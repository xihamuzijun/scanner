# -*- coding: utf-8 -*-
"""
更新日志
- v4
    重采样与逐点特征支持 layer1 几何校正输出的 x_shift。

    关键改动：
    1) resample_observation() 新增 x_shift_mm 输入，统一在物理 x 网格上施加横向平移；
    2) 统计量中显式记录 x_shift_applied_mm；
    3) build_point_features() 将 x_shift 作为全局特征加入逐点回归。
- v3
- v2
- v1
  1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, medfilt

from scanner_config import ScannerConfig


MORPH_FEATURE_KEYS: List[str] = [
    "width_px",
    "sigma_px",
    "asymmetry",
    "skewness",
    "peak_val",
    "intensity_sum",
    "contrast",
    "second_peak_ratio",
    "saturation_ratio",
    "centroid_offset_px",
    "bg_mean",
    "bg_std",
    "snr",
    "quality",
    "mode_code",
]


MODE_TO_CODE = {
    "invalid": 0.0,
    "gaussian": 1.0,
    "centroid": 2.0,
    "threshold": 3.0,
}


GLOBAL_STAT_KEYS = [
    "u_left_actual",
    "u_right_actual",
    "u_span_actual",
    "u_span_support",
    "x_left_actual_mm",
    "x_right_actual_mm",
    "x_span_actual_mm",
    "v_mean",
    "v_std",
    "quality_mean",
    "quality_std",
    "width_mean",
    "width_std",
    "peak_mean",
    "contrast_mean",
    "snr_mean",
    "x_shift_applied_mm",
]


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _parabolic_subpixel(col: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(col) - 1:
        return float(idx)
    y1, y2, y3 = float(col[idx - 1]), float(col[idx]), float(col[idx + 1])
    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-12:
        return float(idx)
    delta = 0.5 * (y1 - y3) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(idx + delta)


def _find_peak_edges(col: np.ndarray, peak_idx: int, bg_mean: float, peak_val: float) -> tuple[int, int]:
    h = len(col)
    thr = bg_mean + 0.35 * max(peak_val - bg_mean, 1.0)

    left = peak_idx
    while left > 0 and col[left] >= thr:
        left -= 1
    right = peak_idx
    while right < h - 1 and col[right] >= thr:
        right += 1

    return int(left), int(right)


def _robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 1e-6
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-6)


def _compute_local_background(col: np.ndarray, peak_idx: int, cfg: ScannerConfig) -> tuple[float, float]:
    h = len(col)
    y0 = max(0, peak_idx - cfg.local_bg_halfwin)
    y1 = min(h, peak_idx + cfg.local_bg_halfwin + 1)
    ex0 = max(0, peak_idx - cfg.local_bg_exclude_halfwin)
    ex1 = min(h, peak_idx + cfg.local_bg_exclude_halfwin + 1)

    win = col[y0:y1]
    keep_mask = np.ones_like(win, dtype=bool)
    keep_mask[(ex0 - y0):(ex1 - y0)] = False
    bg = win[keep_mask]

    if bg.size < 8:
        bg = np.concatenate([col[: max(1, peak_idx - 4)], col[min(h, peak_idx + 5):]])
    if bg.size < 8:
        return float(np.median(col)), float(np.std(col) + 1e-6)

    bg_sorted = np.sort(np.asarray(bg, dtype=np.float64))
    n_keep = max(8, int(cfg.bg_low_fraction * bg_sorted.size))
    bg_low = bg_sorted[:n_keep]

    bg_mean = float(np.median(bg_low))
    bg_std = _robust_std(bg_low)
    return bg_mean, bg_std


def _choose_mode(
    width_px: float,
    asymmetry: float,
    second_peak_ratio: float,
    saturation_ratio: float,
    flat_top_ratio: float,
    snr: float,
    cfg: ScannerConfig,
) -> str:
    if snr < cfg.snr_invalid_threshold:
        return "invalid"
    if width_px < cfg.min_width_px or width_px > cfg.max_width_px * cfg.width_invalid_factor:
        return "invalid"
    if saturation_ratio > cfg.saturation_mode_threshold or flat_top_ratio > cfg.flat_top_mode_threshold:
        return "threshold"
    if second_peak_ratio > cfg.second_peak_invalid_ratio:
        return "invalid"
    if second_peak_ratio > cfg.second_peak_centroid_ratio or abs(asymmetry) > cfg.asymmetry_centroid_threshold:
        return "centroid"
    return "gaussian"


def _compute_quality(
    width_px: float,
    asymmetry: float,
    second_peak_ratio: float,
    saturation_ratio: float,
    snr: float,
    mode: str,
    cfg: ScannerConfig,
) -> float:
    q_snr = _clip01((snr - 2.0) / 10.0)
    q_width = 1.0 if cfg.min_width_px <= width_px <= cfg.max_width_px else 0.4
    q_asym = 1.0 - _clip01(abs(asymmetry))
    q_multi = 1.0 - _clip01(second_peak_ratio)
    q_sat = 1.0 - _clip01(saturation_ratio)
    base = 0.28 * q_snr + 0.18 * q_width + 0.18 * q_asym + 0.18 * q_multi + 0.18 * q_sat
    if mode == "invalid":
        base *= 0.2
    elif mode == "threshold":
        base *= 0.7
    return float(np.clip(base, 0.0, 1.0))


def _interpolate_short_gaps(arr: np.ndarray, valid_mask: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    arr = arr.copy()
    valid_mask = valid_mask.copy()
    n = len(arr)
    i = 0
    while i < n:
        if valid_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not valid_mask[j]:
            j += 1
        gap = j - i
        left = i - 1
        right = j
        if gap <= max_gap and left >= 0 and right < n and valid_mask[left] and valid_mask[right]:
            xs = np.array([left, right], dtype=np.float64)
            ys = np.array([arr[left], arr[right]], dtype=np.float64)
            xi = np.arange(i, j, dtype=np.float64)
            arr[i:j] = np.interp(xi, xs, ys)
            valid_mask[i:j] = True
        i = j
    return arr, valid_mask


def extract_stripe_profile(img: np.ndarray, cfg: ScannerConfig) -> Dict[str, np.ndarray]:
    blur_ksize = _odd(max(3, cfg.blur_ksize))
    blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    h, w = blur.shape
    global_thr = max(cfg.min_peak_height, float(np.percentile(blur, cfg.global_threshold_percentile)))

    center_full = np.full(w, np.nan, dtype=np.float64)
    quality_full = np.zeros(w, dtype=np.float64)
    mode_full = np.zeros(w, dtype=np.float64)
    morph_full = {k: np.full(w, np.nan, dtype=np.float64) for k in MORPH_FEATURE_KEYS if k not in {"quality", "mode_code"}}
    raw_valid = np.zeros(w, dtype=bool)

    for u in range(w):
        col = blur[:, u].astype(np.float64)
        peaks, props = find_peaks(
            col,
            height=max(global_thr * 0.75, cfg.min_peak_height),
            prominence=cfg.min_peak_prominence,
            distance=2,
        )

        if peaks.size == 0:
            peak_idx = int(np.argmax(col))
            if col[peak_idx] < global_thr:
                continue
            peaks = np.array([peak_idx], dtype=np.int64)
            prominences = np.array([max(col[peak_idx] - np.median(col), 0.0)], dtype=np.float64)
            heights = np.array([col[peak_idx]], dtype=np.float64)
        else:
            prominences = props.get("prominences", np.zeros_like(peaks, dtype=np.float64))
            heights = props.get("peak_heights", col[peaks])

        scores = heights + 0.5 * prominences
        sel = int(np.argmax(scores))
        peak_idx = int(peaks[sel])
        peak_val = float(col[peak_idx])
        if peak_val < global_thr:
            continue

        bg_mean, bg_std = _compute_local_background(col, peak_idx, cfg)
        contrast = max(peak_val - bg_mean, 0.0)
        snr = contrast / max(bg_std, 1e-6)
        if snr < 2.0:
            continue

        left, right = _find_peak_edges(col, peak_idx, bg_mean, peak_val)
        width_px = float(max(right - left, 1))
        if width_px < cfg.min_width_px:
            continue

        ys = np.arange(left, right + 1, dtype=np.float64)
        local = col[left:right + 1]
        weights = np.clip(local - bg_mean, a_min=0.0, a_max=None)
        wsum = float(weights.sum())
        if wsum <= 1e-9:
            continue

        centroid = float(np.sum(ys * weights) / wsum)
        sigma_px = float(np.sqrt(np.sum(weights * (ys - centroid) ** 2) / wsum))
        left_hw = max(centroid - left, 1e-6)
        right_hw = max(right - centroid, 1e-6)
        asymmetry = float((right_hw - left_hw) / max(width_px, 1.0))
        skewness = float(np.sum(weights * ((ys - centroid) / max(sigma_px, 1e-6)) ** 3) / wsum)
        saturation_ratio = float(np.mean(local >= cfg.saturation_threshold))
        flat_top_ratio = float(np.mean(local >= (peak_val - cfg.flat_top_delta)))
        intensity_sum = float(wsum)
        centroid_offset_px = float(centroid - peak_idx)

        local_peaks, local_props = find_peaks(
            local,
            height=bg_mean + 0.25 * contrast,
            prominence=max(cfg.min_peak_prominence * 0.5, 0.1 * contrast),
            distance=2,
        )
        if local_peaks.size >= 2:
            local_heights = np.sort(local_props.get("peak_heights", local[local_peaks]))[::-1]
            second_peak_ratio = float(local_heights[1] / max(local_heights[0], 1e-6))
        else:
            second_peak_ratio = 0.0

        mode = _choose_mode(
            width_px=width_px,
            asymmetry=asymmetry,
            second_peak_ratio=second_peak_ratio,
            saturation_ratio=saturation_ratio,
            flat_top_ratio=flat_top_ratio,
            snr=snr,
            cfg=cfg,
        )

        if mode == "gaussian":
            center = _parabolic_subpixel(col, peak_idx)
        elif mode == "centroid":
            center = centroid
        elif mode == "threshold":
            center = 0.5 * (left + right)
        else:
            center = centroid

        quality = _compute_quality(
            width_px=width_px,
            asymmetry=asymmetry,
            second_peak_ratio=second_peak_ratio,
            saturation_ratio=saturation_ratio,
            snr=snr,
            mode=mode,
            cfg=cfg,
        )

        center_full[u] = center
        quality_full[u] = quality
        mode_full[u] = MODE_TO_CODE[mode]
        morph_full["width_px"][u] = width_px
        morph_full["sigma_px"][u] = sigma_px
        morph_full["asymmetry"][u] = asymmetry
        morph_full["skewness"][u] = skewness
        morph_full["peak_val"][u] = peak_val
        morph_full["intensity_sum"][u] = intensity_sum
        morph_full["contrast"][u] = contrast
        morph_full["second_peak_ratio"][u] = second_peak_ratio
        morph_full["saturation_ratio"][u] = saturation_ratio
        morph_full["centroid_offset_px"][u] = centroid_offset_px
        morph_full["bg_mean"][u] = bg_mean
        morph_full["bg_std"][u] = bg_std
        morph_full["snr"][u] = snr
        raw_valid[u] = mode != "invalid"

    empty = {
        "u": np.array([], dtype=np.float64),
        "v": np.array([], dtype=np.float64),
        "quality": np.array([], dtype=np.float64),
        "mode_code": np.array([], dtype=np.float64),
        **{k: np.array([], dtype=np.float64) for k in morph_full.keys()},
    }
    if raw_valid.sum() < cfg.min_valid_columns:
        return empty

    valid_for_cont = raw_valid & np.isfinite(center_full)

    if valid_for_cont.sum() >= max(cfg.min_valid_columns, 3 * cfg.continuity_kernel):
        xs = np.where(valid_for_cont)[0].astype(np.float64)
        ys = center_full[valid_for_cont].astype(np.float64)
        full_x = np.arange(len(center_full), dtype=np.float64)
        ref_interp = np.interp(full_x, xs, ys)
        ref_med = medfilt(ref_interp, kernel_size=_odd(max(3, cfg.continuity_kernel)))
        outlier_mask = valid_for_cont & (np.abs(center_full - ref_med) > cfg.continuity_outlier_px) & (quality_full < 0.85)
        valid_for_cont[outlier_mask] = False

    center_full[~valid_for_cont] = np.nan
    center_full, valid_for_cont = _interpolate_short_gaps(center_full, valid_for_cont, cfg.max_gap_fill_cols)

    for k, arr in morph_full.items():
        arr[~raw_valid] = np.nan
        arr_interp = arr.copy()
        arr_interp, _ = _interpolate_short_gaps(arr_interp, valid_for_cont.copy(), cfg.max_gap_fill_cols)
        morph_full[k] = arr_interp

    quality_full[~valid_for_cont] = 0.0
    mode_full[~valid_for_cont] = 0.0

    valid_idx = np.where(valid_for_cont & np.isfinite(center_full))[0]
    if valid_idx.size < cfg.min_valid_columns:
        return empty

    result = {
        "u": valid_idx.astype(np.float64),
        "v": center_full[valid_idx].astype(np.float64),
        "quality": quality_full[valid_idx].astype(np.float64),
        "mode_code": mode_full[valid_idx].astype(np.float64),
    }
    for k in morph_full.keys():
        result[k] = morph_full[k][valid_idx].astype(np.float64)
    return result


def resample_observation(profile: Dict[str, np.ndarray], cfg: ScannerConfig, x_shift_mm: float = 0.0) -> Dict[str, np.ndarray]:
    """
    以固定 ROI 的全图列坐标为统一支撑网格，对观测条纹做重采样。

    当前版本在物理 x 网格上显式加入 layer1 几何校正输出的 x_shift_mm。
    """
    u_full = np.asarray(profile["u_full"], dtype=np.float64).reshape(-1)
    v_full = np.asarray(profile["v_full"], dtype=np.float64).reshape(-1)
    quality = np.asarray(profile["quality"], dtype=np.float64).reshape(-1)

    if u_full.size < 4:
        raise ValueError("有效条纹点过少，无法重采样")
    if not (u_full.size == v_full.size == quality.size):
        raise ValueError("profile 长度不一致")

    idx = np.argsort(u_full)
    u_full = u_full[idx]
    v_full = v_full[idx]
    quality = quality[idx]

    x_shift_mm = float(cfg.clip_x_shift(x_shift_mm))
    u_grid = cfg.full_u_grid()
    x_rel_grid_nominal = cfg.full_u_to_x_mm(u_grid)
    x_rel_grid = x_rel_grid_nominal + x_shift_mm
    t_grid = np.linspace(-1.0, 1.0, cfg.profile_points, dtype=np.float64)

    actual_left = float(np.nanmin(u_full))
    actual_right = float(np.nanmax(u_full))
    actual_span = max(actual_right - actual_left, 1e-6)
    valid_mask = ((u_grid >= actual_left) & (u_grid <= actual_right)).astype(np.float64)

    out: Dict[str, np.ndarray] = {
        "u_full_grid": u_grid,
        "x_rel_grid_nominal": x_rel_grid_nominal,
        "x_rel_grid": x_rel_grid,
        "t_grid": t_grid,
        "valid_mask": valid_mask,
        "x_shift_applied_mm": np.asarray([x_shift_mm], dtype=np.float64),
    }

    interp_data = {
        "v": v_full,
        "quality": quality,
        "mode_code": np.asarray(profile["mode_code"], dtype=np.float64)[idx],
    }
    for k in MORPH_FEATURE_KEYS:
        if k in {"quality", "mode_code"}:
            continue
        interp_data[k] = np.asarray(profile[k], dtype=np.float64)[idx]

    for name, arr in interp_data.items():
        f = interp1d(u_full, arr, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
        out[name + "_grid"] = np.asarray(f(u_grid), dtype=np.float64)

    stats = {
        "u_left_actual": actual_left,
        "u_right_actual": actual_right,
        "u_span_actual": actual_span,
        "u_span_support": float(cfg.roi_u_right - cfg.roi_u_left),
        "x_left_actual_mm": float(cfg.full_u_to_x_mm(actual_left) + x_shift_mm),
        "x_right_actual_mm": float(cfg.full_u_to_x_mm(actual_right) + x_shift_mm),
        "x_span_actual_mm": float(cfg.full_u_to_x_mm(actual_right) - cfg.full_u_to_x_mm(actual_left)),
        "v_mean": float(np.nanmean(out["v_grid"])),
        "v_std": float(np.nanstd(out["v_grid"])),
        "quality_mean": float(np.nanmean(out["quality_grid"])),
        "quality_std": float(np.nanstd(out["quality_grid"])),
        "width_mean": float(np.nanmean(out["width_px_grid"])),
        "width_std": float(np.nanstd(out["width_px_grid"])),
        "peak_mean": float(np.nanmean(out["peak_val_grid"])),
        "contrast_mean": float(np.nanmean(out["contrast_grid"])),
        "snr_mean": float(np.nanmean(out["snr_grid"])),
        "x_shift_applied_mm": x_shift_mm,
    }
    out["stats"] = stats
    return out


def build_point_features(resampled: Dict[str, np.ndarray], cfg: ScannerConfig) -> np.ndarray:
    """
    构造逐点特征。

    本版本显式加入：
    - u_full_grid / x_rel_grid：绝对列位置与物理 x；
    - z0_grid：逐列参考平面曲线，而非旧版常数基线；
    - x_shift_applied_mm：layer1 几何校正输出的横向校正量；
    - 其余局部形貌特征及梯度。
    """
    t_grid = np.asarray(resampled["t_grid"], dtype=np.float64)
    x_rel_grid = np.asarray(resampled["x_rel_grid"], dtype=np.float64)
    u_full_grid = np.asarray(resampled["u_full_grid"], dtype=np.float64)
    valid_mask = np.asarray(resampled["valid_mask"], dtype=np.float64)
    x_shift_applied_mm = float(np.asarray(resampled["x_shift_applied_mm"], dtype=np.float64).reshape(-1)[0])

    signal_names = [
        "v_grid",
        "z0_grid",
        "quality_grid",
        "width_px_grid",
        "sigma_px_grid",
        "peak_val_grid",
        "contrast_grid",
        "snr_grid",
        "second_peak_ratio_grid",
        "asymmetry_grid",
        "centroid_offset_px_grid",
        "mode_code_grid",
    ]
    signals = {name: np.asarray(resampled[name], dtype=np.float64) for name in signal_names}
    grads = {name: np.gradient(arr) for name, arr in signals.items()}

    pad = cfg.context_halfwin
    padded = {name: np.pad(arr, (pad, pad), mode="edge") for name, arr in signals.items()}
    padded_grads = {name: np.pad(arr, (pad, pad), mode="edge") for name, arr in grads.items()}
    mask_pad = np.pad(valid_mask, (pad, pad), mode="edge")

    x_half = max(float(cfg.physical_window_width_mm) * 0.5, 1e-6)
    u_center = float(cfg.roi_u_center)
    u_half = max(float(cfg.roi_width - 1) * 0.5, 1.0)
    global_feat = [resampled["stats"][k] for k in GLOBAL_STAT_KEYS]

    feats = []
    n = t_grid.size
    for i in range(n):
        j = i + pad
        q_local = padded["quality_grid"][j - pad : j + pad + 1]
        f = [
            t_grid[i],
            valid_mask[i],
            x_rel_grid[i],
            x_rel_grid[i] / x_half,
            u_full_grid[i],
            (u_full_grid[i] - u_center) / u_half,
            x_shift_applied_mm,
            x_shift_applied_mm / max(abs(cfg.xshift_clip_mm), 1e-6),
        ]
        for name in signal_names:
            local = padded[name][j - pad : j + pad + 1]
            f.extend(local.tolist())
            f.extend((local * q_local).tolist())
        for name in ("v_grid", "z0_grid", "quality_grid", "width_px_grid", "contrast_grid", "snr_grid"):
            f.extend(padded_grads[name][j - pad : j + pad + 1].tolist())
        f.extend(mask_pad[j - pad : j + pad + 1].tolist())
        f.extend(global_feat)
        feats.append(f)

    return np.asarray(feats, dtype=np.float64)
