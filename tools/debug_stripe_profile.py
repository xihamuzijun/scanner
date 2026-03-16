from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt

from scanner_config import ScannerConfig
from stripe_extractor import (
    MODE_TO_CODE,
    _choose_mode,
    _compute_local_background,
    _compute_quality,
    _find_peak_edges,
    _interpolate_short_gaps,
    _odd,
    _parabolic_subpixel,
)


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def _apply_roi(img: np.ndarray, cfg: ScannerConfig) -> tuple[np.ndarray, Dict[str, int]]:
    h, w = img.shape[:2]
    if not cfg.use_roi:
        return img, {"x0": 0, "x1": w, "y0": 0, "y1": h}
    x0 = max(0, int(cfg.roi_x0))
    y0 = max(0, int(cfg.roi_y0))
    x1 = w if int(cfg.roi_x1) < 0 else min(w, int(cfg.roi_x1))
    y1 = h if int(cfg.roi_y1) < 0 else min(h, int(cfg.roi_y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI 非法: {(x0, x1, y0, y1)}")
    return img[y0:y1, x0:x1], {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _gap_stats(valid_idx: np.ndarray, width: int) -> Dict[str, float]:
    if valid_idx.size == 0:
        return {
            "coverage_ratio": 0.0,
            "mean_gap": float("nan"),
            "median_gap": float("nan"),
            "p95_gap": float("nan"),
            "max_gap": float("nan"),
            "span_ratio": 0.0,
        }
    if valid_idx.size == 1:
        span = 0.0
        return {
            "coverage_ratio": float(valid_idx.size / max(width, 1)),
            "mean_gap": float("nan"),
            "median_gap": float("nan"),
            "p95_gap": float("nan"),
            "max_gap": float("nan"),
            "span_ratio": span,
        }
    gaps = np.diff(valid_idx)
    span_ratio = float((valid_idx[-1] - valid_idx[0]) / max(width - 1, 1))
    return {
        "coverage_ratio": float(valid_idx.size / max(width, 1)),
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "p95_gap": float(np.percentile(gaps, 95)),
        "max_gap": float(np.max(gaps)),
        "span_ratio": span_ratio,
    }


def _sampling_support_stats(valid_idx: np.ndarray, width: int, output_points: int) -> Dict[str, float]:
    output_points = int(max(output_points, 1))
    if width <= 0:
        return {
            "output_points": output_points,
            "sample_pitch_px": float("nan"),
            "bin_width_px": float("nan"),
            "nonempty_bins": 0,
            "bin_hit_rate": 0.0,
            "max_empty_run_bins": output_points,
            "max_empty_run_px": float("nan"),
            "sample_supported_radius3": 0,
            "sample_supported_radius6": 0,
            "sample_support_rate_r3": 0.0,
            "sample_support_rate_r6": 0.0,
            "nearest_dist_mean": float("nan"),
            "nearest_dist_median": float("nan"),
            "nearest_dist_p95": float("nan"),
            "nearest_dist_max": float("nan"),
        }

    sample_pitch = float((width - 1) / max(output_points - 1, 1))
    bin_edges = np.linspace(0.0, float(width), output_points + 1)
    sample_x = np.linspace(0.0, float(width - 1), output_points)

    if valid_idx.size == 0:
        return {
            "output_points": output_points,
            "sample_pitch_px": sample_pitch,
            "bin_width_px": float(width / output_points),
            "nonempty_bins": 0,
            "bin_hit_rate": 0.0,
            "max_empty_run_bins": output_points,
            "max_empty_run_px": float(output_points * sample_pitch),
            "sample_supported_radius3": 0,
            "sample_supported_radius6": 0,
            "sample_support_rate_r3": 0.0,
            "sample_support_rate_r6": 0.0,
            "nearest_dist_mean": float("nan"),
            "nearest_dist_median": float("nan"),
            "nearest_dist_p95": float("nan"),
            "nearest_dist_max": float("nan"),
        }

    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    # 1) 201-bin 覆盖率
    bin_ids = np.digitize(valid_idx.astype(np.float64), bin_edges[1:-1], right=False)
    bin_hits = np.zeros(output_points, dtype=bool)
    bin_hits[np.clip(bin_ids, 0, output_points - 1)] = True
    nonempty_bins = int(bin_hits.sum())

    max_empty_run = 0
    cur = 0
    for hit in bin_hits:
        if hit:
            max_empty_run = max(max_empty_run, cur)
            cur = 0
        else:
            cur += 1
    max_empty_run = max(max_empty_run, cur)

    # 2) 每个采样点到最近有效列的距离
    insert_pos = np.searchsorted(valid_idx, sample_x, side="left")
    left_idx = np.clip(insert_pos - 1, 0, valid_idx.size - 1)
    right_idx = np.clip(insert_pos, 0, valid_idx.size - 1)
    left_dist = np.abs(sample_x - valid_idx[left_idx].astype(np.float64))
    right_dist = np.abs(sample_x - valid_idx[right_idx].astype(np.float64))
    nearest_dist = np.minimum(left_dist, right_dist)

    support_r3 = int(np.sum(nearest_dist <= 3.0))
    support_r6 = int(np.sum(nearest_dist <= 6.0))

    return {
        "output_points": output_points,
        "sample_pitch_px": sample_pitch,
        "bin_width_px": float(width / output_points),
        "nonempty_bins": nonempty_bins,
        "bin_hit_rate": float(nonempty_bins / max(output_points, 1)),
        "max_empty_run_bins": int(max_empty_run),
        "max_empty_run_px": float(max_empty_run * sample_pitch),
        "sample_supported_radius3": support_r3,
        "sample_supported_radius6": support_r6,
        "sample_support_rate_r3": float(support_r3 / max(output_points, 1)),
        "sample_support_rate_r6": float(support_r6 / max(output_points, 1)),
        "nearest_dist_mean": float(np.mean(nearest_dist)),
        "nearest_dist_median": float(np.median(nearest_dist)),
        "nearest_dist_p95": float(np.percentile(nearest_dist, 95)),
        "nearest_dist_max": float(np.max(nearest_dist)),
    }


def debug_extract_stripe_profile(img: np.ndarray, cfg: ScannerConfig) -> Dict[str, Any]:
    blur_ksize = _odd(max(3, cfg.blur_ksize))
    blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    h, w = blur.shape
    global_thr = max(cfg.min_peak_height, float(np.percentile(blur, cfg.global_threshold_percentile)))

    center_full = np.full(w, np.nan, dtype=np.float64)
    quality_full = np.zeros(w, dtype=np.float64)
    mode_full = np.zeros(w, dtype=np.float64)
    raw_valid = np.zeros(w, dtype=bool)

    reasons = Counter()
    rows = []

    for u in range(w):
        col = blur[:, u].astype(np.float64)
        row: Dict[str, Any] = {
            "u": u,
            "col_max": float(np.max(col)),
            "reason": "",
            "peak_idx": np.nan,
            "peak_val": np.nan,
            "bg_mean": np.nan,
            "bg_std": np.nan,
            "contrast": np.nan,
            "snr": np.nan,
            "left": np.nan,
            "right": np.nan,
            "width_px": np.nan,
            "centroid": np.nan,
            "sigma_px": np.nan,
            "asymmetry": np.nan,
            "skewness": np.nan,
            "saturation_ratio": np.nan,
            "flat_top_ratio": np.nan,
            "second_peak_ratio": np.nan,
            "mode": "",
            "quality": np.nan,
            "raw_valid": 0,
            "valid_after_cont": 0,
        }

        peaks, props = find_peaks(
            col,
            height=max(global_thr * 0.75, cfg.min_peak_height),
            prominence=cfg.min_peak_prominence,
            distance=2,
        )

        used_fallback_argmax = False
        if peaks.size == 0:
            peak_idx = int(np.argmax(col))
            row["peak_idx"] = peak_idx
            row["peak_val"] = float(col[peak_idx])
            if col[peak_idx] < global_thr:
                reasons["nopeak_lt_global"] += 1
                row["reason"] = "nopeak_lt_global"
                rows.append(row)
                continue
            peaks = np.array([peak_idx], dtype=np.int64)
            prominences = np.array([max(col[peak_idx] - np.median(col), 0.0)], dtype=np.float64)
            heights = np.array([col[peak_idx]], dtype=np.float64)
            used_fallback_argmax = True
        else:
            prominences = props.get("prominences", np.zeros_like(peaks, dtype=np.float64))
            heights = props.get("peak_heights", col[peaks])

        scores = heights + 0.5 * prominences
        sel = int(np.argmax(scores))
        peak_idx = int(peaks[sel])
        peak_val = float(col[peak_idx])
        row["peak_idx"] = peak_idx
        row["peak_val"] = peak_val
        row["used_fallback_argmax"] = int(used_fallback_argmax)
        if peak_val < global_thr:
            reasons["peak_lt_global"] += 1
            row["reason"] = "peak_lt_global"
            rows.append(row)
            continue

        bg_mean, bg_std = _compute_local_background(col, peak_idx, cfg)
        contrast = max(peak_val - bg_mean, 0.0)
        snr = contrast / max(bg_std, 1e-6)
        row["bg_mean"] = bg_mean
        row["bg_std"] = bg_std
        row["contrast"] = contrast
        row["snr"] = snr
        if snr < 2.0:
            reasons["snr_lt2"] += 1
            row["reason"] = "snr_lt2"
            rows.append(row)
            continue

        left, right = _find_peak_edges(col, peak_idx, bg_mean, peak_val)
        width_px = float(max(right - left, 1))
        row["left"] = left
        row["right"] = right
        row["width_px"] = width_px
        if width_px < cfg.min_width_px:
            reasons["width_lt_min"] += 1
            row["reason"] = "width_lt_min"
            rows.append(row)
            continue

        ys = np.arange(left, right + 1, dtype=np.float64)
        local = col[left:right + 1]
        weights = np.clip(local - bg_mean, a_min=0.0, a_max=None)
        wsum = float(weights.sum())
        if wsum <= 1e-9:
            reasons["wsum_le0"] += 1
            row["reason"] = "wsum_le0"
            rows.append(row)
            continue

        centroid = float(np.sum(ys * weights) / wsum)
        sigma_px = float(np.sqrt(np.sum(weights * (ys - centroid) ** 2) / wsum))
        left_hw = max(centroid - left, 1e-6)
        right_hw = max(right - centroid, 1e-6)
        asymmetry = float((right_hw - left_hw) / max(width_px, 1.0))
        skewness = float(np.sum(weights * ((ys - centroid) / max(sigma_px, 1e-6)) ** 3) / wsum)
        saturation_ratio = float(np.mean(local >= cfg.saturation_threshold))
        flat_top_ratio = float(np.mean(local >= (peak_val - cfg.flat_top_delta)))

        row["centroid"] = centroid
        row["sigma_px"] = sigma_px
        row["asymmetry"] = asymmetry
        row["skewness"] = skewness
        row["saturation_ratio"] = saturation_ratio
        row["flat_top_ratio"] = flat_top_ratio

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
        row["second_peak_ratio"] = second_peak_ratio

        mode = _choose_mode(
            width_px=width_px,
            asymmetry=asymmetry,
            second_peak_ratio=second_peak_ratio,
            saturation_ratio=saturation_ratio,
            flat_top_ratio=flat_top_ratio,
            snr=snr,
            cfg=cfg,
        )
        row["mode"] = mode
        if mode == "invalid":
            reasons["invalid_mode"] += 1
            row["reason"] = "invalid_mode"
            # still record center-like quantities for diagnostics, but not raw_valid
            rows.append(row)
            continue

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
        raw_valid[u] = True
        row["quality"] = quality
        row["raw_valid"] = 1
        row["reason"] = "accepted_raw"
        reasons[f"mode_{mode}"] += 1
        rows.append(row)

    valid_for_cont = raw_valid & np.isfinite(center_full)
    outlier_mask = np.zeros_like(valid_for_cont, dtype=bool)

    if raw_valid.sum() < cfg.min_valid_columns:
        valid_idx = np.array([], dtype=np.int64)
    else:
        if valid_for_cont.sum() >= max(cfg.min_valid_columns, 3 * cfg.continuity_kernel):
            xs = np.where(valid_for_cont)[0].astype(np.float64)
            ys = center_full[valid_for_cont].astype(np.float64)
            full_x = np.arange(len(center_full), dtype=np.float64)
            ref_interp = np.interp(full_x, xs, ys)
            ref_med = medfilt(ref_interp, kernel_size=_odd(max(3, cfg.continuity_kernel)))
            outlier_mask = valid_for_cont & (np.abs(center_full - ref_med) > cfg.continuity_outlier_px) & (quality_full < 0.85)
            valid_for_cont[outlier_mask] = False

        center_full[~valid_for_cont] = np.nan
        before_fill = int(valid_for_cont.sum())
        center_full, valid_for_cont = _interpolate_short_gaps(center_full, valid_for_cont, cfg.max_gap_fill_cols)
        after_fill = int(valid_for_cont.sum())
        valid_idx = np.where(valid_for_cont & np.isfinite(center_full))[0]

        reasons["cont_outlier_removed"] = int(outlier_mask.sum())
        reasons["gap_fill_added"] = max(after_fill - before_fill, 0)

    for row in rows:
        u = int(row["u"])
        row["valid_after_cont"] = int(valid_for_cont[u]) if u < len(valid_for_cont) else 0
        if outlier_mask[u]:
            row["reason_after_cont"] = "continuity_outlier"
        elif raw_valid[u] and valid_for_cont[u]:
            row["reason_after_cont"] = "kept"
        elif raw_valid[u] and (not valid_for_cont[u]):
            row["reason_after_cont"] = "dropped_after_cont"
        else:
            row["reason_after_cont"] = row["reason"]

    summary = {
        "image_shape": [int(h), int(w)],
        "global_thr": float(global_thr),
        "raw_valid_sum": int(raw_valid.sum()),
        "valid_idx_size": int(valid_idx.size),
        "raw_valid_ratio": float(raw_valid.sum() / max(w, 1)),
        "valid_idx_ratio": float(valid_idx.size / max(w, 1)),
        "raw_gap_stats": _gap_stats(np.where(raw_valid)[0], w),
        "final_gap_stats": _gap_stats(valid_idx, w),
        "sampling_support": _sampling_support_stats(valid_idx, w, cfg.output_points),
        "reason_counts": dict(reasons),
        "config": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in cfg.__dict__.items()
        },
    }
    return {
        "summary": summary,
        "rows": rows,
        "raw_valid": raw_valid,
        "valid_for_cont": valid_for_cont,
        "valid_idx": valid_idx,
        "center_full": center_full,
        "quality_full": quality_full,
        "mode_full": mode_full,
        "blur": blur,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print("Stripe profile debug summary")
    print("=" * 72)
    print(f"image_shape           : {summary['image_shape']}")
    print(f"global_thr            : {summary['global_thr']:.3f}")
    print(f"raw_valid_sum         : {summary['raw_valid_sum']}")
    print(f"valid_idx_size        : {summary['valid_idx_size']}")
    print(f"raw_valid_ratio       : {summary['raw_valid_ratio']:.4f}")
    print(f"valid_idx_ratio       : {summary['valid_idx_ratio']:.4f}")

    print("\n[raw gap stats]")
    for k, v in summary["raw_gap_stats"].items():
        print(f"  {k:18s}: {v:.4f}" if np.isfinite(v) else f"  {k:18s}: nan")

    print("\n[final gap stats]")
    for k, v in summary["final_gap_stats"].items():
        print(f"  {k:18s}: {v:.4f}" if np.isfinite(v) else f"  {k:18s}: nan")

    print("\n[reason counts]")
    for k, v in sorted(summary["reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:24s}: {v}")
    print("=" * 72)


def _write_csv(rows: list[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_overlay(gray: np.ndarray, debug: Dict[str, Any], path: Path, output_points: int = 201) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rows = debug["rows"]
    for row in rows:
        u = int(row["u"])
        if row["raw_valid"]:
            y = row["centroid"] if np.isfinite(row["centroid"]) else row["peak_idx"]
            if np.isfinite(y):
                cv2.circle(vis, (u, int(round(y))), 1, (0, 255, 0), -1)
    for u in debug["valid_idx"]:
        y = debug["center_full"][u]
        if np.isfinite(y):
            cv2.circle(vis, (int(u), int(round(y))), 1, (0, 0, 255), -1)

    # 采样位置是否有近邻支撑：蓝=近邻<=3px，黄=<=6px，紫=>6px
    valid_idx = np.asarray(debug["valid_idx"], dtype=np.int64)
    if output_points > 0:
        xs = np.linspace(0.0, float(gray.shape[1] - 1), int(output_points))
        for x in xs:
            xi = int(round(x))
            color = (255, 0, 255)
            if valid_idx.size > 0:
                pos = int(np.searchsorted(valid_idx, x, side="left"))
                cand = []
                if pos < valid_idx.size:
                    cand.append(abs(float(valid_idx[pos]) - x))
                if pos > 0:
                    cand.append(abs(float(valid_idx[pos - 1]) - x))
                d = min(cand) if cand else float("inf")
                color = (255, 0, 0) if d <= 3.0 else ((0, 255, 255) if d <= 6.0 else (255, 0, 255))
            cv2.line(vis, (xi, 0), (xi, min(15, gray.shape[0] - 1)), color, 1)
    cv2.imwrite(str(path), vis)

def main() -> None:
    img_id='2_5600'
    parser = argparse.ArgumentParser(description="Debug stripe extractor counts and rejection reasons.")
    parser.add_argument("--image",  type=str,default=f"raw_data/layer2/camera/{img_id}.png", help="灰度图路径")
    parser.add_argument("--json", dest="json_out", default=f"{img_id}.json", help="保存 summary.json 的路径")
    parser.add_argument("--csv", dest="csv_out", default=f"{img_id}.csv", help="保存逐列诊断 csv 的路径")
    parser.add_argument("--overlay", dest="overlay_out", default="", help="保存叠加可视化 png 的路径")
    parser.add_argument("--use-roi",type=bool, default=False, help="启用 ROI")
    parser.add_argument("--roi-x0", type=int, default=None)
    parser.add_argument("--roi-x1", type=int, default=None)
    parser.add_argument("--roi-y0", type=int, default=None)
    parser.add_argument("--roi-y1", type=int, default=None)
    parser.add_argument("--global-thr-pct", type=float, default=None)
    parser.add_argument("--min-peak-height", type=float, default=None)
    parser.add_argument("--min-peak-prominence", type=float, default=None)
    parser.add_argument("--min-width", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--continuity-kernel", type=int, default=None)
    parser.add_argument("--continuity-outlier-px", type=float, default=None)
    parser.add_argument("--max-gap-fill-cols", type=int, default=None)
    parser.add_argument("--min-valid-columns", type=int, default=None)
    parser.add_argument("--output-points", type=int, default=None, help="目标等距采样点数")
    args = parser.parse_args()

    cfg = ScannerConfig()
    if args.use_roi:
        cfg.use_roi = True
    for name, value in {
        "roi_x0": args.roi_x0,
        "roi_x1": args.roi_x1,
        "roi_y0": args.roi_y0,
        "roi_y1": args.roi_y1,
        "global_threshold_percentile": args.global_thr_pct,
        "min_peak_height": args.min_peak_height,
        "min_peak_prominence": args.min_peak_prominence,
        "min_width_px": args.min_width,
        "max_width_px": args.max_width,
        "continuity_kernel": args.continuity_kernel,
        "continuity_outlier_px": args.continuity_outlier_px,
        "max_gap_fill_cols": args.max_gap_fill_cols,
        "min_valid_columns": args.min_valid_columns,
        "output_points": args.output_points,
    }.items():
        if value is not None:
            setattr(cfg, name, value)

    gray0 = _read_gray(args.image)
    gray, roi_info = _apply_roi(gray0, cfg)
    debug = debug_extract_stripe_profile(gray, cfg)
    debug["summary"]["roi"] = roi_info
    _print_summary(debug["summary"])

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(debug["summary"], f, ensure_ascii=False, indent=2)
    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(debug["rows"], csv_path)
    if args.overlay_out:
        overlay_path = Path(args.overlay_out)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        _save_overlay(gray, debug, overlay_path, output_points=cfg.output_points)


if __name__ == "__main__":
    main()
