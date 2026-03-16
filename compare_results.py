# -*- coding: utf-8 -*-
"""
更新日志
- v4
    best x-shift 诊断统一改为读取配置范围，避免评估与训练使用不同的搜索边界。
- v3
- v2
    1) 评估改为 NaN 容忍：误差指标仅在有限值点上统计，不再因个别缺失点导致整条曲线指标为 NaN。
    2) 新增覆盖率统计：输出 n_samples / valid_points / nan_points / valid_ratio，区分“误差大”和“缺失多”。
    3) 新增诊断指标：mae_centered_mm / rmse_centered_mm，用于区分“整体偏移”与“局部形状误差”。
    4) 新增 best x-shift 诊断：在给定小范围内扫描横向平移，输出 best_shift_mm / mae_best_shift_mm / rmse_best_shift_mm，用于判断横向配准问题。
    5) 面积误差仅在连续有效区段上积分，避免跨 NaN 缺口硬连线。
    6) 汇总统计改为 NaN-safe 聚合：点级误差按 valid_points 加权，形貌类指标采用 NaN 忽略聚合。
- v1
    1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_reader import (
    build_reference_interpolator,
    ensure_dir,
    extract_xyz_from_pose,
    find_layer_scanner_file,
    find_origin_pose_path,
    get_layer_id,
    index_camera_samples,
    load_pose_npy,
    load_scanner_profile_npy,
    pose_to_relative,
    sample_reference_profile,
)
from scanner_config import ScannerConfig


def _finite_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _pv(arr: np.ndarray) -> float:
    arr = _finite_1d(arr)
    if arr.size == 0:
        return np.nan
    return float(np.max(arr) - np.min(arr))


def _nanmax_abs(arr: np.ndarray) -> float:
    arr = np.abs(_finite_1d(arr))
    if arr.size == 0:
        return np.nan
    return float(np.max(arr))


def _nanmean(arr: np.ndarray) -> float:
    arr = _finite_1d(arr)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _nanweighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != weights.size:
        raise ValueError("values 与 weights 长度不一致")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _trapz_valid_segments(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x 与 y 长度不一致")

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.nan

    total = 0.0
    has_segment = False
    start = None
    n = len(mask)

    for i in range(n):
        if mask[i] and start is None:
            start = i

        end_of_segment = start is not None and ((not mask[i]) or (i == n - 1))
        if end_of_segment:
            end = i if mask[i] else i - 1
            if end - start + 1 >= 2:
                total += _trapezoid(y[start : end + 1], x[start : end + 1])
                has_segment = True
            start = None

    return float(total) if has_segment else np.nan


def _calc_centered_metrics(err: np.ndarray) -> Tuple[float, float]:
    err = _finite_1d(err)
    if err.size == 0:
        return np.nan, np.nan
    centered = err - np.mean(err)
    mae_centered = float(np.mean(np.abs(centered)))
    rmse_centered = float(np.sqrt(np.mean(centered ** 2)))
    return mae_centered, rmse_centered


def _shifted_profile_metrics(
    x: np.ndarray,
    z_pred: np.ndarray,
    z_ref: np.ndarray,
    shift_mm: float,
) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    z_pred = np.asarray(z_pred, dtype=np.float64).reshape(-1)
    z_ref = np.asarray(z_ref, dtype=np.float64).reshape(-1)

    mask_ref = np.isfinite(x) & np.isfinite(z_ref)
    mask_pred = np.isfinite(x) & np.isfinite(z_pred)
    if np.sum(mask_ref) < 2 or np.sum(mask_pred) == 0:
        return np.nan, np.nan, 0

    x_ref_valid = x[mask_ref]
    z_ref_valid = z_ref[mask_ref]
    order = np.argsort(x_ref_valid)
    x_ref_valid = x_ref_valid[order]
    z_ref_valid = z_ref_valid[order]

    x_pred_valid = x[mask_pred]
    z_pred_valid = z_pred[mask_pred]

    x_query = x_pred_valid + float(shift_mm)
    in_range = (x_query >= x_ref_valid.min()) & (x_query <= x_ref_valid.max())
    if np.sum(in_range) < 2:
        return np.nan, np.nan, 0

    z_ref_shifted = np.interp(x_query[in_range], x_ref_valid, z_ref_valid)
    err = z_pred_valid[in_range] - z_ref_shifted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse, int(err.size)


def _find_best_shift(cfg: ScannerConfig, x: np.ndarray, z_pred: np.ndarray, z_ref: np.ndarray) -> Dict[str, float]:
    shifts = np.arange(
        cfg.xshift_search_min_mm,
        cfg.xshift_search_max_mm + 0.5 * cfg.xshift_search_step_mm,
        cfg.xshift_search_step_mm,
        dtype=np.float64,
    )

    best_shift = np.nan
    best_mae = np.nan
    best_rmse = np.nan
    best_valid = 0

    for s in shifts:
        mae, rmse, n_valid = _shifted_profile_metrics(x, z_pred, z_ref, shift_mm=float(s))
        if n_valid <= 0 or not np.isfinite(mae):
            continue
        if not np.isfinite(best_mae) or mae < best_mae:
            best_shift = float(s)
            best_mae = float(mae)
            best_rmse = float(rmse)
            best_valid = int(n_valid)

    return {
        "best_shift_mm": best_shift,
        "mae_best_shift_mm": best_mae,
        "rmse_best_shift_mm": best_rmse,
        "best_shift_valid_points": best_valid,
    }


def compare_one_prediction(
    pred_csv_path: str,
    scanner_file: str,
    pose_path: str,
    origin_pose: np.ndarray,
    cfg: ScannerConfig,
) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv_path)
    if not {"x_mm", "z_mm"}.issubset(pred_df.columns):
        raise ValueError(f"{pred_csv_path} 缺少 x_mm/z_mm 列")

    x_pred = pred_df["x_mm"].to_numpy(dtype=np.float64)
    z_pred = pred_df["z_mm"].to_numpy(dtype=np.float64)

    pose = load_pose_npy(pose_path)
    rel_pose = pose_to_relative(pose, origin_pose)
    dx, _, dz = extract_xyz_from_pose(rel_pose)

    x_ref, z_ref = load_scanner_profile_npy(scanner_file)
    x_unique, z_unique, fz_ref = build_reference_interpolator(x_ref, z_ref)

    z_ref_local = sample_reference_profile(
        fz=fz_ref,
        x_ref_min=float(x_unique.min()),
        x_ref_max=float(x_unique.max()),
        center_x_global_mm=dx + cfg.x_center_offset_mm,
        x_rel_grid_mm=x_pred,
    )
    z_ref_true = z_ref_local + dz

    err = z_pred - z_ref_true
    out = pd.DataFrame(
        {
            "x_mm": x_pred,
            "z_pred_mm": z_pred,
            "z_ref_mm": z_ref_true,
            "err_mm": err,
            "abs_err_mm": np.abs(err),
        }
    )
    return out


def summarize_compare_df(df: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, float]:
    x = df["x_mm"].to_numpy(dtype=np.float64)
    z_pred = df["z_pred_mm"].to_numpy(dtype=np.float64)
    z_ref = df["z_ref_mm"].to_numpy(dtype=np.float64)
    err = df["err_mm"].to_numpy(dtype=np.float64)

    valid_mask = np.isfinite(x) & np.isfinite(err)
    total_points = int(len(df))
    valid_points = int(np.sum(valid_mask))
    nan_points = int(total_points - valid_points)
    valid_ratio = float(valid_points / total_points) if total_points > 0 else np.nan

    pv_pred_mm = _pv(z_pred)
    pv_ref_mm = _pv(z_ref)
    pv_err_mm = float(pv_pred_mm - pv_ref_mm) if np.isfinite(pv_pred_mm) and np.isfinite(pv_ref_mm) else np.nan
    pv_abs_err_mm = float(abs(pv_err_mm)) if np.isfinite(pv_err_mm) else np.nan

    area_abs = _trapz_valid_segments(x, np.abs(err))
    area_signed = _trapz_valid_segments(x, err)

    base = {
        "n_samples": total_points,
        "valid_points": valid_points,
        "nan_points": nan_points,
        "valid_ratio": valid_ratio,
        "pv_pred_mm": pv_pred_mm,
        "pv_ref_mm": pv_ref_mm,
        "pv_err_mm": pv_err_mm,
        "pv_abs_err_mm": pv_abs_err_mm,
        "area_abs_err_mm2": area_abs,
        "area_signed_err_mm2": area_signed,
    }

    if valid_points == 0:
        base.update(
            {
                "mae_mm": np.nan,
                "rmse_mm": np.nan,
                "max_abs_err_mm": np.nan,
                "bias_mm": np.nan,
                "std_err_mm": np.nan,
                "mae_centered_mm": np.nan,
                "rmse_centered_mm": np.nan,
                "best_shift_mm": np.nan,
                "mae_best_shift_mm": np.nan,
                "rmse_best_shift_mm": np.nan,
                "best_shift_valid_points": 0,
            }
        )
        return base

    err_valid = err[valid_mask]
    mae_mm = float(np.mean(np.abs(err_valid)))
    rmse_mm = float(np.sqrt(np.mean(err_valid ** 2)))
    max_abs_err_mm = float(np.max(np.abs(err_valid)))
    bias_mm = float(np.mean(err_valid))
    std_err_mm = float(np.std(err_valid))
    mae_centered_mm, rmse_centered_mm = _calc_centered_metrics(err_valid)
    best_shift_info = _find_best_shift(cfg=cfg, x=x, z_pred=z_pred, z_ref=z_ref)

    base.update(
        {
            "mae_mm": mae_mm,
            "rmse_mm": rmse_mm,
            "max_abs_err_mm": max_abs_err_mm,
            "bias_mm": bias_mm,
            "std_err_mm": std_err_mm,
            "mae_centered_mm": mae_centered_mm,
            "rmse_centered_mm": rmse_centered_mm,
            **best_shift_info,
        }
    )
    return base


def compare_prediction_folder(cfg: ScannerConfig) -> Tuple[pd.DataFrame, List[str]]:
    ensure_dir(cfg.comparison_dir)
    origin_pose = load_pose_npy(find_origin_pose_path(cfg.raw_root))

    records: List[Dict[str, float]] = []
    failed: List[str] = []

    layer_dirs = [d for d in os.listdir(cfg.raw_root) if d.startswith("layer")]
    layer_dirs = [os.path.join(cfg.raw_root, d) for d in sorted(layer_dirs)]

    for layer_dir in layer_dirs:
        layer_id = get_layer_id(layer_dir)
        if layer_id == 1:
            continue

        scanner_file = find_layer_scanner_file(layer_dir)
        pred_layer_dir = os.path.join(cfg.prediction_dir, f"layer{layer_id}")
        out_layer_dir = os.path.join(cfg.comparison_dir, f"layer{layer_id}")
        ensure_dir(out_layer_dir)

        sample_map = {s.sample_idx: s for s in index_camera_samples(layer_dir)}

        if not os.path.isdir(pred_layer_dir):
            failed.append(f"缺少预测目录: {pred_layer_dir}")
            continue

        for fname in sorted(os.listdir(pred_layer_dir)):
            if not fname.endswith("_mid20mm.csv"):
                continue

            pred_path = os.path.join(pred_layer_dir, fname)
            try:
                sample_idx = int(fname.split("_")[1])
                if sample_idx not in sample_map:
                    raise KeyError(f"layer{layer_id} sample {sample_idx} 在原始数据中不存在")

                cmp_df = compare_one_prediction(
                    pred_csv_path=pred_path,
                    scanner_file=scanner_file,
                    pose_path=sample_map[sample_idx].pose_path,
                    origin_pose=origin_pose,
                    cfg=cfg,
                )
                cmp_out = os.path.join(out_layer_dir, fname.replace("_mid20mm.csv", "_compare.csv"))
                cmp_df.to_csv(cmp_out, index=False)

                rec = summarize_compare_df(cmp_df, cfg=cfg)
                rec["scope"] = "by_layer"
                rec["layer_id"] = layer_id
                rec["sample_idx"] = sample_idx
                records.append(rec)
            except Exception as exc:
                failed.append(f"{pred_path} -> {exc}")

    if not records:
        raise RuntimeError("没有生成任何对比结果")

    detail_df = pd.DataFrame(records)
    return detail_df, failed


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        raise ValueError("detail_df 为空，无法汇总")

    def _aggregate(sub: pd.DataFrame, scope: str, layer_id: int) -> Dict[str, float]:
        total_points = int(sub["n_samples"].sum())
        valid_points = int(sub["valid_points"].sum())
        nan_points = int(sub["nan_points"].sum())
        valid_ratio = float(valid_points / total_points) if total_points > 0 else np.nan
        weights = sub["valid_points"].to_numpy(dtype=np.float64)
        shift_weights = sub["best_shift_valid_points"].to_numpy(dtype=np.float64)

        rmse_vals = sub["rmse_mm"].to_numpy(dtype=np.float64)
        rmse_summary = np.nan
        if np.any(np.isfinite(rmse_vals) & np.isfinite(weights) & (weights > 0)):
            rmse_sq = _nanweighted_average(np.square(rmse_vals), weights)
            rmse_summary = float(np.sqrt(rmse_sq)) if np.isfinite(rmse_sq) else np.nan

        rmse_centered_vals = sub["rmse_centered_mm"].to_numpy(dtype=np.float64)
        rmse_centered_summary = np.nan
        if np.any(np.isfinite(rmse_centered_vals) & np.isfinite(weights) & (weights > 0)):
            rmse_centered_sq = _nanweighted_average(np.square(rmse_centered_vals), weights)
            rmse_centered_summary = float(np.sqrt(rmse_centered_sq)) if np.isfinite(rmse_centered_sq) else np.nan

        rmse_best_shift_vals = sub["rmse_best_shift_mm"].to_numpy(dtype=np.float64)
        rmse_best_shift_summary = np.nan
        if np.any(np.isfinite(rmse_best_shift_vals) & np.isfinite(shift_weights) & (shift_weights > 0)):
            rmse_best_shift_sq = _nanweighted_average(np.square(rmse_best_shift_vals), shift_weights)
            rmse_best_shift_summary = float(np.sqrt(rmse_best_shift_sq)) if np.isfinite(rmse_best_shift_sq) else np.nan

        row = {
            "scope": scope,
            "layer_id": int(layer_id),
            "n_profiles": int(len(sub)),
            "n_samples": total_points,
            "valid_points": valid_points,
            "nan_points": nan_points,
            "valid_ratio": valid_ratio,
            "mae_mm": _nanweighted_average(sub["mae_mm"].to_numpy(dtype=np.float64), weights),
            "rmse_mm": rmse_summary,
            "max_abs_err_mm": _nanmax_abs(sub["max_abs_err_mm"].to_numpy(dtype=np.float64)),
            "bias_mm": _nanweighted_average(sub["bias_mm"].to_numpy(dtype=np.float64), weights),
            "std_err_mm": _nanweighted_average(sub["std_err_mm"].to_numpy(dtype=np.float64), weights),
            "mae_centered_mm": _nanweighted_average(sub["mae_centered_mm"].to_numpy(dtype=np.float64), weights),
            "rmse_centered_mm": rmse_centered_summary,
            "best_shift_mm": _nanweighted_average(sub["best_shift_mm"].to_numpy(dtype=np.float64), shift_weights),
            "mae_best_shift_mm": _nanweighted_average(sub["mae_best_shift_mm"].to_numpy(dtype=np.float64), shift_weights),
            "rmse_best_shift_mm": rmse_best_shift_summary,
            "pv_pred_mm": _nanmean(sub["pv_pred_mm"].to_numpy(dtype=np.float64)),
            "pv_ref_mm": _nanmean(sub["pv_ref_mm"].to_numpy(dtype=np.float64)),
            "pv_err_mm": _nanmean(sub["pv_err_mm"].to_numpy(dtype=np.float64)),
            "pv_abs_err_mm": _nanmean(sub["pv_abs_err_mm"].to_numpy(dtype=np.float64)),
            "area_abs_err_mm2": _nanmean(sub["area_abs_err_mm2"].to_numpy(dtype=np.float64)),
            "area_signed_err_mm2": _nanmean(sub["area_signed_err_mm2"].to_numpy(dtype=np.float64)),
        }
        return row

    rows: List[Dict[str, float]] = []
    rows.append(_aggregate(detail_df, scope="overall", layer_id=-1))

    for layer_id, sub in detail_df.groupby("layer_id"):
        rows.append(_aggregate(sub, scope="by_layer", layer_id=int(layer_id)))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="对预测结果与商业扫描仪真值进行对比")
    parser.add_argument("--raw-root", default="debug_data", help="原始数据目录")
    parser.add_argument("--output-dir", default="output_debug", help="输出目录")
    args = parser.parse_args()

    cfg = ScannerConfig(raw_root=args.raw_root, output_dir=args.output_dir)
    ensure_dir(cfg.comparison_dir)

    detail_df, failed = compare_prediction_folder(cfg)
    summary_df = build_summary(detail_df)

    detail_path = os.path.join(cfg.comparison_dir, "detail_metrics.csv")
    summary_path = os.path.join(cfg.comparison_dir, "summary.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[Compare] 详细指标已保存: {detail_path}")
    print(f"[Compare] 汇总指标已保存: {summary_path}")

    if failed:
        fail_path = os.path.join(cfg.comparison_dir, "failed_cases.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"[Compare] 失败样本 {len(failed)} 个，已保存到: {fail_path}")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
