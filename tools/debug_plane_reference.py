from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from data_reader import (
    ensure_dir,
    extract_xyz_from_pose,
    find_origin_pose_path,
    get_layer_id,
    index_camera_samples,
    list_layer_dirs,
    load_pose_npy,
    pose_to_relative,
    read_gray,
)
from plane_reference import (
    IMAGE_FEATURE_KEYS,
    PLANE_FEATURE_KEYS,
    PLANE_TRAIN_FEATURE_KEYS,
    PlaneReferenceModel,
    build_image_features,
)
from scanner_config import ScannerConfig
from stripe_extractor import extract_stripe_profile


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def collect_layer1_debug_rows(cfg: ScannerConfig) -> Tuple[pd.DataFrame, List[Dict[str, object]], str, np.ndarray]:
    layer_dirs = list_layer_dirs(cfg.raw_root)
    layer1_dir = None
    for d in layer_dirs:
        if get_layer_id(d) == 1:
            layer1_dir = d
            break
    if layer1_dir is None:
        raise RuntimeError(f"未找到 layer1: {cfg.raw_root}")

    origin_pose = load_pose_npy(find_origin_pose_path(cfg.raw_root))
    samples = index_camera_samples(layer1_dir)
    if not samples:
        raise RuntimeError(f"{layer1_dir} 中未找到相机样本")

    rows: List[Dict[str, object]] = []
    kept_samples: List[Dict[str, object]] = []

    for s in samples:
        img = read_gray(s["img_path"], cfg)
        h, w = img.shape[:2]
        pose = load_pose_npy(s["pose_path"])
        rel_pose = pose_to_relative(pose, origin_pose)
        dx, dy, dz = extract_xyz_from_pose(rel_pose)

        try:
            profile = extract_stripe_profile(img, cfg)
        except Exception as exc:
            rows.append(
                {
                    "sample_idx": s["sample_idx"],
                    "img_path": s["img_path"],
                    "pose_path": s["pose_path"],
                    "height": h,
                    "width": w,
                    "dx_mm": dx,
                    "dy_mm": dy,
                    "dz_mm": dz,
                    "status": f"extract_error: {exc}",
                    "valid_cols": 0.0,
                    "valid_ratio": 0.0,
                }
            )
            continue

        feat = build_image_features(profile, image_width=w)
        status = "ok" if int(feat["valid_cols"]) >= cfg.min_valid_columns else "too_few_valid_cols"

        row = {
            "sample_idx": s["sample_idx"],
            "img_path": s["img_path"],
            "pose_path": s["pose_path"],
            "height": h,
            "width": w,
            "dx_mm": dx,
            "dy_mm": dy,
            "dz_mm": dz,
            "status": status,
            **feat,
        }
        rows.append(row)

        if status == "ok":
            kept_samples.append({"sample": s, "profile": profile, "row": row, "image_width": w})

    df = pd.DataFrame(rows)
    return df, kept_samples, layer1_dir, origin_pose


def _fit_and_report(name: str, model, x_train, y_train, x_val, y_val) -> Dict[str, float]:
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    return {
        "model": name,
        "train_mae_mm": float(mean_absolute_error(y_train, pred_train)),
        "val_mae_mm": float(mean_absolute_error(y_val, pred_val)),
        "val_r2": float(r2_score(y_val, pred_val)) if y_val.size >= 2 else np.nan,
    }


def evaluate_image_level_models(df_ok: pd.DataFrame, cfg: ScannerConfig) -> pd.DataFrame:
    if len(df_ok) < 8:
        raise RuntimeError("有效图像过少，无法做图像级验证")

    train_df, val_df = train_test_split(
        df_ok,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    y_train = train_df["dz_mm"].to_numpy(dtype=np.float64)
    y_val = val_df["dz_mm"].to_numpy(dtype=np.float64)
    out = []

    baseline_pred = np.full_like(y_val, np.mean(y_train), dtype=np.float64)
    out.append(
        {
            "model": "constant_mean",
            "train_mae_mm": float(mean_absolute_error(y_train, np.full_like(y_train, np.mean(y_train)))),
            "val_mae_mm": float(mean_absolute_error(y_val, baseline_pred)),
            "val_r2": float(r2_score(y_val, baseline_pred)) if y_val.size >= 2 else np.nan,
        }
    )

    for key in ["v_mean", "v_median"]:
        x_train = train_df[[key]].to_numpy(dtype=np.float64)
        x_val = val_df[[key]].to_numpy(dtype=np.float64)
        out.append(_fit_and_report(f"linear_{key}", LinearRegression(), x_train, y_train, x_val, y_val))

    x_train_official = train_df[PLANE_TRAIN_FEATURE_KEYS].to_numpy(dtype=np.float64)
    x_val_official = val_df[PLANE_TRAIN_FEATURE_KEYS].to_numpy(dtype=np.float64)
    out.append(_fit_and_report("official_plane_linear", LinearRegression(), x_train_official, y_train, x_val_official, y_val))

    x_train_hgbr = train_df[IMAGE_FEATURE_KEYS].to_numpy(dtype=np.float64)
    x_val_hgbr = val_df[IMAGE_FEATURE_KEYS].to_numpy(dtype=np.float64)
    hgbr = HistGradientBoostingRegressor(
        max_iter=cfg.plane_max_iter,
        max_depth=cfg.plane_max_depth,
        learning_rate=cfg.plane_learning_rate,
        min_samples_leaf=max(3, min(cfg.plane_min_samples_leaf, len(train_df) // 5 if len(train_df) >= 10 else 3)),
        random_state=cfg.random_state,
    )
    out.append(_fit_and_report("hgbr_image_stats", hgbr, x_train_hgbr, y_train, x_val_hgbr, y_val))

    return pd.DataFrame(out)


def _build_column_features(profile: Dict[str, np.ndarray]) -> np.ndarray:
    cols = [np.asarray(profile[k], dtype=np.float64).reshape(-1) for k in PLANE_FEATURE_KEYS]
    lengths = [c.size for c in cols]
    if len(set(lengths)) != 1:
        raise ValueError("参考平面特征长度不一致")
    return np.column_stack(cols)


def evaluate_column_level_model(kept_samples: List[Dict[str, object]], cfg: ScannerConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(kept_samples) < 8:
        raise RuntimeError("有效图像过少，无法做逐列验证")

    xs = []
    ys = []
    groups = []
    sample_rows = []

    for gid, item in enumerate(kept_samples):
        row = item["row"]
        profile = item["profile"]
        feat = _build_column_features(profile)
        target = np.full(feat.shape[0], float(row["dz_mm"]), dtype=np.float64)
        xs.append(feat)
        ys.append(target)
        groups.append(np.full(feat.shape[0], gid, dtype=np.int64))
        sample_rows.append(
            {
                "group_id": gid,
                "sample_idx": int(row["sample_idx"]),
                "img_path": row["img_path"],
                "dz_mm": float(row["dz_mm"]),
                "n_cols": feat.shape[0],
            }
        )

    x = np.vstack(xs)
    y = np.concatenate(ys)
    group_arr = np.concatenate(groups)
    group_df = pd.DataFrame(sample_rows)

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, val_idx = next(splitter.split(x, y, groups=group_arr))

    model = HistGradientBoostingRegressor(
        max_iter=cfg.plane_max_iter,
        max_depth=cfg.plane_max_depth,
        learning_rate=cfg.plane_learning_rate,
        min_samples_leaf=cfg.plane_min_samples_leaf,
        random_state=cfg.random_state,
    )
    model.fit(x[train_idx], y[train_idx])
    pred_train = model.predict(x[train_idx])
    pred_val = model.predict(x[val_idx])

    summary_rows = [
        {
            "model": "hgbr_column_group_split",
            "train_mae_mm": float(mean_absolute_error(y[train_idx], pred_train)),
            "val_mae_mm": float(mean_absolute_error(y[val_idx], pred_val)),
            "val_r2": float(r2_score(y[val_idx], pred_val)) if val_idx.size >= 2 else np.nan,
        }
    ]

    val_group_ids = np.unique(group_arr[val_idx])
    image_level_rows = []
    row_ids = np.arange(group_arr.size)
    for gid in val_group_ids:
        mask = group_arr == gid
        mask_val = mask & np.isin(row_ids, val_idx)
        true_dz = float(np.mean(y[mask_val]))
        pred_dz_mean = float(np.mean(model.predict(x[mask])))
        sample_row = group_df[group_df["group_id"] == gid].iloc[0]
        image_level_rows.append(
            {
                "group_id": int(gid),
                "sample_idx": int(sample_row["sample_idx"]),
                "img_path": sample_row["img_path"],
                "true_dz_mm": true_dz,
                "pred_dz_mm": pred_dz_mean,
                "abs_err_mm": abs(pred_dz_mean - true_dz),
                "n_cols": int(sample_row["n_cols"]),
            }
        )

    image_level_df = pd.DataFrame(image_level_rows).sort_values("sample_idx").reset_index(drop=True)
    if not image_level_df.empty:
        summary_rows.append(
            {
                "model": "hgbr_column_group_split_image_mean",
                "train_mae_mm": np.nan,
                "val_mae_mm": float(image_level_df["abs_err_mm"].mean()),
                "val_r2": float(r2_score(image_level_df["true_dz_mm"], image_level_df["pred_dz_mm"])) if len(image_level_df) >= 2 else np.nan,
            }
        )

    return pd.DataFrame(summary_rows), image_level_df


def evaluate_official_model_predictions(
    kept_samples: List[Dict[str, object]], cfg: ScannerConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(kept_samples) < 8:
        raise RuntimeError("有效图像过少，无法做官方参考平面模型验证")

    rows = []
    for item in kept_samples:
        row = item["row"]
        feat_dict = {k: float(row[k]) for k in IMAGE_FEATURE_KEYS}
        rows.append(
            {
                "sample_idx": int(row["sample_idx"]),
                "img_path": row["img_path"],
                "dz_mm": float(row["dz_mm"]),
                **feat_dict,
            }
        )
    feat_df = pd.DataFrame(rows)

    train_df, val_df = train_test_split(
        feat_df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    model = LinearRegression()
    model.fit(train_df[PLANE_TRAIN_FEATURE_KEYS].to_numpy(dtype=np.float64), train_df["dz_mm"].to_numpy(dtype=np.float64))
    pred_train = model.predict(train_df[PLANE_TRAIN_FEATURE_KEYS].to_numpy(dtype=np.float64))
    pred_val = model.predict(val_df[PLANE_TRAIN_FEATURE_KEYS].to_numpy(dtype=np.float64))

    metrics_df = pd.DataFrame(
        [
            {
                "model": "official_plane_linear",
                "train_mae_mm": float(mean_absolute_error(train_df["dz_mm"], pred_train)),
                "val_mae_mm": float(mean_absolute_error(val_df["dz_mm"], pred_val)),
                "val_r2": float(r2_score(val_df["dz_mm"], pred_val)) if len(val_df) >= 2 else np.nan,
            }
        ]
    )

    pred_df = val_df[["sample_idx", "img_path", "dz_mm"]].copy()
    pred_df = pred_df.rename(columns={"dz_mm": "true_dz_mm"})
    pred_df["pred_dz_mm"] = pred_val
    pred_df["abs_err_mm"] = np.abs(pred_df["pred_dz_mm"] - pred_df["true_dz_mm"])
    pred_df = pred_df.sort_values("sample_idx").reset_index(drop=True)
    return metrics_df, pred_df


def fit_official_model_on_all(cfg: ScannerConfig, origin_pose: np.ndarray, layer1_dir: str) -> Dict[str, float]:
    model = PlaneReferenceModel(cfg)
    model.fit(layer1_dir=layer1_dir, origin_pose=origin_pose)
    return dict(model.train_metrics)


def save_scatter(df_ok: pd.DataFrame, out_dir: str) -> None:
    plots = [
        ("v_mean", "dz_mm", "dz vs v_mean"),
        ("v_median", "dz_mm", "dz vs v_median"),
        ("valid_ratio", "dz_mm", "dz vs valid_ratio"),
        ("width_mean", "dz_mm", "dz vs width_mean"),
        ("snr_mean", "dz_mm", "dz vs snr_mean"),
        ("dx_mm", "v_median", "v_median vs dx"),
        ("dy_mm", "v_median", "v_median vs dy"),
        ("dz_mm", "v_median", "v_median vs dz"),
    ]
    for x_key, y_key, title in plots:
        if x_key not in df_ok.columns or y_key not in df_ok.columns:
            continue
        d = df_ok[[x_key, y_key]].dropna()
        if len(d) < 3:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.scatter(d[x_key], d[y_key], s=18)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(title)
        fig.tight_layout()
        save_path = os.path.join(out_dir, f"scatter_{x_key}_vs_{y_key}.png")
        fig.savefig(save_path)
        plt.close(fig)


def print_core_summary(
    df_all: pd.DataFrame,
    df_ok: pd.DataFrame,
    image_model_df: pd.DataFrame,
    official_metrics_df: pd.DataFrame,
    column_model_df: pd.DataFrame,
    official_fit_metrics: Dict[str, float],
) -> None:
    print("\n=== 基本统计 ===")
    print(f"总样本数: {len(df_all)}")
    print(f"有效样本数: {len(df_ok)}")
    print(f"有效率: {len(df_ok) / max(len(df_all), 1):.4f}")
    print(
        f'valid_cols: mean={df_ok["valid_cols"].mean():.2f}, '
        f'median={df_ok["valid_cols"].median():.2f}, '
        f'min={df_ok["valid_cols"].min():.0f}, max={df_ok["valid_cols"].max():.0f}'
    )
    print(f'valid_ratio: mean={df_ok["valid_ratio"].mean():.4f}, median={df_ok["valid_ratio"].median():.4f}')
    print(f'dz range: [{df_ok["dz_mm"].min():.4f}, {df_ok["dz_mm"].max():.4f}] mm')

    print("\n=== 相关性检查（越高越好）===")
    print(f'corr(dz, v_mean)   = {_corr(df_ok["dz_mm"], df_ok["v_mean"]):.6f}')
    print(f'corr(dz, v_median) = {_corr(df_ok["dz_mm"], df_ok["v_median"]):.6f}')
    print(f'corr(dx, v_median) = {_corr(df_ok["dx_mm"], df_ok["v_median"]):.6f}')
    print(f'corr(dy, v_median) = {_corr(df_ok["dy_mm"], df_ok["v_median"]):.6f}')
    print(f'corr(dz, valid_ratio) = {_corr(df_ok["dz_mm"], df_ok["valid_ratio"]):.6f}')

    print("\n=== 图像级模型对比 ===")
    print(image_model_df.to_string(index=False))

    print("\n=== 当前正式参考平面模型 ===")
    print(official_metrics_df.to_string(index=False))
    if official_fit_metrics:
        print(pd.Series(official_fit_metrics).to_string())

    print("\n=== 旧逐列模型（用于对照，不建议继续用）===")
    print(column_model_df.to_string(index=False))

    corr_dz = abs(_corr(df_ok["dz_mm"], df_ok["v_median"]))
    corr_dx = abs(_corr(df_ok["dx_mm"], df_ok["v_median"]))
    corr_dy = abs(_corr(df_ok["dy_mm"], df_ok["v_median"]))
    best_axis = max([("dx", corr_dx), ("dy", corr_dy), ("dz", corr_dz)], key=lambda t: (np.nan_to_num(t[1], nan=-1.0), t[0]))[0]

    print("\n=== 初步判断 ===")
    if np.isfinite(corr_dz) and corr_dz < 0.5:
        print("- dz 与 v_median 相关性偏弱。优先怀疑位姿标签、坐标轴定义或条纹中心提取。")
    if best_axis != "dz":
        print(f"- v_median 与 {best_axis} 的相关性强于 dz，优先检查 rob_poz.npy 的坐标定义和 z 轴解释。")

    img_best = float(image_model_df["val_mae_mm"].min()) if not image_model_df.empty else np.nan
    official_val = float(official_metrics_df["val_mae_mm"].iloc[0]) if not official_metrics_df.empty else np.nan
    col_val = float(column_model_df["val_mae_mm"].dropna().min()) if not column_model_df.empty else np.nan
    if np.isfinite(official_val) and np.isfinite(col_val) and official_val < col_val:
        print("- 当前正式方案应坚持图像级标定。逐列监督在参考平面阶段会把列内形态差异误当成深度信息。")
    if np.isfinite(img_best) and np.isfinite(official_val) and official_val <= img_best + 1e-9:
        print("- 正式模型已经和 debug 中最优图像级基线对齐，可以直接替换旧 plane_reference.py。")
    elif np.isfinite(img_best) and np.isfinite(official_val):
        print("- 正式模型略差于最优 debug 基线，可继续尝试二次多项式/分段标定，但不要退回逐列方案。")


def main() -> None:
    parser = argparse.ArgumentParser(description="参考平面模型 debug 脚本（图像级版本）")
    parser.add_argument("--raw-root", default="raw_data", help="原始数据根目录")
    parser.add_argument("--output-dir", default="debug_plane_reference2", help="调试输出目录")
    args = parser.parse_args()

    cfg = ScannerConfig(raw_root=args.raw_root)
    debug_dir = args.output_dir
    ensure_dir(debug_dir)

    print("=== 配置 ===")
    print(pd.Series(asdict(cfg)).to_string())

    df_all, kept_samples, layer1_dir, origin_pose = collect_layer1_debug_rows(cfg)
    df_all.to_csv(os.path.join(debug_dir, "layer1_debug_all.csv"), index=False, encoding="utf-8-sig")

    df_ok = df_all[df_all["status"] == "ok"].copy().reset_index(drop=True)
    df_ok.to_csv(os.path.join(debug_dir, "layer1_debug_ok.csv"), index=False, encoding="utf-8-sig")

    if df_ok.empty:
        raise RuntimeError("没有任何有效样本，无法继续 debug")

    image_model_df = evaluate_image_level_models(df_ok, cfg)
    image_model_df.to_csv(os.path.join(debug_dir, "image_level_metrics.csv"), index=False, encoding="utf-8-sig")

    official_metrics_df, official_pred_df = evaluate_official_model_predictions(kept_samples, cfg)
    official_metrics_df.to_csv(os.path.join(debug_dir, "official_plane_metrics.csv"), index=False, encoding="utf-8-sig")
    official_pred_df.to_csv(os.path.join(debug_dir, "official_plane_val_predictions.csv"), index=False, encoding="utf-8-sig")

    column_model_df, image_pred_df = evaluate_column_level_model(kept_samples, cfg)
    column_model_df.to_csv(os.path.join(debug_dir, "column_level_metrics.csv"), index=False, encoding="utf-8-sig")
    image_pred_df.to_csv(os.path.join(debug_dir, "column_level_val_image_predictions.csv"), index=False, encoding="utf-8-sig")

    official_fit_metrics = fit_official_model_on_all(cfg, origin_pose=origin_pose, layer1_dir=layer1_dir)
    pd.Series(official_fit_metrics).to_csv(
        os.path.join(debug_dir, "official_plane_fit_all_metrics.csv"),
        header=False,
        encoding="utf-8-sig",
    )

    save_scatter(df_ok, debug_dir)

    print(f"\nlayer1 dir: {layer1_dir}")
    print(f"输出目录: {os.path.abspath(debug_dir)}")
    print_core_summary(df_all, df_ok, image_model_df, official_metrics_df, column_model_df, official_fit_metrics)


if __name__ == "__main__":
    main()