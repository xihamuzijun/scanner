# -*- coding: utf-8 -*-
"""
更新日志
- v4
    在 layer1 上增加低自由度横向几何校正模型。

    关键改动：
    1) scalar_model 仍负责预测图像级 dz，但新增 x_shift_model，只在 layer1 平面样本上拟合；
    2) x_shift 标签通过“预测平面曲线 vs 商业扫描仪平面真值”的 best-shift 得到，避免用 layer2 物体形状去污染几何校正；
    3) x_shift_model 采用低自由度连续模型（多项式 + Ridge），只使用少量物理相关特征，避免每样本自由平移式过拟合。
- v3
    参考平面模型。

    本版本将参考平面拆成两部分：
    1) 图像级标量模型：预测当前图像相对 origin 的整体 dz；
    2) layer1 商业扫描仪基线曲线：提供固定 ROI 对应的 z0(x) 基线形状。

    最终 z0_curve(x) = base_plane_curve(x) + dz_scalar。
    这比旧版本“整条曲线共用一个标量 z0”更接近真实几何基线。
- v2
    1) 按方案A重构参考平面输入：layer1使用“原图上固定ROI提条纹 + 全图坐标恢复”。
    2) 参考平面特征优先使用 u_full/v_full，从而学习视场绝对位置相关的系统偏差。
    3) 训练特征中加入 u_left/u_right/u_span 等位置项，并更新 cache_tag。
- v1
    1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from data_reader import (
    attach_full_coordinates,
    build_meta_for_sample,
    build_reference_interpolator,
    extract_xyz_from_pose,
    find_layer_scanner_file,
    index_camera_samples,
    load_pose_npy,
    load_scanner_profile_npy,
    pose_to_relative,
    read_gray,
    sample_reference_profile,
)
from scanner_config import ScannerConfig
from stripe_extractor import extract_stripe_profile


PLANE_TRAIN_FEATURE_KEYS: List[str] = [
    "valid_ratio",
    "u_left",
    "u_right",
    "u_span",
    "v_mean",
    "v_std",
    "quality_mean",
    "width_mean",
    "contrast_mean",
    "snr_mean",
]

XSHIFT_FEATURE_KEYS: List[str] = [
    "dz_scalar_pred_mm",
    "u_center_px",
    "u_span_px",
]


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x))


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.nan
    return float(np.nanstd(x))


def _shifted_profile_metrics(x: np.ndarray, z_pred: np.ndarray, z_ref: np.ndarray, shift_mm: float) -> Tuple[float, float, int]:
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
        "best_shift_mm": float(cfg.clip_x_shift(best_shift)) if np.isfinite(best_shift) else np.nan,
        "mae_best_shift_mm": best_mae,
        "rmse_best_shift_mm": best_rmse,
        "best_shift_valid_points": best_valid,
    }


def build_image_features(profile: Dict[str, np.ndarray], image_width: int) -> Dict[str, float]:
    u = np.asarray(profile.get("u_full", []), dtype=np.float64).reshape(-1)
    v = np.asarray(profile.get("v_full", []), dtype=np.float64).reshape(-1)
    q = np.asarray(profile.get("quality", []), dtype=np.float64).reshape(-1)

    feat = {
        "valid_ratio": float(u.size / max(int(image_width), 1)),
        "u_left": float(np.min(u)) if u.size else np.nan,
        "u_right": float(np.max(u)) if u.size else np.nan,
        "u_span": float(np.max(u) - np.min(u)) if u.size else np.nan,
        "v_mean": _safe_mean(v),
        "v_std": _safe_std(v),
        "quality_mean": _safe_mean(q),
        "width_mean": _safe_mean(profile.get("width_px", [])),
        "contrast_mean": _safe_mean(profile.get("contrast", [])),
        "snr_mean": _safe_mean(profile.get("snr", [])),
    }
    return feat


class PlaneReferenceModel:
    def __init__(self, cfg: ScannerConfig):
        self.cfg = cfg
        self.scalar_model = HistGradientBoostingRegressor(
            max_iter=cfg.plane_max_iter,
            max_depth=cfg.plane_max_depth,
            learning_rate=cfg.plane_learning_rate,
            min_samples_leaf=cfg.plane_min_samples_leaf,
            random_state=cfg.random_state,
        )
        self.xshift_model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=cfg.xshift_poly_degree, include_bias=False)),
                ("ridge", Ridge(alpha=cfg.xshift_ridge_alpha)),
            ]
        )
        self.feature_keys = list(PLANE_TRAIN_FEATURE_KEYS)
        self.xshift_feature_keys = list(XSHIFT_FEATURE_KEYS)
        self.train_metrics: Dict[str, float] = {}
        self.base_curve_x_mm = cfg.train_x_grid()
        self.base_curve_z_mm = np.full_like(self.base_curve_x_mm, np.nan, dtype=np.float64)
        self.cache_tag = "plane_curve_v3_layer1_poly_xshift"
        self.xshift_enabled = False

    def _feature_dict_to_row(self, feature_dict: Dict[str, float]) -> np.ndarray:
        row = np.asarray([feature_dict[k] for k in self.feature_keys], dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(row)):
            bad = [k for k in self.feature_keys if not np.isfinite(feature_dict[k])]
            raise ValueError(f"参考平面图像级特征存在非法值: {bad}")
        return row

    def build_feature_row(self, profile: Dict[str, np.ndarray], image_width: int) -> np.ndarray:
        feature_dict = build_image_features(profile, image_width=image_width)
        return self._feature_dict_to_row(feature_dict)

    def _build_xshift_feature_row_from_feature_dict(self, feat: Dict[str, float], dz_scalar_pred_mm: float) -> np.ndarray:
        u_center_px = 0.5 * (float(feat["u_left"]) + float(feat["u_right"]))
        row = np.asarray(
            [
                float(dz_scalar_pred_mm),
                float(u_center_px),
                float(feat["u_span"]),
            ],
            dtype=np.float64,
        ).reshape(1, -1)
        if not np.all(np.isfinite(row)):
            raise ValueError("x_shift 特征存在非法值")
        return row

    def _build_xshift_feature_row(self, profile: Dict[str, np.ndarray], image_width: int, dz_scalar_pred_mm: float) -> np.ndarray:
        feat = build_image_features(profile, image_width=image_width)
        return self._build_xshift_feature_row_from_feature_dict(feat, dz_scalar_pred_mm)

    def _prepare_profile(self, sample) -> tuple[Dict[str, np.ndarray], int]:
        img = read_gray(sample.img_path)
        img_proc, meta = build_meta_for_sample(img, sample, self.cfg)
        profile_local = extract_stripe_profile(img_proc, self.cfg)
        if profile_local["u"].size < self.cfg.min_valid_columns:
            raise ValueError("条纹有效列过少")
        profile = attach_full_coordinates(profile_local, meta)
        return profile, meta.full_width

    def _collect_training_data(self, layer1_dir: str, origin_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        samples = index_camera_samples(layer1_dir)
        if not samples:
            raise RuntimeError(f"{layer1_dir} 中未找到 layer1 相机样本")

        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        total = 0
        used = 0

        for sample in samples:
            total += 1
            try:
                profile, full_width = self._prepare_profile(sample)
            except Exception:
                continue

            pose = load_pose_npy(sample.pose_path)
            rel_pose = pose_to_relative(pose, origin_pose)
            _, _, dz = extract_xyz_from_pose(rel_pose)

            try:
                feat_row = self.build_feature_row(profile, image_width=full_width)
            except ValueError:
                continue

            x_rows.append(feat_row)
            y_rows.append(float(dz))
            used += 1

        if not x_rows:
            raise RuntimeError("layer1 中没有可用于建立参考平面的有效样本")

        x = np.vstack(x_rows)
        y = np.asarray(y_rows, dtype=np.float64)
        return x, y, used, total - used

    def _fit_base_plane_curve(self, layer1_dir: str) -> None:
        scanner_file = find_layer_scanner_file(layer1_dir)
        x_ref, z_ref = load_scanner_profile_npy(scanner_file)
        x_unique, _, fz_ref = build_reference_interpolator(x_ref, z_ref)
        base_curve = sample_reference_profile(
            fz=fz_ref,
            x_ref_min=float(np.min(x_unique)),
            x_ref_max=float(np.max(x_unique)),
            center_x_global_mm=float(self.cfg.x_center_offset_mm),
            x_rel_grid_mm=self.base_curve_x_mm,
        )
        self.base_curve_z_mm = np.asarray(base_curve, dtype=np.float64)

    def _fit_xshift_model(self, layer1_dir: str, origin_pose: np.ndarray) -> Dict[str, float]:
        scanner_file = find_layer_scanner_file(layer1_dir)
        x_ref, z_ref = load_scanner_profile_npy(scanner_file)
        x_unique, _, fz_ref = build_reference_interpolator(x_ref, z_ref)
        x_ref_min = float(np.min(x_unique))
        x_ref_max = float(np.max(x_unique))

        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        boundary_hits = 0
        skipped = 0

        for sample in index_camera_samples(layer1_dir):
            try:
                profile, full_width = self._prepare_profile(sample)
                feat_dict = build_image_features(profile, image_width=full_width)
                feat_row = self._feature_dict_to_row(feat_dict)
                dz_scalar_pred = float(np.asarray(self.scalar_model.predict(feat_row), dtype=np.float64).reshape(-1)[0])
                xshift_row = self._build_xshift_feature_row_from_feature_dict(feat_dict, dz_scalar_pred_mm=dz_scalar_pred)
                z_pred_curve = np.asarray(self.base_curve_z_mm, dtype=np.float64) + dz_scalar_pred

                pose = load_pose_npy(sample.pose_path)
                rel_pose = pose_to_relative(pose, origin_pose)
                dx, _, dz = extract_xyz_from_pose(rel_pose)
                z_ref_local = sample_reference_profile(
                    fz=fz_ref,
                    x_ref_min=x_ref_min,
                    x_ref_max=x_ref_max,
                    center_x_global_mm=dx + self.cfg.x_center_offset_mm,
                    x_rel_grid_mm=self.base_curve_x_mm,
                )
                z_true = z_ref_local + dz
                shift_info = _find_best_shift(self.cfg, self.base_curve_x_mm, z_pred_curve, z_true)
                shift = shift_info["best_shift_mm"]
                if not np.isfinite(shift):
                    skipped += 1
                    continue
                if abs(shift - self.cfg.xshift_search_min_mm) < 1e-9 or abs(shift - self.cfg.xshift_search_max_mm) < 1e-9:
                    boundary_hits += 1
                x_rows.append(xshift_row)
                y_rows.append(float(shift))
            except Exception:
                skipped += 1
                continue

        metrics: Dict[str, float] = {
            "xshift_label_count": float(len(y_rows)),
            "xshift_boundary_hits": float(boundary_hits),
            "xshift_skipped": float(skipped),
        }
        if len(y_rows) < 6:
            self.xshift_enabled = False
            metrics.update(
                {
                    "xshift_label_abs_mean_mm": np.nan,
                    "xshift_fit_mae_mm": np.nan,
                }
            )
            print("[Plane] layer1 有效 x_shift 标签过少，跳过横向几何校正")
            return metrics

        x_shift_train = np.vstack(x_rows)
        y_shift_train = np.asarray(y_rows, dtype=np.float64)
        self.xshift_model.fit(x_shift_train, y_shift_train)
        pred_shift = np.asarray(self.xshift_model.predict(x_shift_train), dtype=np.float64)
        pred_shift = np.asarray(self.cfg.clip_x_shift(pred_shift), dtype=np.float64)
        self.xshift_enabled = True
        metrics.update(
            {
                "xshift_label_abs_mean_mm": float(np.mean(np.abs(y_shift_train))),
                "xshift_fit_mae_mm": float(mean_absolute_error(y_shift_train, pred_shift)),
            }
        )
        print(
            "[Plane] layer1 横向几何校正训练完成："
            f"样本={int(metrics['xshift_label_count'])}，"
            f"标签平均绝对值={metrics['xshift_label_abs_mean_mm']:.6f} mm，"
            f"拟合 MAE={metrics['xshift_fit_mae_mm']:.6f} mm，"
            f"边界命中={int(metrics['xshift_boundary_hits'])}"
        )
        return metrics

    def fit(self, layer1_dir: str, origin_pose: np.ndarray) -> None:
        x, y, used, skipped = self._collect_training_data(layer1_dir, origin_pose)
        if y.size < 4:
            raise RuntimeError(f"参考平面有效样本过少: {y.size}")

        metrics: Dict[str, float] = {
            "n_samples": float(y.size),
            "dz_min_mm": float(np.min(y)),
            "dz_max_mm": float(np.max(y)),
        }

        if y.size >= 8:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
                shuffle=True,
            )
            eval_model = HistGradientBoostingRegressor(
                max_iter=self.cfg.plane_max_iter,
                max_depth=self.cfg.plane_max_depth,
                learning_rate=self.cfg.plane_learning_rate,
                min_samples_leaf=self.cfg.plane_min_samples_leaf,
                random_state=self.cfg.random_state,
            )
            eval_model.fit(x_train, y_train)
            pred_train = eval_model.predict(x_train)
            pred_val = eval_model.predict(x_val)
            metrics.update(
                {
                    "split_train_mae_mm": float(mean_absolute_error(y_train, pred_train)),
                    "split_val_mae_mm": float(mean_absolute_error(y_val, pred_val)),
                    "split_val_r2": float(r2_score(y_val, pred_val)) if y_val.size >= 2 else np.nan,
                }
            )

        self.scalar_model.fit(x, y)
        pred_full = self.scalar_model.predict(x)
        metrics["fit_all_mae_mm"] = float(mean_absolute_error(y, pred_full))
        self._fit_base_plane_curve(layer1_dir)
        metrics["base_curve_pv_mm"] = float(np.nanmax(self.base_curve_z_mm) - np.nanmin(self.base_curve_z_mm))
        metrics.update(self._fit_xshift_model(layer1_dir, origin_pose))
        self.train_metrics = metrics

        print(
            "[Plane] 参考平面训练完成："
            f"样本={int(metrics['n_samples'])}，标量 dz 训练 MAE={metrics['fit_all_mae_mm']:.6f} mm，"
            f"基线曲线 PV={metrics['base_curve_pv_mm']:.6f} mm"
        )
        if "split_val_mae_mm" in metrics:
            print(
                "[Plane] 保留集评估："
                f"train MAE={metrics['split_train_mae_mm']:.6f} mm，"
                f"val MAE={metrics['split_val_mae_mm']:.6f} mm，"
                f"val R2={metrics['split_val_r2']:.6f}"
            )
        if skipped > 0:
            print(f"[Plane] 跳过样本 {skipped} 张")

    def predict_scalar(self, profile: Dict[str, np.ndarray], image_width: Optional[int] = None) -> float:
        u = np.asarray(profile.get("u_full", []), dtype=np.float64).reshape(-1)
        if u.size == 0:
            raise ValueError("参考平面预测失败：profile 为空")
        if image_width is None:
            image_width = int(np.nanmax(profile.get("full_width", np.array([self.cfg.full_image_width]))))
        feat_row = self.build_feature_row(profile, image_width=image_width)
        pred = float(np.asarray(self.scalar_model.predict(feat_row), dtype=np.float64).reshape(-1)[0])
        return pred

    def predict_x_shift(self, profile: Dict[str, np.ndarray], image_width: Optional[int] = None) -> float:
        if not self.xshift_enabled:
            return 0.0
        if image_width is None:
            image_width = int(np.nanmax(profile.get("full_width", np.array([self.cfg.full_image_width]))))
        dz_scalar_pred = self.predict_scalar(profile=profile, image_width=image_width)
        feat_row = self._build_xshift_feature_row(profile, image_width=image_width, dz_scalar_pred_mm=dz_scalar_pred)
        pred = float(np.asarray(self.xshift_model.predict(feat_row), dtype=np.float64).reshape(-1)[0])
        return float(self.cfg.clip_x_shift(pred))

    def predict_curve(self, profile: Dict[str, np.ndarray], x_rel_grid_mm: np.ndarray, image_width: Optional[int] = None) -> np.ndarray:
        if self.base_curve_z_mm.size == 0 or not np.all(np.isfinite(self.base_curve_z_mm)):
            raise RuntimeError("参考平面基线曲线尚未建立，请先执行 fit()")
        dz_scalar = self.predict_scalar(profile=profile, image_width=image_width)
        z_base = np.interp(
            np.asarray(x_rel_grid_mm, dtype=np.float64),
            self.base_curve_x_mm,
            self.base_curve_z_mm,
            left=np.nan,
            right=np.nan,
        )
        return z_base + dz_scalar

    def save(self, path: str) -> None:
        payload = {
            "scalar_model": self.scalar_model,
            "xshift_model": self.xshift_model,
            "xshift_enabled": self.xshift_enabled,
            "feature_keys": self.feature_keys,
            "xshift_feature_keys": self.xshift_feature_keys,
            "train_metrics": self.train_metrics,
            "base_curve_x_mm": self.base_curve_x_mm,
            "base_curve_z_mm": self.base_curve_z_mm,
            "cache_tag": self.cache_tag,
        }
        joblib.dump(payload, path)
