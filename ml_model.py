# -*- coding: utf-8 -*-
"""
更新日志
- v3
    纯图像 -> 轮廓映射模型。

    本版本的关键改动：
    1) 训练/预测统一使用固定 ROI 的全图列坐标 u_full；
    2) 重采样输出显式 x_rel_grid(mm)，不再把物理 x 隐含在索引里；
    3) 参考平面 z0 改为逐列曲线 z0_curve(x)，而非旧版常数基线；
    4) 删除旧兼容逻辑：不再区分 local/full 两套 mapping 分支，不再保留局部宽度重采样流程。
- v2
    1) 按重构样本读取：layer2+，但在参考平面预测前恢复 u_full/v_full。
    2) 缓存中同时保存局部坐标与全图坐标元数据，plane_model 用全图宽度建模，mapping model 继续在局部ROI宽度上重采样。
    3) cache key 纳入固定ROI策略，避免旧缓存污染新逻辑。
- v1
    1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from data_reader import (
    attach_full_coordinates,
    build_meta_for_sample,
    build_reference_interpolator,
    ensure_dir,
    extract_xyz_from_pose,
    find_layer_scanner_file,
    index_camera_samples,
    load_pose_npy,
    load_scanner_profile_npy,
    pose_to_relative,
    read_gray,
    sample_reference_profile,
)
from plane_reference import PlaneReferenceModel
from scanner_config import ScannerConfig
from stripe_extractor import build_point_features, extract_stripe_profile, resample_observation


class ProfileMappingModel:
    def __init__(self, cfg: ScannerConfig, plane_model: PlaneReferenceModel, origin_pose: np.ndarray):
        self.cfg = cfg
        self.plane_model = plane_model
        self.origin_pose = origin_pose
        self.model_dz = HistGradientBoostingRegressor(
            max_iter=cfg.map_max_iter,
            max_depth=cfg.map_max_depth,
            learning_rate=cfg.map_learning_rate,
            min_samples_leaf=cfg.map_min_samples_leaf,
            random_state=cfg.random_state,
        )

    def _cache_path(self, sample) -> str:
        ensure_dir(self.cfg.feature_cache_dir)
        key = (
            f"{sample.img_path}|{self.cfg.cache_version}|{self.plane_model.cache_tag}|"
            f"{self.cfg.blur_ksize}|{self.cfg.global_threshold_percentile}|{self.cfg.min_peak_prominence}|"
            f"{self.cfg.min_valid_columns}|{self.cfg.context_halfwin}|{self.cfg.roi_x0}|{self.cfg.roi_x1}|"
            f"{self.cfg.roi_y0}|{self.cfg.roi_y1}|{self.cfg.full_image_width}|{self.cfg.full_image_height}|"
            f"{self.cfg.physical_window_width_mm}|{self.cfg.x_center_offset_mm}"
        )
        name = hashlib.md5(key.encode("utf-8")).hexdigest() + ".npz"
        return os.path.join(self.cfg.feature_cache_dir, name)

    def _extract_cached_profile(self, sample) -> Dict[str, np.ndarray]:
        cache_path = self._cache_path(sample)
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            return {k: np.asarray(data[k], dtype=np.float64) for k in data.files}

        img = read_gray(sample.img_path)
        img_proc, meta = build_meta_for_sample(img, sample, self.cfg)
        profile_local = extract_stripe_profile(img_proc, self.cfg)
        if profile_local["u"].size < self.cfg.min_valid_columns:
            raise RuntimeError("条纹有效列过少")

        profile_full = attach_full_coordinates(profile_local, meta)
        sample_npz = {k: np.asarray(v, dtype=np.float64) for k, v in profile_full.items()}
        np.savez_compressed(cache_path, **sample_npz)
        return sample_npz

    def prepare_sequence(self, sample) -> Dict[str, np.ndarray]:
        profile = self._extract_cached_profile(sample)
        resampled = resample_observation(profile=profile, cfg=self.cfg)
        x_rel_grid = np.asarray(resampled["x_rel_grid"], dtype=np.float64)
        z0_grid = self.plane_model.predict_curve(
            profile=profile,
            x_rel_grid_mm=x_rel_grid,
            image_width=int(np.asarray(profile["full_width"], dtype=np.float64).reshape(-1)[0]),
        )
        resampled["z0_grid"] = np.asarray(z0_grid, dtype=np.float64)
        feat = build_point_features(resampled, self.cfg)
        return {
            "feat": feat,
            "x_rel_grid": x_rel_grid,
            "z0_grid": z0_grid,
            "resampled": resampled,
        }

    def build_training_dataset(self, layer_dirs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_all = []
        y_dz_all = []
        z0_all = []
        z_true_all = []
        groups_all = []

        total_samples = 0
        used_samples = 0
        skipped_samples = 0
        group_id = 0

        for layer_dir in layer_dirs:
            scanner_file = find_layer_scanner_file(layer_dir)
            x_ref, z_ref = load_scanner_profile_npy(scanner_file)
            x_unique, _, fz_ref = build_reference_interpolator(x_ref, z_ref)
            x_ref_min = float(np.min(x_unique))
            x_ref_max = float(np.max(x_unique))

            camera_samples = index_camera_samples(layer_dir)
            layer_used = 0
            layer_skipped = 0

            for sample in camera_samples:
                if sample.layer_id == 1:
                    continue
                total_samples += 1
                try:
                    seq = self.prepare_sequence(sample)
                    x_rel_grid = seq["x_rel_grid"]
                    z0_grid = seq["z0_grid"]

                    pose = load_pose_npy(sample.pose_path)
                    rel_pose = pose_to_relative(pose, self.origin_pose)
                    dx, _, dz = extract_xyz_from_pose(rel_pose)
                    z_ref_local = sample_reference_profile(
                        fz=fz_ref,
                        x_ref_min=x_ref_min,
                        x_ref_max=x_ref_max,
                        center_x_global_mm=dx + self.cfg.x_center_offset_mm,
                        x_rel_grid_mm=x_rel_grid,
                    )

                    z_true = z_ref_local + dz
                    y_dz = z_true - z0_grid
                    valid_mask = np.asarray(seq["resampled"]["valid_mask"], dtype=np.float64).reshape(-1) > 0.5
                    keep = valid_mask & np.isfinite(z0_grid) & np.isfinite(z_true)
                    if np.count_nonzero(keep) < 8:
                        raise ValueError("有效训练点过少")

                    x_all.append(seq["feat"][keep])
                    y_dz_all.append(y_dz[keep].reshape(-1, 1))
                    z0_all.append(z0_grid[keep].reshape(-1, 1))
                    z_true_all.append(z_true[keep].reshape(-1, 1))
                    groups_all.append(np.full((int(np.count_nonzero(keep)), 1), group_id, dtype=np.int64))

                    layer_used += 1
                    used_samples += 1
                    group_id += 1
                except Exception as exc:
                    skipped_samples += 1
                    layer_skipped += 1
                    if layer_skipped <= 5:
                        print(f"[Skip] {sample.img_path} -> {exc}")

            layer_id = camera_samples[0].layer_id if camera_samples else -1
            print(f"[Layer] layer{layer_id}: 成功 {layer_used} 张，跳过 {layer_skipped} 张")

        if not x_all:
            raise RuntimeError("没有生成任何训练样本，请检查数据路径和格式")

        x_all = np.vstack(x_all)
        y_dz_all = np.vstack(y_dz_all).reshape(-1)
        z0_all = np.vstack(z0_all).reshape(-1)
        z_true_all = np.vstack(z_true_all).reshape(-1)
        groups_all = np.vstack(groups_all).reshape(-1)
        print(f"[Data] 训练集构建完成：点数={x_all.shape[0]}，图像成功={used_samples}，图像跳过={skipped_samples}")
        return x_all, y_dz_all, z0_all, z_true_all, groups_all

    def fit(self, layer_dirs: List[str]) -> Dict[str, float]:
        x, y_dz, z0, z_true, groups = self.build_training_dataset(layer_dirs)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )
        train_idx, val_idx = next(splitter.split(x, y_dz, groups=groups))

        x_train = x[train_idx]
        y_train = y_dz[train_idx]
        x_val = x[val_idx]
        y_val = y_dz[val_idx]
        z0_val = z0[val_idx]
        z_true_val = z_true[val_idx]

        self.model_dz.fit(x_train, y_train)

        pred_dz = self.model_dz.predict(x_val)
        pred_z = z0_val + pred_dz

        metrics = {
            "mae_dz_mm": float(mean_absolute_error(y_val, pred_dz)),
            "mae_z_mm": float(mean_absolute_error(z_true_val, pred_z)),
        }
        print(f"[Model] 验证集 MAE_dz = {metrics['mae_dz_mm']:.4f} mm")
        print(f"[Model] 验证集 MAE_z  = {metrics['mae_z_mm']:.4f} mm")
        return metrics

    def predict_one_image(self, sample, output_mid_only: bool = True) -> pd.DataFrame:
        seq = self.prepare_sequence(sample)
        dz_pred = self.model_dz.predict(seq["feat"])
        z_pred = seq["z0_grid"] + dz_pred

        df_full = pd.DataFrame(
            {
                "x_mm": seq["x_rel_grid"],
                "z_mm": z_pred,
                "z0_mm": seq["z0_grid"],
                "dz_pred_mm": dz_pred,
            }
        )
        if not output_mid_only:
            return df_full

        x_grid = self.cfg.output_x_grid()
        x_full = df_full["x_mm"].to_numpy(dtype=np.float64)
        z_full = df_full["z_mm"].to_numpy(dtype=np.float64)
        if x_grid.shape == x_full.shape and np.allclose(x_grid, x_full, atol=1e-9, rtol=0.0):
            return df_full[["x_mm", "z_mm"]].copy()
        valid = np.isfinite(x_full) & np.isfinite(z_full)
        if np.count_nonzero(valid) < 2:
            raise ValueError("有效预测点不足，无法导出曲线")
        fz = interp1d(x_full[valid], z_full[valid], kind="linear", bounds_error=False, fill_value=np.nan)
        z_mid = np.asarray(fz(x_grid), dtype=np.float64)
        return pd.DataFrame({"x_mm": x_grid, "z_mm": z_mid})

    def export_predictions(self, layer_dirs: List[str]) -> None:
        ensure_dir(self.cfg.prediction_dir)
        pred_success = 0
        pred_fail = 0

        for layer_dir in layer_dirs:
            camera_samples = index_camera_samples(layer_dir)
            if not camera_samples:
                continue
            layer_id = camera_samples[0].layer_id
            if layer_id == 1:
                continue
            out_layer_dir = os.path.join(self.cfg.prediction_dir, f"layer{layer_id}")
            ensure_dir(out_layer_dir)

            for sample in camera_samples:
                try:
                    df_mid = self.predict_one_image(sample, output_mid_only=True)
                    save_name = f"layer{layer_id}_{sample.sample_idx}_mid20mm.csv"
                    save_path = os.path.join(out_layer_dir, save_name)
                    df_mid.to_csv(save_path, index=False)
                    pred_success += 1
                except Exception as exc:
                    pred_fail += 1
                    if pred_fail <= 10:
                        print(f"[Pred-Fail] {sample.img_path} -> {exc}")

        print(f"[Pred] 导出完成：成功 {pred_success} 张，失败 {pred_fail} 张")

    def save(self) -> None:
        ensure_dir(self.cfg.model_dir)
        joblib.dump(self.model_dz, os.path.join(self.cfg.model_dir, "mapping_model.pkl"))
        print(f"[Save] 映射模型已保存到 {self.cfg.model_dir}")
