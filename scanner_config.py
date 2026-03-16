# -*- coding: utf-8 -*-
"""
更新日志
- v4
    横向几何校正配置。

    本版本将横向校正约束为低自由度、连续的 layer1 几何校正：
    1) x_shift 标签仅从 layer1 平面样本估计，不再从 layer2 为每张图自由回归；
    2) x_shift 搜索范围改为配置项，训练与评估共用同一套范围；
    3) 新增低自由度 x_shift 拟合所需的多项式/裁剪参数。
- v3
    扫描仪训练/预测统一配置。

    本版本去掉了为兼容旧流程保留的分支配置，只保留当前真实数据组织：
    - layer1/camera: 原始全图
    - layer2+/camera: 已手动替换好的固定 ROI 图

    关键改动：
    1) 显式定义“固定 ROI 全图列坐标 -> 相对物理 x(mm)”的仿射映射；
    2) train_x_grid 不再是抽象的 [-10, 10] 索引网格，而是由固定 ROI 列坐标推导出的物理 x 网格；
    3) x_center_offset_mm 仅表示本扫描仪与商业扫描仪在 x 原点上的全局偏移，不再混杂列坐标映射逻辑。
- v2
    1) 引入固定ROI坐标定义：layer1在原图上按固定ROI截取并恢复全图坐标，layer2+读取预裁剪ROI图。
    2) 增加原图尺寸、layer1/layer2图像读取策略等配置项。
    3) 保留旧 use_roi 字段仅为兼容历史代码；当前主流程不再依赖 read_gray 内部再次裁剪。
- v1
    1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScannerConfig:
    raw_root: str = "test_data"
    output_dir: str = "output_test"

    model_dirname: str = "models"
    prediction_dirname: str = "predictions"
    comparison_dirname: str = "comparisons"
    feature_cache_dirname: str = "feature_cache"

    # 固定 ROI 在原图坐标系中的位置。
    roi_x0: int = 545
    roi_x1: int = 1655
    roi_y0: int = 0
    roi_y1: int = 850

    # 原图尺寸。
    full_image_width: int = 2200
    full_image_height: int = 850

    # 条纹提取参数。
    blur_ksize: int = 5
    global_threshold_percentile: float = 78.0
    min_peak_height: float = 12.0
    min_peak_prominence: float = 2.5
    min_width_px: int = 2
    max_width_px: int = 40
    local_bg_halfwin: int = 24
    local_bg_exclude_halfwin: int = 8
    bg_low_fraction: float = 0.35
    saturation_threshold: float = 250.0
    flat_top_delta: float = 4.0

    snr_invalid_threshold: float = 2.5
    width_invalid_factor: float = 1.5
    saturation_mode_threshold: float = 0.45
    flat_top_mode_threshold: float = 0.45
    second_peak_invalid_ratio: float = 0.90
    second_peak_centroid_ratio: float = 0.55
    asymmetry_centroid_threshold: float = 0.65

    continuity_kernel: int = 5
    continuity_outlier_px: float = 10.0
    max_gap_fill_cols: int = 10
    min_valid_columns: int = 32

    # 参考平面图像级标量模型。
    plane_max_iter: int = 250
    plane_max_depth: int = 6
    plane_learning_rate: float = 0.06
    plane_min_samples_leaf: int = 40

    # 逐点 dz 映射模型。
    map_max_iter: int = 280
    map_max_depth: int = 8
    map_learning_rate: float = 0.06
    map_min_samples_leaf: int = 30

    # layer1 几何校正：低自由度 x_shift 模型。
    xshift_poly_degree: int = 2
    xshift_ridge_alpha: float = 1.0
    xshift_search_min_mm: float = -1.50
    xshift_search_max_mm: float = 1.50
    xshift_search_step_mm: float = 0.05
    xshift_clip_mm: float = 1.50

    # 统一的网格设置。
    context_halfwin: int = 2
    profile_points: int = 401
    physical_window_width_mm: float = 20.0

    output_x_min_mm: float = -10.0
    output_x_max_mm: float = 10.0
    output_points: int = 401

    # 本扫描仪与商业扫描仪之间的全局 x 原点偏移。
    x_center_offset_mm: float = -10.0

    random_state: int = 42
    test_size: float = 0.2
    log_every_n_samples: int = 100

    cache_version: str = "v7_abs_u_grid_plane_curve_layer1_xshift"

    @property
    def model_dir(self) -> str:
        return f"{self.output_dir}/{self.model_dirname}"

    @property
    def prediction_dir(self) -> str:
        return f"{self.output_dir}/{self.prediction_dirname}"

    @property
    def comparison_dir(self) -> str:
        return f"{self.output_dir}/{self.comparison_dirname}"

    @property
    def feature_cache_dir(self) -> str:
        return f"{self.output_dir}/{self.feature_cache_dirname}"

    @property
    def roi_width(self) -> int:
        return max(int(self.roi_x1 - self.roi_x0), 1)

    @property
    def roi_height(self) -> int:
        return max(int(self.roi_y1 - self.roi_y0), 1)

    @property
    def roi_u_left(self) -> float:
        return float(self.roi_x0)

    @property
    def roi_u_right(self) -> float:
        return float(self.roi_x1 - 1)

    @property
    def roi_u_center(self) -> float:
        return 0.5 * (self.roi_u_left + self.roi_u_right)

    @property
    def x_mm_per_px(self) -> float:
        denom = max(float(self.roi_width - 1), 1.0)
        return float(self.physical_window_width_mm / denom)

    def output_x_grid(self) -> np.ndarray:
        return np.linspace(
            self.output_x_min_mm,
            self.output_x_max_mm,
            self.output_points,
            dtype=np.float64,
        )

    def full_u_grid(self) -> np.ndarray:
        return np.linspace(self.roi_u_left, self.roi_u_right, self.profile_points, dtype=np.float64)

    def full_u_to_x_mm(self, u_full: np.ndarray | float) -> np.ndarray:
        u = np.asarray(u_full, dtype=np.float64)
        return (u - self.roi_u_center) * self.x_mm_per_px

    def train_x_grid(self) -> np.ndarray:
        return self.full_u_to_x_mm(self.full_u_grid())

    def clip_x_shift(self, shift_mm: float | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(shift_mm, dtype=np.float64), -abs(self.xshift_clip_mm), abs(self.xshift_clip_mm))
