# -*- coding: utf-8 -*-
"""
更新日志
- v3
    数据读取与坐标元数据。

    当前只支持一种数据组织：
    - layer1/camera: 原始全图。训练/预测前按固定 ROI 截取，再恢复全图列坐标。
    - layer2+/camera: 已经手动替换好的固定 ROI 图。读取后按固定 ROI 偏移恢复全图列坐标。

    不再保留 camera_cropped、自动二次裁剪、旧兼容分支。
- v2
    1) 增加方案A的数据读取接口：layer1使用原图+固定ROI信息，layer2+优读取裁剪后的图像。
    2) 增加图像/坐标元数据 dataclass，并支持将ROI内局部坐标恢复到原图全幅坐标。
    3) read_gray 不再在内部执行二次ROI裁剪，避免与预处理后的重复裁剪。
- v1
    1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d

from scanner_config import ScannerConfig


@dataclass
class ImageCoordMeta:
    full_width: int
    full_height: int
    offset_x: int
    offset_y: int
    local_width: int
    local_height: int
    source_path: str


@dataclass
class CameraSample:
    layer_id: int
    sample_idx: int
    img_path: str
    pose_path: str
    is_layer1_full_image: bool


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_layer_dirs(raw_root: str) -> List[str]:
    layer_dirs = glob.glob(os.path.join(raw_root, "layer*"))
    layer_dirs = [d for d in layer_dirs if os.path.isdir(d)]
    return sorted(layer_dirs, key=natural_key)


def get_layer_id(layer_dir: str) -> int:
    name = os.path.basename(layer_dir)
    m = re.match(r"layer(\d+)", name)
    if not m:
        raise ValueError(f"非法 layer 目录名: {layer_dir}")
    return int(m.group(1))


def read_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img


def load_pose_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def pose_to_relative(current_pose: np.ndarray, origin_pose: np.ndarray) -> np.ndarray:
    return current_pose[:3] - origin_pose[:3]


def extract_xyz_from_pose(rel_pose: np.ndarray) -> Tuple[float, float, float]:
    if rel_pose.shape[0] < 3:
        raise ValueError("rob_poz.npy 长度小于 3，无法解析 xyz")
    return float(rel_pose[0]), float(rel_pose[1]), float(rel_pose[2])


def load_scanner_profile_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)[0]
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()

    if isinstance(data, dict):
        keys = {k.lower(): k for k in data.keys()}
        if "x" not in keys or "z" not in keys:
            raise ValueError(f"{path} 中未找到 x/z 字段")
        x = np.asarray(data[keys["x"]], dtype=np.float64).reshape(-1)
        z = np.asarray(data[keys["z"]], dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{path} 数据格式无法解析为 x/z")
        x = arr[:, 0].astype(np.float64)
        z = arr[:, 1].astype(np.float64)

    valid = np.isfinite(x) & np.isfinite(z)
    x = x[valid]
    z = z[valid]

    idx = np.argsort(x)
    x = x[idx]
    z = z[idx]
    x_unique, unique_idx = np.unique(x, return_index=True)
    z_unique = z[unique_idx]
    return x_unique, 500.0 - z_unique


def build_reference_interpolator(x_ref: np.ndarray, z_ref: np.ndarray):
    x_ref = np.asarray(x_ref, dtype=np.float64).reshape(-1)
    z_ref = np.asarray(z_ref, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x_ref) & np.isfinite(z_ref)
    x_ref = x_ref[valid]
    z_ref = z_ref[valid]
    if x_ref.size < 4:
        raise ValueError("商业扫描仪有效点过少，无法建立插值")

    idx = np.argsort(x_ref)
    x_ref = x_ref[idx]
    z_ref = z_ref[idx]
    x_unique, unique_idx = np.unique(x_ref, return_index=True)
    z_unique = z_ref[unique_idx]
    if x_unique.size < 4:
        raise ValueError("商业扫描仪唯一 x 点过少，无法插值")
    fz = interp1d(x_unique, z_unique, kind="linear", bounds_error=True)
    return x_unique, z_unique, fz


def sample_reference_profile(
    fz,
    x_ref_min: float,
    x_ref_max: float,
    center_x_global_mm: float,
    x_rel_grid_mm: np.ndarray,
) -> np.ndarray:
    x_global = center_x_global_mm + np.asarray(x_rel_grid_mm, dtype=np.float64)
    if float(np.nanmin(x_global)) < x_ref_min or float(np.nanmax(x_global)) > x_ref_max:
        raise ValueError(
            f"局部窗口超出商业扫描仪范围: [{np.nanmin(x_global):.3f}, {np.nanmax(x_global):.3f}] "
            f"not in [{x_ref_min:.3f}, {x_ref_max:.3f}]"
        )
    return np.asarray(fz(x_global), dtype=np.float64)


def find_origin_pose_path(raw_root: str) -> str:
    p = os.path.join(raw_root, "layer1", "camera", "1_1_rob_poz.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"未找到原点位姿文件: {p}")
    return p


def find_layer_scanner_file(layer_dir: str) -> str:
    layer_id = get_layer_id(layer_dir)
    pattern = os.path.join(layer_dir, "scanner", f"{layer_id}_1_pt_1_raw_laser_data.npy")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到商业扫描仪文件: {pattern}")
    return files[0]


def _clamp_roi_bounds(cfg: ScannerConfig, full_w: int, full_h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(int(cfg.roi_x0), full_w))
    x1 = max(0, min(int(cfg.roi_x1), full_w))
    y0 = max(0, min(int(cfg.roi_y0), full_h))
    y1 = max(0, min(int(cfg.roi_y1), full_h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"固定 ROI 非法：x=({x0},{x1}), y=({y0},{y1}), full=({full_w},{full_h})")
    return x0, x1, y0, y1


def build_meta_for_sample(img: np.ndarray, sample: CameraSample, cfg: ScannerConfig) -> tuple[np.ndarray, ImageCoordMeta]:
    """
    统一返回“用于提条纹的图像 + 其在原图坐标系中的偏移”。

    - layer1: 输入原始全图，按固定 ROI 截取后提条纹；
    - layer2+: 输入已经裁剪好的 ROI 图，只恢复偏移，不再做二次裁剪。
    """
    if sample.is_layer1_full_image:
        full_h, full_w = img.shape[:2]
        x0, x1, y0, y1 = _clamp_roi_bounds(cfg, full_w, full_h)
        img_proc = img[y0:y1, x0:x1]
        meta = ImageCoordMeta(
            full_width=int(full_w),
            full_height=int(full_h),
            offset_x=int(x0),
            offset_y=int(y0),
            local_width=int(img_proc.shape[1]),
            local_height=int(img_proc.shape[0]),
            source_path=sample.img_path,
        )
        return img_proc, meta

    img_proc = img
    meta = ImageCoordMeta(
        full_width=int(cfg.full_image_width),
        full_height=int(cfg.full_image_height),
        offset_x=int(cfg.roi_x0),
        offset_y=int(cfg.roi_y0),
        local_width=int(img_proc.shape[1]),
        local_height=int(img_proc.shape[0]),
        source_path=sample.img_path,
    )
    return img_proc, meta


def attach_full_coordinates(profile: Dict[str, np.ndarray], meta: ImageCoordMeta) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in profile.items()}
    out["u_full"] = np.asarray(out["u"], dtype=np.float64) + float(meta.offset_x)
    out["v_full"] = np.asarray(out["v"], dtype=np.float64) + float(meta.offset_y)
    out["full_width"] = np.asarray([meta.full_width], dtype=np.float64)
    out["full_height"] = np.asarray([meta.full_height], dtype=np.float64)
    out["offset_x"] = np.asarray([meta.offset_x], dtype=np.float64)
    out["offset_y"] = np.asarray([meta.offset_y], dtype=np.float64)
    out["local_width"] = np.asarray([meta.local_width], dtype=np.float64)
    out["local_height"] = np.asarray([meta.local_height], dtype=np.float64)
    return out


def index_camera_samples(layer_dir: str) -> List[CameraSample]:
    """
    当前真实数据组织：
    - layer1/camera: 原始全图
    - layer2+/camera: 已手动替换好的固定 ROI 图
    """
    layer_id = get_layer_id(layer_dir)
    cam_dir = os.path.join(layer_dir, "camera")
    img_files = glob.glob(os.path.join(cam_dir, f"{layer_id}_*.png"))
    samples: List[CameraSample] = []

    for img_path in sorted(img_files, key=natural_key):
        name = os.path.basename(img_path)
        m = re.match(rf"{layer_id}_(\d+)\.png$", name)
        if not m:
            continue
        idx = int(m.group(1))
        pose_path = os.path.join(cam_dir, f"{layer_id}_{idx}_rob_poz.npy")
        if not os.path.exists(pose_path):
            continue
        samples.append(
            CameraSample(
                layer_id=layer_id,
                sample_idx=idx,
                img_path=img_path,
                pose_path=pose_path,
                is_layer1_full_image=(layer_id == 1),
            )
        )
    return samples
