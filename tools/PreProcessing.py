import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Config:
    raw_root: str = "raw_data"
    output_dir: str = "preprocess_output"
    image_pattern: str = "layer1/camera/*.png"

    # 可视化输出
    save_mean: bool = True
    save_max: bool = True
    save_grid_preview: bool = True

    # 网格参数
    grid_step_px: int = 50

    # 批量裁剪参数（先人工看图，再改这里）
    enable_crop: bool = True
    # crop_x0: int = 350+545
    # crop_x1: int = 2550-545
    crop_x0: int = 545
    crop_x1: int = 2200-crop_x0
    crop_y0: int = 0
    crop_y1: int = 850

    cropped_dir_name: str = "camera_cropped"


CFG = Config()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_all_images(raw_root: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(raw_root, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths, key=natural_key)


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_heat_overlay(images: np.ndarray, threshold_percentile: float = 99.0) -> np.ndarray:
    """
    images: [N, H, W], uint8 or float
    返回每个像素被“高亮激活”的次数热图
    """
    N, H, W = images.shape
    counts = np.zeros((H, W), dtype=np.float32)

    for i in range(N):
        img = images[i].astype(np.float32)
        thr = np.percentile(img, threshold_percentile)
        mask = img >= thr
        counts += mask.astype(np.float32)

    counts = counts / max(N, 1)
    counts_u8 = np.clip(counts * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(counts_u8, cv2.COLORMAP_JET)
    return heat


def draw_grid(img_gray: np.ndarray, step: int = 50) -> np.ndarray:
    """
    给灰度图画网格和坐标标签，方便人工判断裁剪范围
    """
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    H, W = img_gray.shape

    # 网格线
    for x in range(0, W, step):
        cv2.line(vis, (x, 0), (x, H - 1), (0, 255, 0), 1)
    for y in range(0, H, step):
        cv2.line(vis, (0, y), (W - 1, y), (0, 255, 0), 1)

    # 坐标文字
    for x in range(0, W, step):
        cv2.putText(vis, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, H, step):
        cv2.putText(vis, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return vis


def save_preview_images(image_paths: List[str], output_dir: str, cfg: Config):
    if len(image_paths) == 0:
        raise RuntimeError("没有找到任何图像")

    print(f"共找到 {len(image_paths)} 张图像")

    ref_shape = None

    mean_img = read_gray(image_paths[0])
    max_img = read_gray(image_paths[0])
    for i,p in enumerate(image_paths):
        img = read_gray(p)

        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(f"图像尺寸不一致，无法叠加: {p}, shape={img.shape}, expected={ref_shape}")

        max_stack = np.stack([max_img,img], axis=0)
        mean_img = (mean_img*i+img)/(i+1)
        max_img = np.max(max_stack, axis=0)


    if cfg.save_mean:
        cv2.imwrite(os.path.join(output_dir, "mean_image.png"), normalize_to_uint8(mean_img))
        print("[OK] 已保存 mean_image.png")

    if cfg.save_max:
        cv2.imwrite(os.path.join(output_dir, "max_image.png"), normalize_to_uint8(max_img))
        print("[OK] 已保存 max_image.png")

    if cfg.save_grid_preview:
        grid_mean = draw_grid(normalize_to_uint8(mean_img), cfg.grid_step_px)
        grid_max = draw_grid(normalize_to_uint8(max_img), cfg.grid_step_px)

        cv2.imwrite(os.path.join(output_dir, "mean_image_with_grid.png"), grid_mean)
        cv2.imwrite(os.path.join(output_dir, "max_image_with_grid.png"), grid_max)
        print("[OK] 已保存 mean_image_with_grid.png")
        print("[OK] 已保存 max_image_with_grid.png")


def crop_image(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"非法裁剪区域: x=({x0},{x1}), y=({y0},{y1})")

    return img[y0:y1, x0:x1]


def batch_crop_images(image_paths: List[str], raw_root: str, cfg: Config):
    """
    将每个 layer*/camera/*.png 裁剪后保存到对应 layer*/camera_cropped/*.png
    """
    for p in image_paths:
        img = read_gray(p)
        cropped = crop_image(img, cfg.crop_x0, cfg.crop_x1, cfg.crop_y0, cfg.crop_y1)

        # 构造输出路径
        # 例如 raw_data/layer2/camera/2_1.png
        # ->  raw_data/layer2/camera_cropped/2_1.png
        rel = os.path.relpath(p, raw_root)
        parts = rel.split(os.sep)

        if len(parts) < 3 or parts[1] != "camera":
            print(f"[Warn] 跳过非标准路径: {p}")
            continue

        parts[1] = cfg.cropped_dir_name
        out_path = os.path.join(raw_root, *parts)

        ensure_dir(os.path.dirname(out_path))
        cv2.imwrite(out_path, cropped)

    print(f"[OK] 批量裁剪完成，输出目录名: {cfg.cropped_dir_name}")


def main():
    ensure_dir(CFG.output_dir)

    image_paths = list_all_images(CFG.raw_root, CFG.image_pattern)
    if len(image_paths) == 0:
        raise RuntimeError(f"未找到图像: {os.path.join(CFG.raw_root, CFG.image_pattern)}")

    if not CFG.enable_crop:
        print("开始生成叠加预览图...")
        save_preview_images(image_paths, CFG.output_dir, CFG)

    if CFG.enable_crop:
        print("开始批量裁剪...")
        batch_crop_images(image_paths, CFG.raw_root, CFG)
    else:
        print("当前仅生成预览图，未执行裁剪。")
        print("请查看以下文件后手工确定裁剪范围：")
        print(f"  {os.path.join(CFG.output_dir, 'max_image_with_grid.png')}")
        print(f"  {os.path.join(CFG.output_dir, 'mean_image_with_grid.png')}")
        print("确定 crop_x0, crop_x1, crop_y0, crop_y1 后，将 enable_crop 改为 True 再运行。")


if __name__ == "__main__":
    main()