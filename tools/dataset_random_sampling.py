import random
import shutil
from pathlib import Path


def sample_layer2_keep_structure(
    src_layer2,
    dst_dataset_root,
    image_dir_name="camera",
    pose_dir_name="camera",
    sample_ratio=0.10,
    seed=42,
):
    src_layer2 = Path(src_layer2)
    dst_layer2 = Path(dst_dataset_root) / "layer2"

    src_img_dir = src_layer2 / image_dir_name
    src_pose_dir = src_layer2 / pose_dir_name

    dst_img_dir = dst_layer2 / image_dir_name
    dst_pose_dir = dst_layer2 / pose_dir_name

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_pose_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pose_exts = {".json", ".txt", ".csv", ".npy"}

    img_map = {
        f.stem: f for f in src_img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    }
    pose_map = {
        f.stem[:-8]: f for f in src_pose_dir.iterdir()
        if f.is_file() and f.suffix.lower() in pose_exts
    }

    common_keys = sorted(set(img_map) & set(pose_map))
    if not common_keys:
        raise RuntimeError("未找到同名 image/pose 文件对")

    sample_num = max(1, int(len(common_keys) * sample_ratio))
    random.seed(seed)
    sampled_keys = random.sample(common_keys, sample_num)

    for key in sampled_keys:
        shutil.copy2(img_map[key], dst_img_dir / img_map[key].name)
        shutil.copy2(pose_map[key], dst_pose_dir / pose_map[key].name)

    print(f"匹配总数: {len(common_keys)}")
    print(f"抽取数量: {sample_num}")
    print(f"保存到: {dst_layer2}")


if __name__ == "__main__":
    sample_layer2_keep_structure(
        src_layer2=r"raw_data/layer2",
        dst_dataset_root=r"test_data",
        image_dir_name="camera",
        pose_dir_name="camera",
        sample_ratio=0.10,
        seed=42,
    )