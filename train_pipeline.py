# -*- coding: utf-8 -*-
"""
更新日志
- v4
    主脚本输出 layer1 横向几何校正与 layer2 映射训练的联合指标。
- v3
- v2
- v1
  1) 建立文件头更新日志，后续每次修改请在此追加，便于追踪该文件的演化。
"""
from __future__ import annotations

import argparse
import os

from compare_results import build_summary, compare_prediction_folder
from data_reader import ensure_dir, find_origin_pose_path, get_layer_id, list_layer_dirs, load_pose_npy
from ml_model import ProfileMappingModel
from plane_reference import PlaneReferenceModel
from scanner_config import ScannerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="工业风格分层扫描仪训练脚本")
    parser.add_argument("--raw-root", default="test_data")
    parser.add_argument("--output-dir", default="output_test")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    cfg = ScannerConfig(raw_root=args.raw_root, output_dir=args.output_dir)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.model_dir)
    ensure_dir(cfg.prediction_dir)
    ensure_dir(cfg.comparison_dir)
    ensure_dir(cfg.feature_cache_dir)

    layer_dirs = list_layer_dirs(cfg.raw_root)
    if not layer_dirs:
        raise RuntimeError(f"未找到任何 layer 目录: {cfg.raw_root}")

    print("检测到 layer 目录:")
    for d in layer_dirs:
        print("  ", d)

    layer1_dir = None
    object_layer_dirs = []
    for d in layer_dirs:
        if get_layer_id(d) == 1:
            layer1_dir = d
        else:
            object_layer_dirs.append(d)

    if layer1_dir is None:
        raise RuntimeError("缺少 layer1，无法建立参考平面")
    if not object_layer_dirs:
        raise RuntimeError("没有 layer2+ 数据，无法训练映射模型")

    origin_pose = load_pose_npy(find_origin_pose_path(cfg.raw_root))

    print("\n=== 1) 建立参考平面与 layer1 横向几何校正 ===")
    plane_model = PlaneReferenceModel(cfg)
    plane_model.fit(layer1_dir, origin_pose)
    plane_model.save(os.path.join(cfg.model_dir, "plane_model.pkl"))
    print(f"[Save] 参考平面模型已保存到 {cfg.model_dir}")

    print("\n=== 2) 建立 layer2 纯图像 -> 轮廓映射模型 ===")
    mapping_model = ProfileMappingModel(cfg=cfg, plane_model=plane_model, origin_pose=origin_pose)
    mapping_metrics = mapping_model.fit(object_layer_dirs)
    mapping_model.save()

    print("\n=== 3) 导出 layer2+ 预测曲线 ===")
    mapping_model.export_predictions(object_layer_dirs)

    if args.skip_compare:
        print("\n=== 4) 已跳过结果对比 ===")
        return

    print("\n=== 4) 与商业扫描仪结果对比 ===")
    detail_df, failed = compare_prediction_folder(cfg)
    summary_df = build_summary(detail_df)
    detail_df.to_csv(os.path.join(cfg.comparison_dir, "detail_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(cfg.comparison_dir, "summary.csv"), index=False)

    if failed:
        fail_path = os.path.join(cfg.comparison_dir, "failed_cases.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"[Compare] 失败样本 {len(failed)} 个，明细见 {fail_path}")

    print("\n=== 5) 核心结果 ===")
    print(f"验证集 MAE_dz = {mapping_metrics['mae_dz_mm']:.4f} mm")
    print(f"验证集 MAE_z  = {mapping_metrics['mae_z_mm']:.4f} mm")
    print(f"layer1 x_shift 标签平均绝对值 = {plane_model.train_metrics.get('xshift_label_abs_mean_mm', float('nan')):.4f} mm")
    print(f"layer1 x_shift 拟合 MAE      = {plane_model.train_metrics.get('xshift_fit_mae_mm', float('nan')):.4f} mm")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
