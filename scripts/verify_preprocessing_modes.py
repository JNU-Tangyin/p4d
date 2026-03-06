#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据预处理模式切换脚本
用于确认旧版本和新版本逻辑正常工作
"""

import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# 固定随机种子
np.random.seed(0)

def test_old_mode():
    """测试旧版本模式（用于复现 R²=0.784）"""
    print("=" * 60)
    print("测试旧版本模式 (USE_OLD_PREPROCESSING=true)")
    print("=" * 60)

    # 设置环境变量
    os.environ['USE_OLD_PREPROCESSING'] = 'true'

    # 重新导入模块以应用环境变量
    import importlib
    import src.data_preprocessing as dp
    importlib.reload(dp)

    from src.data_preprocessing import load_and_preprocess_data

    # 测试回归任务
    X, y, df, config = load_and_preprocess_data(
        dataset_name='investment_decision',
        target_column='预期利润额（万元）',
        task_type='regression'
    )

    print(f"\n[旧版本] 回归任务:")
    print(f"  特征数: {X.shape[1]}")
    print(f"  样本数: {X.shape[0]}")
    print(f"  目标变量范围: [{y.min():.2f}, {y.max():.2f}]")

    # 验证特征数应该是 26
    assert X.shape[1] == 26, f"期望特征数=26，实际={X.shape[1]}"
    print(f"  [OK] 特征数验证通过 (26)")

    return X.shape[1]

def test_new_mode():
    """测试新版本模式（用于分类任务）"""
    print("\n" + "=" * 60)
    print("测试新版本模式 (默认)")
    print("=" * 60)

    # 清除环境变量
    os.environ['USE_OLD_PREPROCESSING'] = 'false'

    # 重新导入模块以应用环境变量
    import importlib
    import src.data_preprocessing as dp
    importlib.reload(dp)

    from src.data_preprocessing import load_and_preprocess_data

    # 测试分类任务
    X, y, df, config = load_and_preprocess_data(
        dataset_name='investment_decision',
        target_column='是否应投资该项目',
        task_type='classification'
    )

    print(f"\n[新版本] 分类任务:")
    print(f"  特征数: {X.shape[1]}")
    print(f"  样本数: {X.shape[0]}")

    unique_values, counts = np.unique(y, return_counts=True)
    print(f"  类别分布: 类别{int(unique_values[0])}有{counts[0]}个样本, 类别{int(unique_values[1])}有{counts[1]}个样本")

    # 验证
    assert X.shape[1] == 30, f"期望特征数=30，实际={X.shape[1]}"
    print(f"  [OK] 特征数验证通过 (30)")

    assert counts[0] == 13 and counts[1] == 58, f"期望类别分布=[13, 58]，实际={list(counts)}"
    print(f"  [OK] 类别分布验证通过 ([13, 58])")

    return X.shape[1]

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据预处理模式验证")
    print("=" * 60)

    try:
        old_features = test_old_mode()
        new_features = test_new_mode()

        print("\n" + "=" * 60)
        print("验证结果汇总")
        print("=" * 60)
        print(f"旧版本模式（回归）: {old_features} 个特征")
        print(f"新版本模式（分类）: {new_features} 个特征")
        print(f"\n[OK] 所有验证通过！")
        print(f"\n使用方法:")
        print(f"  - 回归任务（复现 R2=0.784）:")
        print(f"    Windows: set USE_OLD_PREPROCESSING=true")
        print(f"    Linux/Mac: USE_OLD_PREPROCESSING=true python main.py ...")
        print(f"  - 分类任务（保持 100% 准确率）:")
        print(f"    直接运行: python main.py --task_type classification ...")

    except AssertionError as e:
        print(f"\n[FAIL] 验证失败: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
