#!/usr/bin/env python3
"""
纯回归任务实验脚本

此脚本现在包含一个可复用的函数 `run_calibrated_regression_experiment`，
该函数封装了完整的、包含线性校准的回归实验流程。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.symbolic_regression import SymbolicRegressionModel
from src.evaluation import get_regression_metrics
from sklearn.linear_model import LinearRegression

def run_calibrated_regression_experiment(epochs=15, test_size=0.2, random_state=0):
    """执行包含线性校准的完整回归实验流程"""
    print("=" * 60)
    print("开始纯回归任务实验 (包含线性校准)...")
    print("=" * 60)

    # --- 1. 数据加载和准备 ---
    print("\n[步骤1] 加载和准备数据...")
    dataset_name = 'investment_decision'
    target_column = '预期利润额（万元）'
    task_type = 'regression'

    X, y, _, _ = load_and_preprocess_data(
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type
    )
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    print(f"数据准备完成。目标变量: {target_column}")

    # --- 2. 模型训练 ---
    print("\n[步骤2] 符号回归模型训练 (回归任务)...")
    X_train_transposed = X_train.T
    X_test_transposed = X_test.T

    model = SymbolicRegressionModel(seed=random_state)
    model.fit(
        X_train_transposed, y_train,
        X_names=[f'X{i}' for i in range(X_train_transposed.shape[0])],
        y_name="Profit",
        epochs=epochs
    )

    best_expr_info = model.get_best_expression()
    print(f"\n[结果] 发现的最佳利润预测公式: {best_expr_info['clean_expression']}")

    # --- 3. 模型评估 (包含线性校准) ---
    print("\n[步骤3] 模型评估 (包含线性校准)...")
    y_train_sr_pred = model.predict(X_train_transposed)
    y_test_sr_pred = model.predict(X_test_transposed)

    print("\n--- 评估 (校准前) ---")
    uncalibrated_metrics = get_regression_metrics(y_test, y_test_sr_pred)
    print("Test Set (Uncalibrated):")
    for name, value in uncalibrated_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n--- 训练线性校准器 ---")
    calibrator = LinearRegression()
    calibrator.fit(y_train_sr_pred.reshape(-1, 1), y_train)
    coef = calibrator.coef_[0]
    intercept = calibrator.intercept_
    print(f"校准模型完成。公式: y_calibrated = {coef:.4f} * y_sr + {intercept:.4f}")

    print("\n--- 评估 (校准后) ---")
    y_test_calibrated_pred = calibrator.predict(y_test_sr_pred.reshape(-1, 1))
    calibrated_metrics = get_regression_metrics(y_test, y_test_calibrated_pred)
    print("Test Set (Calibrated):")
    for name, value in calibrated_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("=" * 40)

    # --- 4. 绘制诊断图 ---
    print("--- 正在生成诊断图 ---")
    results_dir = os.path.join(PROJECT_ROOT, 'results') # 现在直接指向results根目录
    plot_save_path = os.path.join(results_dir, 'regression_actual_vs_pred.png')
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_test_sr_pred, y_test, alpha=0.7, label='Data Points (Uncalibrated Pred vs Actual)')
    min_val = min(np.min(y_test_sr_pred), np.min(y_test)) * 0.9
    max_val = max(np.max(y_test_sr_pred), np.max(y_test)) * 1.1
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y=x)')
    x_fit = np.linspace(min_val, max_val, 100).reshape(-1, 1)
    y_fit = calibrator.predict(x_fit)
    ax.plot(x_fit, y_fit, 'g-', linewidth=2, label=f'Calibrated Fit (y={coef:.2f}x + {intercept:.2f})')
    ax.set_xlabel("Predicted Values (Uncalibrated SR Output)", fontsize=12)
    ax.set_ylabel("Actual Values", fontsize=12)
    ax.set_title("Actual vs. Predicted Values for Regression Diagnosis", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(plot_save_path, dpi=300)
    print(f"诊断图已保存到: {plot_save_path}")

    # --- 5. 保存结果和预测数据 ---
    print("\n[步骤4] 保存回归实验结果...")
    results = {
        'best_expression': str(best_expr_info['clean_expression']),
        'sympy_expression': str(best_expr_info['sympy_expression']),
        'uncalibrated_test_metrics': uncalibrated_metrics,
        'calibrated_test_metrics': calibrated_metrics,
        'calibration_model': {'coef': coef, 'intercept': intercept}
    }
    save_path = os.path.join(results_dir, 'regression_experiment_results.json')
    def default_serializer(o):
        if isinstance(o, (np.number, np.bool_)):
            return o.item()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=default_serializer)
    print(f"结果已保存到: {save_path}")

    # 保存预测数据（用于绘图脚本）
    # 合并训练集和测试集的预测结果（使用校准后的预测）
    all_y_true = np.concatenate([y_train, y_test])
    all_y_pred_sr = np.concatenate([y_train_sr_pred, y_test_sr_pred])

    # 对训练集预测也进行校准
    y_train_calibrated_pred = calibrator.predict(y_train_sr_pred.reshape(-1, 1))
    all_y_pred_calibrated = np.concatenate([y_train_calibrated_pred, y_test_calibrated_pred])

    # 计算残差和误差（使用校准后的预测）
    residuals = all_y_true - all_y_pred_calibrated
    percentage_error = (residuals / all_y_true) * 100

    pred_df = pd.DataFrame({
        'y_true': all_y_true,
        'y_pred': all_y_pred_calibrated,  # 使用校准后的预测
        'residual': residuals,
        'abs_error': np.abs(residuals),
        'percentage_error': percentage_error
    })

    # 保存到 results/investment_decision/
    investment_dir = os.path.join(PROJECT_ROOT, 'results', 'investment_decision')
    os.makedirs(investment_dir, exist_ok=True)
    csv_save_path = os.path.join(investment_dir, 'physo_predictions.csv')
    pred_df.to_csv(csv_save_path, index=False)
    print(f"预测数据已保存到: {csv_save_path}")

    print("\n" + "=" * 60)
    print("纯回归任务实验完成！")
    print("=" * 60)

    return results

def main():
    """主函数，用于直接运行此脚本"""
    run_calibrated_regression_experiment(epochs=15)

if __name__ == "__main__":
    main()