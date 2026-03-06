#!/usr/bin/env python3
"""
生成回归任务的两联图：符号回归 + 线性校准（论文版本）

基于现有的 regression_experiment_results.json 和 physo_predictions.csv 生成图表
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

# 设置绘图参数（论文质量）
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
})


def main():
    """主函数"""
    print("=" * 70)
    print("回归校准图表生成（论文版本）")
    print("=" * 70)

    # 1. 读取回归结果
    print("\n[步骤1] 读取回归实验结果...")
    results_path = os.path.join(PROJECT_ROOT, 'results', 'regression_experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    coef = results['calibration_model']['coef']
    intercept = results['calibration_model']['intercept']
    formula = results['best_expression']

    print(f"符号回归公式: {formula}")
    print(f"校准公式: y_cal = {coef:.2f} * y_sr + {intercept:.2f}")
    print(f"R2: {results['uncalibrated_test_metrics']['R-squared']:.3f} -> {results['calibrated_test_metrics']['R-squared']:.3f}")

    # 2. 读取预测数据
    print("\n[步骤2] 读取预测数据...")
    pred_path = os.path.join(PROJECT_ROOT, 'results', 'investment_decision', 'physo_predictions.csv')
    df_pred = pd.read_csv(pred_path)

    # 反推出未校准的 y_sr
    # y_cal = coef * y_sr + intercept  =>  y_sr = (y_cal - intercept) / coef
    df_pred['y_sr'] = (df_pred['y_pred'] - intercept) / coef

    # 3. 获取训练/测试分割信息
    print("\n[步骤3] 获取数据分割信息...")
    dataset_name = 'investment_decision'
    target_column = '预期利润额（万元）'
    task_type = 'regression'

    X, y, _, _ = load_and_preprocess_data(
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type
    )
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)

    n_train = len(y_train)
    n_test = len(y_test)

    # 标记训练集和测试集
    df_pred['split'] = ['train'] * n_train + ['test'] * n_test

    print(f"训练集: {n_train} 样本")
    print(f"测试集: {n_test} 样本")
    print(f"y_reg 范围: [{y.min():.2f}, {y.max():.2f}] 万元")

    # 4. 保存完整数据
    print("\n[步骤4] 保存完整数据...")
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 重命名列以符合论文要求
    df_export = df_pred.rename(columns={
        'y_true': 'y_reg',
        'y_pred': 'y_cal',
        'y_sr': 'y_sr'
    })[['split', 'y_reg', 'y_sr', 'y_cal']]

    csv_path = os.path.join(figures_dir, 'regression_calibration_points.csv')
    df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存: {csv_path}")

    # 保存校准参数
    params_path = os.path.join(figures_dir, 'regression_calibration_params.txt')
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("回归校准参数记录\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"符号回归公式: {formula}\n\n")
        f.write(f"校准公式: y_cal = {coef:.6f} * y_sr + {intercept:.6f}\n\n")
        f.write(f"校准系数 (a): {coef:.6f}\n")
        f.write(f"截距 (b): {intercept:.6f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("校准前指标 (Test Set)\n")
        f.write("=" * 60 + "\n")
        for name, value in results['uncalibrated_test_metrics'].items():
            f.write(f"  {name}: {value:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("校准后指标 (Test Set)\n")
        f.write("=" * 60 + "\n")
        for name, value in results['calibrated_test_metrics'].items():
            f.write(f"  {name}: {value:.6f}\n")
    print(f"参数已保存: {params_path}")

    # 5. 生成图表
    print("\n[步骤5] 生成图表...")

    # 提取训练集和测试集数据
    train_mask = df_pred['split'] == 'train'
    test_mask = df_pred['split'] == 'test'

    y_true_train = df_pred.loc[train_mask, 'y_true'].values
    y_true_test = df_pred.loc[test_mask, 'y_true'].values
    y_sr_train = df_pred.loc[train_mask, 'y_sr'].values
    y_sr_test = df_pred.loc[test_mask, 'y_sr'].values
    y_cal_train = df_pred.loc[train_mask, 'y_pred'].values
    y_cal_test = df_pred.loc[test_mask, 'y_pred'].values

    # 【修正1】标准化 raw output: z_SR = (ŷ_SR - mean_train) / std_train
    sr_mean_train = np.mean(y_sr_train)
    sr_std_train = np.std(y_sr_train)
    z_sr_train = (y_sr_train - sr_mean_train) / sr_std_train
    z_sr_test = (y_sr_test - sr_mean_train) / sr_std_train

    # 创建图表（1×2 横排，适合双栏论文）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ===== 子图: Standardized raw SR output vs Actual (before calibration) =====
    # 【修正5】train点更透明，test点更醒目
    ax1.scatter(y_true_train, z_sr_train, c='blue', s=18, alpha=0.25,
                marker='s', label='Train (n={})'.format(len(y_true_train)))
    ax1.scatter(y_true_test, z_sr_test, c='red', s=35, alpha=0.7,
                edgecolors='darkred', linewidth=0.6, label='Test (n={})'.format(len(y_true_test)))

    # 【修正1】【修正4】在右下角注明 z_SR 公式（标准数学符号）
    ax1.text(0.98, 0.02, r'$z_{SR} = (\hat{y}_{SR} - \mu_{train}) / \sigma_{train}$',
             transform=ax1.transAxes, fontsize=7, ha='right', va='bottom',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    ax1.set_xlabel('Actual Profit $y_{reg}$ (10K CNY)', fontsize=9)
    ax1.set_ylabel('Standardized raw SR output $z_{SR}$', fontsize=9)
    ax1.set_title('(a) Before calibration: standardized raw SR output vs actual',
                 fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ===== 子图: Calibrated prediction vs Actual (after calibration) + log-log inset =====
    # 【修正5】train点更透明，test点更醒目
    ax2.scatter(y_true_train, y_cal_train, c='blue', s=18, alpha=0.25,
                marker='s', label='Train (n={})'.format(len(y_true_train)))
    ax2.scatter(y_true_test, y_cal_test, c='red', s=35, alpha=0.7,
                edgecolors='darkred', linewidth=0.6, label='Test (n={})'.format(len(y_true_test)))

    # y=x 参考线（Perfect fit）
    min_val_b = min(min(y_true_train), min(y_cal_train))
    max_val_b = max(max(y_true_train), max(y_cal_train))
    ax2.plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'gray',
             linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Fit (y=x)')

    # 【修正2】标注校准后指标（标明 Test Set + Calib. fit on train）
    r2_cal = results['calibrated_test_metrics']['R-squared']
    rmse_cal = results['calibrated_test_metrics']['RMSE']
    mae_cal = results['calibrated_test_metrics']['MAE']

    metrics_text = f'Test Set:\nR² = {r2_cal:.3f}\nRMSE = {rmse_cal:.0f}\nMAE = {mae_cal:.0f}\nCalib. fit on train'
    ax2.text(0.97, 0.97, metrics_text, transform=ax2.transAxes,
             fontsize=7, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.6, edgecolor='none'))

    ax2.set_xlabel('Actual Profit $y_{reg}$ (10K CNY)', fontsize=9)
    ax2.set_ylabel('Calibrated Prediction $\\hat{y}_{cal}$', fontsize=9)
    ax2.set_title('(b) After calibration: calibrated prediction vs actual',
                 fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 【修正3】添加 ln-ln parity inset（自然对数，标签简化）
    # 只显示正值点
    positive_mask_train = (y_true_train > 0) & (y_cal_train > 0)
    positive_mask_test = (y_true_test > 0) & (y_cal_test > 0)

    if np.sum(positive_mask_train) > 0 or np.sum(positive_mask_test) > 0:
        # 创建 inset axes
        from matplotlib.patches import Rectangle
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axins = inset_axes(ax2, width="35%", height="35%", loc='lower right',
                          bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax2.transAxes)

        # 【修正5】inset 中 train点更透明，test点更醒目
        # 【修正3】使用自然对数 ln
        axins.scatter(np.log(y_true_train[positive_mask_train]),
                     np.log(y_cal_train[positive_mask_train]),
                     c='blue', s=8, alpha=0.25, marker='s')
        axins.scatter(np.log(y_true_test[positive_mask_test]),
                     np.log(y_cal_test[positive_mask_test]),
                     c='red', s=12, alpha=0.6, edgecolors='darkred', linewidth=0.3)

        # ln-ln y=x 参考线
        ln_min = min(np.log(y_true_train[positive_mask_train]).min(),
                     np.log(y_cal_train[positive_mask_train]).min())
        ln_max = max(np.log(y_true_train[positive_mask_train]).max(),
                     np.log(y_cal_train[positive_mask_train]).max())
        axins.plot([ln_min, ln_max], [ln_min, ln_max], 'gray',
                  linestyle='--', linewidth=1, alpha=0.7)

        # 【修正3】inset 标签：ln(y) 和 ln(ŷ_cal)，字体更小
        axins.set_xlabel(r'$\ln(y)$', fontsize=5.5)
        axins.set_ylabel(r'$\ln(\hat{y}_{cal})$', fontsize=5.5)
        axins.tick_params(labelsize=4.5)
        axins.grid(True, alpha=0.3, linestyle='--')
        # inset 边框
        for spine in axins.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)

    # 不使用 tight_layout，因为它与 inset_axes 不兼容

    # 保存图片（不使用 bbox_inches='tight'，因为它与 inset_axes 不兼容）
    save_path = os.path.join(figures_dir, 'regression_calibration_ab.png')
    plt.savefig(save_path, dpi=300, facecolor='white', pad_inches=0.1)
    print(f"图表已保存: {save_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("完成！生成的文件：")
    print("  1. figures/regression_calibration_ab.png")
    print("  2. figures/regression_calibration_points.csv")
    print("  3. figures/regression_calibration_params.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
