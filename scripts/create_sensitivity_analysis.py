#!/usr/bin/env python3
"""
模型敏感性分析图生成脚本

分别生成：
1. 分类任务敏感性分析（X4、X0、X2）
2. 回归任务敏感性分析（X6、X8）
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

from src.data_preprocessing import load_and_preprocess_data

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
    'text.usetex': False,  # 关闭LaTeX渲染，避免依赖问题
})


def load_classification_data():
    """加载分类任务数据"""
    print("\n[分类任务] 加载数据...")

    data_path = os.path.join(PROJECT_ROOT, 'data', '项目数据收集表v2.0.xlsx')
    df = pd.read_excel(data_path, header=2)

    # 编码目标变量
    if '是否应投资该项目' in df.columns:
        df['是否应投资该项目'] = df['是否应投资该项目'].map({'是': 1, '否': 0})

    # 删除Unnamed列
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_cols)

    # 只保留数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # 分类公式变量
    X0_col = '去化期间物业管理费（万元）'
    X2_col = '去化期间利息支出或沉没（万元）'
    X4_col = '当期商铺对外每月租金（万元）'

    X0 = df[X0_col].values
    X2 = df[X2_col].values
    X4 = df[X4_col].values

    print(f"X0 ({X0_col}) 原始: 范围 [{X0.min():.2f}, {X0.max():.2f}], 均值 {X0.mean():.2f}")
    print(f"X2 ({X2_col}) 原始: 范围 [{X2.min():.2f}, {X2.max():.2f}], 均值 {X2.mean():.2f}")
    print(f"X4 ({X4_col}) 原始: 范围 [{X4.min():.2f}, {X4.max():.2f}], 均值 {X4.mean():.2f}")

    # 使用DecimalScaler标准化（与模型一致）
    from src.data_preprocessing import DecimalScaler

    X0_scaled = DecimalScaler(X0)
    X2_scaled = DecimalScaler(X2)
    X4_scaled = DecimalScaler(X4)

    print(f"X0 标准化后: 范围 [{X0_scaled.min():.4f}, {X0_scaled.max():.4f}]")
    print(f"X2 标准化后: 范围 [{X2_scaled.min():.4f}, {X2_scaled.max():.4f}]")
    print(f"X4 标准化后: 范围 [{X4_scaled.min():.4f}, {X4_scaled.max():.4f}]")

    return {
        'X0': {'name': 'Property Fee', 'short': 'X0', 'data': X0_scaled, 'data_orig': X0, 'full_name': X0_col},
        'X2': {'name': 'Interest Expense', 'short': 'X2', 'data': X2_scaled, 'data_orig': X2, 'full_name': X2_col},
        'X4': {'name': 'Commercial Rent', 'short': 'X4', 'data': X4_scaled, 'data_orig': X4, 'full_name': X4_col}
    }


def load_regression_data():
    """加载回归任务数据"""
    print("\n[回归任务] 加载数据...")

    dataset_name = 'investment_decision'
    target_column = '预期利润额（万元）'
    task_type = 'regression'

    X, y, df, _ = load_and_preprocess_data(
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type
    )

    # 找到 X6 和 X8 对应的列（需要根据特征选择后的索引）
    # X6: 抵房总面积（㎡）- 原始列名需要确认
    # X8: 当期行业基准收益率（%）- 原始列名需要确认

    # 根据数据文件读取原始列
    data_path = os.path.join(PROJECT_ROOT, 'data', '项目数据收集表v2.0.xlsx')
    df_raw = pd.read_excel(data_path, header=2)

    # 查找包含关键词的列
    X6_col = None
    X8_col = None

    for col in df_raw.columns:
        if '抵房总面积' in str(col) or '面积' in str(col) and '抵房' in str(col):
            X6_col = col
        if '行业基准收益率' in str(col) or '基准收益率' in str(col):
            X8_col = col

    print(f"X6 列名: {X6_col}")
    print(f"X8 列名: {X8_col}")

    if X6_col and X6_col in df_raw.columns:
        X6 = df_raw[X6_col].values
    else:
        # 从特征矩阵中获取（假设X6是第7个特征，索引6）
        X6 = X[6] if X.shape[0] > 6 else np.random.uniform(0, 6000, len(y))
        X6_col = 'X6 (索引6)'

    if X8_col and X8_col in df_raw.columns:
        X8 = df_raw[X8_col].values
    else:
        # 从特征矩阵中获取（假设X8是第9个特征，索引8）
        X8 = X[8] if X.shape[0] > 8 else np.random.uniform(0.05, 0.25, len(y))
        X8_col = 'X8 (索引8)'

    # 处理缺失值
    X6 = np.nan_to_num(X6, nan=0.0, posinf=6000, neginf=0)
    X8 = np.nan_to_num(X8, nan=0.1, posinf=0.25, neginf=0.01)

    print(f"X6 ({X6_col}): 范围 [{X6.min():.2f}, {X6.max():.2f}], 均值 {X6.mean():.2f}")
    print(f"X8 ({X8_col}): 范围 [{X8.min():.4f}, {X8.max():.4f}], 均值 {X8.mean():.4f}")

    # 读取回归结果
    results_path = os.path.join(PROJECT_ROOT, 'results', 'regression_experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        regression_results = json.load(f)

    coef = regression_results['calibration_model']['coef']
    intercept = regression_results['calibration_model']['intercept']

    print(f"校准公式: y_cal = {coef:.2f} * y_sr + {intercept:.2f}")

    return {
        'X6': {'name': 'Total Mortgage Area', 'short': 'X6', 'data': X6, 'full_name': X6_col},
        'X8': {'name': 'Benchmark Yield', 'short': 'X8', 'data': X8, 'full_name': X8_col},
        'coef': coef,
        'intercept': intercept
    }


def classification_formula(X4, X0, X2):
    """分类任务决策得分公式"""
    return (X4**2) * np.cos(X0)**2 / (X2**2 + 1e-10)


def regression_sr_output(X6, X8):
    """回归任务符号回归原始输出（未校准）"""
    return np.abs(np.log(X6 + 1)**2 / (X8**0.5 + 1e-10))


def regression_calibrated(X6, X8, coef, intercept):
    """回归任务校准后输出"""
    y_sr = regression_sr_output(X6, X8)
    return coef * y_sr + intercept


def create_classification_sensitivity_plot(vars_data, save_dir):
    """创建分类任务敏感性分析图"""
    print("\n[分类任务] 生成敏感性分析图...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 计算基准值（使用平均值而非中位数，避免接近0的问题）
    X0_baseline = np.mean(vars_data['X0']['data'])
    X2_baseline = np.mean(vars_data['X2']['data'])
    X4_baseline = np.mean(vars_data['X4']['data'])

    print(f"基准值 - X0: {X0_baseline:.6f}, X2: {X2_baseline:.6f}, X4: {X4_baseline:.6f}")

    # 确保基准值不会导致除零或数值问题
    if X4_baseline < 0.001:
        X4_baseline = 0.05  # 使用合理的非零值
    if X2_baseline < 0.001:
        X2_baseline = 0.05

    print(f"调整后基准值 - X0: {X0_baseline:.6f}, X2: {X2_baseline:.6f}, X4: {X4_baseline:.6f}")

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    var_keys = ['X0', 'X2', 'X4']

    sensitivity_data = {}

    for idx, (ax, var_key) in enumerate(zip(axes, var_keys)):
        var_info = vars_data[var_key]
        var_data_scaled = var_info['data']  # 标准化后的数据
        var_data_orig = var_info['data_orig']  # 原始数据

        # 创建变化范围（使用标准化数据的范围）
        x_range_scaled = np.linspace(var_data_scaled.min(), var_data_scaled.max(), 50)
        # 对应的原始值范围（用于x轴标注）
        x_range_orig = np.linspace(var_data_orig.min(), var_data_orig.max(), 50)

        # 计算得分变化
        scores = []
        for val in x_range_scaled:
            if var_key == 'X0':
                score = classification_formula(X4_baseline, val, X2_baseline)
            elif var_key == 'X2':
                score = classification_formula(X4_baseline, X0_baseline, val)
            else:  # X4
                score = classification_formula(val, X0_baseline, X2_baseline)
            scores.append(score)

        scores = np.array(scores)

        # 标准化得分（相对于最大值）
        scores_max = scores.max()
        if scores_max > 1e-10:
            scores_norm = scores / scores_max
        else:
            scores_norm = scores / (np.abs(scores).max() + 1e-10)

        # 绘制曲线（x轴用原始值，y轴用标准化得分）
        ax.plot(x_range_orig, scores_norm, color=colors[idx], linewidth=2.5,
                label=f"{var_info['short']}")

        # 标注基准值（用原始值）
        ax.axvline(x=var_info['data_orig'].mean(), color='gray', linestyle='--',
                   alpha=0.5, linewidth=1)
        ax.text(var_info['data_orig'].mean(), 0.95, 'mean', rotation=90,
               ha='right', va='top', fontsize=6, color='gray')

        # 填充区域
        ax.fill_between(x_range_orig, 0, scores_norm, alpha=0.15, color=colors[idx])

        # 设置坐标轴
        ax.set_xlabel(f'{var_info["short"]} (10K CNY)', fontsize=9)
        ax.set_ylabel('Normalized Decision Score', fontsize=9)
        ax.set_title(f'({chr(97+idx)}) {var_info["short"]}: {var_info["name"]}',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        # 打印调试信息
        print(f"{var_key}: score范围 [{scores.min():.6f}, {scores.max():.6f}]")

        # 保存数据
        sensitivity_data[var_key] = {
            'x_range_orig': x_range_orig,
            'x_range_scaled': x_range_scaled,
            'scores': scores,
            'scores_norm': scores_norm
        }

    plt.tight_layout()

    # 保存图片（使用时间戳避免缓存）
    import time
    timestamp = int(time.time())
    save_path = os.path.join(save_dir, f'sensitivity_classification_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  分类任务敏感性分析图已保存: {save_path}")
    plt.close()

    return sensitivity_data


def create_regression_sensitivity_plot(vars_data, save_dir):
    """创建回归任务敏感性分析图"""
    print("\n[回归任务] 生成敏感性分析图...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # 计算基准值
    X6_baseline = np.median(vars_data['X6']['data'])
    X8_baseline = np.median(vars_data['X8']['data'])
    coef = vars_data['coef']
    intercept = vars_data['intercept']

    colors = ['#9b59b6', '#f39c12']
    var_keys = ['X6', 'X8']

    sensitivity_data = {}

    for idx, (ax, var_key) in enumerate(zip(axes, var_keys)):
        var_info = vars_data[var_key]
        var_data = var_info['data']

        # 创建变化范围
        if var_key == 'X6':
            x_range = np.linspace(0, var_data.max() * 1.1, 50)
        else:  # X8
            x_range = np.linspace(0.01, var_data.max() * 1.5, 50)

        # 计算预测值变化
        predictions = []
        for val in x_range:
            if var_key == 'X6':
                pred = regression_calibrated(val, X8_baseline, coef, intercept)
            else:  # X8
                pred = regression_calibrated(X6_baseline, val, coef, intercept)
            predictions.append(pred)

        predictions = np.array(predictions)

        # 标准化（相对于范围）
        pred_range = predictions.max() - predictions.min()
        if pred_range > 0:
            predictions_norm = (predictions - predictions.min()) / pred_range
        else:
            predictions_norm = np.zeros_like(predictions)

        # 绘制曲线
        ax.plot(x_range, predictions_norm, color=colors[idx], linewidth=2.5,
                label=f"{var_info['short']}")

        # 标注基准值
        ax.axvline(x=var_info['data'].mean(), color='gray', linestyle='--',
                   alpha=0.5, linewidth=1)
        ax.text(var_info['data'].mean(), 0.95, 'mean', rotation=90,
               ha='right', va='top', fontsize=6, color='gray')

        # 填充区域
        ax.fill_between(x_range, 0, predictions_norm, alpha=0.15, color=colors[idx])

        # 设置坐标轴
        ax.set_xlabel(f'{var_info["short"]} Value', fontsize=9)
        ax.set_ylabel('Normalized Predicted Profit', fontsize=9)
        ax.set_title(f'({chr(97+idx)}) {var_info["short"]}: {var_info["name"]}',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        # 添加公式标注
        if var_key == 'X6':
            ax.text(0.5, 0.1, r'$\log(X6)^2$ (scaled)', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='#666666')
        else:
            ax.text(0.5, 0.1, r'$X8^{-0.5}$ (scaled)', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='#666666')

        # 保存数据
        sensitivity_data[var_key] = {
            'x_range': x_range,
            'predictions': predictions,
            'predictions_norm': predictions_norm
        }

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, 'sensitivity_regression.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  回归任务敏感性分析图已保存: {save_path}")
    plt.close()

    return sensitivity_data


def save_sensitivity_data(class_data, reg_data, save_dir):
    """保存敏感性分析数据到CSV"""
    print("\n[保存数据] 保存敏感性分析数据...")

    # 分类任务数据
    cls_df_list = []
    for var_key, data in class_data.items():
        df = pd.DataFrame({
            f'{var_key}_value_orig': data['x_range_orig'],
            f'{var_key}_value_scaled': data['x_range_scaled'],
            'decision_score': data['scores'],
            'normalized_score': data['scores_norm']
        })
        df['variable'] = var_key
        cls_df_list.append(df)

    cls_all = pd.concat(cls_df_list, ignore_index=True)
    cls_path = os.path.join(save_dir, 'sensitivity_classification_data.csv')
    cls_all.to_csv(cls_path, index=False, encoding='utf-8-sig')
    print(f"  分类任务数据已保存: {cls_path}")

    # 回归任务数据
    reg_df_list = []
    for var_key, data in reg_data.items():
        df = pd.DataFrame({
            f'{var_key}_value': data['x_range'],
            'predicted_profit': data['predictions'],
            'normalized_profit': data['predictions_norm']
        })
        df['variable'] = var_key
        reg_df_list.append(df)

    reg_all = pd.concat(reg_df_list, ignore_index=True)
    reg_path = os.path.join(save_dir, 'sensitivity_regression_data.csv')
    reg_all.to_csv(reg_path, index=False, encoding='utf-8-sig')
    print(f"  回归任务数据已保存: {reg_path}")


def create_combined_comparison_plot(class_data, reg_data, save_dir):
    """创建对比图：展示分类和回归任务的敏感性对比"""
    print("\n[对比图] 生成任务对比敏感性分析图...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 分类任务（左图）
    ax = axes[0]
    colors_cls = {'X0': '#3498db', 'X2': '#e74c3c', 'X4': '#2ecc71'}

    for var_key, data in class_data.items():
        # 计算敏感性指数（标准差）
        sensitivity = np.std(data['scores_norm'])
        ax.barh(var_key, sensitivity, color=colors_cls[var_key],
                alpha=0.7, edgecolor='black', linewidth=1)
        ax.text(sensitivity + 0.01, var_key, f'{sensitivity:.3f}',
               va='center', fontsize=8)

    ax.set_xlabel('Sensitivity Index (Std of Normalized Score)', fontsize=9)
    ax.set_title('(a) Classification Task\nDecision Score Sensitivity',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_ylim(-0.5, 2.5)

    # 回归任务（右图）
    ax = axes[1]
    colors_reg = {'X6': '#9b59b6', 'X8': '#f39c12'}

    for var_key, data in reg_data.items():
        sensitivity = np.std(data['predictions_norm'])
        ax.barh(var_key, sensitivity, color=colors_reg[var_key],
                alpha=0.7, edgecolor='black', linewidth=1)
        ax.text(sensitivity + 0.01, var_key, f'{sensitivity:.3f}',
               va='center', fontsize=8)

    ax.set_xlabel('Sensitivity Index (Std of Normalized Profit)', fontsize=9)
    ax.set_title('(b) Regression Task\nProfit Prediction Sensitivity',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_ylim(-0.5, 1.5)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, 'sensitivity_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  对比敏感性分析图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("模型敏感性分析图生成")
    print("=" * 70)

    # 创建输出目录
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 加载分类任务数据
    class_vars = load_classification_data()

    # 2. 加载回归任务数据
    reg_vars = load_regression_data()

    # 3. 生成分类任务敏感性分析图
    class_data = create_classification_sensitivity_plot(class_vars, figures_dir)

    # 4. 生成回归任务敏感性分析图
    reg_data = create_regression_sensitivity_plot(reg_vars, figures_dir)

    # 5. 保存数据
    save_sensitivity_data(class_data, reg_data, figures_dir)

    # 6. 生成对比图
    create_combined_comparison_plot(class_data, reg_data, figures_dir)

    print("\n" + "=" * 70)
    print("完成！生成的文件：")
    print("  1. figures/sensitivity_classification_<timestamp>.png")
    print("  2. figures/sensitivity_regression.png")
    print("  3. figures/sensitivity_comparison.png")
    print("  4. figures/sensitivity_classification_data.csv")
    print("  5. figures/sensitivity_regression_data.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
