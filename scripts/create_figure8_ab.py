#!/usr/bin/env python3
"""
Figure 8: Factor Relationship Network with Gating Visualization

生成两联图：
(a) 因子关系网络图（修正3个问题）
(b) cos² 门控可视化
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import scipy.stats

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

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

np.random.seed(42)


def load_data():
    """加载投资决策数据集"""
    data_path = os.path.join(PROJECT_ROOT, 'data', '项目数据收集表v2.0.xlsx')
    df = pd.read_excel(data_path, header=2)

    # 先编码目标变量
    if '是否应投资该项目' in df.columns:
        df['是否应投资该项目'] = df['是否应投资该项目'].map({'是': 1, '否': 0})

    # 删除Unnamed列
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_cols)

    # 只保留数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    return df


def DecimalScaler(data):
    """DecimalScaler 标准化（与模型一致）"""
    data = np.array(data, dtype=float)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        max_val = 1.0
    exponent = np.ceil(np.log10(max_val + 1e-10))
    norm_data = data / (10 ** max(exponent, -2))
    return np.clip(norm_data, -1.0, 1.0)


def extract_variables(df):
    """提取公式中使用的变量"""
    X0_col = '去化期间物业管理费（万元）'
    X2_col = '去化期间利息支出或沉没（万元）'
    X4_col = '当期商铺对外每月租金（万元）'

    X0 = df[X0_col].values
    X2 = df[X2_col].values
    X4 = df[X4_col].values

    # DecimalScaler 标准化
    X0_scaled = DecimalScaler(X0)
    X2_scaled = DecimalScaler(X2)
    X4_scaled = DecimalScaler(X4)

    # 计算 cos² 门控值
    cos2_gating = np.cos(X0_scaled) ** 2

    return {
        'X0': {'name': 'Property Management Fee', 'short': 'X0', 'data': X0, 'scaled': X0_scaled},
        'X2': {'name': 'Interest Expense', 'short': 'X2', 'data': X2, 'scaled': X2_scaled},
        'X4': {'name': 'Commercial Rent', 'short': 'X4', 'data': X4, 'scaled': X4_scaled},
    }, cos2_gating


def calculate_spearman_correlation(variables):
    """计算变量之间的Spearman相关系数"""
    data_matrix = np.column_stack([
        variables['X0']['data'],
        variables['X2']['data'],
        variables['X4']['data']
    ])

    correlations = {}
    pairs = [('X0', 'X2'), ('X0', 'X4'), ('X2', 'X4')]

    for var1, var2 in pairs:
        idx1 = ['X0', 'X2', 'X4'].index(var1)
        idx2 = ['X0', 'X2', 'X4'].index(var2)

        rho, p_value = scipy.stats.spearmanr(data_matrix[:, idx1], data_matrix[:, idx2])
        correlations[f'{var1}_{var2}'] = {
            'rho': rho,
            'p_value': p_value,
            'abs_rho': abs(rho)
        }

    return correlations


def draw_fancy_arrow(ax, start, end, color, annotation, force_center=False):
    """绘制从start到end的箭头"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        dx /= length
        dy /= length

    offset = 0.05
    start_pos = (start[0] + dx * offset, start[1] + dy * offset)
    end_pos = (end[0] - dx * offset, end[1] - dy * offset)

    arrow = FancyArrowPatch(start_pos, end_pos,
                           arrowstyle='->,head_width=0.15,head_length=0.1',
                           color=color, linewidth=2.5, zorder=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

    # 添加标注
    mid_x = (start_pos[0] + end_pos[0]) / 2
    mid_y = (start_pos[1] + end_pos[1]) / 2

    if force_center or abs(dy) > abs(dx):
        ha = 'center'
        va = 'center'
    else:
        ha = 'center'
        va = 'bottom' if mid_y > 0 else 'top'

    # 深灰色文字
    text_color = '#444444'
    ax.text(mid_x, mid_y, annotation, fontsize=8, ha=ha, va=va,
           color=text_color, fontweight='normal')


def create_subplot_a(variables, correlations, ax):
    """
    创建子图 (a): 因子关系网络图

    修正：
    1. 标题中的 \n 正常换行
    2. 图例中移除 |ρ|>1 的示例
    3. 图例明确节点颜色与相关性无关
    """
    # 定义节点位置
    positions = {
        'X4': (0, 0.6),
        'X0': (-0.5, 0),
        'X2': (0.5, 0),
        'Score': (0, -0.5)
    }

    # 节点样式
    node_style = {
        'X4': {'color': '#2ecc71', 'label': 'Commercial Rent\n(X4)', 'size': 200},
        'X0': {'color': '#3498db', 'label': 'Property Fee\n(X0)', 'size': 200},
        'X2': {'color': '#e74c3c', 'label': 'Interest Expense\n(X2)', 'size': 200},
        'Score': {'color': '#9b59b6', 'label': 'Decision\nScore', 'size': 250}
    }

    # 绘制耦合边（虚线）
    rho_X4_X2 = correlations['X2_X4']['rho']
    width_X4_X2 = 1 + 4 * correlations['X2_X4']['abs_rho']
    color_X4_X2 = '#27ae60' if rho_X4_X2 > 0 else '#c0392b'

    ax.plot([positions['X4'][0], positions['X2'][0]],
            [positions['X4'][1], positions['X2'][1]],
            linestyle='--', linewidth=width_X4_X2, color=color_X4_X2,
            alpha=0.6, zorder=1)

    # 标注相关系数
    mid_x = (positions['X4'][0] + positions['X2'][0]) / 2
    mid_y = (positions['X4'][1] + positions['X2'][1]) / 2 + 0.05
    ax.text(mid_x, mid_y, f'ρ={rho_X4_X2:.2f}', fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # X0 -- X4
    rho_X0_X4 = correlations['X0_X4']['rho']
    width_X0_X4 = 1 + 4 * correlations['X0_X4']['abs_rho']
    color_X0_X4 = '#27ae60' if rho_X0_X4 > 0 else '#c0392b'

    ax.plot([positions['X0'][0], positions['X4'][0]],
            [positions['X0'][1], positions['X4'][1]],
            linestyle='--', linewidth=width_X0_X4, color=color_X0_X4,
            alpha=0.6, zorder=1)

    mid_x = (positions['X0'][0] + positions['X4'][0]) / 2 - 0.1
    mid_y = (positions['X0'][1] + positions['X4'][1]) / 2 + 0.05
    ax.text(mid_x, mid_y, f'ρ={rho_X0_X4:.2f}', fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # 绘制有向边
    draw_fancy_arrow(ax, positions['X4'], positions['Score'],
                     color='#2ecc71', annotation='positive, squared', force_center=True)

    draw_fancy_arrow(ax, positions['X2'], positions['Score'],
                     color='#e74c3c', annotation='penalty, squared\ndenominator')

    draw_fancy_arrow(ax, positions['X0'], positions['Score'],
                     color='#3498db', annotation='nonlinear\ngating: cos^2')

    # 绘制节点
    for node_id, pos in positions.items():
        style = node_style[node_id]
        circle = plt.Circle(pos, radius=np.sqrt(style['size'])/100,
                           facecolor=style['color'], edgecolor='black',
                           linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], style['label'], fontsize=8, ha='center',
                va='center', fontweight='bold', zorder=3)

    # 设置坐标轴
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # 【修正1】标题正常换行，公式规范化
    ax.set_title('(a) Factor Relationship Network\nFormula: Decision Score = (X4^2 · cos^2(X0)) / X2^2',
                 fontsize=9, fontweight='bold')

    # 【修正2】【修正7】图例移除 |ρ|>1，只保留 "Edge width ~ |rho|"
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Contribution to Score'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Formula-implied Coupling'),
        mpatches.Patch(color='#27ae60', label='Positive Correlation (rho > 0)'),
        mpatches.Patch(color='#c0392b', label='Negative Correlation (rho < 0)'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, label='Edge width ~ |rho|'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.15, 1.0),
              fontsize=7, framealpha=0.9)

    # 添加子图标注
    ax.text(-0.95, 0.9, '(a)', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))


def create_subplot_b(variables, cos2_gating, ax):
    """
    创建子图: cos² 门控可视化

    x轴：DecimalScaler 后的 X0 值
    y轴：cos²(X0)
    散点：71个项目的实际数据
    曲线：理论曲线 y = cos²(x)

    修正：
    2. 右图主要部分缩小一圈（约4/5）
    3. 去掉左上角重复的 "(b)" 标签
    4. High/Low gating 改为虚线阈值 + 小标签
    5. x轴缩小到数据的 1%-99% 分位区间
    """
    X0_scaled = variables['X0']['scaled']

    # 【修正5】x轴缩小到数据的 1%-99% 分位区间（约4/5范围）
    x_min, x_max = np.percentile(X0_scaled, [1, 99])
    # 理论曲线范围也相应缩小（进一步缩小一圈）
    x_theory = np.linspace(x_min - 0.02, x_max + 0.02, 300)
    y_theory = np.cos(x_theory) ** 2

    # 绘制理论曲线
    ax.plot(x_theory, y_theory, 'b-', linewidth=2, alpha=0.7, label='Theory: cos^2(x)')

    # 绘制散点（半透明）
    ax.scatter(X0_scaled, cos2_gating, c='red', s=25, alpha=0.5,
               edgecolors='darkred', linewidth=0.5, label='Project Data (n=71)')

    # 门控阈值标注（数据几乎都在 high gating 区域）
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(x_max * 0.65, 0.92, 'mostly high gating in observed range', fontsize=9,
            color='#444444', ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, edgecolor='none'))

    # 设置坐标轴（y轴收缩到数据范围，进一步缩小）
    y_min, y_max = cos2_gating.min(), cos2_gating.max()
    ax.set_xlabel('Scaled X0 (Property Fee)', fontsize=9)
    ax.set_ylabel('cos^2(X0)', fontsize=9)
    ax.set_xlim(x_min - 0.015, x_max + 0.015)
    ax.set_ylim(max(0.35, y_min - 0.03), min(1.01, y_max + 0.03))

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 图例
    ax.legend(fontsize=7, loc='upper right')

    # 标题
    ax.set_title('(b) Nonlinear Gating Effect: cos^2(X0)', fontsize=9, fontweight='bold')

    # 【修正3】去掉左上角重复的 "(b)" 标签（保留标题里的即可）


def save_gating_data(variables, cos2_gating, save_dir):
    """保存门控数据到CSV"""
    df = pd.DataFrame({
        'X0_original': variables['X0']['data'],
        'X0_scaled': variables['X0']['scaled'],
        'cos2_gating': cos2_gating
    })
    save_path = os.path.join(save_dir, 'factor_relationship_gating.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[结果] 门控数据已保存到: {save_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Figure 8: Factor Relationship Network with Gating Visualization")
    print("=" * 70)

    # 创建输出目录
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    df = load_data()
    print(f"数据加载完成：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 2. 提取变量
    print("\n[步骤2] 提取公式变量...")
    variables, cos2_gating = extract_variables(df)
    print("变量提取完成：")
    for key, var in variables.items():
        data_min = var['scaled'].min()
        data_max = var['scaled'].max()
        print(f"  {key}: {var['name']} (标准化后范围: [{data_min:.3f}, {data_max:.3f}])")

    # 3. 计算相关系数
    print("\n[步骤3] 计算Spearman相关系数...")
    correlations = calculate_spearman_correlation(variables)
    print("相关系数计算完成：")
    for key, corr in correlations.items():
        print(f"  {key}: ρ={corr['rho']:.4f}, p={corr['p_value']:.4f}")

    # 4. 保存门控数据
    print("\n[步骤4] 保存门控数据...")
    save_gating_data(variables, cos2_gating, figures_dir)

    # 5. 创建两联图（1×2 横排，紧凑方形适合双栏论文）
    print("\n[步骤5] 生成 Figure 8 (a)(b)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    create_subplot_a(variables, correlations, ax1)
    create_subplot_b(variables, cos2_gating, ax2)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(figures_dir, 'factor_relationship_network_ab.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[结果] Figure 8 已保存到: {save_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("完成！生成的文件：")
    print(f"  1. {save_path}")
    print(f"  2. {os.path.join(figures_dir, 'factor_relationship_gating.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
