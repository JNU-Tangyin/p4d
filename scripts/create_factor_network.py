#!/usr/bin/env python3
"""
因子关系网络图生成脚本

功能：
1. 绘制变量关系网络图（分类任务公式可视化）
2. 计算Spearman相关系数作为边的属性
3. 输出图片和CSV文件（可复现）

论文公式：投资决策 = X4² × cos²(X0) / X2²
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import scipy.stats
import json

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
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
})

# 固定随机种子以确保可复现
np.random.seed(42)


def load_data():
    """加载投资决策数据集"""
    # 读取原始数据
    data_path = os.path.join(PROJECT_ROOT, 'data', '项目数据收集表v2.0.xlsx')
    df = pd.read_excel(data_path, header=2)

    # 先编码目标变量（在删除非数值列之前）
    if '是否应投资该项目' in df.columns:
        df['是否应投资该项目'] = df['是否应投资该项目'].map({'是': 1, '否': 0})

    # 删除Unnamed列
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_cols)

    # 只保留数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    return df


def extract_variables(df):
    """
    提取公式中使用的变量

    论文公式：投资决策 = X4² × cos²(X0) / X2²
    映射：
    - X0 = 去化期间物业管理费（万元）= 原始数据列21
    - X2 = 去化期间利息支出或沉没（万元）= 原始数据列23
    - X4 = 当期商铺对外每月租金（万元）= 原始数据列24
    """
    # 根据列名提取数据
    X0_col = '去化期间物业管理费（万元）'
    X2_col = '去化期间利息支出或沉没（万元）'
    X4_col = '当期商铺对外每月租金（万元）'
    y_col = '是否应投资该项目'

    # 提取数据
    X0 = df[X0_col].values
    X2 = df[X2_col].values
    X4 = df[X4_col].values
    y_class = df[y_col].values

    # 计算决策分数（使用论文公式）
    # 注意：需要使用DecimalScaler标准化后的值
    def DecimalScaler(data):
        data = np.array(data, dtype=float)
        max_val = np.max(np.abs(data))
        if max_val == 0:
            max_val = 1.0
        exponent = np.ceil(np.log10(max_val + 1e-10))
        norm_data = data / (10 ** max(exponent, -2))
        return np.clip(norm_data, -1.0, 1.0)

    X0_scaled = DecimalScaler(X0)
    X2_scaled = DecimalScaler(X2)
    X4_scaled = DecimalScaler(X4)

    # 计算决策分数
    decision_score = (X4_scaled ** 2) * (np.cos(X0_scaled) ** 2) / (X2_scaled ** 2 + 1e-10)

    return {
        'X0': {'name': 'Property Management Fee', 'short': 'X0', 'data': X0, 'scaled': X0_scaled},
        'X2': {'name': 'Interest Expense', 'short': 'X2', 'data': X2, 'scaled': X2_scaled},
        'X4': {'name': 'Commercial Rent', 'short': 'X4', 'data': X4, 'scaled': X4_scaled},
        'Score': {'name': 'Decision Score', 'short': 'Score', 'data': decision_score}
    }, y_class


def calculate_spearman_correlation(variables):
    """计算变量之间的Spearman相关系数"""
    # 提取原始数据（用于计算相关性）
    data_matrix = np.column_stack([
        variables['X0']['data'],
        variables['X2']['data'],
        variables['X4']['data']
    ])

    # 计算Spearman相关系数
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


def create_network_diagram(variables, correlations, save_dir):
    """
    创建因子关系网络图

    布局：
        X4
       /   \
      /     \
    X0  ---  X2
      \     /
       \   /
        Score
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 定义节点位置（固定坐标）
    # 使用笛卡尔坐标系，中心在(0, 0)
    positions = {
        'X4': (0, 0.6),      # 上
        'X0': (-0.5, 0),     # 左
        'X2': (0.5, 0),      # 右
        'Score': (0, -0.5)   # 下（中心）
    }

    # 节点样式（直径缩小一半：size除以4）
    node_style = {
        'X4': {'color': '#2ecc71', 'label': 'Commercial Rent\n(X4)', 'size': 200},
        'X0': {'color': '#3498db', 'label': 'Property Fee\n(X0)', 'size': 200},
        'X2': {'color': '#e74c3c', 'label': 'Interest Expense\n(X2)', 'size': 200},
        'Score': {'color': '#9b59b6', 'label': 'Decision\nScore', 'size': 250}
    }

    # 绘制耦合边（虚线，表示统计依赖）
    # X4 -- X2 (ratio/trade-off)
    rho_X4_X2 = correlations['X2_X4']['rho']  # 注意：键名是 'X2_X4'
    width_X4_X2 = 1 + 4 * correlations['X2_X4']['abs_rho']  # |rho|映射到粗细 [1, 5]
    color_X4_X2 = '#27ae60' if rho_X4_X2 > 0 else '#c0392b'  # 正绿/负红

    ax.plot([positions['X4'][0], positions['X2'][0]],
            [positions['X4'][1], positions['X2'][1]],
            linestyle='--', linewidth=width_X4_X2, color=color_X4_X2,
            alpha=0.6, zorder=1, label='coupling' if 'X4_X2' == 'X4_X2' else '')

    # 标注相关系数
    mid_x = (positions['X4'][0] + positions['X2'][0]) / 2
    mid_y = (positions['X4'][1] + positions['X2'][1]) / 2 + 0.05
    ax.text(mid_x, mid_y, f'ρ={rho_X4_X2:.2f}', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # X0 -- X4 (modulation)
    rho_X0_X4 = correlations['X0_X4']['rho']
    width_X0_X4 = 1 + 4 * correlations['X0_X4']['abs_rho']
    color_X0_X4 = '#27ae60' if rho_X0_X4 > 0 else '#c0392b'

    ax.plot([positions['X0'][0], positions['X4'][0]],
            [positions['X0'][1], positions['X4'][1]],
            linestyle='--', linewidth=width_X0_X4, color=color_X0_X4,
            alpha=0.6, zorder=1)

    mid_x = (positions['X0'][0] + positions['X4'][0]) / 2 - 0.1
    mid_y = (positions['X0'][1] + positions['X4'][1]) / 2 + 0.05
    ax.text(mid_x, mid_y, f'ρ={rho_X0_X4:.2f}', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # 绘制有向边（实线箭头，表示公式贡献）
    # X4 -> Score (positive, squared)
    draw_fancy_arrow(ax, positions['X4'], positions['Score'],
                     color='#2ecc71', label='positive contribution',
                     annotation='positive, squared')

    # X2 -> Score (penalty, squared denominator)
    draw_fancy_arrow(ax, positions['X2'], positions['Score'],
                     color='#e74c3c', label='penalty',
                     annotation='penalty, squared\ndenominator')

    # X0 -> Score (nonlinear gating: cos^2)
    draw_fancy_arrow(ax, positions['X0'], positions['Score'],
                     color='#3498db', label='nonlinear gating',
                     annotation='nonlinear\ngating: cos²')

    # 绘制节点
    for node_id, pos in positions.items():
        style = node_style[node_id]

        # 绘制圆形节点
        circle = plt.Circle(pos, radius=np.sqrt(style['size'])/100,
                           facecolor=style['color'], edgecolor='black',
                           linewidth=2, zorder=2)
        ax.add_patch(circle)

        # 添加标签
        ax.text(pos[0], pos[1], style['label'], fontsize=10, ha='center',
                va='center', fontweight='bold', zorder=3)

    # 设置坐标轴
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # 创建图例
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Contribution to Score'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Formula-implied Coupling'),
        mpatches.Patch(color='#27ae60', label='Positive Correlation (ρ > 0)'),
        mpatches.Patch(color='#c0392b', label='Negative Correlation (ρ < 0)'),
        plt.Line2D([0], [0], color='gray', linewidth=1, label='|ρ| = 1'),
        plt.Line2D([0], [0], color='gray', linewidth=5, label='|ρ| = 5')
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.15, 1.0),
              fontsize=8, framealpha=0.9)

    plt.title('Factor Relationship Network\\nClassification Formula: Decision = X4² × cos²(X0) / X2²',
              fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, 'factor_relationship_network.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[结果] 网络图已保存到: {save_path}")
    plt.close()

    return positions, correlations


def draw_fancy_arrow(ax, start, end, color, label, annotation):
    """绘制从start到end的箭头"""
    # 计算方向向量
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)

    # 归一化方向向量
    if length > 0:
        dx /= length
        dy /= length

    # 箭头起点和终点（缩短以避免与节点重叠）
    # 节点半径约为 sqrt(size)/100 ≈ 0.045（对于size=200）
    offset = 0.05  # 节点半径
    start_pos = (start[0] + dx * offset, start[1] + dy * offset)
    end_pos = (end[0] - dx * offset, end[1] - dy * offset)

    # 绘制箭头
    arrow = FancyArrowPatch(start_pos, end_pos,
                           arrowstyle='->,head_width=0.15,head_length=0.1',
                           color=color, linewidth=2.5, zorder=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

    # 添加标注（在边的中点）
    mid_x = (start_pos[0] + end_pos[0]) / 2
    mid_y = (start_pos[1] + end_pos[1]) / 2

    # 根据方向调整标注位置
    if abs(dy) > abs(dx):  # 垂直边
        ha = 'right' if start_pos[0] < 0 else 'left'
        va = 'center'
    else:  # 水平或对角边
        ha = 'center'
        va = 'bottom' if mid_y > 0 else 'top'

    ax.text(mid_x + 0.1 if ha == 'left' else mid_x - 0.1 if ha == 'right' else mid_x,
           mid_y + 0.05 if va == 'top' else mid_y - 0.05 if va == 'bottom' else mid_y,
           annotation, fontsize=7, ha=ha, va=va, color=color, fontweight='bold')


def save_edge_data(correlations, save_dir):
    """保存边数据到CSV"""
    edges = []

    # 添加贡献边（变量 -> Score）
    contribution_edges = [
        {'source': 'X4', 'target': 'Score', 'edge_type': 'contribution', 'relation': 'positive, squared', 'rho': '', 'abs_rho': ''},
        {'source': 'X0', 'target': 'Score', 'edge_type': 'contribution', 'relation': 'nonlinear gating: cos²', 'rho': '', 'abs_rho': ''},
        {'source': 'X2', 'target': 'Score', 'edge_type': 'contribution', 'relation': 'penalty, squared denominator', 'rho': '', 'abs_rho': ''}
    ]
    edges.extend(contribution_edges)

    # 添加耦合边（注意：键名是按字母顺序的）
    coupling_edges = [
        {'source': 'X4', 'target': 'X2', 'edge_type': 'coupling', 'relation': 'ratio/trade-off',
         'rho': correlations['X2_X4']['rho'], 'abs_rho': correlations['X2_X4']['abs_rho']},
        {'source': 'X0', 'target': 'X4', 'edge_type': 'coupling', 'relation': 'modulation',
         'rho': correlations['X0_X4']['rho'], 'abs_rho': correlations['X0_X4']['abs_rho']}
    ]
    edges.extend(coupling_edges)

    df = pd.DataFrame(edges)
    save_path = os.path.join(save_dir, 'factor_relationship_edges.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[结果] 边数据已保存到: {save_path}")


def save_node_data(variables, save_dir):
    """保存节点数据到CSV"""
    nodes = []

    for var_key, var_data in variables.items():
        if var_key == 'Score':
            nodes.append({
                'node': 'Score',
                'label': 'Decision Score',
                'role': 'center',
                'description': 'Investability metric from formula: X4² × cos²(X0) / X2²'
            })
        else:
            nodes.append({
                'node': var_data['short'],
                'label': var_data['name'],
                'role': 'input_variable',
                'chinese_name': get_chinese_name(var_data['short']),
                'data_column_index': get_column_index(var_data['short'])
            })

    df = pd.DataFrame(nodes)
    save_path = os.path.join(save_dir, 'factor_relationship_nodes.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[结果] 节点数据已保存到: {save_path}")


def get_chinese_name(var_short):
    """获取变量的中文名称"""
    mapping = {
        'X0': '去化期间物业管理费（万元）',
        'X2': '去化期间利息支出或沉没（万元）',
        'X4': '当期商铺对外每月租金（万元）'
    }
    return mapping.get(var_short, '')


def get_column_index(var_short):
    """获取变量在原始数据中的列索引"""
    # 根据之前的分析
    mapping = {
        'X0': 21,  # 去化期间物业管理费（万元）
        'X2': 23,  # 去化期间利息支出或沉没（万元）
        'X4': 24   # 当期商铺对外每月租金（万元）
    }
    return mapping.get(var_short, '')


def main():
    """主函数"""
    print("=" * 70)
    print("因子关系网络图生成工具")
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
    variables, y_class = extract_variables(df)
    print("变量提取完成：")
    for key, var in variables.items():
        if key != 'Score':
            data_min = var['data'].min()
            data_max = var['data'].max()
            print(f"  {key}: {var['name']} (范围: [{data_min:.2f}, {data_max:.2f}])")

    # 3. 计算相关系数
    print("\n[步骤3] 计算Spearman相关系数...")
    correlations = calculate_spearman_correlation(variables)
    print("相关系数计算完成：")
    for key, corr in correlations.items():
        print(f"  {key}: ρ={corr['rho']:.4f}, p={corr['p_value']:.4f}")

    # 4. 绘制网络图
    print("\n[步骤4] 绘制网络图...")
    positions, correlations = create_network_diagram(variables, correlations, figures_dir)

    # 5. 保存数据
    print("\n[步骤5] 保存边数据和节点数据...")
    save_edge_data(correlations, figures_dir)
    save_node_data(variables, figures_dir)

    print("\n" + "=" * 70)
    print("完成！生成的文件：")
    print(f"  1. {os.path.join(figures_dir, 'factor_relationship_network.png')}")
    print(f"  2. {os.path.join(figures_dir, 'factor_relationship_edges.csv')}")
    print(f"  3. {os.path.join(figures_dir, 'factor_relationship_nodes.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
