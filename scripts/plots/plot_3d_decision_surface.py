#!/usr/bin/env python3
"""
3D Decision Boundary Visualization using PhySO Results
使用PhySO训练结果绘制3D决策曲面
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys
import os
import json
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set matplotlib parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_classification_data():
    """加载分类任务的训练数据"""
    data_path = 'results/classification_data.pkl'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到分类数据文件: {data_path}\n请先运行: python main.py --task_type classification")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"[INFO] 成功加载分类数据")
    print(f"[INFO] PhySO表达式: {data['best_expr_info']['clean_expression']}")
    print(f"[INFO] 目标列: {data['target_column']}")
    print(f"[INFO] 训练准确率: {data['train_metrics']['accuracy']:.1%}")
    print(f"[INFO] 测试准确率: {data['test_metrics']['accuracy']:.1%}")

    return data

def plot_3d_decision_boundary():
    """绘制3D决策边界"""
    print("="*60)
    print("绘制3D决策边界和决策曲面...")
    print("="*60)

    # 加载数据
    data = load_classification_data()

    # 获取训练数据
    X_train = data['X_train'].T  # 转换为 (n_samples, n_features)
    y_train = data['y_train']
    y_train_pred = data['y_train_pred']

    print(f"\n[数据信息]")
    print(f"  训练样本数: {len(y_train)}")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  类别0: {sum(y_train==0)}, 类别1: {sum(y_train==1)}")

    # PCA降维到3D
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)

    print(f"\n[PCA信息]")
    print(f"  主成分1解释方差: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  主成分2解释方差: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  主成分3解释方差: {pca.explained_variance_ratio_[2]:.3f}")
    print(f"  累计解释方差: {sum(pca.explained_variance_ratio_):.3f}")

    # 使用逻辑回归拟合PCA空间（用于可视化决策边界）
    print(f"\n[正在拟合可视化模型...]")
    lr_model = LogisticRegression(random_state=0, max_iter=1000)
    lr_model.fit(X_3d, y_train)
    lr_acc = lr_model.score(X_3d, y_train)
    print(f"  可视化模型准确率: {lr_acc:.1%}")
    print(f"  PhySO模型准确率: {data['train_metrics']['accuracy']:.1%}")

    # 创建网格用于绘制决策曲面
    x_min, x_max = X_3d[:, 0].min() - 1, X_3d[:, 0].max() + 1
    y_min, y_max = X_3d[:, 1].min() - 1, X_3d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 40),
        np.linspace(y_min, y_max, 40)
    )

    # 在每个网格点上预测
    z_mean = X_3d[:, 2].mean()
    Z = np.zeros_like(xx)

    print(f"\n[正在生成决策曲面...]")
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = lr_model.predict_proba([[xx[i, j], yy[i, j], z_mean]])[0, 1]

    # ========== 图1: 3D决策曲面 + 散点 ==========
    print(f"\n[绘制] 3D决策曲面图...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制决策曲面
    surf = ax.plot_surface(xx, yy, Z, cmap='RdYlBu_r', alpha=0.6,
                          linewidth=0, antialiased=True, vmin=0, vmax=1)

    # 绘制数据点
    mask_0 = y_train == 0
    mask_1 = y_train == 1

    ax.scatter(X_3d[mask_0, 0], X_3d[mask_0, 1], X_3d[mask_0, 2],
               c='red', marker='o', s=100, alpha=0.9,
               edgecolors='darkred', linewidths=1.5, label='Not Invest (Class 0)')
    ax.scatter(X_3d[mask_1, 0], X_3d[mask_1, 1], X_3d[mask_1, 2],
               c='blue', marker='^', s=100, alpha=0.9,
               edgecolors='darkblue', linewidths=1.5, label='Invest (Class 1)')

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Investment Probability', fontsize=12)

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.set_title(f'3D Decision Surface - PhySO Classification Results\n' +
                 f'Expression: {data["best_expr_info"]["clean_expression"]}\n' +
                 f'PhySO Accuracy: {data["train_metrics"]["accuracy"]:.1%}',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output1 = 'results/3d_decision_surface_physo.png'
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"  保存: {output1}")
    plt.close()

    # ========== 图2: 不同视角的3D图 ==========
    print(f"[绘制] 多视角3D图...")
    fig = plt.figure(figsize=(18, 6))

    angles = [(20, 45), (20, 135), (60, 45)]
    titles = ['View 1: Front', 'View 2: Side', 'View 3: Top']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # 决策曲面
        ax.plot_surface(xx, yy, Z, cmap='RdYlBu_r', alpha=0.6,
                       linewidth=0, antialiased=True, vmin=0, vmax=1)

        # 数据点
        ax.scatter(X_3d[mask_0, 0], X_3d[mask_0, 1], X_3d[mask_0, 2],
                   c='red', marker='o', s=80, alpha=0.8,
                   edgecolors='darkred', linewidths=1)
        ax.scatter(X_3d[mask_1, 0], X_3d[mask_1, 1], X_3d[mask_1, 2],
                   c='blue', marker='^', s=80, alpha=0.8,
                   edgecolors='darkblue', linewidths=1)

        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_zlabel('PC3', fontsize=10)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output2 = 'results/3d_multi_view_physo.png'
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"  保存: {output2}")
    plt.close()

    # ========== 图3: 2D等高线图 ==========
    print(f"[绘制] 2D等高线决策边界...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：连续概率等高线
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.9)
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    axes[0].scatter(X_3d[y_train==0, 0], X_3d[y_train==0, 1], c='red', marker='o', s=100,
                   edgecolors='black', linewidths=1.5, label='Not Invest', zorder=3)
    axes[0].scatter(X_3d[y_train==1, 0], X_3d[y_train==1, 1], c='blue', marker='^', s=100,
                   edgecolors='black', linewidths=1.5, label='Invest', zorder=3)
    axes[0].set_xlabel('PCA Component 1', fontsize=12)
    axes[0].set_ylabel('PCA Component 2', fontsize=12)
    axes[0].set_title('Decision Probability Contour (PhySO Results)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_aspect('equal')
    cbar1 = fig.colorbar(contour, ax=axes[0])
    cbar1.set_label('Probability', fontsize=11)

    # 右图：二元决策边界
    axes[1].contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ffcccc', '#cce5ff'], alpha=0.9)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    axes[1].scatter(X_3d[y_train==0, 0], X_3d[y_train==0, 1], c='red', marker='o', s=100,
                   edgecolors='black', linewidths=1.5, label='Not Invest', zorder=3)
    axes[1].scatter(X_3d[y_train==1, 0], X_3d[y_train==1, 1], c='blue', marker='^', s=100,
                   edgecolors='black', linewidths=1.5, label='Invest', zorder=3)
    axes[1].set_xlabel('PCA Component 1', fontsize=12)
    axes[1].set_ylabel('PCA Component 2', fontsize=12)
    axes[1].set_title('Binary Decision Boundary (PhySO Results)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    output3 = 'results/2d_decision_boundary_physo.png'
    plt.savefig(output3, dpi=300, bbox_inches='tight')
    print(f"  保存: {output3}")
    plt.close()

    # ========== 图4: 只有散点的3D图 ==========
    print(f"[绘制] 3D散点图（无决策曲面）...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_3d[mask_0, 0], X_3d[mask_0, 1], X_3d[mask_0, 2],
               c='red', marker='o', s=80, alpha=0.7, label='Not Invest (Class 0)')
    ax.scatter(X_3d[mask_1, 0], X_3d[mask_1, 1], X_3d[mask_1, 2],
               c='blue', marker='^', s=80, alpha=0.7, label='Invest (Class 1)')

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.set_title('3D Scatter Plot - PhySO Class Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output4 = 'results/3d_scatter_physo.png'
    plt.savefig(output4, dpi=300, bbox_inches='tight')
    print(f"  保存: {output4}")
    plt.close()

    print("\n" + "="*60)
    print("完成！生成的文件:")
    print(f"  1. {output1} - 3D决策曲面（推荐）")
    print(f"  2. {output2} - 多视角3D图")
    print(f"  3. {output3} - 2D等高线决策边界")
    print(f"  4. {output4} - 3D散点图")
    print(f"\n[说明]")
    print(f"  - 数据点来自PhySO的实际训练集")
    print(f"  - 决策曲面展示PCA空间的分类边界")
    print(f"  - PhySO表达式: {data['best_expr_info']['clean_expression']}")
    print("="*60)

def main():
    """主函数"""
    plot_3d_decision_boundary()

if __name__ == "__main__":
    main()
