#!/usr/bin/env python3
"""
PhySO Training Convergence Plot
基于真实训练数据制作PhySO训练收敛图
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_training_data():
    """加载训练数据"""
    # 查找最新的实验数据目录
    experiment_dir = Path("archive/old_experiments/experiment_data")
    if not experiment_dir.exists():
        print("No experiment_data directory found in archive!")
        return None

    # 查找最新的数据目录
    data_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    if not data_dirs:
        print("No experiment data directories found!")
        return None

    # 按修改时间排序，获取最新的
    latest_dir = max(data_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Loading data from: {latest_dir}")

    # 加载训练曲线数据
    training_file = latest_dir / "training" / "physo_training_curves.csv"
    if not training_file.exists():
        print(f"Training curves file not found: {training_file}")
        return None

    df = pd.read_csv(training_file)
    print(f"Loaded training data: {len(df)} epochs")
    return df

def plot_training_convergence(df):
    """绘制训练收敛图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PhySO Training Convergence Analysis', fontsize=16, fontweight='bold')

    # 1. R² Score Convergence (mean_R and mean_R_train)
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['mean_R'], 'b-', linewidth=2.5, label='Test R²', marker='o', markersize=6)
    ax1.plot(df['epoch'], df['mean_R_train'], 'r-', linewidth=2.5, label='Train R²', marker='s', markersize=6)
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.6)

    # 2. Loss Function
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['loss'], 'g-', linewidth=2.5, marker='^', markersize=6)
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Function Convergence', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Complexity Evolution
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['mean_complexity'], 'purple', linewidth=2.5, marker='D', markersize=6)
    ax3.set_xlabel('Training Epoch', fontsize=12)
    ax3.set_ylabel('Mean Complexity', fontsize=12)
    ax3.set_title('Expression Complexity Evolution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics Comparison
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['max_R'], 'orange', linewidth=2.5, label='Max R²', marker='*', markersize=8)
    ax4.plot(df['epoch'], df['overall_max_R'], 'cyan', linewidth=2.5, label='Overall Max R²', marker='x', markersize=6)
    ax4.set_xlabel('Training Epoch', fontsize=12)
    ax4.set_ylabel('R² Score', fontsize=12)
    ax4.set_title('Performance Metrics Evolution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 0.6)

    plt.tight_layout()

    # 保存图像
    output_file = 'PhySO_Training_Convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training convergence plot saved as: {output_file}")

    plt.show()
    return output_file

def create_summary_statistics(df):
    """创建训练统计摘要"""
    print("\n" + "="*60)
    print("PhySO Training Summary Statistics")
    print("="*60)

    print(f"Total Epochs: {len(df)}")
    print(f"Final Test R2: {df['mean_R'].iloc[-1]:.4f}")
    print(f"Final Train R2: {df['mean_R_train'].iloc[-1]:.4f}")
    print(f"Final Loss: {df['loss'].iloc[-1]:.6f}")
    print(f"Max Test R2 Achieved: {df['mean_R'].max():.4f}")
    print(f"Max Train R2 Achieved: {df['mean_R_train'].max():.4f}")
    print(f"Min Loss Achieved: {df['loss'].min():.6f}")
    print(f"Final Mean Complexity: {df['mean_complexity'].iloc[-1]:.2f}")
    print(f"Complexity Range: {df['mean_complexity'].min():.2f} - {df['mean_complexity'].max():.2f}")

    # 检查收敛性
    r2_converged = abs(df['mean_R'].iloc[-1] - df['mean_R'].iloc[-5]) < 0.01 if len(df) >= 5 else False
    print(f"R2 Converged: {r2_converged}")

    return True

def main():
    """主函数"""
    print("="*60)
    print("PhySO Training Convergence Visualization")
    print("="*60)

    # 加载训练数据
    df = load_training_data()
    if df is None:
        print("Failed to load training data!")
        return

    # 显示数据预览
    print(f"\nTraining Data Preview:")
    print(df[['epoch', 'mean_R', 'mean_R_train', 'loss', 'mean_complexity']].head())

    # 创建统计摘要
    create_summary_statistics(df)

    # 绘制收敛图
    output_file = plot_training_convergence(df)

    print("\n" + "="*60)
    print("Training convergence plot created successfully!")
    print(f"Output file: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()