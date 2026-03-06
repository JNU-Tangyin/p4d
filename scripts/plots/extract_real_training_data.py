#!/usr/bin/env python3
"""
提取真实PhySO训练数据
从已有的训练日志中提取完整的训练曲线数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def extract_training_data_from_log():
    """从训练日志中提取真实训练数据"""
    print("提取真实PhySO训练数据...")

    # 检查文件是否存在
    log_file = "results/demo.log"
    curves_file = "results/demo_curves_data.csv"

    if not Path(log_file).exists():
        print(f"错误: 找不到训练日志文件 {log_file}")
        return None

    if not Path(curves_file).exists():
        print(f"错误: 找不到训练曲线数据文件 {curves_file}")
        return None

    # 读取现有的训练曲线数据
    try:
        df = pd.read_csv(curves_file)
        print(f"现有训练数据: {len(df)} epochs")
        print("数据预览:")
        print(df[['epoch', 'mean_R', 'mean_R_train', 'loss']].to_string())

        return df
    except Exception as e:
        print(f"读取训练数据失败: {e}")
        return None

def extract_programs_from_log():
    """从日志中提取程序信息"""
    log_file = "results/demo.log"
    if not Path(log_file).exists():
        return None

    try:
        df = pd.read_csv(log_file)
        print(f"\n训练日志数据: {len(df)} 条记录")
        print(f"Epoch范围: {df['epoch'].min()} 到 {df['epoch'].max()}")

        # 按epoch分组统计
        epoch_stats = df.groupby('epoch').agg({
            'reward': ['mean', 'max', 'min'],
            'complexity': ['mean', 'max', 'min']
        }).round(6)

        print("\n每个epoch的统计:")
        print(epoch_stats)

        return df
    except Exception as e:
        print(f"读取训练日志失败: {e}")
        return None

def create_training_convergence_plot(df):
    """基于真实数据创建训练收敛图"""
    if df is None or len(df) < 2:
        print("数据不足，无法创建图表")
        return None

    print("\n创建基于真实数据的训练收敛图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PhySO Training Convergence (Real Data)', fontsize=16, fontweight='bold')

    # 1. R² Score Convergence
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['mean_R'], 'b-', linewidth=2.5, label='Test R²', marker='o', markersize=6)
    ax1.plot(df['epoch'], df['mean_R_train'], 'r-', linewidth=2.5, label='Train R²', marker='s', markersize=6)
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 显示实际数值
    for i, (epoch, test_r, train_r) in enumerate(zip(df['epoch'], df['mean_R'], df['mean_R_train'])):
        ax1.annotate(f'{test_r:.4f}', (epoch, test_r), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax1.annotate(f'{train_r:.4f}', (epoch, train_r), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # 2. Loss Function
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['loss'], 'g-', linewidth=2.5, marker='^', markersize=6)
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Function Convergence', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 显示损失数值
    for i, (epoch, loss) in enumerate(zip(df['epoch'], df['loss'])):
        ax2.annotate(f'{loss:.6f}', (epoch, loss), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # 3. Complexity Evolution
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['mean_complexity'], 'purple', linewidth=2.5, marker='D', markersize=6)
    ax3.set_xlabel('Training Epoch', fontsize=12)
    ax3.set_ylabel('Mean Complexity', fontsize=12)
    ax3.set_title('Expression Complexity Evolution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['max_R'], 'orange', linewidth=2.5, label='Max R²', marker='*', markersize=8)
    ax4.plot(df['epoch'], df['overall_max_R'], 'cyan', linewidth=2.5, label='Overall Max R²', marker='x', markersize=6)
    ax4.set_xlabel('Training Epoch', fontsize=12)
    ax4.set_ylabel('R² Score', fontsize=12)
    ax4.set_title('Performance Metrics Evolution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_file = 'PhySO_Training_Convergence_RealData.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"基于真实数据的训练收敛图已保存: {output_file}")

    # 移动到figures目录
    import shutil
    figures_dir = "改写论文/figures/v2_优化优化版/"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(output_file, figures_dir + output_file)
    print(f"图表已复制到: {figures_dir}{output_file}")

    plt.show()
    return output_file

def create_recommended_training_command():
    """创建推荐的完整训练命令"""
    print("\n" + "="*60)
    print("推荐运行完整训练的命令:")
    print("="*60)

    commands = [
        "# 回归任务训练 (完整的15个epoch)",
        "python main.py --dataset_name investment_decision --task_type regression --epochs 15 --seed 0",
        "",
        "# 分类任务训练 (完整的15个epoch)",
        "python main.py --dataset_name investment_decision --task_type classification --epochs 15 --seed 0",
        "",
        "# 带对比的回归任务训练",
        "python main.py --dataset_name investment_decision --task_type regression --epochs 15 --seed 0 --compare-classical"
    ]

    for cmd in commands:
        print(cmd)

    print("\n注意事项:")
    print("1. 确保PhySO环境已正确配置")
    print("2. 训练过程会自动保存到 results/ 目录")
    print("3. 训练完成后会生成 demo_curves.png 和 demo_curves_data.csv")
    print("4. 建议使用 screen 或 tmux 运行长时间训练")
    print("="*60)

def main():
    """主函数"""
    print("="*60)
    print("PhySO真实训练数据提取工具")
    print("="*60)

    # 提取现有训练数据
    training_data = extract_training_data_from_log()

    # 提取训练日志
    log_data = extract_programs_from_log()

    if training_data is not None:
        # 基于真实数据创建图表
        plot_file = create_training_convergence_plot(training_data)

        print(f"\n真实数据总结:")
        print(f"- 已完成epochs: {len(training_data)}")
        print(f"- 最终测试R2: {training_data['mean_R'].iloc[-1]:.4f}")
        print(f"- 最终训练R2: {training_data['mean_R_train'].iloc[-1]:.4f}")
        print(f"- 最终损失: {training_data['loss'].iloc[-1]:.6f}")

        if len(training_data) < 15:
            print(f"\n警告: 只完成 {len(training_data)}/15 个epoch")
            print(f"建议重新运行完整训练以获取完整的收敛数据")
    else:
        print("ERROR: 无法提取训练数据")

    # 显示推荐命令
    create_recommended_training_command()

if __name__ == "__main__":
    main()