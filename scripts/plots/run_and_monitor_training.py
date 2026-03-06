#!/usr/bin/env python3
"""
运行并监控PhySO训练
自动运行main.py并实时监控训练过程，生成准确的训练收敛图
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
from pathlib import Path

def wait_for_training_completion(timeout_minutes=30):
    """等待训练完成并收集数据"""
    print("等待训练完成...")

    results_dir = Path("results")
    curves_file = results_dir / "demo_curves_data.csv"

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        if curves_file.exists():
            try:
                # 检查文件是否完整
                df = pd.read_csv(curves_file)
                if len(df) >= 15:
                    print(f"训练完成! 检测到 {len(df)} epochs")
                    return df
                elif len(df) > 0:
                    print(f"训练进行中... 已完成 {len(df)}/15 epochs")
                    print(f"最新Epoch {len(df)-1}: R²={df['mean_R'].iloc[-1]:.4f}")
            except:
                pass

        time.sleep(10)  # 每10秒检查一次

    print("训练超时或未完成")
    return None

def create_final_training_plot(df):
    """基于最终训练数据创建准确的收敛图"""
    if df is None:
        print("没有有效的训练数据")
        return None

    print(f"\n创建基于真实训练数据的收敛图...")
    print(f"训练数据: {len(df)} epochs")

    # 显示最终统计
    final_data = df.iloc[-1]
    print(f"\n最终训练结果:")
    print(f"- 测试集R²: {final_data['mean_R']:.4f}")
    print(f"- 训练集R²: {final_data['mean_R_train']:.4f}")
    print(f"- 最终损失: {final_data['loss']:.6f}")
    print(f"- 表达式复杂度: {final_data['mean_complexity']:.2f}")

    # 创建高质量的训练收敛图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PhySO Training Convergence Analysis', fontsize=16, fontweight='bold')

    # 1. R² Score Convergence
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['mean_R'], 'b-', linewidth=2.5, label='Test R²', marker='o', markersize=6)
    ax1.plot(df['epoch'], df['mean_R_train'], 'r-', linewidth=2.5, label='Train R²', marker='s', markersize=6)
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 添加最终值标注
    ax1.annotate(f'Final: {final_data["mean_R"]:.4f}',
                xy=(final_data['epoch'], final_data['mean_R']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color='blue')
    ax1.annotate(f'Final: {final_data["mean_R_train"]:.4f}',
                xy=(final_data['epoch'], final_data['mean_R_train']),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color='red')

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

    # 保存高质量图像
    output_file = 'PhySO_Training_Convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"训练收敛图已保存: {output_file}")

    # 复制到论文目录
    figures_dir = Path("改写论文/figures/v2_优化优化版/")
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_file = figures_dir / "PhySO_Training_Convergence.png"
    import shutil
    shutil.copy(output_file, target_file)
    print(f"图表已更新到论文目录: {target_file}")

    plt.show()
    return output_file

def run_training():
    """运行main.py训练"""
    print("="*60)
    print("运行PhySO符号回归训练 (R² 目标: 0.784)")
    print("="*60)

    cmd = [
        sys.executable, "main.py",
        "--dataset_name", "investment_decision",
        "--task_type", "regression",
        "--epochs", "15",
        "--seed", "0"
    ]

    print("执行命令:")
    print(" ".join(cmd))
    print("-" * 60)

    try:
        # 清理旧的结果文件
        results_dir = Path("results")
        if results_dir.exists():
            for file in ["demo_curves_data.csv", "demo_curves.png", "demo.log"]:
                file_path = results_dir / file
                if file_path.exists():
                    file_path.unlink()
                    print(f"清理旧文件: {file_path}")

        print("启动训练...")

        # 启动训练进程 (确保使用当前Python环境)
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        if process.returncode == 0:
            print("训练成功完成!")
            if process.stdout:
                print("\n训练输出:")
                print(process.stdout[-500:])  # 显示最后500个字符
        else:
            print("训练过程中出现错误:")
            print(process.stderr)
            return None

    except Exception as e:
        print(f"运行训练时出错: {e}")
        return None

def main():
    """主函数"""
    # 确保在正确的目录
    PROJECT_ROOT = Path(__file__).parent.absolute()
    os.chdir(PROJECT_ROOT)

    # 步骤1: 运行训练
    run_training()

    # 步骤2: 等待并收集训练数据
    print("\n收集训练数据...")
    training_data = wait_for_training_completion(timeout_minutes=20)

    # 步骤3: 创建准确的训练收敛图
    if training_data is not None:
        plot_file = create_final_training_plot(training_data)

        print("\n" + "="*60)
        print("✅ 训练监控完成!")
        print(f"生成的图表: {plot_file}")
        print("论文中的训练收敛图已更新为基于真实数据的版本")
        print("="*60)
    else:
        print("❌ 未能收集到完整的训练数据")

if __name__ == "__main__":
    main()