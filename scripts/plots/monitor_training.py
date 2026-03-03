#!/usr/bin/env python3
"""
实时监控main.py训练过程
自动记录PhySO训练的真实数据并生成收敛图
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import threading
import signal
import sys

class TrainingMonitor:
    def __init__(self):
        self.monitoring = True
        self.training_data = []
        self.start_time = None

    def monitor_training_files(self):
        """监控训练文件的变化"""
        results_dir = Path("results")

        print("开始监控PhySO训练过程...")
        print(f"监控目录: {results_dir.absolute()}")

        # 等待训练开始
        print("等待训练开始...")
        time.sleep(5)

        last_curves_size = 0
        last_log_size = 0

        while self.monitoring:
            try:
                # 检查训练曲线数据文件
                curves_file = results_dir / "demo_curves_data.csv"
                if curves_file.exists():
                    current_size = curves_file.stat().st_size

                    # 如果文件大小发生变化，读取新数据
                    if current_size != last_curves_size:
                        try:
                            df = pd.read_csv(curves_file)
                            if len(df) > len(self.training_data):
                                self.training_data = df.copy()
                                print(f"检测到新训练数据: {len(df)} epochs")

                                # 显示最新epoch的数据
                                if len(df) > 0:
                                    latest = df.iloc[-1]
                                    print(f"最新Epoch {latest['epoch']}: "
                                          f"Test R²={latest['mean_R']:.4f}, "
                                          f"Train R²={latest['mean_R_train']:.4f}, "
                                          f"Loss={latest['loss']:.6f}")

                        except Exception as e:
                            print(f"读取训练数据时出错: {e}")

                    last_curves_size = current_size

                # 检查是否训练完成
                if len(self.training_data) >= 15:
                    print("训练完成！")
                    break

                time.sleep(2)  # 每2秒检查一次

            except KeyboardInterrupt:
                print("\n监控被中断")
                break
            except Exception as e:
                print(f"监控过程中出错: {e}")
                time.sleep(2)

        return self.training_data

    def create_training_convergence_plot(self, df):
        """创建训练收敛图"""
        if df is None or len(df) == 0:
            print("没有训练数据，无法创建图表")
            return None

        print(f"\n基于真实训练数据创建收敛图: {len(df)} epochs")

        # 显示最终统计
        final_data = df.iloc[-1]
        print(f"最终性能指标:")
        print(f"- 测试集R²: {final_data['mean_R']:.4f}")
        print(f"- 训练集R²: {final_data['mean_R_train']:.4f}")
        print(f"- 损失函数: {final_data['loss']:.6f}")
        print(f"- 表达式复杂度: {final_data['mean_complexity']:.2f}")

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PhySO Training Convergence (Real Data - R² Target: 0.784)', fontsize=16, fontweight='bold')

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
        ax1.axhline(y=final_data['mean_R'], color='b', linestyle='--', alpha=0.5)
        ax1.axhline(y=final_data['mean_R_train'], color='r', linestyle='--', alpha=0.5)

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

        # 保存图表
        output_file = 'PhySO_Training_Convergence_Monitored.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"训练收敛图已保存: {output_file}")

        # 复制到figures目录
        figures_dir = "改写论文/figures/v2_优化优化版/"
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        import shutil
        target_file = figures_dir + "PhySO_Training_Convergence.png"
        shutil.copy(output_file, target_file)
        print(f"图表已更新到论文目录: {target_file}")

        plt.show()
        return output_file

def run_main_training():
    """运行main.py训练"""
    print("="*60)
    print("启动PhySO训练监控")
    print("="*60)

    # 构建训练命令
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
        # 启动训练进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1)

        # 创建监控器
        monitor = TrainingMonitor()

        # 在后台启动监控
        def monitor_thread():
            monitor.monitor_training_files()

        monitor_process = threading.Thread(target=monitor_thread, daemon=True)
        monitor_process.start()

        # 等待训练完成
        process.wait()
        monitor.monitoring = False

        print("训练进程结束")

    except KeyboardInterrupt:
        print("\n训练被中断")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"运行训练时出错: {e}")

def main():
    """主函数"""
    # 确保在正确的目录
    import pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
    os.chdir(PROJECT_ROOT)

    # 运行训练
    run_main_training()

    # 等待一下确保所有数据都写入文件
    time.sleep(2)

    # 创建图表
    monitor = TrainingMonitor()

    # 读取最新的训练数据
    try:
        curves_file = Path("results/demo_curves_data.csv")
        if curves_file.exists():
            df = pd.read_csv(curves_file)
            print(f"\n读取到训练数据: {len(df)} epochs")
            monitor.create_training_convergence_plot(df)
        else:
            print("未找到训练数据文件")
    except Exception as e:
        print(f"读取训练数据失败: {e}")

    print("\n" + "="*60)
    print("训练监控完成")
    print("="*60)

if __name__ == "__main__":
    main()