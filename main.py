#!/usr/bin/env python3
"""
符号回归实验主程序
基于PhySO框架的符号回归模型训练与评估
支持分类和回归任务的一键运行
"""

import os
import sys
import pathlib
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd

# 统一工作目录设置
PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.symbolic_regression import SymbolicRegressionModel
from src.evaluation import ModelEvaluator
from config.datasets_config import DATASETS # 导入DATASETS配置

def main():
    """主函数"""

    # 设置参数
    parser = argparse.ArgumentParser(description='Symbolic Regression Experiment for Investment Decision')
    parser.add_argument('--dataset_name', type=str, default='investment_decision', help='数据集名称')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'], help='任务类型: classification 或 regression')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=0, help='随机种子')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值 (仅分类任务)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--no-sensitivity', action='store_true', help='跳过特征敏感性分析')
    parser.add_argument('--compare-classical', action='store_true', help='运行经典回归模型对比试验 (仅回归任务)')

    args = parser.parse_args()

    # 根据任务类型动态设置目标列
    dataset_config = DATASETS.get(args.dataset_name)
    if not dataset_config:
        raise ValueError(f"未找到数据集配置: {args.dataset_name}")

    # 如果是回归任务，根据参数选择运行标准实验或对比实验
    if args.task_type == 'regression':
        if args.compare_classical:
            print("运行经典回归模型对比试验...")
            from scripts.classical_regression_comparison import run_comprehensive_regression_comparison
            run_comprehensive_regression_comparison(
                epochs=args.epochs,
                test_size=args.test_size,
                random_state=args.random_state,
                include_symbolic=True
            )
        else:
            print("运行标准符号回归实验...")
            from scripts.regression_experiment import run_calibrated_regression_experiment
            run_calibrated_regression_experiment(
                epochs=args.epochs,
                test_size=args.test_size,
                random_state=args.random_state
            )
        # 回归任务已由专属脚本处理完毕，直接退出
        return

    # --- 以下为分类任务或其他通用任务的流程 ---
    if args.task_type == 'classification':
        target_column = dataset_config['target_column'] # 默认分类目标
    else:
        raise ValueError(f"不支持的任务类型: {args.task_type}")

    # 创建结果目录 (现在直接使用根results目录)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"符号回归实验: {args.dataset_name} - 任务类型: {args.task_type}")
    print("=" * 60)

    # 步骤1: 数据预处理
    print("\n[步骤1] 数据预处理...")
    X, y, df_original, config = load_and_preprocess_data(
        dataset_name=args.dataset_name,
        target_column=target_column,
        task_type=args.task_type
    )

    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 步骤2: 数据分割
    print("\n[步骤2] 数据分割...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 转换为模型需要的格式
    X_train = X_train.T
    X_test = X_test.T

    # 确保y是浮点数类型，physo要求
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # 调试打印：检查y的分布
    print(f"\n[调试信息] 训练集y的唯一值和计数: {np.unique(y_train, return_counts=True)}")
    print(f"[调试信息] 测试集y的唯一值和计数: {np.unique(y_test, return_counts=True)}\n")

    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")

    # 步骤3: 模型训练
    print("\n[步骤3] 符号回归模型训练...")
    model = SymbolicRegressionModel(seed=args.seed)

    X_names = [f'X{i}' for i in range(X_train.shape[0])]

    model.fit(
        X_train, y_train,
        X_names=X_names,
        y_name=target_column.replace('（万元）', '').replace('是否应投资该项目', 'Decision'), # 简化y_name
        epochs=args.epochs,
        parallel_mode=False
    )

    # 获取最佳表达式
    best_expr_info = model.get_best_expression()

    print("\n[结果] 最佳符号表达式:")
    print(f"原始表达式: {best_expr_info['expression']}")
    print(f"简化表达式: {best_expr_info['clean_expression']}")

    # 步骤4: 模型评估
    print("\n[步骤4] 模型评估...")
    evaluator = ModelEvaluator(task_type=args.task_type, threshold=args.threshold)

    # 训练集评估
    y_train_pred = model.predict(X_train)
    # 调试打印：检查预测结果的分布
    print(f"\n[调试信息] 训练集预测y的唯一值和计数: {np.unique(y_train_pred, return_counts=True)}")
    train_metrics = evaluator.evaluate(y_train, y_train_pred)
    evaluator.print_evaluation_report(train_metrics, "Training")

    # 测试集评估
    y_test_pred = model.predict(X_test)
    # 调试打印：检查预测结果的分布
    print(f"[调试信息] 测试集预测y的唯一值和计数: {np.unique(y_test_pred, return_counts=True)}\n")
    test_metrics = evaluator.evaluate(y_test, y_test_pred)
    evaluator.print_evaluation_report(test_metrics, "Test")

    # 初始化文件列表
    generated_files = []

    # 保存训练数据（用于后续可视化，PhySO模型无法直接序列化）
    import pickle
    # 只保存可序列化的表达式信息，避免pickle包含不可序列化的内部对象
    serializable_expr_info = {
        'sympy_expression': best_expr_info['sympy_expression'],
        'clean_expression': best_expr_info['clean_expression']
    }
    model_data = {
        'best_expr_info': serializable_expr_info,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_names': X_names,
        'y_train_pred': y_train_pred,  # 保存预测结果
        'y_test_pred': y_test_pred,
        'target_column': target_column,
        'task_type': args.task_type,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

    # 根据任务类型使用不同的文件名
    if args.task_type == 'classification':
        model_save_path = os.path.join(results_dir, 'classification_data.pkl')
        print(f"[INFO] 分类训练数据已保存到: {model_save_path}")
    elif args.task_type == 'regression':
        model_save_path = os.path.join(results_dir, 'regression_data.pkl')
        print(f"[INFO] 回归训练数据已保存到: {model_save_path}")

        # 另外保存为CSV供plot_improved_regression_charts.py使用
        csv_save_path = os.path.join(results_dir, 'physo_predictions.csv')
        # 合并训练集和测试集的预测结果
        all_y_true = np.concatenate([y_train, y_test])
        all_y_pred = np.concatenate([y_train_pred, y_test_pred])
        residuals = all_y_true - all_y_pred
        percentage_error = (residuals / all_y_true) * 100

        pred_df = pd.DataFrame({
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'residual': residuals,
            'abs_error': np.abs(residuals),
            'percentage_error': percentage_error
        })
        pred_df.to_csv(csv_save_path, index=False)
        print(f"[INFO] 回归预测数据已保存到: {csv_save_path}")
        generated_files.append(csv_save_path)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    generated_files.append(model_save_path)

    # 步骤5: 可视化
    print("\n[步骤5] 生成可视化图表...")
    # generated_files 已经在保存模型时初始化

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    if args.task_type == 'classification':
        # ROC曲线
        if 'fpr' in test_metrics and 'tpr' in test_metrics:
            fig_path = os.path.join(results_dir, 'roc_curve.png')
            plt.figure(figsize=(10, 8))
            plt.plot(test_metrics['fpr'], test_metrics['tpr'], color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {test_metrics["auc_score"]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            generated_files.append(fig_path)

        # KS曲线
        if 'fpr' in test_metrics and 'tpr' in test_metrics:
            fig_path = os.path.join(results_dir, 'ks_curve.png')
            plt.figure(figsize=(10, 8))
            plt.plot(test_metrics['tpr'], label='TPR')
            plt.plot(test_metrics['fpr'], label='FPR')
            plt.title(f'KS Curve (KS Statistic = {test_metrics["ks_score"]:.2f})')
            plt.legend()
            plt.grid(True)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            generated_files.append(fig_path)
    elif args.task_type == 'regression':
        # 回归任务的可视化可以根据需要添加，例如预测值 vs 真实值散点图
        print("  (回归任务暂无特定可视化图表生成)")

    if not args.no_sensitivity:
        try:
            print("\n[步骤5a] 特征敏感性分析...")
            # 确保X_train是DataFrame，以便plot_feature_sensitivity使用列名
            X_train_df = pd.DataFrame(X_train.T, columns=X_names)
            evaluator.plot_feature_sensitivity(model, X_train_df, X_names, results_dir)
            generated_files.append(os.path.join(results_dir, 'sensitivity_X2.png')) # 假设生成这个文件
            generated_files.append(os.path.join(results_dir, 'sensitivity_X4.png')) # 假设生成这个文件
        except Exception as e:
            print(f"[警告] 特征敏感性分析失败: {e}")

    # 步骤6: 保存结果
    print("\n[步骤6] 保存结果...")

    def serialize_metric(value):
        """序列化评估指标，处理不同类型的数据"""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            return [float(x) if isinstance(x, (int, float, np.number)) else x for x in value]
        elif isinstance(value, dict):
            return {k: serialize_metric(v) for k, v in value.items()}
        else:
            return str(value)

    results = {
        'best_expression': str(best_expr_info['clean_expression']),
        'train_metrics': {k: serialize_metric(v) for k, v in train_metrics.items()},
        'test_metrics': {k: serialize_metric(v) for k, v in test_metrics.items()}
    }

    results_file_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    generated_files.append(results_file_path)

    print(f"\n[完成] 实验结果已保存到 {results_dir} 目录")
    print("=" * 60)

    # 步骤7: 结果清单
    print("\n" + "=" * 60)
    print("实验结果清单:")
    print("=" * 60)
    print(f"最佳表达式: {best_expr_info['clean_expression']}")
    print(f"测试集 {args.task_type} 评估指标:")
    for metric, value in test_metrics.items():
        if isinstance(value, (float, int)):
            print(f"  - {metric}: {value:.4f}")
        else:
            print(f"  - {metric}: {value}")
    print("\n生成的文件:")
    for f_path in generated_files:
        print(f"- {f_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
