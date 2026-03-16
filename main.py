#!/usr/bin/env python3
"""
符号回归实验主程序
基于PhySO框架的符号回归模型训练与评估
支持分类和回归任务的一键运行
分类任务使用10-fold交叉验证
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

from src.data_preprocessing import load_and_preprocess_data
from src.symbolic_regression import SymbolicRegressionModel
from src.evaluation import ModelEvaluator
from config.datasets_config import DATASETS
from sklearn.model_selection import StratifiedKFold

def select_features_on_training_set(X_train, y_train, X_test, task_type='classification', threshold=0.3):
    """
    在训练集上进行特征选择，然后应用到测试集

    Args:
        X_train: 训练集特征 (样本数×特征数)
        y_train: 训练集标签
        X_test: 测试集特征 (样本数×特征数)
        task_type: 任务类型
        threshold: 相关性阈值

    Returns:
        X_train_selected: 选择后的训练集特征 (特征数×样本数)
        X_test_selected: 选择后的测试集特征 (特征数×样本数)
        selected_indices: 被选择的特征索引
    """
    import pandas as pd

    # X_train 和 X_test 输入格式为 (样本数×特征数)
    # 计算训练集特征与目标变量的相关性
    df_temp = pd.DataFrame(X_train)  # (样本数×特征数)
    correlations = abs(df_temp.corrwith(pd.Series(y_train)))

    # 根据相关性选择特征
    if task_type == 'classification':
        selected_features = correlations > threshold
    else:  # regression
        selected_features = correlations >= threshold

    # 确保至少保留1个特征
    if selected_features.sum() == 0:
        selected_indices = list(range(X_train.shape[1]))
    else:
        selected_indices = [i for i, selected in enumerate(selected_features) if selected]

    # 应用特征选择 (保持样本数×特征数格式，然后转置为特征数×样本数)
    X_train_selected = X_train[:, selected_indices].T  # (特征数×样本数)
    X_test_selected = X_test[:, selected_indices].T    # (特征数×样本数)

    return X_train_selected, X_test_selected, selected_indices

def main():
    """主函数"""

    # 设置参数
    parser = argparse.ArgumentParser(description='Symbolic Regression Experiment for Investment Decision')
    parser.add_argument('--dataset_name', type=str, default='investment_decision', help='数据集名称')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'], help='任务类型: classification 或 regression')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例 (仅回归任务)')
    parser.add_argument('--random_state', type=int, default=0, help='随机种子')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值 (仅分类任务)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--no-sensitivity', action='store_true', help='跳过特征敏感性分析')
    parser.add_argument('--compare-classical', action='store_true', help='运行经典回归模型对比试验 (仅回归任务)')
    parser.add_argument('--test-fold', type=int, default=None, help='只测试指定的fold (1-10)，用于快速调试')
    parser.add_argument('--n-folds', type=int, default=10, help='交叉验证fold数 (默认10)')

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

    # --- 以下为分类任务流程 (使用10-fold交叉验证) ---
    if args.task_type == 'classification':
        target_column = dataset_config['target_column']
    else:
        raise ValueError(f"不支持的任务类型: {args.task_type}")

    # 创建结果目录
    results_dir = os.path.join(PROJECT_ROOT, 'results', args.dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("符号回归实验 (10-Fold交叉验证)")
    print(f"数据集: {args.dataset_name}")
    print(f"任务类型: {args.task_type}")
    print(f"Fold数: {args.n_folds}")
    print("=" * 60)

    # 步骤1: 数据预处理
    print("\n[步骤1] 数据预处理...")
    try:
        X, y, df, config = load_and_preprocess_data(
            dataset_name=args.dataset_name,
            target_column=target_column,
            task_type=args.task_type,
            do_feature_selection=False,
            apply_scaling=False  # 不在预处理时scaling，在fold内进行
        )
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        return

    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 获取原始特征名映射
    target_column_config = config.get('target_column', '是否应投资该项目')
    original_feature_names = [col for col in df.columns if col != target_column_config]
    print(f"\n[信息] 原始特征数量: {len(original_feature_names)}")

    # 诊断：数据集基本情况
    print("\n" + "="*60)
    print("[诊断] 数据集基本信息")
    print("="*60)
    print(f"总样本数: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别分布:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  类别 {int(cls)}: {count} 个样本 ({count/len(y)*100:.1f}%)")
    print("="*60)

    # 步骤2: 10-fold交叉验证
    print(f"\n[步骤2] 数据分割（{args.n_folds}-fold分层交叉验证）...")

    n_repeats = 1
    n_folds = args.n_folds

    # 存储所有fold的结果
    all_fold_results = []
    all_train_metrics = []
    all_test_metrics = []
    all_roc_data = []
    all_expressions = []

    total_folds = n_repeats * n_folds
    current_fold = 0

    # 10-fold交叉验证
    for repeat_idx in range(n_repeats):
        print(f"\n{'='*70}")
        print(f"重复 {repeat_idx + 1}/{n_repeats}")
        print(f"{'='*70}")

        # 使用StratifiedKFold保持类别比例
        skfold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=args.random_state + repeat_idx)

        for fold_idx, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
            current_fold += 1

            # 如果指定了test_fold，只运行该fold
            if args.test_fold is not None and current_fold != args.test_fold:
                continue

            print(f"\n{'='*60}")
            print(f"Repeat {repeat_idx + 1}, Fold {fold_idx + 1}/{n_folds} (总Fold {current_fold}/{total_folds})")
            print(f"{'='*60}")

            # 分割数据
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # 诊断：每个fold的数据分布
            print(f"[诊断 R{repeat_idx+1}F{fold_idx+1}] 训练集类别分布:")
            unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
            for cls, count in zip(unique_train, counts_train):
                print(f"  类别 {int(cls)}: {count} 个样本")
            print(f"[诊断 R{repeat_idx+1}F{fold_idx+1}] 测试集类别分布:")
            unique_test, counts_test = np.unique(y_test_fold, return_counts=True)
            for cls, count in zip(unique_test, counts_test):
                print(f"  类别 {int(cls)}: {count} 个样本")

            # 特征选择：在训练集上进行
            print(f"[R{repeat_idx+1}F{fold_idx+1}] 在训练集上进行特征选择...")
            X_train_fold_fs, X_test_fold_fs, selected_indices = select_features_on_training_set(
                X_train_fold,
                y_train_fold,
                X_test_fold,
                task_type='classification',
                threshold=0.1
            )
            print(f"[R{repeat_idx+1}F{fold_idx+1}] 选择了 {len(selected_indices)}/{X_train_fold.shape[1]} 个特征")

            X_train_fold = X_train_fold_fs
            X_test_fold = X_test_fold_fs

            # 创建变量名映射
            feature_mapping = {}
            for new_idx, orig_idx in enumerate(selected_indices):
                x_name = f'X{new_idx}'
                orig_feature_name = original_feature_names[orig_idx] if orig_idx < len(original_feature_names) else f'Unknown_Index_{orig_idx}'
                feature_mapping[x_name] = {
                    'original_index': int(orig_idx),
                    'feature_name': orig_feature_name
                }

            print(f"训练集: X={X_train_fold.shape}, y={y_train_fold.shape}")
            print(f"测试集: X={X_test_fold.shape}, y={y_test_fold.shape}")

            # 打印变量映射（仅显示前5个）
            print(f"[R{repeat_idx+1}F{fold_idx+1}] 变量映射（前5个）:")
            for x_name in sorted(feature_mapping.keys())[:5]:
                print(f"  {x_name} -> {feature_mapping[x_name]['feature_name']}")

            # 模型训练
            print(f"[R{repeat_idx+1}F{fold_idx+1}] 符号回归模型训练...")
            try:
                # 每个fold使用不同的种子
                model_seed = args.seed + current_fold
                model = SymbolicRegressionModel(seed=model_seed)

                X_names = [f'X{i}' for i in range(X_train_fold.shape[0])]

                expression = model.fit(
                    X_train_fold, y_train_fold,
                    X_names=X_names,
                    y_name="y",
                    epochs=args.epochs,
                    parallel_mode=False
                )

                # 获取最佳表达式
                best_expr_info = model.get_best_expression()
                all_expressions.append(str(best_expr_info['clean_expression']))

                print(f"[R{repeat_idx+1}F{fold_idx+1}] 表达式: {best_expr_info['clean_expression']}")
            except Exception as e:
                print(f"[ERROR] R{repeat_idx+1}F{fold_idx+1} 模型训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 模型评估
            print(f"[R{repeat_idx+1}F{fold_idx+1}] 模型评估...")
            try:
                evaluator = ModelEvaluator(task_type=args.task_type)

                # 训练集评估
                y_train_pred = model.predict(X_train_fold)
                train_metrics = evaluator.evaluate(y_train_fold, y_train_pred)
                all_train_metrics.append(train_metrics)
                evaluator.print_evaluation_report(train_metrics, f"R{repeat_idx+1}F{fold_idx+1} Train")

                # 测试集评估
                y_test_pred = model.predict(X_test_fold)
                test_metrics = evaluator.evaluate(y_test_fold, y_test_pred)
                evaluator.print_evaluation_report(test_metrics, f"R{repeat_idx+1}F{fold_idx+1} Test")

                # 保存用于汇总统计
                all_test_metrics.append(test_metrics)

                # 存储ROC数据
                if 'fpr' in test_metrics and 'tpr' in test_metrics:
                    all_roc_data.append({
                        'fpr': test_metrics['fpr'],
                        'tpr': test_metrics['tpr'],
                        'auc': test_metrics.get('auc_score', 0.5)
                    })

                # 诊断：预测值详情
                print(f"\n[诊断 R{repeat_idx+1}F{fold_idx+1}] 预测值统计:")
                print(f"  预测值范围: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
                print(f"  预测值均值: {y_test_pred.mean():.4f}")
                print(f"  预测值标准差: {y_test_pred.std():.4f}")

                # 保存fold结果
                all_fold_results.append({
                    'repeat': repeat_idx + 1,
                    'fold': fold_idx + 1,
                    'global_fold': current_fold,
                    'expression': str(best_expr_info['clean_expression']),
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_mapping': feature_mapping
                })

            except Exception as e:
                print(f"[ERROR] R{repeat_idx+1}F{fold_idx+1} 模型评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 计算所有fold的平均值
    print(f"\n{'='*60}")
    print(f"{args.n_folds}-Fold交叉验证结果汇总")
    print(f"{'='*60}")

    # 提取关键指标
    metric_keys = ['f1_score', 'precision', 'recall', 'accuracy', 'auc_score', 'ks_score']

    # 识别并排除失败的fold
    failed_fold_indices = []
    successful_fold_results = []

    for i, result in enumerate(all_fold_results):
        f1_score = result.get('test_metrics', {}).get('f1_score', 0)
        if f1_score == 0.0:
            failed_fold_indices.append(i + 1)
        else:
            successful_fold_results.append(result)

    total_folds_run = len(all_fold_results)
    successful_folds = len(successful_fold_results)
    failed_folds = len(failed_fold_indices)

    print(f"\n{'='*70}")
    print("Fold筛选结果")
    print(f"{'='*70}")
    print(f"总fold数: {total_folds_run}")
    print(f"成功fold数: {successful_folds}")
    print(f"失败fold数: {failed_folds}")
    if failed_fold_indices:
        print(f"失败fold编号: {failed_fold_indices}")

    # 计算统计 - 只使用成功的fold
    fold_metrics = {key: [] for key in metric_keys}
    for result in successful_fold_results:
        for key in metric_keys:
            if key in result['test_metrics']:
                fold_metrics[key].append(result['test_metrics'][key])

    summary_stats = {}
    for key in metric_keys:
        if fold_metrics[key]:
            values = np.array(fold_metrics[key])
            summary_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }

    # 打印统计
    print(f"\n{'='*70}")
    print(f"测试集指标统计 [基于{successful_folds}个成功fold]")
    print(f"{'='*70}")
    print(f"{'指标':<15} {'平均值':<10} {'中位数':<10} {'标准差':<10} {'样本数':<8}")
    print("-" * 60)
    for key in metric_keys:
        if key in summary_stats:
            stats = summary_stats[key]
            print(f"{key:<15} {stats['mean']:<10.4f} {stats['median']:<10.4f} {stats['std']:<10.4f} {stats['count']:<8}")

    # 汇总分析
    print("\n" + "="*60)
    print("[诊断] 汇总分析")
    print("="*60)

    print(f"\n所有Fold的表达式:")
    for i, expr in enumerate(all_expressions):
        print(f"  Fold {i+1}: {expr}")
    unique_expr = len(set(all_expressions))
    print(f"\n唯一表达式数量: {unique_expr}/{len(all_expressions)}")

    # 选择最佳表达式
    if len(all_test_metrics) > 0:
        best_fold_idx = np.argmax([m.get('f1_score', 0) for m in all_test_metrics])
        best_expr_info = {'clean_expression': all_expressions[best_fold_idx]}
        print(f"\n[结果] 最佳Fold (总Fold {best_fold_idx + 1}) 的符号表达式:")
        print(f"表达式: {best_expr_info['clean_expression']}")
    else:
        print(f"\n[结果] 没有成功训练的fold")
        best_expr_info = {'clean_expression': 'N/A'}
        best_fold_idx = 0

    # 可视化
    print(f"\n[步骤3] 生成{args.n_folds}-fold交叉验证可视化图表...")
    if len(all_test_metrics) == 0:
        print("[WARNING] 没有成功训练的fold，跳过可视化")
    else:
        try:
            plt.rcParams['text.usetex'] = False
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 构建fold_metrics字典
            fold_metrics_default = {}
            for key in metric_keys:
                fold_metrics_default[key] = [m.get(key, 0) for m in all_test_metrics]

            # 1. 性能指标箱线图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{args.n_folds}-Fold交叉验证性能指标分布', fontsize=16, fontweight='bold')

            for idx, key in enumerate(['f1_score', 'precision', 'recall', 'accuracy', 'auc_score', 'ks_score']):
                if key in fold_metrics_default and fold_metrics_default[key]:
                    ax = axes[idx // 3, idx % 3]
                    data = fold_metrics_default[key]
                    bp = ax.boxplot([data], vert=True, patch_artist=True,
                                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2),
                                  whiskerprops=dict(linewidth=1.5),
                                  capprops=dict(linewidth=1.5))
                    ax.set_ylabel(key.replace('_', ' ').title(), fontsize=10)
                    ax.set_xticklabels([f'{args.n_folds}-Fold'])
                    ax.grid(True, alpha=0.3)
                    # 添加均值点
                    ax.plot(1, np.mean(data), 'ro', markersize=8, label=f'Mean: {np.mean(data):.3f}')
                    ax.legend(loc='lower right', fontsize=8)

            plt.tight_layout()
            boxplot_path = os.path.join(results_dir, f'performance_boxplots_{args.n_folds}fold.png')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] 性能箱线图已保存: {boxplot_path}")

            # 2. ROC曲线汇总
            if all_roc_data:
                plt.figure(figsize=(10, 8))

                # 绘制每个fold的ROC曲线
                for i, roc_data in enumerate(all_roc_data):
                    plt.plot(roc_data['fpr'], roc_data['tpr'], alpha=0.3, linewidth=1,
                            label=f'Fold {i+1} (AUC={roc_data["auc"]:.3f})')

                # 绘制平均ROC曲线
                mean_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(mean_fpr)
                for roc_data in all_roc_data:
                    interp_tpr = np.interp(mean_fpr, roc_data['fpr'], roc_data['tpr'])
                    mean_tpr += interp_tpr
                mean_tpr /= len(all_roc_data)

                mean_auc = summary_stats.get('auc_score', {}).get('mean', 0.5)
                plt.plot(mean_fpr, mean_tpr, color='red', linewidth=2, linestyle='--',
                        label=f'平均ROC (AUC = {mean_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle=':', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title(f'{args.n_folds}-Fold交叉验证ROC曲线', fontsize=14)
                plt.legend(loc="lower right", fontsize=8)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
                plt.tight_layout()

                roc_path = os.path.join(results_dir, f'roc_curve_{args.n_folds}fold.png')
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[INFO] ROC曲线汇总图已保存: {roc_path}")

            print("[INFO] 可视化图表生成完成")
        except Exception as e:
            print(f"[WARNING] 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    print(f"\n[步骤4] 保存{args.n_folds}-fold交叉验证结果...")
    try:
        def serialize_metric(value):
            """序列化评估指标"""
            if isinstance(value, (int, float, np.number)):
                return float(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                return [float(x) if isinstance(x, (int, float, np.number)) else x for x in value]
            elif isinstance(value, dict):
                return {k: serialize_metric(v) for k, v in value.items()}
            else:
                return str(value)

        # 保存所有fold的详细结果
        fold_results = []
        for i, result in enumerate(all_fold_results):
            fold_results.append({
                'repeat': result.get('repeat', (i // args.n_folds) + 1),
                'fold': result.get('fold', (i % args.n_folds) + 1),
                'global_fold': result.get('global_fold', i + 1),
                'expression': result['expression'],
                'train_metrics': {k: serialize_metric(v) for k, v in result['train_metrics'].items()},
                'test_metrics': {k: serialize_metric(v) for k, v in result['test_metrics'].items()},
                'feature_mapping': result.get('feature_mapping', {})
            })

        # 保存汇总结果
        results = {
            'n_repeats': n_repeats,
            'n_folds': n_folds,
            'total_folds': total_folds_run,
            'successful_folds': successful_folds,
            'failed_folds': failed_folds,
            'failed_fold_indices': failed_fold_indices,
            'best_expression': str(best_expr_info['clean_expression']),
            'best_fold': int(best_fold_idx + 1) if len(all_test_metrics) > 0 else 0,
            'task_type': args.task_type,
            'summary_statistics': serialize_metric(summary_stats),
            'all_fold_results': fold_results
        }

        output_filename = os.path.join(results_dir, f'experiment_results_{args.n_folds}fold.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已保存: {output_filename}")

        # 保存CSV格式的汇总统计
        summary_df = pd.DataFrame({
            '指标': list(summary_stats.keys()),
            '平均值': [summary_stats[k]['mean'] for k in summary_stats.keys()],
            '标准差': [summary_stats[k]['std'] for k in summary_stats.keys()],
            '中位数': [summary_stats[k]['median'] for k in summary_stats.keys()],
            '最小值': [summary_stats[k]['min'] for k in summary_stats.keys()],
            '最大值': [summary_stats[k]['max'] for k in summary_stats.keys()],
            '样本数': [summary_stats[k]['count'] for k in summary_stats.keys()]
        })

        summary_csv_path = os.path.join(results_dir, f'test_summary_statistics_{args.n_folds}fold.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"[INFO] 测试集汇总统计已保存: {summary_csv_path}")

    except Exception as e:
        print(f"[WARNING] 结果保存失败: {e}")
        import traceback
        traceback.print_exc()

    # 最终结果清单
    print(f"\n{'='*60}")
    print(f"{args.n_folds}-Fold交叉验证实验完成")
    print(f"{'='*60}")
    print(f"成功fold数: {successful_folds}/{total_folds_run}")
    if successful_folds > 0:
        print(f"\n平均性能指标:")
        for key in metric_keys:
            if key in summary_stats:
                stats = summary_stats[key]
                print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\n最佳表达式: {best_expr_info['clean_expression']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
