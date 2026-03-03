#!/usr/bin/env python3
"""
经典回归模型对比试验脚本

此脚本实现了多种经典回归模型与符号回归模型的对比试验。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple, Any

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.symbolic_regression import SymbolicRegressionModel
from src.evaluation import get_regression_metrics

# 经典机器学习模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

class ClassicalRegressionModels:
    """经典回归模型集合类"""

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.model_names = list(self.models.keys())

    def _initialize_models(self) -> Dict[str, Any]:
        """初始化所有经典回归模型"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=1.0, random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'Support Vector Machine': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ]),
            'K-Nearest Neighbors': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor(n_neighbors=5))
            ]),
            'Neural Network': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=self.random_state,
                    early_stopping=True
                ))
            ])
        }
        return models

    def train_and_predict(self, model_name: str, X_train: np.ndarray,
                         X_test: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        训练指定模型并进行预测

        Returns:
            Tuple[predictions, training_time]
        """
        model = self.models[model_name]

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        predictions = model.predict(X_test)
        return predictions, training_time

    def get_model_params(self, model_name: str) -> Dict:
        """获取模型参数信息"""
        model = self.models[model_name]
        if hasattr(model, 'get_params'):
            return model.get_params()
        return {}

def run_comprehensive_regression_comparison(
    epochs=15,
    test_size=0.2,
    random_state=0,
    include_symbolic=True
):
    """
    运行全面的回归模型对比试验

    Args:
        epochs: 符号回归训练轮数
        test_size: 测试集比例
        random_state: 随机种子
        include_symbolic: 是否包含符号回归模型
    """
    print("=" * 80)
    print("开始综合回归模型对比试验...")
    print("=" * 80)

    # --- 1. 数据加载和准备 ---
    print("\n[步骤1] 加载和准备数据...")
    dataset_name = 'investment_decision'
    target_column = '预期利润额（万元）'
    task_type = 'regression'

    X, y, _, _ = load_and_preprocess_data(
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type
    )
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    print(f"数据准备完成。训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"目标变量: {target_column}")

    # --- 2. 初始化模型 ---
    print("\n[步骤2] 初始化回归模型...")
    classical_models = ClassicalRegressionModels(random_state=random_state)
    all_model_names = classical_models.model_names.copy()

    if include_symbolic:
        all_model_names.append('Symbolic Regression (PhySO)')

    print(f"参与对比的模型数量: {len(all_model_names)}")
    print("模型列表:")
    for i, name in enumerate(all_model_names, 1):
        print(f"  {i}. {name}")

    # --- 3. 模型训练和评估 ---
    print("\n[步骤3] 模型训练和评估...")

    results = {}

    # 经典模型训练和评估
    print("\n--- 经典回归模型训练 ---")
    for model_name in classical_models.model_names:
        print(f"\n正在训练和评估: {model_name}")
        try:
            # 训练和预测
            predictions, training_time = classical_models.train_and_predict(
                model_name, X_train, X_test, y_train
            )

            # 计算评估指标
            metrics = get_regression_metrics(y_test, predictions)

            # 记录结果
            results[model_name] = {
                'predictions': predictions.tolist(),
                'metrics': metrics,
                'training_time': training_time,
                'model_params': classical_models.get_model_params(model_name)
            }

            print(f"  训练时间: {training_time:.3f}秒")
            print(f"  R2: {metrics['R-squared']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")

        except Exception as e:
            print(f"  [错误] {model_name} 训练失败: {e}")
            continue

    # 符号回归模型训练和评估
    if include_symbolic:
        print(f"\n--- 符号回归模型训练 ---")
        print("正在训练和评估: Symbolic Regression (PhySO)")
        try:
            start_time = time.time()

            # 转置数据格式以适应PhySO要求
            X_train_transposed = X_train.T
            X_test_transposed = X_test.T

            # 训练符号回归模型
            sr_model = SymbolicRegressionModel(seed=random_state)
            sr_model.fit(
                X_train_transposed, y_train,
                X_names=[f'X{i}' for i in range(X_train_transposed.shape[0])],
                y_name="Profit",
                epochs=epochs
            )

            training_time = time.time() - start_time

            # 获取最佳表达式
            best_expr_info = sr_model.get_best_expression()
            print(f"  发现的最佳公式: {best_expr_info['clean_expression']}")

            # 预测（原始结果）
            y_train_sr_pred = sr_model.predict(X_train_transposed)
            y_test_sr_pred = sr_model.predict(X_test_transposed)

            print("\n--- 符号回归线性校准 ---")
            # 训练线性校准器
            calibrator = LinearRegression()
            calibrator.fit(y_train_sr_pred.reshape(-1, 1), y_train)
            coef = calibrator.coef_[0]
            intercept = calibrator.intercept_
            print(f"  校准模型: y_calibrated = {coef:.4f} * y_sr + {intercept:.4f}")

            # 应用校准
            predictions = calibrator.predict(y_test_sr_pred.reshape(-1, 1))

            # 计算评估指标（校准后）
            calibrated_metrics = get_regression_metrics(y_test, predictions)

            # 也记录校准前的指标用于对比
            uncalibrated_metrics = get_regression_metrics(y_test, y_test_sr_pred)

            # 记录结果
            results['Symbolic Regression (PhySO)'] = {
                'predictions': predictions.tolist(),
                'metrics': calibrated_metrics,
                'uncalibrated_metrics': uncalibrated_metrics,
                'training_time': training_time,
                'calibration_model': {'coef': coef, 'intercept': intercept},
                'best_expression': best_expr_info['clean_expression'],
                'sympy_expression': str(best_expr_info['sympy_expression'])
            }

            print(f"  训练时间: {training_time:.3f}秒")
            print(f"  校准前 R2: {uncalibrated_metrics['R-squared']:.4f}")
            print(f"  校准前 RMSE: {uncalibrated_metrics['RMSE']:.4f}")
            print(f"  校准前 MAE: {uncalibrated_metrics['MAE']:.4f}")
            print(f"  校准后 R2: {calibrated_metrics['R-squared']:.4f}")
            print(f"  校准后 RMSE: {calibrated_metrics['RMSE']:.4f}")
            print(f"  校准后 MAE: {calibrated_metrics['MAE']:.4f}")

        except Exception as e:
            print(f"  [错误] 符号回归训练失败: {e}")

    # --- 4. 结果分析和排序 ---
    print("\n[步骤4] 模型性能对比分析...")

    # 按R2分数排序
    model_comparison = []
    for model_name, result in results.items():
        if 'metrics' in result and 'R-squared' in result['metrics']:
            model_comparison.append({
                'model': model_name,
                'r2': result['metrics']['R-squared'],
                'rmse': result['metrics']['RMSE'],
                'mae': result['metrics']['MAE'],
                'training_time': result.get('training_time', 0)
            })

    # 按R2降序排序
    model_comparison.sort(key=lambda x: x['r2'], reverse=True)

    print("\n--- 模型性能排名 (按R2分数) ---")
    for i, model_info in enumerate(model_comparison, 1):
        print(f"  {i}. {model_info['model']}")
        print(f"     R2: {model_info['r2']:.4f}")
        print(f"     RMSE: {model_info['rmse']:.4f}")
        print(f"     MAE: {model_info['mae']:.4f}")
        print(f"     训练时间: {model_info['training_time']:.3f}秒")
        print()

    # --- 5. 可视化结果 ---
    print("[步骤5] 生成对比图表...")

    # 确保results目录存在
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 生成性能对比图
    _create_performance_comparison_plot(model_comparison, results_dir)

    # 生成预测结果对比图
    _create_prediction_comparison_plot(results, y_test, results_dir)

    # --- 6. 保存详细结果 ---
    print("[步骤6] 保存对比试验结果...")

    # 准备保存的结果
    final_results = {
        'experiment_info': {
            'dataset_name': dataset_name,
            'target_column': target_column,
            'test_size': test_size,
            'random_state': random_state,
            'epochs': epochs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'model_ranking': model_comparison,
        'detailed_results': results
    }

    # 保存JSON结果
    save_path = os.path.join(results_dir, 'regression_model_comparison.json')

    def default_serializer(o):
        if isinstance(o, (np.number, np.bool_)):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4, default=default_serializer)

    print(f"对比试验结果已保存到: {save_path}")

    print("\n" + "=" * 80)
    print("综合回归模型对比试验完成！")
    print("=" * 80)

    return final_results

def _create_performance_comparison_plot(model_comparison: List[Dict], results_dir: str):
    """创建性能对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    models = [item['model'] for item in model_comparison]
    r2_scores = [item['r2'] for item in model_comparison]
    rmse_scores = [item['rmse'] for item in model_comparison]
    mae_scores = [item['mae'] for item in model_comparison]
    training_times = [item['training_time'] for item in model_comparison]

    # R2分数对比
    bars1 = ax1.barh(models, r2_scores, color='skyblue')
    ax1.set_xlabel('R2 Score')
    ax1.set_title('R2 Score Comparison')
    ax1.set_xlim(-1, 1)
    ax1.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}',
                va='center', fontsize=9)

    # RMSE对比
    bars2 = ax2.barh(models, rmse_scores, color='lightcoral')
    ax2.set_xlabel('RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
        ax2.text(score + max(rmse_scores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)

    # MAE对比
    bars3 = ax3.barh(models, mae_scores, color='lightgreen')
    ax3.set_xlabel('MAE')
    ax3.set_title('MAE Comparison')
    ax3.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars3, mae_scores)):
        ax3.text(score + max(mae_scores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)

    # 训练时间对比 (对数坐标)
    bars4 = ax4.barh(models, training_times, color='gold')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_title('Training Time Comparison (log scale)')
    ax4.set_xscale('log')
    ax4.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, (bar, time_val) in enumerate(zip(bars4, training_times)):
        ax4.text(time_val * 1.1, bar.get_y() + bar.get_height()/2, f'{time_val:.3f}s',
                va='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'regression_models_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能对比图已保存到: {save_path}")

def _create_prediction_comparison_plot(results: Dict, y_true: np.ndarray, results_dir: str):
    """创建预测结果对比图"""
    n_models = len(results)
    if n_models == 0:
        return

    # 计算子图布局
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    y_true_array = np.array(y_true)

    for i, (model_name, result) in enumerate(results.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        y_pred = np.array(result['predictions'])

        # 散点图
        ax.scatter(y_pred, y_true_array, alpha=0.6, s=30)

        # 完美预测线
        min_val = min(np.min(y_pred), np.min(y_true_array)) * 0.9
        max_val = max(np.max(y_pred), np.max(y_true_array)) * 1.1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # 计算R2
        r2 = result['metrics']['R-squared']

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_title(f'{model_name}\nR2 = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # 隐藏多余的子图
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'regression_predictions_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测对比图已保存到: {save_path}")

def main():
    """主函数，用于直接运行此脚本"""
    run_comprehensive_regression_comparison(epochs=15)

if __name__ == "__main__":
    main()