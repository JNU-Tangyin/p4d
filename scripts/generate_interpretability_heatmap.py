#!/usr/bin/env python3
"""
生成可解释性代理指标热力图和CSV文件

输出：
1. figures/model_interpretability_proxy_heatmap.png - 热力图
2. figures/model_interpretability_proxy_metrics.csv - 原始指标表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess_data, split_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class InterpretabilityMetricsCalculator:
    """可解释性指标计算器"""

    def __init__(self):
        pass

    def count_operators_in_expression(self, expr: str) -> int:
        """计算表达式中的操作符数量"""
        if not isinstance(expr, str):
            return 0
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs']
        count = 0
        for op in operators:
            if '**' in expr:
                count += expr.count('**') * 2
            else:
                count += len(re.findall(r'\b' + op + r'\b', expr) if op not in ['+', '-', '*'] else expr.count(op))
        return count

    def get_expression_complexity_score(self, expr: str) -> float:
        """计算表达式复杂度分数 (0-1)"""
        if not isinstance(expr, str):
            return 0.5

        import re
        op_count = self.count_operators_in_expression(expr)
        variables = len(set(re.findall(r'\bX\d+\b', expr)))
        expr_len = len(expr)

        # 计算嵌套深度
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        complexity = (
            np.log1p(op_count) / 10 +
            np.log1p(variables) / 10 +
            np.log1p(max_depth) / 5 +
            np.log1p(expr_len) / 100
        )
        return min(complexity, 1.0)

    def get_rf_complexity_metrics(self, rf_model) -> dict:
        """获取随机森林复杂度指标"""
        if hasattr(rf_model, 'named_steps') and 'rf' in rf_model.named_steps:
            actual_model = rf_model.named_steps['rf']
        elif hasattr(rf_model, 'estimators_'):
            actual_model = rf_model
        else:
            return {'n_trees': 0, 'total_leaves': 0, 'avg_depth': 0}

        n_trees = len(actual_model.estimators_)
        total_leaves = sum(estimator.get_n_leaves() for estimator in actual_model.estimators_)
        avg_depth = np.mean([estimator.get_depth() for estimator in actual_model.estimators_])

        return {
            'n_trees': n_trees,
            'total_leaves': total_leaves,
            'avg_depth': avg_depth
        }

    def get_rf_complexity_score(self, rf_model) -> float:
        """计算随机森林复杂度分数"""
        metrics = self.get_rf_complexity_metrics(rf_model)
        score = (
            min(metrics['n_trees'] / 200, 1.0) * 0.4 +
            min(metrics['total_leaves'] / 5000, 1.0) * 0.4 +
            min(metrics['avg_depth'] / 30, 1.0) * 0.2
        )
        return score

    def get_nn_param_count(self, nn_model) -> int:
        """获取神经网络参数量"""
        if hasattr(nn_model, 'coefs_'):
            params = 0
            for i in range(len(nn_model.coefs_)):
                params += nn_model.coefs_[i].size
                if i < len(nn_model.intercepts_):
                    params += nn_model.intercepts_[i].size
            return params
        return 0

    def get_nn_complexity_score(self, nn_model) -> float:
        """计算神经网络复杂度分数"""
        param_count = self.get_nn_param_count(nn_model)
        return min(np.log1p(param_count) / np.log1p(100000), 1.0)

    def get_model_size_complexity(self, model, model_name: str, expr: str = None) -> float:
        """获取模型大小/复杂度指标"""
        if 'Symbolic Regression' in model_name and expr:
            return self.get_expression_complexity_score(expr)
        elif 'Random Forest' in model_name and model:
            return self.get_rf_complexity_score(model)
        elif 'Gradient Boosting' in model_name and model:
            # 获取GBM模型
            if hasattr(model, 'named_steps') and 'gbm' in model.named_steps:
                gbm_model = model.named_steps['gbm']
            elif hasattr(model, 'estimators_'):
                gbm_model = model
            else:
                return 0.5

            estimators = gbm_model.estimators_ if hasattr(gbm_model, 'estimators_') else []
            n_trees = len(estimators)
            if n_trees > 0:
                avg_depth = np.mean([est[0].get_depth() for est in estimators])
            else:
                avg_depth = 0
            return min(n_trees * avg_depth / 3000, 1.0)
        elif 'Neural Network' in model_name and model:
            if hasattr(model, 'named_steps') and 'mlp' in model.named_steps:
                nn_model = model.named_steps['mlp']
            elif hasattr(model, 'coefs_'):
                nn_model = model
            else:
                return 0.5
            return self.get_nn_complexity_score(nn_model)
        elif 'Decision Tree' in model_name and model:
            if hasattr(model, 'named_steps') and 'dt' in model.named_steps:
                dt_model = model.named_steps['dt']
            elif hasattr(model, 'get_depth'):
                dt_model = model
            else:
                return 0.5
            depth = dt_model.get_depth()
            return min(depth / 30, 1.0)
        elif 'Linear' in model_name or 'Ridge' in model_name or 'Lasso' in model_name:
            return 0.05
        else:
            return 0.5

    def get_global_interpretability(self, model_name: str) -> float:
        """全局可解释性打分"""
        scoring = {
            'Symbolic Regression (PhySO)': 1.0,
            'Linear Regression': 1.0,
            'Ridge Regression': 1.0,
            'Lasso Regression': 1.0,
            'Decision Tree': 0.75,
            'Random Forest': 0.5,
            'Gradient Boosting': 0.4,
            'K-Nearest Neighbors': 0.1,
            'Support Vector Machine': 0.1,
            'Neural Network': 0.0
        }
        return scoring.get(model_name, 0.0)

    def get_local_explanation_cost(self, model_name: str) -> float:
        """解释成本（基于额外工具/步骤需求）"""
        scoring = {
            'Linear Regression': 0.05,
            'Ridge Regression': 0.05,
            'Lasso Regression': 0.05,
            'Symbolic Regression (PhySO)': 0.05,
            'Decision Tree': 0.20,
            'Random Forest': 0.60,
            'Gradient Boosting': 0.70,
            'K-Nearest Neighbors': 0.85,
            'Support Vector Machine': 0.90,
            'Neural Network': 1.00
        }
        return scoring.get(model_name, 0.5)

    def get_auditability(self, model_name: str) -> float:
        """可审计性/可追溯性打分"""
        scoring = {
            'Symbolic Regression (PhySO)': 1.0,
            'Linear Regression': 1.0,
            'Ridge Regression': 1.0,
            'Lasso Regression': 1.0,
            'Decision Tree': 0.75,
            'Random Forest': 0.4,
            'Gradient Boosting': 0.3,
            'K-Nearest Neighbors': 0.2,
            'Support Vector Machine': 0.1,
            'Neural Network': 0.05
        }
        return scoring.get(model_name, 0.0)


def load_sr_expression():
    """加载符号回归表达式"""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    possible_paths = [
        os.path.join(results_dir, 'regression_experiment_results.json'),
        os.path.join(results_dir, 'simple_regression_comparison', 'regression_comparison_results.json'),
        os.path.join(PROJECT_ROOT, 'report_result', 'regression', 'regression_experiment_results.json')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'best_expression' in data:
                    return data['best_expression']
    return "X4**2/Abs((X1 + X20)**2)"  # 默认表达式


def train_models(X_train, y_train, random_state=0):
    """训练所有模型"""
    models = {}

    # Linear Regression
    models['Linear Regression'] = LinearRegression().fit(X_train, y_train)

    # Ridge Regression
    models['Ridge Regression'] = Ridge(alpha=1.0, random_state=random_state).fit(X_train, y_train)

    # Lasso Regression
    models['Lasso Regression'] = Lasso(alpha=1.0, random_state=random_state).fit(X_train, y_train)

    # Decision Tree
    models['Decision Tree'] = DecisionTreeRegressor(
        max_depth=10, min_samples_split=5, random_state=random_state
    ).fit(X_train, y_train)

    # Random Forest
    models['Random Forest'] = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=random_state
    ).fit(X_train, y_train)

    # Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state
    ).fit(X_train, y_train)

    # SVM
    models['Support Vector Machine'] = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
    ]).fit(X_train, y_train)

    # KNN
    models['K-Nearest Neighbors'] = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=5))
    ]).fit(X_train, y_train)

    # Neural Network
    models['Neural Network'] = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=random_state,
            early_stopping=True
        ))
    ]).fit(X_train, y_train)

    return models


def calculate_metrics(models, sr_expr, X_train):
    """计算所有模型的可解释性指标"""
    calculator = InterpretabilityMetricsCalculator()

    results = []

    for model_name, model in models.items():
        # 计算各项指标
        complexity = calculator.get_model_size_complexity(model, model_name, sr_expr if 'Symbolic' in model_name else None)
        global_interp = calculator.get_global_interpretability(model_name)
        local_cost = calculator.get_local_explanation_cost(model_name)
        auditability = calculator.get_auditability(model_name)

        # 综合分数
        overall = (
            global_interp * 0.4 +
            (1 - local_cost) * 0.2 +
            auditability * 0.3 +
            (1 - complexity) * 0.1
        )

        results.append({
            'Model': model_name,
            'Model_Size_Complexity': complexity,
            'Global_Interpretability': global_interp,
            'Local_Explanation_Cost': local_cost,
            'Auditability_Traceability': auditability,
            'Overall_Interpretability': overall
        })

    return pd.DataFrame(results)


def main():
    """主函数"""
    print("=" * 80)
    print("生成可解释性代理指标热力图和CSV文件")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    X, y, _, _ = load_and_preprocess_data(
        dataset_name='investment_decision',
        target_column='预期利润额（万元）',
        task_type='regression'
    )
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
    print(f"数据加载完成：{X_train.shape[0]} 训练样本")

    # 2. 训练模型
    print("\n[步骤2] 训练模型...")
    models = train_models(X_train, y_train, random_state=0)

    # 3. 加载符号回归表达式
    print("\n[步骤3] 加载符号回归表达式...")
    sr_expr = load_sr_expression()
    print(f"符号回归表达式: {sr_expr}")

    # 添加符号回归到结果中
    calculator = InterpretabilityMetricsCalculator()

    # 4. 计算指标
    print("\n[步骤4] 计算可解释性指标...")
    df = calculate_metrics(models, sr_expr, X_train)

    # 添加符号回归
    sr_complexity = calculator.get_expression_complexity_score(sr_expr)
    sr_metrics = {
        'Model': 'Symbolic Regression (PhySO)',
        'Model_Size_Complexity': sr_complexity,
        'Global_Interpretability': 1.0,
        'Local_Explanation_Cost': 0.05,
        'Auditability_Traceability': 1.0,
        'Overall_Interpretability': (
            1.0 * 0.4 + (1 - 0.05) * 0.2 + 1.0 * 0.3 + (1 - sr_complexity) * 0.1
        )
    }
    df = pd.concat([pd.DataFrame([sr_metrics]), df], ignore_index=True)
    df = df.sort_values('Overall_Interpretability', ascending=False)

    # 5. 创建figures目录
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 6. 保存CSV
    csv_path = os.path.join(figures_dir, 'model_interpretability_proxy_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[结果] CSV文件已保存到: {csv_path}")

    # 7. 生成热力图
    print("\n[步骤5] 生成热力图...")

    # 准备数据
    models_short = [m.replace(' (PhySO)', '').replace(' Regression', '').replace('Neural', 'NN')
                    for m in df['Model'].values]

    metrics_data = df[[
        'Model_Size_Complexity',
        'Global_Interpretability',
        'Local_Explanation_Cost',
        'Auditability_Traceability',
        'Overall_Interpretability'
    ]].values.T

    metric_names = [
        'Model Size\n/ Complexity\n(Higher=Worse)',
        'Global\nInterpretability',
        'Explanation\nCost\n(Higher=Worse)',
        'Auditability\n& Traceability',
        'Overall\nInterpretability'
    ]

    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 9))

    # 使用RdYlGn_r颜色映射（红色=差，绿色=好）
    im = ax.imshow(metrics_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(models_short)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=12)
    ax.set_xticklabels(models_short, fontsize=10, rotation=45, ha='right')

    # 在每个单元格中添加数值
    for i in range(len(metric_names)):
        for j in range(len(models_short)):
            value = metrics_data[i, j]

            # 根据值选择文本颜色
            if value > 0.5:
                text_color = 'black'
            else:
                text_color = 'white'

            ax.text(j, i, f'{value:.3f}',
                   ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score (0=Bad, 1=Good)', rotation=270, labelpad=20, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # 设置标题
    ax.set_title('Model Interpretability Proxy Metrics Heatmap\n(Normalized scores: 0=Least Interpretable, 1=Most Interpretable)',
                 fontsize=14, fontweight='bold', pad=20)

    # 添加网格线
    ax.set_xticks(np.arange(len(models_short)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(metric_names)) + 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5, alpha=0.3)

    plt.tight_layout()

    # 保存热力图
    heatmap_path = os.path.join(figures_dir, 'model_interpretability_proxy_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"[结果] 热力图已保存到: {heatmap_path}")
    plt.close()

    # 8. 打印结果表
    print("\n" + "=" * 80)
    print("可解释性代理指标表")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("完成！生成的文件：")
    print(f"  1. {csv_path}")
    print(f"  2. {heatmap_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
