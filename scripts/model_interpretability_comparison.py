#!/usr/bin/env python3
"""
模型可解释性对比分析脚本

功能：
1. 计算各模型的可解释性指标
2. 生成对比表（归一化到0-1）
3. 生成可视化图表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from typing import Dict, List, Tuple, Any
import ast

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.evaluation import get_regression_metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time


class InterpretabilityMetrics:
    """可解释性指标计算类"""

    def __init__(self):
        self.metrics = {}

    def count_operators_in_expression(self, expr: str) -> int:
        """计算表达式中的操作符数量"""
        if not isinstance(expr, str):
            return 0

        # 统计数学操作符
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs']
        count = 0

        for op in operators:
            if op in ['+', '-', '*']:
                count += expr.count(op)
            elif '**' in expr:
                count += expr.count('**') * 2  # 幂运算算2个操作符
            else:
                count += len(re.findall(r'\b' + op + r'\b', expr))

        return count

    def count_variables_in_expression(self, expr: str) -> int:
        """计算表达式中的变量数量（去重）"""
        if not isinstance(expr, str):
            return 0

        # 匹配X0, X1, ..., X38等变量
        variables = re.findall(r'\bX\d+\b', expr)
        return len(set(variables))

    def get_expression_complexity_score(self, expr: str) -> float:
        """
        计算表达式复杂度分数 (归一化到0-1)

        考虑因素：
        1. 操作符数量
        2. 变量数量
        3. 嵌套深度
        4. 表达式长度
        """
        if not isinstance(expr, str):
            return 0.5  # 默认中等复杂度

        op_count = self.count_operators_in_expression(expr)
        var_count = self.count_variables_in_expression(expr)
        expr_len = len(expr)

        # 计算嵌套深度（通过括号匹配）
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        # 综合复杂度分数 (0-1，值越大越复杂)
        # 使用对数缩放避免极端值
        complexity = (
            np.log1p(op_count) / 10 +  # 操作符贡献
            np.log1p(var_count) / 10 +  # 变量贡献
            np.log1p(max_depth) / 5 +   # 嵌套深度贡献
            np.log1p(expr_len) / 100    # 长度贡献
        )

        return min(complexity, 1.0)

    def get_rf_complexity_metrics(self, rf_model) -> Dict[str, float]:
        """获取随机森林复杂度指标"""
        # 处理Pipeline中的模型
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
        """
        计算随机森林复杂度分数 (归一化到0-1)

        基于树的数量、总叶子数和平均深度
        """
        metrics = self.get_rf_complexity_metrics(rf_model)

        # 归一化假设（基于典型值）
        # n_trees: 1-200
        # total_leaves: 1-5000
        # avg_depth: 1-30

        score = (
            min(metrics['n_trees'] / 200, 1.0) * 0.4 +
            min(metrics['total_leaves'] / 5000, 1.0) * 0.4 +
            min(metrics['avg_depth'] / 30, 1.0) * 0.2
        )

        return score

    def get_nn_param_count(self, nn_model) -> int:
        """获取神经网络参数量"""
        if hasattr(nn_model, 'coefs_'):
            # MLPRegressor
            params = 0
            for i in range(len(nn_model.coefs_)):
                params += nn_model.coefs_[i].size
                if i < len(nn_model.intercepts_):
                    params += nn_model.intercepts_[i].size
            return params
        return 0

    def get_nn_complexity_score(self, nn_model) -> float:
        """
        计算神经网络复杂度分数 (归一化到0-1)

        基于参数量，假设典型范围 100-100000
        """
        param_count = self.get_nn_param_count(nn_model)

        # 对数缩放
        score = min(np.log1p(param_count) / np.log1p(100000), 1.0)
        return score

    def get_global_interpretability_score(self, model_type: str) -> float:
        """
        全局可解释性打分 (0/0.5/1)

        规则：
        - SR (Symbolic Regression): 1.0 (显式全局公式)
        - LR (Linear Regression): 1.0 (显式线性公式)
        - RF (Random Forest): 0.5 (可用特征重要性但无闭式规则)
        - DT (Decision Tree): 0.75 (有规则但可能复杂)
        - GBM (Gradient Boosting): 0.4 (比RF更难解释)
        - NN (Neural Network): 0.0 (无内生全局规则)
        - SVM/KNN: 0.1 (极难解释)
        """
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

        return scoring.get(model_type, 0.0)

    def get_local_explanation_cost(self, model_info: Dict) -> float:
        """
        解释成本 - 生成解释所需的额外工具/步骤成本 (0-1，值越大成本越高)

        新定义（基于额外工具/步骤需求）：
        - Linear/Ridge/Lasso: 0.05（解释无需额外工具）
        - Symbolic Regression (PhySO): 0.05（公式直接解释，无需额外解释器）
        - Decision Tree: 0.20（单棵树路径可解释，成本低）
        - Random Forest: 0.60（需汇总多树重要性/局部路径，成本中高）
        - Gradient Boosting: 0.70（集成模型+非线性，解释需额外汇总，成本较高）
        - KNN: 0.85（解释依赖邻域样本与距离度量，通常需额外说明，成本高）
        - SVM: 0.90（核方法不可直接解释，通常需LIME/SHAP等，成本高）
        - Neural Network: 1.00（通常依赖SHAP/LIME等解释器，成本最高）
        """
        model_type = model_info.get('model_type', '')

        # 固定成本值（基于额外工具/步骤需求）
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

        return scoring.get(model_type, 0.5)

    def get_auditability_score(self, model_type: str) -> float:
        """
        可审计性/可追溯性打分 (0-1)

        评分规则：
        1.0: 完全可审计（显式公式，可直接验证）
        0.75: 高度可审计（有明确规则但可能复杂）
        0.5: 中等可审计（可审计但需要工具）
        0.25: 低可审计性（需要额外工具和专家）
        0.0: 不可审计（黑盒）
        """
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

        return scoring.get(model_type, 0.0)


def load_existing_results():
    """加载已有的模型对比结果"""
    results_dir = os.path.join(PROJECT_ROOT, 'report_result', 'regression')
    json_path = os.path.join(results_dir, 'regression_model_comparison.json')

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def train_models_for_complexity_analysis(X_train, y_train, random_state=0):
    """训练模型以获取复杂度信息"""

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


def load_sr_expression():
    """加载符号回归的表达式"""
    results_dir = os.path.join(PROJECT_ROOT, 'results')

    # 尝试从多个可能的位置加载
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

    return None


def create_interpretability_comparison_table():
    """创建可解释性对比表"""

    print("=" * 80)
    print("开始计算模型可解释性指标...")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    dataset_name = 'investment_decision'
    target_column = '预期利润额（万元）'
    task_type = 'regression'

    X, y, _, _ = load_and_preprocess_data(
        dataset_name=dataset_name,
        target_column=target_column,
        task_type=task_type
    )
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)

    print(f"数据加载完成：{X_train.shape[0]} 训练样本, {X_test.shape[0]} 测试样本")

    # 2. 训练模型获取复杂度信息
    print("\n[步骤2] 训练模型获取复杂度信息...")
    models = train_models_for_complexity_analysis(X_train, y_train, random_state=0)

    # 3. 加载符号回归表达式
    print("\n[步骤3] 加载符号回归表达式...")
    sr_expr = load_sr_expression()
    if sr_expr:
        print(f"符号回归表达式: {sr_expr}")
    else:
        print("警告: 未找到符号回归表达式，使用默认表达式")
        sr_expr = "X4**2/Abs((X1 + X20)**2)"  # 使用论文中的表达式

    # 4. 计算所有指标
    print("\n[步骤4] 计算可解释性指标...")

    calculator = InterpretabilityMetrics()
    results = []

    # 定义模型顺序
    model_order = [
        'Symbolic Regression (PhySO)',
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Decision Tree',
        'Random Forest',
        'Gradient Boosting',
        'K-Nearest Neighbors',
        'Support Vector Machine',
        'Neural Network'
    ]

    for model_name in model_order:
        print(f"\n--- 处理: {model_name} ---")

        model = models.get(model_name, None)

        # 1. Model Size / Complexity
        if 'Symbolic Regression' in model_name:
            complexity = calculator.get_expression_complexity_score(sr_expr)
            detail = f"expr_len={len(sr_expr)}, ops={calculator.count_operators_in_expression(sr_expr)}"
        elif 'Random Forest' in model_name and model:
            rf_metrics = calculator.get_rf_complexity_metrics(model.named_steps['rf'] if hasattr(model, 'named_steps') else model)
            complexity = calculator.get_rf_complexity_score(model.named_steps['rf'] if hasattr(model, 'named_steps') else model)
            detail = f"trees={rf_metrics['n_trees']}, avg_depth={rf_metrics['avg_depth']:.1f}"
        elif 'Gradient Boosting' in model_name and model:
            # 获取实际的GBM模型
            if hasattr(model, 'named_steps') and 'gbm' in model.named_steps:
                gbm_model = model.named_steps['gbm']
            elif hasattr(model, 'estimators_'):
                gbm_model = model
            else:
                complexity = 0.5
                detail = "N/A"

            # GradientBoostingRegressor的estimators_是numpy数组
            estimators = gbm_model.estimators_ if hasattr(gbm_model, 'estimators_') else []
            n_trees = len(estimators)
            if n_trees > 0:
                # 处理numpy数组中的决策树
                avg_depth = np.mean([est[0].get_depth() for est in estimators])
            else:
                avg_depth = 0
            complexity = min(n_trees * avg_depth / 3000, 1.0)
            detail = f"trees={n_trees}, avg_depth={avg_depth:.1f}"
        elif 'Neural Network' in model_name and model:
            # 获取实际的NN模型
            if hasattr(model, 'named_steps') and 'mlp' in model.named_steps:
                nn_model = model.named_steps['mlp']
            elif hasattr(model, 'coefs_'):
                nn_model = model
            else:
                complexity = 0.5
                detail = "N/A"

            param_count = calculator.get_nn_param_count(nn_model)
            complexity = calculator.get_nn_complexity_score(nn_model)
            detail = f"params={param_count}"
        elif 'Decision Tree' in model_name and model:
            # 获取实际的DT模型
            if hasattr(model, 'named_steps') and 'dt' in model.named_steps:
                dt_model = model.named_steps['dt']
            elif hasattr(model, 'get_depth'):
                dt_model = model
            else:
                complexity = 0.5
                detail = "N/A"

            depth = dt_model.get_depth()
            complexity = min(depth / 30, 1.0)
            detail = f"depth={depth}"
        elif 'Linear' in model_name or 'Ridge' in model_name or 'Lasso' in model_name:
            complexity = 0.05  # 线性模型复杂度很低
            detail = f"features={X_train.shape[1]}"
        else:
            complexity = 0.5
            detail = "N/A"

        # 2. Global Interpretability
        global_interp = calculator.get_global_interpretability_score(model_name)

        # 3. Local Explanation Cost
        model_info = {'model_type': model_name, 'model': model, 'expression': sr_expr if 'Symbolic' in model_name else None}
        local_cost = calculator.get_local_explanation_cost(model_info)

        # 4. Auditability / Traceability
        auditability = calculator.get_auditability_score(model_name)

        results.append({
            'Model': model_name,
            'Model_Size_Complexity': complexity,
            'Complexity_Detail': detail,
            'Global_Interpretability': global_interp,
            'Local_Explanation_Cost': local_cost,
            'Auditability_Traceability': auditability
        })

        print(f"  复杂度: {complexity:.3f} ({detail})")
        print(f"  全局可解释性: {global_interp:.2f}")
        print(f"  局部解释成本: {local_cost:.3f}")
        print(f"  可审计性: {auditability:.2f}")

    # 5. 创建DataFrame
    df = pd.DataFrame(results)

    # 6. 计算综合可解释性分数
    # 权重: Global=0.4, Local=0.2, Auditability=0.3, Complexity_penalty=0.1
    df['Overall_Interpretability'] = (
        df['Global_Interpretability'] * 0.4 +
        (1 - df['Local_Explanation_Cost']) * 0.2 +  # 成本越低越好
        df['Auditability_Traceability'] * 0.3 +
        (1 - df['Model_Size_Complexity']) * 0.1  # 复杂度越低越好
    )

    # 7. 添加排名列
    # 全模型排名 - 使用method='dense'使并列分数获得相同排名，下一个排名连续递增
    df['Rank_All_Models'] = df['Overall_Interpretability'].rank(ascending=False, method='dense').astype(int)

    # 定义非线性模型（排除线性模型）
    nonlinear_models = ['Symbolic Regression (PhySO)', 'Decision Tree', 'Random Forest',
                       'Gradient Boosting', 'K-Nearest Neighbors', 'Support Vector Machine',
                       'Neural Network']

    # 非线性模型子集排名
    df_nonlinear = df[df['Model'].isin(nonlinear_models)].copy()
    df_nonlinear['Rank_Nonlinear_Subset'] = df_nonlinear['Overall_Interpretability'].rank(ascending=False, method='dense').astype(int)

    # 将非线性排名合并回原DataFrame（保持为整数类型，用于后续处理）
    df = df.merge(df_nonlinear[['Model', 'Rank_Nonlinear_Subset']], on='Model', how='left')

    # 对于显示，将NaN转换为空字符串
    df['Rank_Nonlinear_Subset_Display'] = df['Rank_Nonlinear_Subset'].fillna('-').astype(str)

    # 8. 排序（按综合可解释性分数降序）
    df = df.sort_values('Overall_Interpretability', ascending=False)

    # 9. 保存结果
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 保存完整表格
    csv_path = os.path.join(results_dir, 'model_interpretability_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[结果] 完整对比表已保存到: {csv_path}")

    # 保存简化表格（不含detail列）
    display_df = df.drop(columns=['Complexity_Detail'])
    display_csv_path = os.path.join(results_dir, 'model_interpretability_comparison_display.csv')
    display_df.to_csv(display_csv_path, index=False, encoding='utf-8-sig')
    print(f"[结果] 展示用对比表已保存到: {display_csv_path}")

    print("\n" + "=" * 80)
    print("模型可解释性对比表 (按综合可解释性分数排序):")
    print("=" * 80)
    print(display_df.to_string(index=False))

    # 9. 生成可视化图表
    print("\n[步骤5] 生成可视化图表...")
    create_interpretability_visualization(df, results_dir)

    return df


def create_interpretability_visualization(df: pd.DataFrame, save_dir: str):
    """创建可解释性对比可视化图表"""

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 准备数据
    models = df['Model'].values
    complexity = df['Model_Size_Complexity'].values
    global_interp = df['Global_Interpretability'].values
    local_cost = df['Local_Explanation_Cost'].values
    auditability = df['Auditability_Traceability'].values
    overall = df['Overall_Interpretability'].values

    # 颜色映射
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))

    # 1. Model Size / Complexity (条形图)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(len(models)), complexity, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([m.replace(' (PhySO)', '') for m in models], fontsize=9)
    ax1.set_xlabel('Complexity Score (0=Simple, 1=Complex)', fontsize=10)
    ax1.set_title('Model Size / Complexity', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, complexity)):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    # 2. Global Interpretability (条形图)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(len(models)), global_interp, color=plt.cm.GnBu(global_interp))
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels([m.replace(' (PhySO)', '') for m in models], fontsize=9)
    ax2.set_xlabel('Interpretability Score (0-1)', fontsize=10)
    ax2.set_title('Global Interpretability', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, global_interp)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    # 3. Local Explanation Cost (条形图)
    ax3 = axes[1, 0]
    bars3 = ax3.barh(range(len(models)), local_cost, color=plt.cm.OrRd(local_cost))
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([m.replace(' (PhySO)', '') for m in models], fontsize=9)
    ax3.set_xlabel('Cost Score (0=Low, 1=High)', fontsize=10)
    ax3.set_title('Local Explanation Cost', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, local_cost)):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    # 4. Auditability / Traceability (条形图)
    ax4 = axes[1, 1]
    bars4 = ax4.barh(range(len(models)), auditability, color=plt.cm.PuBuGn(auditability))
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels([m.replace(' (PhySO)', '') for m in models], fontsize=9)
    ax4.set_xlabel('Auditability Score (0-1)', fontsize=10)
    ax4.set_title('Auditability / Traceability', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars4, auditability)):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(save_dir, 'model_interpretability_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[结果] 可解释性对比图已保存到: {save_path}")
    plt.close()

    # 创建雷达图
    create_radar_chart(df, save_dir)


def create_radar_chart(df: pd.DataFrame, save_dir: str):
    """创建雷达图展示综合可解释性"""

    # 选择前8个模型（避免太拥挤）
    top_models = df.head(8)
    models = top_models['Model'].values
    models_short = [m.replace(' (PhySO)', '').replace(' Regression', '') for m in models]

    # 指标
    metrics = ['Complexity\n(Inverted)', 'Global\nInterpretability',
               'Local Cost\n(Inverted)', 'Auditability']

    # 准备数据
    data = np.array([
        1 - top_models['Model_Size_Complexity'].values,  # 复杂度反转
        top_models['Global_Interpretability'].values,
        1 - top_models['Local_Explanation_Cost'].values,  # 成本反转
        top_models['Auditability_Traceability'].values
    ]).T

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 绘制
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, (model, row, color) in enumerate(zip(models_short, data, colors)):
        values = row.tolist() + [row[0]]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # 设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.title('Model Interpretability Radar Chart\n(Top 8 Models)', fontsize=14, fontweight='bold', pad=20)

    # 保存
    save_path = os.path.join(save_dir, 'model_interpretability_radar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[结果] 雷达图已保存到: {save_path}")
    plt.close()

    # 创建热力图
    create_heatmap_chart(df, save_dir)


def create_heatmap_chart(df: pd.DataFrame, save_dir: str):
    """创建热力图展示所有指标"""

    # 准备数据
    models = df['Model'].values
    models_short = [m.replace(' (PhySO)', '').replace(' Regression', '').replace('Neural', 'NN') for m in models]

    # 选择要展示的指标列
    metrics_data = df[[
        'Model_Size_Complexity',
        'Global_Interpretability',
        'Local_Explanation_Cost',
        'Auditability_Traceability',
        'Overall_Interpretability'
    ]].values.T

    # 指标名称
    metric_names = [
        'Complexity\n(Higher=Worse)',
        'Global\nInterpretability',
        'Explanation\nCost\n(Higher=Worse)',
        'Auditability\nTraceability',
        'Overall\nInterpretability'
    ]

    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 使用RdYlGn_r颜色映射（红色=差，绿色=好）
    im = ax.imshow(metrics_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(models_short)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.set_xticklabels(models_short, fontsize=9, rotation=45, ha='right')

    # 在每个单元格中添加数值
    for i in range(len(metric_names)):
        for j in range(len(models_short)):
            text = ax.text(j, i, f'{metrics_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score (0=Bad, 1=Good)', rotation=270, labelpad=20, fontsize=10)

    # 添加排名标注
    for j, (model, rank_all, rank_nl_display) in enumerate(zip(models, df['Rank_All_Models'], df['Rank_Nonlinear_Subset_Display'])):
        # 在顶部添加全模型排名
        ax.text(j, -0.6, f'#{int(rank_all)}', ha='center', va='center',
                color='blue', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # 在底部添加非线性排名（如果有）
        if rank_nl_display != '-' and rank_nl_display != '':
            ax.text(j, len(metric_names) - 0.4, f'NL#{rank_nl_display}', ha='center', va='center',
                    color='darkgreen', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))

    ax.set_title('Model Interpretability Heatmap\n(Blue #=All Models Rank, Green #=Nonlinear Subset Rank)',
                 fontsize=14, fontweight='bold', pad=25)

    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, 'model_interpretability_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[结果] 热力图已保存到: {save_path}")
    plt.close()


def create_summary_statistics_table(df: pd.DataFrame):
    """创建汇总统计表"""

    print("\n" + "=" * 80)
    print("可解释性指标汇总统计")
    print("=" * 80)

    summary = pd.DataFrame({
        '指标': ['Model Size / Complexity', 'Global Interpretability',
                'Local Explanation Cost', 'Auditability / Traceability',
                'Overall Interpretability'],
        '最小值': [
            df['Model_Size_Complexity'].min(),
            df['Global_Interpretability'].min(),
            df['Local_Explanation_Cost'].min(),
            df['Auditability_Traceability'].min(),
            df['Overall_Interpretability'].min()
        ],
        '最大值': [
            df['Model_Size_Complexity'].max(),
            df['Global_Interpretability'].max(),
            df['Local_Explanation_Cost'].max(),
            df['Auditability_Traceability'].max(),
            df['Overall_Interpretability'].max()
        ],
        '平均值': [
            df['Model_Size_Complexity'].mean(),
            df['Global_Interpretability'].mean(),
            df['Local_Explanation_Cost'].mean(),
            df['Auditability_Traceability'].mean(),
            df['Overall_Interpretability'].mean()
        ],
        '标准差': [
            df['Model_Size_Complexity'].std(),
            df['Global_Interpretability'].std(),
            df['Local_Explanation_Cost'].std(),
            df['Auditability_Traceability'].std(),
            df['Overall_Interpretability'].std()
        ]
    })

    print(summary.to_string(index=False))

    # 保存
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    summary_path = os.path.join(results_dir, 'interpretability_summary_statistics.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n[结果] 汇总统计已保存到: {summary_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("模型可解释性对比分析工具")
    print("=" * 80)

    # 创建对比表
    df = create_interpretability_comparison_table()

    # 创建汇总统计
    create_summary_statistics_table(df)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. model_interpretability_comparison.csv - 完整对比表")
    print("  2. model_interpretability_comparison_display.csv - 展示用对比表")
    print("  3. model_interpretability_comparison.png - 对比图表")
    print("  4. model_interpretability_radar_chart.png - 雷达图")
    print("  5. interpretability_summary_statistics.csv - 汇总统计")


if __name__ == "__main__":
    main()
