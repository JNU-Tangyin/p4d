#!/usr/bin/env python3
"""
模型对比分析脚本

功能：
1. 加载投资决策数据集。
2. 训练和评估多个标准机器学习分类器。
3. 与符号回归模型进行性能对比。
4. 将结果保存为CSV文件。
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径，以便导入自定义模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.symbolic_regression import SymbolicRegressionModel
from src.evaluation import get_classification_metrics # 导入新的函数

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def main():
    """主函数"""
    print("=" * 60)
    print("开始模型对比分析...")
    print("=" * 60)

    # --- 1. 数据加载和准备 ---
    print("\n[步骤1] 加载和准备数据...")
    X, y, _, _ = load_and_preprocess_data(dataset_name='investment_decision')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
    
    # 确保y是浮点数类型以满足PhySO的要求
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    # 标准化数据，因为逻辑回归和SVM对此敏感
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"数据准备完成。训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # --- 2. 定义要评估的模型 ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'SVM': SVC(probability=True, random_state=0), # probability=True for AUC
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }

    results = []

    # --- 3. 训练和评估标准模型 ---
    print("\n[步骤2] 训练和评估标准机器学习模型...")
    for name, model in models.items():
        print(f"--- 正在评估: {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # 预测概率用于评估
        
        metrics = get_classification_metrics(y_test, y_pred_proba)
        metrics['Model'] = name
        results.append(metrics)
        
        print(f"{name} 评估完成。 F1-Score: {metrics.get('F1-Score', 0):.4f}, AUC-ROC: {metrics.get('AUC-ROC', 0):.4f}")

    # --- 4. 单独处理和评估符号回归模型 ---
    print("--- 正在评估: Symbolic Regression ---")
    sr_model = SymbolicRegressionModel(seed=0)
    
    # PhySO需要转置后的数据: (n_features, n_samples)
    X_train_transposed = X_train.T
    X_test_transposed = X_test.T

    sr_model.fit(X_train_transposed, y_train, X_names=[f'X{i}' for i in range(X_train_transposed.shape[0])], epochs=10)
    y_pred_sr_proba = sr_model.predict(X_test_transposed) # PhySO直接输出概率
    
    sr_metrics = get_classification_metrics(y_test, y_pred_sr_proba)
    sr_metrics['Model'] = 'Symbolic Regression'
    results.append(sr_metrics)

    print(f"Symbolic Regression 评估完成。 F1-Score: {sr_metrics.get('F1-Score', 0):.4f}, AUC-ROC: {sr_metrics.get('AUC-ROC', 0):.4f}")

    # --- 5. 结果汇总和保存 ---
    # 调整字典键以匹配DataFrame列名
    df_results = []
    for res in results:
        df_row = {
            'Model': res['Model'],
            'Accuracy': res['Accuracy'],
            'Precision': res['Precision'],
            'Recall': res['Recall'],
            'F1-Score': res['F1-Score'],
            'AUC-ROC': res['AUC-ROC']
            # 混淆矩阵不方便在表格中展示，故省略
        }
        df_results.append(df_row)

    results_df = pd.DataFrame(df_results)
    # 将符号回归模型的结果放在第一行
    sr_row = results_df[results_df['Model'] == 'Symbolic Regression']
    other_rows = results_df[results_df['Model'] != 'Symbolic Regression']
    results_df = pd.concat([sr_row, other_rows], ignore_index=True)

    print("\n[步骤3] 模型性能对比结果:")
    print(results_df.to_string(index=False))

    # 保存到CSV
    save_path = os.path.join(PROJECT_ROOT, 'results', 'model_comparison_table.csv')
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {save_path}")

    print("\n" + "=" * 60)
    print("模型对比分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
