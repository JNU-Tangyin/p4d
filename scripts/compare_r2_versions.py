#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2变化原因排查脚本
对比旧版本 (efd65be4) 和当前版本 (HEAD) 的数据处理差异
"""

import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 固定所有随机种子
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)
import torch
torch.manual_seed(RANDOM_STATE)

print("=" * 70)
print("R2变化原因排查 - 对比测试")
print("=" * 70)

# === 步骤1: 加载原始数据 ===
print("\n[步骤1] 加载原始数据...")
data_path = 'data/项目数据收集表v2.0.xlsx'
df_raw = pd.read_excel(data_path, header=2)
print(f"原始数据形状: {df_raw.shape}")
print(f"原始列名 (前15个): {list(df_raw.columns)[:15]}")

# === 步骤2: 旧版本处理逻辑 (模拟 efd65be4) ===
print("\n[步骤2] 旧版本数据预处理 (efd65be4 逻辑)...")

df_old = df_raw.copy()

# 旧版本: 删除完全为空的行和列
df_old.dropna(how='all', inplace=True)
df_old.dropna(axis=1, how='all', inplace=True)

# 目标列
target_column = '预期利润额（万元）'

# 旧版本: 将所有列转换为数值
for col in df_old.columns:
    df_old[col] = pd.to_numeric(df_old[col], errors='coerce')

# 删除目标列为NaN的行
df_old.dropna(subset=[target_column], inplace=True)

# 分离特征和目标
y_old = df_old[target_column].values
X_old_df = df_old.drop(columns=[target_column])

# 旧版本: 用0填充特征中的剩余NaN值
X_old_df.fillna(0, inplace=True)
X_old = X_old_df.values

print(f"旧版本 - 数据形状: X={X_old.shape}, y={y_old.shape}")
print(f"旧版本 - 特征数: {X_old.shape[1]}")

# 旧版本特征选择 (相关性阈值 0.1)
if X_old.shape[1] > 1:
    correlations_old = abs(pd.DataFrame(X_old).corrwith(pd.Series(y_old)))
    selected_features_old = correlations_old[correlations_old >= 0.1].index.tolist()
    if selected_features_old:
        X_old = X_old[:, selected_features_old]
        print(f"旧版本 - 特征选择后: {len(selected_features_old)}/{correlations_old.size} 个特征")

# 旧版本标准化
from src.data_preprocessing import DecimalScaler
X_old_clipped = np.clip(X_old, -1e2, 1e2)
X_old_shifted = np.where(X_old_clipped <= 0, X_old_clipped + 1e-3, X_old_clipped)
X_old_scaled = DecimalScaler(X_old_shifted)

print(f"旧版本 - 最终形状: X={X_old_scaled.shape}")

# === 步骤3: 新版本处理逻辑 (当前 HEAD) ===
print("\n[步骤3] 新版本数据预处理 (当前 HEAD 逻辑)...")

df_new = df_raw.copy()

# 新版本: 删除所有列都是NaN的行
df_new = df_new.dropna(how='all')

# 新版本: 删除 Unnamed 列
unnamed_cols = [col for col in df_new.columns if col.startswith('Unnamed:')]
if unnamed_cols:
    df_new = df_new.drop(columns=unnamed_cols)
    print(f"新版本 - 删除 {len(unnamed_cols)} 个 Unnamed 列")

# 新版本: 移除非数值列 (不包含目标列)
numeric_cols_new = df_new.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = set(df_new.columns) - set(numeric_cols_new)
if non_numeric_cols:
    print(f"新版本 - 移除非数值列: {non_numeric_cols}")
df_new = df_new[numeric_cols_new]

# 数据类型转换
df_new = df_new.astype(float)

# 分离特征和目标
y_new = df_new[target_column].values
X_new_df = df_new.drop(columns=[target_column])
X_new = X_new_df.values

print(f"新版本 - 数据形状: X={X_new.shape}, y={y_new.shape}")
print(f"新版本 - 特征数: {X_new.shape[1]}")

# 新版本特征选择 (相关性阈值 0.1)
if X_new.shape[1] > 1:
    correlations_new = abs(pd.DataFrame(X_new).corrwith(pd.Series(y_new)))
    selected_features_new = correlations_new[correlations_new >= 0.1]
    if selected_features_new.sum() > 0:
        selected_indices_new = [i for i, selected in enumerate(selected_features_new) if selected]
        X_new = X_new[:, selected_indices_new]
        print(f"新版本 - 特征选择后: {len(selected_indices_new)}/{correlations_new.size} 个特征")

# 新版本标准化
X_new_clipped = np.clip(X_new, -1e2, 1e2)
X_new_shifted = np.where(X_new_clipped <= 0, X_new_clipped + 1e-3, X_new_clipped)
X_new_scaled = DecimalScaler(X_new_shifted)

print(f"新版本 - 最终形状: X={X_new_scaled.shape}")

# === 步骤4: 对比数据切分 ===
print("\n[步骤4] 数据切分对比...")

X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(
    X_old_scaled, y_old, test_size=0.2, random_state=RANDOM_STATE
)

X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
    X_new_scaled, y_new, test_size=0.2, random_state=RANDOM_STATE
)

print(f"旧版本 - 训练集: {X_old_train.shape}, 测试集: {X_old_test.shape}")
print(f"新版本 - 训练集: {X_new_train.shape}, 测试集: {X_new_test.shape}")

# 检查数据是否一致
print(f"\n数据一致性检查:")
print(f"  y 是否相同: {np.allclose(y_old, y_new)}")
print(f"  X_old_scaled 和 X_new_scaled 形状差异: {X_old_scaled.shape} vs {X_new_scaled.shape}")

if X_old_scaled.shape == X_new_scaled.shape:
    print(f"  X 是否相同: {np.allclose(X_old_scaled, X_new_scaled)}")
else:
    print(f"  X 形状不同，无法直接比较")

# === 步骤5: 模拟 PhySO 训练和校准 ===
print("\n[步骤5] 模拟 PhySO 训练和校准...")

# 由于 PhySO 训练时间较长，这里我们使用简化的模拟来展示 R² 变化的原因
# 我们假设 PhySO 表达式在两个版本中相同，但输入特征不同

# 模拟 PhySO 预测 (使用一个简单的线性函数来模拟)
# 在实际情况中，这是 PhySO 训练出来的表达式

# 旧版本
y_old_train_sr_pred = X_old_train.mean(axis=1) * 100 + 50  # 简化模拟
y_old_test_sr_pred = X_old_test.mean(axis=1) * 100 + 50

# 新版本
y_new_train_sr_pred = X_new_train.mean(axis=1) * 100 + 50
y_new_test_sr_pred = X_new_test.mean(axis=1) * 100 + 50

# 线性校准
calibrator_old = LinearRegression()
calibrator_old.fit(y_old_train_sr_pred.reshape(-1, 1), y_old_train)
y_old_test_calibrated_pred = calibrator_old.predict(y_old_test_sr_pred.reshape(-1, 1))
r2_old = r2_score(y_old_test, y_old_test_calibrated_pred)

calibrator_new = LinearRegression()
calibrator_new.fit(y_new_train_sr_pred.reshape(-1, 1), y_new_train)
y_new_test_calibrated_pred = calibrator_new.predict(y_new_test_sr_pred.reshape(-1, 1))
r2_new = r2_score(y_new_test, y_new_test_calibrated_pred)

print(f"\n模拟结果 (简化预测函数):")
print(f"  旧版本 R2: {r2_old:.4f}")
print(f"  新版本 R2: {r2_new:.4f}")
print(f"  差异: {r2_new - r2_old:.4f}")

# === 步骤6: 根本原因分析 ===
print("\n" + "=" * 70)
print("根本原因分析:")
print("=" * 70)

print("\n[关键差异]")
print(f"1. 特征数量变化: {X_old_scaled.shape[1]} -> {X_new_scaled.shape[1]}")
print(f"2. 数据清洗逻辑变化:")
print(f"   - 旧版本: 保留 Unnamed 列，用 0 填充 NaN")
print(f"   - 新版本: 删除 Unnamed 列，删除非数值列")

print("\n[R2变化原因]")
if X_old_scaled.shape[1] != X_new_scaled.shape[1]:
    print(f"WARNING: 特征数量不同导致 PhySO 训练出的表达式不同")
    print(f"WARNING: 即使使用相同的 random_state，不同数量的特征会导致:")
    print(f"    - 不同的特征空间")
    print(f"    - 不同的 PhySO 搜索路径")
    print(f"    - 不同的最终表达式")
    print(f"    - 不同的校准结果")
else:
    print(f"OK 特征数量相同，但特征值可能有细微差异")

print("\n[建议]")
print("1. 如果需要复现 0.784 的结果，需要使用旧版本的数据预处理逻辑")
print("2. 或者重新训练 PhySO 并报告新的 R2 值")
print("3. 建议使用多次运行的平均值 +/- 标准差来报告结果，减少随机性影响")
print("4. 考虑使用交叉验证来获得更稳健的 R2 估计")
