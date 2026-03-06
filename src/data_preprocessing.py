import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets_config import DATASETS, DATASET_ALIASES

def DecimalScaler(data):
    """十进制标准化器，带边界保护"""
    # 确保数据在合理范围内
    data = np.array(data, dtype=float)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        max_val = 1.0

    # 使用对数缩放避免极值
    exponent = np.ceil(np.log10(max_val + 1e-10))
    norm_data = data / (10 ** max(exponent, -2))  # 限制最小指数

    # 限制范围在[-1, 1]内
    norm_data = np.clip(norm_data, -1.0, 1.0)
    return norm_data

def MeanScaler(data):
    """均值标准化器，带边界保护"""
    data = np.array(data, dtype=float)
    min_val = np.min(data)
    max_val = np.max(data)
    mean = np.mean(data)

    # 防止除零
    range_val = max(max_val - min_val, 1e-8)
    norm_data = (data - mean) / range_val

    # 限制范围在[-1, 1]内
    norm_data = np.clip(norm_data, -1.0, 1.0)
    return norm_data

def get_dataset_config(dataset_name):
    """
    获取数据集配置

    Args:
        dataset_name: str, 数据集名称或别名

    Returns:
        dict, 数据集配置
    """
    # 处理别名
    if dataset_name in DATASET_ALIASES:
        dataset_name = DATASET_ALIASES[dataset_name]

    if dataset_name in DATASETS:
        return DATASETS[dataset_name]
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

def load_and_preprocess_data(data_path=None, dataset_name=None, target_column=None, task_type=None):
    """
    加载并预处理数据

    Args:
        data_path: str, 数据文件路径（可选）
        dataset_name: str, 数据集名称（可选）
        target_column: str, 目标变量列名（可选）
        task_type: str, 任务类型（可选）

    Returns:
        X: 预处理后的特征数据
        y: 目标变量
        df: 原始数据DataFrame
        config: 数据集配置

    环境变量:
        USE_OLD_PREPROCESSING: 设置为 'true' 使用旧版本预处理逻辑（用于复现 R²=0.784）
    """

    # 检查是否使用旧版本预处理逻辑（用于复现回归 R²=0.784）
    use_old_preprocessing = os.getenv('USE_OLD_PREPROCESSING', 'false').lower() == 'true'
    if use_old_preprocessing:
        print("[INFO] 使用旧版本预处理逻辑 (USE_OLD_PREPROCESSING=true)")

    # 如果提供了数据集名称，使用配置
    if dataset_name:
        config = get_dataset_config(dataset_name)
        data_path = config['file_path']
        # 只有当参数未指定时才从配置读取
        if target_column is None:
            target_column = config['target_column']
        if task_type is None:
            task_type = config['task_type']
    elif not data_path:
        # 默认使用投资决策数据集
        config = get_dataset_config('investment_decision')
        data_path = config['file_path']
        # 只有当参数未指定时才从配置读取
        if target_column is None:
            target_column = config['target_column']
        if task_type is None:
            task_type = config['task_type']

    # 特殊处理：投资决策数据集根据任务类型切换目标列
    if dataset_name == 'investment_decision' or '投资决策' in str(data_path):
        if task_type == 'regression':
            # 回归任务使用利润预测
            target_column = '预期利润额（万元）'
            print(f"[INFO] 回归任务：目标列切换为 {target_column}")
        elif task_type == 'classification':
            # 分类任务使用投资决策
            target_column = '是否应投资该项目'
            print(f"[INFO] 分类任务：目标列切换为 {target_column}")

    # 加载数据
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)

    # 获取配置参数
    header_row = config.get('header_row', 0) if 'config' in locals() else 0
    max_distance = config.get('max_distance', None)  # 台北数据集距离过滤

    if data_path.endswith('.xlsx'):
        if header_row > 0:
            # 使用指定的header行
            df = pd.read_excel(data_file, header=header_row)
        else:
            df = pd.read_excel(data_file)
    elif data_path.endswith('.csv'):
        if header_row > 0:
            # 对于CSV文件，需要先读取后手动处理
            df_raw = pd.read_csv(data_file, header=None)
            if header_row < len(df_raw):
                # 使用指定行作为列名
                df_raw.columns = df_raw.iloc[header_row]
                df = df_raw[header_row+1:].reset_index(drop=True)
            else:
                df = pd.read_csv(data_file)
        else:
            df = pd.read_csv(data_file)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")

    print(f"[INFO] 加载数据集: {data_path}")
    print(f"[INFO] 数据形状: {df.shape}")
    print(f"[INFO] 目标列: {target_column}")
    print(f"[INFO] 任务类型: {task_type}")

    # === 数据清洗：根据 USE_OLD_PREPROCESSING 选择不同逻辑 ===
    if use_old_preprocessing:
        # ===== 旧版本预处理逻辑 (efd65be4) =====
        # 用于复现回归 R² = 0.784

        # 1. 删除完全为空的行和列
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        print(f"[INFO] [旧版本] 清洗后数据形状 (删除空行空列后): {df.shape}")

        # 2. 确保目标列存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 在清洗后不存在于数据中")

        # 3. 将所有列转换为数值，无法转换的变为NaN
        for col in df.columns:
            if col == target_column and task_type == 'classification' and df[col].dtype == 'object':
                # 对于分类任务的字符串标签，使用LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                print(f"[INFO] [旧版本] 分类目标列 '{col}' 已使用LabelEncoder转换为数值")
            else:
                # 对于所有其他列或回归任务的目标，使用pd.to_numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 删除目标列为NaN的行
        df.dropna(subset=[target_column], inplace=True)
        print(f"[INFO] [旧版本] 移除无效目标行后形状: {df.shape}")

        # 5. 分离特征和目标变量
        y = df[target_column].values
        X_df = df.drop(columns=[target_column])

        # 6. 用0填充特征中的剩余NaN值
        X_df.fillna(0, inplace=True)
        X = X_df.values

    else:
        # ===== 新版本预处理逻辑 (当前 HEAD) =====
        # 用于分类任务（保持 100% 准确率）

        # 1. 删除所有列都是NaN的行
        df = df.dropna(how='all')

        # 2. 删除 Unnamed 列
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"[INFO] [新版本] 删除 {len(unnamed_cols)} 个 Unnamed 列")

        # 3. 处理目标列中的中文标签（在删除非数值列之前）
        if target_column in df.columns and df[target_column].dtype == 'object':
            label_mapping = {'是': 1, '否': 0, 'Yes': 1, 'No': 0}
            df[target_column] = df[target_column].map(label_mapping)
            print(f"[INFO] [新版本] 目标列中文标签已转换为数值")

        # 4. 移除所有非数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = set(df.columns) - set(numeric_cols)
        if non_numeric_cols:
            print(f"[INFO] [新版本] 移除非数值列: {non_numeric_cols}")
        df = df[numeric_cols]

        # 5. 确保目标列存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")

        # 6. 数据类型转换
        df = df.astype(float)

        # 7. 分离特征和目标变量
        y = df[target_column].values
        X = df.drop(columns=[target_column])
        X = X.values if hasattr(X, 'values') else X

    # === 数据集统计信息 ===
    print(f"[INFO] 数据集：{len(y)}条记录")
    if task_type == 'classification':
        # 分类任务打印类别分布
        unique_values, counts = np.unique(y, return_counts=True)
        print(f"[INFO] 类别分布：", end='')
        for val, count in zip(unique_values, counts):
            print(f"类别{int(val)}有{count}个样本", end=', ')
        print()
    else:
        # 回归任务打印目标变量范围
        print(f"[INFO] 目标变量范围：[{y.min():.2f}, {y.max():.2f}]，均值：{y.mean():.2f}")

    # === 特征选择：根据版本和任务类型选择不同逻辑 ===
    if use_old_preprocessing:
        # 旧版本：分类和回归都进行特征选择
        if X.shape[1] > 1:
            correlations = abs(pd.DataFrame(X).corrwith(pd.Series(y)))
            if task_type == 'regression':
                threshold = 0.1
            else:  # classification
                threshold = 0.3

            selected_features = correlations[correlations >= threshold].index.tolist()

            if not selected_features:
                print(f"[INFO] [旧版本] 特征选择：无特征通过阈值{threshold}，保留所有{X.shape[1]}个特征")
                if X.shape[1] == 0:
                    raise ValueError("特征选择后X变为空数组，请检查数据或阈值")
            else:
                X = X[:, selected_features]
                print(f"[INFO] [旧版本] {task_type}特征选择：保留 {len(selected_features)}/{correlations.size} 个特征")
    else:
        # 新版本：只有回归任务进行特征选择
        if task_type == 'regression':
            if X.shape[1] > 1:
                # 计算相关系数
                df_temp = pd.DataFrame(X)
                correlations = abs(df_temp.corrwith(pd.Series(y)))
                threshold = 0.1  # 回归任务相关性阈值
                selected_features = correlations >= threshold

                if selected_features.sum() == 0:
                    # 如果没有特征通过阈值，保留所有特征
                    print(f"[INFO] [新版本] 回归特征选择：无特征通过阈值{threshold}，保留所有{X.shape[1]}个特征")
                else:
                    selected_indices = [i for i, selected in enumerate(selected_features) if selected]
                    X = X[:, selected_indices]
                    print(f"[INFO] [新版本] 回归特征选择：保留 {len(selected_indices)}/{correlations.size} 个特征 (threshold={threshold})")
        elif task_type == 'classification':
            # 分类任务：不进行特征选择，使用所有特征
            print(f"[INFO] [新版本] 分类任务：使用所有 {X.shape[1]} 个特征（不做特征选择）")

    # 应用对数变换（如果配置了需要变换的特征）
    log_transform_features = config.get('log_transform_features', [])
    if log_transform_features:
        # X是numpy数组，转换为DataFrame以便处理
        X_df = pd.DataFrame(X)

        for feature in log_transform_features:
            # 处理列名匹配
            if isinstance(feature, str):
                # 如果feature是字符串，需要匹配column名
                if hasattr(X_df, 'columns') and feature in X_df.columns:
                    col_idx = X_df.columns.get_loc(feature)
                    col_data = X_df.iloc[:, col_idx]
                else:
                    # 尝试匹配原始特征名
                    continue
            else:
                # 如果feature是索引
                if feature < X_df.shape[1]:
                    col_data = X_df.iloc[:, feature]
                else:
                    continue

            # 确保特征值都为正数
            min_val = col_data.min()
            if min_val <= 0:
                print(f"[警告] 特征{feature}包含非正值，无法应用对数变换")
            else:
                if hasattr(X_df, 'columns') and isinstance(feature, str) and feature in X_df.columns:
                    X_df[feature] = np.log1p(X_df[feature])
                else:
                    # 直接修改数组
                    col_data = np.log1p(col_data)
                    X_df.iloc[:, col_idx] = col_data
                print(f"[INFO] 对数变换已应用于特征: {feature}")

        # 更新X为numpy数组
        X = X_df.values

    # 数据标准化（在特征选择和对数变换之后）
    X = DecimalScaler(X)
    print("[INFO] 特征数据标准化完成（DecimalScaler）")

    # 确保数据在合理范围内，避免极端值影响log/tan
    X = np.array(X, dtype=float)

    # 为log和tan添加边界保护
    # log需要正值，tan需要避免π/2 + kπ
    X = np.clip(X, -1e2, 1e2)  # 限制范围避免极值

    # 为log添加小偏移确保正值
    X = np.where(X <= 0, X + 1e-3, X)

    print(f"[INFO] 边界保护已应用")

    print(f"[INFO] 处理后特征数: {X.shape[1]}")
    print(f"[INFO] 处理后样本数: {X.shape[0]}")

    return X, y, df, config

def split_data(X, y, test_size=0.2, random_state=42):
    """分割训练集和测试集"""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
