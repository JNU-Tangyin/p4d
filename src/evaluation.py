import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve, auc
)
import scipy.stats
import torch

class ModelEvaluator:
    """模型评估器类"""

    def __init__(self, task_type='auto', threshold=0.5):
        """
        初始化评估器

        Args:
            task_type: str, 'regression' 或 'classification' 或 'auto'
            threshold: float, 分类阈值（仅分类任务使用）
        """
        self.task_type = task_type
        self.threshold = threshold
        self.detected_type = None

    def _detect_task_type(self, y_true):
        """
        自动检测任务类型

        Args:
            y_true: np.ndarray, 真实值

        Returns:
            str, 任务类型 ('regression' 或 'classification')
        """
        y_true = np.array(y_true)

        # 检查是否为整数且唯一值较少
        unique_values = np.unique(y_true)

        # 规则1: 如果是整数且唯一值 <= 10，认为是分类
        if all(val == int(val) for val in unique_values) and len(unique_values) <= 10:
            return 'classification'

        # 规则2: 如果是浮点数且唯一值较多，认为是回归
        if len(unique_values) > 10:
            return 'regression'

        # 规则3: 如果值在[0,1]之间且看起来像是概率，可能是分类
        if all(0 <= val <= 1 for val in unique_values):
            # 检查是否为0/1或者接近0/1的值
            binary_like = all(abs(val - round(val)) < 0.01 for val in unique_values)
            if binary_like:
                return 'classification'

        # 规则4: 默认回归
        return 'regression'

    def evaluate_regression(self, y_true, y_pred):
        """
        评估回归任务性能

        Args:
            y_true: np.ndarray, 真实值
            y_pred: np.ndarray, 预测值

        Returns:
            dict, 评估指标
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 基本误差指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # 相关系数（与PhySO一致）
        correlation, _ = scipy.stats.pearsonr(y_true, y_pred)

        # R²分数
        r2 = r2_score(y_true, y_pred)

        # 计算相对误差（处理可能的除零）
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'r2': r2,
            'mape': mape
        }

    def evaluate_classification(self, y_true, y_pred, result_values=None):
        """
        评估分类任务性能

        Args:
            y_true: np.ndarray, 真实标签
            y_pred: np.ndarray, 预测概率值
            result_values: np.ndarray, 原始预测值（用于ROC曲线）

        Returns:
            dict, 评估指标
        """
        if result_values is None:
            result_values = y_pred

        # 检查并处理NaN和Inf值
        result_values = np.array(result_values).flatten()
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 处理NaN和Inf
        mask = np.isfinite(result_values) & np.isfinite(y_pred)
        if not np.all(mask):
            print(f"[WARNING] 发现 {np.sum(~mask)} 个NaN/Inf值，已移除")
            result_values = result_values[mask]
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        # 如果数据为空，返回默认值
        if len(y_true) == 0:
            return {
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'fpr': [0.0, 1.0],
                'tpr': [0.0, 1.0],
                'auc_score': 0.5,
                'ks_score': 0.0
            }

        # 多分类处理
        if len(np.unique(y_true)) > 2:
            # 多分类：将连续预测转为离散类别
            y_pred_classes = np.round(y_pred).astype(int)
            y_pred_classes = np.clip(y_pred_classes, int(y_true.min()), int(y_true.max()))

            # 多分类指标
            f1 = f1_score(y_true, y_pred_classes, average='macro')
            precision = precision_score(y_true, y_pred_classes, average='macro')
            recall = recall_score(y_true, y_pred_classes, average='macro')
            accuracy = accuracy_score(y_true, y_pred_classes)
            conf_matrix = confusion_matrix(y_true, y_pred_classes)
        else:
            # 二分类处理
            y_pred_binary = np.where(y_pred >= self.threshold, 1, 0)

            # 二分类指标
            f1 = f1_score(y_true, y_pred_binary, average='binary')
            precision = precision_score(y_true, y_pred_binary, average='binary')
            recall = recall_score(y_true, y_pred_binary, average='binary')
            accuracy = accuracy_score(y_true, y_pred_binary)
            conf_matrix = confusion_matrix(y_true, y_pred_binary)

        # ROC和KS值计算
        try:
            if len(np.unique(y_true)) == 2:
                # 二分类：计算ROC和KS
                fpr, tpr, thresholds = roc_curve(y_true, result_values)
                auc_score = auc(fpr, tpr)

                # 计算KS统计量
                ks_score = np.max(tpr - fpr)
            else:
                # 多分类：返回安全默认值
                auc_score = 0.5
                ks_score = 0.0
                fpr, tpr, thresholds = [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        except Exception as e:
            print(f"[WARNING] ROC/KS计算失败: {e}")
            auc_score = 0.5
            ks_score = 0.0
            fpr, tpr, thresholds = [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]

        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist() if hasattr(conf_matrix, 'tolist') else conf_matrix,
            'auc_score': auc_score,
            'ks_score': ks_score,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }

    def evaluate(self, y_true, y_pred):
        """
        根据任务类型自动选择评估方法

        Args:
            y_true: np.ndarray, 真实值
            y_pred: np.ndarray, 预测值

        Returns:
            dict, 评估指标
        """
        # 如果设置为auto，自动检测任务类型
        if self.task_type == 'auto':
            actual_type = self._detect_task_type(y_true)
            self.detected_type = actual_type
            print(f"[INFO] 自动检测到任务类型: {actual_type}")
        else:
            actual_type = self.task_type
            self.detected_type = actual_type

        if actual_type == 'regression':
            return self.evaluate_regression(y_true, y_pred)
        elif actual_type == 'classification':
            return self.evaluate_classification(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported task type: {actual_type}")


    def plot_feature_sensitivity(self, model, X_train, X_names,
                               save_dir='results/'):
        """
        绘制特征敏感性分析图

        Args:
            model: 训练好的符号回归模型
            X_train: np.ndarray, 训练数据
            X_names: list, 特征名称列表
            save_dir: str, 保存目录
        """
        import sympy as sp

        best_expr_info = model.get_best_expression()
        best_expr_sympy = best_expr_info['sympy_expression']

        # 获取表达式中的符号
        symbols_list = list(best_expr_sympy.free_symbols)

        if len(symbols_list) >= 2:
            # 绘制前两个特征的敏感性分析
            for i, symbol in enumerate(symbols_list[:2]):
                try:
                    func_diff = sp.diff(best_expr_sympy, symbol)
                except Exception:
                    # 如果直接求导失败，使用数值近似
                    print(f"[警告] 符号 {symbol} 的解析求导失败，使用数值近似")
                    func_diff = None

                if func_diff is not None:
                    # 转换为可执行函数，指定使用numpy和sympy模块
                    func_diff_lambdified = sp.lambdify(symbols_list, func_diff, modules=['numpy', 'sympy'])

                    # 生成数据点
                    symbol_values = np.linspace(
                        X_train[i].min(), X_train[i].max(), 10000
                    )

                    # 计算导数值
                    other_values = [
                        np.mean(X_train[j]) for j in range(len(symbols_list))
                        if j != i
                    ]

                    # 构建输入参数
                    args = []
                    for j in range(len(symbols_list)):
                        if j == i:
                            args.append(symbol_values)
                        else:
                            idx = 0 if j < i else (j-1)
                            args.append(other_values[idx] * np.ones_like(symbol_values))

                    # 计算敏感性
                    try:
                        sensitivity = func_diff_lambdified(*args)
                    except Exception:
                        # 如果函数计算失败，使用数值近似
                        print(f"[警告] 敏感性计算失败，使用数值近似")
                        sensitivity = np.zeros_like(symbol_values)

                    # 确保敏感性是数组形式
                    if np.isscalar(sensitivity):
                        sensitivity = np.full_like(symbol_values, sensitivity)
                    elif len(np.array(sensitivity).shape) == 0:
                        sensitivity = np.full_like(symbol_values, sensitivity)
                    elif not np.isfinite(sensitivity).all():
                        # 处理NaN/Inf值
                        sensitivity = np.nan_to_num(sensitivity, nan=0.0, posinf=1e6, neginf=-1e6)
                else:
                    # 使用数值近似
                    print(f"[警告] 使用数值近似计算敏感性")
                    symbol_values = np.linspace(
                        X_train[i].min(), X_train[i].max(), 1000
                    )

                    # 数值近似导数
                    delta = 1e-5
                    f_original = sp.lambdify(symbols_list, best_expr_sympy, modules=['numpy'])

                    sensitivity = []
                    for val in symbol_values:
                        # 计算f(x+delta)
                        args_pos = []
                        for j in range(len(symbols_list)):
                            if j == i:
                                args_pos.append(val + delta)
                            else:
                                args_pos.append(np.mean(X_train[j]))

                        # 计算f(x-delta)
                        args_neg = []
                        for j in range(len(symbols_list)):
                            if j == i:
                                args_neg.append(val - delta)
                            else:
                                args_neg.append(np.mean(X_train[j]))

                        try:
                            f_plus = f_original(*args_pos)
                            f_minus = f_original(*args_neg)
                            deriv = (f_plus - f_minus) / (2 * delta)
                            sensitivity.append(deriv)
                        except:
                            sensitivity.append(0.0)

                    sensitivity = np.array(sensitivity)
                    sensitivity = np.nan_to_num(sensitivity, nan=0.0, posinf=1e6, neginf=-1e6)

                # 绘制敏感性图
                import matplotlib.pyplot as plt
                plt.rcParams['text.usetex'] = False  # 禁用LaTeX
                plt.rcParams['font.family'] = 'DejaVu Sans'

                plt.figure(figsize=(12, 8))
                plt.plot(symbol_values, sensitivity, linewidth=2)
                plt.xlabel(str(symbol), fontsize=12)
                plt.ylabel('Sensitivity', fontsize=12)
                plt.title(f'Sensitivity Analysis for {symbol}', fontsize=14)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()

                save_path = f'{save_dir}sensitivity_{symbol}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

    def print_evaluation_report(self, metrics, dataset_name="Test"):
        """
        打印评估报告

        Args:
            metrics: dict, 评估指标
            dataset_name: str, 数据集名称
        """
        task_display = self.detected_type or self.task_type
        print(f"\n=== {dataset_name} Set Evaluation Report ===")
        print(f"任务类型: {task_display}")

        if 'correlation' in metrics:  # 回归指标
            print(f"Correlation (R): {metrics['correlation']:.4f}")
            print(f"R2 Score: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"MSE: {metrics['mse']:.4f}")
            if metrics['mape'] != np.inf:
                print(f"MAPE: {metrics['mape']:.2f}%")
            else:
                print("MAPE: N/A (zero values in target)")
        elif 'f1_score' in metrics:  # 分类指标
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            if 'auc_score' in metrics:
                print(f"AUC Score: {metrics['auc_score']:.4f}")
                print(f"KS Score: {metrics['ks_score']:.4f}")
            print(f"Confusion Matrix:")
            print(metrics['confusion_matrix'])

        print("=" * 40)


# 以下为0c8964b版本兼容性函数
def get_regression_metrics(y_true, y_pred):
    """
    Calculates a standard set of metrics for a regression task.
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    metrics = {
        'R-squared': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    return metrics

def get_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculates a standard set of metrics for a binary classification task.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_pred_binary = (np.array(y_pred_proba) >= threshold).astype(int)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_binary),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Confusion Matrix': confusion_matrix(y_true, y_pred_binary).tolist()
    }
    return metrics
