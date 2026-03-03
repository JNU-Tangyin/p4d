"""
Enhanced Data Collection Module for PhySO Training Process
用于记录PhySO训练过程的详细数据，支持可视化需求
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import logging

class PhySODataCollector:
    """PhySO训练过程数据收集器"""

    def __init__(self, experiment_name="physo_experiment"):
        """
        初始化数据收集器

        Args:
            experiment_name: str, 实验名称
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建实验数据目录
        self.base_dir = f"experiment_data/{experiment_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(f"{self.base_dir}/training", exist_ok=True)
        os.makedirs(f"{self.base_dir}/regression", exist_ok=True)
        os.makedirs(f"{self.base_dir}/classification", exist_ok=True)

        # 初始化数据记录
        self.training_curves = []
        self.pareto_frontier = []
        self.expression_evolution = []
        self.model_predictions = []
        self.experiment_config = {}

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.base_dir}/experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def record_experiment_config(self, config):
        """记录实验配置"""
        self.experiment_config.update(config)
        self.logger.info(f"Experiment config recorded: {config}")

    def start_training_recording(self, X_train, y_train, X_names, y_name, op_names, epochs):
        """开始训练记录"""
        config = {
            "dataset_size": len(y_train),
            "n_features": X_train.shape[0],
            "X_names": X_names,
            "y_name": y_name,
            "op_names": op_names,
            "epochs": epochs,
            "start_time": datetime.now().isoformat()
        }
        self.record_experiment_config(config)
        self.logger.info(f"Training recording started: {len(y_train)} samples, {epochs} epochs")

    def record_epoch_data(self, epoch, loss, best_r2, train_r2, best_expression, complexity, n_expressions_evaluated):
        """记录每个epoch的训练数据"""
        epoch_data = {
            "epoch": epoch,
            "loss": float(loss),
            "best_r2": float(best_r2),
            "train_r2": float(train_r2),
            "complexity": int(complexity),
            "n_expressions_evaluated": int(n_expressions_evaluated),
            "best_expression": str(best_expression),
            "simplified_expression": self._simplify_expression(best_expression),
            "timestamp": datetime.now().isoformat()
        }

        self.training_curves.append(epoch_data)

        # 记录帕累托前沿点
        pareto_point = {
            "epoch": epoch,
            "complexity": int(complexity),
            "r2": float(best_r2),
            "expression": str(best_expression)
        }
        self.pareto_frontier.append(pareto_point)

        # 记录表达式演进
        expr_evolution = {
            "epoch": epoch,
            "expression": str(best_expression),
            "r2": float(best_r2),
            "complexity": int(complexity)
        }
        self.expression_evolution.append(expr_evolution)

        if epoch % 5 == 0 or epoch == 0:  # 每5个epoch或第一个epoch记录日志
            self.logger.info(f"Epoch {epoch}: Best R²={best_r2:.4f}, Complexity={complexity}, Loss={loss:.6f}")

    def record_final_results(self, final_expression, final_r2, train_r2, test_r2, test_predictions, test_actual):
        """记录最终结果"""
        final_data = {
            "final_expression": str(final_expression),
            "simplified_expression": self._simplify_expression(final_expression),
            "final_r2": float(final_r2),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "n_train_samples": len(train_r2) if hasattr(train_r2, '__len__') else 1,
            "n_test_samples": len(test_actual),
            "end_time": datetime.now().isoformat()
        }

        self.record_experiment_config(final_data)

        # 记录预测结果
        for i, (pred, actual) in enumerate(zip(test_predictions, test_actual)):
            prediction_data = {
                "sample_id": i,
                "y_true": float(actual),
                "y_pred": float(pred),
                "residual": float(actual - pred),
                "abs_error": float(abs(actual - pred)),
                "percentage_error": float(abs(actual - pred) / (actual + 1e-8) * 100)
            }
            self.model_predictions.append(prediction_data)

        self.logger.info(f"Final results recorded: R²={final_r2:.4f}, Test R²={test_r2:.4f}")

    def record_classification_results(self, y_true, y_pred_proba, y_pred_class, class_names=None):
        """记录分类结果"""
        from sklearn.metrics import confusion_matrix, classification_report

        if class_names is None:
            class_names = ['Class_0', 'Class_1']

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred_class)

        classification_data = {
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_true, y_pred_class,
                                                         target_names=class_names,
                                                         output_dict=True),
            "accuracy": float(np.mean(y_true == y_pred_class)),
            "n_samples": len(y_true),
            "class_names": class_names
        }

        self.record_experiment_config({"classification_results": classification_data})

        # 记录分类预测详情
        for i, (true_val, pred_proba, pred_class) in enumerate(zip(y_true, y_pred_proba, y_pred_class)):
            class_data = {
                "sample_id": i,
                "y_true": int(true_val),
                "y_pred_class": int(pred_class),
                "y_pred_proba": [float(p) for p in pred_proba],
                "correct": int(true_val == pred_class)
            }
            # 这里可以扩展为单独的分类预测记录文件

        self.logger.info(f"Classification results: Accuracy={classification_data['accuracy']:.4f}")

    def record_sensitivity_analysis(self, feature_names, base_values, sensitivity_results):
        """记录敏感性分析结果"""
        sensitivity_data = []

        for i, (feature_name, base_val) in enumerate(zip(feature_names, base_values)):
            if i < len(sensitivity_results):
                result = sensitivity_results[i]
                sensitivity_data.append({
                    "feature_name": feature_name,
                    "base_value": float(base_val),
                    "sensitivity_score": float(result.get('sensitivity', 0)),
                    "impact_level": str(result.get('impact', 'Medium')),
                    "low_change": float(result.get('low_change', base_val * 0.8)),
                    "high_change": float(result.get('high_change', base_val * 1.2))
                })

        sensitivity_df = pd.DataFrame(sensitivity_data)
        sensitivity_df.to_csv(f"{self.base_dir}/analysis/sensitivity_analysis.csv", index=False)
        self.logger.info(f"Sensitivity analysis recorded for {len(sensitivity_data)} features")

    def _simplify_expression(self, expression):
        """简化表达式字符串，便于显示"""
        try:
            expr_str = str(expression)
            # 移除过长的表达式，只保留核心部分
            if len(expr_str) > 100:
                return expr_str[:100] + "..."
            return expr_str
        except:
            return str(expression)

    def save_all_data(self):
        """保存所有记录的数据"""
        # 保存训练曲线
        if self.training_curves:
            training_df = pd.DataFrame(self.training_curves)
            training_df.to_csv(f"{self.base_dir}/training/physo_training_curves.csv", index=False)

            # 保存帕累托前沿
            pareto_df = pd.DataFrame(self.pareto_frontier)
            pareto_df.to_csv(f"{self.base_dir}/training/pareto_frontier.csv", index=False)

            # 保存表达式演进
            evolution_df = pd.DataFrame(self.expression_evolution)
            evolution_df.to_csv(f"{self.base_dir}/training/expression_evolution.csv", index=False)

        # 保存预测结果
        if self.model_predictions:
            predictions_df = pd.DataFrame(self.model_predictions)
            predictions_df.to_csv(f"{self.base_dir}/regression/physo_predictions.csv", index=False)

        # 保存实验配置
        with open(f"{self.base_dir}/experiment_config.json", 'w') as f:
            json.dump(self.experiment_config, f, indent=2, default=str)

        self.logger.info(f"All data saved to {self.base_dir}")

    def get_training_curves_df(self):
        """获取训练曲线数据框"""
        return pd.DataFrame(self.training_curves) if self.training_curves else pd.DataFrame()

    def get_predictions_df(self):
        """获取预测结果数据框"""
        return pd.DataFrame(self.model_predictions) if self.model_predictions else pd.DataFrame()

    def get_pareto_frontier_df(self):
        """获取帕累托前沿数据框"""
        return pd.DataFrame(self.pareto_frontier) if self.pareto_frontier else pd.DataFrame()


class EnhancedRunLogger:
    """增强的运行记录器，配合数据收集器使用"""

    def __init__(self, data_collector):
        self.data_collector = data_collector

    def __call__(self):
        return self

    def log_epoch(self, epoch, logger, run):
        """记录epoch数据"""
        try:
            # 获取最佳表达式信息
            best_expr = run.best_program
            best_r2 = run.best_val_R
            train_r2 = run.best_train_R
            loss = run.best_loss

            # 计算复杂度（表达式长度）
            complexity = len(str(best_expr)) if best_expr else 0

            # 记录到数据收集器
            self.data_collector.record_epoch_data(
                epoch=epoch,
                loss=loss,
                best_r2=best_r2,
                train_r2=train_r2,
                best_expression=best_expr,
                complexity=complexity,
                n_expressions_evaluated=len(run.population) if hasattr(run, 'population') else 0
            )

        except Exception as e:
            print(f"Error logging epoch {epoch}: {e}")

    def log_final_results(self, final_expression, final_r2, test_r2, test_predictions, test_actual):
        """记录最终结果"""
        self.data_collector.record_final_results(
            final_expression=final_expression,
            final_r2=final_r2,
            train_r2=final_r2,  # 这里需要根据实际情况调整
            test_r2=test_r2,
            test_predictions=test_predictions,
            test_actual=test_actual
        )