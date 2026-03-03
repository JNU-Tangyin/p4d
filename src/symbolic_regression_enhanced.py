"""
Enhanced Symbolic Regression Model with Data Collection
增强版符号回归模型，集成数据收集功能
"""

import physo
import physo.learn.monitoring as monitoring
import numpy as np
import pandas as pd
from .data_collector import PhySODataCollector, EnhancedRunLogger
import logging

class EnhancedSymbolicRegressionModel:
    """增强版符号回归模型类，集成数据收集功能"""

    def __init__(self, config_name='config2', seed=0, experiment_name="physo_symbolic_regression"):
        """
        初始化增强版符号回归模型

        Args:
            config_name: str, 配置名称（推荐使用config2科学级配置）
            seed: int, 随机种子
            experiment_name: str, 实验名称
        """
        self.config_name = config_name
        self.seed = seed
        self.experiment_name = experiment_name
        self.expression = None
        self.config = self._get_config()

        # 初始化数据收集器
        self.data_collector = PhySODataCollector(experiment_name)
        self.logger = logging.getLogger(__name__)

    def _get_config(self):
        """获取配置 - 使用科学级配置"""
        if self.config_name == 'config2':
            import physo.config.config2 as config2
            config = config2.config2
        elif self.config_name == 'config1':
            import physo.config.config1 as config1
            config = config1.config1
        else:
            # 默认使用config2
            import physo.config.config2 as config2
            config = config2.config2

        return config

    def fit(self, X_train, y_train, X_names=None, y_name="y",
            op_names=None, epochs=15, parallel_mode=False, X_test=None, y_test=None):
        """
        训练增强版符号回归模型

        Args:
            X_train: np.ndarray, 训练特征数据 (n_features, n_samples)
            y_train: np.ndarray, 训练目标数据
            X_names: list, 特征名称列表
            y_name: str, 目标变量名称
            op_names: list, 操作符名称列表
            epochs: int, 训练轮数
            parallel_mode: bool, 是否使用并行模式
            X_test: np.ndarray, 测试特征数据 (可选)
            y_test: np.ndarray, 测试目标数据 (可选)
        """
        # 设置随机种子
        np.random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        import os

        # 转换数据维度（PhySO需要 (n_features, n_samples)格式）
        if X_train.shape[0] > X_train.shape[1]:
            # 如果是 (n_samples, n_features)，转置为 (n_features, n_samples)
            X_train = X_train.T
            self.logger.info(f"Transposed X_train to shape: {X_train.shape}")

        # 设置特征名称
        if X_names is None:
            X_names = [f'X{i}' for i in range(X_train.shape[0])]

        # 设置操作符 - 使用完整操作符集
        if op_names is None:
            op_names = ["mul", "add", "sub", "div", "inv", "neg",
                       "log", "exp", "sin", "cos", "tan", "n2", "sqrt", "abs"]

        # 设置监控器
        save_path_training_curves = f'{self.data_collector.base_dir}/training/physo_curves.png'
        save_path_log = f'{self.data_collector.base_dir}/training/physo_training.log'

        # 开始训练记录
        self.data_collector.start_training_recording(X_train, y_train, X_names, y_name, op_names, epochs)

        # 创建增强的运行记录器
        run_logger = lambda: EnhancedRunLogger(self.data_collector)

        # 创建可视化器（保持原有功能）
        run_visualiser = lambda: monitoring.RunVisualiser(
            epoch_refresh_rate=1,
            save_path=save_path_training_curves,
            do_show=False,
            do_prints=True,
            do_save=True)

        self.logger.info(f"Starting PhySO training: {X_train.shape[1]} samples, {epochs} epochs")
        self.logger.info(f"Features: {X_names}")
        self.logger.info(f"Operators: {op_names}")

        # 训练模型
        try:
            self.expression, self.logs = physo.SR(
                X_train, y_train,
                X_names=X_names,
                y_name=y_name,
                op_names=op_names,
                get_run_logger=run_logger,
                get_run_visualiser=run_visualiser,
                run_config=self.config,
                parallel_mode=parallel_mode,
                epochs=epochs
            )

            # 如果有测试数据，进行测试评估
            if X_test is not None and y_test is not None:
                if X_test.shape[0] > X_test.shape[1]:
                    X_test = X_test.T

                y_pred = self.predict(X_test)
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                test_r2 = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)

                self.logger.info(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

                # 记录最终结果
                final_r2 = self.logs.best_val_R if hasattr(self.logs, 'best_val_R') else 0.0
                train_r2 = self.logs.best_train_R if hasattr(self.logs, 'best_train_R') else 0.0

                self.data_collector.record_final_results(
                    final_expression=self.expression,
                    final_r2=final_r2,
                    train_r2=train_r2,
                    test_r2=test_r2,
                    test_predictions=y_pred,
                    test_actual=y_test
                )

            # 保存所有数据
            self.data_collector.save_all_data()

            self.logger.info("PhySO training completed successfully")

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise

        return self.expression

    def fit_classification(self, X_train, y_train, X_names=None, X_test=None, y_test=None, epochs=15):
        """
        训练分类任务（使用PhySO的class symbolic regression）

        Args:
            X_train: np.ndarray, 训练特征数据
            y_train: np.ndarray, 训练标签 (0或1)
            X_names: list, 特征名称列表
            X_test: np.ndarray, 测试特征数据 (可选)
            y_test: np.ndarray, 测试标签 (可选)
            epochs: int, 训练轮数
        """
        # 记录这是分类任务
        self.data_collector.record_experiment_config({
            "task_type": "classification",
            "n_classes": 2,
            "class_distribution": {
                "class_0": int(np.sum(y_train == 0)),
                "class_1": int(np.sum(y_train == 1))
            }
        })

        # 运行训练
        expression = self.fit(X_train, y_train, X_names=X_names, y_name="Decision", epochs=epochs)

        # 如果有测试数据，进行分类评估
        if X_test is not None and y_test is not None:
            y_pred_proba = self.predict_proba(X_test)
            y_pred_class = (y_pred_proba >= 0.5).astype(int)

            # 记录分类结果
            self.data_collector.record_classification_results(
                y_true=y_test,
                y_pred_proba=y_pred_proba,
                y_pred_class=y_pred_class,
                class_names=['No_Invest', 'Invest']
            )

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_test, y_pred_class)
            precision = precision_score(y_test, y_pred_class)
            recall = recall_score(y_test, y_pred_class)
            f1 = f1_score(y_test, y_pred_class)

            self.logger.info(f"Classification Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

        return expression

    def get_best_expression(self):
        """获取最佳表达式"""
        if self.expression is None:
            raise ValueError("Model has not been trained yet")

        from physo.benchmark.utils import symbolic_utils as su
        import sympy

        best_expr = self.expression
        best_expr_sympy = best_expr.get_infix_sympy(evaluate_consts=True)
        best_expr_sympy = best_expr_sympy[0]

        # 简化和四舍五入常数
        clean_expr = su.clean_sympy_expr(best_expr_sympy, round_decimal=4)

        return {
            'expression': best_expr,
            'sympy_expression': best_expr_sympy,
            'clean_expression': clean_expr
        }

    def predict(self, X):
        """
        使用训练好的模型进行预测

        Args:
            X: np.ndarray, 输入特征数据

        Returns:
            np.ndarray, 预测结果
        """
        if self.expression is None:
            raise ValueError("Model has not been trained yet")

        # 确保数据维度正确
        if X.shape[0] > X.shape[1]:
            X = X.T

        import sympy as sp
        from sympy import symbols, lambdify
        import numpy as np

        X_names = [f"X{i}" for i in range(X.shape[0])]

        best_expr_info = self.get_best_expression()
        best_expr_sympy = best_expr_info['sympy_expression']

        # 确保表达式是实数
        best_expr_sympy = sp.re(best_expr_sympy)

        # 创建lambda函数
        best_prog_func = sp.lambdify(X_names, best_expr_sympy, modules=['numpy'])

        # 处理输入数据
        X_values = [X[i, :] for i in range(X.shape[0])]

        try:
            predictions = best_prog_func(*X_values)
            # 处理可能的复数结果
            if np.iscomplexobj(predictions):
                predictions = np.real(predictions)
            return np.array(predictions).astype(np.float32)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            # 如果表达式有问题，返回合理的中性值
            return np.zeros(X.shape[1])

    def predict_proba(self, X):
        """
        预测分类概率（用于分类任务）

        Args:
            X: np.ndarray, 输入特征数据

        Returns:
            np.ndarray, 预测概率
        """
        # 使用sigmoid函数将预测值转换为概率
        predictions = self.predict(X)
        probabilities = 1 / (1 + np.exp(-predictions))
        return probabilities

    def get_training_curves(self):
        """获取训练曲线数据"""
        return self.data_collector.get_training_curves_df()

    def get_predictions_data(self):
        """获取预测结果数据"""
        return self.data_collector.get_predictions_df()

    def get_experiment_data_path(self):
        """获取实验数据路径"""
        return self.data_collector.base_dir