"""
Enhanced Monitoring Module for PhySO
增强版PhySO监控模块，集成数据收集功能
"""

import physo.learn.monitoring as monitoring
import numpy as np
import pandas as pd

class EnhancedRunLogger:
    """增强的运行记录器，每轮记录详细数据"""

    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.epoch_counter = 0

    def __call__(self):
        return self

    def __call__(self, run, epoch=None):
        """记录epoch数据"""
        if epoch is None:
            epoch = self.epoch_counter

        try:
            # 获取最佳表达式信息
            best_expr = run.best_program

            # 从run对象中提取性能指标
            if hasattr(run, 'best_val_R'):
                best_r2 = run.best_val_R
            elif hasattr(run, 'val_R'):
                best_r2 = np.max(run.val_R) if run.val_R else 0.0
            else:
                best_r2 = 0.0

            if hasattr(run, 'best_train_R'):
                train_r2 = run.best_train_R
            elif hasattr(run, 'train_R'):
                train_r2 = np.max(run.train_R) if run.train_R else 0.0
            else:
                train_r2 = 0.0

            # 计算损失（如果没有直接的损失，使用1-R²作为代理）
            if hasattr(run, 'best_loss'):
                loss = run.best_loss
            else:
                loss = 1.0 - best_r2

            # 计算复杂度（表达式长度）
            complexity = len(str(best_expr)) if best_expr else 0

            # 估算评估的表达式数量
            n_expressions = len(run.population) if hasattr(run, 'population') else 0

            # 记录到数据收集器
            self.data_collector.record_epoch_data(
                epoch=epoch,
                loss=loss,
                best_r2=best_r2,
                train_r2=train_r2,
                best_expression=best_expr,
                complexity=complexity,
                n_expressions_evaluated=n_expressions
            )

        except Exception as e:
            print(f"Warning: Error logging epoch {epoch}: {e}")
            # 继续执行，不中断训练
            pass

        self.epoch_counter += 1

    def log_final_results(self, final_expression, final_r2, train_r2, test_r2, test_predictions, test_actual):
        """记录最终结果"""
        try:
            self.data_collector.record_final_results(
                final_expression=final_expression,
                final_r2=final_r2,
                train_r2=train_r2,
                test_r2=test_r2,
                test_predictions=test_predictions,
                test_actual=test_actual
            )
        except Exception as e:
            print(f"Warning: Error logging final results: {e}")

class CustomRunLogger:
    """自定义运行记录器，适配PhySO接口"""

    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.enhanced_logger = EnhancedRunLogger(data_collector)

    def __call__(self):
        """PhySO调用的接口"""
        return self

    def __call__(self, run, epoch=None):
        """记录epoch数据"""
        self.enhanced_logger(run, epoch)


def create_enhanced_logger(data_collector):
    """创建增强的logger工厂函数"""
    def get_enhanced_logger():
        return CustomRunLogger(data_collector)
    return get_enhanced_logger


# 扩展原有的监控功能
class EnhancedRunVisualiser(monitoring.RunVisualiser):
    """增强的运行可视化器，保留原有功能的同时支持数据收集"""

    def __init__(self, *args, data_collector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collector = data_collector

    def __call__(self, run, epoch=None):
        """在原有可视化功能基础上添加数据记录"""
        # 执行原有的可视化逻辑
        super().__call__(run, epoch)

        # 添加数据记录逻辑
        if self.data_collector and epoch is not None:
            try:
                # 获取运行状态
                best_r2 = getattr(run, 'best_val_R', 0.0)
                if hasattr(run, 'val_R') and len(run.val_R) > 0:
                    best_r2 = max(best_r2, max(run.val_R))

                train_r2 = getattr(run, 'best_train_R', 0.0)
                if hasattr(run, 'train_R') and len(run.train_R) > 0:
                    train_r2 = max(train_r2, max(run.train_R))

                best_expr = getattr(run, 'best_program', None)
                loss = getattr(run, 'best_loss', 1.0 - best_r2)

                complexity = len(str(best_expr)) if best_expr else 0
                n_expressions = len(run.population) if hasattr(run, 'population') else 0

                self.data_collector.record_epoch_data(
                    epoch=epoch,
                    loss=loss,
                    best_r2=best_r2,
                    train_r2=train_r2,
                    best_expression=best_expr,
                    complexity=complexity,
                    n_expressions_evaluated=n_expressions
                )

            except Exception as e:
                print(f"Warning: Enhanced visualization logging error: {e}")
                pass


def create_enhanced_visualiser(data_collector, **kwargs):
    """创建增强的可视化器工厂函数"""
    def get_enhanced_visualiser():
        return EnhancedRunVisualiser(data_collector=data_collector, **kwargs)
    return get_enhanced_visualiser


class PhySOTrainingMonitor:
    """PhySO训练监控器，统一管理数据收集和可视化"""

    def __init__(self, experiment_name="physo_experiment", base_dir="experiment_data"):
        """
        初始化训练监控器

        Args:
            experiment_name: str, 实验名称
            base_dir: str, 基础数据目录
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir

        # 创建数据收集器
        self.data_collector = PhySODataCollector(experiment_name)

    def get_logger(self):
        """获取增强的logger"""
        return create_enhanced_logger(self.data_collector)

    def get_visualiser(self, **kwargs):
        """获取增强的可视化器"""
        return create_enhanced_visualiser(self.data_collector, **kwargs)

    def record_training_start(self, X_train, y_train, X_names, y_name, op_names, epochs):
        """记录训练开始"""
        self.data_collector.start_training_recording(X_train, y_train, X_names, y_name, op_names, epochs)

    def record_training_end(self, model, X_test=None, y_test=None):
        """记录训练结束"""
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            from sklearn.metrics import r2_score

            test_r2 = r2_score(y_test, y_pred)
            train_r2 = getattr(model.logs, 'best_train_R', 0.0) if hasattr(model, 'logs') else 0.0
            final_r2 = getattr(model.logs, 'best_val_R', 0.0) if hasattr(model, 'logs') else 0.0

            self.data_collector.record_final_results(
                final_expression=model.expression,
                final_r2=final_r2,
                train_r2=train_r2,
                test_r2=test_r2,
                test_predictions=y_pred,
                test_actual=y_test
            )

    def save_all_data(self):
        """保存所有数据"""
        self.data_collector.save_all_data()

    def get_data_path(self):
        """获取数据路径"""
        return self.data_collector.base_dir

    def get_training_curves(self):
        """获取训练曲线"""
        return self.data_collector.get_training_curves_df()

    def get_predictions_data(self):
        """获取预测数据"""
        return self.data_collector.get_predictions_df()