import physo
import physo.learn.monitoring as monitoring
import numpy as np

class SymbolicRegressionModel:
    """符号回归模型类"""
    
    def __init__(self, config_name='config0', seed=0):
        """
        初始化符号回归模型
        
        Args:
            config_name: str, 配置名称
            seed: int, 随机种子
        """
        self.config_name = config_name
        self.seed = seed
        self.expression = None
        self.config = self._get_config()
        
    def _get_config(self):
        """获取配置 - 使用科学级配置"""
        import physo.config.config2 as config2
        config = config2.config2  # 使用科学配置而非演示配置
        
        # 科学配置参数优化
        return config
    
    def fit(self, X_train, y_train, X_names=None, y_name="y", 
            op_names=None, epochs=10, parallel_mode=False):
        """
        训练符号回归模型
        
        Args:
            X_train: np.ndarray, 训练特征数据
            y_train: np.ndarray, 训练目标数据
            X_names: list, 特征名称列表
            y_name: str, 目标变量名称
            op_names: list, 操作符名称列表
            epochs: int, 训练轮数
            parallel_mode: bool, 是否使用并行模式
        """
        # 设置随机种子
        np.random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        import os # 导入os模块
        
        # 设置特征名称 - 使用极短名称避免长度限制
        if X_names is None:
            X_names = [f'x{i}' for i in range(X_train.shape[0])]
        
        # 设置操作符 - 使用完整操作符集
        if op_names is None:
            op_names = ["mul", "add", "sub", "div", "inv", "neg", 
                       "log", "exp", "sin", "cos", "tan", "n2", "sqrt", "abs"]
        
        # 设置监控器
        save_path_training_curves = 'results/demo_curves.png' # 保持在results根目录
        save_path_log = 'results/demo.log' # 保持在results根目录
        
        # 确保目录存在
        os.makedirs('results', exist_ok=True)
        
        run_logger = lambda: monitoring.RunLogger(
            save_path=save_path_log, do_save=True)
        
        run_visualiser = lambda: monitoring.RunVisualiser(
            epoch_refresh_rate=1,
            save_path=save_path_training_curves,
            do_show=False,
            do_prints=True,
            do_save=True)
        
        # 训练模型
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
        
        return self.expression
    
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
            # 如果表达式有问题，返回合理的默认值
            print(f"预测时发生错误: {e}")
            return np.zeros(X.shape[1])