"""
随机种子管理工具
确保符号回归实验的可重现性
"""

import os
import random
import numpy as np
import torch

def set_global_seed(seed=0):
    """
    设置全局随机种子，确保所有随机源的一致性

    Args:
        seed: int, 随机种子值
    """
    # 设置环境变量以确保完全可重现
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python内置随机模块
    random.seed(seed)

    # NumPy随机数生成器
    np.random.seed(seed)

    # PyTorch随机数生成器
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 启用确定性行为（牺牲性能保证可重现性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[INFO] 全局随机种子已设置为: {seed}")
    print("[INFO] 所有随机源已同步，确保实验可重现性")

def reset_random_state():
    """
    重置随机状态（用于测试）
    """
    set_global_seed(42)  # 使用固定值测试

def check_random_state():
    """
    检查当前随机状态
    """
    print("当前随机状态检查:")
    print(f"Python random: {random.getstate()[1][:3]}")
    print(f"NumPy random: {np.random.get_state()[1][:3]}")
    if torch.cuda.is_available():
        print(f"PyTorch CPU: {torch.get_rng_state()[0:3]}")
        print(f"PyTorch CUDA: {torch.cuda.get_rng_state()[0:3]}")
    else:
        print(f"PyTorch CPU: {torch.get_rng_state()[0:3]}")

def get_random_state():
    """
    获取当前随机状态信息

    Returns:
        dict: 包含各随机源状态的字典
    """
    state = {
        'python_random': str(random.getstate()[1][:3]),
        'numpy_random': str(np.random.get_state()[1][:3]),
        'pytorch_cpu': str(torch.get_rng_state()[0:3])
    }

    if torch.cuda.is_available():
        state['pytorch_cuda'] = str(torch.cuda.get_rng_state()[0:3])

    return state
