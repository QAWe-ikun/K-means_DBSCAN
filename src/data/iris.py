"""
鸢尾花数据集加载模块
使用sklearn.datasets加载Iris数据集
"""
import numpy as np
from typing import Tuple
from sklearn.datasets import load_iris as sklearn_load_iris


def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    使用sklearn加载鸢尾花数据集

    Returns:
        X: 特征矩阵 (150, 4)
        y: 标签向量 (150,)
    """
    iris = sklearn_load_iris()
    X = iris.data.astype(np.float64) # type: ignore
    y = iris.target.astype(int) # type: ignore
    return X, y


def normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score标准化

    Args:
        X: 特征矩阵

    Returns:
        X_normalized: 标准化后的特征矩阵
        mean: 均值
        std: 标准差
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # 避免除零
    std[std == 0] = 1
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def min_max_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-Max归一化到[0, 1]

    Args:
        X: 特征矩阵

    Returns:
        X_normalized: 归一化后的特征矩阵
        min_val: 最小值
        max_val: 最大值
    """
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    # 避免除零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    X_normalized = (X - min_val) / range_val
    return X_normalized, min_val, max_val


if __name__ == "__main__":
    # 测试数据加载
    X, y = load_iris()
    print(f"数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    X_norm, mean, std = normalize(X)
    print(f"\n标准化后均值: {mean}")
    print(f"标准化后标准差: {std}")