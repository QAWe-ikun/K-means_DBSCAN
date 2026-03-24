"""
鸢尾花数据集加载模块
从网络下载Iris数据集并缓存到本地
"""
import numpy as np
from typing import Tuple
import urllib.request
import os


# 数据集缓存路径
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'iris.csv')


def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    加载鸢尾花数据集（优先从本地缓存加载，否则从网络下载）

    Returns:
        X: 特征矩阵 (150, 4)
        y: 标签向量 (150,)
        
    Raises:
        RuntimeError: 无法加载数据时抛出异常
    """
    csv_data = None
    
    # 尝试从本地缓存加载
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                csv_data = f.read()
            print(f"从本地缓存加载数据: {CACHE_FILE}")
        except Exception as e:
            print(f"读取本地缓存失败: {e}")
            csv_data = None
    
    # 如果本地没有，从网络下载
    if csv_data is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        
        try:
            print(f"从网络下载数据集: {url}")
            with urllib.request.urlopen(url, timeout=15) as response:
                csv_data = response.read().decode('utf-8')
        except Exception as e:
            raise RuntimeError(
                f"无法从网络下载鸢尾花数据集: {e}\n"
                f"请检查网络连接，或手动下载数据集到: {CACHE_FILE}"
            )
        
        # 保存到本地缓存（添加列名）
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                # 写入列名
                f.write("sepal_length,sepal_width,petal_length,petal_width,species\n")
                # 写入数据（去除首尾空白）
                f.write(csv_data.strip())
            print(f"数据已缓存到: {CACHE_FILE}")
        except Exception as e:
            print(f"警告: 无法保存缓存文件: {e}")
    
    # 解析CSV数据
    lines = [line.strip() for line in csv_data.strip().split('\n') if line.strip()]
    X_list = []
    y_list = []
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    
    for line in lines:
        parts = line.split(',')
        if len(parts) == 5:
            try:
                X_list.append([float(x) for x in parts[:4]])
                label = parts[4].strip()
                y_list.append(label_map.get(label, -1))
            except ValueError:
                continue
    
    if len(X_list) == 0:
        raise RuntimeError("解析数据失败，数据为空")
    
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)
    
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