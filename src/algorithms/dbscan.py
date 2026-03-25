"""
DBSCAN聚类算法实现
核心逻辑：计算邻域 → 识别核心点 → 扩展簇 → 标记噪声点
"""
import numpy as np
from typing import Tuple, Optional, List, Set


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 聚类算法
    
    Parameters:
        eps: 邻域半径
        min_samples: 核心点所需的最小邻域样本数
        metric: 距离度量方式 ('euclidean' 或 'manhattan')
    """
    
    # 噪声点标签
    NOISE = -1
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        计算距离矩阵
        
        Args:
            X: 数据矩阵 (n_samples, n_features)
            
        Returns:
            distance_matrix: 距离矩阵 (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        
        if self.metric == 'euclidean':
            # 欧氏距离: ||a - b||_2
            # 使用广播机制高效计算
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.metric == 'manhattan':
            # 曼哈顿距离: ||a - b||_1
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            distance_matrix = np.sum(np.abs(diff), axis=2)
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")
        
        return distance_matrix
    
    def _get_neighbors(self, distance_matrix: np.ndarray, point_idx: int) -> np.ndarray:
        """
        获取点在eps邻域内的所有邻居
        
        Args:
            distance_matrix: 距离矩阵
            point_idx: 点的索引
            
        Returns:
            neighbors: 邻居点的索引数组
        """
        distances = distance_matrix[point_idx]
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors
    
    def _expand_cluster(self, X: np.ndarray, distance_matrix: np.ndarray,
                        point_idx: int, neighbors: np.ndarray, 
                        cluster_id: int, labels: np.ndarray,
                        visited: np.ndarray) -> None:
        """
        扩展簇：从核心点开始，将密度可达的点加入簇
        
        Args:
            X: 数据矩阵
            distance_matrix: 距离矩阵
            point_idx: 当前核心点索引
            neighbors: 当前点的邻居
            cluster_id: 当前簇ID
            labels: 标签数组
            visited: 访问标记数组
        """
        # 将当前点加入簇
        labels[point_idx] = cluster_id
        
        # 使用队列进行广度优先搜索
        queue = list(neighbors)
        queue.remove(point_idx)  # 移除自身
        
        while queue:
            current_idx = queue.pop(0)
            
            if not visited[current_idx]:
                visited[current_idx] = True
                current_neighbors = self._get_neighbors(distance_matrix, current_idx)
                
                # 如果当前点也是核心点，将其邻居加入队列
                if len(current_neighbors) >= self.min_samples:
                    for neighbor_idx in current_neighbors:
                        if labels[neighbor_idx] == -1:  # 噪声点
                            labels[neighbor_idx] = cluster_id
                        elif labels[neighbor_idx] == 0:  # 未分类点
                            labels[neighbor_idx] = cluster_id
                            if not visited[neighbor_idx]:
                                queue.append(neighbor_idx)
            else:
                # 已访问的点，如果是噪声点则转为边界点
                if labels[current_idx] == -1:
                    labels[current_idx] = cluster_id
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        训练DBSCAN模型
        
        Args:
            X: 训练数据 (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # 计算距离矩阵
        distance_matrix = self._compute_distance_matrix(X)
        
        # 初始化
        labels = np.zeros(n_samples, dtype=int)  # 0表示未分类
        visited = np.zeros(n_samples, dtype=bool)
        core_points = np.zeros(n_samples, dtype=bool)
        
        # 识别核心点
        for i in range(n_samples):
            neighbors = self._get_neighbors(distance_matrix, i)
            if len(neighbors) >= self.min_samples:
                core_points[i] = True
        
        # 聚类
        cluster_id = 0
        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._get_neighbors(distance_matrix, i)
            
            if len(neighbors) < self.min_samples:
                # 噪声点
                labels[i] = self.NOISE
            else:
                # 核心点，开始新簇
                cluster_id += 1
                self._expand_cluster(X, distance_matrix, i, neighbors, 
                                    cluster_id, labels, visited)
        
        # 存储结果
        self.labels_ = labels
        self.core_sample_indices_ = np.where(core_points)[0]
        self.n_clusters_ = cluster_id
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        训练模型并返回簇标签
        
        Args:
            X: 训练数据
            
        Returns:
            labels: 簇标签 (-1表示噪声点)
        """
        self.fit(X)
        return self.labels_ # type: ignore
    
    def get_n_noise(self) -> int:
        """获取噪声点数量"""
        return np.sum(self.labels_ == self.NOISE)


def compute_core_distances(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    计算每个点到其第k近邻的距离（用于确定eps参数）
    
    Args:
        X: 数据矩阵
        k: 近邻数
        
    Returns:
        core_distances: 每个点的核心距离
    """
    n_samples = X.shape[0]
    core_distances = np.zeros(n_samples)
    
    for i in range(n_samples):
        distances = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
        distances.sort()
        core_distances[i] = distances[k]  # 第k近邻的距离
    
    return core_distances


def suggest_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """
    基于k-距离图建议eps值
    
    Args:
        X: 数据矩阵
        min_samples: min_samples参数
        
    Returns:
        suggested_eps: 建议的eps值
    """
    core_distances = compute_core_distances(X, min_samples)
    core_distances.sort()
    
    # 使用k-距离图的"肘部"作为建议值
    # 这里简单返回排序后距离的中位数
    suggested_eps = np.median(core_distances)
    
    return suggested_eps # type: ignore


if __name__ == "__main__":
    # 测试DBSCAN
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.data.iris import load_iris, normalize
    from src.evaluation.metrics import evaluate_clustering

    X, y = load_iris()
    X_norm, _, _ = normalize(X)

    # 建议eps值
    suggested = suggest_eps(X_norm, min_samples=9)
    print(f"建议的eps值: {suggested:.4f}")

    # 测试DBSCAN
    dbscan = DBSCAN(eps=0.62, min_samples=9)
    labels = dbscan.fit_predict(X_norm)

    print(f"\n=== DBSCAN 聚类结果 ===")
    print(f"簇数量: {dbscan.n_clusters_}")
    print(f"噪声点数量: {dbscan.get_n_noise()}")
    print(f"核心点数量: {len(dbscan.core_sample_indices_)}")  # type: ignore
    
    # 评估指标
    metrics = evaluate_clustering(X_norm, labels, y)
    print(f"\n=== 评估指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"轮廓系数: {metrics['silhouette_score']:.4f}")
    print(f"CH指数: {metrics['calinski_harabasz_score']:.2f}")
    
    # 标签分布
    print(f"\n=== 标签分布 ===")
    print(f"预测标签分布: {np.bincount(labels + 1)}")  # +1因为噪声点为-1
    print(f"真实标签分布: {np.bincount(y)}")