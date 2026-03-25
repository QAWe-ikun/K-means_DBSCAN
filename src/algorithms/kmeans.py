"""
K-means聚类算法实现
核心逻辑：初始化质心 → 分配样本 → 更新质心 → 迭代收敛
"""
import numpy as np
from typing import Tuple, Optional


class KMeans:
    """
    K-means聚类算法

    Parameters:
        n_clusters: 簇的数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        random_state: 随机种子
        init: 初始化方法 ('random' 或 'kmeans++')
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 init: str = 'kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self._rng = None

    def _get_rng(self) -> np.random.RandomState:
        """获取随机数生成器"""
        if self._rng is None:
            self._rng = np.random.RandomState(self.random_state)
        return self._rng

    def _init_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """随机初始化质心"""
        rng = self._get_rng()
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-means++初始化质心"""
        rng = self._get_rng()
        n_samples = X.shape[0]
        centroids = []

        # 随机选择第一个质心
        first_idx = rng.randint(0, n_samples)
        centroids.append(X[first_idx].copy())

        # 选择剩余的质心
        for _ in range(1, self.n_clusters):
            # 计算每个点到最近质心的距离
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for c in centroids:
                    dist = np.sum((X[i] - c) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist

            # 按距离的平方成比例选择下一个质心
            probs = distances / distances.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx].copy())

        return np.array(centroids)

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        将每个样本分配到最近的质心

        Args:
            X: 数据矩阵 (n_samples, n_features)
            centroids: 质心矩阵 (n_clusters, n_features)

        Returns:
            labels: 每个样本的簇标签
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            min_dist = float('inf')
            for j in range(self.n_clusters):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        return labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        更新质心为每个簇的均值

        Args:
            X: 数据矩阵
            labels: 当前簇标签

        Returns:
            new_centroids: 新的质心
        """
        rng = self._get_rng()
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # 如果簇为空，随机重新初始化
                new_centroids[k] = X[rng.randint(0, X.shape[0])]

        return new_centroids

    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray,
                           centroids: np.ndarray) -> float:
        """计算簇内平方和(SSE)"""
        inertia = 0.0
        for i in range(X.shape[0]):
            inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)
        return inertia

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        训练K-means模型

        Args:
            X: 训练数据 (n_samples, n_features)

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        
        # 重置随机数生成器以确保可复现性
        self._rng = np.random.RandomState(self.random_state)

        # 初始化质心
        if self.init == 'kmeans++':
            centroids = self._init_centroids_kmeans_plus_plus(X)
        else:
            centroids = self._init_centroids_random(X)

        labels = np.zeros(X.shape[0], dtype=int)

        # 迭代优化
        for iteration in range(self.max_iter):
            # 分配样本到最近的簇
            new_labels = self._assign_clusters(X, centroids)

            # 更新质心
            new_centroids = self._update_centroids(X, new_labels)

            # 检查收敛
            centroid_shift = np.sum((new_centroids - centroids) ** 2)

            centroids = new_centroids
            labels = new_labels

            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels, centroids)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本的簇标签

        Args:
            X: 样本数据

        Returns:
            labels: 预测的簇标签
        """
        X = np.asarray(X, dtype=np.float64)
        return self._assign_clusters(X, self.cluster_centers_)  # type: ignore

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        训练模型并返回簇标签

        Args:
            X: 训练数据

        Returns:
            labels: 簇标签
        """
        self.fit(X)
        return self.labels_  # type: ignore


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算欧氏距离"""
    return np.sqrt(np.sum((a - b) ** 2))


if __name__ == "__main__":
    # 测试K-means
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.data.iris import load_iris, normalize

    X, y = load_iris()
    X_norm, _, _ = normalize(X)

    # 测试K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_norm)

    print(f"迭代次数: {kmeans.n_iter_}")
    print(f"SSE: {kmeans.inertia_:.4f}")
    print(f"簇中心形状: {kmeans.cluster_centers_.shape}")  # type: ignore
    print(f"标签分布: {np.bincount(labels)}")