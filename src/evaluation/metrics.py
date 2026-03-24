"""
聚类评估指标模块
实现准确率、轮廓系数、Calinski-Harabasz指数
"""
import numpy as np
from typing import Tuple, Dict
from scipy.optimize import linear_sum_assignment


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算聚类准确率（使用匈牙利算法匹配标签）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        accuracy: 准确率 [0, 1]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # 处理噪声点（DBSCAN中标签为-1的点）
    # 将噪声点视为一个单独的类别
    if np.min(y_pred) < 0:
        # 找出噪声点
        noise_mask = y_pred < 0
        if np.any(noise_mask):
            # 将噪声点分配一个新的标签
            max_label = np.max(y_pred)
            y_pred = y_pred.copy()
            y_pred[noise_mask] = max_label + 1
    
    # 获取标签类别
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    
    n_classes_true = len(classes_true)
    n_classes_pred = len(classes_pred)
    
    # 构建混淆矩阵（代价矩阵）
    # 行: 真实类别, 列: 预测类别
    cost_matrix = np.zeros((n_classes_true, n_classes_pred))
    
    for i, true_label in enumerate(classes_true):
        for j, pred_label in enumerate(classes_pred):
            # 计算将真实类别i预测为类别j的样本数
            cost_matrix[i, j] = -np.sum((y_true == true_label) & (y_pred == pred_label))
    
    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 计算正确匹配的样本数
    correct = 0
    for i, j in zip(row_ind, col_ind):
        true_label = classes_true[i]
        pred_label = classes_pred[j]
        correct += np.sum((y_true == true_label) & (y_pred == pred_label))
    
    accuracy = correct / len(y_true)
    return accuracy


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算轮廓系数
    
    轮廓系数衡量样本与同簇样本的相似度 vs 与其他簇样本的相似度
    取值范围: [-1, 1], 越大越好
    
    Args:
        X: 数据矩阵 (n_samples, n_features)
        labels: 簇标签
        
    Returns:
        silhouette: 平均轮廓系数
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    
    n_samples = X.shape[0]
    
    # 处理噪声点（标签为负数）
    noise_mask = labels < 0
    valid_mask = ~noise_mask
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    # 只计算有效样本
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    # 重新映射标签到连续的整数
    unique_labels = np.unique(labels_valid)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels_mapped = np.array([label_map[l] for l in labels_valid])
    
    n_valid = len(labels_valid)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # 计算所有样本对之间的距离
    silhouette_values = np.zeros(n_valid)
    
    for i in range(n_valid):
        # 当前样本的簇
        current_label = labels_mapped[i]
        
        # 计算a(i): 样本i到同簇其他样本的平均距离
        same_cluster_mask = labels_mapped == current_label
        same_cluster_count = np.sum(same_cluster_mask) - 1  # 排除自身
        
        if same_cluster_count == 0:
            # 簇中只有一个样本
            silhouette_values[i] = 0.0
            continue
        
        # 计算到同簇样本的距离
        same_cluster_indices = np.where(same_cluster_mask)[0]
        a_i = np.mean([np.sqrt(np.sum((X_valid[i] - X_valid[j]) ** 2)) 
                       for j in same_cluster_indices if j != i])
        
        # 计算b(i): 样本i到其他簇的最小平均距离
        b_i = float('inf')
        for label in range(n_clusters):
            if label == current_label:
                continue
            other_cluster_mask = labels_mapped == label
            other_cluster_indices = np.where(other_cluster_mask)[0]
            if len(other_cluster_indices) > 0:
                avg_dist = np.mean([np.sqrt(np.sum((X_valid[i] - X_valid[j]) ** 2)) 
                                   for j in other_cluster_indices])
                b_i = min(b_i, avg_dist) # type: ignore
        
        # 计算轮廓系数
        if b_i == float('inf'):
            silhouette_values[i] = 0.0
        else:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i) # type: ignore
    
    return np.mean(silhouette_values) # type: ignore


def silhouette_score_optimized(X: np.ndarray, labels: np.ndarray) -> float:
    """
    优化版本的轮廓系数计算（使用向量化操作）
    
    Args:
        X: 数据矩阵 (n_samples, n_features)
        labels: 簇标签
        
    Returns:
        silhouette: 平均轮廓系数
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    
    n_samples = X.shape[0]
    
    # 处理噪声点
    noise_mask = labels < 0
    valid_mask = ~noise_mask
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    unique_labels = np.unique(labels_valid)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    n_valid = len(labels_valid)
    
    # 计算距离矩阵（向量化）
    # 使用广播机制: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = np.sum(X_valid ** 2, axis=1)
    distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X_valid, X_valid.T)
    distances_sq = np.maximum(distances_sq, 0)  # 避免数值误差导致的负值
    distances = np.sqrt(distances_sq)
    
    silhouette_values = np.zeros(n_valid)
    
    for i in range(n_valid):
        current_label = labels_valid[i]
        
        # 同簇样本
        same_cluster_mask = labels_valid == current_label
        same_cluster_count = np.sum(same_cluster_mask) - 1
        
        if same_cluster_count == 0:
            silhouette_values[i] = 0.0
            continue
        
        # a(i): 到同簇样本的平均距离
        a_i = (np.sum(distances[i, same_cluster_mask]) - 0) / same_cluster_count  # 减去自身距离0
        
        # b(i): 到其他簇的最小平均距离
        b_i = float('inf')
        for label in unique_labels:
            if label == current_label:
                continue
            other_cluster_mask = labels_valid == label
            if np.sum(other_cluster_mask) > 0:
                avg_dist = np.mean(distances[i, other_cluster_mask])
                b_i = min(b_i, avg_dist)
        
        if b_i == float('inf'):
            silhouette_values[i] = 0.0
        else:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(silhouette_values) # type: ignore


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算Calinski-Harabasz指数
    
    CH指数 = (BG / (k-1)) / (WG / (n-k))
    其中:
    - BG: 簇间离散度（Between-Group Dispersion）
    - WG: 簇内离散度（Within-Group Dispersion）
    - k: 簇数量
    - n: 样本数量
    
    取值范围: [0, +∞), 越大越好
    
    Args:
        X: 数据矩阵 (n_samples, n_features)
        labels: 簇标签
        
    Returns:
        ch_score: Calinski-Harabasz指数
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    
    # 处理噪声点
    noise_mask = labels < 0
    valid_mask = ~noise_mask
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_samples, n_features = X_valid.shape
    unique_labels = np.unique(labels_valid)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # 计算全局均值
    global_mean = np.mean(X_valid, axis=0)
    
    # 计算簇间离散度 (BG)
    bg = 0.0
    for label in unique_labels:
        cluster_mask = labels_valid == label
        cluster_size = np.sum(cluster_mask)
        cluster_mean = np.mean(X_valid[cluster_mask], axis=0)
        bg += cluster_size * np.sum((cluster_mean - global_mean) ** 2)
    
    # 计算簇内离散度 (WG)
    wg = 0.0
    for label in unique_labels:
        cluster_mask = labels_valid == label
        cluster_points = X_valid[cluster_mask]
        cluster_mean = np.mean(cluster_points, axis=0)
        wg += np.sum((cluster_points - cluster_mean) ** 2)
    
    # 计算CH指数
    if wg == 0:
        return 0.0
    
    ch_score = (bg / (n_clusters - 1)) / (wg / (n_samples - n_clusters))
    
    return ch_score


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, 
                       y_true: np.ndarray = None) -> Dict[str, float]: # type: ignore
    """
    综合评估聚类结果
    
    Args:
        X: 数据矩阵
        labels: 预测的簇标签
        y_true: 真实标签（可选，用于计算准确率）
        
    Returns:
        metrics: 包含各评估指标的字典
    """
    metrics = {}
    
    # 轮廓系数
    metrics['silhouette_score'] = silhouette_score_optimized(X, labels)
    
    # Calinski-Harabasz指数
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    
    # 簇数量
    unique_labels = np.unique(labels[labels >= 0])  # 排除噪声点
    metrics['n_clusters'] = len(unique_labels)
    
    # 噪声点数量
    metrics['n_noise'] = np.sum(labels < 0)
    
    # 如果有真实标签，计算准确率
    if y_true is not None:
        metrics['accuracy'] = accuracy_score(y_true, labels)
    
    return metrics


if __name__ == "__main__":
    # 测试评估指标
    import sys
    sys.path.append('..')
    from data.iris import load_iris, normalize
    from algorithms.kmeans import KMeans
    from algorithms.dbscan import DBSCAN
    
    X, y = load_iris()
    X_norm, _, _ = normalize(X)
    
    print("=" * 50)
    print("K-means聚类评估")
    print("=" * 50)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_norm)
    
    metrics_kmeans = evaluate_clustering(X_norm, labels_kmeans, y)
    for key, value in metrics_kmeans.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("DBSCAN聚类评估")
    print("=" * 50)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_norm)
    
    metrics_dbscan = evaluate_clustering(X_norm, labels_dbscan, y)
    for key, value in metrics_dbscan.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")