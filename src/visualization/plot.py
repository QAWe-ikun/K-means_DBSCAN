"""
聚类可视化模块
实现聚类结果的2D可视化和参数-指标变化曲线
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Tuple

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def pca_reduce(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    使用PCA降维到指定维度
    
    Args:
        X: 数据矩阵 (n_samples, n_features)
        n_components: 目标维度
        
    Returns:
        X_reduced: 降维后的数据 (n_samples, n_components)
    """
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    
    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 按特征值降序排列
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # 选择前n_components个主成分
    principal_components = eigenvectors_sorted[:, :n_components]
    
    # 投影
    X_reduced = np.dot(X_centered, principal_components)
    
    return X_reduced


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, 
                     title: str = "聚类结果",
                     centers: np.ndarray = None, # type: ignore
                     true_labels: np.ndarray = None, # type: ignore
                     figsize: Tuple[int, int] = (10, 8),
                     save_path: Optional[str] = None) -> Figure:
    """
    绘制2D聚类结果散点图
    
    Args:
        X: 数据矩阵 (如果维度>2，会自动PCA降维)
        labels: 簇标签
        title: 图表标题
        centers: 簇中心点（可选）
        true_labels: 真实标签（可选，用于对比）
        figsize: 图表大小
        save_path: 保存路径（可选）
        
    Returns:
        fig: matplotlib Figure对象
    """
    # 如果维度>2，使用PCA降维
    if X.shape[1] > 2:
        X_plot = pca_reduce(X, n_components=2)
    else:
        X_plot = X.copy()
    
    # 创建图表
    if true_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # 定义颜色
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # 排除噪声点
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters + 1, 3))) # type: ignore
    
    # 绘制聚类结果
    for i, label in enumerate(unique_labels):
        if label < 0:
            # 噪声点用黑色x标记
            mask = labels == label
            ax1.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                       c='black', marker='x', s=50, label=f'噪声点')
        else:
            mask = labels == label
            ax1.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                       c=[colors[i]], s=50, label=f'簇 {label}', alpha=0.7)
    
    # 绘制簇中心
    if centers is not None:
        if centers.shape[1] > 2:
            centers_plot = pca_reduce(np.vstack([X, centers]), n_components=2)[-len(centers):]
        else:
            centers_plot = centers
        ax1.scatter(centers_plot[:, 0], centers_plot[:, 1], 
                   c='red', marker='*', s=200, edgecolors='black', 
                   linewidths=1.5, label='簇中心')
    
    ax1.set_xlabel('主成分 1')
    ax1.set_ylabel('主成分 2')
    ax1.set_title(title)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 如果有真实标签，绘制对比图
    if true_labels is not None:
        unique_true = np.unique(true_labels)
        true_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_true))) # type: ignore
        
        for i, label in enumerate(unique_true):
            mask = true_labels == label
            label_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            name = label_names.get(label, f'类别 {label}')
            ax2.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                       c=[true_colors[i]], s=50, label=name, alpha=0.7)
        
        ax2.set_xlabel('主成分 1')
        ax2.set_ylabel('主成分 2')
        ax2.set_title('真实标签')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_kmeans_metrics(k_range: List[int], 
                        metrics_dict: Dict[str, List[float]],
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None) -> Figure:
    """
    绘制K-means参数-指标变化曲线
    
    Args:
        k_range: k值范围
        metrics_dict: 指标字典，如 {'accuracy': [...], 'silhouette': [...], ...}
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        fig: matplotlib Figure对象
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 4:
            break
        ax = axes[i]
        ax.plot(k_range, values, marker='o', color=colors[i], 
                linewidth=2, markersize=8)
        ax.set_xlabel('簇数 k')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs k')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_range)
    
    # 隐藏多余的子图
    for i in range(len(metrics_dict), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('K-means 参数敏感性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_dbscan_metrics(eps_range: List[float], 
                        min_samples_range: List[int],
                        metrics_matrix: Dict[str, np.ndarray],
                        figsize: Tuple[int, int] = (14, 10),
                        save_path: Optional[str] = None) -> Figure:
    """
    绘制DBSCAN参数-指标热力图
    
    Args:
        eps_range: eps值范围
        min_samples_range: min_samples值范围
        metrics_matrix: 指标矩阵字典，每个指标为2D数组
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        fig: matplotlib Figure对象
    """
    n_metrics = len(metrics_matrix)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (metric_name, matrix) in enumerate(metrics_matrix.items()):
        if i >= 4:
            break
        ax = axes[i]
        
        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        
        ax.set_xticks(range(len(eps_range)))
        ax.set_xticklabels([f'{e:.2f}' for e in eps_range], rotation=45)
        ax.set_yticks(range(len(min_samples_range)))
        ax.set_yticklabels(min_samples_range)
        
        ax.set_xlabel('eps')
        ax.set_ylabel('min_samples')
        ax.set_title(metric_name)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 在每个格子中显示数值
        for row in range(len(min_samples_range)):
            for col in range(len(eps_range)):
                value = matrix[row, col]
                text_color = 'white' if im.norm(value) > 0.5 else 'black'
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    # 隐藏多余的子图
    for i in range(len(metrics_matrix), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('DBSCAN 参数敏感性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(kmeans_metrics: Dict[str, float],
                    dbscan_metrics: Dict[str, float],
                    figsize: Tuple[int, int] = (12, 6),
                    save_path: Optional[str] = None) -> Figure:
    """
    绘制K-means和DBSCAN性能对比柱状图（双Y轴）

    Args:
        kmeans_metrics: K-means评估指标
        dbscan_metrics: DBSCAN评估指标
        figsize: 图表大小
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure对象
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 左Y轴：准确率和轮廓系数（0-1范围）
    x1 = np.array([0, 1])
    width = 0.35
    
    kmeans_small = [kmeans_metrics.get('accuracy', 0), 
                    kmeans_metrics.get('silhouette_score', 0)]
    dbscan_small = [dbscan_metrics.get('accuracy', 0), 
                    dbscan_metrics.get('silhouette_score', 0)]
    
    bars1 = ax1.bar(x1 - width/2, kmeans_small, width, label='K-means', 
                    color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x1 + width/2, dbscan_small, width, label='DBSCAN', 
                    color='#ff7f0e', alpha=0.8)
    
    ax1.set_ylabel('准确率 / 轮廓系数', color='#1f77b4', fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xticks([0, 1, 2.5])
    ax1.set_xticklabels(['准确率', '轮廓系数', 'CH指数'])
    
    # 在柱子上显示数值
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#1f77b4')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#ff7f0e')
    
    # 右Y轴：CH指数
    ax2 = ax1.twinx()
    x2 = np.array([2.5])
    
    kmeans_ch = kmeans_metrics.get('calinski_harabasz_score', 0)
    dbscan_ch = dbscan_metrics.get('calinski_harabasz_score', 0)
    
    bars3 = ax2.bar(x2 - width/2, [kmeans_ch], width, color='#1f77b4', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, [dbscan_ch], width, color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('CH指数', color='#2ca02c', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    # 在柱子上显示数值
    for bar, val in zip(bars3, [kmeans_ch]):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#1f77b4')
    
    for bar, val in zip(bars4, [dbscan_ch]):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#ff7f0e')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='K-means'),
        Patch(facecolor='#ff7f0e', alpha=0.8, label='DBSCAN')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    ax1.set_title('K-means vs DBSCAN 性能对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # 测试可视化模块
    import sys
    sys.path.append('..')
    from data.iris import load_iris, normalize
    from algorithms.kmeans import KMeans
    from algorithms.dbscan import DBSCAN
    
    X, y = load_iris()
    X_norm, _, _ = normalize(X)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_norm)
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_norm)
    
    # 绘制聚类结果
    fig1 = plot_clusters_2d(X_norm, labels_kmeans, 
                           title="K-means 聚类结果 (k=3)",
                           centers=kmeans.cluster_centers_, # type: ignore
                           true_labels=y)
    
    fig2 = plot_clusters_2d(X_norm, labels_dbscan, 
                           title="DBSCAN 聚类结果 (eps=0.5, min_samples=5)",
                           true_labels=y)
    
    plt.show()