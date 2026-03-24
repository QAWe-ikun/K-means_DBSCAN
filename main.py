"""
聚类算法对比实验主程序
在鸢尾花数据集上实现K-means和DBSCAN算法对比实验
"""
import os
import numpy as np
from src.data.iris import load_iris, normalize
from src.algorithms.kmeans import KMeans
from src.algorithms.dbscan import DBSCAN, suggest_eps
from src.evaluation.metrics import evaluate_clustering
from src.visualization.plot import (plot_clusters_2d, plot_kmeans_metrics,
                                     plot_dbscan_metrics, plot_comparison)


def experiment_kmeans(X: np.ndarray, y: np.ndarray, output_dir: str):
    """
    K-means聚类实验：调整k值观察聚类结果变化

    Args:
        X: 标准化后的特征矩阵
        y: 真实标签
        output_dir: 输出目录
    """
    print("=" * 60)
    print("K-means 聚类实验")
    print("=" * 60)

    # 测试不同的k值
    k_range = range(2, 6)
    results = []

    # 存储各指标用于绘图
    metrics_dict = {
        '准确率': [],
        '轮廓系数': [],
        'CH指数': [],
        'SSE': []
    }

    for k in k_range:
        print(f"\n--- k = {k} ---")

        kmeans = KMeans(n_clusters=k, random_state=42, init='kmeans++')
        labels = kmeans.fit_predict(X)

        metrics = evaluate_clustering(X, labels, y)
        metrics['sse'] = kmeans.inertia_  # type: ignore

        results.append({
            'k': k,
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'metrics': metrics,
            'n_iter': kmeans.n_iter_
        })

        # 打印结果
        print(f"迭代次数: {kmeans.n_iter_}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"轮廓系数: {metrics['silhouette_score']:.4f}")
        print(f"CH指数: {metrics['calinski_harabasz_score']:.4f}")
        print(f"SSE: {metrics['sse']:.4f}")
        print(f"簇大小分布: {np.bincount(labels)}")

        # 收集指标用于绘图
        metrics_dict['准确率'].append(metrics['accuracy'])
        metrics_dict['轮廓系数'].append(metrics['silhouette_score'])
        metrics_dict['CH指数'].append(metrics['calinski_harabasz_score'])
        metrics_dict['SSE'].append(metrics['sse'])

    # 绘制k=3时的聚类结果（最佳k值）
    best_idx = np.argmax([r['metrics']['accuracy'] for r in results])
    best_result = results[best_idx]

    fig = plot_clusters_2d(X, best_result['labels'],
                          title=f"K-means 聚类结果 (k={best_result['k']})",
                          centers=best_result['centers'],
                          true_labels=y)
    fig.savefig(os.path.join(output_dir, 'kmeans_best_result.png'), dpi=150)

    # 绘制参数-指标变化曲线
    fig = plot_kmeans_metrics(list(k_range), metrics_dict)
    fig.savefig(os.path.join(output_dir, 'kmeans_metrics.png'), dpi=150)

    # 绘制每个k值的聚类结果
    for result in results:
        fig = plot_clusters_2d(X, result['labels'],
                              title=f"K-means 聚类结果 (k={result['k']})",
                              centers=result['centers'],
                              true_labels=y)
        fig.savefig(os.path.join(output_dir, f"kmeans_k{result['k']}.png"), dpi=150)

    return results


def experiment_dbscan(X: np.ndarray, y: np.ndarray, output_dir: str):
    """
    DBSCAN聚类实验：调整eps和min_samples观察聚类结果变化

    Args:
        X: 标准化后的特征矩阵
        y: 真实标签
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("DBSCAN 聚类实验")
    print("=" * 60)

    # 建议eps值
    suggested_eps = suggest_eps(X, min_samples=5)
    print(f"\n建议的eps值: {suggested_eps:.4f}")

    # 测试不同的参数组合
    eps_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    min_samples_range = [3, 4, 5, 6, 7]

    results = []

    # 存储指标矩阵用于热力图
    accuracy_matrix = np.zeros((len(min_samples_range), len(eps_range)))
    silhouette_matrix = np.zeros((len(min_samples_range), len(eps_range)))
    ch_matrix = np.zeros((len(min_samples_range), len(eps_range)))
    n_clusters_matrix = np.zeros((len(min_samples_range), len(eps_range)))

    for i, min_samples in enumerate(min_samples_range):
        for j, eps in enumerate(eps_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            # 如果所有点都是噪声点或只有一个簇，跳过
            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) < 2:
                accuracy_matrix[i, j] = 0
                silhouette_matrix[i, j] = 0
                ch_matrix[i, j] = 0
                n_clusters_matrix[i, j] = len(unique_labels)
                continue

            metrics = evaluate_clustering(X, labels, y)

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'metrics': metrics,
                'n_clusters': dbscan.n_clusters_,
                'n_noise': dbscan.get_n_noise()
            })

            accuracy_matrix[i, j] = metrics['accuracy']
            silhouette_matrix[i, j] = metrics['silhouette_score']
            ch_matrix[i, j] = metrics['calinski_harabasz_score']
            n_clusters_matrix[i, j] = metrics['n_clusters']

    # 打印部分关键结果
    print("\n关键参数组合结果:")
    print("-" * 80)
    print(f"{'eps':<8} {'min_samples':<12} {'准确率':<10} {'轮廓系数':<10} {'CH指数':<12} {'簇数':<6} {'噪声点':<8}")
    print("-" * 80)

    for result in results:
        print(f"{result['eps']:<8.2f} {result['min_samples']:<12} "
              f"{result['metrics']['accuracy']:<10.4f} "
              f"{result['metrics']['silhouette_score']:<10.4f} "
              f"{result['metrics']['calinski_harabasz_score']:<12.2f} "
              f"{result['n_clusters']:<6} {result['n_noise']:<8}")

    # 找到最佳参数组合（基于准确率）
    valid_results = [r for r in results if r['metrics']['n_clusters'] >= 2]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['metrics']['accuracy'])

        print(f"\n最佳参数组合: eps={best_result['eps']}, min_samples={best_result['min_samples']}")
        print(f"准确率: {best_result['metrics']['accuracy']:.4f}")

        # 绘制最佳结果
        fig = plot_clusters_2d(X, best_result['labels'],
                              title=f"DBSCAN 聚类结果 (eps={best_result['eps']}, min_samples={best_result['min_samples']})",
                              true_labels=y)
        fig.savefig(os.path.join(output_dir, 'dbscan_best_result.png'), dpi=150)

    # 绘制参数热力图
    metrics_matrix = {
        '准确率': accuracy_matrix,
        '轮廓系数': silhouette_matrix,
        'CH指数': ch_matrix,
        '簇数量': n_clusters_matrix
    }

    fig = plot_dbscan_metrics(eps_range, min_samples_range, metrics_matrix)
    fig.savefig(os.path.join(output_dir, 'dbscan_metrics_heatmap.png'), dpi=150)

    # 返回最佳结果（基于准确率）
    best_result = max(valid_results, key=lambda r: r['metrics']['accuracy']) if valid_results else None
    return results, best_result


def experiment_dbscan_optimized(X: np.ndarray, y: np.ndarray, output_dir: str):
    """
    优化的DBSCAN聚类实验：使用花瓣特征提升准确率

    Args:
        X: 标准化后的特征矩阵
        y: 真实标签
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("DBSCAN 优化实验（使用花瓣特征）")
    print("=" * 60)

    # 只使用花瓣特征（petal_length, petal_width），这两个特征区分度更高
    X_petal = X[:, 2:4]

    print("\n使用花瓣特征（petal_length, petal_width）进行聚类")
    print("原因：花瓣特征对三个类别的区分度更高")

    # 搜索最佳参数
    best_result = None
    best_acc = 0

    for eps in np.arange(0.3, 0.8, 0.05):
        for min_samples in range(3, 8):
            dbscan = DBSCAN(eps=round(eps, 2), min_samples=min_samples) # type: ignore
            labels = dbscan.fit_predict(X_petal)

            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) < 2:
                continue

            metrics = evaluate_clustering(X_petal, labels, y)

            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_result = {
                    'eps': round(eps, 2),
                    'min_samples': min_samples,
                    'labels': labels,
                    'metrics': metrics,
                    'n_clusters': dbscan.n_clusters_,
                    'n_noise': dbscan.get_n_noise()
                }

    if best_result:
        print(f"\n优化后最佳参数: eps={best_result['eps']}, min_samples={best_result['min_samples']}")
        print(f"准确率: {best_result['metrics']['accuracy']:.4f}")
        print(f"轮廓系数: {best_result['metrics']['silhouette_score']:.4f}")
        print(f"CH指数: {best_result['metrics']['calinski_harabasz_score']:.2f}")
        print(f"簇数量: {best_result['n_clusters']}")
        print(f"噪声点: {best_result['n_noise']}")

        # 绘制优化后的结果
        fig = plot_clusters_2d(X_petal, best_result['labels'],
                              title=f"DBSCAN 优化结果 (花瓣特征, eps={best_result['eps']})",
                              true_labels=y)
        fig.savefig(os.path.join(output_dir, 'dbscan_optimized_result.png'), dpi=150)

    return best_result


def compare_algorithms(kmeans_results: list, dbscan_result: dict,
                       dbscan_optimized: dict, X: np.ndarray, y: np.ndarray, 
                       output_dir: str):
    """
    对比K-means和DBSCAN的性能

    Args:
        kmeans_results: K-means实验结果
        dbscan_result: DBSCAN实验结果
        dbscan_optimized: 优化后的DBSCAN结果
        X: 特征矩阵
        y: 真实标签
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("算法性能对比")
    print("=" * 60)

    # 选择K-means最佳结果（k=3，因为鸢尾花有3个类别）
    kmeans_best = None
    for r in kmeans_results:
        if r['k'] == 3:
            kmeans_best = r
            break
    if kmeans_best is None:
        kmeans_best = max(kmeans_results, key=lambda r: r['metrics']['accuracy'])

    print("\nK-means 最佳结果 (k=3):")
    print(f"  准确率: {kmeans_best['metrics']['accuracy']:.4f}")
    print(f"  轮廓系数: {kmeans_best['metrics']['silhouette_score']:.4f}")
    print(f"  CH指数: {kmeans_best['metrics']['calinski_harabasz_score']:.2f}")

    if dbscan_result:
        print(f"\nDBSCAN 最佳结果 (eps={dbscan_result['eps']}, min_samples={dbscan_result['min_samples']}):")
        print(f"  准确率: {dbscan_result['metrics']['accuracy']:.4f}")
        print(f"  轮廓系数: {dbscan_result['metrics']['silhouette_score']:.4f}")
        print(f"  CH指数: {dbscan_result['metrics']['calinski_harabasz_score']:.2f}")
        print(f"  簇数量: {dbscan_result['n_clusters']}")
        print(f"  噪声点: {dbscan_result['n_noise']}")

    if dbscan_optimized:
        print(f"\nDBSCAN 优化结果 (花瓣特征, eps={dbscan_optimized['eps']}):")
        print(f"  准确率: {dbscan_optimized['metrics']['accuracy']:.4f}")
        print(f"  轮廓系数: {dbscan_optimized['metrics']['silhouette_score']:.4f}")
        print(f"  CH指数: {dbscan_optimized['metrics']['calinski_harabasz_score']:.2f}")
        print(f"  簇数量: {dbscan_optimized['n_clusters']}")
        print(f"  噪声点: {dbscan_optimized['n_noise']}")

    # 绘制对比图（使用优化后的DBSCAN）
    if dbscan_optimized:
        fig = plot_comparison(kmeans_best['metrics'], dbscan_optimized['metrics'])
        fig.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=150)

    # 绘制并排对比图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # K-means结果
    from src.visualization.plot import pca_reduce
    X_2d = pca_reduce(X, 2)

    # K-means
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 3))  # type: ignore
    for i in range(3):
        mask = kmeans_best['labels'] == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], s=50, alpha=0.7)
    ax.set_title(f"K-means (k=3)\n准确率: {kmeans_best['metrics']['accuracy']:.4f}")
    ax.set_xlabel('主成分 1')
    ax.set_ylabel('主成分 2')

    # DBSCAN（4特征）
    if dbscan_result:
        ax = axes[1]
        unique_labels = np.unique(dbscan_result['labels'])
        dbscan_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))  # type: ignore
        for i, label in enumerate(unique_labels):
            mask = dbscan_result['labels'] == label
            if label < 0:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', s=50)
            else:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[dbscan_colors[i]], s=50, alpha=0.7)
        ax.set_title(f"DBSCAN (4特征)\n准确率: {dbscan_result['metrics']['accuracy']:.4f}")
        ax.set_xlabel('主成分 1')
        ax.set_ylabel('主成分 2')

    # DBSCAN（花瓣特征）
    if dbscan_optimized:
        ax = axes[2]
        X_petal_2d = X[:, 2:4]  # 花瓣特征已经是2D
        unique_labels = np.unique(dbscan_optimized['labels'])
        dbscan_opt_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # type: ignore
        for i, label in enumerate(unique_labels):
            mask = dbscan_optimized['labels'] == label
            if label < 0:
                ax.scatter(X_petal_2d[mask, 0], X_petal_2d[mask, 1], c='black', marker='x', s=50)
            else:
                ax.scatter(X_petal_2d[mask, 0], X_petal_2d[mask, 1], c=[dbscan_opt_colors[i]], s=50, alpha=0.7)
        ax.set_title(f"DBSCAN (花瓣特征)\n准确率: {dbscan_optimized['metrics']['accuracy']:.4f}")
        ax.set_xlabel('花瓣长度')
        ax.set_ylabel('花瓣宽度')

    # 真实标签
    ax = axes[3]
    true_colors = plt.cm.Set1(np.linspace(0, 1, 3))  # type: ignore
    label_names = ['Setosa', 'Versicolor', 'Virginica']
    for i in range(3):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[true_colors[i]], s=50,
                  label=label_names[i], alpha=0.7)
    ax.set_title('真实标签')
    ax.set_xlabel('主成分 1')
    ax.set_ylabel('主成分 2')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_comparison.png'), dpi=150)

    return kmeans_best, dbscan_result, dbscan_optimized


def main():
    """主函数"""
    import matplotlib.pyplot as plt
    
    # 设置全局随机种子，确保结果可复现
    np.random.seed(42)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("鸢尾花数据集聚类算法对比实验")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    X, y = load_iris()
    print(f"数据形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 数据标准化
    print("\n数据标准化...")
    X_norm, mean, std = normalize(X)
    print(f"标准化后均值: {mean}")
    print(f"标准化后标准差: {std}")

    # K-means实验
    kmeans_results = experiment_kmeans(X_norm, y, output_dir)

    # DBSCAN实验
    dbscan_results, dbscan_best = experiment_dbscan(X_norm, y, output_dir)

    # DBSCAN优化实验
    dbscan_optimized = experiment_dbscan_optimized(X_norm, y, output_dir)

    # 算法对比
    compare_algorithms(kmeans_results, dbscan_best, dbscan_optimized, X_norm, y, output_dir) # type: ignore

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"\n可视化结果: {output_dir}")

    # 显示所有图表
    plt.show()


if __name__ == "__main__":
    main()