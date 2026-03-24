# 聚类算法对比实验

在鸢尾花（Iris）数据集上实现 K-means 和 DBSCAN 聚类算法的对比实验。

## 项目结构

```
Task3/
├── main.py                 # 主实验程序
├── requirements.txt        # 依赖包
├── README.md              # 项目说明
├── output/                # 输出结果
│   ├── algorithm_comparison.png      # 算法对比图
│   ├── dbscan_best_result.png        # DBSCAN最佳结果
│   ├── dbscan_metrics_heatmap.png    # DBSCAN参数热力图
│   ├── kmeans_best_result.png        # K-means最佳结果
│   ├── kmeans_k2~5.png               # 不同k值的聚类结果
│   └── kmeans_metrics.png            # K-means参数曲线
└── src/                    # 源代码
    ├── __init__.py
    ├── data/              # 数据模块
    │   ├── __init__.py
    │   └── iris.py        # 鸢尾花数据集加载
    ├── algorithms/        # 算法模块
    │   ├── __init__.py
    │   ├── kmeans.py      # K-means算法实现
    │   └── dbscan.py      # DBSCAN算法实现
    ├── evaluation/        # 评估模块
    │   ├── __init__.py
    │   └── metrics.py     # 评估指标实现
    └── visualization/     # 可视化模块
        ├── __init__.py
        └── plot.py        # 可视化函数
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖包：
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0

## 运行实验

```bash
python main.py
```

## 算法实现

### K-means 算法

**核心逻辑：**
1. 初始化质心（支持随机初始化和 K-means++ 初始化）
2. 将每个样本分配到最近的质心
3. 更新质心为每个簇的均值
4. 重复步骤2-3直到收敛

**可调参数：**
- `n_clusters`: 簇数量 k
- `max_iter`: 最大迭代次数
- `tol`: 收敛阈值
- `init`: 初始化方法（'random' 或 'kmeans++'）

### DBSCAN 算法

**核心逻辑：**
1. 计算所有样本对之间的距离
2. 识别核心点（邻域内样本数 ≥ min_samples）
3. 从核心点出发扩展簇（密度可达）
4. 标记噪声点（不属于任何簇）

**可调参数：**
- `eps`: 邻域半径
- `min_samples`: 核心点所需的最小邻域样本数
- `metric`: 距离度量方式（'euclidean' 或 'manhattan'）

## 评估指标

### 1. 准确率（Accuracy）
使用匈牙利算法匹配预测标签与真实标签，计算正确分类的样本比例。

### 2. 轮廓系数（Silhouette Score）
衡量样本与同簇样本的相似度 vs 与其他簇样本的相似度。
- 取值范围：[-1, 1]
- 越大越好，表示聚类效果越好

### 3. Calinski-Harabasz 指数
簇间离散度与簇内离散度的比值。
- 取值范围：[0, +∞)
- 越大越好，表示簇间差异大、簇内紧凑

## 实验结果

### K-means 参数敏感性分析

| k值 | 准确率 | 轮廓系数 | CH指数 | SSE |
|-----|--------|----------|--------|-----|
| 2   | 0.6667 | 0.5802   | 248.90 | 223.73 |
| 3   | **0.8467** | 0.4557 | 238.78 | 141.22 |
| 4   | 0.6933 | 0.4140   | 203.75 | 115.68 |
| 5   | 0.6133 | 0.3566   | 165.76 | 107.67 |

**结论：** k=3 时准确率最高（84.67%），与鸢尾花数据集的真实类别数一致。

### DBSCAN 参数敏感性分析

| eps | min_samples | 准确率 | 轮廓系数 | 簇数 | 噪声点 |
|-----|-------------|--------|----------|------|--------|
| 0.7 | 3           | **0.6800** | 0.6035 | 2    | 6      |
| 0.5 | 4           | 0.6800 | 0.6523 | 2    | 34     |
| 0.6 | 7           | 0.6800 | 0.6470 | 2    | 32     |

**结论：** DBSCAN 在鸢尾花数据集上倾向于将两个重叠的类别（Versicolor 和 Virginica）合并为一个簇，导致只能识别出2个簇。

### 算法对比

| 指标 | K-means (k=3) | DBSCAN (eps=0.7, min_samples=3) |
|------|---------------|----------------------------------|
| 准确率 | **0.8467** | 0.6800 |
| 轮廓系数 | 0.4557 | **0.6035** |
| CH指数 | 238.78 | **287.31** |

**分析：**
- K-means 在准确率上表现更好，因为鸢尾花数据集的三个类别在特征空间中相对分离
- DBSCAN 的轮廓系数和CH指数更高，因为它能识别出更紧凑的簇结构
- DBSCAN 将部分样本识别为噪声点，这在某些应用场景中是有价值的

## 可视化结果

运行程序后，`output/` 目录下会生成以下图表：

1. **kmeans_best_result.png** - K-means 最佳聚类结果（k=3）
2. **kmeans_metrics.png** - K-means 参数-指标变化曲线
3. **kmeans_k2~5.png** - 不同 k 值的聚类结果对比
4. **dbscan_best_result.png** - DBSCAN 最佳聚类结果
5. **dbscan_metrics_heatmap.png** - DBSCAN 参数热力图
6. **algorithm_comparison.png** - 两种算法性能对比柱状图

## 注意事项

- 本项目禁止直接调用 scikit-learn 等完整工具包的聚类算法
- 所有核心算法逻辑均为手动实现
- 仅使用 numpy 进行数值计算，scipy 用于数据加载和匈牙利算法

## 作者

数据科学与工程课程实验