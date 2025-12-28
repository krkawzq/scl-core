# Batch 8 内核

> Batch 8: 聚类、标记基因、矩阵合并和质量评估

## 概述

本批次包含以下功能的内核：

- **Louvain 聚类** - 多层级社区检测算法
- **标记基因选择** - 基因标记识别和特异性评分
- **矩阵合并** - 水平和垂直矩阵拼接
- **质量评估** - 聚类和整合质量评估指标

## 文件列表

| 文件 | 说明 | 主要 API |
|------|------|----------|
| [louvain.hpp](./louvain) | Louvain 社区检测 | `cluster`, `compute_modularity`, `community_sizes`, `get_community_members` |
| [markers.hpp](./markers) | 标记基因选择 | `find_markers`, `specificity_score` |
| [merge.hpp](./merge) | 矩阵合并操作 | `vstack`, `hstack` |
| [metrics.hpp](./metrics) | 质量评估指标 | `silhouette_score`, `adjusted_rand_index`, `normalized_mutual_information`, `graph_connectivity`, `batch_entropy`, `lisi` |

## 快速开始

### Louvain 聚类

```cpp
#include "scl/kernel/louvain.hpp"

Sparse<Real, true> adjacency = /* 邻接矩阵 */;
Array<Index> labels(n_nodes);

scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.0);
```

### 标记基因选择

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* 表达矩阵 */;
Array<Index> cluster_labels = /* 聚类标签 */;
Index* marker_genes = /* 分配内存 */;
Real* marker_scores = /* 分配内存 */;

scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers = 50
);
```

### 矩阵合并

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* 第一个矩阵 */;
Sparse<Real, true> matrix2 = /* 第二个矩阵 */;

// 垂直堆叠
auto vstacked = scl::kernel::merge::vstack(matrix1, matrix2);

// 水平堆叠
auto hstacked = scl::kernel::merge::hstack(matrix1, matrix2);
```

### 质量评估

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* 成对距离 */;
Array<Index> labels = /* 聚类标签 */;

// Silhouette 分数
Real score = scl::kernel::metrics::silhouette_score(distances, labels);

// 调整兰德指数
Array<Index> labels1 = /* 第一个聚类 */;
Array<Index> labels2 = /* 第二个聚类 */;
Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);
```

## 相关内容

- [Leiden 聚类](./leiden) - 替代社区检测算法
- [近邻](./neighbors) - 用于聚类的 KNN 图构建
- [统计](./statistics) - 统计检验和分析

