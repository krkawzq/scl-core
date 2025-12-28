# 聚类度量

用于评估分析结果的聚类和整合质量度量。

## 概述

`metrics` 模块提供全面的评估度量：

- **轮廓分数**: 测量聚类质量和分离度
- **ARI/NMI**: 比较聚类（调整兰德指数、归一化互信息）
- **图连通性**: 测量图中的聚类连通性
- **批次混合**: 评估批次校正质量（LISI、批次熵）

所有操作都：
- 在适用时并行化
- 统计严谨
- 内存高效

## 聚类质量度量

### silhouette_score

计算所有样本的平均轮廓系数。

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* 成对距离矩阵 */;
Array<const Index> labels = /* 聚类分配 */;

Real score = scl::kernel::metrics::silhouette_score(distances, labels);
```

**参数:**
- `distances` [in] - 成对距离矩阵（细胞 x 细胞，CSR）
- `labels` [in] - 每个细胞的聚类分配

**返回:**
- 平均轮廓分数，范围 [-1, 1]
- 1 表示密集、分离良好的聚类
- -1 表示错误的聚类

**前置条件:**
- `distances.rows() == labels.len`
- 至少 2 个细胞和 2 个聚类才能得到有意义的结果
- 距离值应为非负

**后置条件:**
- 如果少于 2 个细胞或聚类，返回 0
- 单例聚类被排除在计算之外

**复杂度:**
- 时间: O(n * nnz_per_row * n_clusters)
- 空间: O(n_clusters) 辅助空间

**线程安全:** 不安全 - 顺序实现

### silhouette_samples

计算每个样本的轮廓系数。

```cpp
Array<Real> scores(n_cells);
scl::kernel::metrics::silhouette_samples(distances, labels, scores);
```

**参数:**
- `distances` [in] - 成对距离矩阵
- `labels` [in] - 聚类分配
- `scores` [out] - 每个样本的轮廓分数

**前置条件:**
- `distances.rows() == labels.len == scores.len`
- 至少 2 个细胞和 2 个聚类

**后置条件:**
- `scores[i]` = 细胞 i 的轮廓，范围 [-1, 1]
- 单例聚类中的细胞分数为 0

**复杂度:**
- 时间: O(n * nnz_per_row * n_clusters)
- 空间: O(n_clusters * n_threads) 用于线程本地缓冲区

**线程安全:** 安全 - 使用 WorkspacePool 按细胞并行化

## 聚类比较

### adjusted_rand_index

计算两个聚类之间的调整兰德指数。

```cpp
Array<const Index> labels1 = /* 第一个聚类 */;
Array<const Index> labels2 = /* 第二个聚类 */;

Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);
```

**参数:**
- `labels1` [in] - 第一个聚类分配
- `labels2` [in] - 第二个聚类分配

**返回:**
- ARI 分数，范围 [-1, 1]
- 1 表示相同的聚类
- 0 表示随机标记

**前置条件:**
- `labels1.len == labels2.len`
- 标签必须是非负整数

**后置条件:**
- 对于相同的聚类返回 1.0
- 对于随机聚类平均返回 0.0

**复杂度:**
- 时间: O(n + n_clusters1 * n_clusters2)
- 空间: O(n_clusters1 * n_clusters2) 用于列联表

**线程安全:** 不安全 - 顺序实现

### normalized_mutual_information

计算两个聚类之间的归一化互信息。

```cpp
Real nmi = scl::kernel::metrics::normalized_mutual_information(labels1, labels2);
```

**参数:**
- `labels1` [in] - 第一个聚类分配
- `labels2` [in] - 第二个聚类分配

**返回:**
- NMI 分数，范围 [0, 1]
- 1 表示完全一致

**前置条件:**
- `labels1.len == labels2.len`
- 标签必须是非负整数

**后置条件:**
- 对于相同的聚类返回 1.0
- 对于独立的聚类返回 0.0

**复杂度:**
- 时间: O(n + n_clusters1 * n_clusters2)
- 空间: O(n_clusters1 * n_clusters2)

**线程安全:** 不安全 - 顺序实现

## 批次整合度量

### batch_entropy

计算每个细胞邻域中批次分布的归一化熵。

```cpp
Sparse<Index, true> neighbors = /* KNN 图 */;
Array<const Index> batch_labels = /* 批次分配 */;
Array<Real> entropy_scores(n_cells);

scl::kernel::metrics::batch_entropy(neighbors, batch_labels, entropy_scores);
```

**参数:**
- `neighbors` [in] - KNN 图（细胞 x 细胞，CSR）
- `batch_labels` [in] - 每个细胞的批次分配
- `entropy_scores` [out] - 每个细胞的归一化熵

**前置条件:**
- `neighbors.rows() == batch_labels.len == entropy_scores.len`
- 批次标签必须是非负整数

**后置条件:**
- `entropy_scores[i]` 在 [0, 1] 范围内
- 1 表示完美的批次混合（均匀分布）
- 0 表示邻域中只有单个批次

**复杂度:**
- 时间: O(n * k)，其中 k = 每个细胞的平均邻居数
- 空间: O(n_batches * n_threads) 用于线程本地计数器

**线程安全:** 安全 - 使用 WorkspacePool 并行化

### lisi

计算局部逆辛普森指数以测量标签多样性。

```cpp
Array<Real> lisi_scores(n_cells);
scl::kernel::metrics::lisi(neighbors, labels, lisi_scores);
```

**参数:**
- `neighbors` [in] - KNN 图
- `labels` [in] - 标签分配（批次或细胞类型）
- `lisi_scores` [out] - 每个细胞的 LISI 分数

**前置条件:**
- `neighbors.rows() == labels.len == lisi_scores.len`
- 标签必须是非负整数

**后置条件:**
- `lisi_scores[i] >= 1`
- LISI = 1 当所有邻居具有相同标签时
- LISI 接近 n_labels 对于均匀分布

**复杂度:**
- 时间: O(n * k)，其中 k = 平均邻居数
- 空间: O(n_labels * n_threads) 用于线程本地计数器

**线程安全:** 安全 - 使用 WorkspacePool 并行化

### graph_connectivity

测量聚类连通性，作为完全连通的聚类比例。

```cpp
Sparse<Real, true> adjacency = /* 细胞邻域图 */;
Array<const Index> labels = /* 聚类分配 */;

Real connectivity = scl::kernel::metrics::graph_connectivity(adjacency, labels);
```

**参数:**
- `adjacency` [in] - 细胞邻域图
- `labels` [in] - 聚类分配

**返回:**
- 连通的聚类比例，范围 [0, 1]

**前置条件:**
- `adjacency.rows() == labels.len`
- 对于无向连通性，邻接矩阵应对称

**后置条件:**
- 如果所有聚类完全连通，返回 1.0
- 如果所有聚类碎片化，返回 0.0

**复杂度:**
- 时间: O(n + nnz)
- 空间: O(n) 用于分量 ID 和 BFS 队列

**线程安全:** 不安全 - 顺序 BFS

## 配置

```cpp
namespace scl::kernel::metrics::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real LOG2_E = Real(1.4426950408889634);
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## 使用场景

### 聚类评估

```cpp
// 评估聚类质量
Sparse<Real, true> distances = /* 成对距离 */;
Array<const Index> labels = /* 聚类标签 */;

Real silhouette = scl::kernel::metrics::silhouette_score(distances, labels);
// 越高越好（接近 1）

// 比较两个聚类
Array<const Index> labels2 = /* 替代聚类 */;
Real ari = scl::kernel::metrics::adjusted_rand_index(labels, labels2);
Real nmi = scl::kernel::metrics::normalized_mutual_information(labels, labels2);
```

### 批次整合评估

```cpp
// 评估批次校正
Sparse<Index, true> knn = /* KNN 图 */;
Array<const Index> batches = /* 批次标签 */;
Array<Real> entropy(n_cells);

scl::kernel::metrics::batch_entropy(knn, batches, entropy);
Real mean_entropy = scl::kernel::metrics::mean_batch_entropy(knn, batches);
// 更高的熵 = 更好的批次混合

// 用于批次多样性的 LISI
Array<Real> lisi(n_cells);
scl::kernel::metrics::lisi(knn, batches, lisi);
Real mean_lisi = scl::kernel::metrics::mean_lisi(knn, batches);
// 更高的 LISI = 邻域中更多样化的批次
```

## 性能

- **并行化**: 随细胞数量线性扩展
- **内存高效**: 大型数据集的最小分配
- **SIMD 加速**: 均值计算的向量化操作
- **工作空间池**: 并行操作的线程本地缓冲区

---

::: tip 度量选择
- **轮廓**: 最适合在没有真实标签的情况下评估聚类质量
- **ARI/NMI**: 最适合比较聚类或针对真实标签评估
- **LISI/批次熵**: 最适合评估批次校正质量
- **图连通性**: 最适合评估空间/空间转录组学聚类
:::

