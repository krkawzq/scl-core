# 采样

用于细胞选择和数据缩减的采样和下采样内核。

## 概述

`sampling` 模块提供多种采样策略：

- **几何草图**: 通过均匀流形覆盖保留稀有群体
- **密度保持**: 保持局部密度分布
- **地标选择**: KMeans++ 风格的多样化采样
- **代表性细胞**: 选择最接近聚类质心的细胞
- **平衡/分层**: 跨组/层的相等表示
- **均匀/蓄水池**: 简单随机采样

所有操作都：
- 内存高效
- 使用种子可重现
- 针对大型数据集优化

## 核心函数

### geometric_sketching

使用几何草图采样细胞以保留稀有群体。

```cpp
#include "scl/kernel/sampling.hpp"

Sparse<Real, true> data = /* 表达矩阵 */;
Index* selected = new Index[target_size];
Size n_selected;

scl::kernel::sampling::geometric_sketching(
    data, target_size, selected, n_selected, 42
);
```

**参数:**
- `data` [in] - 表达矩阵（细胞 x 基因，CSR）
- `target_size` [in] - 要选择的细胞数量
- `selected_indices` [out] - 选中细胞的索引
- `n_selected` [out] - 实际选中的细胞数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `selected_indices` 容量 >= min(target_size, data.rows())
- `target_size > 0`

**后置条件:**
- `n_selected <= target_size`
- `selected_indices[0..n_selected)` 包含选中细胞的索引
- 细胞从几何网格桶中均匀采样

**复杂度:**
- 时间: O(n * d + n log n)，其中 n = 细胞数，d = 特征数
- 空间: O(n + d) 辅助空间

**线程安全:** 不安全 - 顺序实现

### density_preserving

在保持局部密度分布的同时采样细胞。

```cpp
Sparse<Index, true> neighbors = /* KNN 图 */;
scl::kernel::sampling::density_preserving(
    data, neighbors, target_size, selected, n_selected
);
```

**参数:**
- `data` [in] - 表达矩阵
- `neighbors` [in] - KNN 图（CSR）
- `target_size` [in] - 期望的细胞数量
- `selected_indices` [out] - 选中细胞的索引
- `n_selected` [out] - 实际选中的数量

**前置条件:**
- `data.rows() == neighbors.rows()`
- `selected_indices` 容量 >= min(target_size, data.rows())

**后置条件:**
- 来自稀疏区域的细胞更可能被选中
- 样本中保持局部密度分布

**复杂度:**
- 时间: O(n)
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序实现

### landmark_selection

使用 KMeans++ 初始化选择多样化的地标细胞。

```cpp
scl::kernel::sampling::landmark_selection(
    data, n_landmarks, landmark_indices, n_selected, 42
);
```

**参数:**
- `data` [in] - 表达矩阵
- `n_landmarks` [in] - 要选择的地标数量
- `landmark_indices` [out] - 选中地标的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `landmark_indices` 容量 >= min(n_landmarks, data.rows())

**后置条件:**
- `n_selected = min(n_landmarks, data.rows())`
- 地标在表达空间中最大程度分散

**复杂度:**
- 时间: O(n_landmarks * n * d) 用于稀疏距离计算
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序 KMeans++

### representative_cells

从每个聚类中选择代表性细胞。

```cpp
Array<const Index> cluster_labels = /* 聚类分配 */;
Index* representatives = new Index[max_representatives];
Size n_selected;

scl::kernel::sampling::representative_cells(
    data, cluster_labels, per_cluster, representatives, n_selected, 42
);
```

**参数:**
- `data` [in] - 表达矩阵
- `cluster_labels` [in] - 每个细胞的聚类分配
- `per_cluster` [in] - 每个聚类的代表数量
- `representatives` [out] - 代表性细胞的索引
- `n_selected` [out] - 总代表数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `data.rows() == cluster_labels.len`
- `representatives` 有足够的容量

**后置条件:**
- `n_selected = sum(min(per_cluster, cluster_size))` 跨聚类
- 代表是每个聚类质心最近的细胞

**复杂度:**
- 时间: O(n * d + n_clusters * cluster_size * per_cluster)
- 空间: O(n + d * n_clusters) 辅助空间

**线程安全:** 不安全 - 顺序实现

## 平衡和分层采样

### balanced_sampling

从每个组/标签类别中采样相等数量。

```cpp
Array<const Index> labels = /* 组标签 */;
Index* selected = new Index[target_size];
Size n_selected;

scl::kernel::sampling::balanced_sampling(
    labels, target_size, selected, n_selected, 42
);
```

**参数:**
- `labels` [in] - 每个元素的组标签
- `target_size` [in] - 总期望样本大小
- `selected_indices` [out] - 选中元素的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `selected_indices` 容量 >= target_size
- 标签是非负整数

**后置条件:**
- 每个非空组贡献大约 target_size / n_groups 个样本
- 余数分配给前几个组

**复杂度:**
- 时间: O(n)
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序实现

### stratified_sampling

从通过分箱连续变量定义的层中采样。

```cpp
Array<const Real> values = /* 连续值 */;
scl::kernel::sampling::stratified_sampling(
    values, n_strata, target_size, selected, n_selected, 42
);
```

**参数:**
- `values` [in] - 用于分层的连续值
- `n_strata` [in] - 要创建的层数
- `target_size` [in] - 总期望样本大小
- `selected_indices` [out] - 选中元素的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `values.len > 0`
- `n_strata > 0`

**后置条件:**
- 元素被分箱到 n_strata 个等宽层
- 对层标签应用 balanced_sampling

**复杂度:**
- 时间: O(n)
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序实现

## 简单采样

### uniform_sampling

简单的无放回均匀随机采样。

```cpp
scl::kernel::sampling::uniform_sampling(
    n, target_size, selected_indices, n_selected, 42
);
```

**参数:**
- `n` [in] - 总体大小
- `target_size` [in] - 期望样本大小
- `selected_indices` [out] - 选中元素的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `selected_indices` 容量 >= min(target_size, n)

**后置条件:**
- `n_selected = min(target_size, n)`
- 每个元素具有相等的选择概率

**复杂度:**
- 时间: O(n) 用于初始化，O(target_size) 用于采样
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序实现

### importance_sampling

以与给定权重成比例的概率采样元素。

```cpp
Array<const Real> weights = /* 采样权重 */;
scl::kernel::sampling::importance_sampling(
    weights, target_size, selected_indices, n_selected, 42
);
```

**参数:**
- `weights` [in] - 采样权重（非负）
- `target_size` [in] - 要抽取的样本数量
- `selected_indices` [out] - 选中元素的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `weights.len > 0`
- 所有权重 >= 0

**后置条件:**
- `n_selected = target_size`
- P(选择 i) 与 weights[i] 成比例
- 同一元素可能出现多次（有放回）

**复杂度:**
- 时间: O(n + target_size * log n)
- 空间: O(n) 辅助空间

**线程安全:** 不安全 - 顺序实现

### reservoir_sampling

使用蓄水池采样从流中均匀随机选择 k 个项目。

```cpp
scl::kernel::sampling::reservoir_sampling(
    stream_size, reservoir_size, reservoir, n_selected, 42
);
```

**参数:**
- `stream_size` [in] - 流中的项目总数
- `reservoir_size` [in] - 要选择的项目数量
- `reservoir` [out] - 选中项目的索引
- `n_selected` [out] - 实际选中的数量
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `reservoir` 容量 >= min(reservoir_size, stream_size)

**后置条件:**
- `n_selected = min(reservoir_size, stream_size)`
- 每个项目在蓄水池中的概率相等

**复杂度:**
- 时间: O(stream_size)
- 空间: O(reservoir_size)

**线程安全:** 不安全 - 顺序实现

## 配置

```cpp
namespace scl::kernel::sampling::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_BINS = 64;
    constexpr Size MAX_ITERATIONS = 1000;
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## 使用场景

### 保留稀有群体

```cpp
// 使用几何草图保留稀有细胞类型
Sparse<Real, true> expression = /* ... */;
Index* selected = new Index[10000];
Size n_selected;

scl::kernel::sampling::geometric_sketching(
    expression, 10000, selected, n_selected, 42
);
// 选中的细胞在表达空间中具有均匀覆盖
```

### 聚类代表

```cpp
// 从每个聚类中选择代表性细胞
Array<const Index> clusters = /* 聚类标签 */;
Index* reps = new Index[n_clusters * 5];
Size n_reps;

scl::kernel::sampling::representative_cells(
    expression, clusters, 5, reps, n_reps, 42
);
// 每个聚类 5 个代表，最接近质心
```

### 平衡采样

```cpp
// 从每个批次中采样相等数量
Array<const Index> batches = /* 批次标签 */;
Index* selected = new Index[1000];
Size n_selected;

scl::kernel::sampling::balanced_sampling(
    batches, 1000, selected, n_selected, 42
);
// 每个批次贡献大致相等数量的细胞
```

### 按表达分层

```cpp
// 按总 UMI 计数分层
Array<Real> total_counts = /* 计算行和 */;
Index* selected = new Index[5000];
Size n_selected;

scl::kernel::sampling::stratified_sampling(
    total_counts, 10, 5000, selected, n_selected, 42
);
// 从 10 个 UMI 计数层中采样
```

## 性能

- **内存高效**: 大型数据集的最小分配
- **可重现**: 固定种子时具有确定性
- **快速 RNG**: Xoshiro128+ 用于高质量随机性
- **可扩展**: 高效处理数百万细胞

---

::: tip 方法选择
- **几何草图**: 最适合保留稀有群体
- **密度保持**: 最适合保持局部结构
- **地标选择**: 最适合多样化覆盖
- **代表性细胞**: 最适合聚类摘要
- **平衡/分层**: 最适合相等表示
- **均匀**: 最简单，随机采样最快
:::

