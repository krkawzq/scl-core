# 邻居搜索

带稀疏矩阵优化和 Cauchy-Schwarz 剪枝的 K 近邻（KNN）计算。

## 概述

邻居操作提供：

- **KNN 搜索** - 为每个样本找到 K 个最近邻
- **欧氏距离** - L2 距离计算
- **稀疏优化** - 高效的稀疏点积
- **剪枝** - Cauchy-Schwarz 下界用于早期终止

## 基本用法

### knn

为稀疏矩阵中的每一行找到 K 个最近邻。

```cpp
#include "scl/kernel/neighbors.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // n_samples x n_features

// 预计算平方范数
Array<Real> norms_sq(matrix.primary_dim());
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// 找到 K 个最近邻
Size k = 15;
Array<Index> indices(matrix.primary_dim() * k);
Array<Real> distances(matrix.primary_dim() * k);

scl::kernel::neighbors::knn(
    matrix,
    norms_sq,
    k,
    indices,
    distances
);
```

**参数：**
- `matrix` [in] - 稀疏矩阵（n_samples x n_features）
- `norms_sq` [in] - 从 `compute_norms()` 预计算的平方范数
- `k` [in] - 要找到的邻居数
- `out_indices` [out] - 邻居索引，形状 (n_samples * k)
- `out_distances` [out] - 邻居距离，形状 (n_samples * k)

**前置条件：**
- `norms_sq.len >= matrix.primary_dim()`
- `norms_sq` 包含来自 `compute_norms()` 的有效平方范数
- `out_indices.len >= matrix.primary_dim() * k`
- `out_distances.len >= matrix.primary_dim() * k`
- `k > 0`

**后置条件：**
- 对于每个样本 i：
  - `out_indices[i*k : i*k+k]` 包含 k 个最近邻的索引
  - `out_distances[i*k : i*k+k]` 包含到邻居的欧氏距离
  - 邻居按距离排序（升序）
  - 自身（i）从邻居中排除
- 如果存在的邻居少于 k 个：剩余槽位填充 index=-1 和 distance=infinity

**算法：**
对每个样本 i 并行：
1. 维护大小为 k 的最大堆用于最近邻
2. 对于每个候选 j != i：
   a. Cauchy-Schwarz 剪枝：如果 |norm_i - norm_j| >= current_max 则跳过
   b. 使用自适应策略计算稀疏点积：
      - 线性合并：用于相似大小的向量
      - 二分搜索：用于比例 >= 32
      - 跳跃搜索：用于比例 >= 256
   c. 计算距离：sqrt(norm_i + norm_j - 2*dot)
   d. 如果距离 < 当前最大值则更新堆
3. 对最终堆排序以获得升序

**稀疏点积优化：**
- 非重叠索引范围的 8 路/4 路跳过
- 合并循环中的预取
- 不相交范围的早期退出（O(1) 检查）

**复杂度：**
- 时间: O(n^2 * avg_nnz) 最坏情况，使用剪枝通常好得多
- 空间: 每个线程 O(k) 用于堆存储

**线程安全：**
安全 - 按样本并行，使用线程本地工作空间

**数值说明：**
- 距离计算为 sqrt(norm_i + norm_j - 2*dot)
- 数值误差导致的负值被钳制为 0
- Cauchy-Schwarz 下界实现显著剪枝

## 辅助函数

### compute_norms

计算稀疏矩阵每行/列的平方 L2 范数。

```cpp
Array<Real> norms_sq(matrix.primary_dim());
scl::kernel::neighbors::compute_norms(matrix, norms_sq);
```

**参数：**
- `matrix` [in] - 稀疏矩阵（CSR 或 CSC）
- `norms_sq` [out] - 预分配的平方范数缓冲区

**后置条件：**
- `norms_sq[i] = 行/列 i 的平方值之和`

**算法：**
对每行并行：
- 使用 SIMD 优化的 `scl::vectorize::sum_squared`

**复杂度：**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全：**
安全 - 按行并行，无共享可变状态

**抛出：**
`SCL_CHECK_DIM` - 如果 norms_sq 大小不足

## 使用场景

### 构建 KNN 图

为下游分析构建 KNN 图：

```cpp
Sparse<Real, true> data = /* ... */;  // 细胞 x 基因

// 计算范数
Array<Real> norms_sq(data.primary_dim());
scl::kernel::neighbors::compute_norms(data, norms_sq);

// 找到邻居
Size k = 15;
Array<Index> knn_indices(data.primary_dim() * k);
Array<Real> knn_distances(data.primary_dim() * k);

scl::kernel::neighbors::knn(
    data, norms_sq, k, knn_indices, knn_distances
);

// 构建邻接矩阵或图结构
// 使用 knn_indices 和 knn_distances 构建图
```

### UMAP / t-SNE 预处理

为降维准备 KNN 图：

```cpp
// 标准预处理
Sparse<Real, true> normalized_data = /* ... */;

Array<Real> norms_sq(normalized_data.primary_dim());
scl::kernel::neighbors::compute_norms(normalized_data, norms_sq);

Size k = 15;
Array<Index> knn_indices(normalized_data.primary_dim() * k);
Array<Real> knn_distances(normalized_data.primary_dim() * k);

scl::kernel::neighbors::knn(
    normalized_data, norms_sq, k, knn_indices, knn_distances
);

// 将 KNN 图传递给 UMAP/t-SNE
```

### 聚类预处理

为 Leiden/Louvain 聚类计算 KNN：

```cpp
// 归一化并计算 KNN
Sparse<Real, true> data = /* ... */;
scl::kernel::normalize::normalize_rows_inplace(data, NormMode::L2);

Array<Real> norms_sq(data.primary_dim());
scl::kernel::neighbors::compute_norms(data, norms_sq);

Array<Index> knn_indices(data.primary_dim() * 15);
Array<Real> knn_distances(data.primary_dim() * 15);

scl::kernel::neighbors::knn(
    data, norms_sq, 15, knn_indices, knn_distances
);

// 转换为图并聚类
// 与 scl::kernel::leiden 或 scl::kernel::louvain 一起使用
```

## 性能

### 剪枝优化

Cauchy-Schwarz 下界实现早期终止：
- 跳过 |norm_i - norm_j| >= current_max_distance 的候选
- 在典型情况下减少 50-90% 的计算

### 稀疏点积

基于向量大小比例的自适应策略：
- 线性合并：O(nnz1 + nnz2) 用于相似大小
- 二分搜索：O(nnz1 * log(nnz2)) 用于比例 >= 32
- 跳跃搜索：O(nnz1 * log(nnz2/nnz1)) 用于比例 >= 256

### SIMD 优化

- SIMD 优化的范数计算
- 合并循环中的预取
- 非重叠范围的 8 路/4 路跳过

### 并行化

- 按样本并行
- 线程本地堆存储
- 无同步开销

## 算法细节

### 距离计算

欧氏距离: sqrt(norm_i + norm_j - 2*dot)

其中：
- norm_i = ||x_i||^2（预计算）
- norm_j = ||x_j||^2（预计算）
- dot = x_i · x_j（通过稀疏点积计算）

### Cauchy-Schwarz 剪枝

下界: |norm_i - norm_j| <= ||x_i - x_j||

如果 |norm_i - norm_j| >= current_max_distance：
- 跳过候选 j（不可能比当前最大值更近）

### 堆管理

大小为 k 的最大堆：
- 维护到目前为止看到的 k 个最近邻
- O(log k) 插入
- 最终排序: O(k log k)

## 参见

- [BBKNN](/zh/cpp/kernels/bbknn) - 批次平衡 KNN
- [空间](/zh/cpp/kernels/spatial) - 空间邻居搜索

