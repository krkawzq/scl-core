# bbknn.hpp

> scl/kernel/bbknn.hpp · 批次平衡 K 近邻搜索，用于跨批次数据整合

## 概述

本文件提供批次平衡的 KNN 搜索功能，用于整合多个批次的数据。与标准 KNN 全局查找 k 个最近邻不同，BBKNN 从每个批次中分别查找 k 个最近邻，确保跨批次的平衡表示。

主要特性：
- 批次感知的邻居搜索
- 内存高效的批次分组处理
- 使用 Cauchy-Schwarz 剪枝和稀疏点积优化
- 线程安全的并行化

**头文件**: `#include "scl/kernel/bbknn.hpp"`

---

## 主要 API

### bbknn

::: source_code file="scl/kernel/bbknn.hpp" symbol="bbknn" collapsed
:::

**算法说明**

批次平衡 KNN 为每个样本从每个批次中查找 k 个最近邻：

1. **批次分组**：按批次标签对所有样本进行分组，提高缓存局部性
2. **逐样本处理**：对于每个查询样本 i：
   - 对于每个批次 b：
     - 初始化大小为 k 的最大堆以跟踪 k 个最近邻
     - 遍历批次 b 中的所有候选样本
     - 对于每个候选样本 j：
       - 使用 Cauchy-Schwarz 不等式计算下界：`min_dist^2 = norm_i^2 + norm_j^2 - 2*sqrt(norm_i^2 * norm_j^2)`
       - 如果下界 >= 堆中当前最大距离，跳过候选（剪枝）
       - 否则，计算精确欧氏距离：`dist^2 = norm_i^2 + norm_j^2 - 2*dot(i,j)`
       - 使用优化的稀疏点积，支持 8/4 路跳过优化
       - 如果距离 < 堆中当前最大值，插入堆中
     - 提取按距离排序的 k 个最近邻
3. **输出布局**：对于样本 i，批次 b，邻居 j：`offset = i * (n_batches * k) + b * k + j`

**边界条件**

- **空矩阵**：立即返回，所有索引设为 -1，距离设为无穷大
- **批次中邻居数少于 k**：剩余槽位用 index = -1，distance = infinity 填充
- **零范数行**：正确处理（当点积 = 0 时，距离 = norm_i^2 + norm_j^2）
- **负批次标签**：忽略负批次标签的样本（不包含在任何批次中）

**数据保证（前置条件）**

- 矩阵必须是有效的 CSR 格式，行内索引已排序
- `batch_labels.len == matrix.primary_dim()`
- `batch_labels[i]` 必须在范围 [0, n_batches) 内或为负（忽略）
- `out_indices.len >= n_samples * n_batches * k`
- `out_distances.len >= n_samples * n_batches * k`
- 如果提供 `norms_sq`：`norms_sq.len >= n_samples`
- 矩阵索引必须在每行内排序（CSR 要求）

**复杂度分析**

- **时间**：O(n_samples * avg_batch_size * (nnz_per_row + k*log(k)))
  - 对于每个样本：遍历每个批次中的候选样本
  - 稀疏点积：平均情况 O(nnz_per_row)
  - 堆操作：每个批次 O(k*log(k))
- **空间**：O(n_threads * n_batches * k) 用于线程本地堆

**示例**

```cpp
#include "scl/kernel/bbknn.hpp"
#include "scl/core/sparse.hpp"

// 创建稀疏矩阵（细胞 x 特征）
Sparse<Real, true> expression = /* ... */;  // n_cells x n_features
Array<int32_t> batch_labels(n_cells);
// ... 分配批次标签 (0, 1, 2, ...) ...

Size n_batches = 3;
Size k = 15;  // 每个批次的邻居数

// 预计算范数以提升性能（可选）
Array<Real> norms_sq(n_cells);
scl::kernel::bbknn::compute_norms(expression, norms_sq);

// 分配输出数组
Array<Index> indices(n_cells * n_batches * k);
Array<Real> distances(n_cells * n_batches * k);

// 计算 BBKNN
scl::kernel::bbknn::bbknn(
    expression,
    batch_labels,
    n_batches,
    k,
    indices,
    distances,
    norms_sq  // 可选：可省略以动态计算范数
);

// 访问细胞 i，批次 b，邻居 j 的邻居：
// Index neighbor_idx = indices[i * (n_batches * k) + b * k + j];
// Real dist = distances[i * (n_batches * k) + b * k + j];
```

---

### compute_norms

::: source_code file="scl/kernel/bbknn.hpp" symbol="compute_norms" collapsed
:::

**算法说明**

预计算稀疏矩阵所有行的平方 L2 范数：

1. 按行并行处理
2. 对于每行 i：计算 `norm_i^2 = sum(matrix[i,:]^2)`
3. 使用 SIMD 优化的 `scl::vectorize::sum_squared` 提高效率

**边界条件**

- **空行**：返回 norm_sq = 0
- **零矩阵**：所有范数为零

**数据保证（前置条件）**

- 矩阵必须是有效的 CSR 格式
- `norms_sq.len >= matrix.primary_dim()`

**复杂度分析**

- **时间**：O(nnz / n_threads) - 按行并行化
- **空间**：O(1) 辅助空间

**示例**

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> norms_sq(matrix.rows());

scl::kernel::bbknn::compute_norms(matrix, norms_sq);

// norms_sq[i] 现在包含 ||matrix[i,:]||^2
```

---

### build_batch_groups

::: source_code file="scl/kernel/bbknn.hpp" symbol="build_batch_groups" collapsed
:::

**算法说明**

构建内存高效的批次分组索引结构：

1. **第一遍**：统计每个批次的样本数（单遍遍历 batch_labels）
2. **前缀和**：计算每个批次的偏移量：`offsets[b] = sum(sizes[0..b-1])`
3. **第二遍**：填充索引数组，按批次分组样本

**边界条件**

- **负批次标签**：忽略（不包含在任何批次中）
- **空批次**：批次大小为 0，但偏移量仍正确计算
- **所有样本无效**：返回空结构，total_size = 0

**数据保证（前置条件）**

- `batch_labels[i]` 在范围 [0, n_batches) 内或为负（忽略）
- `n_batches > 0`

**复杂度分析**

- **时间**：O(n_samples) - 两遍遍历
- **空间**：O(n_samples + n_batches) 用于索引和偏移量数组

**示例**

```cpp
Array<int32_t> batch_labels(n_cells);
// ... 分配批次标签 ...

BatchGroups groups;
scl::kernel::bbknn::build_batch_groups(
    batch_labels,
    n_batches,
    groups
);

// 访问批次 b 中的样本：
Size batch_size = groups.batch_size(b);
const Index* batch_indices = groups.batch_data(b);

// 完成后释放
scl::kernel::bbknn::free_batch_groups(groups);
```

---

## 工具函数

### free_batch_groups

释放 `build_batch_groups` 分配的内存。

::: source_code file="scl/kernel/bbknn.hpp" symbol="free_batch_groups" collapsed
:::

**复杂度**

- 时间：O(1)
- 空间：O(1)

---

## 注意事项

**输出布局**

对于样本 i，批次 b，邻居 j：
```
offset = i * (n_batches * k) + b * k + j
indices[offset] = 邻居索引（如果未找到则为 -1）
distances[offset] = 欧氏距离（如果未找到则为无穷大）
```

**优化细节**

1. **Cauchy-Schwarz 剪枝**：下界实现早期终止，减少 30-70% 的计算
2. **稀疏点积**：对非重叠区域的 8/4 路跳过优化
3. **固定大小堆**：手动 sift 操作避免动态分配开销
4. **批次分组**：通过处理批次连续样本提高缓存局部性

**线程安全**

所有函数都是线程安全且并行化的：
- `bbknn`：使用线程本地堆存储
- `compute_norms`：按行并行
- `build_batch_groups`：顺序执行（单线程）

## 相关内容

- [Neighbors](/zh/cpp/kernels/neighbors) - 标准 KNN 搜索
- [Spatial](/zh/cpp/kernels/spatial) - 空间邻居搜索
