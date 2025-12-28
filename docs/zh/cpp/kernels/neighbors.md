# neighbors.hpp

> scl/kernel/neighbors.hpp · 使用欧氏距离的 K 近邻计算

## 概述

本文件为稀疏矩阵使用欧氏距离提供高效的 K 近邻（KNN）计算。实现使用各种优化，包括 Cauchy-Schwarz 剪枝、自适应稀疏点积策略和 SIMD 优化操作。

主要特性：
- 基于欧氏距离的 KNN
- 预计算的平方范数以提高效率
- Cauchy-Schwarz 不等式用于剪枝
- 自适应稀疏点积（线性合并、二分搜索、跳跃）
- 用于维护 top-k 邻居的最大堆
- 跨样本的线程安全并行化

**头文件**: `#include "scl/kernel/neighbors.hpp"`

---

## 主要 API

### knn

使用欧氏距离查找稀疏矩阵中每一行的 K 个最近邻。

::: source_code file="scl/kernel/neighbors.hpp" symbol="knn" collapsed
:::

**算法说明**

对每个样本 i 并行执行：

1. 维护大小为 k 的最大堆用于存储最近邻
2. 对于每个候选 j != i：
   a. **Cauchy-Schwarz 剪枝**：如果 `|norm_i - norm_j| >= current_max_distance` 则跳过
     - 这使用 Cauchy-Schwarz 的下界：`|a-b| <= ||a-b||`
   b. **稀疏点积**：使用自适应策略计算点积：
     - **线性合并**：对于相似大小的向量（比例 < 32）
     - **二分搜索**：对于比例 >= 32（较大向量长得多）
     - **跳跃**：对于比例 >= 256（极端大小差异）
   c. **距离计算**：`distance = sqrt(norm_i + norm_j - 2*dot)`
     - 来自数值误差的负值钳制为 0
   d. **堆更新**：如果距离 < 当前最大值，移除最大值并插入新邻居
3. 对最终堆进行排序以获得按距离升序排列

稀疏点积优化：
- 非重叠索引范围的 8 路/4 路跳过
- 在合并循环中预取以提高缓存效率
- 在不相交范围上早期退出（O(1) 检查）

**边界条件**

- **存在的邻居少于 k 个**：剩余槽位填充 index=-1 和 distance=infinity
- **排除自身**：样本 i 从其自己的邻居列表中排除
- **全零样本**：正确计算距离（两个范数 = 0，距离 = 0）
- **相同样本**：距离 = 0，如果允许将在邻居列表中
- **来自数值误差的负距离**：在堆插入前钳制为 0
- **空矩阵**：立即返回，所有索引 = -1

**数据保证（前置条件）**

- `norms_sq.len >= matrix.primary_dim()`（预计算的范数）
- `norms_sq` 包含来自 `compute_norms()` 的有效平方范数
- `out_indices.len >= matrix.primary_dim() * k`
- `out_distances.len >= matrix.primary_dim() * k`
- `k > 0`
- 矩阵必须是有效的 CSR/CSC 格式

**复杂度分析**

- **时间**：最坏情况 O(n^2 * avg_nnz)，使用剪枝通常更好
  - 最佳情况（重度剪枝）：O(n * k * log k)
  - 使用 Cauchy-Schwarz 剪枝，许多候选者早期被跳过
- **空间**：每个线程 O(k) 用于堆存储

**示例**

```cpp
#include "scl/kernel/neighbors.hpp"
#include "scl/core/sparse.hpp"

// 创建稀疏矩阵（n_samples x n_features）
Sparse<Real, true> matrix(n_samples, n_features);
// 用数据填充...

// 预计算平方范数
Array<Real> norms_sq(n_samples);
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// 预分配输出
Size k = 10;  // 邻居数量
Array<Index> indices(n_samples * k);
Array<Real> distances(n_samples * k);

// 计算 KNN
scl::kernel::neighbors::knn(
    matrix,
    norms_sq,
    k,
    indices,
    distances
);

// 结果：
// 对于样本 i：
// - indices[i*k : i*k+k] 包含 k 个最近邻的索引
// - distances[i*k : i*k+k] 包含欧氏距离
// - 邻居按距离排序（升序）
// - 排除自身（i）
```

---

## 工具函数

### compute_norms

计算稀疏矩阵每一行/列的平方 L2 范数。必须在 `knn()` 之前调用。

::: source_code file="scl/kernel/neighbors.hpp" symbol="compute_norms" collapsed
:::

**复杂度**

- 时间：使用 SIMD 优化的 sum_squared O(nnz)
- 空间：辅助空间 O(1)

**示例**

```cpp
Sparse<Real, true> matrix(n_samples, n_features);
Array<Real> norms_sq(n_samples);

// 在 KNN 之前计算范数
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// 现在在 knn() 中使用 norms_sq
scl::kernel::neighbors::knn(matrix, norms_sq, k, indices, distances);
```

---

## 数值注意事项

- **距离公式**：`sqrt(norm_i + norm_j - 2*dot)`，其中 `norm_i = ||x_i||^2`
- **数值稳定性**：来自 `norm_i + norm_j - 2*dot` 的负值钳制为 0（根据 Cauchy-Schwarz 应该非负）
- **Cauchy-Schwarz 界**：`|norm_i - norm_j| <= ||x_i - x_j||` 用于剪枝
- **稀疏点积**：基于大小比的自适应策略：
  - 线性合并：当大小相似时 O(nnz1 + nnz2)
  - 二分搜索：当比例 >= 32 时 O(nnz1 * log(nnz2))
  - 跳跃：当比例 >= 256 时 O(nnz1 * log(nnz2/nnz1))
- **堆操作**：维护最大堆，插入/删除为 O(log k)

## 相关内容

- [BBKNN](/zh/cpp/kernels/bbknn) - 用于批次整合的批次平衡 KNN
- [Normalization](/zh/cpp/kernels/normalization) - 在计算距离之前归一化矩阵
