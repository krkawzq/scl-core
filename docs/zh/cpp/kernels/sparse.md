# sparse.hpp

> scl/kernel/sparse.hpp · 稀疏矩阵统计内核

## 概述

本文件提供用于计算稀疏矩阵统计、格式转换、数据清理和验证的高性能内核。包括沿主维度计算和、均值、方差的函数，导出为连续数组（CSR/CSC）和 COO 格式，消除零值、修剪和内存分析。

**头文件**: `#include "scl/kernel/sparse.hpp"`

---

## 主要 API

### primary_sums

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_sums" collapsed
:::

**算法说明**

计算沿每个主维度（CSR 的行，CSC 的列）的值和：

1. 并行处理每个主索引：
   - 使用稀疏矩阵访问器获取主切片的值的跨度
   - 使用 `scl::vectorize::sum` 进行 SIMD 优化的归约
   - 累加切片中所有非零值的和
2. 空切片产生和 = 0
3. 在主维度上并行化以提高效率

**边界条件**

- **空切片**: 返回无非零的切片和 = 0
- **全零**: 返回零和
- **单个非零**: 返回该值
- **非常稀疏**: 高效处理非零较少的切片

**数据保证（前置条件）**

- `output.len == matrix.primary_dim()`
- 矩阵是有效的稀疏格式（CSR 或 CSC）
- 输出数组可写

**复杂度分析**

- **时间**: O(nnz) - 每个非零处理一次
- **空间**: O(1) 辅助空间 - 每个线程仅累加器

**示例**

```cpp
#include "scl/kernel/sparse.hpp"

Sparse<Real, true> matrix = /* 稀疏矩阵，CSR */;
Array<Real> row_sums(matrix.rows());

scl::kernel::sparse::primary_sums(matrix, row_sums);

// row_sums[i] = 第 i 行所有非零的和
```

---

### primary_means

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_means" collapsed
:::

**算法说明**

计算沿每个主维度的值的均值，考虑隐式零：

1. 并行处理每个主索引：
   - 使用 `primary_sums` 计算和
   - 除以 secondary_dim（不是 nnz）：`mean = sum / secondary_dim`
2. 这考虑了稀疏表示中的隐式零
3. 空切片产生均值 = 0

**边界条件**

- **空切片**: 返回均值 = 0
- **零 secondary_dim**: 避免除以零，返回 0
- **常数切片**: 返回常数值
- **稀疏切片**: 均值反映隐式零

**数据保证（前置条件）**

- `output.len == matrix.primary_dim()`
- `matrix.secondary_dim() > 0`（对于有意义的均值）
- 矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(nnz) - 由和计算主导
- **空间**: O(1) 辅助空间 - 每个线程

**示例**

```cpp
Array<Real> row_means(matrix.rows());

scl::kernel::sparse::primary_means(matrix, row_means);

// row_means[i] = sum(row_i) / n_cols（包括隐式零）
```

---

### primary_variances

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_variances" collapsed
:::

**算法说明**

使用融合和与平方和计算沿每个主维度的方差：

1. 并行处理每个主索引：
   - 使用 SIMD 融合和+平方和辅助函数（4 路展开加 FMA）
   - 计算均值：`mean = sum / secondary_dim`
   - 计算方差：`var = (sumsq - sum*mean) / (secondary_dim - ddof)`
   - 将负值限制为零以保持数值稳定性
2. 使用补偿求和模式以提高准确性
3. ddof（自由度增量）默认为 1 用于样本方差

**边界条件**

- **零方差**: 返回常数切片的 0
- **负方差**: 限制为 0（数值稳定性）
- **ddof >= secondary_dim**: 避免除以零
- **空切片**: 返回方差 = 0

**数据保证（前置条件）**

- `output.len == matrix.primary_dim()`
- `ddof >= 0 且 ddof < secondary_dim`
- 矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(nnz) - 单次遍历，带融合计算
- **空间**: O(1) 辅助空间 - 每个线程

**示例**

```cpp
Array<Real> row_variances(matrix.rows());

scl::kernel::sparse::primary_variances(matrix, row_variances, 1);

// row_variances[i] = 第 i 行的样本方差（ddof=1）
```

---

### to_contiguous_arrays

::: source_code file="scl/kernel/sparse.hpp" symbol="to_contiguous_arrays" collapsed
:::

**算法说明**

将稀疏矩阵导出为带注册表注册数组的连续 CSR/CSC 格式：

1. 通过注册表分配 data、indices、indptr 数组
2. 通过行/列长度的累积和构建 indptr
3. 顺序复制值和索引到连续数组
4. 使用 HandlerRegistry 注册数组以进行内存管理
5. 返回带注册指针的结构

**边界条件**

- **空矩阵 (nnz=0)**: 返回全空指针，分配 indptr
- **已连续**: 仍创建新数组（无零拷贝）
- **分配失败**: 返回全空指针
- **注册表已满**: 返回全空指针

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式
- 有足够的内存可用于分配

**复杂度分析**

- **时间**: O(nnz) - 所有元素的顺序复制
- **空间**: O(nnz + primary_dim) 用于输出数组

**示例**

```cpp
auto arrs = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrs.data、arrs.indices、arrs.indptr 已注册到注册表
// arrs.nnz = 总非零数
// arrs.primary_dim = 行数（CSR）或列数（CSC）

// 在取消注册之前将所有权转移到 Python
```

---

### to_coo_arrays

::: source_code file="scl/kernel/sparse.hpp" symbol="to_coo_arrays" collapsed
:::

**算法说明**

将稀疏矩阵导出为带注册表注册数组的 COO（坐标）格式：

1. 计算每个主切片的偏移量
2. 并行转换：每个线程处理一个或多个主切片
3. 将 COO 三元组（行、列、值）写入预计算位置
4. 条目按行主序（CSR）或列主序（CSC）
5. 使用 HandlerRegistry 注册数组

**边界条件**

- **空矩阵**: 返回全空指针
- **分配失败**: 返回全空指针
- **并行开销**: 小矩阵可能比顺序慢

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式
- 有足够的内存可用

**复杂度分析**

- **时间**: O(nnz / n_threads + primary_dim) - 并行转换
- **空间**: O(nnz + primary_dim) 用于输出数组和偏移量

**示例**

```cpp
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// coo.row_indices[i]、coo.col_indices[i]、coo.values[i] 形成一个条目
// coo.nnz = 总非零数
```

---

### eliminate_zeros

::: source_code file="scl/kernel/sparse.hpp" symbol="eliminate_zeros" collapsed
:::

**算法说明**

从稀疏矩阵中移除零值元素：

1. 并行统计按容差过滤后每行/列的非零数
2. 分配具有减少 nnz 的新矩阵
3. 仅并行复制非零元素（其中 |value| > tolerance）
4. 索引保持排序
5. 创建新矩阵，原始矩阵不变

**边界条件**

- **无零**: 返回原始矩阵的副本
- **全零**: 返回空矩阵
- **容差 > 0**: 也移除近零
- **非常稀疏**: 高效处理具有许多零的矩阵

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式
- `tolerance >= 0`

**复杂度分析**

- **时间**: O(nnz / n_threads) - 并行过滤和复制
- **空间**: O(nnz_output + primary_dim) 用于结果和计数

**示例**

```cpp
Sparse<Real, true> cleaned = scl::kernel::sparse::eliminate_zeros(
    matrix,
    1e-10  // 容差
);

// cleaned 已移除所有 |value| <= 1e-10
```

---

### prune

::: source_code file="scl/kernel/sparse.hpp" symbol="prune" collapsed
:::

**算法说明**

从稀疏矩阵中移除小值，可选择保留结构：

1. 如果 `keep_structure = true`：
   - 将 |value| < threshold 的值设置为零
   - 保持矩阵结构（索引、指针）不变
2. 如果 `keep_structure = false`：
   - 完全移除 |value| < threshold 的元素
   - 压缩结构（减少 nnz）
3. 创建新矩阵，原始矩阵不变

**边界条件**

- **所有值被修剪**: 返回空矩阵（如果 !keep_structure）
- **无值被修剪**: 返回原始矩阵的副本
- **keep_structure = true**: 矩阵大小不变，某些值为零
- **keep_structure = false**: 矩阵大小减少

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式
- `threshold >= 0`

**复杂度分析**

- **时间**: O(nnz) - 单次遍历过滤
- **空间**: O(nnz) 用于结果（keep_structure）或 O(nnz_output)（移除）

**示例**

```cpp
// 修剪并移除结构
Sparse<Real, true> pruned = scl::kernel::sparse::prune(
    matrix,
    0.01,      // 阈值
    false      // 移除结构
);

// 修剪但保留结构
Sparse<Real, true> pruned_keep = scl::kernel::sparse::prune(
    matrix,
    0.01,
    true       // 保留结构
);
```

---

## 工具函数

### primary_nnz

获取每个主切片中的非零元素数量。

::: source_code file="scl/kernel/sparse.hpp" symbol="primary_nnz" collapsed
:::

**复杂度**

- 时间: O(primary_dim)
- 空间: O(1) 辅助空间

---

### from_contiguous_arrays

从连续 CSR/CSC 格式数组创建稀疏矩阵。

::: source_code file="scl/kernel/sparse.hpp" symbol="from_contiguous_arrays" collapsed
:::

**复杂度**

- 时间: O(primary_dim) 用于元数据设置
- 空间: O(primary_dim) 用于元数据数组

---

### validate

验证稀疏矩阵结构的完整性。

::: source_code file="scl/kernel/sparse.hpp" symbol="validate" collapsed
:::

**复杂度**

- 时间: O(nnz)
- 空间: O(1)

---

### memory_info

获取稀疏矩阵的详细内存使用信息。

::: source_code file="scl/kernel/sparse.hpp" symbol="memory_info" collapsed
:::

**复杂度**

- 时间: O(primary_dim) 用于块计数
- 空间: O(1)

---

### make_contiguous

如果尚未连续，则将稀疏矩阵转换为连续存储布局。

::: source_code file="scl/kernel/sparse.hpp" symbol="make_contiguous" collapsed
:::

**复杂度**

- 时间: 如果需要转换则为 O(nnz)，如果已连续则为 O(primary_dim)
- 空间: 如果需要转换则为 O(nnz)

---

### resize_secondary

调整稀疏矩阵的次维度（仅元数据）。

::: source_code file="scl/kernel/sparse.hpp" symbol="resize_secondary" collapsed
:::

**复杂度**

- 时间: 发布模式下为 O(1)，调试模式下缩小为 O(nnz)
- 空间: O(1)

---

## 相关内容

- [稀疏矩阵核心](../core/sparse) - 核心稀疏矩阵类型
- [内存模块](../core/memory) - 内存管理

