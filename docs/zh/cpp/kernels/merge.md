# merge.hpp

> scl/kernel/merge.hpp · 矩阵合并和拼接操作

## 概述

高效的稀疏矩阵拼接操作，用于沿主维度（行）或次维度（列）合并矩阵。这些操作对于数据整合、批次合并和单细胞分析中的特征组合至关重要。

本文件提供：
- 垂直堆叠 (vstack) - 沿行拼接
- 水平堆叠 (hstack) - 沿列拼接
- SIMD 优化的索引偏移操作
- 大型矩阵的并行内存复制

**头文件**: `#include "scl/kernel/merge.hpp"`

---

## 主要 API

### vstack

::: source_code file="scl/kernel/merge.hpp" symbol="vstack" collapsed
:::

**算法说明**

通过沿主维度（CSR 为行，CSC 为列）拼接来垂直堆叠两个稀疏矩阵：

1. **维度验证**：检查矩阵是否可以垂直堆叠（次维度可以不同）
2. **结果大小计算**：
   - 结果主维度 = matrix1.主维度 + matrix2.主维度
   - 结果次维度 = max(matrix1.次维度, matrix2.次维度)
   - 结果 nnz = matrix1.nnz + matrix2.nnz
3. **内存分配**：使用指定的块策略分配结果矩阵
4. **数据复制**（并行处理行）：
   - 将 matrix1 的行复制到 result[0 : n1]（索引不变）
   - 将 matrix2 的行复制到 result[n1 : n1+n2]（索引不变，如果次维度不同可能有间隙）
5. **索引指针设置**：为合并结构正确设置 indptr 数组

对于 CSR 矩阵：matrix1 的行放在前面，然后是 matrix2 的行。列索引保持不变。

**边界条件**

- **空 matrix1**：返回 matrix2 的副本
- **空 matrix2**：返回 matrix1 的副本
- **两者都为空**：返回具有正确维度的空矩阵
- **不同的次维度**：使用最大维度，matrix2 索引保持有效（不需要偏移）
- **一个矩阵的次维度为零**：正确处理，结果使用非零维度

**数据保证（前置条件）**

- 两个矩阵都必须是有效的稀疏矩阵
- 矩阵必须具有相同的格式（都是 CSR 或都是 CSC）
- 对于 CSR：次维度（列）可以不同
- 对于 CSC：次维度（行）可以不同
- 使用块策略进行结果分配

**复杂度分析**

- **时间**：O(nnz1 + nnz2)，其中 nnz1 和 nnz2 是每个矩阵的非零元素数量。并行复制操作减少了有效时间。
- **空间**：O(nnz1 + nnz2) 用于结果矩阵存储

**示例**

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* n1 x m1 稀疏矩阵 */;
Sparse<Real, true> matrix2 = /* n2 x m2 稀疏矩阵 */;

// 垂直堆叠（对于 CSR：堆叠行）
auto vstacked = scl::kernel::merge::vstack(matrix1, matrix2);

// 结果是 (n1+n2) x max(m1, m2) 稀疏矩阵
// 行 0 到 n1-1 来自 matrix1
// 行 n1 到 n1+n2-1 来自 matrix2

// 使用自定义块分配策略
auto vstacked_custom = scl::kernel::merge::vstack(
    matrix1, matrix2, BlockStrategy::adaptive()
);
```

---

### hstack

::: source_code file="scl/kernel/merge.hpp" symbol="hstack" collapsed
:::

**算法说明**

通过沿次维度（CSR 为列，CSC 为行）拼接来水平堆叠两个稀疏矩阵：

1. **维度验证**：验证主维度匹配（CSR 为行，CSC 为列）
2. **结果大小计算**：
   - 结果主维度 = matrix1.主维度（不变）
   - 结果次维度 = matrix1.次维度 + matrix2.次维度
   - 结果 nnz = matrix1.nnz + matrix2.nnz
3. **内存分配**：分配结果矩阵
4. **数据复制**（对于 CSR 并行处理行）：
   - 对每一行并行处理：
     - 将 matrix1 的值和索引复制到结果
     - 将 matrix2 的值复制到结果
     - 使用 SIMD 优化的加法将偏移（matrix1.次维度）添加到 matrix2 索引
     - 如果需要，合并排序的索引（两个矩阵都应有排序的索引）
5. **索引合并**：合并每行中来自两个矩阵的索引，保持排序顺序

对于 CSR 矩阵：每行包含来自 matrix1 的列，后跟来自 matrix2 的列。Matrix2 的列索引偏移了 matrix1.cols()。

**边界条件**

- **空 matrix1**：返回 matrix2，索引不变（不需要偏移）
- **空 matrix2**：返回 matrix1 的副本
- **两者都为空**：返回具有正确维度的空矩阵
- **主维度不匹配**：抛出 DimensionError
- **零偏移**：早期退出优化（不需要索引调整）

**数据保证（前置条件）**

- 两个矩阵都必须是有效的稀疏矩阵
- 矩阵必须具有相同的格式（都是 CSR 或都是 CSC）
- 主维度必须匹配：matrix1.主维度 == matrix2.主维度
- 索引应在每行（CSR）或每列（CSC）内排序以获得最佳性能
- 使用块策略进行结果分配

**复杂度分析**

- **时间**：O(nnz1 + nnz2) 用于数据复制。索引偏移加法使用 SIMD 优化进行批量操作。索引合并（如需要）增加 O(n * log(k))，其中 n 是行数，k 是每行平均非零元素数。
- **空间**：O(nnz1 + nnz2) 用于结果矩阵

**示例**

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* n x m1 稀疏矩阵 */;
Sparse<Real, true> matrix2 = /* n x m2 稀疏矩阵 */;  // 相同的行数

// 水平堆叠（对于 CSR：堆叠列）
auto hstacked = scl::kernel::merge::hstack(matrix1, matrix2);

// 结果是 n x (m1+m2) 稀疏矩阵
// 每行包含来自 matrix1 的列 0 到 m1-1，
// 后跟来自 matrix2 的列 m1 到 m1+m2-1
// Matrix2 的列索引偏移了 m1

// 使用自定义块策略
auto hstacked_custom = scl::kernel::merge::hstack(
    matrix1, matrix2, BlockStrategy::adaptive()
);
```

---

## 实现细节

### SIMD 优化

`hstack` 函数使用 SIMD（单指令多数据）指令进行高效的索引偏移加法：

- 当偏移 > 0 时，批量加法使用 2 路 SIMD 展开循环
- 标量清理处理剩余元素
- 当偏移 == 0 时早期退出优化（直接 memcpy）

### 并行处理

`vstack` 和 `hstack` 都使用并行处理：

- **vstack**：并行复制行/列（主维度）
- **hstack**：并行处理行（CSR）或列（CSC）
- 大型数据块使用带预取的并行 memcpy

### 内存管理

- 结果矩阵使用 `BlockStrategy` 分配以进行高效的稀疏存储
- 默认策略是 `BlockStrategy::adaptive()`，它选择最优的块大小
- 在可能的情况下连续分配内存以获得更好的缓存性能

## 注意事项

- **索引排序**：为了获得最佳性能，输入矩阵应在每行（CSR）或每列（CSC）内具有排序的索引。实现在结果中保持排序顺序。
- **格式一致性**：两个输入矩阵必须使用相同的存储格式（都是 CSR 或都是 CSC）。
- **维度约束**：
  - `vstack`：次维度可以不同（使用最大值）
  - `hstack`：主维度必须完全匹配
- **稀疏效率**：这些操作针对稀疏矩阵进行了优化，并高效地保留稀疏结构。

## 相关内容

- [稀疏矩阵](../core/sparse) - 稀疏矩阵数据结构文档
- [内存管理](../core/memory) - 块分配策略
