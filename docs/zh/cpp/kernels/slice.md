# slice.hpp

> scl/kernel/slice.hpp · 稀疏矩阵切片内核

## 概述

本文件提供用于沿主维度和次维度切片稀疏矩阵的高性能内核。支持高效的检查（统计非零）、物化（复制到预分配数组）和创建新稀疏矩阵的完整切片操作。所有操作都经过并行化并针对缓存效率进行了优化。

**头文件**: `#include "scl/kernel/slice.hpp"`

---

## 主要 API

### slice_primary

::: source_code file="scl/kernel/slice.hpp" symbol="slice_primary" collapsed
:::

**算法说明**

创建包含选定主维度切片（CSR 的行，CSC 的列）的新稀疏矩阵：

1. 调用 `inspect_slice_primary` 统计选定切片中的总非零数
2. 分配适当大小的输出数组（data、indices、indptr）
3. 调用 `materialize_slice_primary` 复制选定的切片
4. 将数组包装为新 Sparse 矩阵
5. 结果保留 `keep_indices` 的顺序并保持次维度

**边界条件**

- **空选择**: 返回零行/列的空矩阵
- **选择所有索引**: 返回原始矩阵的副本
- **无效索引**: 如果索引超出范围，行为未定义
- **重复索引**: keep_indices 中的重复索引导致重复行/列

**数据保证（前置条件）**

- `keep_indices` 中的所有索引在范围 [0, primary_dim) 内
- 源矩阵是有效的 CSR 或 CSC 格式
- `keep_indices` 可能未排序

**复杂度分析**

- **时间**: O(nnz_output / n_threads + n_keep) - 并行复制加上元数据设置
- **空间**: O(nnz_output) 用于结果矩阵

**示例**

```cpp
#include "scl/kernel/slice.hpp"

Sparse<Real, true> matrix = /* 源矩阵，CSR */;
Array<const Index> keep_indices = /* 要保留的行索引 */;

Sparse<Real, true> sliced = scl::kernel::slice::slice_primary(
    matrix,
    keep_indices
);

// sliced 仅包含选定的行，列保留
```

---

### filter_secondary

::: source_code file="scl/kernel/slice.hpp" symbol="filter_secondary" collapsed
:::

**算法说明**

通过次维度掩码（CSR 的列，CSC 的行）过滤创建新稀疏矩阵：

1. 构建从旧到新次维度索引的索引映射（紧凑范围）
2. 调用 `inspect_filter_secondary` 统计过滤后的非零数
3. 分配适当大小的输出数组
4. 调用 `materialize_filter_secondary` 复制并重新映射索引
5. 结果具有紧凑的次维度 [0, new_secondary_dim)

**边界条件**

- **全零掩码**: 返回空矩阵（零列/行）
- **全一掩码**: 返回原始矩阵的副本
- **稀疏掩码**: 高效处理只有少数 1 的掩码
- **索引重新映射**: 旧索引重新映射到紧凑范围

**数据保证（前置条件）**

- `mask.len >= secondary_dim`
- 掩码值为 0 或 1
- 源矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(nnz / n_threads + secondary_dim) - 并行过滤加上映射
- **空间**: O(nnz_output + secondary_dim) 用于结果和索引映射

**示例**

```cpp
Array<const uint8_t> mask = /* 列的布尔掩码 */;

Sparse<Real, true> filtered = scl::kernel::slice::filter_secondary(
    matrix,
    mask
);

// filtered 仅包含 mask[col] == 1 的列
// 列索引重新映射到 [0, new_n_cols)
```

---

## 工具函数

### inspect_slice_primary

统计选定主维度切片中的总非零数。

::: source_code file="scl/kernel/slice.hpp" symbol="inspect_slice_primary" collapsed
:::

**复杂度**

- 时间: O(n_keep / n_threads)
- 空间: O(n_threads) 用于部分和

---

### materialize_slice_primary

将选定的主维度切片复制到预分配的输出数组。

::: source_code file="scl/kernel/slice.hpp" symbol="materialize_slice_primary" collapsed
:::

**复杂度**

- 时间: O(nnz_output / n_threads + n_keep)
- 空间: O(1) 超出输出

---

### inspect_filter_secondary

统计通过次维度掩码过滤后的非零数。

::: source_code file="scl/kernel/slice.hpp" symbol="inspect_filter_secondary" collapsed
:::

**复杂度**

- 时间: O(nnz / n_threads)
- 空间: O(n_threads) 用于部分和

---

### materialize_filter_secondary

将通过次掩码的元素复制到预分配的输出，并进行索引重新映射。

::: source_code file="scl/kernel/slice.hpp" symbol="materialize_filter_secondary" collapsed
:::

**复杂度**

- 时间: O(nnz / n_threads + primary_dim)
- 空间: O(1) 超出输出

---

## 配置

内部配置常量：

- `PARALLEL_THRESHOLD_ROWS = 512`: 并行处理的最小行数
- `PARALLEL_THRESHOLD_NNZ = 10000`: 并行处理的最小非零数
- `MEMCPY_THRESHOLD = 8`: memcpy 与循环的最小元素数

---

## 性能说明

### 并行化

- 主维度切片：并行归约用于计数，并行复制用于物化
- 次维度过滤：在主维度上并行，使用 8 路展开计数
- 缓存高效：使用预取和批处理

### 内存效率

- 两阶段方法：先检查以确定输出大小，然后物化
- 预分配数组：允许调用者管理内存
- 零拷贝潜力：可以在适当所有权下包装现有数组

---

## 相关内容

- [稀疏矩阵](../core/sparse) - 稀疏矩阵操作
- [内存模块](../core/memory) - 内存管理
