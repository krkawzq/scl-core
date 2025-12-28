# 归一化

用于稀疏矩阵的归一化操作，支持 SIMD 优化。

## 概述

`normalize` 模块为稀疏矩阵提供高效的归一化操作：

- **行/列求和**: 沿主维度计算总和
- **缩放**: 按因子缩放主维度
- **掩码操作**: 使用基因/细胞掩码计算总和
- **检测**: 检测高表达基因

所有操作都：
- 使用 SIMD 加速，向量化操作
- 按行/列并行化
- 大多数操作零分配

## 函数

### compute_row_sums

计算稀疏矩阵每行的值总和。

```cpp
#include "scl/kernel/normalize.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<Real> row_sums(matrix.rows());

scl::kernel::normalize::compute_row_sums(matrix, row_sums);
```

**参数:**
- `matrix` [in] - 稀疏矩阵（CSR 或 CSC）
- `output` [out] - 行总和的预分配缓冲区 [n_rows]

**前置条件:**
- `output.len >= matrix.rows()`
- 矩阵必须是有效的稀疏格式

**后置条件:**
- `output[i]` 包含行 i 的总和
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按行并行化

**算法:**
对每行并行处理：
1. 遍历行中的非零元素
2. 使用向量化求和计算值的总和
3. 将结果写入输出

### scale_primary

按因子缩放每个主维度（CSR 为行，CSC 为列）。

```cpp
Array<Real> scales(matrix.rows());
// 设置缩放因子
for (Index i = 0; i < matrix.rows(); ++i) {
    scales.ptr[i] = target_sum / row_sums.ptr[i];
}

scl::kernel::normalize::scale_primary(matrix, scales);
```

**参数:**
- `matrix` [in,out] - 稀疏矩阵，原地修改
- `scales` [in] - 缩放因子 [primary_dim]

**前置条件:**
- `scales.len >= matrix.primary_dim()`
- 矩阵值必须可变

**后置条件:**
- 每个主维度按对应因子缩放
- 矩阵结构（indices, indptr）不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按主维度并行化

**算法:**
对每个主维度并行处理：
1. 如果 scale == 1.0，跳过
2. 加载此维度的值
3. 使用 SIMD 操作缩放
4. 将缩放后的值存储回去

### primary_sums_masked

计算每个主维度的值总和，仅计算 mask[indices[i]] == 0 的元素。

```cpp
Array<Byte> mask(n_genes);  // 0 = 包含，非零 = 排除
Array<Real> masked_sums(matrix.rows());

scl::kernel::normalize::primary_sums_masked(matrix, mask, masked_sums);
```

**参数:**
- `matrix` [in] - 稀疏矩阵（CSR 或 CSC）
- `mask` [in] - 掩码数组，0 表示应计算该元素 [secondary_dim]
- `output` [out] - 总和的预分配缓冲区 [primary_dim]

**前置条件:**
- `output.len >= matrix.primary_dim()`
- `mask.len >= matrix.secondary_dim()`
- 矩阵必须是有效的稀疏格式

**后置条件:**
- `output[i]` 包含主维度 i 中未掩码元素的总和
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按主维度并行化

**算法:**
对每个主维度并行处理：
1. 遍历非零元素
2. 检查 mask[indices[j]] == 0
3. 使用优化的 SIMD 路径对未掩码值求和
4. 将结果写入输出

### detect_highly_expressed

检测每个细胞中高表达的基因，其中表达超过总表达的一定比例。

```cpp
Array<Real> row_sums(matrix.rows());
scl::kernel::normalize::compute_row_sums(matrix, row_sums);

Array<Byte> highly_expressed(n_genes, 0);
Real max_fraction = 0.1;  // 总表达的 10%

scl::kernel::normalize::detect_highly_expressed(
    matrix, row_sums, max_fraction, highly_expressed
);
```

**参数:**
- `matrix` [in] - 表达矩阵（细胞 x 基因，CSR）
- `row_sums` [in] - 预计算的行总和 [n_cells]
- `max_fraction` [in] - 每个基因的最大总表达比例
- `out_mask` [out] - 输出掩码，1 表示高表达 [n_genes]

**前置条件:**
- `row_sums.len == matrix.rows()`
- `out_mask.len >= matrix.cols()`
- `max_fraction` 在 (0, 1] 范围内

**后置条件:**
- `out_mask[g] == 1` 如果基因 g 在任何细胞中高表达
- `out_mask[g] == 0` 否则
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 使用原子操作更新掩码

**算法:**
对每个细胞并行处理：
1. 计算阈值 = row_sums[cell] * max_fraction
2. 对细胞中每个表达的基因：
   a. 如果值 > 阈值，设置 out_mask[gene] = 1（原子操作）

## 配置

```cpp
namespace scl::kernel::normalize::config {
    constexpr Size PREFETCH_DISTANCE = 64;
}
```

## 使用场景

### 总计数归一化

```cpp
// 计算行总和
Array<Real> row_sums(matrix.rows());
scl::kernel::normalize::compute_row_sums(matrix, row_sums);

// 计算缩放因子
Array<Real> scales(matrix.rows());
Real target_sum = 10000.0;
for (Index i = 0; i < matrix.rows(); ++i) {
    scales.ptr[i] = target_sum / row_sums.ptr[i];
}

// 缩放矩阵
scl::kernel::normalize::scale_primary(matrix, scales);
```

### 掩码归一化

```cpp
// 从归一化中排除线粒体基因
Array<Byte> mask(n_genes, 0);
for (Index g = 0; g < n_genes; ++g) {
    if (gene_names[g].starts_with("MT-")) {
        mask.ptr[g] = 1;  // 排除
    }
}

// 计算排除线粒体基因的总和
Array<Real> masked_sums(matrix.rows());
scl::kernel::normalize::primary_sums_masked(matrix, mask, masked_sums);

// 使用掩码总和归一化
Array<Real> scales(matrix.rows());
for (Index i = 0; i < matrix.rows(); ++i) {
    scales.ptr[i] = target_sum / masked_sums.ptr[i];
}
scl::kernel::normalize::scale_primary(matrix, scales);
```

### 高表达基因检测

```cpp
// 检测在任何细胞中表达 >5% 的基因
Array<Real> row_sums(matrix.rows());
scl::kernel::normalize::compute_row_sums(matrix, row_sums);

Array<Byte> highly_expressed(n_genes, 0);
scl::kernel::normalize::detect_highly_expressed(
    matrix, row_sums, 0.05, highly_expressed
);

// 过滤掉高表达基因
// （在下游分析中使用掩码）
```

## 性能

- **SIMD 加速**: 向量化操作以获得最大吞吐量
- **并行化**: 随 CPU 核心数线性扩展
- **零分配**: 大多数操作需要预分配的输出数组
- **缓存友好**: 按行处理最大化缓存重用

---

::: tip 性能提示
预计算行总和一次，并在多个操作（缩放、检测等）中重复使用。
:::

