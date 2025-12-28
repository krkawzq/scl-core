# 质量控制

用于单细胞数据的质量控制指标计算，支持 SIMD 优化。

## 概述

`qc` 模块提供单细胞分析中常用的高效质量控制指标计算：

- **基因计数**: 每个细胞表达的基因数量
- **总计数**: 每个细胞的总 UMI 计数
- **子集百分比**: 来自特定基因子集（如线粒体基因）的计数百分比

所有操作都：
- 使用 SIMD 加速，融合操作
- 按细胞并行化
- 零分配（输出数组需预分配）

## 函数

### compute_basic_qc

计算基本质量控制指标：每个细胞的基因数量和总计数。

```cpp
#include "scl/kernel/qc.hpp"

Sparse<Real, true> matrix = /* 表达矩阵 [n_cells x n_genes] */;
Array<Index> n_genes(matrix.rows());
Array<Real> total_counts(matrix.rows());

scl::kernel::qc::compute_basic_qc(matrix, n_genes, total_counts);
```

**参数:**
- `matrix` [in] - 表达矩阵（细胞 x 基因，CSR）
- `out_n_genes` [out] - 每个细胞表达的基因数量 [n_cells]
- `out_total_counts` [out] - 每个细胞的总 UMI 计数 [n_cells]

**前置条件:**
- `out_n_genes.len == matrix.rows()`
- `out_total_counts.len == matrix.rows()`
- 矩阵必须是有效的 CSR 格式

**后置条件:**
- `out_n_genes[i]` 包含细胞 i 中非零基因的数量
- `out_total_counts[i]` 包含细胞 i 中所有计数的总和
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按细胞并行化

**算法:**
对每个细胞并行处理：
1. 计算非零元素数量（基因数量）
2. 使用 SIMD 优化的求和计算所有值的总和
3. 将结果写入输出数组

### compute_subset_pct

计算每个细胞中来自基因子集（如线粒体基因）的计数占总计数的百分比。

```cpp
Array<const uint8_t> mito_mask(n_genes);  // 1 表示线粒体基因
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_subset_pct(matrix, mito_mask, mito_pct);
```

**参数:**
- `matrix` [in] - 表达矩阵（细胞 x 基因，CSR）
- `subset_mask` [in] - 掩码数组，非零表示子集基因 [n_genes]
- `out_pcts` [out] - 百分比值 [n_cells]

**前置条件:**
- `out_pcts.len == matrix.rows()`
- `subset_mask.len >= matrix.cols()`
- 矩阵必须是有效的 CSR 格式

**后置条件:**
- `out_pcts[i]` 包含细胞 i 中子集计数的百分比（0-100）
- 如果总计数为零，返回 0.0
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按细胞并行化

**算法:**
对每个细胞并行处理：
1. 使用融合 SIMD 操作计算总计数和子集计数
2. 计算百分比 = (子集 / 总计) * 100
3. 将结果写入输出

### compute_fused_qc

在一次遍历中计算所有 QC 指标：基因计数、总计数和子集百分比。

```cpp
Array<const uint8_t> mito_mask(n_genes);
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_fused_qc(
    matrix, mito_mask, n_genes, total_counts, mito_pct
);
```

**参数:**
- `matrix` [in] - 表达矩阵（细胞 x 基因，CSR）
- `subset_mask` [in] - 子集基因的掩码数组 [n_genes]
- `out_n_genes` [out] - 每个细胞表达的基因数量 [n_cells]
- `out_total_counts` [out] - 每个细胞的总 UMI 计数 [n_cells]
- `out_pcts` [out] - 每个细胞的子集百分比 [n_cells]

**前置条件:**
- 所有输出数组的长度 == matrix.rows()
- `subset_mask.len >= matrix.cols()`
- 矩阵必须是有效的 CSR 格式

**后置条件:**
- 为每个细胞计算所有指标
- 矩阵不变

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按细胞并行化

**算法:**
对每个细胞并行处理：
1. 计算非零元素数量
2. 使用融合 SIMD 操作计算总计数和子集计数
3. 计算百分比
4. 将所有结果写入输出数组

## 配置

```cpp
namespace scl::kernel::qc::config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = Real(100);
}
```

## 使用场景

### 标准 QC 流程

```cpp
// 加载表达矩阵
Sparse<Real, true> expression = /* ... */;

// 创建线粒体基因掩码
Array<uint8_t> mito_mask(n_genes, 0);
for (Index g = 0; g < n_genes; ++g) {
    if (gene_names[g].starts_with("MT-")) {
        mito_mask.ptr[g] = 1;
    }
}

// 一次遍历计算所有 QC 指标
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_fused_qc(
    expression, mito_mask, n_genes, total_counts, mito_pct
);

// 根据 QC 指标过滤细胞
for (Index i = 0; i < n_cells; ++i) {
    if (n_genes.ptr[i] < 200 || total_counts.ptr[i] < 1000 || 
        mito_pct.ptr[i] > 20.0) {
        // 标记细胞为移除
    }
}
```

### 单独指标计算

```cpp
// 仅计算基本指标
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
scl::kernel::qc::compute_basic_qc(expression, n_genes, total_counts);

// 仅计算子集百分比
Array<Real> mito_pct(n_cells);
scl::kernel::qc::compute_subset_pct(expression, mito_mask, mito_pct);
```

## 性能

- **融合操作**: 一次遍历计算多个指标，减少内存流量
- **SIMD 加速**: 4 路展开累加以获得最大吞吐量
- **并行化**: 随 CPU 核心数线性扩展
- **零分配**: 所有输出数组必须预分配

---

::: tip 性能提示
当需要多个指标时，使用 `compute_fused_qc` - 它比分别调用各个函数更快。
:::

