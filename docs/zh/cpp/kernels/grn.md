# 基因调控网络

从表达数据推断基因调控网络。

## 概述

`grn` 模块为单细胞表达数据提供高效的基因调控网络（GRN）推断：

- **基于相关性**: 简单的相关性网络
- **偏相关**: 控制其他基因的网络
- **互信息**: 基于信息论的网络
- **GENIE3**: 基于树的集成方法

所有操作都：
- 按基因对并行化
- 内存高效的稀疏存储
- 可配置阈值

## 函数

### infer_grn

从表达数据推断基因调控网络。

```cpp
#include "scl/kernel/grn.hpp"

Sparse<Real, true> expression = /* 表达矩阵 [n_cells x n_genes] */;
Real* network = /* 预分配 [n_genes * n_genes] */;

scl::kernel::grn::infer_grn(
    expression, n_cells, n_genes, network,
    scl::kernel::grn::GRNMethod::Correlation, 0.3
);
```

**参数:**
- `expression` [in] - 表达矩阵（细胞 x 基因，CSR）
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `network` [out] - GRN 邻接矩阵 [n_genes * n_genes]
- `method` [in] - 推断方法（默认: Correlation）
- `threshold` [in] - 相关性阈值（默认: 0.3）

**前置条件:**
- `network` 容量 >= n_genes * n_genes

**后置条件:**
- `network[i * n_genes + j]` 包含从基因 i 到 j 的边权重

**复杂度:**
- 时间: O(n_genes^2 * n_cells) 用于相关性
- 空间: O(n_genes^2) 辅助空间

**线程安全:** 安全 - 按基因对并行化

**方法:**
- `GRNMethod::Correlation` - Pearson 相关性
- `GRNMethod::PartialCorrelation` - 偏相关
- `GRNMethod::MutualInformation` - 基于 MI 的边
- `GRNMethod::GENIE3` - 树集成方法
- `GRNMethod::Combined` - 加权组合

### partial_correlation

计算偏相关矩阵（控制其他基因）。

```cpp
Real* partial_corr = /* 预分配 [n_genes * n_genes] */;
scl::kernel::grn::partial_correlation(
    expression, n_cells, n_genes, partial_corr
);
```

**参数:**
- `expression` [in] - 表达矩阵（细胞 x 基因，CSR）
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `partial_corr` [out] - 偏相关矩阵 [n_genes * n_genes]

**前置条件:**
- `partial_corr` 容量 >= n_genes * n_genes

**后置条件:**
- `partial_corr[i * n_genes + j]` 包含偏相关

**复杂度:**
- 时间: O(n_genes^3 * n_cells)
- 空间: O(n_genes^2) 辅助空间

**线程安全:** 安全 - 并行化

**算法:**
1. 计算完整相关矩阵
2. 反转相关矩阵得到精度矩阵
3. 偏相关 = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])

## 配置

```cpp
namespace scl::kernel::grn::config {
    constexpr Real DEFAULT_CORRELATION_THRESHOLD = Real(0.3);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Index DEFAULT_N_TREES = 100;
    constexpr Index DEFAULT_SUBSAMPLE = 500;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## 使用场景

### 基本相关性网络

```cpp
// 推断简单的基于相关性的 GRN
Sparse<Real, true> expression = /* ... */;
Real* network = new Real[n_genes * n_genes];

scl::kernel::grn::infer_grn(
    expression, n_cells, n_genes, network,
    scl::kernel::grn::GRNMethod::Correlation, 0.3
);

// 按阈值过滤边
for (Index i = 0; i < n_genes; ++i) {
    for (Index j = 0; j < n_genes; ++j) {
        if (std::abs(network[i * n_genes + j]) < 0.3) {
            network[i * n_genes + j] = 0.0;
        }
    }
}
```

### 偏相关网络

```cpp
// 推断偏相关网络（去除间接效应）
Real* partial_corr = new Real[n_genes * n_genes];
scl::kernel::grn::partial_correlation(
    expression, n_cells, n_genes, partial_corr
);

// 偏相关去除其他基因的混杂
// 对直接调控关系更准确
```

## 性能

- **并行化**: 随基因对数量平方扩展
- **内存高效**: 尽可能使用稀疏矩阵操作
- **阈值过滤**: 减少大型网络的内存
- **SIMD 加速**: 向量化相关性计算

---

::: tip 网络大小
对于大型基因集（>1000 个基因），考虑先按方差过滤或使用特征选择以减少计算成本。
:::

