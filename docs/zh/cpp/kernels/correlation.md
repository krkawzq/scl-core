# correlation.hpp

> scl/kernel/correlation.hpp · 带 SIMD 优化的 Pearson 相关内核

## 概述

本文件提供稀疏特征矩阵的 Pearson 相关矩阵的高效计算。包括统计计算（均值和逆标准差）和完整的成对相关矩阵计算。所有操作都经过 SIMD 加速并使用缓存分块优化进行并行化。

**头文件**: `#include "scl/kernel/correlation.hpp"`

---

## 主要 API

### compute_stats

::: source_code file="scl/kernel/correlation.hpp" symbol="compute_stats" collapsed
:::

**算法说明**

并行计算每行（特征）的均值和逆标准差：

1. 并行处理每行：
   - 使用 4 路 SIMD 展开进行融合求和与平方和计算
   - 每次迭代加载 4 个值，在双累加器中累积（v_sum0, v_sum1, v_sq0, v_sq1）
   - 使用 FMA（融合乘加）计算平方和：`v_sq = v_sq + v * v`
   - 使用 SumOfLanes 归约为标量
2. 计算均值：`mean = sum / n_samples`
3. 计算方差：`var = sum_sq / n_samples - mean^2`
4. 计算逆标准差：如果 var > 0 则 `inv_std = 1 / sqrt(var)`，否则为 0
5. 方差被限制为 >= 0 以保证数值稳定性

**边界条件**

- **零方差行**: 方差为零的行得到 `inv_std = 0`（防止除以零）
- **空行**: 没有非零值的行均值为 0，inv_std = 0
- **常数行**: 所有值相同的行方差 = 0，inv_std = 0
- **数值精度**: 方差被限制为 >= 0 以处理浮点误差

**数据保证（前置条件）**

- `out_means.len >= matrix.primary_dim()`
- `out_inv_stds.len >= matrix.primary_dim()`
- 矩阵是有效的稀疏格式（CSR 或 CSC）
- 矩阵形状为 (n_features, n_samples)

**复杂度分析**

- **时间**: O(nnz / n_threads) - 在行上并行化，带 SIMD 加速
- **空间**: O(1) 辅助空间 - 仅需要累加器

**示例**

```cpp
#include "scl/kernel/correlation.hpp"

Sparse<Real, true> matrix = /* 稀疏矩阵 (n_features, n_samples) */;
Array<Real> means(matrix.primary_dim());
Array<Real> inv_stds(matrix.primary_dim());

scl::kernel::correlation::compute_stats(matrix, means, inv_stds);

// 使用预计算的统计量进行相关计算
Array<Real> corr_matrix(n_features * n_features);
scl::kernel::correlation::pearson(matrix, means, inv_stds, corr_matrix);
```

---

### pearson

::: source_code file="scl/kernel/correlation.hpp" symbol="pearson" collapsed
:::

**算法说明**

使用多种优化计算完整的成对 Pearson 相关矩阵：

1. **对称计算**: 仅计算上三角，复制到下三角
2. **稀疏中心化点积**: 使用代数恒等式避免物化密集向量：
   - `cov(a,b) = sum(a*b) - mean_a*sum(b) - mean_b*sum(a) + n*mean_a*mean_b`
   - 其中 sum(a*b) 通过非零索引的稀疏合并计算
3. **8/4 路跳过优化**: 在稀疏合并中，当索引相距较远时一次跳过 8 个元素
4. **基于块的并行化**: 以块为单位处理行以提高缓存局部性
5. **早期跳过**: 跳过零方差特征（与所有其他特征的相关性 = 0）
6. **相关公式**: `corr(a,b) = cov(a,b) / (std_a * std_b) = cov(a,b) * inv_std_a * inv_std_b`
7. **限制**: 相关值被限制在 [-1, 1] 以保证数值稳定性

**边界条件**

- **零方差特征**: 方差为零的特征与所有其他特征的相关性为 0
- **完美相关**: 对于完美相关的特征返回恰好 1.0 或 -1.0
- **稀疏特征**: 非零值很少的特征通过稀疏合并高效处理
- **数值溢出**: 限制防止值超出 [-1, 1] 范围

**数据保证（前置条件）**

- `output.len >= n_features^2`
- 如果使用带 means/inv_stds 的重载：`means.len >= n_features` 且 `inv_stds.len >= n_features`
- 矩阵是有效的稀疏格式
- 矩阵形状为 (n_features, n_samples)
- 如果提供统计量，它们必须匹配矩阵维度

**复杂度分析**

- **时间**: O(n_features^2 * avg_nnz_per_row / n_threads) - 与特征数成二次关系，与稀疏度成线性关系
- **空间**: O(1) 超出输出 - 不需要临时矩阵

**示例**

```cpp
// 选项 1：内部计算统计量
Array<Real> corr_matrix(n_features * n_features);
scl::kernel::correlation::pearson(matrix, corr_matrix);

// 选项 2：使用预计算的统计量（如果已计算统计量则更快）
Array<Real> means(n_features);
Array<Real> inv_stds(n_features);
scl::kernel::correlation::compute_stats(matrix, means, inv_stds);
scl::kernel::correlation::pearson(matrix, means, inv_stds, corr_matrix);

// 访问特征 i 和 j 之间的相关性
Real corr_ij = corr_matrix[i * n_features + j];  // 对称：corr_ij == corr_ji
```

---

## 配置

内部配置常量（不在 API 中公开）：

- `CHUNK_SIZE = 64`: 缓存分块的行块大小
- `STAT_CHUNK = 256`: 统计计算的块大小
- `PREFETCH_DISTANCE = 32`: 为缓存优化预取的元素数

---

## 性能说明

### SIMD 加速

- 4 路展开操作以获得最大吞吐量
- FMA（融合乘加）指令用于平方和
- 双累加器减少依赖链

### 缓存优化

- 基于块的处理提高缓存局部性
- 预取减少内存延迟
- 稀疏合并避免物化密集向量

### 并行化

- 与 CPU 核心数成线性扩展
- 基于块的调度以平衡负载
- 无共享状态冲突

---

## 相关内容

- [向量化模块](../core/vectorize) - SIMD 优化操作
- [稀疏矩阵](../core/sparse) - 稀疏矩阵操作
