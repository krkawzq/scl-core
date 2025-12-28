# ttest.hpp

> scl/kernel/ttest.hpp · T 检验计算内核

## 概述

本文件为稀疏单细胞数据中比较两组提供高性能 t 检验计算。它支持 Welch t 检验（不等方差）和 Student t 检验（等方差），并具有优化的稀疏矩阵操作。

**头文件**: `#include "scl/kernel/ttest.hpp"`

主要特性：
- Welch 和 Student t 检验
- 稀疏矩阵优化
- Log2 倍数变化计算
- 组统计量计算
- SIMD 优化操作

---

## 主要 API

### ttest

::: source_code file="scl/kernel/ttest.hpp" symbol="ttest" collapsed
:::

**算法说明**

计算每个特征比较两组的 Welch 或 Student t 检验：

1. 对于每个特征 f（并行）：
   - 按组分割非零值（4 路展开，预取）
   - 在分割期间累加 sum 和 sum_sq
   - 计算包括零的均值：`mean = sum / n_total`
   - 使用零调整和 Bessel 校正计算方差：
     - 对于 n > 1：`var = (sum_sq - sum^2/n) / (n-1)`
   - 计算标准误差：
     - Welch：`se = sqrt(var1/n1 + var2/n2)`
     - 合并：`se = sqrt(pooled_var * (1/n1 + 1/n2))`
   - 计算 t 统计量：`t = (mean2 - mean1) / se`
   - 通过正态近似计算 p 值（快速 erfc）
   - 计算 log2 倍数变化：`log2FC = log2((mean2 + eps) / (mean1 + eps))`

**边界条件**

- **空组**：抛出 ArgumentError
- **恒定特征**：t 统计量为 0，p 值为 1.0
- **零方差**：标准误差被限制为 SIGMA_MIN (1e-12)
- **完美分离**：非常大的 t 统计量，非常小的 p 值

**数据保证（前置条件）**

- `matrix.secondary_dim() == group_ids.len`
- 输出数组大小 >= `matrix.primary_dim()`
- `group_ids` 仅包含值 0 或 1
- 两组必须至少有一个成员

**复杂度分析**

- **时间**：O(features * nnz_per_feature) - 对特征并行
- **空间**：O(threads * max_row_length) 工作空间

**示例**

```cpp
#include "scl/kernel/ttest.hpp"

scl::Sparse<Real, true> matrix = /* 特征矩阵 [n_features x n_samples] */;
scl::Array<int32_t> group_ids = /* 二元组分配 */;
scl::Array<Real> t_stats(n_features);
scl::Array<Real> p_values(n_features);
scl::Array<Real> log2_fc(n_features);

scl::kernel::ttest::ttest(
    matrix, group_ids,
    t_stats, p_values, log2_fc,
    true  // use_welch
);

// t_stats[i] = 特征 i 的 t 统计量
// p_values[i] = 双尾 p 值
// log2_fc[i] = log2 倍数变化 (group1/group0)
```

---

### compute_group_stats

::: source_code file="scl/kernel/ttest.hpp" symbol="compute_group_stats" collapsed
:::

**算法说明**

计算每个特征的每组均值、方差和计数：

1. 对于每个特征 f（并行）：
   - 对于每组 g：
     - 提取组 g 中样本的非零值
     - 计算均值：`mean = sum(values) / n_nonzero`
     - 计算方差：`var = sum((values - mean)^2) / (n_nonzero - 1)`
     - 计数：`count = n_nonzero`
2. 输出布局是行主序：`[feat0_g0, feat0_g1, ..., feat1_g0, ...]`

**边界条件**

- **组中无样本**：均值 = 0，方差 = 0，计数 = 0
- **组中单个样本**：方差未定义（设置为 0）
- **组中全零**：均值 = 0，方差 = 0，计数 = 0

**数据保证（前置条件）**

- `group_ids[i]` 在范围 [0, n_groups) 内或为负（忽略）
- 输出数组大小 >= `n_features * n_groups`
- 矩阵必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**：O(features * nnz_per_feature)
- **空间**：O(threads * max_row_length) 工作空间

**示例**

```cpp
Index n_groups = 3;
scl::Array<Real> means(n_features * n_groups);
scl::Array<Real> vars(n_features * n_groups);
scl::Array<Size> counts(n_features * n_groups);

scl::kernel::ttest::compute_group_stats(
    matrix, group_ids, n_groups,
    means, vars, counts
);

// means[f * n_groups + g] = 特征 f 在组 g 中的均值
// vars[f * n_groups + g] = 特征 f 在组 g 中的方差
// counts[f * n_groups + g] = 特征 f 在组 g 中的非零计数
```

---

## 工具函数

### count_groups

计算两组中每个组的元素数。

::: source_code file="scl/kernel/ttest.hpp" symbol="count_groups" collapsed
:::

**复杂度**

- 时间：O(n) 使用 SIMD 优化
- 空间：O(1)

---

## 注意事项

- 建议对不等方差使用 Welch t 检验（默认）
- Student t 检验假设等方差（use_welch=false）
- Log2 倍数变化使用 epsilon (1e-9) 以确保数值稳定性
- P 值默认是双尾的
- 稀疏矩阵操作针对单细胞数据进行了优化

## 相关内容

- [多重检验模块](./multiple_testing) - 用于 FDR 校正
- [统计模块](../math/statistics) - 用于其他统计检验
