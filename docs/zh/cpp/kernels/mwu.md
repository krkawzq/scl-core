# mwu.hpp

> scl/kernel/mwu.hpp · 用于比较两组的 Mann-Whitney U 检验

## 概述

本文件提供 Mann-Whitney U 检验（也称为 Wilcoxon 秩和检验），这是一种用于比较两个独立组的非参数统计检验。与 t 检验等参数检验不同，它不假设数据分布，基于秩进行比较。

主要特性：
- 非参数检验（无分布假设）
- 使用排序秩的基于秩的比较
- 稀疏矩阵的特征级并行计算
- 额外度量：U 统计量、p 值、log2 倍数变化、AUROC
- SIMD 优化的排序和秩计算

**头文件**: `#include "scl/kernel/mwu.hpp"`

---

## 主要 API

### mwu_test

对稀疏矩阵中的每个特征（行/列）执行 Mann-Whitney U 检验，比较两组样本。

::: source_code file="scl/kernel/mwu.hpp" symbol="mwu_test" collapsed
:::

**算法说明**

Mann-Whitney U 检验通过对所有值进行排序并计算 U 统计量来比较两组：

1. 对每个特征并行执行：
   a. 使用预分配的缓冲区按组划分非零值
   b. 使用 VQSort（SIMD 优化）对每组进行排序
   c. 合并排序后的数组，使用并列校正计算秩和
   d. 计算 U 统计量：U = R1 - n1*(n1+1)/2，其中 R1 是组 1 的秩和
   e. 应用连续性校正和正态近似计算 p 值：
      - 均值：mu = n1*n2/2
      - 方差：sigma^2 = n1*n2*(n1+n2+1)/12（含并列校正）
      - Z = (U - mu) / sigma（含连续性校正）
      - 从标准正态分布得到双边 p 值
   f. 从组均值计算 log2 倍数变化
   g. 可选计算 AUROC = U / (n1 * n2)

优化：
- 使用二分搜索查找负/正边界（O(log n)）
- 在合并循环中预取以提高缓存效率
- 预计算倒数以避免除法
- 4 路展开的分区循环
- 用于排序缓冲区的线程本地工作空间

**边界条件**

- **全零特征**：返回 U=0, p=1, AUROC=0.5（组间无差异）
- **空组**：抛出 SCL_CHECK_ARG 错误（两组必须至少有一个成员）
- **值中的并列**：对方差估计应用并列校正
- **非常小的组（n1, n2 < 10）**：正态近似可能不太准确
- **相同的组**：U ≈ n1*n2/2, p ≈ 1.0, AUROC ≈ 0.5

**数据保证（前置条件）**

- `matrix.secondary_dim() == group_ids.len`（每个样本一个标签）
- `out_u_stats.len == out_p_values.len == out_log2_fc.len == matrix.primary_dim()`
- `group_ids` 仅包含值 0 或 1
- 两组必须至少有一个成员
- 如果提供 `out_auroc`：`out_auroc.len == matrix.primary_dim()`

**复杂度分析**

- **时间**：每个特征 O(features * nnz_per_feature * log(nnz_per_feature))
- **空间**：每个线程 O(max_nnz) 用于排序缓冲区

**示例**

```cpp
#include "scl/kernel/mwu.hpp"
#include "scl/core/sparse.hpp"

// 创建稀疏矩阵（特征 x 样本）
Sparse<Real, true> matrix(n_features, n_samples);

// 创建组标签（每个样本为 0 或 1）
Array<int32_t> group_ids(n_samples);
// 填充：group_ids[i] = 0 或 1

// 预分配输出数组
Array<Real> u_stats(n_features);
Array<Real> p_values(n_features);
Array<Real> log2_fc(n_features);
Array<Real> auroc(n_features);  // 可选

// 执行 Mann-Whitney U 检验
scl::kernel::mwu::mwu_test(
    matrix,
    group_ids,
    u_stats,
    p_values,
    log2_fc,
    auroc  // 可选：仅省略以获得 U, p, log2FC
);

// 结果：
// - u_stats[i]: 特征 i 的 U 统计量
// - p_values[i]: 双边 p 值（越小 = 越显著）
// - log2_fc[i]: Log2(mean_group1 / mean_group0)
// - auroc[i]: ROC 曲线下面积 = P(group1 > group0) + 0.5*P(ties)
```

---

## 工具函数

### count_groups

计算每组（0 和 1）中的样本数。使用 SIMD 优化计数。

::: source_code file="scl/kernel/mwu.hpp" symbol="count_groups" collapsed
:::

**复杂度**

- 时间：使用 SIMD 优化 O(n)
- 空间：O(1)

---

## 数值注意事项

- **正态近似**：适用于 n1, n2 >= 10。对于较小的样本，应使用精确的 U 分布（未实现）。
- **连续性校正**：在计算 Z 分数时应用（U ± 0.5）以改善离散到连续的近似。
- **并列校正**：当存在并列时调整方差估计：sigma^2 *= (1 - sum(t^3-t)/(n^3-n))，其中 t 是并列组大小。
- **倍数变化计算**：向均值添加 EPS (1e-9) 以防止除以零：log2((mean1+EPS)/(mean0+EPS))。
- **AUROC 解释**：
  - AUROC = 0.5：组间无差异
  - AUROC = 1.0：所有组1值 > 所有组0值
  - AUROC = 0.0：所有组0值 > 所有组1值
- **U 统计量范围**：U 在 [0, n1*n2] 范围内。在零假设下，E[U] = n1*n2/2。

## 相关内容

- [Multiple Testing](/zh/cpp/kernels/multiple_testing) - p 值的 FDR 校正
- [T 检验](/zh/cpp/kernels/ttest) - MWU 检验的参数替代方法
