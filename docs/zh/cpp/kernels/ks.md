# ks.hpp

> scl/kernel/stat/ks.hpp · Kolmogorov-Smirnov 两样本检验

## 概述

本文件提供 Kolmogorov-Smirnov (KS) 两样本检验，用于比较两组之间的分布。它通过比较经验累积分布函数 (ECDF) 来检验两个样本是否来自同一分布。

**头文件**: `#include "scl/kernel/stat/ks.hpp"`

---

## 主要 API

### ks_test

::: source_code file="scl/kernel/stat/ks.hpp" symbol="ks_test" collapsed
:::

**算法说明**

计算每个特征的两样本 Kolmogorov-Smirnov 检验：

1. 对于每个特征并行处理：
   - 按组（组 0 和组 1）划分非零值
   - 使用 VQSort（高性能排序）对每组排序
   - 合并排序数组同时跟踪 ECDF 差异：
     - 在 ECDF 计算中显式处理稀疏零
     - 值 < 0: 在零点之前贡献到 ECDF
     - 零（隐式）: 在 ECDF 中 x=0 处创建跳跃
     - 值 > 0: 在零点之后贡献到 ECDF
   - 计算 D 统计量：
     - D = max |F1(x) - F2(x)| 对所有 x
     - 其中 F1 和 F2 是组 1 和组 2 的 ECDF
   - 通过 Kolmogorov 分布计算 p 值：
     - n_eff = n1 * n2 / (n1 + n2)
     - lambda = (sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * D
     - P(D > d) = 2 * sum_{k=1}^inf (-1)^{k+1} * exp(-2*k^2*lambda^2)
     - 使用 100 项限制的级数展开

2. D 统计量衡量 ECDF 之间的最大差异

3. P 值使用渐近 Kolmogorov 分布（对 n1, n2 >= 25 准确）

**边界条件**

- **组 0 或组 1 为空**: 抛出 ArgumentError
- **两组中所有值相同**: D = 0, p 值 = 1.0
- **组间无重叠**: D = 1.0, p 值接近 0
- **每组单个值**: 计算 D 但可能功效较低
- **稀疏零**: 在 ECDF 计算中显式处理

**数据保证（前置条件）**

- `matrix.secondary_dim() == group_ids.len`（样本维度必须匹配）
- 输出数组必须具有大小 >= matrix.primary_dim()
- 两组（0 和 1）必须至少有一个成员
- `group_ids` 必须仅包含 0 或 1 值
- 矩阵必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(features * nnz_per_feature * log(nnz_per_feature)) - 排序占主导，然后是 ECDF 计算
- **空间**: O(threads * max_row_length) - 用于排序和 ECDF 跟踪的线程局部工作空间

**示例**

```cpp
#include "scl/kernel/stat/ks.hpp"

// 准备数据
Sparse<Real, true> matrix = /* 特征 x 样本 */;
Array<int32_t> group_ids = /* 二值组分配 (0 或 1) */;

// 预分配输出
Size n_features = matrix.rows();
Array<Real> D_stats(n_features);
Array<Real> p_values(n_features);

// 计算 KS 检验
scl::kernel::stat::ks::ks_test(
    matrix, group_ids,
    D_stats, p_values
);

// 解释结果
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "特征 " << i 
                  << ": D = " << D_stats[i]
                  << ", p = " << p_values[i]
                  << " (分布不同)\n";
    }
}

// 过滤具有不同分布的特征
std::vector<Size> different_features;
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05 && D_stats[i] > 0.3) {
        // D > 0.3 表示显著差异
        different_features.push_back(i);
    }
}
```

---

## 注意事项

**何时使用**: Kolmogorov-Smirnov 检验适用于：
- 比较两组之间的分布
- 无需假设分布形状
- 检测分布的任何差异（位置、尺度、形状）
- t 检验的非参数替代方法

**解释**: 
- D 统计量范围从 0 到 1
- D = 0: 分布相同
- D = 1: 分布无重叠
- 大的 D 和小的 p 值: 分布显著不同
- D > 0.3 通常表示显著差异

**稀疏数据处理**: 算法在 ECDF 计算中显式处理稀疏零：
- 负值在零之前贡献到 ECDF
- 隐式零在 x=0 处创建跳跃
- 正值在零之后贡献到 ECDF
- 这确保稀疏表达数据的正确 ECDF

**准确性**: P 值计算使用渐近 Kolmogorov 分布：
- 对 n1, n2 >= 25 准确
- 使用 100 项限制的级数展开
- 对于较小样本，考虑置换检验

**线程安全**: 使用线程局部工作空间在特征上并行处理，对并发执行安全。

**与其他检验的比较**:
- **KS 检验**: 检测任何分布差异，非参数
- **t 检验**: 假设正态性，检测均值差异
- **Mann-Whitney U**: 非参数，检测位置偏移

---

## 相关内容

- [Mann-Whitney U](/zh/cpp/kernels/mwu) - 非参数位置检验
- [Permutation Test](/zh/cpp/kernels/permutation_stat) - 精确置换检验
- [T-test](/zh/cpp/kernels/ttest) - 参数均值比较

