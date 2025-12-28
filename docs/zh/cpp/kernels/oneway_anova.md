# oneway_anova.hpp

> scl/kernel/stat/oneway_anova.hpp · 用于参数组比较的单因素方差分析 F 检验

## 概述

本文件提供单因素方差分析 (ANOVA) F 检验，用于比较 k 组之间的均值。这是一个参数检验，假设正态性和方差齐性。

**头文件**: `#include "scl/kernel/stat/oneway_anova.hpp"`

---

## 主要 API

### oneway_anova

::: source_code file="scl/kernel/stat/oneway_anova.hpp" symbol="oneway_anova" collapsed
:::

**算法说明**

计算 k 组的单因素方差分析 F 检验（参数）：

1. 对于每个特征并行处理：
   - 计算组和和总均值（包括稀疏零）：
     - 对所有组的所有值求和
     - 计算总样本数（包括隐式零）
     - grand_mean = total_sum / total_count
   - 计算组均值：
     - 对于每组 g: mean_g = sum_g / n_g
     - 其中 sum_g 是组 g 中值的总和，n_g 是组 g 的大小
   - 计算平方和：
     - SS_between = sum(n_g * (mean_g - grand_mean)^2)
     - SS_total = sum((x_i - grand_mean)^2) 对所有观测值
     - SS_within = SS_total - SS_between
   - 计算 F 统计量：
     - df_between = k - 1
     - df_within = N - k
     - MS_between = SS_between / df_between
     - MS_within = SS_within / df_within
     - F = MS_between / MS_within
   - 从 F 分布计算 p 值：
     - 使用 Wilson-Hilferty 近似进行 F 分布
     - P 值 = P(F(df1, df2) >= F_observed)

2. F 统计量衡量组间方差与组内方差的比率

3. 大的 F 和小的 p 值表示组均值显著不同

**边界条件**

- **n_groups < 2**: 抛出 ArgumentError
- **少于 2 个组有成员**: 抛出 ArgumentError
- **N <= k (总样本数 <= 组数)**: 抛出 ArgumentError（自由度不足）
- **所有值相同**: F = 0, p 值 = 1.0
- **每组单个值**: 计算 F 但可能功效较低
- **负 group_ids**: 忽略（视为缺失数据）
- **空特征**: F = NaN, p 值 = NaN

**数据保证（前置条件）**

- `n_groups >= 2`（必须至少有 2 个组）
- `group_ids[i]` 在范围 [0, n_groups) 内或为负数（忽略）
- 至少 2 个组必须有成员
- `N > k`（总样本数 > 组数，用于自由度）
- 输出数组必须具有大小 >= matrix.primary_dim()
- 矩阵必须是有效的 CSR 或 CSC 格式
- 假设正态性和方差齐性（不强制执行，但需要有效性）

**复杂度分析**

- **时间**: O(features * nnz_per_feature) - 每个特征单次遍历数据
- **空间**: O(threads * n_groups) - 用于组统计的线程局部工作空间

**示例**

```cpp
#include "scl/kernel/stat/oneway_anova.hpp"

// 准备数据
Sparse<Real, true> matrix = /* 特征 x 样本 */;
Array<int32_t> group_ids = /* 组分配 [0, k-1] */;
Size n_groups = 3;  // k 个组

// 预分配输出
Size n_features = matrix.rows();
Array<Real> F_stats(n_features);
Array<Real> p_values(n_features);

// 计算单因素方差分析
scl::kernel::stat::oneway_anova::oneway_anova(
    matrix, group_ids, n_groups,
    F_stats, p_values
);

// 解释结果
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "特征 " << i 
                  << ": F = " << F_stats[i]
                  << ", p = " << p_values[i]
                  << " (组均值不同)\n";
    }
}

// 过滤显著特征
std::vector<Size> significant_features;
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        significant_features.push_back(i);
    }
}
```

---

## 工具函数

### count_k_groups

计算 k 组中每组的样本数。

::: source_code file="scl/kernel/stat/oneway_anova.hpp" symbol="count_k_groups" collapsed
:::

**复杂度**

- 时间: O(n_samples)
- 空间: O(1) 辅助空间

---

## 注意事项

**何时使用**: 单因素方差分析适用于：
- 数据服从正态分布（或样本量大）
- 组间方差齐性
- 比较 k 组之间的均值（k >= 2）
- Kruskal-Wallis 的参数替代方法

**假设**:
- **正态性**: 每组中的数据应服从正态分布
- **方差齐性**: 组间方差应相等
- **独立性**: 样本应独立

**解释**: 
- 大的 F 统计量表示组均值显著不同
- P 值 < 0.05 表明至少一组均值与其他组不同
- 需要事后检验（如 Tukey HSD）来识别哪些组不同
- F > 1 表示组间方差超过组内方差

**稀疏数据处理**: 算法正确处理稀疏零：
- 隐式零包含在均值和方差计算中
- 总计数包括零以正确计算自由度
- 组均值考虑所有样本（包括零）

**敏感性**: 方差分析对以下情况敏感：
- 异常值（考虑稳健替代方法）
- 非正态性（考虑 Kruskal-Wallis）
- 不等方差（考虑 Welch 方差分析）

**线程安全**: 使用线程局部工作空间在特征上并行处理，对并发执行安全。

**与 Kruskal-Wallis 的比较**: 
- **ANOVA**: 参数，假设正态性，满足假设时功效更高
- **Kruskal-Wallis**: 非参数，对异常值稳健，无需正态性假设

---

## 相关内容

- [Kruskal-Wallis](/zh/cpp/kernels/kruskal_wallis) - 非参数替代方法
- [T-test](/zh/cpp/kernels/ttest) - 两组参数检验
- [Permutation Test](/zh/cpp/kernels/permutation_stat) - 精确置换检验

