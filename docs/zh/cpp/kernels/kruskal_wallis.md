# kruskal_wallis.hpp

> scl/kernel/stat/kruskal_wallis.hpp · 用于非参数方差分析的 Kruskal-Wallis H 检验

## 概述

本文件提供 Kruskal-Wallis H 检验，这是单因素方差分析的非参数替代方法。它检验不同组的样本是否来自同一分布，无需假设正态性。

**头文件**: `#include "scl/kernel/stat/kruskal_wallis.hpp"`

---

## 主要 API

### kruskal_wallis

::: source_code file="scl/kernel/stat/kruskal_wallis.hpp" symbol="kruskal_wallis" collapsed
:::

**算法说明**

计算 k 组的 Kruskal-Wallis H 检验（非参数方差分析）：

1. 对于每个特征并行处理：
   - 提取带组标签的非零值
   - 使用 argsort 对值排序（保留索引映射）
   - 计算每组的秩和，处理并列值：
     - 为并列值分配平均秩
     - 累加每组的秩
   - 计算 H 统计量：
     - H = 12/(N(N+1)) * sum(R_i^2/n_i) - 3(N+1)
     - 其中 R_i 是组 i 的秩和，n_i 是组 i 的大小，N 是总样本数
   - 应用并列校正：
     - H_corrected = H / (1 - sum(t^3-t)/(N^3-N))
     - 其中 t 是每个秩的并列数
   - 从自由度为 k - 1 的卡方分布计算 p 值

2. H 统计量衡量组间是否存在显著差异

3. P 值使用卡方近似（对大样本准确）

**边界条件**

- **n_groups < 2**: 抛出 ArgumentError
- **少于 2 个组有成员**: 抛出 ArgumentError
- **所有值相同**: H = 0, p 值 = 1.0
- **每组单个值**: 计算 H 统计量但可能功效较低
- **负 group_ids**: 忽略（视为缺失数据）
- **空特征**: H = NaN, p 值 = NaN

**数据保证（前置条件）**

- `n_groups >= 2`（必须至少有 2 个组）
- `group_ids[i]` 在范围 [0, n_groups) 内或为负数（忽略）
- 至少 2 个组必须有成员
- 输出数组必须具有大小 >= matrix.primary_dim()
- 矩阵必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(features * nnz_per_feature * log(nnz_per_feature)) - 排序占主导，然后是秩计算
- **空间**: O(threads * (max_row_length + n_groups)) - 用于排序和组统计的线程局部工作空间

**示例**

```cpp
#include "scl/kernel/stat/kruskal_wallis.hpp"

// 准备数据
Sparse<Real, true> matrix = /* 特征 x 样本 */;
Array<int32_t> group_ids = /* 组分配 [0, k-1] */;
Size n_groups = 3;  // k 个组

// 预分配输出
Size n_features = matrix.rows();
Array<Real> H_stats(n_features);
Array<Real> p_values(n_features);

// 计算 Kruskal-Wallis 检验
scl::kernel::stat::kruskal_wallis::kruskal_wallis(
    matrix, group_ids, n_groups,
    H_stats, p_values
);

// 解释结果
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "特征 " << i 
                  << ": H = " << H_stats[i]
                  << ", p = " << p_values[i]
                  << " (显著)\n";
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

## 注意事项

**何时使用**: Kruskal-Wallis 适用于：
- 数据不服从正态分布
- 样本量小
- 存在异常值
- 方差齐性假设不满足

**解释**: 
- 大的 H 统计量表示组间存在显著差异
- P 值 < 0.05 表明至少一组与其他组不同
- 需要事后检验（如 Dunn 检验）来识别哪些组不同

**并列处理**: 算法通过分配平均秩并应用并列校正到 H 统计量来正确处理并列值。

**线程安全**: 使用线程局部工作空间在特征上并行处理，对并发执行安全。

**与 ANOVA 的比较**: 
- Kruskal-Wallis: 非参数，对异常值稳健，无需正态性假设
- 单因素 ANOVA: 参数，假设正态性，满足假设时功效更高

---

## 相关内容

- [One-way ANOVA](/zh/cpp/kernels/oneway_anova) - 参数替代方法
- [Mann-Whitney U](/zh/cpp/kernels/mwu) - 两组非参数检验
- [Permutation Test](/zh/cpp/kernels/permutation_stat) - 精确置换检验

