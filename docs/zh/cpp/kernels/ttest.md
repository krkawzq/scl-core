# T 检验

用于比较两组样本的 Welch 和 Student t 检验。

## 概述

T 检验操作提供：

- **参数检验** - 假设正态分布
- **两种变体** - Welch（不等方差）或 Student（等方差）
- **多特征** - 并行测试所有特征
- **附加指标** - T 统计量、p 值、log2 倍数变化

## 基本用法

### ttest

对稀疏矩阵中的每个特征执行 t 检验，比较两组。

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // 特征 x 样本
Array<int32_t> group_ids = /* ... */;    // 每个样本为 0 或 1

Array<Real> t_stats(matrix.primary_dim());
Array<Real> p_values(matrix.primary_dim());
Array<Real> log2_fc(matrix.primary_dim());

// Welch t 检验（默认，处理不等方差）
scl::kernel::ttest::ttest(
    matrix,
    group_ids,
    t_stats,
    p_values,
    log2_fc,
    true  // use_welch = true
);

// Student t 检验（假设等方差）
scl::kernel::ttest::ttest(
    matrix,
    group_ids,
    t_stats,
    p_values,
    log2_fc,
    false  // use_welch = false
);
```

**参数：**
- `matrix` [in] - 稀疏矩阵（特征 x 样本）
- `group_ids` [in] - 二分组分配（0 或 1）
- `out_t_stats` [out] - 每个特征的 T 统计量
- `out_p_values` [out] - 每个特征的双侧 p 值
- `out_log2_fc` [out] - Log2 倍数变化（组1 / 组0）
- `use_welch` [in] - 使用 Welch t 检验（默认: true）

**前置条件：**
- `matrix.secondary_dim() == group_ids.len`
- 所有输出数组长度为 `matrix.primary_dim()`
- `group_ids` 仅包含值 0 或 1
- 两组必须至少各有一个成员

**后置条件：**
- `out_t_stats[i]` 包含特征 i 的 t 统计量
- `out_p_values[i]` 包含双侧 p 值
- `out_log2_fc[i] = log2((mean_group1 + eps) / (mean_group0 + eps))`

**算法：**
对每个特征并行：
1. 按组划分非零值（4 路展开，预取）
2. 在划分期间累加 sum 和 sum_sq
3. 计算包括零的均值：mean = sum / n_total
4. 使用零调整和 Bessel 校正计算方差
5. 计算标准误差：
   - Welch: se = sqrt(var1/n1 + var2/n2)
   - Pooled: se = sqrt(pooled_var * (1/n1 + 1/n2))
6. t_stat = (mean2 - mean1) / se
7. 通过正态近似计算 p 值（快速 erfc）

**复杂度：**
- 时间: O(特征数 * 每特征非零数)
- 空间: O(线程数 * 最大行长度) 用于工作空间

**线程安全：**
安全 - 按特征并行，使用线程本地工作空间

**抛出：**
`ArgumentError` - 如果任一组为空

**数值说明：**
- EPS = 1e-9 添加到均值以稳定 log2FC
- SIGMA_MIN = 1e-12 有效标准误差的阈值
- 方差被钳制为 >= 0 以保持数值稳定性
- 使用快速 erfc 近似（最大误差 < 1.5e-7）

## 辅助函数

### count_groups

统计两组中的元素数。

```cpp
Size n1, n2;
scl::kernel::ttest::count_groups(group_ids, n1, n2);
// n1 = 组 0 的计数
// n2 = 组 1 的计数
```

**参数：**
- `group_ids` [in] - 组分配数组（0 或 1）
- `out_n1` [out] - group_id == 0 的元素计数
- `out_n2` [out] - group_id == 1 的元素计数

**复杂度：**
- 时间: O(n)，带 SIMD 优化
- 空间: O(1)

## 遗留函数

### compute_group_stats

计算每个特征的每组均值、方差和计数。为向后兼容而保留。

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<int32_t> group_ids = /* ... */;
Size n_groups = 2;

Array<Real> means(matrix.primary_dim() * n_groups);
Array<Real> vars(matrix.primary_dim() * n_groups);
Array<Size> counts(matrix.primary_dim() * n_groups);

scl::kernel::ttest::compute_group_stats(
    matrix,
    group_ids,
    n_groups,
    means,
    vars,
    counts
);
```

**参数：**
- `matrix` [in] - 稀疏矩阵，形状 (n_features, n_samples)
- `group_ids` [in] - 每个样本的组分配
- `n_groups` [in] - 组数
- `out_means` [out] - 组均值，大小 = n_features * n_groups
- `out_vars` [out] - 组方差，大小 = n_features * n_groups
- `out_counts` [out] - 组计数，大小 = n_features * n_groups

**输出布局：**
行主序: [feat0_g0, feat0_g1, ..., feat1_g0, feat1_g1, ...]

**注意：**
对于两组比较，考虑直接使用 `ttest()`。此函数为 k 组场景保留。

## 使用场景

### 差异表达分析

比较两种条件之间的基因表达：

```cpp
Sparse<Real, true> expression_matrix = /* ... */;  // 基因 x 细胞
Array<int32_t> condition_labels = /* ... */;      // 0=对照, 1=处理

Array<Real> t_stats(expression_matrix.primary_dim());
Array<Real> p_values(expression_matrix.primary_dim());
Array<Real> log2_fc(expression_matrix.primary_dim());

scl::kernel::ttest::ttest(
    expression_matrix,
    condition_labels,
    t_stats,
    p_values,
    log2_fc
);

// 查找显著差异表达的基因
for (Size i = 0; i < expression_matrix.primary_dim(); ++i) {
    if (p_values[i] < 0.05 && std::abs(log2_fc[i]) > 1.0) {
        // 基因 i 显著差异表达
    }
}
```

### 多组比较

对 k 组场景使用 compute_group_stats：

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<int32_t> group_ids = /* ... */;  // 0, 1, 2, ...
Size n_groups = 3;

Array<Real> means(matrix.primary_dim() * n_groups);
Array<Real> vars(matrix.primary_dim() * n_groups);
Array<Size> counts(matrix.primary_dim() * n_groups);

scl::kernel::ttest::compute_group_stats(
    matrix, group_ids, n_groups, means, vars, counts
);

// 访问特征 f、组 g 的统计：
// mean = means[f * n_groups + g]
// var = vars[f * n_groups + g]
// count = counts[f * n_groups + g]
```

### Welch vs Student t 检验

根据方差假设选择：

```cpp
// Welch t 检验（默认）- 处理不等方差
scl::kernel::ttest::ttest(
    matrix, group_ids, t_stats, p_values, log2_fc, true
);

// Student t 检验 - 假设等方差（更快）
scl::kernel::ttest::ttest(
    matrix, group_ids, t_stats, p_values, log2_fc, false
);
```

## 性能

### 并行化

- 按特征并行
- 线程本地工作空间用于划分缓冲区
- 无同步开销

### SIMD 优化

- SIMD 优化的组大小计数
- 4 路展开的划分循环
- 间接访问的预取

### 内存效率

- 预分配工作空间池
- 可重用划分缓冲区
- 最小分配

## 统计细节

### Welch t 检验

对于不等方差：
- 标准误差: se = sqrt(var1/n1 + var2/n2)
- 自由度: 通过 Satterthwaite 公式近似
- 当方差不同时更稳健

### Student t 检验

对于等方差：
- 合并方差: pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
- 标准误差: se = sqrt(pooled_var * (1/n1 + 1/n2))
- 自由度: n1 + n2 - 2
- 计算更快

### P 值计算

使用快速 erfc 近似：
- 双侧检验: p = 2 * erfc(|t| / sqrt(2))
- 最大误差 < 1.5e-7
- 比精确计算快得多

## 参见

- [Mann-Whitney U](/zh/cpp/kernels/mwu) - 非参数替代方法
- [统计](/zh/cpp/kernels/statistics) - 其他统计检验
