# Mann-Whitney U 检验

Mann-Whitney U 检验（Wilcoxon 秩和检验），用于比较两组样本。

## 概述

Mann-Whitney U 检验提供：

- **非参数检验** - 无需分布假设
- **基于秩** - 使用秩而非原始值
- **多特征** - 并行测试所有特征
- **附加指标** - U 统计量、p 值、倍数变化、AUROC

## 基本用法

### mwu_test

对稀疏矩阵中的每个特征执行 Mann-Whitney U 检验。

```cpp
#include "scl/kernel/mwu.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // 特征 x 样本
Array<int32_t> group_ids = /* ... */;    // 每个样本为 0 或 1

Array<Real> u_stats(matrix.primary_dim());
Array<Real> p_values(matrix.primary_dim());
Array<Real> log2_fc(matrix.primary_dim());
Array<Real> auroc(matrix.primary_dim());

scl::kernel::mwu::mwu_test(
    matrix,
    group_ids,
    u_stats,
    p_values,
    log2_fc,
    auroc  // 可选
);
```

**参数：**
- `matrix` [in] - 稀疏矩阵（特征 x 样本）
- `group_ids` [in] - 每个样本的组标签（0 或 1）
- `out_u_stats` [out] - 每个特征的 U 统计量
- `out_p_values` [out] - 每个特征的双侧 p 值
- `out_log2_fc` [out] - Log2 倍数变化（组1 / 组0）
- `out_auroc` [out] - 可选：AUROC 值（U / (n1 * n2)）

**前置条件：**
- `matrix.secondary_dim() == group_ids.len`
- 所有输出数组长度为 `matrix.primary_dim()`
- `group_ids` 仅包含值 0 或 1
- 两组必须至少各有一个成员

**后置条件：**
- `out_u_stats[i]` 包含特征 i 的 U 统计量
- `out_p_values[i]` 包含双侧 p 值（正态近似）
- `out_log2_fc[i]` 包含 log2(mean_group1 / mean_group0)
- `out_auroc[i]`（如果提供）包含 AUROC = U / (n1 * n2)
- 对于全零特征：U=0, p=1, AUROC=0.5

**算法：**
对每个特征并行：
1. 使用预分配缓冲区按组划分非零值
2. 使用 VQSort（SIMD 优化）对每组排序
3. 合并排序数组以计算带结校正的秩和
4. 计算 U 统计量：U = R1 - n1*(n1+1)/2
5. 应用连续性校正和正态近似计算 p 值
6. 从组均值计算 log2 倍数变化
7. 可选计算 AUROC = U / (n1 * n2)

**优化：**
- 负/正边界二分搜索（O(log n)）
- 合并循环中的预取以提高缓存效率
- 预计算倒数以避免除法
- 4 路展开的划分循环

**复杂度：**
- 时间: O(特征数 * 每特征非零数 * log(每特征非零数))
- 空间: 每个线程 O(最大非零数) 用于排序缓冲区

**线程安全：**
安全 - 按特征并行，使用线程本地工作空间

**抛出：**
`SCL_CHECK_ARG` - 如果任一组为空

**数值说明：**
- 使用正态近似计算 p 值（适用于 n1, n2 >= 10）
- 应用连续性校正用于离散到连续近似
- 对方差估计应用结校正
- 在倍数变化的均值中添加 EPS (1e-9) 以防止除零
- AUROC 在 [0, 1] 中表示 P(组1 > 组0) + 0.5 * P(相等)

## 辅助函数

### count_groups

统计每组中的样本数。

```cpp
Size n1, n2;
scl::kernel::mwu::count_groups(group_ids, n1, n2);
// n1 = 组 0 的计数
// n2 = 组 1 的计数
```

**参数：**
- `group_ids` [in] - 组标签数组（0 或 1）
- `out_n1` [out] - 组 0 的样本数
- `out_n2` [out] - 组 1 的样本数

**算法：**
使用 SIMD 优化的 `scl::vectorize::count` 进行并行计数。

**复杂度：**
- 时间: O(n)
- 空间: O(1)

## 使用场景

### 差异表达分析

比较两种条件之间的基因表达：

```cpp
Sparse<Real, true> expression_matrix = /* ... */;  // 基因 x 细胞
Array<int32_t> condition_labels = /* ... */;      // 0=对照, 1=处理

Array<Real> u_stats(expression_matrix.primary_dim());
Array<Real> p_values(expression_matrix.primary_dim());
Array<Real> log2_fc(expression_matrix.primary_dim());

scl::kernel::mwu::mwu_test(
    expression_matrix,
    condition_labels,
    u_stats,
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

### 特征选择

按效应大小对特征排序：

```cpp
// 计算 AUROC 用于排序
Array<Real> auroc(matrix.primary_dim());
scl::kernel::mwu::mwu_test(
    matrix, group_ids, u_stats, p_values, log2_fc, auroc
);

// 按 AUROC 排序（越高 = 分离越好）
std::vector<std::pair<Real, Size>> ranked;
for (Size i = 0; i < matrix.primary_dim(); ++i) {
    ranked.push_back({auroc[i], i});
}
std::sort(ranked.rbegin(), ranked.rend());
```

### 质量控制

检查组是否分离良好：

```cpp
Array<Real> auroc(matrix.primary_dim());
scl::kernel::mwu::mwu_test(
    matrix, group_ids, u_stats, p_values, log2_fc, auroc
);

Real mean_auroc = 0.0;
for (Size i = 0; i < matrix.primary_dim(); ++i) {
    mean_auroc += auroc[i];
}
mean_auroc /= matrix.primary_dim();

// mean_auroc 接近 0.5: 组相似
// mean_auroc 接近 1.0: 组分离良好
```

## 性能

### 并行化

- 按特征并行
- 线程本地工作空间用于排序缓冲区
- 无同步开销

### SIMD 优化

- SIMD 优化的组大小计数
- VQSort 快速排序（SIMD 优化）
- 合并循环中的预取

### 内存效率

- 预分配工作空间池
- 可重用排序缓冲区
- 最小分配

## 统计细节

### U 统计量

U = R1 - n1*(n1+1)/2

其中：
- R1 = 组 1 的秩和
- n1 = 组 1 的大小

### P 值计算

使用正态近似：
- 均值: n1*n2/2
- 方差: n1*n2*(n1+n2+1)/12（带结校正）
- 连续性校正: ±0.5

### AUROC 解释

AUROC = U / (n1 * n2)

- 0.5: 无分离（随机）
- 1.0: 完美分离
- < 0.5: 组 0 倾向于有更高的值

## 参见

- [T 检验](/zh/cpp/kernels/ttest) - 参数替代方法
- [统计](/zh/cpp/kernels/statistics) - 其他统计检验

