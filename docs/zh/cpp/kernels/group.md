# group.hpp

> scl/kernel/group.hpp · 用于计算分组统计量的分组聚合内核

## 概述

本文件为稀疏矩阵提供高性能的分组聚合操作。它计算每个特征在每个组中的均值和方差统计量，支持多个组和灵活的零值处理选项。

本文件提供：
- 分组统计量计算（均值和方差）
- 支持任意数量的组
- 可选的零值包含/排除统计
- 4路展开的 SIMD 优化累加

**头文件**: `#include "scl/kernel/group.hpp"`

---

## 主要 API

### group_stats

::: source_code file="scl/kernel/group.hpp" symbol="group_stats" collapsed
:::

**算法说明**

计算稀疏矩阵中每个特征在每个组中的均值和方差：

1. **并行特征处理**：对每个特征并行处理：
   - 初始化线程局部累加器（每个组的 sum 和 sum_sq）
   - 使用 4 路展开循环遍历非零元素
   - 使用预取进行间接组访问（通过 group_ids[indices[k]]）
   - 使用间接寻址累加每个组的 sum 和 sum_sq

2. **统计量最终化**：对每个组：
   - 确定样本数 N：
     - 如果 include_zeros：N = group_sizes[g]
     - 否则：N = nnz_counts[g]（组中非零值的计数）
   - 计算均值：mean[g] = sum[g] / N[g]
   - 计算方差：var[g] = (sum_sq[g] - N[g] * mean[g]^2) / (N[g] - ddof)
   - 处理边界情况：如果 N <= ddof，设置 mean=0 和 var=0

3. **输出布局**：结果按特征以行主序格式存储：
   - 特征 f、组 g 的索引：(f * n_groups + g)
   - 布局：[feat0_g0, feat0_g1, ..., feat0_gN, feat1_g0, ...]

**边界条件**

- **空组**：N <= ddof 的组具有 mean=0, var=0
- **负组 ID**：忽略具有负 group_ids 的样本
- **无效组 ID**：忽略 [0, n_groups) 范围外的组 ID
- **零方差**：为数值稳定性，方差被限制为 >= 0
- **组中全为零**：如果 include_zeros=false 且组没有非零值，则 N=0, mean=0, var=0

**数据保证（前置条件）**

- `group_ids[i]` 必须在 [0, n_groups) 范围内或为负（将被忽略）
- `group_sizes.len >= n_groups`（必须包含每个组的大小）
- `out_means.len >= n_features * n_groups`（预分配的输出缓冲区）
- `out_vars.len >= n_features * n_groups`（预分配的输出缓冲区）
- 矩阵必须是有效的 CSR 格式（排序的索引，无重复）
- 如果 include_zeros=false，需要额外的内存用于每个线程的 nnz_counts

**复杂度分析**

- **时间**：O(nnz + n_features * n_groups)
  - O(nnz) 用于遍历所有非零元素
  - O(n_features * n_groups) 用于最终化统计量
- **空间**：每个线程 O(n_groups) 用于累加器
  - 如果 include_zeros=false，每个线程额外 O(n_groups) 用于 nnz_counts
  - n_groups <= 256 时使用栈分配，更大时使用堆分配

**示例**

```cpp
#include "scl/kernel/group.hpp"
#include "scl/core/sparse.hpp"

// 表达矩阵：基因 x 细胞
Sparse<Real, true> expression = /* ... */;
Index n_genes = expression.rows();
Index n_cells = expression.cols();

// 每个细胞的细胞类型标签
Array<int32_t> cell_type_labels(n_cells);
// ... 分配标签：0, 1, 2, ... (n_types - 1) ...

Size n_types = 5;

// 计算组大小
Array<Size> type_sizes(n_types, 0);
for (Index i = 0; i < n_cells; ++i) {
    if (cell_type_labels[i] >= 0 && cell_type_labels[i] < n_types) {
        type_sizes[cell_type_labels[i]]++;
    }
}

// 预分配输出缓冲区
Array<Real> means(n_genes * n_types);
Array<Real> vars(n_genes * n_types);

// 计算分组统计量
scl::kernel::group::group_stats(
    expression,
    cell_type_labels,
    n_types,
    type_sizes,
    means,
    vars,
    1,      // ddof = 1（样本方差）
    true    // include_zeros = true
);

// 访问基因 g 在细胞类型 t 中的平均表达
Real mean_expr = means[g * n_types + t];
Real var_expr = vars[g * n_types + t];

// 示例：比较组间表达
Real mean_type0 = means[gene_idx * n_types + 0];
Real mean_type1 = means[gene_idx * n_types + 1];
Real fold_change = mean_type1 / (mean_type0 + 1e-10);
```

---

## 工具函数

### detail::finalize_stats

内部辅助函数，将累加的和转换为均值和方差。

::: source_code file="scl/kernel/group.hpp" symbol="detail::finalize_stats" collapsed
:::

**复杂度**

- 时间：O(n_groups)
- 空间：O(1)

---

## 注意事项

**内存优化**：
- n_groups <= 256 时使用栈分配（典型情况）
- 更大的组数使用堆分配以避免栈溢出

**数值稳定性**：
- 使用 Welford 风格的方差公式以提高数值稳定性
- 方差被限制为 >= 0 以处理浮点舍入误差
- 优雅地处理除零情况（N <= ddof 情况）

**零值处理**：
- `include_zeros=true`：计算组中所有样本，包括零值
  - 均值反映包括非表达样本在内的总体表达
  - 适用于比例/百分比计算
- `include_zeros=false`：仅计算非零样本
  - 均值反映表达样本中的平均表达水平
  - 适用于平均表达水平计算

**输出布局**：
结果按特征以行主序格式存储：
- 对于特征 f，组 g：索引 = f * n_groups + g
- 此布局便于下游分析的高效访问模式

## 相关内容

- [T-test](/zh/cpp/kernels/ttest) - 两组比较的统计检验
- [Statistics](/zh/cpp/kernels/statistics) - 其他统计操作
