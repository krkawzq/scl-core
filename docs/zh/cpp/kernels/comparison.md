# comparison.hpp

> scl/kernel/comparison.hpp · 组比较和差异丰度分析

## 概述

本文件提供用于比较组、检验差异丰度和计算效应量的统计内核。支持组成分析、丰度检验、多样本差异丰度（DAseq/Milo 风格）和条件响应分析。

**头文件**: `#include "scl/kernel/comparison.hpp"`

---

## 主要 API

### composition_analysis

::: source_code file="scl/kernel/comparison.hpp" symbol="composition_analysis" collapsed
:::

**算法说明**

使用卡方检验分析跨条件的细胞类型组成：

1. 单次遍历中统计每个类型每个条件的细胞数
2. 计算比例：`proportions[t * n_conditions + c] = count[t,c] / total_cells[c]`
3. 对于每个细胞类型：
   - 在零假设（等比例）下计算期望计数
   - 计算卡方统计量：sum((observed - expected)^2 / expected)
   - 使用 Wilson-Hilferty 近似转换为 p 值以提高准确性
4. 在细胞类型上并行化以提高效率

**边界条件**

- **空条件**: 零细胞的条件产生 NaN 比例
- **单一细胞类型**: 如果只存在一种类型，p 值为 1.0（无变异）
- **零计数**: 在所有条件中零细胞的类型产生 NaN p 值

**数据保证（前置条件）**

- `cell_types.len == conditions.len == n_cells`
- `proportions` 容量 >= `n_types * n_conditions`
- `p_values` 容量 >= n_types
- 所有细胞类型索引 < n_types
- 所有条件索引 < n_conditions

**复杂度分析**

- **时间**: O(n_cells + n_types * n_conditions) - 线性扫描加上成对计算
- **空间**: O(n_types * n_conditions) 辅助空间 - 计数矩阵

**示例**

```cpp
#include "scl/kernel/comparison.hpp"

Array<const Index> cell_types = /* 细胞类型标签 [n_cells] */;
Array<const Index> conditions = /* 条件标签 [n_cells] */;
Real* proportions = new Real[n_types * n_conditions];
Real* p_values = new Real[n_types];

scl::kernel::comparison::composition_analysis(
    cell_types,
    conditions,
    proportions,
    p_values,
    n_types,
    n_conditions
);

// 检查显著的组成变化
for (Size t = 0; t < n_types; ++t) {
    if (p_values[t] < 0.05) {
        // 类型 t 在条件间显示显著的组成变化
    }
}
```

---

### abundance_test

::: source_code file="scl/kernel/comparison.hpp" symbol="abundance_test" collapsed
:::

**算法说明**

使用 Fisher 精确检验检验两个条件之间簇的差异丰度：

1. 统计每个簇每个条件的细胞数（条件必须是二元的：0 或 1）
2. 计算比例：`prop[c, cond] = count[c, cond] / total_cells[cond]`
3. 对于每个簇：
   - 计算 log2 倍数变化：`log2(prop[c, 1] / prop[c, 0])`
   - 构建 2x2 列联表：[count[c,0], count[other,0]; count[c,1], count[other,1]]
   - 使用卡方近似计算 Fisher 精确检验 p 值
4. 在簇上并行化

**边界条件**

- **条件 0 中零比例**: 倍数变化为 +infinity
- **条件 1 中零比例**: 倍数变化为 -infinity
- **两者都为零**: 倍数变化为 NaN，p 值为 1.0
- **小计数**: 卡方近似在计数 >= 5 时有效

**数据保证（前置条件）**

- `cluster_labels.len == condition.len`
- `fold_changes` 容量 >= n_clusters
- `p_values` 容量 >= n_clusters
- 条件标签严格为 0 或 1（二元）

**复杂度分析**

- **时间**: O(n_cells + n_clusters) - 单次遍历加上簇级计算
- **空间**: O(n_clusters) 辅助空间 - 计数数组

**示例**

```cpp
Array<const Index> cluster_labels = /* 簇分配 [n_cells] */;
Array<const Index> condition = /* 条件标签（0 或 1）[n_cells] */;
Array<Real> fold_changes(n_clusters);
Array<Real> p_values(n_clusters);

scl::kernel::comparison::abundance_test(
    cluster_labels,
    condition,
    fold_changes,
    p_values
);

// 找到显著富集的簇
for (Index c = 0; c < n_clusters; ++c) {
    if (p_values[c] < 0.05 && fold_changes[c] > 1.0) {
        // 簇 c 在条件 1 中显著富集
    }
}
```

---

### differential_abundance

::: source_code file="scl/kernel/comparison.hpp" symbol="differential_abundance" collapsed
:::

**算法说明**

使用 Wilcoxon 秩和检验检验跨样本的差异丰度（DAseq/Milo 风格）：

1. 将每个样本映射到其条件
2. 统计每个簇每个样本的细胞数
3. 计算每个样本的比例：`prop[s, c] = count[s, c] / total_cells[s]`
4. 对于每个簇：
   - 分别收集条件 0 和条件 1 的比例
   - 计算 log2 倍数变化：`log2(mean(prop[cond1]) / mean(prop[cond0]))`
   - 对条件间的比例执行 Wilcoxon 秩和检验
5. 使用工作空间池进行线程局部缓冲区

**边界条件**

- **每个条件单一样本**: 无法计算 Wilcoxon 检验，返回 NaN
- **样本中零细胞**: 样本从分析中排除
- **所有样本相同条件**: 返回 NaN p 值
- **并列秩**: 在 Wilcoxon 检验中使用标准并列校正

**数据保证（前置条件）**

- `cluster_labels.len == sample_ids.len == conditions.len`
- `da_scores` 容量 >= n_clusters
- `p_values` 容量 >= n_clusters
- 至少需要 2 个条件和 2 个样本

**复杂度分析**

- **时间**: O(n_cells + n_clusters * n_samples) - 计数加上每个簇的统计
- **空间**: O(n_clusters * n_samples) 辅助空间 - 比例矩阵

**示例**

```cpp
Array<const Index> cluster_labels = /* 簇标签 [n_cells] */;
Array<const Index> sample_ids = /* 样本 ID [n_cells] */;
Array<const Index> conditions = /* 条件标签 [n_cells] */;
Array<Real> da_scores(n_clusters);
Array<Real> p_values(n_clusters);

scl::kernel::comparison::differential_abundance(
    cluster_labels,
    sample_ids,
    conditions,
    da_scores,
    p_values
);

// 过滤显著的 DA 簇
for (Index c = 0; c < n_clusters; ++c) {
    if (p_values[c] < 0.05 && std::abs(da_scores[c]) > 1.0) {
        // 簇 c 显示显著的差异丰度
    }
}
```

---

### condition_response

::: source_code file="scl/kernel/comparison.hpp" symbol="condition_response" collapsed
:::

**算法说明**

使用 Wilcoxon 秩和检验检验条件间基因表达响应：

1. 并行处理每个基因：
   - 使用稀疏矩阵中的二分搜索提取条件 0 的表达值
   - 使用二分搜索提取条件 1 的表达值
   - 计算每个条件的平均表达
   - 计算 log2 倍数变化：`log2(mean[cond1] / mean[cond0])`
   - 对条件间的表达值执行 Wilcoxon 秩和检验
2. 使用工作空间池进行高效的稀疏矩阵访问
3. 二分搜索优化用于稀疏矩阵行访问

**边界条件**

- **两个条件中零表达**: 倍数变化为 NaN，p 值为 1.0
- **一个条件中零表达**: 倍数变化为 +/-infinity
- **稀疏基因**: 非零值很少的基因可能统计不可靠
- **并列值**: 在 Wilcoxon 检验中应用标准并列校正

**数据保证（前置条件）**

- `expression.rows() == conditions.len`
- `response_scores` 容量 >= n_genes
- `p_values` 容量 >= n_genes
- 至少需要 2 个条件
- 表达矩阵是有效的 CSR 格式

**复杂度分析**

- **时间**: O(n_genes * n_cells * log(nnz_per_cell)) - 每个基因每个细胞的二分搜索
- **空间**: O(n_cells) 辅助空间 - 每个线程的表达提取工作空间

**示例**

```cpp
Sparse<Real, true> expression = /* cells x genes, CSR */;
Array<const Index> conditions = /* 条件标签 [n_cells] */;
Real* response_scores = new Real[n_genes];
Real* p_values = new Real[n_genes];

scl::kernel::comparison::condition_response(
    expression,
    conditions,
    response_scores,
    p_values,
    n_genes
);

// 找到显著响应的基因
for (Index g = 0; g < n_genes; ++g) {
    if (p_values[g] < 0.05 && std::abs(response_scores[g]) > 1.0) {
        // 基因 g 显示对条件变化的显著响应
    }
}
```

---

## 工具函数

### effect_size

计算两组之间的 Cohen's d 效应量。

::: source_code file="scl/kernel/comparison.hpp" symbol="effect_size" collapsed
:::

**复杂度**

- 时间: O(n1 + n2)
- 空间: O(1) 辅助空间

---

### glass_delta

使用对照组标准差计算 Glass's delta 效应量。

::: source_code file="scl/kernel/comparison.hpp" symbol="glass_delta" collapsed
:::

**复杂度**

- 时间: O(n_control + n_treatment)
- 空间: O(1) 辅助空间

---

### hedges_g

计算 Hedges' g 偏差校正效应量。

::: source_code file="scl/kernel/comparison.hpp" symbol="hedges_g" collapsed
:::

**复杂度**

- 时间: O(n1 + n2)
- 空间: O(1) 辅助空间

---

## 配置

`scl::kernel::comparison::config` 中的默认参数：

- `EPSILON = 1e-10`: 数值稳定性常数
- `MIN_CELLS_PER_GROUP = 3`: 每组所需的最小细胞数以获得可靠的统计
- `PERMUTATION_COUNT = 1000`: 默认置换计数（如果使用）
- `PARALLEL_THRESHOLD = 32`: 并行处理的最小大小

---

## 相关内容

- [通讯模块](./communication) - L-R 相互作用分析
- [多重检验模块](./multiple_testing) - P 值校正
