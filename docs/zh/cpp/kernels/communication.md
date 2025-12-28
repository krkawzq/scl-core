# communication.hpp

> scl/kernel/communication.hpp · 细胞间通讯分析（CellChat/CellPhoneDB 风格）

## 概述

本文件提供高性能的配体-受体（L-R）相互作用分析内核，支持多种评分方法、基于置换的统计检验、批量处理和空间上下文感知的通讯分析。

**头文件**: `#include "scl/kernel/communication.hpp"`

---

## 主要 API

### lr_score_matrix

::: source_code file="scl/kernel/communication.hpp" symbol="lr_score_matrix" collapsed
:::

**算法说明**

计算所有细胞类型对之间的配体-受体相互作用分数：

1. 提取所有细胞中配体和受体基因的表达值
2. 按细胞类型分组细胞
3. 对于每个发送者-接收者类型对：
   - 计算发送者类型中配体的平均表达
   - 计算接收者类型中受体的平均表达
   - 应用评分方法（MeanProduct、GeometricMean、MinMean、Product 或 Natmi）
4. 将结果存储在分数矩阵中，`score_matrix[s * n_types + r]` 包含发送者 s 和接收者 r 的分数

**边界条件**

- **零表达**: 如果基因在某个细胞类型中表达为零，分数贡献为零
- **缺失细胞类型**: 空细胞类型被优雅处理，分数为零
- **无效基因索引**: 必须是表达矩阵内的有效索引

**数据保证（前置条件）**

- `score_matrix` 容量 >= `n_types * n_types`
- 所有细胞类型标签有效（0 <= label < n_types）
- 表达矩阵是有效的 CSR 格式
- 配体和受体基因索引在有效范围内

**复杂度分析**

- **时间**: O(n_cells + n_types^2) - 线性扫描细胞加上成对评分
- **空间**: O(n_cells + n_types) 辅助空间 - 类型分组的临时数组

**示例**

```cpp
#include "scl/kernel/communication.hpp"

Sparse<Real, true> expression = /* cells x genes, CSR 格式 */;
Array<const Index> cell_type_labels = /* 每个细胞的细胞类型 */;
Real* score_matrix = new Real[n_types * n_types];

scl::kernel::communication::lr_score_matrix(
    expression,
    cell_type_labels,
    ligand_gene,      // 配体基因索引
    receptor_gene,    // 受体基因索引
    n_cells,
    n_types,
    score_matrix,
    scl::kernel::communication::ScoreMethod::MeanProduct
);

// 访问发送者类型 s 和接收者类型 r 的分数：
Real score = score_matrix[s * n_types + r];
```

---

### lr_score_with_permutation

::: source_code file="scl/kernel/communication.hpp" symbol="lr_score_with_permutation" collapsed
:::

**算法说明**

计算 L-R 相互作用分数及基于置换的 p 值以评估统计显著性：

1. 使用选定评分方法计算指定发送者-接收者对的观测分数
2. 执行 n_permutations 次随机打乱细胞类型标签
3. 对于每次置换：
   - 使用打乱后的标签计算 L-R 分数
   - 跟踪有多少置换分数 >= 观测分数
4. P 值 = (count + 1) / (n_permutations + 1)，使用标准置换检验公式
5. 使用线程局部随机数生成器进行并行执行

**边界条件**

- **零观测分数**: 如果所有置换分数也为零，p 值为 1.0
- **完美分数**: 如果观测分数远高于置换分数，p 值接近 0
- **小 n_permutations**: 建议最小值为 1000 以获得可靠的 p 值

**数据保证（前置条件）**

- 发送者和接收者类型索引有效（0 <= index < n_types）
- 表达矩阵是有效的 CSR 格式
- 细胞类型标签数组长度 == n_cells
- 所有类型标签有效

**复杂度分析**

- **时间**: O(n_permutations * n_cells) - 每次置换需要完整扫描细胞
- **空间**: O(n_cells) 辅助空间 - 每个线程的打乱标签临时数组

**示例**

```cpp
Real observed_score;
Real p_value;

scl::kernel::communication::lr_score_with_permutation(
    expression,
    cell_type_labels,
    ligand_gene,
    receptor_gene,
    sender_type,      // 发送者细胞类型索引
    receiver_type,    // 接收者细胞类型索引
    n_cells,
    observed_score,   // 输出：观测相互作用强度
    p_value,          // 输出：置换 p 值
    1000,             // 置换次数
    scl::kernel::communication::ScoreMethod::MeanProduct,
    42                // 随机种子
);

if (p_value < 0.05) {
    // 检测到显著相互作用
}
```

---

### batch_lr_scoring

::: source_code file="scl/kernel/communication.hpp" symbol="batch_lr_scoring" collapsed
:::

**算法说明**

高效并行计算多个配体-受体对的 L-R 分数：

1. 并行处理每个 L-R 对
2. 对于每个对：
   - 提取配体和受体表达向量
   - 按类型分组细胞
   - 使用选定方法计算类型对分数
   - 存储在输出数组的偏移量 `p * n_types^2 + s * n_types + r` 处
3. 使用带动态调度的 parallel_for 进行负载平衡

**边界条件**

- **空对列表**: 立即返回，不进行计算
- **无效基因索引**: 具有无效索引的对产生零分数
- **内存限制**: 非常大的 n_pairs 可能需要分块处理

**数据保证（前置条件）**

- `scores` 数组容量 >= `n_pairs * n_types * n_types`
- `ligand_genes` 和 `receptor_genes` 数组长度 == n_pairs
- 所有基因索引有效（0 <= index < n_genes）
- 表达矩阵是有效的 CSR 格式

**复杂度分析**

- **时间**: O(n_pairs * (n_cells + n_types^2)) - 与对数成线性关系
- **空间**: O(n_cells * max_gene) 辅助空间 - 每个线程的表达提取工作空间

**示例**

```cpp
const Index* ligand_genes = /* 配体基因索引数组 [n_pairs] */;
const Index* receptor_genes = /* 受体基因索引数组 [n_pairs] */;
Real* scores = new Real[n_pairs * n_types * n_types];

scl::kernel::communication::batch_lr_scoring(
    expression,
    cell_type_labels,
    ligand_genes,
    receptor_genes,
    n_pairs,
    n_cells,
    n_types,
    scores,
    scl::kernel::communication::ScoreMethod::MeanProduct
);

// 访问对 p、发送者 s、接收者 r 的分数：
Real score = scores[p * n_types * n_types + s * n_types + r];
```

---

### batch_lr_permutation_test

::: source_code file="scl/kernel/communication.hpp" symbol="batch_lr_permutation_test" collapsed
:::

**算法说明**

同时计算多个 L-R 对的置换 p 值：

1. 并行处理每个 L-R 对
2. 对于每个对：
   - 计算所有类型组合的观测分数
   - 执行 n_permutations 次随机打乱
   - 计算每个类型组合的 p 值
3. 结果存储在扁平数组中：`scores[p * n_types^2 + s * n_types + r]` 和 `p_values[p * n_types^2 + s * n_types + r]`
4. 在可能时使用早期停止优化

**边界条件**

- **零分数**: 零观测分数的 p 值为 1.0
- **大 n_pairs**: 内存使用与 n_pairs * n_types^2 成比例
- **线程竞争**: 使用线程局部 RNG 的并行执行避免竞争

**数据保证（前置条件）**

- `scores` 和 `p_values` 数组容量 >= `n_pairs * n_types^2`
- 所有输入数组长度匹配
- 表达矩阵是有效的 CSR 格式

**复杂度分析**

- **时间**: O(n_pairs * n_permutations * n_cells) - 与对数和置换数成二次关系
- **空间**: O(n_cells) 辅助空间 - 每个线程的打乱标签数组

**示例**

```cpp
Real* scores = new Real[n_pairs * n_types * n_types];
Real* p_values = new Real[n_pairs * n_types * n_types];

scl::kernel::communication::batch_lr_permutation_test(
    expression,
    cell_type_labels,
    ligand_genes,
    receptor_genes,
    n_pairs,
    n_cells,
    n_types,
    scores,          // 输出：观测分数
    p_values,       // 输出：p 值
    1000,           // 每个对的置换次数
    scl::kernel::communication::ScoreMethod::MeanProduct,
    42              // 随机种子
);

// 过滤显著相互作用
for (Index p = 0; p < n_pairs; ++p) {
    for (Index s = 0; s < n_types; ++s) {
        for (Index r = 0; r < n_types; ++r) {
            Index idx = p * n_types * n_types + s * n_types + r;
            if (p_values[idx] < 0.05) {
                // 显著相互作用
            }
        }
    }
}
```

---

### spatial_communication_score

::: source_code file="scl/kernel/communication.hpp" symbol="spatial_communication_score" collapsed
:::

**算法说明**

使用空间邻居图计算空间上下文感知的通讯分数：

1. 并行处理每个细胞：
   - 提取当前细胞的配体表达
   - 遍历空间邻居（来自 spatial_graph）
   - 对于每个邻居，提取受体表达
   - 累积加权 L-R 相互作用：ligand_i * receptor_j * weight_ij
2. 结果是反映局部空间上下文的每细胞通讯分数
3. 使用稀疏矩阵迭代进行高效的邻居访问

**边界条件**

- **孤立细胞**: 没有邻居的细胞得分为零
- **自环**: 如果空间图包含自环，它们会贡献到分数
- **断开图**: 每个连通分量独立计算

**数据保证（前置条件）**

- `cell_scores` 容量 >= n_cells
- 空间图是有效的 CSR 格式，维度为 n_cells x n_cells
- 表达矩阵行数 == n_cells
- 空间图表示有效的邻居结构

**复杂度分析**

- **时间**: O(n_cells * avg_neighbors) - 与细胞数和平均度数成线性关系
- **空间**: O(n_cells) 辅助空间 - 仅输出数组

**示例**

```cpp
Sparse<Index, true> spatial_graph = /* 空间邻居图，CSR */;
Real* cell_scores = new Real[n_cells];

scl::kernel::communication::spatial_communication_score(
    expression,
    spatial_graph,
    ligand_gene,
    receptor_gene,
    n_cells,
    cell_scores
);

// cell_scores[i] 包含细胞 i 的空间通讯分数
```

---

### expression_specificity

::: source_code file="scl/kernel/communication.hpp" symbol="expression_specificity" collapsed
:::

**算法说明**

计算基因在细胞类型间的表达特异性：

1. 提取指定基因在所有细胞中的表达值
2. 按类型分组细胞并计算每个类型的平均表达
3. 使用公式计算每个类型的特异性分数：
   - specificity[t] = mean_t / (所有类型的均值之和 + epsilon)
4. 归一化以确保特异性之和有意义
5. 更高的特异性表示基因对该细胞类型更特异

**边界条件**

- **零表达**: 表达为零的类型特异性为零
- **均匀表达**: 如果基因在所有类型中表达相等，特异性均匀
- **单类型表达**: 仅在一种类型中表达的基因在该类型中特异性为 1.0

**数据保证（前置条件）**

- `specificity` 数组容量 >= n_types
- 基因索引有效（0 <= gene < n_genes）
- 表达矩阵是有效的 CSR 格式
- 细胞类型标签有效

**复杂度分析**

- **时间**: O(n_cells) - 单次遍历细胞
- **空间**: O(n_cells + n_types) 辅助空间 - 表达提取和类型分组

**示例**

```cpp
Real* specificity = new Real[n_types];

scl::kernel::communication::expression_specificity(
    expression,
    cell_type_labels,
    gene,            // 基因索引
    n_cells,
    n_types,
    specificity
);

// 找到最特异的类型
Index max_type = 0;
for (Index t = 1; t < n_types; ++t) {
    if (specificity[t] > specificity[max_type]) {
        max_type = t;
    }
}
```

---

## 工具函数

### filter_significant_interactions

按 p 值阈值过滤显著的 L-R 相互作用。

::: source_code file="scl/kernel/communication.hpp" symbol="filter_significant_interactions" collapsed
:::

**复杂度**

- 时间: O(n_pairs * n_types^2)
- 空间: O(1) 辅助空间

---

### aggregate_to_network

将 L-R 分数聚合为细胞类型通讯网络。

::: source_code file="scl/kernel/communication.hpp" symbol="aggregate_to_network" collapsed
:::

**复杂度**

- 时间: O(n_pairs * n_types^2)
- 空间: O(1) 辅助空间

---

## 配置

`scl::kernel::communication::config` 中的默认参数：

- `DEFAULT_N_PERM = 1000`: 默认置换次数
- `DEFAULT_PVAL_THRESHOLD = 0.05`: 默认 p 值阈值
- `EPSILON = 1e-15`: 数值稳定性常数
- `MIN_EXPRESSION = 0.1`: 最小表达阈值
- `MIN_PERCENT_EXPRESSED = 0.1`: 最小表达细胞百分比

---

## 评分方法

`ScoreMethod` 枚举提供不同的评分方法：

- `MeanProduct`: mean(ligand) * mean(receptor) - 标准方法
- `GeometricMean`: sqrt(mean(ligand) * mean(receptor)) - 平衡评分
- `MinMean`: min(mean(ligand), mean(receptor)) - 保守评分
- `Product`: 直接乘积 - 简单乘法
- `Natmi`: NATMI 风格评分 - 与 NATMI 工具兼容

---

## 相关内容

- [比较模块](./comparison) - 统计检验工具
- [稀疏矩阵](../core/sparse) - 稀疏矩阵操作
