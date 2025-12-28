# enrichment.hpp

> scl/kernel/enrichment.hpp · 基因集富集和通路分析

## 概述

本文件提供用于基因集富集分析（GSEA）、过度表达分析（ORA）和通路活性评分的统计方法。所有方法都针对稀疏表达矩阵进行了优化，并支持并行处理。

**头文件**: `#include "scl/kernel/enrichment.hpp"`

主要特性：
- 超几何和 Fisher 精确检验
- GSEA 富集评分
- 过度表达分析（ORA）
- 通路活性计算
- FDR 校正（Benjamini-Hochberg）

---

## 主要 API

### hypergeometric_test

::: source_code file="scl/kernel/enrichment.hpp" symbol="hypergeometric_test" collapsed
:::

**算法说明**

计算过度表达的超几何检验 p 值：

1. 给定列联表：
   - 样本大小为 n 中有 k 个成功
   - 总体大小为 N 中有 K 个成功
2. 计算 p 值：观察到 k 个或更多成功的概率
3. 使用超几何分布：P(X >= k) = sum_{i=k}^{min(n,K)} C(K,i) * C(N-K, n-i) / C(N,n)
4. 使用对数空间计算以避免溢出

**边界条件**

- **k = 0**：返回 p 值 = 1.0（无富集）
- **k = n**：根据总体成功率返回 p 值
- **k > K**：无效输入（不应发生）
- **n = 0**：返回 1.0

**数据保证（前置条件）**

- `k <= n <= N`
- `K <= N`
- `k <= K`
- 所有值都是非负整数

**复杂度分析**

- **时间**：O(min(k, n-k)) - 与较小尾部成正比
- **空间**：O(1) 辅助

**示例**

```cpp
#include "scl/kernel/enrichment.hpp"

// 检验：100 个样本中有 10 个 DE 基因，20000 个总基因中有 50 个通路基因
Index k = 10;  // 通路中的 DE 基因
Index n = 100; // 总 DE 基因
Index K = 50;  // 通路大小
Index N = 20000; // 总基因

Real p_value = scl::kernel::enrichment::hypergeometric_test(k, n, K, N);

// p_value 是偶然观察到 10+ 个通路基因的概率
```

---

### fisher_exact_test

::: source_code file="scl/kernel/enrichment.hpp" symbol="fisher_exact_test" collapsed
:::

**算法说明**

计算 2x2 列联表的 Fisher 精确检验 p 值：

1. 给定 2x2 表：
   ```
   [a  b]
   [c  d]
   ```
2. 使用超几何分布计算双尾 p 值
3. 对具有相同边际和相等或更极端优势比的所有表的概率求和
4. 使用对数空间计算以确保数值稳定性

**边界条件**

- **零计数**：正确处理（返回适当的 p 值）
- **全零**：返回 1.0
- **完美关联（b=0 或 c=0）**：返回非常小的 p 值

**数据保证（前置条件）**

- 所有计数 >= 0
- 每行和每列至少有一个计数

**复杂度分析**

- **时间**：O(min(a, b, c, d)) - 与最小计数成正比
- **空间**：O(1) 辅助

**示例**

```cpp
// 列联表：
//           在通路中  不在通路中
// DE 基因      15         85
// 非 DE        35       19865

Index a = 15;  // DE 且在通路中
Index b = 85;  // DE 但不在通路中
Index c = 35;  // 非 DE 且在通路中
Index d = 19865; // 非 DE 且不在通路中

Real p_value = scl::kernel::enrichment::fisher_exact_test(a, b, c, d);

// p_value 是双尾 Fisher 精确检验 p 值
```

---

### gsea

::: source_code file="scl/kernel/enrichment.hpp" symbol="gsea" collapsed
:::

**算法说明**

计算基因集富集分析富集分数和 p 值：

1. 按统计量对基因进行排序（例如，倍数变化、t 统计量）
2. 计算运行和：
   - 当基因在集合中时：添加 |statistic| / sum(|statistics in set|)
   - 当基因不在集合中时：减去 1 / (n_genes - n_in_set)
3. 富集分数（ES）= 与零的最大绝对偏差
4. 通过零假设下的期望值对 ES 进行归一化（NES）
5. 通过置换检验计算 p 值：
   - 将基因集成员身份打乱 n_permutations 次
   - 计算 |NES| >= 观察到的 |NES| 的比例

**边界条件**

- **空基因集**：ES = 0，p 值 = 1.0
- **所有基因在集合中**：ES = 1.0，p 值 = 0.0
- **无富集**：ES 接近 0，p 值接近 1.0
- **n_permutations = 0**：p 值未定义

**数据保证（前置条件）**

- `ranked_genes` 包含有效的基因索引
- `in_gene_set.len >= n_genes`
- `ranked_genes` 中的基因按统计量排序（降序）
- `n_permutations > 0` 以获得有效的 p 值

**复杂度分析**

- **时间**：O(n_genes * n_permutations)
- **空间**：O(n_genes) 辅助

**示例**

```cpp
// 按差异表达统计量对基因进行排序
scl::Array<Index> ranked_genes = /* 按统计量排序 */;
scl::Array<bool> in_gene_set(n_genes);
// ... 如果基因 i 在通路中，设置 in_gene_set[i] = true ...

Real es, nes, p_value;
scl::kernel::enrichment::gsea(
    ranked_genes, in_gene_set, n_genes,
    es, nes, p_value,
    1000,  // n_permutations
    42     // 种子
);

// es = 富集分数
// nes = 归一化富集分数
// p_value = 基于置换的 p 值
```

---

### ora

::: source_code file="scl/kernel/enrichment.hpp" symbol="ora" collapsed
:::

**算法说明**

对多个通路执行过度表达分析：

1. 对于每个通路 p（并行）：
   - 计算重叠：同时在 DE 集合和通路中的基因
   - 计算超几何 p 值
   - 计算优势比：(overlap / (n_de - overlap)) / ((pathway_size - overlap) / (n_total - n_de - pathway_size + overlap))
   - 计算倍数富集：(overlap / n_de) / (pathway_size / n_total)
2. 返回所有通路的 p 值、优势比和倍数富集

**边界条件**

- **无重叠**：p 值 = 1.0，odds_ratio = 0，fold_enrichment = 0
- **完美重叠**：非常小的 p 值，大的 odds_ratio
- **空通路**：未定义（不应发生）

**数据保证（前置条件）**

- 所有输出数组长度 >= n_pathways
- 所有基因索引有效
- `pathway_genes` 是数组的数组（每个通路一个）
- `pathway_sizes` 包含每个通路的大小

**复杂度分析**

- **时间**：O(n_pathways * avg_pathway_size)
- **空间**：O(n_total_genes) 辅助

**示例**

```cpp
scl::Array<Index> de_genes = /* DE 基因索引 */;
const Index* pathway_genes[] = {pathway1, pathway2, ...};
const Index pathway_sizes[] = {50, 30, ...};
Index n_pathways = 100;

scl::Array<Real> p_values(n_pathways);
scl::Array<Real> odds_ratios(n_pathways);
scl::Array<Real> fold_enrichments(n_pathways);

scl::kernel::enrichment::ora(
    de_genes,
    pathway_genes, pathway_sizes, n_pathways,
    n_total_genes,
    p_values, odds_ratios, fold_enrichments
);

// p_values[p] = 通路 p 的超几何 p 值
// odds_ratios[p] = 通路 p 的优势比
// fold_enrichments[p] = 通路 p 的倍数富集
```

---

### pathway_activity

::: source_code file="scl/kernel/enrichment.hpp" symbol="pathway_activity" collapsed
:::

**算法说明**

计算每个细胞的通路活性分数：

1. 对于每个细胞 i（并行）：
   - 提取通路基因的表达
   - 计算平均表达：`activity[i] = mean(expression[pathway_genes])`
2. 活性表示每个细胞通路基因的平均表达
3. 使用稀疏矩阵操作以提高效率

**边界条件**

- **空通路**：所有活性为 0
- **无表达**：所有活性为 0
- **缺失基因**：忽略（不在表达矩阵中）

**数据保证（前置条件）**

- `activity_scores.len >= n_cells`
- 所有基因索引有效
- X 必须是 CSR 格式（细胞 x 基因）

**复杂度分析**

- **时间**：O(nnz * n_pathway_genes / n_genes) - 与通路大小成正比
- **空间**：O(n_genes) 辅助用于基因查找

**示例**

```cpp
scl::Array<Index> pathway_genes = /* 通路基因索引 */;
scl::Array<Real> activity_scores(n_cells);

scl::kernel::enrichment::pathway_activity(
    X, pathway_genes, n_cells, n_genes, activity_scores
);

// activity_scores[i] = 细胞 i 中通路基因的平均表达
```

---

### benjamini_hochberg

::: source_code file="scl/kernel/enrichment.hpp" symbol="benjamini_hochberg" collapsed
:::

**算法说明**

对富集 p 值应用 Benjamini-Hochberg FDR 校正：

1. 按升序对 p 值进行排序（保留原始索引）
2. 对于排名 i 的每个 p 值：
   - 计算调整后的 p 值：`q = p * n / rank`
   - 确保单调性：`q[i] = min(q[i], q[i+1], ..., q[n-1])`
3. 返回 q 值（FDR 调整后的 p 值）

**边界条件**

- **所有 p 值 = 0**：所有 q 值 = 0
- **所有 p 值 = 1**：所有 q 值 = 1
- **空数组**：返回空数组

**数据保证（前置条件）**

- `q_values.len >= p_values.len`
- 所有 p 值在 [0, 1] 范围内

**复杂度分析**

- **时间**：O(n log n) 用于排序
- **空间**：O(n) 辅助

**示例**

```cpp
scl::Array<Real> p_values = /* 富集 p 值 */;
scl::Array<Real> q_values(p_values.len);

scl::kernel::enrichment::benjamini_hochberg(p_values, q_values);

// q_values[i] = FDR 调整后的 p 值（q 值）
// q < 0.05 表示 FDR < 5%
```

---

## 工具函数

### odds_ratio

从 2x2 列联表计算优势比。

::: source_code file="scl/kernel/enrichment.hpp" symbol="odds_ratio" collapsed
:::

**复杂度**

- 时间：O(1)
- 空间：O(1)

---

### gsea_running_sum

计算 GSEA 运行和用于可视化。

::: source_code file="scl/kernel/enrichment.hpp" symbol="gsea_running_sum" collapsed
:::

**复杂度**

- 时间：O(n_genes)
- 空间：O(1) 辅助

---

### leading_edge_genes

识别前导边缘基因（对富集峰值有贡献的基因）。

::: source_code file="scl/kernel/enrichment.hpp" symbol="leading_edge_genes" collapsed
:::

**复杂度**

- 时间：O(n_genes)
- 空间：O(1) 辅助

---

## 注意事项

- 超几何检验最常用于 ORA
- GSEA 需要排序的基因 - 确保正确排序
- FDR 校对于多重检验至关重要
- 通路活性可用于细胞类型评分

## 相关内容

- [多重检验模块](./multiple_testing) - 其他校正方法
- [统计模块](../math/statistics) - 统计检验
