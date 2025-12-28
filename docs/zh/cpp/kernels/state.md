# state.hpp

> scl/kernel/state.hpp · 细胞状态评分内核（干性、分化、增殖等细胞状态）

## 概述

本文件为单细胞 RNA-seq 分析提供全面的细胞状态评分方法。它基于基因表达特征计算各种细胞状态的分数，包括干性、分化潜能、增殖、应激、细胞周期阶段、代谢状态和凋亡。

本文件提供：
- 干性分数计算
- 分化潜能（CytoTRACE 风格）评分
- 增殖和细胞周期评分
- 应激和凋亡评分
- 代谢状态评分（糖酵解/OXPHOS）
- 自定义基因特征评分
- 表达多样性和复杂性测量

**头文件**: `#include "scl/kernel/state.hpp"`

---

## 主要 API

### stemness_score

::: source_code file="scl/kernel/state.hpp" symbol="stemness_score" collapsed
:::

**算法说明**

基于干性基因表达计算每个细胞的干性分数：

1. **基因表达聚合**：计算干性基因的平均表达
2. **Z 分数归一化**：对所有细胞的分数进行 Z 分数归一化

**边界条件**

- 无干性基因：如果基因列表为空，返回零分数
- 常数值表达：如果所有细胞表达相同，归一化后返回零分数

**数据保证（前置条件）**

- `scores.len == expression.rows()`
- 所有基因索引必须有效

**复杂度分析**

- **时间**：O(n_cells * n_stemness_genes * log(nnz_per_cell))
- **空间**：O(n_cells)

**示例**

```cpp
#include "scl/kernel/state.hpp"

Sparse<Real, true> expression = /* ... */;
Array<Index> stemness_genes = {0, 1, 2, 3};
Array<Real> stemness_scores(n_cells);

scl::kernel::state::stemness_score(
    expression, stemness_genes, stemness_scores
);
```

---

### differentiation_potential

::: source_code file="scl/kernel/state.hpp" symbol="differentiation_potential" collapsed
:::

**算法说明**

计算分化潜能分数（CytoTRACE 风格）：

1. **基因计数相关性**：计算每个基因与基因计数的相关性
2. **加权表达和**：计算顶级基因表达的加权和
3. **归一化**：将分数归一化到 [0, 1] 范围

**复杂度分析**

- **时间**：O(n_cells * n_genes * log(nnz_per_cell) + n_genes * log(n_genes))
- **空间**：O(n_cells + n_genes)

---

### cell_cycle_score

::: source_code file="scl/kernel/state.hpp" symbol="cell_cycle_score" collapsed
:::

**算法说明**

计算细胞周期阶段分数（G1/S/G2M）并分配阶段标签：

1. **阶段分数计算**：计算 S 期和 G2/M 期分数
2. **Z 分数归一化**：独立归一化两个分数
3. **阶段分配**：根据分数分配阶段标签

**复杂度分析**

- **时间**：O(n_cells * (n_s_genes + n_g2m_genes) * log(nnz_per_cell))
- **空间**：O(n_cells)

**示例**

```cpp
Array<Index> s_genes = {10, 11, 12};
Array<Index> g2m_genes = {20, 21, 22};
Array<Real> s_scores(n_cells);
Array<Real> g2m_scores(n_cells);
Array<Index> phase_labels(n_cells);

scl::kernel::state::cell_cycle_score(
    expression, s_genes, g2m_genes,
    s_scores, g2m_scores, phase_labels
);
```

---

### signature_score

::: source_code file="scl/kernel/state.hpp" symbol="signature_score" collapsed
:::

**算法说明**

计算加权基因特征分数：

1. **加权和**：计算加权基因表达和
2. **Z 分数归一化**：归一化分数

**复杂度分析**

- **时间**：O(n_cells * n_signature_genes * log(nnz_per_cell))
- **空间**：O(n_cells)

---

### state_entropy

::: source_code file="scl/kernel/state.hpp" symbol="state_entropy" collapsed
:::

**算法说明**

计算表达熵（可塑性），按最大可能熵归一化：

1. **Shannon 熵**：计算每个细胞的 Shannon 熵
2. **归一化**：按最大熵（log(n_genes)）归一化

**复杂度分析**

- **时间**：O(nnz)
- **空间**：O(1) 每个细胞

---

## 工具函数

### proliferation_score

基于增殖基因表达计算增殖分数。

::: source_code file="scl/kernel/state.hpp" symbol="proliferation_score" collapsed
:::

**复杂度**

- 时间：O(n_cells * n_proliferation_genes * log(nnz_per_cell))
- 空间：O(n_cells)

---

### stress_score

基于应激基因表达计算应激分数。

::: source_code file="scl/kernel/state.hpp" symbol="stress_score" collapsed
:::

**复杂度**

- 时间：O(n_cells * n_stress_genes * log(nnz_per_cell))
- 空间：O(n_cells)

---

### quiescence_score

计算静止分数（静止与增殖分数的差）。

::: source_code file="scl/kernel/state.hpp" symbol="quiescence_score" collapsed
:::

**复杂度**

- 时间：O(n_cells * (n_quiescence_genes + n_proliferation_genes) * log(nnz_per_cell))
- 空间：O(n_cells)

---

### metabolic_score

计算糖酵解和 OXPHOS 分数。

::: source_code file="scl/kernel/state.hpp" symbol="metabolic_score" collapsed
:::

**复杂度**

- 时间：O(n_cells * (n_glycolysis_genes + n_oxphos_genes) * log(nnz_per_cell))
- 空间：O(n_cells)

---

### apoptosis_score

基于凋亡基因表达计算凋亡分数。

::: source_code file="scl/kernel/state.hpp" symbol="apoptosis_score" collapsed
:::

**复杂度**

- 时间：O(n_cells * n_apoptosis_genes * log(nnz_per_cell))
- 空间：O(n_cells)

---

### multi_signature_score

同时计算多个基因特征的分数。

::: source_code file="scl/kernel/state.hpp" symbol="multi_signature_score" collapsed
:::

**复杂度**

- 时间：O(n_signatures * n_cells * avg_signature_size * log(nnz_per_cell))
- 空间：O(n_cells * n_signatures)

---

### transcriptional_diversity

计算表达分布的 Simpson 多样性指数。

::: source_code file="scl/kernel/state.hpp" symbol="transcriptional_diversity" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1) 每个细胞

---

### expression_complexity

计算表达复杂性（高于阈值的基因表达比例）。

::: source_code file="scl/kernel/state.hpp" symbol="expression_complexity" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1) 每个细胞

---

### combined_state_score

从多个加权基因集计算组合状态分数。

::: source_code file="scl/kernel/state.hpp" symbol="combined_state_score" collapsed
:::

**复杂度**

- 时间：O(n_gene_sets * n_cells * avg_gene_set_size * log(nnz_per_cell))
- 空间：O(n_gene_sets * n_cells)

---

## 注意事项

**分数归一化**：
- 大多数分数进行 Z 分数归一化（均值=0，标准差=1）
- 熵和复杂性分数归一化到 [0, 1]
- 分化潜能归一化到 [0, 1]

**基因特征列表**：
- 基因索引必须有效且在 [0, n_genes) 范围内
- 空基因列表导致零分数
- 无效索引被忽略

**性能**：
- 所有函数按细胞并行化
- 稀疏矩阵访问优化以提高效率
- 内存高效计算

## 相关内容

- [Expression Analysis](/zh/cpp/kernels/feature) - 其他表达分析工具
- [Statistics](/zh/cpp/kernels/statistics) - 统计操作
