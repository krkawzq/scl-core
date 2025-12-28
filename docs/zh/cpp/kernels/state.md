# 状态

用于干性、分化、增殖和其他细胞状态的细胞状态评分内核。

## 概述

状态模块提供：

- **干性评分** - 基于干细胞标记的评分
- **分化潜能** - CytoTRACE 风格的潜能评分
- **增殖评分** - 细胞周期和生长评分
- **应激评分** - 应激反应评分
- **细胞周期** - G1/S/G2M 期分配
- **代谢评分** - 糖酵解和 OXPHOS 评分
- **凋亡评分** - 细胞死亡评分
- **特征评分** - 自定义基因特征评分
- **多样性度量** - 表达熵和复杂性

## 基本状态评分

### stemness_score

基于干性基因表达计算每个细胞的干性评分：

```cpp
#include "scl/kernel/state.hpp"

Sparse<Real, true> expression = /* ... */;  // 细胞 x 基因
Array<const Index> stemness_genes = /* ... */;  // 干性基因索引
Array<Real> scores(n_cells);

scl::kernel::state::stemness_score(
    expression,
    stemness_genes,
    scores
);
```

**参数：**
- `expression`: 表达矩阵（细胞 x 基因，CSR）
- `stemness_genes`: 干性基因索引 [n_stemness_genes]
- `scores`: 输出的 Z 分数归一化干性评分 [n_cells]

**后置条件：**
- `scores[i]` 包含细胞 i 的 Z 分数归一化干性评分
- 评分均值为 0，标准差为 1
- 矩阵不变

**算法：**
1. 对于每个细胞，计算干性基因的平均表达
2. 对所有细胞的评分进行 Z 分数归一化

**复杂度：**
- 时间：O(n_cells * n_stemness_genes * log(nnz_per_cell))
- 空间：O(n_cells) 辅助空间

**使用场景：**
- 干细胞识别
- 多能性评估
- 发育阶段分析

### differentiation_potential

计算每个细胞的分化潜能评分（CytoTRACE 风格）：

```cpp
Array<Real> potency_scores(n_cells);

scl::kernel::state::differentiation_potential(
    expression,
    potency_scores
);
```

**参数：**
- `expression`: 表达矩阵（细胞 x 基因，CSR）
- `potency_scores`: 归一化潜能评分 [0, 1] [n_cells]

**后置条件：**
- `potency_scores[i]` 包含归一化潜能评分，范围 [0, 1]
- 更高的评分表示更大的分化潜能
- 矩阵不变

**算法：**
1. 统计每个细胞的表达基因数
2. 计算每个基因与基因计数的相关性
3. 选择相关性最高的基因
4. 计算顶级基因表达的加权和
5. 归一化到 [0, 1] 范围

**复杂度：**
- 时间：O(n_cells * n_genes * log(nnz_per_cell) + n_genes * log(n_genes))
- 空间：O(n_cells + n_genes) 辅助空间

**使用场景：**
- 分化潜能排序
- 发育轨迹分析
- 干性与分化

### proliferation_score

基于增殖基因表达计算每个细胞的增殖评分：

```cpp
Array<const Index> proliferation_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::proliferation_score(
    expression,
    proliferation_genes,
    scores
);
```

**参数：**
- `proliferation_genes`: 增殖基因索引 [n_proliferation_genes]
- `scores`: 输出的 Z 分数归一化增殖评分 [n_cells]

**后置条件：**
- `scores[i]` 包含 Z 分数归一化增殖评分
- 评分均值为 0，标准差为 1

**使用场景：**
- 细胞周期活性
- 生长速率估计
- 增殖细胞识别

### stress_score

基于应激基因表达计算每个细胞的应激评分：

```cpp
Array<const Index> stress_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::stress_score(
    expression,
    stress_genes,
    scores
);
```

**参数：**
- `stress_genes`: 应激基因索引 [n_stress_genes]
- `scores`: 输出的 Z 分数归一化应激评分 [n_cells]

**使用场景：**
- 应激反应检测
- 细胞健康评估
- 质量控制

## 细胞周期

### cell_cycle_score

计算细胞周期期相评分（G1/S/G2M）并分配期相标签：

```cpp
Array<const Index> s_genes = /* ... */;  // S 期基因
Array<const Index> g2m_genes = /* ... */;  // G2/M 期基因
Array<Real> s_scores(n_cells);
Array<Real> g2m_scores(n_cells);
Array<Index> phase_labels(n_cells);

scl::kernel::state::cell_cycle_score(
    expression,
    s_genes,
    g2m_genes,
    s_scores,
    g2m_scores,
    phase_labels
);
```

**参数：**
- `s_genes`: S 期基因索引 [n_s_genes]
- `g2m_genes`: G2/M 期基因索引 [n_g2m_genes]
- `s_scores`: 输出的 Z 分数归一化 S 期评分 [n_cells]
- `g2m_scores`: 输出的 Z 分数归一化 G2/M 期评分 [n_cells]
- `phase_labels`: 输出的期相标签：0=G1, 1=S, 2=G2M [n_cells]

**后置条件：**
- `s_scores[i]` 和 `g2m_scores[i]` 是 Z 分数归一化的
- `phase_labels[i]` 表示分配的期相（0、1 或 2）

**算法：**
1. 计算每个细胞的 S 期和 G2/M 期评分
2. 对两个评分进行 Z 分数归一化
3. 分配期相：如果 s_score > 0 且 s_score > g2m_score 则为 S，
   如果 g2m_score > 0 且 g2m_score > s_score 则为 G2M，否则为 G1

**复杂度：**
- 时间：O(n_cells * (n_s_genes + n_g2m_genes) * log(nnz_per_cell))
- 空间：O(n_cells) 辅助空间

**使用场景：**
- 细胞周期期相分配
- 增殖分析
- 细胞周期回归

## 代谢评分

### metabolic_score

计算糖酵解和 OXPHOS（氧化磷酸化）评分：

```cpp
Array<const Index> glycolysis_genes = /* ... */;
Array<const Index> oxphos_genes = /* ... */;
Array<Real> glycolysis_scores(n_cells);
Array<Real> oxphos_scores(n_cells);

scl::kernel::state::metabolic_score(
    expression,
    glycolysis_genes,
    oxphos_genes,
    glycolysis_scores,
    oxphos_scores
);
```

**参数：**
- `glycolysis_genes`: 糖酵解基因索引 [n_glycolysis_genes]
- `oxphos_genes`: OXPHOS 基因索引 [n_oxphos_genes]
- `glycolysis_scores`: 输出的 Z 分数归一化糖酵解评分 [n_cells]
- `oxphos_scores`: 输出的 Z 分数归一化 OXPHOS 评分 [n_cells]

**后置条件：**
- 两个评分都是 Z 分数归一化的
- 矩阵不变

**使用场景：**
- 代谢通路活性
- 能量产生分析
- 代谢状态分类

## 凋亡

### apoptosis_score

基于凋亡基因表达计算每个细胞的凋亡评分：

```cpp
Array<const Index> apoptosis_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::apoptosis_score(
    expression,
    apoptosis_genes,
    scores
);
```

**参数：**
- `apoptosis_genes`: 凋亡基因索引 [n_apoptosis_genes]
- `scores`: 输出的 Z 分数归一化凋亡评分 [n_cells]

**使用场景：**
- 细胞死亡检测
- 凋亡通路活性
- 细胞活力评估

## 特征评分

### signature_score

计算每个细胞的加权基因特征评分：

```cpp
Array<const Index> gene_indices = /* ... */;  // 特征基因索引
Array<const Real> gene_weights = /* ... */;  // 基因权重
Array<Real> scores(n_cells);

scl::kernel::state::signature_score(
    expression,
    gene_indices,
    gene_weights,
    scores
);
```

**参数：**
- `gene_indices`: 特征基因索引 [n_signature_genes]
- `gene_weights`: 每个特征基因的权重 [n_signature_genes]
- `scores`: 输出的 Z 分数归一化特征评分 [n_cells]

**后置条件：**
- `scores[i]` 包含细胞 i 的加权特征评分
- 评分是 Z 分数归一化的

**算法：**
1. 对于每个细胞，计算特征基因表达的加权和
2. 按绝对权重和归一化
3. 对所有细胞进行 Z 分数归一化

**复杂度：**
- 时间：O(n_cells * n_signature_genes * log(nnz_per_cell))
- 空间：O(n_cells) 辅助空间

**使用场景：**
- 自定义基因特征评分
- 通路活性评估
- 功能状态分析

### multi_signature_score

同时计算多个基因特征的评分：

```cpp
const Index* signature_gene_indices = /* ... */;  // 扁平数组
const Size* signature_offsets = /* ... */;  // [n_signatures + 1]
Real* score_matrix = /* 分配 n_cells * n_signatures */;

scl::kernel::state::multi_signature_score(
    expression,
    signature_gene_indices,
    signature_offsets,
    n_signatures,
    score_matrix
);
```

**参数：**
- `signature_gene_indices`: 所有基因索引的扁平数组 [total_genes]
- `signature_offsets`: 每个特征的起始偏移 [n_signatures + 1]
- `n_signatures`: 特征数量
- `score_matrix`: 输出的评分矩阵 [n_cells * n_signatures]

**后置条件：**
- `score_matrix[i * n_signatures + s]` 包含细胞 i 和特征 s 的 Z 分数归一化评分
- 每个特征列独立进行 Z 分数归一化

**复杂度：**
- 时间：O(n_signatures * n_cells * avg_signature_size * log(nnz_per_cell))
- 空间：O(n_cells * n_signatures) 辅助空间

**使用场景：**
- 多特征评分
- 通路活性矩阵
- 功能状态分析

## 多样性度量

### state_entropy

计算每个细胞的表达熵（可塑性）：

```cpp
Array<Real> entropy_scores(n_cells);

scl::kernel::state::state_entropy(
    expression,
    entropy_scores
);
```

**参数：**
- `entropy_scores`: 归一化熵评分 [0, 1] [n_cells]

**后置条件：**
- `entropy_scores[i]` 包含细胞 i 的归一化 Shannon 熵
- 评分在 [0, 1] 范围内，其中 1 表示最大多样性

**算法：**
对于每个细胞：
1. 计算总表达
2. 计算 Shannon 熵：-sum(p_i * log(p_i))
3. 按最大可能熵（log(n_genes)）归一化

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间（每个细胞）

**使用场景：**
- 表达多样性
- 细胞可塑性
- 状态异质性

### transcriptional_diversity

计算表达分布的 Simpson 多样性指数：

```cpp
Array<Real> diversity_scores(n_cells);

scl::kernel::state::transcriptional_diversity(
    expression,
    diversity_scores
);
```

**参数：**
- `diversity_scores`: 多样性评分 [0, 1] [n_cells]

**后置条件：**
- `diversity_scores[i]` 包含细胞 i 的 Simpson 多样性指数
- 评分在 [0, 1] 范围内，其中 1 表示最大多样性

**算法：**
对于每个细胞：
1. 计算总表达和平方表达和
2. Simpson 指数 = 1 - sum(p_i^2)，其中 p_i = value_i / total

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间（每个细胞）

**使用场景：**
- 表达多样性
- 熵的替代方法
- 多样性量化

### expression_complexity

计算表达复杂性，即表达超过阈值的基因比例：

```cpp
Array<Real> complexity_scores(n_cells);

scl::kernel::state::expression_complexity(
    expression,
    expression_threshold,  // 最小表达值
    complexity_scores
);
```

**参数：**
- `expression_threshold`: 计为表达的最小表达值
- `complexity_scores`: 复杂性评分 [0, 1] [n_cells]

**后置条件：**
- `complexity_scores[i]` = n_expressed_genes / n_total_genes（细胞 i）
- 评分在 [0, 1] 范围内

**算法：**
对于每个细胞：
1. 统计表达 > 阈值的基因数
2. 按总基因数归一化

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间（每个细胞）

**使用场景：**
- 基因表达复杂性
- 转录活性
- 细胞状态复杂性

## 组合评分

### quiescence_score

计算静止评分，作为静止和增殖之间的差异：

```cpp
Array<const Index> quiescence_genes = /* ... */;
Array<const Index> proliferation_genes = /* ... */;
Array<Real> scores(n_cells);

scl::kernel::state::quiescence_score(
    expression,
    quiescence_genes,
    proliferation_genes,
    scores
);
```

**参数：**
- `quiescence_genes`: 静止基因索引 [n_quiescence_genes]
- `proliferation_genes`: 增殖基因索引 [n_proliferation_genes]
- `scores`: 静止评分 [n_cells]

**后置条件：**
- `scores[i]` = quiescence_score[i] - proliferation_score[i]
- 评分是 Z 分数归一化的差异

**使用场景：**
- 静止细胞识别
- 生长与静止平衡
- 细胞状态分类

### combined_state_score

从多个基因集计算组合状态评分（带权重）：

```cpp
const Index* const* gene_sets = /* ... */;  // 基因集指针数组
const Size* gene_set_sizes = /* ... */;
const Real* weights = /* ... */;
Array<Real> combined_scores(n_cells);

scl::kernel::state::combined_state_score(
    expression,
    gene_sets,
    gene_set_sizes,
    weights,
    n_gene_sets,
    combined_scores
);
```

**参数：**
- `gene_sets`: 基因集指针数组 [n_gene_sets]
- `gene_set_sizes`: 每个基因集的大小 [n_gene_sets]
- `weights`: 每个基因集的权重 [n_gene_sets]
- `n_gene_sets`: 基因集数量
- `combined_scores`: 组合评分 [n_cells]

**后置条件：**
- `combined_scores[i]` 包含所有基因集评分的加权组合
- 矩阵不变

**算法：**
1. 计算每个基因集的单独评分
2. 对每个基因集评分进行 Z 分数归一化
3. 计算加权组合

**复杂度：**
- 时间：O(n_gene_sets * n_cells * avg_gene_set_size * log(nnz_per_cell))
- 空间：O(n_gene_sets * n_cells) 辅助空间

**使用场景：**
- 多基因集评分
- 复合状态评分
- 加权特征组合

## 配置

`scl::kernel::state::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size MIN_GENES_FOR_SCORE = 3;
    constexpr Real PSEUDOCOUNT = 1.0;
    constexpr Size PARALLEL_THRESHOLD = 64;
}
```

## 性能考虑

### 并行化

所有状态评分函数都并行化：
- `stemness_score`: 在细胞上并行
- `differentiation_potential`: 在细胞和基因上并行
- `cell_cycle_score`: 在细胞上并行
- `multi_signature_score`: 在特征上并行

### 内存效率

- 预分配的输出缓冲区
- 高效的稀疏矩阵访问
- 最少的临时分配

---

::: tip 基因集
使用经过充分验证的基因集（例如，来自 MSigDB、CellMarker）以获得可靠的状态评分。
:::

::: warning 归一化
所有评分都是 Z 分数归一化的。在解释结果时考虑评分的分布。
:::

