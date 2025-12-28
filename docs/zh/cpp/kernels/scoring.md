# scoring.hpp

> scl/kernel/scoring.hpp · 基因集评分操作

## 概述

本文件提供用于计算跨细胞的基因集分数的高性能内核。支持多种评分方法，包括平均表达、基于排名的 AUC 分数、加权和、Seurat 风格模块分数和 z 分数归一化分数。还包括用于细胞周期评分和多签名批量处理的专用函数。

**头文件**: `#include "scl/kernel/scoring.hpp"`

---

## 主要 API

### mean_score

::: source_code file="scl/kernel/scoring.hpp" symbol="mean_score" collapsed
:::

**算法说明**

计算每个细胞在基因集上的平均表达分数：

1. 构建位集查找表以实现 O(1) 基因成员资格检查
2. 对于 CSR 格式：并行处理细胞，扫描每行查找基因集成员
3. 对于 CSC 格式：并行处理基因，原子累加到细胞
4. 计算均值：`scores[c] = sum(X[c, g] for g in gene_set) / |gene_set|`
5. 使用高效的稀疏矩阵迭代和位集成员资格测试

**边界条件**

- **空基因集**: 返回所有细胞的零分数
- **基因不在矩阵中**: 无效基因索引导致未定义行为
- **零表达**: 基因集中所有基因表达为零的细胞得分为 0
- **稀疏基因**: 高效处理非零值较少的基因

**数据保证（前置条件）**

- `scores.len >= n_cells`
- `gene_set` 中的所有基因索引在范围 [0, n_genes) 内
- 表达矩阵是有效的 CSR 或 CSC 格式
- 矩阵维度匹配 n_cells 和 n_genes

**复杂度分析**

- **时间**: CSR 为 O(nnz)，CSC 为 O(|gene_set| * avg_col_nnz)
- **空间**: O(n_genes / 64) 用于位集查找表

**示例**

```cpp
#include "scl/kernel/scoring.hpp"

Sparse<Real, true> expression = /* cells x genes, CSR */;
Array<const Index> gene_set = /* 基因集中的基因索引 */;
Array<Real> scores(n_cells);

scl::kernel::scoring::mean_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes
);

// scores[c] 包含细胞 c 中基因集的平均表达
```

---

### weighted_score

::: source_code file="scl/kernel/scoring.hpp" symbol="weighted_score" collapsed
:::

**算法说明**

计算每个细胞在基因集上的加权和分数：

1. 构建从基因索引到权重的权重映射
2. 并行处理每个细胞：
   - 迭代细胞行中的非零元素
   - 如果基因在集合中，累加 `weight[gene] * expression[cell, gene]`
3. 按权重和归一化：`scores[c] = sum(weight[i] * X[c, gene_set[i]]) / sum(weight)`
4. 对于 CSC 格式使用原子操作跨基因累加

**边界条件**

- **零权重**: 权重为零的基因不贡献
- **负权重**: 允许，可产生负分数
- **空基因集**: 返回零分数
- **权重和为零**: 返回零分数（避免除以零）

**数据保证（前置条件）**

- `scores.len >= n_cells`
- `gene_weights.len >= gene_set.len`
- 所有基因索引在 [0, n_genes) 范围内
- 表达矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(nnz) - 与非零数量成线性关系
- **空间**: O(n_genes) 用于权重映射

**示例**

```cpp
Array<const Index> gene_set = /* 基因索引 */;
Array<const Real> gene_weights = /* 每个基因的权重 */;
Array<Real> scores(n_cells);

scl::kernel::scoring::weighted_score(
    expression,
    gene_set,
    gene_weights,
    scores,
    n_cells,
    n_genes
);

// scores[c] = 细胞 c 中基因集表达的加权平均值
```

---

### auc_score

::: source_code file="scl/kernel/scoring.hpp" symbol="auc_score" collapsed
:::

**算法说明**

使用每个细胞的表达排名计算基于 AUC 的分数：

1. 并行处理每个细胞：
   - 提取所有基因的表达值
   - 使用 shell sort + insertion sort 混合算法计算排名
   - 统计基因集中有多少基因在顶部分位数（例如，前 5%）
   - 分数 = 基因集中基因在顶部分位数的比例
2. 使用 WorkspacePool 进行线程局部缓冲区以避免分配
3. 针对典型基因数量优化的高效排名算法

**边界条件**

- **空基因集**: 返回零分数
- **Quantile = 0**: 返回零分数（顶部 0% 中没有基因）
- **Quantile = 1**: 如果集合中所有基因都表达，返回 1.0
- **并列排名**: 应用标准并列处理

**数据保证（前置条件）**

- `scores.len >= n_cells`
- `0 < quantile <= 1`
- 表达矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(n_cells * n_genes * log(n_genes)) - 排名占主导
- **空间**: O(n_genes) 每个线程的工作空间

**示例**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::auc_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes,
    0.05  // 前 5% 分位数
);

// scores[c] = 细胞 c 中基因集在前 5% 表达中的比例
```

---

### module_score

::: source_code file="scl/kernel/scoring.hpp" symbol="module_score" collapsed
:::

**算法说明**

计算 Seurat 风格的模块分数，使用表达匹配的对照基因：

1. 计算所有细胞的基因均值
2. 按表达水平对基因进行分箱（默认 25 个箱）
3. 对于集合中的每个基因：
   - 从相同表达箱中采样对照基因
   - 默认：每个目标基因 1 个对照
4. 对于每个细胞：
   - 计算基因集的平均表达
   - 计算对照基因的平均表达
   - 分数 = gene_set_mean - control_mean
5. 使用随机种子进行可重现的对照选择

**边界条件**

- **空基因集**: 返回零分数
- **对照不足**: 如果箱中的基因少于请求数，使用所有可用基因
- **零表达基因**: 在分箱中正确处理
- **随机种子**: 相同种子产生相同的对照选择

**数据保证（前置条件）**

- `scores.len >= n_cells`
- 表达矩阵是有效的稀疏格式
- 有足够的基因可用于对照匹配

**复杂度分析**

- **时间**: O(nnz + n_genes * n_bins) - 均值计算加分箱
- **空间**: O(n_genes) 用于箱和对照基因数组

**示例**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::module_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes,
    1,      // 每个基因 1 个对照
    25,     // 25 个表达箱
    42      // 随机种子
);

// scores[c] = 细胞 c 的 mean(gene_set) - mean(control_genes)
```

---

### zscore_score

::: source_code file="scl/kernel/scoring.hpp" symbol="zscore_score" collapsed
:::

**算法说明**

计算 z 分数归一化的基因集分数：

1. 计算所有细胞的基因级均值和标准差
2. 预计算零表达的 z 分数：`z_zero = (0 - mean) / std`
3. 并行处理每个细胞：
   - 提取集合中基因的表达值
   - 转换为 z 分数：`z = (expr - mean) / std` 或使用预计算的 z_zero
   - 平均 z 分数：`scores[c] = mean(z-scores for genes in set)`
4. 使用 WorkspacePool 进行线程局部缓冲区

**边界条件**

- **零方差基因**: 方差为零的基因 z 分数为 0
- **空基因集**: 返回零分数
- **零表达**: 使用预计算的 z_zero 以提高效率
- **负 z 分数**: 允许，表示低于平均表达

**数据保证（前置条件）**

- `scores.len >= n_cells`
- 表达矩阵是有效的稀疏格式
- 至少需要 2 个细胞进行方差计算

**复杂度分析**

- **时间**: O(nnz + n_cells * |gene_set|) - 统计计算加上每个细胞的评分
- **空间**: O(n_genes + |gene_set|) 每个线程的工作空间

**示例**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::zscore_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes
);

// scores[c] = 细胞 c 中基因集的平均 z 分数
```

---

### cell_cycle_score

::: source_code file="scl/kernel/scoring.hpp" symbol="cell_cycle_score" collapsed
:::

**算法说明**

计算细胞周期阶段分数和分配：

1. 使用 S 期基因集计算 S 期分数
2. 使用 G2/M 期基因集计算 G2/M 期分数
3. 对于每个细胞：
   - 如果两个分数 <= 0：分配 G1 期（标签 0）
   - 否则：分配具有最高正分数的阶段
     - S 分数最高：S 期（标签 1）
     - G2M 分数最高：G2/M 期（标签 2）
4. 内部使用 mean_score 进行阶段评分

**边界条件**

- **两个分数都为负**: 细胞分配到 G1 期
- **分数相等**: 如果两者都为正，G2/M 优先
- **空基因集**: 返回零分数和 G1 分配
- **在零处并列**: 分配到 G1

**数据保证（前置条件）**

- `s_scores.len >= n_cells`
- `g2m_scores.len >= n_cells`
- `phase_labels.len >= n_cells`
- 表达矩阵是有效的稀疏格式

**复杂度分析**

- **时间**: O(nnz) - 由 mean_score 调用主导
- **空间**: O(n_genes) 用于位集查找

**示例**

```cpp
Array<const Index> s_genes = /* S 期基因索引 */;
Array<const Index> g2m_genes = /* G2/M 期基因索引 */;
Array<Real> s_scores(n_cells);
Array<Real> g2m_scores(n_cells);
Array<Index> phase_labels(n_cells);

scl::kernel::scoring::cell_cycle_score(
    expression,
    s_genes,
    g2m_genes,
    s_scores,
    g2m_scores,
    phase_labels,
    n_cells,
    n_genes
);

// phase_labels[c] = 0 (G1)、1 (S) 或 2 (G2/M)
```

---

## 工具函数

### compute_gene_means

计算所有细胞中每个基因的平均表达。

::: source_code file="scl/kernel/scoring.hpp" symbol="compute_gene_means" collapsed
:::

**复杂度**

- 时间: O(nnz)
- 空间: O(n_genes) 用于原子计数器（仅 CSR）

---

### gene_set_score

通用基因集评分调度器，路由到适当的方法。

::: source_code file="scl/kernel/scoring.hpp" symbol="gene_set_score" collapsed
:::

**复杂度**

- 时间: 取决于所选方法
- 空间: 取决于所选方法

---

### differential_score

计算正负基因集之间的差异分数。

::: source_code file="scl/kernel/scoring.hpp" symbol="differential_score" collapsed
:::

**复杂度**

- 时间: O(nnz)
- 空间: O(n_genes) 用于位集查找

---

### quantile_score

计算每个细胞中基因集表达的分位数。

::: source_code file="scl/kernel/scoring.hpp" symbol="quantile_score" collapsed
:::

**复杂度**

- 时间: O(n_cells * |gene_set| * log(|gene_set|))
- 空间: O(|gene_set|) 每个线程

---

### multi_signature_score

并行评分多个基因签名。

::: source_code file="scl/kernel/scoring.hpp" symbol="multi_signature_score" collapsed
:::

**复杂度**

- 时间: O(n_sets * nnz) - 与签名数量成线性关系
- 空间: O(n_genes) 每个签名的位集

---

## 评分方法

`ScoringMethod` 枚举提供不同的评分方法：

- `Mean`: 基因表达的简单平均值
- `RankBased`: 使用表达排名的基于 AUC 的分数
- `Weighted`: 用户提供权重的加权和
- `SeuratModule`: 带对照基因的 Seurat 风格模块分数
- `ZScore`: Z 分数归一化平均值

---

## 细胞周期阶段

`CellCyclePhase` 枚举表示细胞周期阶段：

- `G1 = 0`: Gap 1 期
- `S = 1`: 合成期
- `G2M = 2`: G2/有丝分裂期

---

## 相关内容

- [归一化模块](./normalize) - 表达归一化
- [比较模块](./comparison) - 统计比较

