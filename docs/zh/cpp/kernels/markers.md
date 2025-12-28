# markers.hpp

> scl/kernel/markers.hpp · 标记基因选择和特异性评分

## 概述

用于单细胞 RNA-seq 分析的标记基因识别和评分函数。这些函数识别在特定聚类或细胞类型中差异表达的基因，支持细胞类型注释和生物学解释。

本文件提供：
- 通过差异表达分析发现标记基因
- 多种排序方法（倍数变化、效应大小、p 值、组合）
- 聚类特异性基因评分
- 针对大型数据集的并行处理

**头文件**: `#include "scl/kernel/markers.hpp"`

---

## 主要 API

### find_markers

::: source_code file="scl/kernel/markers.hpp" symbol="find_markers" collapsed
:::

**算法说明**

使用差异表达分析识别每个聚类的标记基因：

1. **对于每个聚类 c**（并行处理）：
   - 提取属于聚类 c 的细胞（组内）
   - 提取不属于聚类 c 的细胞（组外）
   - 对于每个基因 g：
     - 计算组内和组外的平均表达
     - 计算倍数变化：FC = (mean_in + pseudo_count) / (mean_out + pseudo_count)
     - 计算统计检验（t 检验或 Mann-Whitney U 检验）p 值
     - 计算效应大小（Cohen's d 或类似指标）
2. **根据所选方法对基因排序**：
   - FoldChange：按倍数变化排序（降序）
   - EffectSize：按效应大小排序（降序）
   - PValue：按 p 值排序（升序）
   - Combined：倍数变化、p 值和效应大小的加权组合
3. **过滤标记基因**：
   - 保留 FC >= min_fc 的基因
   - 保留 p 值 <= max_pval 的基因
   - 每个聚类选择前 max_markers 个基因
4. **存储结果**在 marker_genes 和 marker_scores 数组中

**边界条件**

- **空聚类**：未找到标记基因，所有 marker_genes 设置为无效索引
- **单细胞聚类**：统计检验可能不可靠，已优雅处理
- **所有基因被过滤**：每个聚类返回少于 max_markers 个基因
- **得分并列**：对于并列情况，排序是确定性的但顺序任意
- **零表达**：使用 pseudo_count (1.0) 避免除零

**数据保证（前置条件）**

- 表达矩阵必须是有效的稀疏矩阵（CSR 格式）
- 聚类标签数组长度必须等于细胞数（n_cells）
- 标记数组必须具有容量 >= n_clusters * max_markers
- 所有聚类标签必须在范围 [0, n_clusters-1] 内
- 聚类数必须 > 0

**复杂度分析**

- **时间**：最坏情况下为 O(n_clusters * n_genes * n_cells)。对聚类进行并行处理可减少有效时间。统计检验主导复杂度。
- **空间**：每个聚类 O(n_cells) 辅助空间用于存储组内/组外索引（并行化后，随线程数扩展）

**示例**

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* 表达矩阵 [n_cells x n_genes] */;
Array<Index> cluster_labels = /* 聚类分配 [n_cells] */;

Index n_cells = expression.rows();
Index n_genes = expression.cols();
Index n_clusters = /* ... */;
Index max_markers = 50;

Index* marker_genes = new Index[n_clusters * max_markers];
Real* marker_scores = new Real[n_clusters * max_markers];

// 使用默认参数查找标记基因
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers
);

// 使用自定义阈值查找标记基因
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers,
    min_fc = 2.0,        // 最小 2 倍变化
    max_pval = 0.01,     // 最大 p 值 0.01
    method = scl::kernel::markers::RankingMethod::Combined
);

// 访问聚类 c 的标记基因：
// marker_genes[c * max_markers + i] = 基因索引
// marker_scores[c * max_markers + i] = 得分
```

---

### specificity_score

::: source_code file="scl/kernel/markers.hpp" symbol="specificity_score" collapsed
:::

**算法说明**

计算单个基因的聚类特异性表达分数：

1. 计算基因在目标聚类中的平均表达
2. 计算基因在所有其他聚类中的平均表达
3. 将特异性计算为比率或差异度量：
   - 通常使用对数倍数变化或 z 分数标准化
   - 较高的值表示对目标聚类的特异性更强

具体公式可能有所不同，但通常衡量该基因在目标聚类中与其他聚类相比的表达程度。

**边界条件**

- **基因未表达**：返回低特异性分数（接近零或负值）
- **均匀表达**：返回接近零的特异性
- **空目标聚类**：返回未定义/无效分数
- **单细胞聚类**：特异性方差可能较大

**数据保证（前置条件）**

- 表达矩阵必须是有效的稀疏矩阵
- 聚类标签数组长度必须等于 n_cells
- 目标聚类 ID 必须有效（在范围 [0, n_clusters-1] 内）
- 基因索引必须有效（在范围 [0, n_genes-1] 内）

**复杂度分析**

- **时间**：O(n_cells) - 必须检查给定基因在所有细胞中的表达
- **空间**：O(1) 辅助空间 - 仅累积统计信息

**示例**

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* 表达矩阵 */;
Array<Index> cluster_labels = /* 聚类标签 */;
Index target_cluster = 3;
Index gene_index = 125;  // 感兴趣的基因
Index n_cells = expression.rows();

Real specificity;
scl::kernel::markers::specificity_score(
    expression, cluster_labels, gene_index, target_cluster, n_cells, specificity
);

// specificity 现在包含聚类特异性表达分数
// 较高的值表示该基因对聚类 3 更具特异性
```

---

## 配置

命名空间 `scl::kernel::markers::config` 提供配置常量：

- `DEFAULT_MIN_FC = 1.5`：默认最小倍数变化阈值
- `DEFAULT_MIN_PCT = 0.1`：默认表达该基因的细胞最小百分比
- `DEFAULT_MAX_PVAL = 0.05`：默认最大 p 值阈值
- `MIN_EXPR = 1e-9`：最小表达值（数值稳定性）
- `PSEUDO_COUNT = 1.0`：添加的伪计数以避免 log(0) 和除零
- `PARALLEL_THRESHOLD = 500`：并行处理的最小聚类大小
- `SIMD_THRESHOLD = 32`：SIMD 优化的最小向量长度

## 排序方法

`RankingMethod` 枚举提供不同的标记基因排序策略：

- `FoldChange`：按倍数变化排序（降序）
- `EffectSize`：按效应大小排序（降序）
- `PValue`：按 p 值排序（升序）
- `Combined`：多个指标的加权组合

## 注意事项

- 标记基因识别对表达矩阵预处理（标准化、对数变换）敏感。在标记分析之前确保一致的预处理。
- 对于小聚类（< 5 个细胞），统计检验可能不可靠。考虑仅使用效应大小或倍数变化。
- 添加伪计数可防止零表达值的数值问题，但可能影响极低表达基因的结果。
- 标记基因列表应在生物系统背景下解释。高表达基因可能主导排名，即使它们没有生物学意义。

## 相关内容

- [多重检验](./multiple_testing) - 多重假设检验的统计校正
- [统计](./statistics) - 标记基因识别中使用的统计检验
- [评分](./scoring) - 基因选择的替代评分方法

