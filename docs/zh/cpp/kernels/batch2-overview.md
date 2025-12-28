# Batch 2 内核

> Batch 2: 邻居搜索、中心性、克隆型和共表达分析

## 概述

本批次包含以下内核：
- **批次平衡 KNN** - 跨批次邻居搜索
- **图中心性** - 网络重要性度量
- **克隆型分析** - 免疫库分析
- **共表达** - 基因模块检测（WGCNA 风格）

## 文件列表

| 文件 | 说明 | 主要 API |
|------|------|----------|
| [bbknn.hpp](./bbknn) | 批次平衡 KNN | `bbknn`, `compute_norms`, `build_batch_groups` |
| [centrality.hpp](./centrality) | 图中心性度量 | `pagerank`, `betweenness_centrality`, `degree_centrality` |
| [clonotype.hpp](./clonotype) | 克隆型分析 | `clonal_diversity`, `clone_expansion`, `clone_phenotype_association` |
| [coexpression.hpp](./coexpression) | 共表达模块 | `correlation_matrix`, `detect_modules`, `module_eigengene` |

## 快速开始

### 批次平衡 KNN

```cpp
#include "scl/kernel/bbknn.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<int32_t> batch_labels = /* ... */;
Array<Index> indices(/* ... */);
Array<Real> distances(/* ... */);

scl::kernel::bbknn::bbknn(
    matrix, batch_labels, n_batches, k, indices, distances
);
```

### 图中心性

```cpp
#include "scl/kernel/centrality.hpp"

Sparse<Real, true> adjacency = /* ... */;
Array<Real> scores(n_nodes);

scl::kernel::centrality::pagerank(adjacency, scores);
```

### 克隆型多样性

```cpp
#include "scl/kernel/clonotype.hpp"

Array<Index> clone_ids = /* ... */;
Real shannon, simpson, gini;

scl::kernel::clonotype::clonal_diversity(
    clone_ids, n_cells, shannon, simpson, gini
);
```

### 共表达模块

```cpp
#include "scl/kernel/coexpression.hpp"

Sparse<Real, true> expression = /* ... */;
Real* corr_matrix = /* allocate */;
Index* module_labels = /* allocate */;

scl::kernel::coexpression::correlation_matrix(
    expression, n_cells, n_genes, corr_matrix
);
Index n_modules = scl::kernel::coexpression::detect_modules(
    dissim, n_genes, module_labels
);
```

## 相关内容

- [Neighbors](/zh/cpp/kernels/neighbors) - 标准 KNN 搜索
- [Statistics](/zh/cpp/kernels/statistics) - 统计分析

