# 细胞类型注释

基于参考数据集的细胞类型注释，支持多种方法。

## 概述

注释模块提供：

- **参考映射** - 基于 KNN 的细胞类型分配
- **相关性分配** - SingleR 风格的相关性匹配
- **标记基因评分** - scType 风格的基于标记的注释
- **共识方法** - 组合多种注释方法
- **新类型检测** - 识别未注释的细胞类型

## 参考映射

### reference_mapping

使用参考数据集的 KNN 投票来注释查询细胞：

```cpp
#include "scl/kernel/annotation.hpp"

Sparse<Real, true> query_expression = /* ... */;
Sparse<Real, true> reference_expression = /* ... */;
Array<const Index> reference_labels = /* ... */;
Sparse<Index, true> query_to_ref_neighbors = /* ... */;  // KNN 图

Array<Index> query_labels(n_query);
Array<Real> confidence_scores(n_query);

scl::kernel::annotation::reference_mapping(
    query_expression,
    reference_expression,
    reference_labels,
    query_to_ref_neighbors,
    n_query,
    n_ref,
    n_types,
    query_labels,
    confidence_scores
);
```

**参数：**
- `query_expression`: 查询表达矩阵（细胞 x 基因，CSR）
- `reference_expression`: 参考表达矩阵（细胞 x 基因，CSR）
- `reference_labels`: 参考细胞类型标签 [n_ref]
- `query_to_ref_neighbors`: 从查询到参考的 KNN 图（CSR）
- `query_labels`: 输出的注释查询标签 [n_query]
- `confidence_scores`: 输出的置信度分数 [n_query]

**算法：**
1. 对于每个查询细胞，收集 K 个最近的参考邻居
2. 统计邻居中每种细胞类型的投票数
3. 分配获得最多投票的类型
4. 计算置信度为归一化的投票分数

**复杂度：**
- 时间：O(n_query * k * n_genes)
- 空间：O(k * n_types) 辅助空间（每线程）

**使用场景：**
- 从参考图谱进行细胞类型注释
- 从标记数据集的迁移学习
- 使用参考进行批次校正

### count_cell_types

统计标签数组中不同细胞类型的数量：

```cpp
Index n_types = scl::kernel::annotation::count_cell_types(
    reference_labels,
    n_ref
);
```

**返回：** 不同类型的数量（max_label + 1）

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

## 相关性分配

### correlation_assignment

使用与参考谱的相关性来分配细胞类型（SingleR 风格）：

```cpp
Sparse<Real, true> query_expression = /* ... */;
Sparse<Real, true> reference_profiles = /* ... */;  // 类型 x 基因

Array<Index> assigned_labels(n_query);
Array<Real> correlation_scores(n_query);
Array<Real> all_correlations(n_query * n_types);  // 可选

scl::kernel::annotation::correlation_assignment(
    query_expression,
    reference_profiles,
    n_query,
    n_types,
    n_genes,
    assigned_labels,
    correlation_scores,
    all_correlations  // 可选：所有相关性
);
```

**参数：**
- `query_expression`: 查询表达矩阵（细胞 x 基因，CSR）
- `reference_profiles`: 参考类型谱（类型 x 基因，CSR）
- `assigned_labels`: 输出的分配标签 [n_query]
- `correlation_scores`: 输出的最佳相关性分数 [n_query]
- `all_correlations`: 可选的输出，包含所有相关性 [n_query * n_types]

**算法：**
1. 对于每个查询细胞，计算与每个参考谱的相关性
2. 分配相关性最高的类型
3. 将最佳相关性存储为置信度

**复杂度：**
- 时间：O(n_query * n_types * n_genes)
- 空间：O(n_types) 辅助空间（每个查询细胞）

**使用场景：**
- SingleR 风格的注释
- 大型数据集的快速注释
- 当参考谱已预计算时

### build_reference_profiles

为每种细胞类型构建平均表达谱：

```cpp
Real* profiles = /* 分配 n_types * n_genes */;

scl::kernel::annotation::build_reference_profiles(
    reference_expression,
    reference_labels,
    n_ref,
    n_types,
    n_genes,
    profiles
);
```

**参数：**
- `reference_expression`: 参考表达矩阵（细胞 x 基因，CSR）
- `reference_labels`: 参考细胞类型标签 [n_ref]
- `profiles`: 输出的类型谱 [n_types * n_genes]

**后置条件：**
- `profiles[t * n_genes + g]` 包含类型 t 在基因 g 上的平均表达

**复杂度：**
- 时间：O(nnz_ref)
- 空间：O(n_types) 辅助空间

**使用场景：**
- 为相关性分配预计算谱
- 创建参考签名
- 类型特异性表达分析

## 标记基因评分

### marker_gene_score

使用标记基因表达来评分细胞（scType 风格）：

```cpp
const Index* const* marker_genes = /* 每种类型的标记数组数组 */;
const Index* marker_counts = /* 每种类型的标记数量 */;
Real* scores = /* 分配 n_cells * n_types */;

scl::kernel::annotation::marker_gene_score(
    expression,
    marker_genes,
    marker_counts,
    n_cells,
    n_genes,
    n_types,
    scores,
    true  // normalize = true
);
```

**参数：**
- `expression`: 表达矩阵（细胞 x 基因，CSR）
- `marker_genes`: 每种类型的标记基因数组 [n_types]
- `marker_counts`: 每种类型的标记数量 [n_types]
- `scores`: 输出的标记分数 [n_cells * n_types]
- `normalize`: 如果为 true，按细胞归一化分数

**算法：**
1. 对于每个细胞和类型，对标记基因的表达求和
2. 可选地按总标记表达归一化
3. 更高的分数表示更好的匹配

**复杂度：**
- 时间：O(n_cells * sum(marker_counts))
- 空间：O(n_genes) 辅助空间（每线程）

**使用场景：**
- scType 风格的注释
- 当标记基因已知时
- 特定细胞类型的快速注释

### assign_from_scores

通过选择最大值从分数矩阵分配细胞类型：

```cpp
Array<Index> labels(n_cells);
Array<Real> confidence(n_cells);

scl::kernel::annotation::assign_from_scores(
    scores,      // 分数矩阵 [n_cells * n_types]
    n_cells,
    n_types,
    labels,
    confidence
);
```

**参数：**
- `scores`: 分数矩阵 [n_cells * n_types]
- `labels`: 输出的分配标签 [n_cells]
- `confidence`: 输出的置信度分数 [n_cells]

**后置条件：**
- `labels[i]` 包含分数最高的类型
- `confidence[i]` 包含归一化的最大分数

**复杂度：**
- 时间：O(n_cells * n_types)
- 空间：O(1) 辅助空间

## 共识方法

### consensus_annotation

组合来自多种注释方法的预测：

```cpp
const Index* const* predictions = /* 预测数组数组 */;
const Real* const* confidences = /* 可选的置信度数组 */;
Array<Index> consensus_labels(n_cells);
Array<Real> consensus_confidence(n_cells);

scl::kernel::annotation::consensus_annotation(
    predictions,
    confidences,  // 可选
    n_methods,
    n_cells,
    n_types,
    consensus_labels,
    consensus_confidence
);
```

**参数：**
- `predictions`: 预测数组数组 [n_methods][n_cells]
- `confidences`: 可选的置信度数组 [n_methods][n_cells]
- `consensus_labels`: 输出的共识标签 [n_cells]
- `consensus_confidence`: 输出的共识置信度 [n_cells]

**算法：**
1. 对于每个细胞，收集所有方法的预测
2. 使用多数投票或加权投票（如果提供了置信度）
3. 计算一致性度量作为置信度

**复杂度：**
- 时间：O(n_cells * n_methods * n_types)
- 空间：O(n_types) 辅助空间（每个细胞）

**使用场景：**
- 组合 KNN、相关性和基于标记的预测
- 提高注释准确性
- 处理方法分歧

## 新类型检测

### detect_novel_cell_types

检测与任何参考类型匹配不佳的细胞：

```cpp
Array<Byte> is_novel(n_query);
Array<Real> distance_to_assigned(n_query);  // 可选

scl::kernel::annotation::detect_novel_cell_types(
    query_expression,
    reference_profiles,
    assigned_labels,
    n_query,
    n_types,
    n_genes,
    is_novel,
    config::DEFAULT_NOVELTY_THRESHOLD,  // distance_threshold
    distance_to_assigned  // 可选
);
```

**参数：**
- `query_expression`: 查询表达矩阵（细胞 x 基因，CSR）
- `reference_profiles`: 参考类型谱 [n_types * n_genes]
- `assigned_labels`: 先前分配的标签 [n_query]
- `is_novel`: 输出的新类型标志 [n_query]
- `distance_threshold`: 新类型的距离阈值
- `distance_to_assigned`: 可选，到分配类型的距离 [n_query]

**后置条件：**
- `is_novel[i] == 1` 如果细胞 i 是新类型
- 新类型细胞与分配类型谱的相似度低

**复杂度：**
- 时间：O(n_query * n_genes)
- 空间：O(n_genes) 辅助空间（每个查询细胞）

**使用场景：**
- 识别未注释的细胞类型
- 注释的质量控制
- 新细胞类型的发现

## 配置

`scl::kernel::annotation::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real DEFAULT_CONFIDENCE_THRESHOLD = 0.5;
    constexpr Real EPSILON = 1e-15;
    constexpr Index DEFAULT_K = 15;
    constexpr Real DEFAULT_NOVELTY_THRESHOLD = 0.3;
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## 注释方法

### AnnotationMethod 枚举

```cpp
enum class AnnotationMethod {
    KNNVoting,      // K 近邻投票
    Correlation,    // 与参考谱的相关性
    MarkerScore,    // 标记基因评分
    Weighted        // 加权组合
};
```

### DistanceMetric 枚举

```cpp
enum class DistanceMetric {
    Cosine,
    Euclidean,
    Correlation,
    Manhattan
};
```

## 性能考虑

### 并行化

所有注释函数都已并行化：
- `reference_mapping`: 在查询细胞上并行
- `correlation_assignment`: 在查询细胞上并行
- `marker_gene_score`: 在细胞上并行
- `consensus_annotation`: 在细胞上并行

### 内存效率

- 预分配的输出缓冲区
- 在并行循环中重用工作空间
- 最少的临时分配

## 最佳实践

### 1. 预计算参考谱

```cpp
// 构建谱一次
Real* profiles = /* 分配 */;
scl::kernel::annotation::build_reference_profiles(
    reference_expression,
    reference_labels,
    n_ref, n_types, n_genes,
    profiles
);

// 为多个查询重用
for (auto& query : queries) {
    scl::kernel::annotation::correlation_assignment(
        query, profiles, /* ... */
    );
}
```

### 2. 使用共识提高准确性

```cpp
// 运行多种方法
Array<Index> knn_labels(n_cells);
Array<Index> corr_labels(n_cells);
Array<Index> marker_labels(n_cells);

// ... 计算预测 ...

// 使用共识组合
const Index* predictions[] = {
    knn_labels.ptr,
    corr_labels.ptr,
    marker_labels.ptr
};
Array<Index> consensus_labels(n_cells);
scl::kernel::annotation::consensus_annotation(
    predictions, nullptr, 3, n_cells, n_types,
    consensus_labels, /* ... */
);
```

### 3. 检测新类型

```cpp
// 注释后，检查新类型
Array<Byte> is_novel(n_query);
scl::kernel::annotation::detect_novel_cell_types(
    query_expression,
    reference_profiles,
    assigned_labels,
    n_query, n_types, n_genes,
    is_novel,
    0.3  // 阈值
);

// 处理新类型细胞
for (Index i = 0; i < n_query; ++i) {
    if (is_novel[i]) {
        // 标记为未注释或新类型
    }
}
```

---

::: tip 多种方法
使用共识注释来组合来自多种方法的预测，以提高准确性。
:::

::: warning 新类型
注释后始终检查新类型细胞，以识别未注释的群体。
:::

