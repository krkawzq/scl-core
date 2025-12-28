# 异常值检测

用于单细胞数据质量控制的异常值和异常检测内核。

## 概述

`outlier` 模块提供全面的异常值检测方法：

- **隔离分数**: 与总体的统计偏差
- **局部异常因子 (LOF)**: 基于密度的异常值检测
- **环境 RNA 检测**: 污染检测
- **空液滴检测**: EmptyDrops 风格算法
- **双联体检测**: 基于表达差异的评分

所有操作都：
- 稀疏输入内存高效
- 统计严谨
- 针对大型数据集优化

## 核心函数

### isolation_score

基于与总体细胞群体的统计偏差计算隔离分数。

```cpp
#include "scl/kernel/outlier.hpp"

Sparse<Real, true> expression = /* 表达矩阵 */;
Array<Real> scores(n_cells);

scl::kernel::outlier::isolation_score(expression, scores);
```

**参数:**
- `data` [in] - 表达矩阵（细胞 x 基因，CSR）
- `scores` [out] - 每个细胞的隔离分数

**前置条件:**
- `data.rows() == scores.len`
- `scores` 数组已预分配

**后置条件:**
- `scores[i] >= 0`，更高的值表示更孤立的细胞
- 分数是均值偏差和方差偏差的平均值

**复杂度:**
- 时间: O(nnz + n_cells * n_features)
- 空间: O(n_cells) 辅助空间

**线程安全:** 不安全 - 顺序实现

### local_outlier_factor

基于局部密度计算每个细胞的局部异常因子。

```cpp
Sparse<Index, true> neighbors = /* KNN 索引 */;
Sparse<Real, true> distances = /* KNN 距离 */;
Array<Real> lof_scores(n_cells);

scl::kernel::outlier::local_outlier_factor(
    expression, neighbors, distances, lof_scores
);
```

**参数:**
- `data` [in] - 表达矩阵
- `neighbors` [in] - KNN 邻居索引（CSR）
- `distances` [in] - KNN 距离（CSR）
- `lof_scores` [out] - 每个细胞的 LOF 分数

**前置条件:**
- 所有矩阵具有相同的行数
- `neighbors` 和 `distances` 对齐（相同结构）
- `lof_scores.len == data.rows()`

**后置条件:**
- `lof_scores[i] >= 0`
- LOF ~ 1 表示正常密度
- LOF > 1.5 通常表示异常值

**复杂度:**
- 时间: O(n_cells * k^2)，其中 k = 每个细胞的邻居数
- 空间: O(n_cells + k) 辅助空间

**线程安全:** 不安全 - 顺序实现

## 环境 RNA 检测

### ambient_detection

计算每个细胞的环境 RNA 污染分数。

```cpp
Array<Real> ambient_scores(n_cells);
scl::kernel::outlier::ambient_detection(expression, ambient_scores);
```

**参数:**
- `expression` [in] - 表达矩阵（细胞 x 基因）
- `ambient_scores` [out] - 每个细胞的环境污染分数

**前置条件:**
- `expression.rows() == ambient_scores.len`

**后置条件:**
- `ambient_scores[i]` 在 [0, 1] 范围内
- 1 表示与环境谱高度相关
- 0 表示无相关性

**复杂度:**
- 时间: O(n_cells * n_genes + nnz)
- 空间: O(n_cells + n_genes) 辅助空间

**线程安全:** 不安全 - 顺序实现

**算法:**
1. 计算每个细胞的总 UMI
2. 将底部 10% UMI 细胞识别为环境参考
3. 从参考细胞构建环境谱
4. 计算每个细胞与环境谱的余弦相似度

### empty_drops

使用与环境谱的偏差检验识别空液滴。

```cpp
Array<bool> is_empty(n_cells);
scl::kernel::outlier::empty_drops(raw_counts, is_empty, 0.01);
```

**参数:**
- `raw_counts` [in] - 原始 UMI 计数矩阵（细胞 x 基因）
- `is_empty` [out] - 空液滴的布尔掩码
- `fdr_threshold` [in] - 调用细胞的 FDR 阈值（默认: 0.01）

**前置条件:**
- `raw_counts.rows() == is_empty.len`
- `fdr_threshold` 在 (0, 1) 范围内

**后置条件:**
- `is_empty[i]` = true 如果细胞 i 可能是空的
- UMI < EMPTY_DROPS_MIN_UMI 的细胞标记为空

**复杂度:**
- 时间: O(n_cells * n_genes + n_cells log n_cells)
- 空间: O(n_cells + n_genes) 辅助空间

**线程安全:** 不安全 - 顺序实现

## 基因异常值检测

### outlier_genes

识别具有异常离散特征的基因。

```cpp
Index* outlier_indices = new Index[n_genes];
Size n_outliers;
scl::kernel::outlier::outlier_genes(
    expression, outlier_indices, n_outliers, 3.0
);
```

**参数:**
- `expression` [in] - 表达矩阵
- `outlier_gene_indices` [out] - 异常值基因的索引
- `n_outliers` [out] - 找到的异常值数量
- `threshold` [in] - 异常值检测的 Z 分数阈值（默认: 3.0）

**前置条件:**
- `outlier_gene_indices` 容量 >= expression.cols()
- `threshold > 0`

**后置条件:**
- `n_outliers` = 找到的异常值基因数量
- `outlier_gene_indices[0..n_outliers)` 包含异常值基因索引

**复杂度:**
- 时间: O(nnz + n_genes log n_genes)
- 空间: O(n_genes) 辅助空间

**线程安全:** 不安全 - 顺序实现

## 双联体检测

### doublet_score

基于与局部邻域的表达差异计算双联体分数。

```cpp
Sparse<Index, true> neighbors = /* KNN 图 */;
Array<Real> scores(n_cells);

scl::kernel::outlier::doublet_score(expression, neighbors, scores);
```

**参数:**
- `expression` [in] - 表达矩阵
- `neighbors` [in] - KNN 图
- `scores` [out] - 每个细胞的双联体分数

**前置条件:**
- `expression.rows() == neighbors.rows() == scores.len`

**后置条件:**
- `scores[i] >= 0`
- 更高的分数表示潜在的双联体

**复杂度:**
- 时间: O(n_cells * k * nnz_per_row)，其中 k = 每个细胞的邻居数
- 空间: O(1) 每个细胞辅助空间

**线程安全:** 不安全 - 顺序实现

## 线粒体异常值

### mitochondrial_outliers

识别线粒体基因含量高的细胞。

```cpp
Array<const Index> mito_genes = /* 线粒体基因索引 */;
Array<Real> mito_fraction(n_cells);
Array<bool> is_outlier(n_cells);

scl::kernel::outlier::mitochondrial_outliers(
    expression, mito_genes, mito_fraction, is_outlier, 0.2
);
```

**参数:**
- `expression` [in] - 表达矩阵
- `mito_genes` [in] - 线粒体基因的索引
- `mito_fraction` [out] - 每个细胞的线粒体分数
- `is_outlier` [out] - 高线粒体细胞的布尔掩码
- `threshold` [in] - 异常值状态的分数阈值（默认: 0.2）

**前置条件:**
- 所有数组大小与 expression.rows() 匹配
- `mito_genes` 索引在 expression.cols() 范围内
- `threshold` 在 (0, 1) 范围内

**后置条件:**
- `mito_fraction[i]` = 细胞 i 的 mito_UMI / total_UMI
- `is_outlier[i]` = (mito_fraction[i] > threshold)

**复杂度:**
- 时间: O(nnz + n_cells)
- 空间: O(max_gene_idx) 用于基因查找

**线程安全:** 不安全 - 顺序实现

## QC 过滤

### qc_filter

基于多个标准应用组合 QC 过滤。

```cpp
Array<const Index> mito_genes = /* 线粒体基因索引 */;
Array<bool> pass_qc(n_cells);

scl::kernel::outlier::qc_filter(
    expression,
    200,      // min_genes
    5000,     // max_genes
    1000,     // min_counts
    50000,    // max_counts
    0.2,      // max_mito_fraction
    mito_genes,
    pass_qc
);
```

**参数:**
- `expression` [in] - 表达矩阵
- `min_genes` [in] - 每个细胞的最小检测基因数
- `max_genes` [in] - 每个细胞的最大检测基因数
- `min_counts` [in] - 每个细胞的最小总计数
- `max_counts` [in] - 每个细胞的最大总计数
- `max_mito_fraction` [in] - 最大线粒体分数
- `mito_genes` [in] - 线粒体基因索引
- `pass_qc` [out] - 通过细胞的布尔掩码

**前置条件:**
- `pass_qc.len == expression.rows()`
- `min_genes <= max_genes`
- `min_counts <= max_counts`
- `max_mito_fraction` 在 [0, 1] 范围内

**后置条件:**
- `pass_qc[i]` = true 如果细胞通过所有 QC 标准

**复杂度:**
- 时间: O(nnz + n_cells)
- 空间: O(max_gene_idx) 用于线粒体基因查找

**线程安全:** 不安全 - 顺序实现

## 配置

```cpp
namespace scl::kernel::outlier::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_K_NEIGHBORS = 5;
    constexpr Size DEFAULT_K = 20;
    constexpr Real LOF_THRESHOLD = Real(1.5);
    constexpr Real AMBIENT_THRESHOLD = Real(0.1);
    constexpr Size EMPTY_DROPS_MIN_UMI = 100;
    constexpr Size EMPTY_DROPS_MAX_AMBIENT = 10;
    constexpr Size MONTE_CARLO_ITERATIONS = 10000;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## 使用场景

### 标准 QC 流程

```cpp
// 1. 检测空液滴
Array<bool> is_empty(n_cells);
scl::kernel::outlier::empty_drops(raw_counts, is_empty, 0.01);

// 2. 检测线粒体异常值
Array<const Index> mito_genes = /* MT- 基因 */;
Array<Real> mito_fraction(n_cells);
Array<bool> high_mito(n_cells);
scl::kernel::outlier::mitochondrial_outliers(
    expression, mito_genes, mito_fraction, high_mito, 0.2
);

// 3. 组合 QC 过滤
Array<bool> pass_qc(n_cells);
scl::kernel::outlier::qc_filter(
    expression, 200, 5000, 1000, 50000, 0.2, mito_genes, pass_qc
);

// 4. 过滤细胞
// (使用 pass_qc 掩码移除低质量细胞)
```

### 异常值检测

```cpp
// 基于 LOF 的异常值检测
Sparse<Index, true> knn = /* KNN 图 */;
Sparse<Real, true> distances = /* KNN 距离 */;
Array<Real> lof(n_cells);
scl::kernel::outlier::local_outlier_factor(
    expression, knn, distances, lof
);

// 过滤 LOF > 1.5 的细胞
for (Index i = 0; i < n_cells; ++i) {
    if (lof.ptr[i] > 1.5) {
        // 标记为异常值
    }
}
```

## 性能

- **内存高效**: 稀疏矩阵操作
- **统计严谨**: 适当的 FDR 控制和假设检验
- **可扩展**: 高效处理大型数据集
- **数值稳定**: 使用稳健的统计方法

---

::: tip 阈值选择
- **LOF**: > 1.5 通常表示异常值
- **线粒体**: > 0.2 分数通常表示受损细胞
- **空液滴**: FDR < 0.01 用于严格过滤
- **双联体分数**: 使用基于百分位数的阈值（例如，前 5%）
:::

