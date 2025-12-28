# 关联

跨模态（RNA + ATAC）的特征关联分析，用于多组学整合。

## 概述

关联模块提供：

- **基因-峰相关性** - RNA 表达与 ATAC 可及性之间的相关性
- **顺式调控元件** - 识别顺式连接的调控元件

## 基因-峰相关性

### gene_peak_correlation

计算基因与可及峰之间的相关性：

```cpp
#include "scl/kernel/association.hpp"

Sparse<Real, true> rna_expression = /* ... */;  // 细胞 x 基因
Sparse<Real, true> atac_peaks = /* ... */;  // 细胞 x 峰
Real* correlations = /* 分配 n_genes * n_peaks */;

scl::kernel::association::gene_peak_correlation(
    rna_expression,
    atac_peaks,
    n_cells,
    n_genes,
    n_peaks,
    correlations
);
```

**参数：**
- `rna_expression`: RNA 表达矩阵（细胞 x 基因，CSR）
- `atac_peaks`: ATAC 峰矩阵（细胞 x 峰，CSR）
- `n_cells`: 细胞数量
- `n_genes`: 基因数量
- `n_peaks`: 峰数量
- `correlations`: 输出的相关性矩阵 [n_genes * n_peaks]

**后置条件：**
- `correlations[g * n_peaks + p]` 包含基因 g 与峰 p 之间的相关性

**复杂度：**
- 时间：O(n_genes * n_peaks * n_cells)
- 空间：O(n_cells) 辅助空间（每个基因-峰对）

**使用场景：**
- 多组学整合
- 调控元件识别
- 基因-峰关联

## 顺式调控元件

### cis_regulatory_elements

识别与基因连接的顺式调控元件：

```cpp
const Real* correlations = /* ... */;  // 基因-峰相关性
const Index* peak_positions = /* ... */;  // [n_peaks * 2] (起始, 结束)
const Index* gene_positions = /* ... */;  // [n_genes * 2] (起始, 结束)
Index* linked_pairs = /* 分配 max_results * 2 */;
Real* link_scores = /* 分配 max_results */;

Index n_linked = scl::kernel::association::cis_regulatory_elements(
    correlations,
    peak_positions,
    gene_positions,
    n_genes,
    n_peaks,
    max_distance,  // 顺式连接的最大距离
    linked_pairs,
    link_scores,
    max_results
);
```

**参数：**
- `correlations`: 基因-峰相关性 [n_genes * n_peaks]
- `peak_positions`: 峰基因组位置 [n_peaks * 2] (起始, 结束)
- `gene_positions`: 基因位置 [n_genes * 2] (起始, 结束)
- `n_genes`: 基因数量
- `n_peaks`: 峰数量
- `max_distance`: 顺式连接的最大距离
- `linked_pairs`: 输出的连接基因-峰对 [max_results * 2]
- `link_scores`: 输出的连接评分 [max_results]
- `max_results`: 最大结果数

**返回：** 找到的连接对数量

**后置条件：**
- `linked_pairs[i * 2]` = 基因索引，`linked_pairs[i * 2 + 1]` = 峰索引
- 仅包含在 max_distance 内的对

**算法：**
对于每个基因并行：
1. 查找 max_distance 内的峰
2. 检查相关性阈值
3. 存储带评分的连接对

**复杂度：**
- 时间：O(n_genes * n_peaks)
- 空间：O(1) 辅助空间

**使用场景：**
- 顺式调控元件发现
- 增强子-启动子连接
- 调控网络构建

## 配置

`scl::kernel::association::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real EPSILON = 1e-10;
    constexpr Real MIN_CORRELATION = 0.1;
    constexpr Size MIN_CELLS_FOR_CORRELATION = 10;
    constexpr Size MAX_LINKS_PER_GENE = 1000;
    constexpr Size PARALLEL_THRESHOLD = 32;
}
```

## 性能考虑

### 并行化

- `gene_peak_correlation`: 在基因-峰对上并行
- `cis_regulatory_elements`: 在基因上并行

### 内存效率

- 高效的稀疏矩阵访问
- 预分配的输出缓冲区
- 最少的临时分配

---

::: tip 距离过滤
使用基因组距离（例如，1 Mb）过滤顺式调控元件，因为大多数调控相互作用是局部的。
:::

::: warning 相关性阈值
按最小阈值（默认：0.1）过滤相关性，以减少来自噪声的假阳性。
:::

