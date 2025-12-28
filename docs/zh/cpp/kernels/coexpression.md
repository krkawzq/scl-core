# 共表达

用于基因网络分析的高性能共表达模块检测（WGCNA 风格）。

## 概述

共表达模块提供：

- **相关性计算** - 成对基因相关性（Pearson、Spearman、Bicor）
- **邻接矩阵** - 使用软幂将相关性转换为邻接
- **拓扑重叠** - TOM 矩阵计算
- **层次聚类** - 通过聚类进行模块检测
- **模块特征基因** - 每个模块的第一主成分
- **模块-性状相关性** - 与外部性状的关联

## 相关性计算

### correlation_matrix

计算基因的成对相关性矩阵：

```cpp
#include "scl/kernel/coexpression.hpp"

Sparse<Real, true> expression = /* ... */;  // 细胞 x 基因
Real* corr_matrix = /* 分配 n_genes * n_genes */;

scl::kernel::coexpression::correlation_matrix(
    expression,
    n_cells,
    n_genes,
    corr_matrix,
    CorrelationType::Pearson  // 或 Spearman, Bicor
);
```

**参数：**
- `expression`: 表达矩阵（细胞 x 基因，CSR）
- `corr_matrix`: 输出的相关性矩阵 [n_genes * n_genes]（上三角）
- `corr_type`: 相关性类型（Pearson、Spearman、Bicor）

**后置条件：**
- `corr_matrix[i * n_genes + j]` 包含基因 i 和 j 之间的相关性
- 矩阵是对称的

**复杂度：**
- 时间：O(n_genes^2 * n_cells)
- 空间：O(n_genes * n_cells) 辅助空间

**使用场景：**
- 基因共表达分析
- 网络构建
- 模块检测流程

## 邻接矩阵

### adjacency_matrix

使用软幂将相关性矩阵转换为邻接矩阵：

```cpp
Real* adjacency = /* 分配 n_genes * n_genes */;

scl::kernel::coexpression::adjacency_matrix(
    corr_matrix,
    n_genes,
    adjacency,
    config::DEFAULT_SOFT_POWER,  // power = 6
    AdjacencyType::Unsigned      // 或 Signed, SignedHybrid
);
```

**参数：**
- `corr_matrix`: 相关性矩阵 [n_genes * n_genes]
- `adjacency`: 输出的邻接矩阵 [n_genes * n_genes]
- `power`: 软幂（默认 6）
- `adj_type`: 邻接类型（Unsigned、Signed、SignedHybrid）

**后置条件：**
- `adjacency[i * n_genes + j]` 包含邻接值
- 对于无符号，值在 [0, 1] 范围内；对于有符号，值在 [-1, 1] 范围内

**算法：**
- Unsigned: `adj = |corr|^power`
- Signed: `adj = (0.5 + 0.5 * corr)^power`
- SignedHybrid: 如果 corr > 0，则 `adj = corr^power`，否则为 0

**复杂度：**
- 时间：O(n_genes^2)
- 空间：O(1) 辅助空间

**使用场景：**
- WGCNA 风格的网络构建
- 加权基因网络
- 模块检测预处理

## 拓扑重叠

### topological_overlap_matrix

从邻接矩阵计算拓扑重叠矩阵（TOM）：

```cpp
Real* tom = /* 分配 n_genes * n_genes */;

scl::kernel::coexpression::topological_overlap_matrix(
    adjacency,
    n_genes,
    tom
);
```

**参数：**
- `adjacency`: 邻接矩阵 [n_genes * n_genes]
- `tom`: 输出的 TOM 矩阵 [n_genes * n_genes]

**后置条件：**
- `tom[i * n_genes + j]` 包含 TOM 值
- TOM 测量基因之间的共享邻居

**算法：**
`TOM[i,j] = (sum_k(min(A[i,k], A[j,k])) + A[i,j]) / (min(degree[i], degree[j]) + 1 - A[i,j])`

**复杂度：**
- 时间：O(n_genes^3)
- 空间：O(n_genes) 辅助空间

**使用场景：**
- 模块检测（WGCNA）
- 网络平滑
- 共享邻居分析

### tom_dissimilarity

将 TOM 矩阵转换为用于聚类的相异性矩阵：

```cpp
Real* dissim = /* 分配 n_genes * n_genes */;

scl::kernel::coexpression::tom_dissimilarity(
    tom,
    n_genes,
    dissim
);
```

**参数：**
- `tom`: TOM 矩阵 [n_genes * n_genes]
- `dissim`: 输出的相异性矩阵 [n_genes * n_genes]

**后置条件：**
- `dissim[i * n_genes + j] = 1 - tom[i * n_genes + j]`

**复杂度：**
- 时间：O(n_genes^2)
- 空间：O(1) 辅助空间

**使用场景：**
- 层次聚类输入
- 基于距离的模块检测

## 模块检测

### hierarchical_clustering

在相异性矩阵上执行层次聚类：

```cpp
Index* merge_order = /* 分配 2 * (n_genes - 1) */;
Real* merge_heights = /* 分配 n_genes - 1 */;

scl::kernel::coexpression::hierarchical_clustering(
    dissim,
    n_genes,
    merge_order,
    merge_heights,
    nullptr  // 可选树状图
);
```

**参数：**
- `dissim`: 相异性矩阵 [n_genes * n_genes]
- `merge_order`: 输出的合并顺序 [2 * (n_genes - 1)]
- `merge_heights`: 输出的合并高度 [n_genes - 1]
- `dendrogram`: 可选的树状图结构

**后置条件：**
- `merge_order` 包含合并的聚类对
- `merge_heights` 包含合并距离

**复杂度：**
- 时间：O(n_genes^2 * log(n_genes))
- 空间：O(n_genes^2) 辅助空间

**使用场景：**
- 模块检测
- 基因聚类
- 层次结构分析

### cut_tree

在指定高度处切割层次聚类树：

```cpp
Index* module_labels = /* 分配 n_genes */;

Index n_modules = scl::kernel::coexpression::cut_tree(
    merge_order,
    merge_heights,
    n_genes,
    cut_height,      // 切割高度阈值
    module_labels
);
```

**参数：**
- `merge_order`: 来自层次聚类的合并顺序 [2*(n-1)]
- `merge_heights`: 合并高度 [n-1]
- `cut_height`: 切割高度阈值
- `module_labels`: 输出的模块标签 [n_genes]

**返回：** 检测到的模块数量

**后置条件：**
- `module_labels[i]` 包含基因 i 的模块 ID

**复杂度：**
- 时间：O(n_genes)
- 空间：O(n_genes) 辅助空间

**使用场景：**
- 模块分配
- 动态树切割
- 模块大小控制

### detect_modules

从相异性矩阵检测共表达模块：

```cpp
Index* module_labels = /* 分配 n_genes */;

Index n_modules = scl::kernel::coexpression::detect_modules(
    dissim,
    n_genes,
    module_labels,
    config::DEFAULT_MIN_MODULE_SIZE,  // min_module_size = 30
    config::DEFAULT_MERGE_CUT_HEIGHT  // merge_cut_height = 0.25
);
```

**参数：**
- `dissim`: 相异性矩阵 [n_genes * n_genes]
- `module_labels`: 输出的模块标签 [n_genes]
- `min_module_size`: 最小模块大小
- `merge_cut_height`: 合并切割高度

**返回：** 检测到的模块数量

**复杂度：**
- 时间：O(n_genes^2 * log(n_genes))
- 空间：O(n_genes^2) 辅助空间

**使用场景：**
- 完整的模块检测流程
- WGCNA 风格分析
- 基因网络模块

## 模块特征基因

### module_eigengene

计算模块的特征基因（第一主成分）：

```cpp
Array<Real> eigengene(n_cells);

scl::kernel::coexpression::module_eigengene(
    expression,
    module_labels,
    module_id,      // 要计算特征基因的模块 ID
    n_cells,
    n_genes,
    eigengene
);
```

**参数：**
- `expression`: 表达矩阵（细胞 x 基因，CSR）
- `module_labels`: 模块标签 [n_genes]
- `module_id`: 要计算特征基因的模块 ID
- `eigengene`: 输出的模块特征基因 [n_cells]

**后置条件：**
- `eigengene[i]` 包含细胞 i 中模块表达的第一主成分

**复杂度：**
- 时间：O(n_cells * n_module_genes)
- 空间：O(n_cells) 辅助空间

**使用场景：**
- 模块表示
- 模块-性状相关性
- 降维

### all_module_eigengenes

计算所有模块的特征基因：

```cpp
Real* eigengenes = /* 分配 n_cells * n_modules */;

scl::kernel::coexpression::all_module_eigengenes(
    expression,
    module_labels,
    n_modules,
    n_cells,
    n_genes,
    eigengenes
);
```

**参数：**
- `eigengenes`: 输出的特征基因矩阵 [n_cells * n_modules]

**后置条件：**
- `eigengenes[i * n_modules + m]` 包含细胞 i 中模块 m 的特征基因

**复杂度：**
- 时间：O(n_modules * n_cells * avg_module_size)
- 空间：O(n_cells) 辅助空间（每个模块）

**使用场景：**
- 模块汇总统计
- 模块-性状分析
- 模块可视化

## 模块-性状相关性

### module_trait_correlation

计算模块特征基因与性状之间的相关性：

```cpp
Real* correlations = /* 分配 n_modules * n_traits */;
Real* p_values = /* 分配 n_modules * n_traits */;  // 可选

scl::kernel::coexpression::module_trait_correlation(
    eigengenes,     // 模块特征基因 [n_samples * n_modules]
    traits,         // 性状值 [n_samples * n_traits]
    n_samples,
    n_modules,
    n_traits,
    correlations,
    p_values        // 可选
);
```

**参数：**
- `eigengenes`: 模块特征基因 [n_samples * n_modules]
- `traits`: 性状值 [n_samples * n_traits]
- `correlations`: 输出的相关性矩阵 [n_modules * n_traits]
- `p_values`: 可选的 p 值 [n_modules * n_traits]

**后置条件：**
- `correlations[m * n_traits + t]` 包含相关性

**复杂度：**
- 时间：O(n_modules * n_traits * n_samples)
- 空间：O(n_samples) 辅助空间

**使用场景：**
- 模块-性状关联
- 生物学解释
- 疾病模块识别

## 配置

`scl::kernel::coexpression::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real DEFAULT_SOFT_POWER = 6;
    constexpr Real EPSILON = 1e-15;
    constexpr Index DEFAULT_MIN_MODULE_SIZE = 30;
    constexpr Index DEFAULT_DEEP_SPLIT = 2;
    constexpr Real DEFAULT_MERGE_CUT_HEIGHT = 0.25;
    constexpr Index MAX_ITERATIONS = 100;
    constexpr Size PARALLEL_THRESHOLD = 64;
    constexpr Size SIMD_THRESHOLD = 16;
}
```

## 相关性类型

### CorrelationType 枚举

```cpp
enum class CorrelationType {
    Pearson,   // Pearson 相关性
    Spearman,  // Spearman 秩相关性
    Bicor      // 双权重中相关性（鲁棒）
};
```

### AdjacencyType 枚举

```cpp
enum class AdjacencyType {
    Unsigned,      // |corr|^power
    Signed,        // (0.5 + 0.5*corr)^power
    SignedHybrid   // 如果 corr > 0，则 corr^power，否则为 0
};
```

## 性能考虑

### 并行化

大多数操作都已并行化：
- `correlation_matrix`: 在基因对上并行
- `adjacency_matrix`: 在矩阵元素上并行
- `topological_overlap_matrix`: 在基因对上并行
- `module_eigengene`: 已并行化

### 内存效率

- 预分配的输出缓冲区
- 尽可能使用原地操作
- 高效的矩阵操作

## 最佳实践

### 1. 完整的 WGCNA 流程

```cpp
// 1. 计算相关性
Real* corr_matrix = /* 分配 */;
scl::kernel::coexpression::correlation_matrix(
    expression, n_cells, n_genes, corr_matrix, CorrelationType::Pearson
);

// 2. 构建邻接
Real* adjacency = /* 分配 */;
scl::kernel::coexpression::adjacency_matrix(
    corr_matrix, n_genes, adjacency, 6, AdjacencyType::Unsigned
);

// 3. 计算 TOM
Real* tom = /* 分配 */;
scl::kernel::coexpression::topological_overlap_matrix(
    adjacency, n_genes, tom
);

// 4. 转换为相异性
Real* dissim = /* 分配 */;
scl::kernel::coexpression::tom_dissimilarity(tom, n_genes, dissim);

// 5. 检测模块
Index* module_labels = /* 分配 */;
Index n_modules = scl::kernel::coexpression::detect_modules(
    dissim, n_genes, module_labels
);

// 6. 计算特征基因
Real* eigengenes = /* 分配 n_cells * n_modules */;
scl::kernel::coexpression::all_module_eigengenes(
    expression, module_labels, n_modules, n_cells, n_genes, eigengenes
);
```

### 2. 选择适当的相关性

```cpp
// 对于正态数据
CorrelationType::Pearson

// 对于非正态数据
CorrelationType::Spearman

// 对于鲁棒分析
CorrelationType::Bicor
```

### 3. 调整软幂

```cpp
// 测试不同的幂
for (Real power = 4; power <= 20; power += 2) {
    Real* adjacency = /* 分配 */;
    scl::kernel::coexpression::adjacency_matrix(
        corr_matrix, n_genes, adjacency, power, AdjacencyType::Unsigned
    );
    
    // 检查无标度拓扑
    // 选择最适合无标度网络的幂
}
```

---

::: tip WGCNA 流程
遵循完整的 WGCNA 流程：相关性 -> 邻接 -> TOM -> 模块 -> 特征基因 -> 性状相关性。
:::

::: warning 内存
TOM 计算在时间和空间上都是 O(n_genes^3)。适用于中等大小的基因集（n_genes < 10000）。
:::

