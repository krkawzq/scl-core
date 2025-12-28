# 插补

用于单细胞表达数据的高性能插补内核。

## 概述

`impute` 模块为稀疏单细胞数据提供高效的插补方法：

- **KNN 插补**: K 近邻平均
- **扩散插补**: MAGIC 风格的基于扩散的插补
- **ALRA**: 自适应阈值低秩近似
- **加权 KNN**: 距离加权邻居平均

所有操作都：
- 按细胞并行化
- 稀疏输入内存高效
- 支持稀疏和密集输出

## 核心函数

### knn_impute_dense

使用 K 近邻平均在密集输出上插补缺失值。

```cpp
#include "scl/kernel/impute.hpp"

Sparse<Real, true> X_sparse = /* 稀疏表达矩阵 */;
Sparse<Real, true> affinity = /* 细胞-细胞亲和矩阵 */;
Real* X_imputed = /* 预分配 [n_cells * n_genes] */;

scl::kernel::impute::knn_impute_dense(
    X_sparse, affinity, n_cells, n_genes, X_imputed
);
```

**参数:**
- `X_sparse` [in] - 输入稀疏表达矩阵（n_cells x n_genes）
- `affinity` [in] - 细胞-细胞亲和矩阵（n_cells x n_cells）
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `X_imputed` [out] - 密集插补矩阵（n_cells x n_genes，行主序）

**前置条件:**
- `X_sparse` 必须是 CSR 格式（细胞 x 基因）
- `affinity` 必须行归一化（行和为 1）
- `X_imputed` 必须预分配 n_cells * n_genes 个元素

**后置条件:**
- `X_imputed[i, j]` = 细胞 i 的邻居中基因 j 的加权平均
- 亲和矩阵的权重定义邻居贡献
- 密集输出适用于下游密集操作

**复杂度:**
- 时间: O(n_cells * avg_neighbors * n_genes)
- 空间: O(n_cells * n_genes) 用于输出

**线程安全:** 安全 - 按细胞并行化，每个写入独立内存

### magic_impute

MAGIC（基于马尔可夫亲和图的细胞插补）算法。

```cpp
Sparse<Real, true> transition_matrix = /* MAGIC 扩散算子 */;
Index t = 3;  // 扩散时间

scl::kernel::impute::magic_impute(
    X_sparse, transition_matrix, n_cells, n_genes, t, X_imputed
);
```

**参数:**
- `X_sparse` [in] - 输入稀疏表达矩阵
- `transition_matrix` [in] - 扩散算子（来自 MAGIC）
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `t` [in] - 扩散时间参数
- `X_imputed` [out] - 密集插补矩阵

**前置条件:**
- `transition_matrix` 来自 MAGIC 预处理（对称归一化）
- `t >= 1`，通常 t 在 [1, 5] 范围内
- `X_imputed` 必须预分配

**后置条件:**
- `X_imputed = (T^t) * X`
- 去噪和插补的表达值
- 保留整体表达结构

**复杂度:**
- 时间: O(t * n_cells * avg_nnz * n_genes)
- 空间: O(2 * n_cells * n_genes)

**线程安全:** 安全 - 所有操作并行化

**参考文献:**
- van Dijk et al., MAGIC, Cell 2018

### alra_impute

ALRA（自适应阈值低秩近似）插补。

```cpp
const Real* X_dense = /* 密集归一化表达 */;
Index n_components = 50;

scl::kernel::impute::alra_impute(
    X_dense, n_cells, n_genes, n_components, X_imputed, 5, 42
);
```

**参数:**
- `X_dense` [in] - 密集归一化表达 [n_cells x n_genes]
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `n_components` [in] - SVD 组件数量（秩）
- `X_imputed` [out] - 密集插补矩阵
- `n_iter` [in] - SVD 的幂迭代次数（默认: 5）
- `seed` [in] - 随机种子（默认: 42）

**前置条件:**
- `X_dense` 已对数归一化
- `n_components <= min(n_cells, n_genes)`
- `X_imputed` 必须预分配

**后置条件:**
- `X_imputed = U * S * V^T`（秩 k 近似）
- 负值设置为零（生物学约束）
- 在插补 < 原始时保留原始非零值

**复杂度:**
- 时间: O(n_iter * n_cells * n_genes * n_components)
- 空间: O(n_cells * n_components + n_genes * n_components)

**线程安全:** 安全 - 并行矩阵操作

**参考文献:**
- Linderman et al., ALRA, bioRxiv 2018

### diffusion_impute_sparse_transition

使用稀疏转移矩阵的基于扩散的插补。

```cpp
Sparse<Real, true> transition_matrix = /* 行随机转移矩阵 */;
Index n_steps = 3;

scl::kernel::impute::diffusion_impute_sparse_transition(
    X_sparse, transition_matrix, n_cells, n_genes, n_steps, X_imputed
);
```

**参数:**
- `X_sparse` [in] - 输入稀疏表达矩阵
- `transition_matrix` [in] - 行随机转移矩阵
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `n_steps` [in] - 扩散步数
- `X_imputed` [out] - 密集插补矩阵

**前置条件:**
- `transition_matrix` 必须是行随机的（行和为 1）
- `n_steps >= 1`
- `X_imputed` 必须预分配

**后置条件:**
- `X_imputed = T^n_steps * X`，其中 T 是转移矩阵
- 更高的 n_steps = 更多平滑/插补

**复杂度:**
- 时间: O(n_steps * n_cells * avg_nnz_per_row * n_genes)
- 空间: O(2 * n_cells * n_genes) 用于双缓冲

**线程安全:** 安全 - 使用并行 SpMM 双缓冲

## 辅助函数

### impute_selected_genes

仅插补基因子集以提高效率。

```cpp
const Index* gene_indices = /* 要插补的基因 [n_selected] */;
Real* X_imputed = /* 输出 [n_cells * n_selected] */;

scl::kernel::impute::impute_selected_genes(
    X_sparse, affinity, gene_indices, n_selected, n_cells, X_imputed
);
```

**参数:**
- `X_sparse` [in] - 输入稀疏表达矩阵
- `affinity` [in] - 细胞-细胞亲和矩阵
- `gene_indices` [in] - 要插补的基因索引
- `n_selected` [in] - 要插补的基因数量
- `n_cells` [in] - 细胞数量
- `X_imputed` [out] - 选中基因的插补值 [n_cells x n_selected]

**前置条件:**
- 所有基因索引在 [0, n_genes) 范围内
- `X_imputed` 预分配 n_cells * n_selected 个元素

**后置条件:**
- `X_imputed[i, j]` 包含细胞 i、选中基因 j 的插补值
- 仅计算指定基因的插补（内存高效）

**复杂度:**
- 时间: O(n_cells * avg_neighbors * n_selected)
- 空间: O(n_cells * n_selected)

**线程安全:** 安全 - 按细胞并行化

### smooth_expression

使用局部平均平滑表达谱。

```cpp
Real alpha = 0.5;  // 平滑因子
scl::kernel::impute::smooth_expression(
    X_sparse, affinity, n_cells, n_genes, alpha, X_smooth
);
```

**参数:**
- `X_sparse` [in] - 输入稀疏表达矩阵
- `affinity` [in] - 细胞-细胞亲和矩阵
- `n_cells` [in] - 细胞数量
- `n_genes` [in] - 基因数量
- `alpha` [in] - 平滑因子（0 = 原始，1 = 完全邻居平均）
- `X_smooth` [out] - 平滑的密集矩阵

**前置条件:**
- `alpha` 在 [0, 1] 范围内
- `affinity` 行归一化
- `X_smooth` 必须预分配

**后置条件:**
- `X_smooth[i] = (1 - alpha) * X[i] + alpha * neighbor_average[i]`
- 在原始和完全平滑之间插值

**复杂度:**
- 时间: O(n_cells * avg_neighbors * n_genes)
- 空间: O(n_cells * n_genes)

**线程安全:** 安全 - 按细胞并行化

## 配置

```cpp
namespace scl::kernel::impute::config {
    constexpr Real DISTANCE_EPSILON = Real(1e-10);
    constexpr Real DEFAULT_ALPHA = Real(1.0);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Index DEFAULT_N_COMPONENTS = 50;
    constexpr Size PARALLEL_THRESHOLD = 32;
    constexpr Size GENE_BLOCK_SIZE = 64;
    constexpr Size CELL_BLOCK_SIZE = 32;
}
```

## 使用场景

### KNN 插补

```cpp
// 1. 计算细胞-细胞亲和（例如，从邻居）
Sparse<Real, true> affinity = /* 从 KNN 图计算 */;

// 2. 归一化亲和矩阵（行和为 1）
scl::kernel::normalize::normalize_rows_inplace(affinity, NormMode::L1);

// 3. 插补表达
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::knn_impute_dense(
    X_sparse, affinity, n_cells, n_genes, X_imputed
);
```

### MAGIC 插补

```cpp
// 1. 构建 MAGIC 转移矩阵（来自扩散核）
Sparse<Real, true> transition = /* MAGIC 转移矩阵 */;

// 2. 应用 MAGIC 插补
Index t = 3;  // 扩散时间
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::magic_impute(
    X_sparse, transition, n_cells, n_genes, t, X_imputed
);
```

### ALRA 插补

```cpp
// 1. 归一化和对数变换
Sparse<Real, true> X_normalized = /* 归一化表达 */;
scl::kernel::log1p::log1p_inplace(X_normalized);

// 2. 转换为密集
Real* X_dense = /* 将稀疏转换为密集 */;

// 3. 应用 ALRA
Index n_components = 50;
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::alra_impute(
    X_dense, n_cells, n_genes, n_components, X_imputed
);
```

### 选择性基因插补

```cpp
// 仅插补高变基因
Array<Index> hvg_indices = /* 高变基因索引 */;
Real* X_hvg_imputed = new Real[n_cells * n_hvgs];

scl::kernel::impute::impute_selected_genes(
    X_sparse, affinity, hvg_indices.ptr, n_hvgs, n_cells, X_hvg_imputed
);
```

## 性能

- **并行化**: 随细胞数量线性扩展
- **内存高效**: 稀疏输入，可选密集输出
- **块处理**: 针对缓存局部性优化
- **SIMD 加速**: 向量化平均操作

---

::: tip 方法选择
- **KNN**: 快速，适用于小数据集
- **MAGIC**: 最适合保留生物学结构
- **ALRA**: 适用于大型数据集，基于秩的去噪
- **加权 KNN**: 当有距离信息时更好
:::

