# impute.hpp

> scl/kernel/impute.hpp · 单细胞表达数据的高性能插补内核

## 概述

本文件为稀疏单细胞 RNA-seq 数据提供高效的插补方法。插补填充缺失值（dropout）以恢复真实表达信号并提高下游分析质量。

本文件提供：
- K 最近邻（KNN）插补
- 基于扩散的插补（MAGIC 风格）
- ALRA（自适应阈值低秩近似）
- 距离加权 KNN 插补
- Dropout 检测和质量评估

**头文件**: `#include "scl/kernel/impute.hpp"`

---

## 主要 API

### knn_impute_dense

::: source_code file="scl/kernel/impute.hpp" symbol="knn_impute_dense" collapsed
:::

**算法说明**

使用 K 最近邻平均在密集输出上插补缺失值：

1. **亲和力加权平均**：对每个细胞并行计算加权和
2. **边缘情况处理**：如果行和 < epsilon，复制原始行
3. **输出格式**：密集矩阵，适用于下游密集操作

**边界条件**

- 无邻居：亲和力行和为零的细胞复制原始值
- 空亲和力矩阵：返回转换为密集的原始稀疏矩阵

**数据保证（前置条件）**

- `X_sparse` 必须是 CSR 格式（细胞 x 基因）
- `affinity` 必须行归一化（行和为 1）
- `X_imputed` 必须预分配

**复杂度分析**

- **时间**：O(n_cells * avg_neighbors * n_genes)
- **空间**：O(n_cells * n_genes) 用于密集输出

**示例**

```cpp
#include "scl/kernel/impute.hpp"

Sparse<Real, true> X_sparse = /* ... */;
Sparse<Real, true> affinity = /* ... */;
Array<Real> X_imputed(n_cells * n_genes);

scl::kernel::impute::knn_impute_dense(
    X_sparse, affinity, n_cells, n_genes, X_imputed.data()
);
```

---

### magic_impute

::: source_code file="scl/kernel/impute.hpp" symbol="magic_impute" collapsed
:::

**算法说明**

MAGIC（基于马尔可夫亲和图的细胞插补）算法用于基于扩散的插补：

1. **扩散过程**：应用 t 步扩散
2. **双缓冲**：使用两个缓冲区避免数据竞争
3. **收敛**：更高的 t = 更多平滑/插补

**边界条件**

- t=0：返回原始矩阵（无扩散）
- 孤立细胞：无邻居的细胞保持不变

**数据保证（前置条件）**

- `transition_matrix` 来自 MAGIC 预处理
- `t >= 1`，通常 t 在 [1, 5]

**复杂度分析**

- **时间**：O(t * n_cells * avg_nnz * n_genes)
- **空间**：O(2 * n_cells * n_genes) 用于双缓冲

---

### alra_impute

::: source_code file="scl/kernel/impute.hpp" symbol="alra_impute" collapsed
:::

**算法说明**

ALRA（自适应阈值低秩近似）插补使用随机 SVD：

1. **随机 SVD**：计算 rank-k 近似
2. **阈值化**：将负值设为零
3. **保留原始值**：在插补值 < 原始非零值处保留原始值

**边界条件**

- k > min(n_cells, n_genes)：限制为最小维度
- 零方差基因：从 SVD 中排除

**数据保证（前置条件）**

- `X_dense` 已对数归一化
- `n_components <= min(n_cells, n_genes)`

**复杂度分析**

- **时间**：O(n_iter * n_cells * n_genes * n_components)
- **空间**：O(n_cells * n_components + n_genes * n_components)

---

### diffusion_impute_sparse_transition

::: source_code file="scl/kernel/impute.hpp" symbol="diffusion_impute_sparse_transition" collapsed
:::

**算法说明**

使用稀疏转移矩阵的基于扩散的插补。

**复杂度分析**

- **时间**：O(n_steps * n_cells * avg_nnz_per_row * n_genes)
- **空间**：O(2 * n_cells * n_genes)

---

### knn_impute_weighted_dense

::: source_code file="scl/kernel/impute.hpp" symbol="knn_impute_weighted_dense" collapsed
:::

**算法说明**

使用距离加权 KNN 贡献进行插补。

**复杂度分析**

- **时间**：O(n_cells * k * n_genes)
- **空间**：O(n_cells * n_genes)

---

## 工具函数

### impute_selected_genes

仅插补基因子集以提高效率。

::: source_code file="scl/kernel/impute.hpp" symbol="impute_selected_genes" collapsed
:::

**复杂度**

- 时间：O(n_cells * avg_neighbors * n_selected)
- 空间：O(n_cells * n_selected)

---

### detect_dropouts

检测可能的 dropout 事件（技术零 vs 生物零）。

::: source_code file="scl/kernel/impute.hpp" symbol="detect_dropouts" collapsed
:::

**复杂度**

- 时间：O(n_cells * n_genes)
- 空间：O(n_genes)

---

### imputation_quality

计算插补质量指标（与保留数据的相关性）。

::: source_code file="scl/kernel/impute.hpp" symbol="imputation_quality" collapsed
:::

**复杂度**

- 时间：O(n_cells * n_genes)
- 空间：O(n_threads)

---

### smooth_expression

使用局部平均平滑表达谱。

::: source_code file="scl/kernel/impute.hpp" symbol="smooth_expression" collapsed
:::

**复杂度**

- 时间：O(n_cells * avg_neighbors * n_genes)
- 空间：O(n_cells * n_genes)

---

## 注意事项

**方法选择**：
- **KNN**：快速、简单，适用于大多数情况
- **MAGIC**：强平滑，保留结构，可能过度插补
- **ALRA**：低秩近似，适用于去噪，保留全局结构
- **加权 KNN**：考虑距离，比均匀 KNN 更准确

**性能**：
- 所有方法按细胞并行化
- 尽可能保留稀疏输入
- 密集输出用于下游分析

## 相关内容

- [Neighbors](/zh/cpp/kernels/neighbors) - 用于亲和力矩阵的 KNN 计算
- [Normalization](/zh/cpp/kernels/normalize) - 插补前的表达归一化
