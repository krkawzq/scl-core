# mmd.hpp

> scl/kernel/mmd.hpp · 使用 RBF 核的最大均值差异计算

## 概述

本文件提供使用 RBF（径向基函数）核高效计算两个分布之间的最大均值差异（MMD）。MMD 是通过将概率分布嵌入到再生核希尔伯特空间（RKHS）中来比较两个概率分布的度量。

主要特性：
- 稀疏矩阵的特征级 MMD 计算
- 使用 SIMD 优化的 RBF 核计算
- 块平铺以提高缓存效率
- 跨特征的线程安全并行化

**头文件**: `#include "scl/kernel/mmd.hpp"`

---

## 主要 API

### mmd_rbf

::: source_code file="scl/kernel/mmd.hpp" symbol="mmd_rbf" collapsed
:::

**算法说明**

使用 RBF 核计算两个分布 X 和 Y 之间的 MMD^2：

```
MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
```

对每个特征并行执行：
1. 从两个分布中提取该特征的非零值
2. 计算单目 exp 和：对每个分布计算 `sum(exp(-gamma * x^2))` 并缓存
3. 计算自核和：使用对称性计算 `E[k(X,X)]` 和 `E[k(Y,Y)]`
4. 计算交叉核和：使用块平铺（64x512 块）计算 `E[k(X,Y)]`
5. 组合结果：`(K_xx/n_x^2) + (K_yy/n_y^2) - 2*(K_xy/(n_x*n_y))`

自核计算利用对称性：
- 零-零对：`(N-nnz)^2`（核值 = 1）
- 零-非零对：`2 * (N-nnz) * sum_unary`
- 对角线：`nnz`（核值 = 1）
- 非对角线：`2 * sum_{i<j} exp(-gamma * (x_i - x_j)^2)` 使用 2 路 SIMD 展开

交叉核使用块平铺以提高缓存效率，在每个块内使用 2 路 SIMD 展开计算非零-非零对。

**边界条件**

- **全零特征**：返回 MMD^2 = 0（相同的零分布）
- **空分布**：优雅处理，核和为零
- **每个分布只有一个样本**：对 n=1 正确计算自核
- **数值误差导致的负值**：钳制为 0（MMD^2 必须非负）

**数据保证（前置条件）**

- `mat_x.primary_dim() == mat_y.primary_dim()`（特征数相同）
- `output.len == mat_x.primary_dim()`（输出大小匹配特征数）
- `gamma > 0`（核参数必须为正）
- 矩阵必须是有效的 CSR/CSC 格式

**复杂度分析**

- **时间**：每个特征 O(features * (nnz_x^2 + nnz_y^2 + nnz_x*nnz_y))
- **空间**：每个线程 O(max(nnz_x, nnz_y)) 用于缓存 exp 项

**示例**

```cpp
#include "scl/kernel/mmd.hpp"
#include "scl/core/sparse.hpp"

// 为两个分布创建稀疏矩阵
// 每列是一个样本，每行是一个特征
Sparse<Real, true> mat_x(n_features, n_samples_x);
Sparse<Real, true> mat_y(n_features, n_samples_y);

// 用数据填充矩阵...

// 预分配输出
Array<Real> mmd_values(n_features);

// 使用默认 gamma = 1.0 计算 MMD^2
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, mmd_values);

// 或使用自定义 gamma
Real gamma = 0.5;  // 较小的 gamma = 更宽的核
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, mmd_values, gamma);

// mmd_values[i] 现在包含特征 i 的分布之间的 MMD^2
// 较高的值表示较大的分布差异。
```

---

## 工具函数

### unary_exp_sum

计算非零值的 exp(-gamma * x^2) 之和，缓存各个 exp 项以便在自核和交叉核计算中重用。

::: source_code file="scl/kernel/mmd.hpp" symbol="unary_exp_sum" collapsed
:::

**复杂度**

- 时间：O(nnz)，使用 8 路 SIMD 展开
- 空间：O(nnz) 用于缓存数组

---

### self_kernel_sum

计算单个分布内所有对的 RBF 核之和，包括隐式零。利用对称性提高效率。

::: source_code file="scl/kernel/mmd.hpp" symbol="self_kernel_sum" collapsed
:::

**复杂度**

- 时间：O(nnz^2)，使用 2 路 SIMD 展开
- 空间：O(1) 辅助空间

---

### cross_kernel_sum

计算来自两个分布的所有对之间的 RBF 核之和。使用块平铺（BLOCK_X=64, BLOCK_Y=512）以提高缓存效率。

::: source_code file="scl/kernel/mmd.hpp" symbol="cross_kernel_sum" collapsed
:::

**复杂度**

- 时间：O(nnz_x * nnz_y)，使用 2 路 SIMD 展开
- 空间：O(1) 辅助空间

---

## 数值注意事项

- **RBF 核**：k(x,y) = exp(-gamma * (x-y)^2)
- **MMD^2 性质**：非负度量（可能因数值误差而略为负，钳制为 0）
- **全零特征**：MMD^2 = 0（相同分布）
- **Gamma 参数**：控制核宽度；较大的 gamma = 更窄的核，对小差异更敏感
- **归一化**：每项按对数归一化（自核为 n^2，交叉核为 n_x*n_y）

## 相关内容

- [Neighbors](/zh/cpp/kernels/neighbors) - KNN 算法
- [Statistics](/zh/cpp/kernels/statistics) - 统计测试和度量
