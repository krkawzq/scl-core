# diffusion.hpp

> scl/kernel/diffusion.hpp · 稀疏图上的高性能扩散过程

## 概述

本文件提供用于轨迹分析、伪时间计算和稀疏图上信号传播的高效扩散操作。所有操作使用并行化的稀疏矩阵-向量乘法（SpMV）和稀疏矩阵-矩阵乘法（SpMM）以获得最佳性能。

**头文件**: `#include "scl/kernel/diffusion.hpp"`

主要特性：
- 通过转移矩阵进行向量和矩阵扩散
- 扩散距离计算
- 从根细胞计算伪时间
- 带重启的随机游走（RWR）评分
- 扩散映射嵌入

---

## 主要 API

### diffuse_vector

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffuse_vector" collapsed
:::

**算法说明**

对密集向量应用多步扩散算子：

1. 对于从 1 到 n_steps 的每一步：
   - 计算稀疏矩阵-向量乘积：`x_new = transition * x`
   - 用 x_new 更新 x
2. 每一步使用并行化的 SpMV
3. 每一步将信号通过转移图传播一跳

**边界条件**

- **空向量**：立即返回，无变化
- **零转移矩阵**：向量保持不变
- **n_steps = 0**：立即返回，向量不变

**数据保证（前置条件）**

- `x.len >= transition.primary_dim()`
- 转移矩阵必须是行随机矩阵（每行和为 1.0）
- 转移矩阵必须是有效的 CSR 格式
- 索引必须在行内排序

**复杂度分析**

- **时间**：O(n_steps * nnz)，其中 nnz 是转移矩阵中的非零元素数
- **空间**：O(n_nodes) 辅助空间用于临时向量

**示例**

```cpp
#include "scl/kernel/diffusion.hpp"

// 创建转移矩阵（行随机）
scl::Sparse<Real, true> transition = /* ... */;

// 初始化信号向量
scl::Array<Real> x(n_nodes);
// ... 初始化 x ...

// 应用 3 步扩散
scl::kernel::diffusion::diffuse_vector(transition, x, 3);

// x 现在包含 3 步后的扩散信号
```

---

### diffuse_matrix

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffuse_matrix" collapsed
:::

**算法说明**

对密集矩阵应用扩散算子（同时处理多个特征）：

1. 对于从 1 到 n_steps 的每一步：
   - 对于每个特征列：
     - 计算与特征列的稀疏矩阵-向量乘积
     - 原地更新特征列
2. 使用分块稀疏矩阵-矩阵乘法（SpMM）以提高效率
3. 所有特征并行扩散

**边界条件**

- **空矩阵**：立即返回
- **n_features = 0**：立即返回
- **n_steps = 0**：矩阵不变

**数据保证（前置条件）**

- `X.len >= n_nodes * n_features`
- 矩阵 X 是行主序布局：`X[i * n_features + j]` 是节点 i 的特征 j
- 转移矩阵必须是有效的 CSR 格式

**复杂度分析**

- **时间**：O(n_steps * nnz * n_features)
- **空间**：O(n_nodes * n_features) 辅助空间

**示例**

```cpp
// 特征矩阵：n_nodes 行，n_features 列
scl::Array<Real> X(n_nodes * n_features);
// ... 初始化 X ...

// 对所有特征进行 3 步扩散
scl::kernel::diffusion::diffuse_matrix(
    transition, X, n_nodes, n_features, 3
);

// X 中的所有特征现在都已扩散
```

---

### diffusion_distance

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_distance" collapsed
:::

**算法说明**

计算所有节点对之间的扩散距离矩阵：

1. 对于每个源节点 i：
   - 在节点 i 处初始化单位向量
   - 应用 n_steps 步扩散
   - 将结果分布存储在距离矩阵的第 i 行
2. 节点 i 和 j 之间的距离从扩散分布计算
3. 对源节点进行并行化

**边界条件**

- **n_nodes = 0**：返回空距离矩阵
- **n_steps = 0**：返回类似恒等的距离
- **孤立节点**：距离保持较高

**数据保证（前置条件）**

- `distances` 容量 >= n_nodes * n_nodes
- 输出矩阵是行主序：`distances[i * n_nodes + j]` 是从 i 到 j 的距离
- 转移矩阵必须是有效的 CSR 格式

**复杂度分析**

- **时间**：O(n_nodes^2 * n_steps * nnz)
- **空间**：O(n_nodes^2) 辅助空间

**示例**

```cpp
scl::Array<Real> distances(n_nodes * n_nodes);

scl::kernel::diffusion::diffusion_distance(
    transition, distances, 3
);

// distances[i * n_nodes + j] 包含从节点 i 到节点 j
// 经过 3 步后的扩散距离
```

---

### diffusion_pseudotime

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_pseudotime" collapsed
:::

**算法说明**

从根细胞计算扩散伪时间：

1. 将所有节点的伪时间初始化为无穷大
2. 对于每个根细胞：
   - 在根处初始化单位质量
   - 迭代扩散直到收敛：
     - 应用转移矩阵
     - 将伪时间更新为从任何根的最小距离
   - 继续直到收敛或达到 max_iter
3. 伪时间表示从最近根的最小扩散距离

**边界条件**

- **无根细胞（n_roots = 0）**：所有伪时间值保持无穷大
- **孤立节点**：伪时间保持无穷大
- **达到 max_iter**：返回当前伪时间值（可能未收敛）

**数据保证（前置条件）**

- `pseudotime.len >= transition.primary_dim()`
- `root_cells` 包含 [0, n_nodes) 范围内的有效节点索引
- 所有根细胞索引互不相同
- 转移矩阵必须是有效的 CSR 格式

**复杂度分析**

- **时间**：O(n_roots * max_iter * nnz)
- **空间**：O(n_nodes * n_roots) 辅助空间

**示例**

```cpp
// 定义根细胞（例如，早期发育阶段）
scl::Array<const Index> root_cells = {0, 5, 10};
Index n_roots = 3;

scl::Array<Real> pseudotime(n_nodes);

scl::kernel::diffusion::diffusion_pseudotime(
    transition, root_cells, n_roots, pseudotime, 100
);

// pseudotime[i] 包含从最近根细胞的伪时间
```

---

### random_walk_with_restart

::: source_code file="scl/kernel/diffusion.hpp" symbol="random_walk_with_restart" collapsed
:::

**算法说明**

计算带重启的随机游走（RWR）分数：

1. 初始化分数：在种子节点上的均匀分布
2. 迭代直到收敛：
   - 以概率 alpha：在种子节点重启
   - 以概率 (1-alpha)：通过转移矩阵走一步
   - 更新分数：`scores = alpha * seed_distribution + (1-alpha) * transition * scores`
3. 继续直到变化 < tol 或达到 max_iter
4. 分数表示在每个节点处的稳态概率

**边界条件**

- **无种子节点（n_seeds = 0）**：分数保持为零
- **alpha = 1.0**：分数保持在初始种子分布
- **alpha = 0.0**：纯随机游走（可能不收敛）
- **孤立的种子节点**：仅在种子处有高分

**数据保证（前置条件）**

- `scores.len >= transition.primary_dim()`
- `seed_nodes` 包含有效的节点索引
- `alpha` 在 (0, 1] 范围内以确保正确收敛
- 转移矩阵必须是行随机矩阵

**复杂度分析**

- **时间**：O(max_iter * nnz)
- **空间**：O(n_nodes) 辅助空间

**示例**

```cpp
// 种子节点（例如，标记细胞）
scl::Array<const Index> seed_nodes = {100, 200, 300};
Index n_seeds = 3;

scl::Array<Real> scores(n_nodes);

scl::kernel::diffusion::random_walk_with_restart(
    transition, seed_nodes, scores, 0.85, 100, 1e-6
);

// scores[i] 包含节点 i 的 RWR 分数（概率）
// 更高的分数表示更接近种子节点
```

---

### diffusion_map

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_map" collapsed
:::

**算法说明**

使用特征分解计算扩散映射嵌入：

1. 对于从 1 到 n_components 的每个分量：
   - 使用幂方法找到主特征向量
   - 相对于先前分量进行正交化
   - 存储为嵌入列
2. 嵌入捕获图的扩散几何
3. 使用带重正交化的迭代幂方法

**边界条件**

- **n_components = 0**：返回空嵌入
- **对称转移矩阵**：保证实特征值
- **非对称矩阵**：可能有复特征值（已处理）

**数据保证（前置条件）**

- `embedding` 容量 >= n_nodes * n_components
- 嵌入是行主序：`embedding[i * n_components + j]` 是节点 i 的分量 j
- 转移矩阵必须是有效的 CSR 格式
- n_components <= n_nodes

**复杂度分析**

- **时间**：O(n_components * max_iter * nnz)
- **空间**：O(n_nodes * n_components) 辅助空间

**示例**

```cpp
Index n_components = 10;
scl::Array<Real> embedding(n_nodes * n_components);

scl::kernel::diffusion::diffusion_map(
    transition, embedding, n_nodes, n_components, 100
);

// embedding[i * n_components + j] 包含节点 i 的第 j 个
// 扩散映射坐标
```

---

## 配置

默认参数在 `scl::kernel::diffusion::config` 中定义：

- `DEFAULT_N_STEPS = 3`：默认扩散步数
- `DEFAULT_ALPHA = 0.85`：RWR 的默认重启概率
- `CONVERGENCE_TOL = 1e-6`：收敛容差
- `MAX_ITER = 100`：迭代方法的最大迭代次数

---

## 注意事项

- 转移矩阵应该是行随机矩阵（归一化行）以确保正确的扩散行为
- CSR 格式是 SpMV 最佳性能所必需的
- 所有操作都是线程安全的并自动并行化
- 矩阵扩散使用分块 SpMM 以提高缓存效率

## 相关内容

- [稀疏矩阵操作](../core/sparse)
- [邻居模块](./neighbors) - 用于构建转移矩阵
