# transition.hpp

> scl/kernel/transition.hpp · 细胞状态转换分析

## 概述

本文件提供使用马尔可夫链理论分析细胞状态转换的函数。包括转移矩阵构建、稳态分析、吸收概率和亚稳态识别。

**头文件**: `#include "scl/kernel/transition.hpp"`

主要特性：
- 从速度向量构建转移矩阵
- 稳态分布计算
- 到终态的吸收概率
- 亚稳态识别（PCCA+）
- 谱系驱动基因识别
- 前向提交概率

---

## 主要 API

### transition_matrix_from_velocity

::: source_code file="scl/kernel/transition.hpp" symbol="transition_matrix_from_velocity" collapsed
:::

**算法说明**

从速度向量和 kNN 图构建转移矩阵：

1. 对于每个细胞 i（并行）：
   - 对于 kNN 图中的每个邻居 j：
     - 计算方向向量：`direction = expression[j] - expression[i]`
     - 计算余弦相似度：`cos_sim = dot(velocity[i], direction) / (|velocity[i]| * |direction|)`
     - 如果 `trans_type == Forward`：直接使用 cos_sim
     - 如果 `trans_type == Backward`：使用 -cos_sim
     - 如果 `trans_type == Symmetric`：使用 |cos_sim|
   - 对邻居相似度应用 softmax 以获得转移概率
   - 将概率存储在转移矩阵的第 i 行
2. 矩阵是行随机的（每行和为 1.0）

**边界条件**

- **无邻居**：行和为 0（孤立细胞）
- **零速度**：所有转移具有相等概率
- **所有邻居相同**：均匀转移概率
- **空 kNN 图**：所有转移为零

**数据保证（前置条件）**

- 速度向量是行主序：`velocity[i * n_genes + j]` 是细胞 i 的基因 j
- kNN 图必须是有效的 CSR 格式
- `transition_out` 必须预分配正确的维度
- kNN 图应根据需要是对称的或有向的

**复杂度分析**

- **时间**：O(n_cells * k * n_genes)，其中 k 是平均邻居数
- **空间**：O(n_genes) 每线程用于临时向量

**示例**

```cpp
#include "scl/kernel/transition.hpp"

const Real* velocity = /* 速度向量 [n_cells * n_genes] */;
scl::Sparse<Real, true> knn = /* kNN 图 */;
scl::Sparse<Real, true> transition(n_cells, n_cells);

scl::kernel::transition::transition_matrix_from_velocity(
    velocity, knn, transition, n_cells, n_genes,
    scl::kernel::transition::TransitionType::Forward
);

// transition 现在是行随机转移矩阵
```

---

### stationary_distribution

::: source_code file="scl/kernel/transition.hpp" symbol="stationary_distribution" collapsed
:::

**算法说明**

使用幂迭代计算稳态分布：

1. 初始化：`pi = [1/n, 1/n, ..., 1/n]`（均匀分布）
2. 迭代直到收敛：
   - 计算 `pi_new = pi * T`（与转置的稀疏矩阵-向量乘积）
   - 每 10 次迭代应用 Aitken delta-squared 加速
   - 归一化：`pi = pi_new / sum(pi_new)`
   - 检查收敛：`||pi_new - pi|| < tol`
3. 返回稳态分布，其中 `pi * T = pi`

**边界条件**

- **断开连接的图**：可能有多个稳态分布
- **吸收态**：稳态分布集中在吸收态
- **周期链**：可能不收敛（由 max_iter 处理）
- **达到 max_iter**：返回当前分布（可能未收敛）

**数据保证（前置条件）**

- 矩阵必须是行随机的（行和为 1.0）
- 矩阵应该是不可约的（单个遍历类）以获得唯一解
- `pi` 必须预分配大小为 n

**复杂度分析**

- **时间**：O(max_iter * nnz) - 每次迭代的 SpMV
- **空间**：O(n) 辅助空间

**示例**

```cpp
scl::Array<Real> pi(n_cells);

scl::kernel::transition::stationary_distribution(
    transition, pi.data(), n_cells,
    1e-6,   // tol
    1000    // max_iter
);

// pi[i] 包含状态 i 的稳态概率
// pi * T = pi（稳态条件）
```

---

### absorption_probability

::: source_code file="scl/kernel/transition.hpp" symbol="absorption_probability" collapsed
:::

**算法说明**

计算到终态的吸收概率：

1. 分割转移矩阵：
   - Q = 瞬态到瞬态子矩阵
   - R = 瞬态到终态子矩阵
2. 使用 SOR 迭代求解线性系统：`(I - Q) * B = R`
   - B[i, j] = 从瞬态 i 被终态 j 吸收的概率
   - omega = 1.5 用于过松弛（更快收敛）
3. 返回吸收概率矩阵

**边界条件**

- **无终态**：所有概率为 0
- **所有状态都是终态**：单位矩阵（立即吸收）
- **不可达终态**：概率为 0
- **多条路径**：概率对所有路径求和

**数据保证（前置条件）**

- `terminal_mask` 标识终态（1 = 终态，0 = 瞬态）
- 矩阵必须是有效的转移矩阵
- `absorb_probs` 是行主序：`absorb_probs[i * n_terminal + j]` 是从状态 i 到终态 j 的概率

**复杂度分析**

- **时间**：O(max_iter * nnz * n_terminal) - SOR 迭代
- **空间**：O(n * n_terminal) 用于输出矩阵

**示例**

```cpp
uint8_t* terminal_mask = /* 终态掩码 */;
scl::Array<Real> absorb_probs(n_cells * n_terminal);

scl::kernel::transition::absorption_probability(
    transition, terminal_mask, absorb_probs.data(),
    n_cells, n_terminal,
    1e-6,   // tol
    1000    // max_iter
);

// absorb_probs[i * n_terminal + j] = 从状态 i 被终态 j
// 吸收的概率
```

---

### metastable_states

::: source_code file="scl/kernel/transition.hpp" symbol="metastable_states" collapsed
:::

**算法说明**

使用谱聚类（PCCA+）识别亚稳态：

1. 在特征向量空间上的 K-means++ 初始化：
   - 基于特征向量距离选择初始质心
2. 并行 k-means 分配和更新：
   - 将每个状态分配到特征向量空间中最近的质心
   - 将质心更新为分配状态的均值
   - 迭代直到收敛
3. 计算软成员资格：
   - Membership[i, j] = 到亚稳态 j 的基于距离的权重
   - 归一化为每个状态和为 1.0

**边界条件**

- **k = 1**：所有状态分配到单个亚稳态
- **k >= n**：每个状态是自己的亚稳态
- **相同的特征向量**：可能有退化聚类

**数据保证（前置条件）**

- `eigenvectors` 是行主序：`eigenvectors[i * n + j]` 是特征向量 i 的分量 j
- `assignments` 和 `membership` 必须预分配
- k 应该 <= n（状态数）

**复杂度分析**

- **时间**：O(n * k^2 * n_iter) - k-means 迭代
- **空间**：O(k * k) 用于质心

**示例**

```cpp
const Real* eigenvectors = /* 前 k 个特征向量 [k * n] */;
scl::Array<Index> assignments(n_cells);
scl::Array<Real> membership(n_cells * k);

scl::kernel::transition::metastable_states(
    eigenvectors, n_cells, k,
    assignments.data(), membership.data(),
    42  // 种子
);

// assignments[i] = 到亚稳态的硬分配
// membership[i * k + j] = 软成员资格概率
```

---

### lineage_drivers

::: source_code file="scl/kernel/transition.hpp" symbol="lineage_drivers" collapsed
:::

**算法说明**

识别驱动谱系转换的基因：

1. 对于每个基因 g 和终态 t（并行）：
   - 提取基因表达：`expr = expression[:, g]`
   - 提取吸收概率：`probs = absorb_probs[:, t]`
   - 计算 Pearson 相关性：`corr = corr(expr, probs)`
   - 存储为驱动分数：`driver_scores[g * n_terminal + t] = corr`
2. 更高的分数表示与承诺到终态相关的基因
3. 使用 SIMD 优化的点积进行相关性计算

**边界条件**

- **恒定表达**：相关性为 0（未定义）
- **恒定概率**：相关性为 0（未定义）
- **完美相关**：分数接近 1.0

**数据保证（前置条件）**

- 表达矩阵必须是有效的 CSR 格式
- `absorb_probs` 是行主序：`absorb_probs[i * n_terminal + t]`
- `driver_scores` 是行主序：`driver_scores[g * n_terminal + t]`

**复杂度分析**

- **时间**：O(n_genes * n_terminal * n_cells) - 每个基因-终态对的相关性
- **空间**：O(n_cells) 每线程

**示例**

```cpp
const Real* absorb_probs = /* 吸收概率 */;
scl::Array<Real> driver_scores(n_genes * n_terminal);

scl::kernel::transition::lineage_drivers(
    expression, absorb_probs, driver_scores.data(),
    n_cells, n_genes, n_terminal
);

// driver_scores[g * n_terminal + t] = 基因 g 表达与
// 承诺到终态 t 之间的相关性
```

---

### forward_committor

::: source_code file="scl/kernel/transition.hpp" symbol="forward_committor" collapsed
:::

**算法说明**

计算源和目标之间的前向提交概率：

1. 求解线性系统：`(I - Q) * q = r`，其中：
   - Q = 中间状态之间的转移概率
   - r = 从中间到目标的转移概率
   - q = 提交概率
2. 边界条件：
   - 如果 i 是源状态，则 `committor[i] = 0`
   - 如果 i 是目标状态，则 `committor[i] = 1`
3. 使用过松弛的迭代求解器（SOR）

**边界条件**

- **源 = 目标**：所有提交概率为 0（平凡）
- **从源到目标无路径**：提交概率为 0
- **所有状态都是目标**：所有提交概率为 1

**数据保证（前置条件）**

- `source_mask` 和 `target_mask` 标识源和目标状态
- 矩阵必须是有效的转移矩阵
- `committor` 必须预分配大小为 n

**复杂度分析**

- **时间**：O(max_iter * nnz) - 迭代求解器
- **空间**：O(n) 辅助空间

**示例**

```cpp
uint8_t* source_mask = /* 源状态 */;
uint8_t* target_mask = /* 目标状态 */;
scl::Array<Real> committor(n_cells);

scl::kernel::transition::forward_committor(
    transition, source_mask, target_mask, committor.data(),
    n_cells, 1e-6, 1000
);

// committor[i] = 从状态 i 在源之前命中目标的概率
```

---

## 工具函数

### sparse_matvec

稀疏矩阵-向量乘积：y = A * x

::: source_code file="scl/kernel/transition.hpp" symbol="sparse_matvec" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

### sparse_matvec_transpose

带转置的稀疏矩阵-向量乘积：y = A^T * x

::: source_code file="scl/kernel/transition.hpp" symbol="sparse_matvec_transpose" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

### is_stochastic

检查矩阵是否是行随机的（行和为 1）。

::: source_code file="scl/kernel/transition.hpp" symbol="is_stochastic" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

### row_normalize_to_stochastic

将矩阵行归一化为和为 1（使行随机）。

::: source_code file="scl/kernel/transition.hpp" symbol="row_normalize_to_stochastic" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

### symmetrize_transition

对称化转移矩阵：T_sym = 0.5 * (T + T^T)

::: source_code file="scl/kernel/transition.hpp" symbol="symmetrize_transition" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(nnz) 用于输出

---

### identify_terminal_states

识别马尔可夫链中的终态（吸收态）。

::: source_code file="scl/kernel/transition.hpp" symbol="identify_terminal_states" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

### hitting_time

计算到目标状态的预期命中时间。

::: source_code file="scl/kernel/transition.hpp" symbol="hitting_time" collapsed
:::

**复杂度**

- 时间：O(max_iter * nnz)
- 空间：O(n)

---

### time_to_absorption

计算瞬态状态的预期吸收时间。

::: source_code file="scl/kernel/transition.hpp" symbol="time_to_absorption" collapsed
:::

**复杂度**

- 时间：O(max_iter * nnz)
- 空间：O(n)

---

### compute_top_eigenvectors

使用带收缩的幂迭代计算前 k 个特征向量。

::: source_code file="scl/kernel/transition.hpp" symbol="compute_top_eigenvectors" collapsed
:::

**复杂度**

- 时间：O(k * max_iter * nnz)
- 空间：O(k * n)

---

### coarse_grain_transition

计算亚稳态之间的粗粒度转移矩阵。

::: source_code file="scl/kernel/transition.hpp" symbol="coarse_grain_transition" collapsed
:::

**复杂度**

- 时间：O(nnz * k^2)
- 空间：O(k^2)

---

### directional_score

计算每个细胞的方向偏差分数。

::: source_code file="scl/kernel/transition.hpp" symbol="directional_score" collapsed
:::

**复杂度**

- 时间：O(nnz)
- 空间：O(1)

---

## 注意事项

- 转移矩阵应该是行随机的，以便进行适当的马尔可夫链分析
- 稳态分布需要不可约链（单个遍历类）
- 吸收概率对谱系承诺分析很有用
- 亚稳态识别长期存在的中间状态
- 谱系驱动基因有助于识别关键调控基因

## 相关内容

- [速度模块](./velocity) - 用于速度向量计算
- [扩散模块](./diffusion) - 用于基于扩散的分析
- [伪时间模块](./pseudotime) - 用于伪时间计算

