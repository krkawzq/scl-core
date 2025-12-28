# 传播

用于半监督学习和基于图的分类的标签传播内核。

## 概述

传播模块提供：

- **标签传播** - 硬标签多数投票
- **标签扩散** - 带正则化的软概率标签
- **归纳迁移** - 从参考到查询的标签迁移
- **置信度传播** - 置信度加权标签传播
- **调和函数** - 半监督回归
- **工具函数** - 标签转换和初始化

## 标签传播

### label_propagation

使用硬标签多数投票进行半监督分类的标签传播：

```cpp
#include "scl/kernel/propagation.hpp"

Sparse<Real, true> adjacency = /* ... */;  // 图邻接矩阵
Array<Index> labels(n_nodes);
// 初始化：labels[i] = class_id 用于标记，-1 用于未标记

scl::kernel::propagation::label_propagation(
    adjacency,
    labels,
    config::DEFAULT_MAX_ITER,  // max_iter = 100
    42                         // seed
);
```

**参数：**
- `adjacency`: 图邻接矩阵（权重作为边相似性）
- `labels`: 节点标签（UNLABELED=-1 用于未标记节点），原地修改
- `max_iter`: 最大迭代次数
- `seed`: 用于节点排序的随机种子

**后置条件：**
- 未标记节点分配给多数邻居类
- 当迭代中没有标签改变时收敛
- 原始标记节点的标签保持不变

**算法：**
对于每次迭代：
1. 使用 Fisher-Yates 打乱节点顺序
2. 对于打乱顺序中的每个节点：
   - 计算来自邻居的加权投票
   - 分配多数类标签
3. 如果没有标签改变则停止

**复杂度：**
- 时间：O(max_iter * edges) 期望
- 空间：O(n + n_classes) 辅助空间

**使用场景：**
- 半监督分类
- 基于图的学习
- 当只有少数标签可用时

## 标签扩散

### label_spreading

使用软概率标签进行正则化标签扩散：

```cpp
Array<Real> label_probs(n_nodes * n_classes);  // 软概率
const bool* is_labeled = /* ... */;  // 标记节点掩码

scl::kernel::propagation::label_spreading(
    adjacency,
    label_probs,
    n_classes,
    is_labeled,
    config::DEFAULT_ALPHA,      // alpha = 0.99
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE  // tol = 1e-6
);
```

**参数：**
- `adjacency`: 图邻接矩阵
- `label_probs`: 软标签概率 [n_nodes * n_classes]，原地修改
- `n_classes`: 不同类的数量
- `is_labeled`: 标记节点的布尔掩码
- `alpha`: 传播参数（0 到 1）
- `max_iter`: 最大迭代次数
- `tol`: 收敛容差（L1 范数）

**后置条件：**
- 软标签收敛或达到 max_iter
- label_probs 的每一行总和为 1（归一化）
- 标记节点保留 (1-alpha) 部分的初始标签

**算法：**
使用归一化图拉普拉斯 S = D^(-1/2) * W * D^(-1/2)：
1. 计算行和和 D^(-1/2)
2. 迭代：Y_new = alpha * S * Y + (1-alpha) * Y0
3. 将每一行归一化为总和为 1
4. 检查 L1 收敛

**复杂度：**
- 时间：O(max_iter * edges * n_classes)
- 空间：O(n * n_classes) 辅助空间

**使用场景：**
- 软分类
- 概率估计
- 当需要置信度分数时

## 归纳迁移

### inductive_transfer

使用加权 k-NN 投票从参考数据集迁移标签到查询数据集：

```cpp
Sparse<Real, true> ref_to_query = /* ... */;  // 相似性矩阵
Array<const Index> reference_labels = /* ... */;
Array<Index> query_labels(n_query);

scl::kernel::propagation::inductive_transfer(
    ref_to_query,
    reference_labels,
    query_labels,
    n_classes,
    Real(0.5)  // confidence_threshold
);
```

**参数：**
- `ref_to_query`: 相似性矩阵（行=查询，列=参考）
- `reference_labels`: 参考节点标签
- `query_labels`: 查询节点的预测标签
- `n_classes`: 不同类的数量
- `confidence_threshold`: 分配标签的最小置信度

**后置条件：**
- `query_labels[i]` = 预测类或 UNLABELED（如果置信度 < 阈值）
- 置信度 = best_votes / total_votes

**算法：**
在查询节点上并行：
1. 从参考邻居累积加权投票
2. 找到具有最大投票的类
3. 如果置信度 >= 阈值则分配，否则 UNLABELED

**复杂度：**
- 时间：O(nnz_ref_to_query)
- 空间：O(n_classes) 每线程

**使用场景：**
- 迁移学习
- 基于参考的注释
- 跨数据集标记

## 置信度传播

### confidence_propagation

使用调节投票权重的置信度分数进行标签传播：

```cpp
Array<Index> labels(n_nodes);
Array<Real> confidence(n_nodes);  // 置信度分数 [0, 1]

scl::kernel::propagation::confidence_propagation(
    adjacency,
    labels,
    confidence,
    n_classes,
    config::DEFAULT_ALPHA,  // 自投票权重乘数
    config::DEFAULT_MAX_ITER
);
```

**参数：**
- `adjacency`: 图邻接矩阵
- `labels`: 节点标签，原地修改
- `confidence`: 节点置信度分数 [0, 1]，原地修改
- `n_classes`: 类的数量
- `alpha`: 自投票权重乘数

**后置条件：**
- 使用置信度加权投票传播标签
- 置信度更新以反映投票确定性
- 当没有标签改变时收敛

**算法：**
对于每次迭代：
1. 对于每个节点：累积置信度加权邻居投票
2. 添加权重为 alpha * own_confidence 的自投票
3. 分配多数类
4. 更新置信度 = best_votes / total_votes

**复杂度：**
- 时间：O(max_iter * edges)
- 空间：O(n + n_classes) 辅助空间

**使用场景：**
- 置信度感知传播
- 质量控制
- 不确定性量化

## 调和函数

### harmonic_function

求解半监督回归的调和函数：

```cpp
Array<Real> values(n_nodes);
const bool* is_known = /* ... */;  // 已知值掩码

scl::kernel::propagation::harmonic_function(
    adjacency,
    values,
    is_known,
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**参数：**
- `adjacency`: 图邻接矩阵
- `values`: 节点值（已知值固定，未知值插值），原地修改
- `is_known`: 已知值节点的布尔掩码
- `max_iter`: 最大迭代次数
- `tol`: 收敛容差（最大绝对变化）

**后置条件：**
- 未知值收敛到调和解
- 已知值不变
- 未知值[i] = weighted_avg(neighbors[i])

**算法：**
Gauss-Seidel / Jacobi 风格迭代：
1. 对于每个未知节点：value = sum(w_ij * value_j) / sum(w_ij)
2. 跟踪最大变化
3. 当 max_change < tol 时停止

**复杂度：**
- 时间：O(max_iter * edges)
- 空间：O(n) 辅助空间

**使用场景：**
- 半监督回归
- 值插值
- 缺失数据插补

## 工具函数

### get_hard_labels

通过 argmax 将软概率标签转换为硬类分配：

```cpp
Array<const Real> probs = /* ... */;  // [n_nodes * n_classes]
Array<Index> labels(n_nodes);
Array<Real> max_probs(n_nodes);  // 可选

scl::kernel::propagation::get_hard_labels(
    probs,
    n_nodes,
    n_classes,
    labels,
    max_probs  // 可选
);
```

**参数：**
- `probs`: 软标签概率 [n_nodes * n_classes]
- `n_nodes`: 节点数量
- `n_classes`: 类的数量
- `labels`: 硬标签分配
- `max_probs`: 每个节点的可选最大概率

**后置条件：**
- `labels[i]` = argmax_c(probs[i * n_classes + c])
- `max_probs[i]` = max_c(probs[i * n_classes + c])（如果提供）

**复杂度：**
- 时间：O(n_nodes * n_classes)
- 空间：O(1) 辅助空间

### init_soft_labels

从硬标签初始化软标签概率矩阵：

```cpp
Array<const Index> hard_labels = /* ... */;  // -1 用于未标记
Array<Real> soft_labels(n_nodes * n_classes);

scl::kernel::propagation::init_soft_labels(
    hard_labels,
    n_classes,
    soft_labels,
    Real(1.0),   // labeled_confidence
    Real(0.0)    // unlabeled_prior (0=uniform)
);
```

**参数：**
- `hard_labels`: 硬标签分配（UNLABELED=-1 用于未知）
- `n_classes`: 类的数量
- `soft_labels`: 输出概率矩阵 [n * n_classes]
- `labeled_confidence`: 标记类上的概率质量
- `unlabeled_prior`: 未标记节点的先验概率（0=均匀）

**后置条件：**
- 标记节点：prob[label] = confidence，其他 = (1-conf)/(n-1)
- 未标记节点：均匀 1/n_classes 或指定先验
- 每行总和为 1

**复杂度：**
- 时间：O(n_nodes * n_classes)
- 空间：O(1) 辅助空间

## 配置

`scl::kernel::propagation::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real DEFAULT_ALPHA = 0.99;
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = 1e-6;
    constexpr Index UNLABELED = -1;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## 性能考虑

### 并行化

所有传播函数都已并行化：
- `label_propagation`: 使用 WorkspacePool 并行
- `label_spreading`: 在节点上并行，使用 SIMD
- `inductive_transfer`: 在查询节点上并行
- `confidence_propagation`: 使用 WorkspacePool 并行

### 内存效率

- WorkspacePool 用于线程本地缓冲区
- 尽可能使用原地修改
- 最少的临时分配

## 最佳实践

### 1. 正确初始化标签

```cpp
// 对于标签传播
Array<Index> labels(n_nodes, config::UNLABELED);
labels[seed_nodes] = /* 类分配 */;

// 对于标签扩散
Array<Real> probs(n_nodes * n_classes);
scl::kernel::propagation::init_soft_labels(
    hard_labels, n_classes, probs
);
```

### 2. 选择适当的方法

```cpp
// 硬分类
scl::kernel::propagation::label_propagation(adjacency, labels);

// 软概率
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled
);

// 回归/插值
scl::kernel::propagation::harmonic_function(
    adjacency, values, is_known
);
```

### 3. 调整 Alpha 参数

```cpp
// 更高的 alpha：在图中传播标签更远
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled, 0.99
);

// 更低的 alpha：更信任初始标签
scl::kernel::propagation::label_spreading(
    adjacency, probs, n_classes, is_labeled, 0.5
);
```

---

::: tip Alpha 参数
更高的 alpha（接近 1）在图中传播标签更远。更低的 alpha 更信任初始标签。
:::

::: warning 收敛
标签传播可能对某些图不收敛。检查迭代次数并考虑使用带容差的标签扩散。
:::

