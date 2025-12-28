# 中心性

用于网络分析的高性能图中心性度量。

## 概述

中心性模块提供：

- **度中心性** - 基于度的简单重要性
- **PageRank** - 基于随机游走的重要性
- **HITS** - 枢纽和权威分数
- **特征向量中心性** - 主导特征向量
- **Katz 中心性** - 广义特征向量
- **接近中心性** - 平均最短路径长度
- **介数中心性** - 最短路径中介
- **调和中心性** - 逆距离之和
- **随机游走中心性** - 基于访问频率

## 基本中心性

### degree_centrality

计算度中心性（边权重之和）：

```cpp
#include "scl/kernel/centrality.hpp"

Sparse<Real, true> adjacency = /* ... */;
Array<Real> centrality(n_nodes);

scl::kernel::centrality::degree_centrality(
    adjacency,
    centrality,
    true  // normalize = true
);
```

**参数：**
- `adjacency`: 邻接矩阵（CSR 或 CSC）
- `centrality`: 输出的度中心性分数 [n_nodes]
- `normalize`: 如果为 true，按最大度归一化

**后置条件：**
- `centrality[i]` 包含节点 i 的度
- 如果 normalize=true，值在 [0, 1] 范围内

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间

**使用场景：**
- 快速重要性排序
- 网络结构分析
- 其他度量的预处理

## 随机游走中心性

### pagerank

使用幂迭代计算 PageRank 中心性：

```cpp
Array<Real> scores(n_nodes);

scl::kernel::centrality::pagerank(
    adjacency,
    scores,
    config::DEFAULT_DAMPING,      // damping = 0.85
    config::DEFAULT_MAX_ITER,     // max_iter = 100
    config::DEFAULT_TOLERANCE     // tol = 1e-6
);
```

**参数：**
- `adjacency`: 邻接矩阵（CSR 或 CSC）
- `scores`: 输出的 PageRank 分数 [n_nodes]
- `damping`: 阻尼因子（默认 0.85）
- `max_iter`: 最大迭代次数
- `tol`: 收敛容差

**算法：**
幂迭代：`scores = (1-d) * teleport + d * A^T * scores`

**后置条件：**
- 分数总和为 1.0
- 收敛到平稳分布

**复杂度：**
- 时间：O(max_iter * nnz)
- 空间：O(n_nodes) 辅助空间

**使用场景：**
- 网页排名
- 网络中的节点重要性
- 影响传播

### personalized_pagerank

使用自定义传送向量计算个性化 PageRank：

```cpp
Array<const Real> personalization(n_nodes);
// ... 设置个性化概率（总和为 1.0）...

Array<Real> scores(n_nodes);

scl::kernel::centrality::personalized_pagerank(
    adjacency,
    personalization,
    scores,
    config::DEFAULT_DAMPING,
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**参数：**
- `personalization`: 传送概率 [n_nodes]（必须总和为 1.0）
- `scores`: 输出的个性化 PageRank 分数 [n_nodes]

**使用场景：**
- 基于种子的重要性
- 围绕特定节点的局部重要性
- 个性化推荐

### hits

计算 HITS（超链接诱导主题搜索）枢纽和权威分数：

```cpp
Array<Real> hub_scores(n_nodes);
Array<Real> authority_scores(n_nodes);

scl::kernel::centrality::hits(
    adjacency,
    hub_scores,
    authority_scores,
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**参数：**
- `hub_scores`: 输出的枢纽分数 [n_nodes]
- `authority_scores`: 输出的权威分数 [n_nodes]

**后置条件：**
- `auth[j] = sum_i(hub[i] * A[i,j])`
- `hub[i] = sum_j(auth[j] * A[i,j])`
- 两个分数都已归一化

**复杂度：**
- 时间：O(max_iter * nnz)
- 空间：O(n_nodes) 辅助空间

**使用场景：**
- 有向网络分析
- 识别枢纽和权威
- 网络搜索排名

## 特征向量中心性

### eigenvector_centrality

计算特征向量中心性（主导特征向量）：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::eigenvector_centrality(
    adjacency,
    centrality,
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**参数：**
- `centrality`: 输出的特征向量中心性 [n_nodes]

**后置条件：**
- 中心性包含主导特征向量
- 向量已 L2 归一化

**复杂度：**
- 时间：O(max_iter * nnz)
- 空间：O(n_nodes) 辅助空间

**使用场景：**
- 长期影响
- 网络结构分析
- 连通分量重要性

### katz_centrality

计算 Katz 中心性：`centrality = alpha * A * centrality + beta`：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::katz_centrality(
    adjacency,
    centrality,
    Real(0.1),  // alpha（衰减因子）
    Real(1.0),  // beta（常数项）
    config::DEFAULT_MAX_ITER,
    config::DEFAULT_TOLERANCE
);
```

**参数：**
- `alpha`: 衰减因子（默认 0.1，必须 < 1 / lambda_max）
- `beta`: 常数项（默认 1.0）

**使用场景：**
- 广义特征向量中心性
- 加权重要性
- 网络影响

## 路径中心性

### closeness_centrality

计算接近中心性（平均最短路径长度的倒数）：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::closeness_centrality(
    adjacency,
    centrality,
    true  // normalize = true
);
```

**参数：**
- `centrality`: 输出的接近中心性 [n_nodes]
- `normalize`: 如果为 true，按 (n-1) 归一化

**后置条件：**
- `centrality[i] = (n-1) / sum(distances[i])`
- 孤立节点的中心性为 0

**复杂度：**
- 时间：O(n_nodes * nnz)（从每个节点进行 BFS）
- 空间：O(n_nodes) 辅助空间（每线程）

**使用场景：**
- 信息传播效率
- 网络可达性
- 中心位置识别

### betweenness_centrality

使用 Brandes 算法计算介数中心性：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::betweenness_centrality(
    adjacency,
    centrality,
    true  // normalize = true
);
```

**参数：**
- `centrality`: 输出的介数中心性 [n_nodes]
- `normalize`: 如果为 true，按 (n-1)*(n-2)/2 归一化

**算法：**
Brandes 算法：从每个源进行 BFS，累积依赖关系

**后置条件：**
- `centrality[i]` 包含通过 i 的最短路径比例
- 归一化值在 [0, 1] 范围内

**复杂度：**
- 时间：O(n_nodes * nnz)（未加权）
- 空间：O(n_nodes) 辅助空间（每线程）

**使用场景：**
- 桥梁识别
- 网络瓶颈分析
- 关键路径检测

### approximate_betweenness

使用随机采样计算近似介数中心性：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::approximate_betweenness(
    adjacency,
    centrality,
    n_samples,  // 要采样的源节点数
    true,        // normalize = true
    42           // seed
);
```

**参数：**
- `n_samples`: 要采样的源节点数
- `normalize`: 如果为 true，归一化
- `seed`: 随机种子

**复杂度：**
- 时间：O(n_samples * nnz)
- 空间：O(n_nodes) 辅助空间（每线程）

**使用场景：**
- 大型网络分析
- 当精确介数计算过于昂贵时
- 近似排序

### harmonic_centrality

计算调和中心性（逆距离之和）：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::harmonic_centrality(
    adjacency,
    centrality,
    true  // normalize = true
);
```

**参数：**
- `centrality`: 输出的调和中心性 [n_nodes]
- `normalize`: 如果为 true，按 (n-1) 归一化

**后置条件：**
- `centrality[i] = sum(1 / dist(i,j))` 对于 j != i
- 适用于不连通图（与接近度不同）

**复杂度：**
- 时间：O(n_nodes * nnz)
- 空间：O(n_nodes) 辅助空间（每线程）

**使用场景：**
- 不连通网络
- 对孤立节点鲁棒
- 接近度的替代方案

## 随机游走中心性

### random_walk_centrality

基于随机游走访问频率计算中心性：

```cpp
Array<Real> centrality(n_nodes);

scl::kernel::centrality::random_walk_centrality(
    adjacency,
    centrality,
    n_walks,      // 随机游走数量
    walk_length,  // 每次游走的长度
    42            // seed
);
```

**参数：**
- `n_walks`: 随机游走数量
- `walk_length`: 每次游走的长度
- `seed`: 随机种子

**后置条件：**
- `centrality[i]` 包含归一化的访问频率
- 值总和为 1.0

**复杂度：**
- 时间：O(n_walks * walk_length)
- 空间：O(n_nodes) 辅助空间（每线程）

**使用场景：**
- 概率重要性
- 随机游走分析
- 网络探索

## 配置

`scl::kernel::centrality::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real DEFAULT_DAMPING = 0.85;
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = 1e-6;
    constexpr Real MIN_SCORE = 1e-15;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
}
```

## 性能考虑

### 并行化

所有中心性函数都已并行化：
- `degree_centrality`: 在节点上并行
- `pagerank`: 使用原子累加并行
- `betweenness_centrality`: 在源节点上并行
- `closeness_centrality`: 在源节点上并行

### 迭代方法

幂迭代方法（PageRank、特征向量、Katz）：
- 通常在 O(log(n)) 次迭代中收敛
- 达到容差时提前停止
- 每次迭代检查收敛

### 路径方法

介数和接近度：
- 使用 BFS 计算最短路径
- 在源节点上并行
- 高效的基于队列的实现

## 最佳实践

### 1. 选择适当的度量

```cpp
// 快速排序
Array<Real> degree(n_nodes);
scl::kernel::centrality::degree_centrality(adjacency, degree);

// 影响分析
Array<Real> pagerank(n_nodes);
scl::kernel::centrality::pagerank(adjacency, pagerank);

// 桥梁检测
Array<Real> betweenness(n_nodes);
scl::kernel::centrality::betweenness_centrality(adjacency, betweenness);
```

### 2. 对大型网络使用近似方法

```cpp
// 精确介数计算昂贵
if (n_nodes > 10000) {
    // 使用近似
    scl::kernel::centrality::approximate_betweenness(
        adjacency, centrality, 1000  // 采样 1000 个节点
    );
} else {
    // 使用精确
    scl::kernel::centrality::betweenness_centrality(
        adjacency, centrality
    );
}
```

### 3. 调整迭代参数

```cpp
// 为了更快收敛，降低容差
scl::kernel::centrality::pagerank(
    adjacency,
    scores,
    0.85,  // damping
    50,    // max_iter（更少迭代）
    1e-4   // tol（更宽松的容差）
);
```

---

::: tip 性能
对于大型网络（n_nodes > 10000），使用近似介数以减少计算时间。
:::

::: warning 收敛
迭代方法（PageRank、特征向量）可能对某些图不收敛。检查迭代次数。
:::

