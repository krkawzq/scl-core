# louvain.hpp

> scl/kernel/louvain.hpp · 多层级 Louvain 社区检测算法

## 概述

高性能的 Louvain 算法实现，用于图中的社区检测。Louvain 方法是一种贪婪优化算法，通过最大化模块度来识别大型网络中的社区。

本文件提供：
- 多层级分层社区检测
- 带分辨率参数的模块度优化
- 针对大型图的并行处理
- 社区分析工具函数

**头文件**: `#include "scl/kernel/louvain.hpp"`

---

## 主要 API

### cluster

::: source_code file="scl/kernel/louvain.hpp" symbol="cluster" collapsed
:::

**算法说明**

多层级 Louvain 社区检测算法：

1. **初始化**：每个节点最初被分配到自己的社区
2. **局部移动阶段**：
   - 对于每个节点，计算移动到每个邻居社区所带来的模块度增益
   - 将节点移动到能产生最大正模块度增益的社区
   - 重复直到无法进一步改进
3. **聚合阶段**：
   - 构建一个粗化图，其中节点代表上一层的社区
   - 边权重是社区间边的总和
   - 适当标准化边权重
4. **迭代**：在粗化图上重复步骤 2-3，直到收敛或达到 max_iter

算法使用带线程本地工作空间的并行处理，以高效处理大型图。

**边界条件**

- **空图**：返回所有节点标记为 0（单一社区）
- **非连通图**：每个连通分量形成独立的社区
- **单节点**：节点被分配到社区 0
- **达到最大迭代次数**：算法停止并返回当前划分

**数据保证（前置条件）**

- 邻接矩阵必须是有效的稀疏矩阵（CSR 或 CSC 格式）
- 邻接矩阵应表示无向图（推荐对称矩阵）
- 标签数组长度必须 >= adjacency.primary_dim()
- 分辨率参数必须 > 0
- 最大迭代次数必须 > 0

**复杂度分析**

- **时间**：对于稀疏图，预期为 O(n * log(n) * avg_degree)，其中 n 是节点数。对数因子来自多层级层次结构，avg_degree 是平均节点度。
- **空间**：工作内存为 O(n + nnz)，包括线程本地工作空间和中间数据结构

**示例**

```cpp
#include "scl/kernel/louvain.hpp"

// 创建邻接矩阵（CSR 格式）
Sparse<Real, true> adjacency = /* ... */;  // n_nodes x n_nodes 稀疏矩阵
Array<Index> labels(adjacency.rows());

// 使用默认分辨率 (1.0) 进行标准聚类
scl::kernel::louvain::cluster(adjacency, labels);

// 使用更高分辨率进行聚类（产生更多、更小的社区）
scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.5);

// 使用自定义迭代限制进行聚类
scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.0, max_iter = 200);

// 社区 ID 现在在 labels[i] 中，范围从 0 到 n_communities-1
```

---

### compute_modularity

::: source_code file="scl/kernel/louvain.hpp" symbol="compute_modularity" collapsed
:::

**算法说明**

计算给定图和聚类的模块度分数 Q：

Q = (1/2m) * Σᵢⱼ [Aᵢⱼ - resolution * (kᵢ * kⱼ) / (2m)] * δ(cᵢ, cⱼ)

其中：
- Aᵢⱼ = 节点 i 和 j 之间的边权重
- m = 总边权重 / 2（所有边权重的一半和）
- kᵢ = 节点 i 的加权度
- cᵢ = 节点 i 的社区分配
- δ(x, y) = 如果 x == y 则为 1，否则为 0（Kronecker delta）
- resolution = 分辨率参数（默认 1.0）

算法步骤：
1. 并行计算总边权重 m 和节点度 kᵢ
2. 对于每条边，检查端点是否在同一社区
3. 累积模块度贡献

**边界条件**

- **单一社区**：返回 Q = 0（无结构）
- **每个节点独立**：返回负 Q（结构较差）
- **完美社区**：返回接近 1.0 的 Q
- **空图**：返回 Q = 0

**数据保证（前置条件）**

- 邻接矩阵必须是有效的稀疏矩阵
- 标签数组长度必须 >= adjacency.primary_dim()
- 所有 labels[i] 必须 >= 0
- 分辨率参数应 > 0（默认：1.0）

**复杂度分析**

- **时间**：O(n + nnz)，其中 n 是节点数，nnz 是非零边数。使用并行处理计算节点度。
- **空间**：O(n) 辅助空间用于存储节点度和社区总和

**示例**

```cpp
#include "scl/kernel/louvain.hpp"

Sparse<Real, true> adjacency = /* ... */;
Array<Index> labels = /* 聚类分配 */;

// 使用默认分辨率计算模块度
Real modularity = scl::kernel::louvain::compute_modularity(adjacency, labels);

// 使用自定义分辨率计算
Real mod_res15 = scl::kernel::louvain::compute_modularity(adjacency, labels, resolution = 1.5);

// 模块度范围从 -0.5 到 1.0
// Q > 0 表示社区结构强于随机
// Q 接近 1.0 表示非常强的社区结构
```

---

### community_sizes

::: source_code file="scl/kernel/louvain.hpp" symbol="community_sizes" collapsed
:::

**算法说明**

统计每个社区中的节点数：

1. 将 sizes 数组初始化为零
2. 遍历所有标签，对每个节点 i 递增 sizes[labels[i]]
3. 社区数量为 max(labels) + 1

**边界条件**

- **空标签数组**：返回 n_communities = 0，sizes 不变
- **所有节点在一个社区**：返回 n_communities = 1，sizes[0] = n_nodes
- **每个节点在独立社区**：返回 n_communities = n_nodes，每个 sizes[i] = 1

**数据保证（前置条件）**

- 所有 labels[i] 必须 >= 0
- Sizes 数组长度必须 >= max(labels) + 1

**复杂度分析**

- **时间**：O(n)，其中 n 是标签数组的长度。单次遍历标签。
- **空间**：O(1) 辅助空间（输出 sizes 数组由调用者提供）

**示例**

```cpp
#include "scl/kernel/louvain.hpp"

Array<Index> labels = /* 来自 cluster() 的聚类标签 */;
Index max_label = /* ... */;  // 确定最大标签值
Array<Index> sizes(max_label + 1);
Index n_communities;

scl::kernel::louvain::community_sizes(labels, sizes, n_communities);

// sizes[c] 现在包含社区 c 中的节点数
// n_communities 包含社区总数
```

---

### get_community_members

::: source_code file="scl/kernel/louvain.hpp" symbol="get_community_members" collapsed
:::

**算法说明**

提取属于特定社区的所有节点的索引：

1. 遍历所有标签
2. 对于每个满足 labels[i] == community 的节点 i，将 i 添加到 members 数组
3. 统计总成员数并存储在 n_members 中

**边界条件**

- **社区不存在**：返回 n_members = 0
- **所有节点在目标社区**：返回 n_members = n_nodes
- **空社区**：返回 n_members = 0
- **成员数组太小**：只存储前 members.len 个索引

**数据保证（前置条件）**

- 社区 ID 必须 >= 0
- 成员数组应足够大以容纳所有成员（可以安全地传递大缓冲区）

**复杂度分析**

- **时间**：O(n)，其中 n 是标签数组的长度。单次遍历标签。
- **空间**：O(1) 辅助空间（输出 members 数组由调用者提供）

**示例**

```cpp
#include "scl/kernel/louvain.hpp"

Array<Index> labels = /* 聚类标签 */;
Index target_community = 5;
Array<Index> members(labels.len);  // 分配足够大的缓冲区
Index n_members;

scl::kernel::louvain::get_community_members(
    labels, target_community, members, n_members
);

// members[0..n_members-1] 现在包含社区 5 中的节点索引
```

---

## 配置

命名空间 `scl::kernel::louvain::config` 提供配置常量：

- `DEFAULT_RESOLUTION = 1.0`：默认分辨率参数
- `DEFAULT_MAX_ITER = 100`：默认最大迭代次数
- `MODULARITY_EPSILON = 1e-8`：模块度变化的收敛阈值
- `PARALLEL_THRESHOLD = 1000`：并行处理的最小图大小
- `MAX_LEVELS = 100`：多层级层次结构的最大深度

## 注意事项

- 由于节点排序的贪婪性质，Louvain 算法在不同运行中可能产生不同的结果。为了可重现的结果，如果内部使用了随机数生成器，请考虑设置种子。
- 更高的分辨率值（例如 1.5、2.0）产生更多、更小的社区。
- 更低的分辨率值（例如 0.5）产生更少、更大的社区。
- 算法针对稀疏图进行了优化。对于密集图，请考虑使用替代算法。

## 相关内容

- [Leiden 聚类](./leiden) - 保证质量改进的替代社区检测算法
- [质量评估](./metrics) - 评估聚类结果的质量指标
