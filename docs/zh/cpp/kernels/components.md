# components.hpp

> scl/kernel/components.hpp · 高性能连通分量和图连通性分析

## 概述

本文件提供用于分析稀疏图的高效图连通性算法。包括连通分量检测、广度优先搜索（BFS）、图度量（直径、三角形计数）和连通性检查。所有操作都针对大规模稀疏图进行了优化，支持并行执行。

**头文件**: `#include "scl/kernel/components.hpp"`

---

## 主要 API

### connected_components

::: source_code file="scl/kernel/components.hpp" symbol="connected_components" collapsed
:::

**算法说明**

使用并行并查集在无向图中查找所有连通分量：

1. 初始化并查集数据结构，每个节点作为自己的分量
2. 并行处理每条边 (u, v)：
   - 查找 u 的根和 v 的根
   - 如果根不同，使用无锁原子操作合并分量
3. 路径压缩和按秩合并优化以提高效率
4. 最终遍历为所有节点分配分量标签
5. 返回找到的不同分量数量

**边界条件**

- **空图**: 如果不存在节点，返回 n_components = 0
- **断开图**: 每个孤立分量获得唯一标签
- **自环**: 正确处理（节点连接到自身）
- **重复边**: 并查集自然处理相同节点之间的多条边

**数据保证（前置条件）**

- `component_labels.len >= adjacency.primary_dim()`
- 邻接矩阵表示无向图（对称）
- 图是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(nnz * α(n))，其中 α 是反阿克曼函数（实际上约为 O(nnz)）
- **空间**: O(n_nodes) 辅助空间 - 并查集数组

**示例**

```cpp
#include "scl/kernel/components.hpp"

Sparse<Real, true> adjacency = /* 邻接矩阵，CSR */;
Array<Index> component_labels(adjacency.rows());
Index n_components;

scl::kernel::components::connected_components(
    adjacency,
    component_labels,
    n_components
);

// 过滤到最大分量
Index largest_component = 0;
Array<Index> component_sizes(n_components, 0);
for (Index i = 0; i < adjacency.rows(); ++i) {
    component_sizes[component_labels[i]]++;
    if (component_sizes[component_labels[i]] > 
        component_sizes[largest_component]) {
        largest_component = component_labels[i];
    }
}
```

---

### largest_component

::: source_code file="scl/kernel/components.hpp" symbol="largest_component" collapsed
:::

**算法说明**

提取最大连通分量中的节点：

1. 调用 `connected_components` 查找所有分量
2. 统计每个分量的节点数
3. 识别具有最大大小的分量
4. 创建二进制掩码，其中 `node_mask[i] == 1` 如果节点 i 在最大分量中
5. 返回最大分量的大小

**边界条件**

- **空图**: 返回 component_size = 0
- **大小并列**: 返回第一个具有最大大小的分量
- **全部孤立**: 每个节点是自己的分量，返回大小 1

**数据保证（前置条件）**

- `node_mask.len >= adjacency.primary_dim()`
- 邻接矩阵是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(nnz) - 由 connected_components 调用主导
- **空间**: O(n_nodes) 辅助空间 - 分量标签和掩码

**示例**

```cpp
Array<Byte> node_mask(adjacency.rows());
Index component_size;

scl::kernel::components::largest_component(
    adjacency,
    node_mask,
    component_size
);

// 将图过滤到最大分量
// （使用 node_mask 选择节点）
```

---

### bfs

::: source_code file="scl/kernel/components.hpp" symbol="bfs" collapsed
:::

**算法说明**

从源节点执行广度优先搜索以计算最短路径距离：

1. 初始化距离数组为 -1（未访问）
2. 创建队列并将源节点入队，距离为 0
3. 当队列不为空时：
   - 出队节点 u
   - 对于 u 的每个邻居 v：
     - 如果 v 未访问（距离 == -1），设置 distance[v] = distance[u] + 1
     - 将 v 入队
4. 顺序实现以保证正确性

**边界条件**

- **不可达节点**: 无法从源到达的节点距离保持 -1
- **自环**: 源节点到自身的距离为 0
- **空图**: 除源（距离 0）外，所有距离保持 -1

**数据保证（前置条件）**

- `distances.len >= adjacency.primary_dim()`
- `visited.len >= adjacency.primary_dim()`（如果提供）
- 源是有效的节点索引（0 <= source < n_nodes）

**复杂度分析**

- **时间**: O(nnz) 对于连通分量 - 每条边访问一次
- **空间**: O(n_nodes) 辅助空间 - 队列和距离数组

**示例**

```cpp
Index source = 0;  // 根节点
Array<Index> distances(adjacency.rows());
Array<Index> visited(adjacency.rows());

scl::kernel::components::bfs(
    adjacency,
    source,
    distances,
    visited
);

// 找到距离为 k 的节点
for (Index i = 0; i < adjacency.rows(); ++i) {
    if (distances[i] == k) {
        // 节点 i 距离源 k 跳
    }
}
```

---

### parallel_bfs

::: source_code file="scl/kernel/components.hpp" symbol="parallel_bfs" collapsed
:::

**算法说明**

使用方向优化算法执行并行 BFS：

1. 使用位向量前沿进行高效的并行处理
2. 方向优化：根据前沿密度在自上而下和自下而上 BFS 之间切换
3. 自上而下：从当前前沿扩展（稀疏前沿）
4. 自下而上：检查所有未访问节点是否有前沿中的邻居（密集前沿）
5. 前沿节点/边的并行处理
6. 对于大图比顺序 BFS 更高效

**边界条件**

- **不可达节点**: 距离保持 -1
- **非常稀疏的图**: 偏好自上而下方法
- **非常密集的图**: 自动切换到自下而上

**数据保证（前置条件）**

- `distances.len >= adjacency.primary_dim()`
- 源是有效的节点索引

**复杂度分析**

- **时间**: O(nnz) 对于连通分量 - 与顺序相同但并行化
- **空间**: O(n_nodes) 辅助空间 - 每个线程的位向量前沿

**示例**

```cpp
Index source = find_root_cell();
Array<Index> distances(adjacency.rows());

scl::kernel::components::parallel_bfs(
    adjacency,
    source,
    distances
);

// 使用距离进行下游分析（例如，轨迹推断）
```

---

### is_connected

::: source_code file="scl/kernel/components.hpp" symbol="is_connected" collapsed
:::

**算法说明**

检查图是否连通（具有单个连通分量）：

1. 调用 `connected_components` 查找所有分量
2. 检查 n_components == 1
3. 如果单个分量返回 true，否则返回 false

**边界条件**

- **空图**: 返回 false（无节点，无法连通）
- **单个节点**: 返回 true（平凡连通）
- **断开**: 如果存在多个分量，返回 false

**数据保证（前置条件）**

- 图至少有一个节点
- 邻接矩阵是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(nnz) - 由 connected_components 主导
- **空间**: O(n_nodes) 辅助空间

**示例**

```cpp
bool connected = scl::kernel::components::is_connected(adjacency);

if (!connected) {
    // 图有多个分量，可能需要过滤
}
```

---

### graph_diameter

::: source_code file="scl/kernel/components.hpp" symbol="graph_diameter" collapsed
:::

**算法说明**

计算图直径（任意两个节点之间的最长最短路径）：

1. 并行处理每个节点：
   - 从该节点运行 BFS
   - 在 BFS 结果中找到最大距离
2. 对所有源节点取最大值
3. 返回直径值

**边界条件**

- **断开图**: 行为未定义（应首先检查连通性）
- **单个节点**: 返回直径 0
- **线性图**: 返回 n_nodes - 1

**数据保证（前置条件）**

- 图是连通的（首先使用 `is_connected`）
- 邻接矩阵是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**: O(n_nodes * nnz) - 从每个节点运行 BFS
- **空间**: O(n_nodes) 辅助空间 - 每个线程的 BFS 工作空间

**示例**

```cpp
if (scl::kernel::components::is_connected(adjacency)) {
    Index diameter = scl::kernel::components::graph_diameter(adjacency);
    // 使用直径进行图分析
}
```

---

### triangle_count

::: source_code file="scl/kernel/components.hpp" symbol="triangle_count" collapsed
:::

**算法说明**

使用优化的稀疏算法计算无向图中的三角形数量：

1. 并行处理每个节点 u：
   - 对于 u 的每个邻居 v，其中 v > u（避免重复计数）：
     - 对于 u 的每个邻居 w，其中 w > v：
       - 检查边 (v, w) 是否存在（在邻接矩阵中二分搜索）
       - 如果存在，增加三角形计数
2. 使用原子操作进行线程安全计数
3. 针对稀疏图进行优化，具有早期终止

**边界条件**

- **无三角形**: 返回 0
- **完全图**: 返回 n_nodes * (n_nodes-1) * (n_nodes-2) / 6
- **自环**: 不计为三角形

**数据保证（前置条件）**

- 图是无向的（对称邻接）
- 邻接矩阵是有效的 CSR 格式

**复杂度分析**

- **时间**: O(nnz^1.5) 对于稀疏图 - 嵌套循环加二分搜索
- **空间**: O(n_nodes) 辅助空间 - 原子计数器

**示例**

```cpp
Size n_triangles = scl::kernel::components::triangle_count(adjacency);

// 计算聚类系数
Real clustering = 3.0 * n_triangles / (n_nodes * (n_nodes - 1));
```

---

## 配置

`scl::kernel::components::config` 中的默认参数：

- `INVALID_COMPONENT = -1`: 无效分量的标记值
- `UNVISITED = -1`: 未访问节点的标记值
- `PARALLEL_NODES_THRESHOLD = 1000`: 并行处理的最小节点数
- `PARALLEL_EDGES_THRESHOLD = 10000`: 并行处理的最小边数
- `DENSE_DEGREE_THRESHOLD = 64`: 密集与稀疏节点处理的阈值

---

## 相关内容

- [邻居模块](./neighbors) - KNN 图构建
- [稀疏矩阵](../core/sparse) - 稀疏矩阵操作
