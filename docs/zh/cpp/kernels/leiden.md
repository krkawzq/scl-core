# leiden.hpp

> scl/kernel/leiden.hpp · 用于社区检测的高性能 Leiden 聚类

## 概述

本文件提供用于图中社区检测的 Leiden 算法。Leiden 算法是对 Louvain 算法的改进，通过细化步骤保证社区的良好连通性。

**头文件**: `#include "scl/kernel/leiden.hpp"`

---

## 主要 API

### cluster

::: source_code file="scl/kernel/leiden.hpp" symbol="cluster" collapsed
:::

**算法说明**

使用多级优化在邻接图上执行 Leiden 聚类：

1. **局部移动阶段**：
   - 对于每个节点，计算移动到相邻社区的模块度增益
   - 将节点移动到具有最高正增益的社区
   - 重复直到没有正移动可能

2. **细化阶段**：
   - 合并社区内的节点以改善连通性
   - 确保社区良好连通（Leiden 算法的保证）

3. **聚合阶段**：
   - 创建新图，其中节点是上一级的社区
   - 边权重是社区之间边的总和

4. **迭代**：
   - 重复局部移动、细化和聚合
   - 继续直到收敛或达到 max_iter

**边界条件**

- **空图**: 返回所有节点的单个社区
- **不连通图**: 每个连通分量形成独立社区
- **零分辨率**: 所有节点在单个社区中
- **极高分辨率**: 每个节点成为自己的社区
- **孤立节点**: 分配到自己的社区

**数据保证（前置条件）**

- `adjacency` 必须是有效的 CSR 或 CSC 稀疏矩阵
- `labels` 必须具有容量 >= adjacency.primary_dim()
- 矩阵应表示无向图（对称邻接）
- 允许自环，但通常在社区检测中移除

**复杂度分析**

- **时间**: O(max_iter * nnz * log(n_nodes)) - 每次迭代处理所有边，对数因子来自细化
- **空间**: O(n_nodes) 辅助空间 - 存储社区分配、节点度数和临时数据结构

**示例**

```cpp
#include "scl/kernel/leiden.hpp"

// 创建或加载邻接矩阵
Sparse<Real, true> adjacency = /* ... */;  // CSR 格式
Index n_nodes = adjacency.rows();

// 预分配输出标签
Array<Index> labels(n_nodes);

// 执行 Leiden 聚类
scl::kernel::leiden::cluster(
    adjacency, labels,
    resolution = 1.0,    // 越高 = 更多社区
    max_iter = 10,       // 最大迭代次数
    seed = 42            // 随机种子用于可重现性
);

// labels[i] 包含节点 i 的社区 ID
// 社区良好连通（Leiden 算法保证）

// 分析结果
std::map<Index, Size> community_sizes;
for (Index i = 0; i < n_nodes; ++i) {
    community_sizes[labels[i]]++;
}

std::cout << "找到 " << community_sizes.size() << " 个社区\n";
for (const auto& [comm_id, size] : community_sizes) {
    std::cout << "社区 " << comm_id << ": " << size << " 个节点\n";
}
```

---

### modularity

::: source_code file="scl/kernel/leiden.hpp" symbol="modularity" collapsed
:::

**算法说明**

计算分区的模块度 Q，衡量社区结构质量：

1. 计算总边权重 m = 所有边权重的总和
2. 对于每个社区 c：
   - 计算社区内边权重总和 (e_c)
   - 计算社区内节点度数总和 (a_c)
3. 模块度 Q = sum_c (e_c/m - (a_c/(2*m))^2) - resolution * sum_c (a_c/(2*m))^2

分辨率参数控制社区数量和大小之间的权衡。

**边界条件**

- **空分区**: 返回 0.0
- **单个社区**: 返回负值（所有节点在一个社区中）
- **每个节点独立**: 返回负值（没有社区内边）
- **零分辨率**: 没有分辨率参数的标准模块度

**数据保证（前置条件）**

- `adjacency` 必须是有效的 CSR 或 CSC 稀疏矩阵
- `labels` 必须具有长度 >= adjacency.primary_dim()
- 标签应该是有效的社区 ID（非负整数）

**复杂度分析**

- **时间**: O(nnz) - 遍历所有边以计算社区内权重
- **空间**: O(n_nodes) 辅助空间 - 存储社区度数总和

**示例**

```cpp
#include "scl/kernel/leiden.hpp"

// 执行聚类
Sparse<Real, true> adjacency = /* ... */;
Array<Index> labels(n_nodes);
scl::kernel::leiden::cluster(adjacency, labels, resolution = 1.0);

// 计算模块度
Real q = scl::kernel::leiden::modularity(
    adjacency, labels,
    resolution = 1.0
);

std::cout << "模块度: " << q << "\n";
// 更高的值表示更好的社区结构
// 典型范围: [-1, 1]，值 > 0.3 被认为良好

// 比较不同分辨率
for (Real res = 0.5; res <= 2.0; res += 0.5) {
    scl::kernel::leiden::cluster(adjacency, labels, resolution = res);
    Real q = scl::kernel::leiden::modularity(adjacency, labels, resolution = res);
    std::cout << "分辨率 " << res << ": Q = " << q << "\n";
}
```

---

## 配置

### 默认参数

```cpp
namespace scl::kernel::leiden::config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Index DEFAULT_MAX_MOVES = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-10);
    constexpr Real THETA = Real(0.05);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size HASH_LOAD_FACTOR_INV = 2;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index MIN_COMMUNITY_SIZE = 1;
    constexpr Real AGGREGATION_THRESHOLD = 0.8;
}
```

---

## 注意事项

**分辨率参数**: 控制社区数量。更高分辨率（如 2.0）创建更多、更小的社区。更低分辨率（如 0.5）创建更少、更大的社区。默认 1.0 是良好的起点。

**Leiden vs. Louvain**: Leiden 算法通过细化步骤保证社区良好连通，而 Louvain 可能产生不连通的社区。对于生物网络，通常首选 Leiden。

**收敛性**: 算法通常在 5-10 次迭代中收敛。如果不收敛，增加 max_iter 或检查图连通性。

**线程安全**: 使用原子操作进行并行更新，对并发执行安全。

---

## 相关内容

- [Louvain](/zh/cpp/kernels/louvain) - Louvain 聚类算法
- [Neighbors](/zh/cpp/kernels/neighbors) - 用于图构建的 K 近邻

