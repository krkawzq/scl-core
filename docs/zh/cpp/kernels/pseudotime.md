# 伪时间

用于轨迹分析和发育排序的伪时间推断内核。

## 概述

伪时间模块提供：

- **最短路径伪时间** - 从根细胞的图距离
- **扩散伪时间** - 扩散图距离（DPT）
- **根选择** - 基于标记或外围根选择
- **分支检测** - 识别轨迹中的分支点
- **轨迹分割** - 将细胞分配到轨迹段
- **伪时间平滑** - 基于邻域平滑
- **基因相关性** - 与伪时间的相关性
- **速度整合** - RNA 速度加权伪时间

## 最短路径伪时间

### dijkstra_shortest_path

使用 Dijkstra 算法计算从单个源节点的最短路径距离：

```cpp
#include "scl/kernel/pseudotime.hpp"

Sparse<Real, true> adjacency = /* ... */;  // 图邻接矩阵
Array<Real> distances(n_nodes);

scl::kernel::pseudotime::dijkstra_shortest_path(
    adjacency,
    source,      // 源节点索引
    distances
);
```

**参数：**
- `adjacency`: 图邻接矩阵（边权重作为距离）
- `source`: 源节点索引
- `distances`: 从源到所有节点的最短距离

**后置条件：**
- `distances[i]` = 从源到 i 的最短路径距离
- `distances[source]` = 0
- 不可达节点的距离 = INF_DISTANCE

**算法：**
4 叉堆 Dijkstra：
1. 将所有距离初始化为 INF，源为 0
2. 从堆中弹出最小值，松弛邻居
3. 继续直到堆为空

**复杂度：**
- 时间：O((V + E) * log_4(V))
- 空间：O(V) 用于堆和距离数组

**使用场景：**
- 最短路径计算
- 图距离分析
- 基于根的伪时间

### graph_pseudotime

计算从根细胞归一化的最短路径距离作为伪时间：

```cpp
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::graph_pseudotime(
    adjacency,
    root_cell,   // 轨迹的起始细胞
    pseudotime
);
```

**参数：**
- `adjacency`: 细胞邻域图
- `root_cell`: 轨迹的起始细胞
- `pseudotime`: 归一化的伪时间值 [0, 1]

**后置条件：**
- `pseudotime[root_cell]` = 0
- `pseudotime[i]` 在所有可达细胞中为 [0, 1]
- 不可达细胞的伪时间 = 1

**算法：**
1. 从 root_cell 运行 Dijkstra
2. 通过除以最大值将距离归一化到 [0, 1]
3. 将不可达细胞设置为 1

**复杂度：**
- 时间：O((V + E) * log_4(V))
- 空间：O(V) 辅助空间

**使用场景：**
- 简单伪时间推断
- 基于根的排序
- 发育轨迹

## 扩散伪时间

### diffusion_pseudotime

使用扩散图距离计算扩散伪时间（DPT）：

```cpp
Sparse<Real, true> transition_matrix = /* ... */;  // 马尔可夫转移矩阵
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::diffusion_pseudotime(
    transition_matrix,
    root_cell,
    pseudotime,
    config::DEFAULT_N_DCS,        // n_dcs = 10
    config::DEFAULT_N_ITERATIONS  // n_iterations = 100
);
```

**参数：**
- `transition_matrix`: 马尔可夫转移矩阵（行随机）
- `root_cell`: 轨迹的起始细胞
- `pseudotime`: 输出的 DPT 值 [0, 1]
- `n_dcs`: 扩散分量数量
- `n_iterations`: 幂迭代迭代次数

**后置条件：**
- 基于从根的扩散距离的伪时间
- 值归一化到 [0, 1]
- 捕获超出图距离的连通性结构

**算法：**
1. 初始化随机扩散分量 [n x n_dcs]
2. 幂迭代：DC = T * DC（应用转移）
3. 使用修改的 Gram-Schmidt 正交归一化
4. 在 DC 空间中计算从根的欧几里得距离
5. 归一化到 [0, 1]

**复杂度：**
- 时间：O(n_iterations * nnz * n_dcs)
- 空间：O(n * n_dcs) 用于扩散分量

**使用场景：**
- 鲁棒伪时间推断
- 抗噪声排序
- 复杂轨迹结构

## 根选择

### select_root_cell

选择具有最小标记基因表达的根细胞：

```cpp
Array<const Real> marker_expression = /* ... */;  // 干/早期标记

Index root = scl::kernel::pseudotime::select_root_cell(
    adjacency,
    marker_expression
);
```

**返回：** 具有最小标记表达的细胞索引

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 基于标记的根选择
- 干细胞识别
- 早期发育阶段

### select_root_peripheral

选择最外围节点作为根细胞：

```cpp
Index root = scl::kernel::pseudotime::select_root_peripheral(
    adjacency
);
```

**返回：** 最外围细胞的索引

**算法：**
1. 对于每个细胞：计算到邻居的平均边权重
2. 返回具有最大平均值（最孤立）的细胞

**复杂度：**
- 时间：O(nnz)
- 空间：O(n) 辅助空间

**使用场景：**
- 外围根选择
- 轨迹端点
- 当标记不可用时

## 分支检测

### detect_branch_points

基于伪时间拓扑识别轨迹中的分支点：

```cpp
Array<const Real> pseudotime = /* ... */;
Array<Index> branch_points(n_nodes);

Index n_branches = scl::kernel::pseudotime::detect_branch_points(
    adjacency,
    pseudotime,
    branch_points,
    config::DEFAULT_THRESHOLD  // threshold = 0.1
);
```

**参数：**
- `adjacency`: 细胞邻域图
- `pseudotime`: 预计算的伪时间值
- `branch_points`: 检测到的分支点细胞索引
- `threshold`: 用于邻居分类的伪时间差阈值

**返回：** 检测到的分支点数量

**后置条件：**
- `branch_points[0..return_value)` 包含分支细胞索引
- 分支定义为：(>=1 个更早和 >=2 个更晚) 或 (>=2 个更早和 >=1 个更晚)

**算法：**
在细胞上并行：
1. 统计伪时间 < (pt_i - threshold) 的邻居（更早）
2. 统计伪时间 > (pt_i + threshold) 的邻居（更晚）
3. 如果邻居分布不对称则标记为分支

**复杂度：**
- 时间：O(nnz)
- 空间：O(n * n_threads) 用于线程本地缓冲区

**使用场景：**
- 轨迹分支分析
- 细胞命运决策点
- 发育分叉

### segment_trajectory

基于分支点将细胞分配到轨迹段：

```cpp
Array<Index> segment_labels(n_nodes);

scl::kernel::pseudotime::segment_trajectory(
    adjacency,
    pseudotime,
    branch_points,
    n_branch_points,
    segment_labels
);
```

**参数：**
- `adjacency`: 细胞邻域图
- `pseudotime`: 伪时间值
- `branch_points`: 检测到的分支点索引
- `n_branch_points`: 分支点数量
- `segment_labels`: 每个细胞的段分配

**后置条件：**
- `segment_labels[i]` 在 [0, n_branch_points] 中
- 第一个分支之前的细胞：段 0
- 分支 k 和 k+1 之间的细胞：段 k+1

**算法：**
1. 按伪时间对分支点排序
2. 对于每个细胞：通过比较分支伪时间找到段

**复杂度：**
- 时间：O(n * n_branch_points)
- 空间：O(n_branch_points) 辅助空间

**使用场景：**
- 轨迹分割
- 分支特定分析
- 发育阶段分配

## 伪时间平滑

### smooth_pseudotime

使用邻域平均平滑伪时间值：

```cpp
Array<Real> pseudotime = /* ... */;

scl::kernel::pseudotime::smooth_pseudotime(
    adjacency,
    pseudotime,
    10,         // n_iterations
    Real(0.5)   // alpha（平滑强度）
);
```

**参数：**
- `adjacency`: 细胞邻域图
- `pseudotime`: 要平滑的伪时间值，原地修改
- `n_iterations`: 平滑迭代次数
- `alpha`: 平滑强度 [0, 1]

**后置条件：**
- 伪时间平滑：pt = (1-alpha)*pt + alpha*avg(neighbors)
- 重复 n_iterations 次

**复杂度：**
- 时间：O(n_iterations * nnz)
- 空间：O(n) 辅助空间

**使用场景：**
- 降噪
- 平滑轨迹排序
- 局部一致性

## 基因相关性

### pseudotime_correlation

计算伪时间与每个基因之间的 Pearson 相关性：

```cpp
Sparse<Real, true> X = /* ... */;  // 表达矩阵（细胞 x 基因）
Array<const Real> pseudotime = /* ... */;
Array<Real> correlations(n_genes);

scl::kernel::pseudotime::pseudotime_correlation(
    X,
    pseudotime,
    n_cells,
    n_genes,
    correlations
);
```

**参数：**
- `X`: 基因表达矩阵（细胞 x 基因，CSR）
- `pseudotime`: 伪时间值
- `n_cells`: 细胞数量
- `n_genes`: 基因数量
- `correlations`: 与伪时间的每个基因相关性

**后置条件：**
- `correlations[g]` = Pearson(pseudotime, gene_g_expression)
- 在方差计算中考虑稀疏零

**算法：**
两遍算法：
1. 第一遍：计算基因和与伪时间的协方差
2. 第二遍：计算基因方差
3. 并行相关性计算

**复杂度：**
- 时间：O(nnz + n_genes)
- 空间：O(n_genes) 辅助空间

**使用场景：**
- 轨迹相关基因
- 发育标记
- 时间依赖性表达

## 速度整合

### velocity_weighted_pseudotime

使用 RNA 速度方向信息细化伪时间：

```cpp
Array<const Real> initial_pseudotime = /* ... */;
Array<const Real> velocity_field = /* ... */;  // 每个细胞的速度
Array<Real> refined_pseudotime(n_nodes);

scl::kernel::pseudotime::velocity_weighted_pseudotime(
    adjacency,
    initial_pseudotime,
    velocity_field,
    refined_pseudotime,
    20  // n_iterations
);
```

**参数：**
- `adjacency`: 细胞邻域图
- `initial_pseudotime`: 初始伪时间估计
- `velocity_field`: 每个细胞的速度大小/方向
- `refined_pseudotime`: 速度细化的伪时间
- `n_iterations`: 细化迭代次数

**后置条件：**
- 细化的伪时间包含速度信息
- 归一化到 [0, 1]

**算法：**
对于每次迭代：
1. 对于每个细胞：使用速度调整权重的加权平均
2. 伪时间中更早的邻居：权重为 1/(1+velocity)
3. 更晚的邻居：权重为 (1+velocity)
4. 重新归一化到 [0, 1]

**复杂度：**
- 时间：O(n_iterations * nnz)
- 空间：O(n) 辅助空间

**使用场景：**
- RNA 速度整合
- 方向感知伪时间
- 改进的轨迹排序

## 工具函数

### find_terminal_states

识别终端（结束）状态作为伪时间百分位数以上的细胞：

```cpp
Array<Index> terminal_cells(n_nodes);

Index n_terminal = scl::kernel::pseudotime::find_terminal_states(
    adjacency,
    pseudotime,
    terminal_cells,
    Real(0.95)  // percentile
);
```

**返回：** 识别的终端细胞数量

**后置条件：**
- `terminal_cells[0..return_value)` 包含 pt >= threshold 的细胞
- threshold = 伪时间分布的百分位数

**复杂度：**
- 时间：O(n log n) 用于百分位数，O(n) 用于选择
- 空间：O(n) 辅助空间

**使用场景：**
- 终端状态识别
- 端点分析
- 成熟细胞类型

### compute_backbone

选择沿伪时间均匀分布的代表性骨干细胞：

```cpp
Array<Index> backbone_indices(n_backbone_cells);

Index n_backbone = scl::kernel::pseudotime::compute_backbone(
    adjacency,
    pseudotime,
    n_backbone_cells,
    backbone_indices
);
```

**返回：** 实际选择的骨干细胞数量

**后置条件：**
- `backbone_indices` 包含在伪时间中均匀采样的细胞
- 覆盖从最早到最新伪时间的完整范围

**算法：**
1. 按伪时间对细胞排序
2. 在均匀伪时间间隔处选择细胞

**复杂度：**
- 时间：O(n log n) 用于排序
- 空间：O(n) 辅助空间

**使用场景：**
- 代表性细胞选择
- 轨迹可视化
- 用于分析的减少数据集

### compute_pseudotime

带方法选择的通用伪时间计算：

```cpp
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::compute_pseudotime(
    adjacency,
    root_cell,
    pseudotime,
    PseudotimeMethod::DiffusionPseudotime,  // method
    config::DEFAULT_N_DCS  // n_dcs (for DPT)
);
```

**参数：**
- `adjacency`: 细胞邻域图或转移矩阵
- `root_cell`: 根细胞索引
- `pseudotime`: 计算的伪时间值
- `method`: 使用的算法
- `n_dcs`: 扩散分量（用于 DPT 方法）

**后置条件：**
- 使用指定方法计算的伪时间
- 值归一化到 [0, 1]

**使用场景：**
- 统一接口
- 方法比较
- 快速伪时间推断

## 配置

`scl::kernel::pseudotime::config` 中的默认参数：

```cpp
namespace config {
    constexpr Index DEFAULT_N_DCS = 10;
    constexpr Index DEFAULT_N_ITERATIONS = 100;
    constexpr Real DEFAULT_THRESHOLD = 0.1;
    constexpr Real DEFAULT_DAMPING = 0.85;
    constexpr Real CONVERGENCE_TOL = 1e-6;
    constexpr Real INF_DISTANCE = 1e30;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index HEAP_ARITY = 4;
}
```

## 性能考虑

### 并行化

- `dijkstra_multi_source`: 在源上并行
- `diffusion_pseudotime`: 并行 SpMM 和距离计算
- `detect_branch_points`: 在细胞上并行
- `pseudotime_correlation`: 并行相关性计算

### 内存效率

- 4 叉堆用于高效 Dijkstra
- WorkspacePool 用于线程本地缓冲区
- 最少的临时分配

## 最佳实践

### 1. 选择适当的方法

```cpp
// 简单轨迹
scl::kernel::pseudotime::graph_pseudotime(adjacency, root, pseudotime);

// 复杂/噪声轨迹
scl::kernel::pseudotime::diffusion_pseudotime(
    transition_matrix, root, pseudotime
);
```

### 2. 适当选择根

```cpp
// 基于标记
Index root = scl::kernel::pseudotime::select_root_cell(
    adjacency, marker_expression
);

// 外围
Index root = scl::kernel::pseudotime::select_root_peripheral(adjacency);
```

### 3. 平滑以获得更好结果

```cpp
// 初始计算后
scl::kernel::pseudotime::smooth_pseudotime(
    adjacency, pseudotime, 10, 0.5
);
```

---

::: tip 方法选择
对简单轨迹使用 graph_pseudotime，对复杂或噪声数据使用 diffusion_pseudotime。
:::

::: warning 根选择
根选择显著影响伪时间排序。尽可能使用基于标记的选择。
:::

