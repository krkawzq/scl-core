# metrics.hpp

> scl/kernel/metrics.hpp · 聚类和整合质量评估指标

## 概述

用于评估聚类结果、批次整合质量和标签相似性的综合质量指标套件。这些指标对于评估算法性能、数据整合成功和生物学解释至关重要。

本文件提供：
- 聚类质量指标（silhouette、ARI、NMI、purity）
- 批次整合指标（batch entropy、LISI）
- 基于图的连通性度量
- 标签相似性比较

**头文件**: `#include "scl/kernel/metrics.hpp"`

---

## 主要 API

### silhouette_score

::: source_code file="scl/kernel/metrics.hpp" symbol="silhouette_score" collapsed
:::

**算法说明**

计算所有样本的平均轮廓系数，衡量每个样本与其自身聚类相比与其他聚类的相似程度：

1. **对于每个细胞 i**，其聚类标签为 c：
   - 计算 a(i) = 与同一聚类 c 中其他细胞的平均距离
   - 计算 b(i) = 与其他每个聚类的平均距离的最小值
   - 计算轮廓系数：s(i) = (b(i) - a(i)) / max(a(i), b(i))
2. **返回平均值**：所有 s(i) 值的平均值

轮廓分数范围从 -1 到 1：
- **1**：细胞聚类良好（比到其他聚类更接近自身聚类）
- **0**：细胞位于聚类边界
- **-1**：细胞更接近其他聚类而非自身聚类

**边界条件**

- **少于 2 个聚类**：返回 0
- **少于 2 个细胞**：返回 0
- **单例聚类**：从计算中排除（单例成员的分数 = 0）
- **每个聚类单个细胞**：返回 0（无有意义的比较）
- **非连通距离矩阵**：优雅处理，仅考虑连通分量

**数据保证（前置条件）**

- 距离矩阵行数必须等于标签数组长度
- 距离值应为非负（负值可能导致问题）
- 至少需要 2 个细胞和 2 个聚类才能得到有意义的结果
- 距离矩阵应对称（虽然不是严格要求）

**复杂度分析**

- **时间**：O(n * nnz_per_row * n_clusters)，其中 n 是细胞数，nnz_per_row 是每行平均非零数（邻居数），n_clusters 是聚类数。对于每个细胞，我们检查到所有邻居的距离并为所有聚类计算平均值。
- **空间**：O(n_clusters) 辅助空间用于存储每个聚类的统计信息

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* 成对距离矩阵 [n_cells x n_cells] */;
Array<Index> labels = /* 聚类分配 [n_cells] */;

// 计算平均轮廓分数
Real score = scl::kernel::metrics::silhouette_score(distances, labels);

// 分数范围从 -1 到 1
// 较高的分数表示更好的聚类
// score > 0.5: 强聚类
// score < 0: 差聚类（细胞更接近其他聚类）
```

---

### adjusted_rand_index

::: source_code file="scl/kernel/metrics.hpp" symbol="adjusted_rand_index" collapsed
:::

**算法说明**

计算两个聚类之间的调整兰德指数（ARI），衡量经随机性调整后的相似性：

1. **构建列联表**：统计样本对的数量，这些样本对：
   - 在两个聚类中都在同一聚类（n_ij）
   - 仅在第一个聚类中在同一聚类（a_i）
   - 仅在第二个聚类中在同一聚类（b_j）
2. **计算总和**：
   - sum_nij = 所有聚类对的 C(n_ij, 2) 之和
   - sum_ai = 第一个聚类中所有聚类的 C(a_i, 2) 之和
   - sum_bj = 第二个聚类中所有聚类的 C(b_j, 2) 之和
3. **计算 ARI**：
   - expected = (sum_ai * sum_bj) / C(n, 2)
   - ARI = (sum_nij - expected) / (mean - expected)
   - 其中 mean = (sum_ai + sum_bj) / 2

ARI 范围从 -1 到 1：
- **1**：相同的聚类
- **0**：随机标记（平均而言）
- **-1**：最大不一致

**边界条件**

- **相同的聚类**：返回 1.0
- **独立聚类**：返回约 0.0（平均而言）
- **空标签**：返回 0.0
- **两者都是单聚类**：返回 1.0（平凡一致性）
- **不同的聚类数**：通过列联表正确处理

**数据保证（前置条件）**

- labels1 和 labels2 必须具有相等长度
- 所有标签值必须是非负整数
- 标签值不需要连续（将在内部映射）

**复杂度分析**

- **时间**：O(n + n_clusters1 * n_clusters2)，其中 n 是样本数，n_clusters1 和 n_clusters2 是每个聚类中的聚类数。构建列联表是 O(n)，计算总和是 O(n_clusters1 * n_clusters2)。
- **空间**：O(n_clusters1 * n_clusters2) 用于列联表存储

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Array<Index> labels1 = /* 第一个聚类 [n] */;
Array<Index> labels2 = /* 第二个聚类 [n] */;

// 计算两个聚类之间的 ARI
Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);

// ARI = 1.0: 完美一致
// ARI = 0.0: 随机一致（平均而言）
// ARI < 0: 一致性低于随机
```

---

### normalized_mutual_information

::: source_code file="scl/kernel/metrics.hpp" symbol="normalized_mutual_information" collapsed
:::

**算法说明**

计算两个聚类之间的归一化互信息（NMI）：

1. **构建列联表**：统计共现次数 n_ij（在第一个聚类的聚类 i 和第二个聚类的聚类 j 中的样本）
2. **计算熵**：
   - H(labels1) = -sum_i (n_i / n) * log2(n_i / n)
   - H(labels2) = -sum_j (n_j / n) * log2(n_j / n)
3. **计算互信息**：
   - MI = sum_ij (n_ij / n) * log2((n_ij * n) / (n_i * n_j))
4. **归一化**：
   - NMI = 2 * MI / (H(labels1) + H(labels2))

NMI 范围从 0 到 1：
- **1**：完美一致
- **0**：独立聚类（无互信息）

**边界条件**

- **相同的聚类**：返回 1.0
- **独立聚类**：返回 0.0
- **空标签**：返回 0.0
- **一个聚类中单聚类**：返回 0.0（无信息）

**数据保证（前置条件）**

- labels1 和 labels2 必须具有相等长度
- 所有标签值必须是非负整数

**复杂度分析**

- **时间**：O(n + n_clusters1 * n_clusters2) - 构建列联表是 O(n)，计算熵和 MI 是 O(n_clusters1 * n_clusters2)
- **空间**：O(n_clusters1 * n_clusters2) 用于列联表

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Array<Index> labels1 = /* 第一个聚类 */;
Array<Index> labels2 = /* 第二个聚类 */;

Real nmi = scl::kernel::metrics::normalized_mutual_information(labels1, labels2);

// NMI 范围从 0 到 1
// 较高的值表示更好的一致性
```

---

### graph_connectivity

::: source_code file="scl/kernel/metrics.hpp" symbol="graph_connectivity" collapsed
:::

**算法说明**

测量聚类连通性，即图中完全连通（单个连通分量）的聚类比例：

1. **对于每个聚类 c**：
   - 提取由标签为 c 的节点诱导的子图
   - 执行 BFS（广度优先搜索）查找连通分量
   - 统计连通分量数
2. **统计连通聚类**：恰好有一个分量的聚类
3. **返回比率**：connected_clusters / total_clusters

此指标对于单细胞分析很重要，其中聚类应在细胞-细胞相似性图中形成连通区域。

**边界条件**

- **所有聚类连通**：返回 1.0
- **所有聚类碎片化**：返回 0.0
- **空邻接矩阵**：返回 0.0
- **非连通图**：完整图中的每个连通分量形成单独的聚类分量
- **单例聚类**：计为连通（单个节点 = 单个分量）

**数据保证（前置条件）**

- 邻接矩阵行数必须等于标签数组长度
- 邻接矩阵应对称以进行无向连通性分析
- 图应表示细胞-细胞相似性（例如，KNN 图）

**复杂度分析**

- **时间**：O(n + nnz)，其中 n 是细胞数，nnz 是边数。每个聚类的 BFS 访问该聚类中的所有节点和边。
- **空间**：O(n) 用于分量 ID 和 BFS 队列存储

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> adjacency = /* 细胞-细胞相似性图 [n_cells x n_cells] */;
Array<Index> labels = /* 聚类标签 [n_cells] */;

Real connectivity = scl::kernel::metrics::graph_connectivity(adjacency, labels);

// connectivity = 1.0: 所有聚类完全连通
// connectivity = 0.5: 一半的聚类连通
// 较低的值表示碎片化的聚类
```

---

### batch_entropy

::: source_code file="scl/kernel/metrics.hpp" symbol="batch_entropy" collapsed
:::

**算法说明**

计算每个细胞的邻域中批次分布的归一化熵，衡量批次混合质量：

1. **对于每个细胞 i**（并行处理）：
   - 获取邻域：KNN 图中的邻居（包括自身）
   - 统计批次出现次数：统计有多少邻居属于每个批次
   - 计算香农熵：H = -sum_b (p_b * log2(p_b))，其中 p_b 是批次 b 中邻居的比例
   - 归一化：normalized_entropy = H / log2(n_batches)
2. **存储每细胞分数**：entropy_scores[i] = 细胞 i 的归一化熵

归一化熵范围从 0 到 1：
- **1**：完美的批次混合（邻域中跨批次均匀分布）
- **0**：无混合（所有邻居来自单个批次）

**边界条件**

- **单个批次**：所有分数为 0（log(n_batches) = 0，已优雅处理）
- **完美混合**：所有分数接近 1.0
- **无混合**：分数为 0.0
- **小邻域**：熵可能受邻域大小影响

**数据保证（前置条件）**

- 邻居矩阵行数必须等于 batch_labels 数组长度
- 邻居矩阵行数必须等于 entropy_scores 数组长度
- 批次标签必须是非负整数
- KNN 图应对称（虽然不是严格要求）

**复杂度分析**

- **时间**：O(n * k)，其中 n 是细胞数，k 是每个细胞的平均邻居数。对于每个细胞，我们检查 k 个邻居。
- **空间**：O(n_batches * n_threads) 用于并行处理期间的线程本地批次计数器

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Index, true> neighbors = /* KNN 图 [n_cells x n_cells] */;
Array<Index> batch_labels = /* 批次分配 [n_cells] */;
Array<Real> entropy_scores(n_cells);

scl::kernel::metrics::batch_entropy(neighbors, batch_labels, entropy_scores);

// entropy_scores[i] = 细胞 i 邻域中的归一化批次熵
// 较高的值表示更好的批次混合
```

---

### lisi

::: source_code file="scl/kernel/metrics.hpp" symbol="lisi" collapsed
:::

**算法说明**

计算局部逆辛普森指数（LISI），用于测量局部邻域中的标签多样性：

1. **对于每个细胞 i**（并行处理）：
   - 获取邻域：KNN 图中的邻居（包括自身）
   - 统计标签出现次数：统计有多少邻居具有每个标签值
   - 计算辛普森指数：SI = sum_b (p_b^2)，其中 p_b 是具有标签 b 的邻居比例
   - 计算 LISI：LISI = 1 / SI
2. **存储每细胞分数**：lisi_scores[i] = 细胞 i 的 LISI

LISI 范围从 1 到 n_labels：
- **1**：所有邻居具有相同标签（无多样性）
- **n_labels**：完美多样性（跨所有标签均匀分布）

较高的 LISI 值表示邻域中更大的标签多样性。

**边界条件**

- **均匀分布**：LISI 接近唯一标签数
- **邻域中单个标签**：LISI = 1.0
- **空邻域**：优雅处理（仅自身）
- **比例并列**：正确计算 LISI

**数据保证（前置条件）**

- 邻居矩阵行数必须等于标签数组长度
- 邻居矩阵行数必须等于 lisi_scores 数组长度
- 标签必须是非负整数
- KNN 图应表示有意义的细胞-细胞相似性

**复杂度分析**

- **时间**：O(n * k)，其中 n 是细胞数，k 是每个细胞的平均邻居数
- **空间**：O(n_labels * n_threads) 用于并行处理期间的线程本地标签计数器

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Index, true> neighbors = /* KNN 图 */;
Array<Index> labels = /* 标签（批次或细胞类型）[n_cells] */;
Array<Real> lisi_scores(n_cells);

scl::kernel::metrics::lisi(neighbors, labels, lisi_scores);

// lisi_scores[i] = 细胞 i 的 LISI
// LISI = 1: 无多样性（邻域中单个标签）
// 较高的 LISI: 邻域中更多样化的标签
```

---

### silhouette_samples

::: source_code file="scl/kernel/metrics.hpp" symbol="silhouette_samples" collapsed
:::

**算法说明**

计算每个单独样本的轮廓系数（与 silhouette_score 相同的算法，但返回每样本分数）：

1. **对于每个细胞 i**（并行处理）：
   - 计算 a(i) = 与同一聚类中其他细胞的平均距离
   - 计算 b(i) = 与其他聚类中细胞的平均距离的最小值
   - 计算 s(i) = (b(i) - a(i)) / max(a(i), b(i))
2. **存储每样本分数**：scores[i] = s(i)

每个分数范围从 -1 到 1，解释与 silhouette_score 相同。

**边界条件**

- **单例聚类**：成员的分数为 0
- **单细胞聚类**：分数为 0
- **少于 2 个聚类**：所有分数为 0

**数据保证（前置条件）**

- 距离矩阵行数必须等于标签数组长度
- 标签数组长度必须等于分数数组长度
- 至少需要 2 个聚类才能得到有意义的分数

**复杂度分析**

- **时间**：O(n * nnz_per_row * n_clusters) - 与 silhouette_score 相同，但计算所有每样本值
- **空间**：O(n_clusters * n_threads) 用于并行处理期间的线程本地缓冲区

**示例**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* 成对距离 */;
Array<Index> labels = /* 聚类标签 */;
Array<Real> scores(n_cells);

scl::kernel::metrics::silhouette_samples(distances, labels, scores);

// scores[i] = 细胞 i 的轮廓分数
// 可以识别聚类较差的细胞（scores < 0）
```

---

## 工具函数

### fowlkes_mallows_index

衡量聚类之间的相似性，作为精确率和召回率的几何平均。

::: source_code file="scl/kernel/metrics.hpp" symbol="fowlkes_mallows_index" collapsed
:::

**复杂度**
- 时间：O(n + n_clusters1 * n_clusters2)
- 空间：O(n_clusters1 * n_clusters2)

---

### v_measure

同质性和完整性的调和平均。

::: source_code file="scl/kernel/metrics.hpp" symbol="v_measure" collapsed
:::

**复杂度**
- 时间：O(n + n_classes * n_clusters)
- 空间：O(n_classes * n_clusters)

---

### homogeneity_score

衡量每个聚类是否仅包含单个类的成员。

::: source_code file="scl/kernel/metrics.hpp" symbol="homogeneity_score" collapsed
:::

**复杂度**
- 时间：O(n + n_classes * n_clusters)
- 空间：O(n_classes * n_clusters)

---

### completeness_score

衡量一个类的所有成员是否被分配到同一聚类。

::: source_code file="scl/kernel/metrics.hpp" symbol="completeness_score" collapsed
:::

**复杂度**
- 时间：O(n + n_classes * n_clusters)
- 空间：O(n_classes * n_clusters)

---

### purity_score

正确分配样本的比例（每个聚类中的多数类）。

::: source_code file="scl/kernel/metrics.hpp" symbol="purity_score" collapsed
:::

**复杂度**
- 时间：O(n + n_classes * n_clusters)
- 空间：O(n_classes * n_clusters)

---

### mean_lisi

计算所有细胞的平均 LISI 分数。

::: source_code file="scl/kernel/metrics.hpp" symbol="mean_lisi" collapsed
:::

**复杂度**
- 时间：O(n * k)
- 空间：O(n) 用于中间分数

---

### mean_batch_entropy

计算所有细胞的平均批次熵。

::: source_code file="scl/kernel/metrics.hpp" symbol="mean_batch_entropy" collapsed
:::

**复杂度**
- 时间：O(n * k)
- 空间：O(n) 用于中间分数

---

## 配置

命名空间 `scl::kernel::metrics::config` 提供配置常量：

- `EPSILON = 1e-10`：数值稳定性常数
- `LOG2_E = 1.4426950408889634`：log2(e) 用于熵计算
- `PARALLEL_THRESHOLD = 256`：并行处理的最小大小

## 注意事项

- **指标解释**：不同的指标强调聚类质量的不同方面。使用多个指标进行全面评估。
- **批次整合**：批次熵和 LISI 专门设计用于评估批次校正和整合质量。
- **计算成本**：某些指标（如 silhouette_score）对于大型数据集可能很昂贵。对于非常大的数据集，请考虑采样。
- **标签编码**：所有基于标签的指标假设非负整数标签，但不需要连续标记。

## 相关内容

- [Louvain 聚类](./louvain) - 社区检测算法
- [Leiden 聚类](./leiden) - 替代聚类算法
- [近邻](./neighbors) - 用于连通性指标的 KNN 图构建
