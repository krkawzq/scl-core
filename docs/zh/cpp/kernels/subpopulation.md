# subpopulation.hpp

> scl/kernel/subpopulation.hpp · 亚群分析和聚类细化

## 概述

本文件提供单细胞数据中亚群分析和聚类细化的函数。包括递归子聚类和使用自举重采样的聚类稳定性评估。

**头文件**: `#include "scl/kernel/subpopulation.hpp"`

主要特性：
- 在现有聚类内进行递归子聚类
- 通过自举评估聚类稳定性
- 分层聚类细化

---

## 主要 API

### recursive_subclustering

::: source_code file="scl/kernel/subpopulation.hpp" symbol="recursive_subclustering" collapsed
:::

**算法说明**

在聚类内执行递归子聚类：

1. 对于每个初始聚类：
   - 如果聚类大小 >= min_size 且深度 < max_depth：
     - 应用聚类算法（例如，k-means、Leiden）细分聚类
     - 递归应用于每个子聚类
     - 分配分层标签：`subcluster_labels[i] = parent_cluster * base + subcluster_id`
   - 否则：保持原始聚类标签
2. 构建最多 max_depth 层的分层聚类树
3. 对独立聚类使用并行处理

**边界条件**

- **max_depth = 0**：返回原始聚类标签不变
- **min_size 太大**：没有聚类被细分
- **空聚类**：在递归中跳过
- **单细胞聚类**：无法细分

**数据保证（前置条件）**

- `subcluster_labels` 容量 >= n_cells
- `cluster_labels` 包含有效的聚类索引
- 表达矩阵必须是有效的 CSR 格式
- `min_size >= 2` 以进行有意义的细分

**复杂度分析**

- **时间**：O(max_depth * n_cells * log(n_cells)) - 每层聚类
- **空间**：O(n_cells) 辅助空间

**示例**

```cpp
#include "scl/kernel/subpopulation.hpp"

scl::Sparse<Real, true> expression = /* 表达矩阵 */;
scl::Array<Index> cluster_labels = /* 初始聚类 */;
scl::Array<Index> subcluster_labels(n_cells);

scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells,
    subcluster_labels, 3,  // max_depth
    10                     // min_size
);

// subcluster_labels 包含分层子聚类分配
```

---

### cluster_stability

::: source_code file="scl/kernel/subpopulation.hpp" symbol="cluster_stability" collapsed
:::

**算法说明**

使用自举重采样评估聚类稳定性：

1. 对于每次自举迭代（并行）：
   - 有放回地采样细胞（自举样本）
   - 对自举样本重新聚类
   - 计算与原始聚类的聚类重叠
   - 累加稳定性指标
2. 对于每个聚类：
   - 稳定性分数 = 跨自举迭代的平均 Jaccard 相似度
   - 更高的分数表示更稳定的聚类
3. 返回 [0, 1] 范围内的稳定性分数

**边界条件**

- **n_bootstrap = 0**：返回零稳定性分数
- **小聚类**：由于采样方差可能具有低稳定性
- **完美稳定性**：所有自举迭代产生相同的聚类

**数据保证（前置条件）**

- `stability_scores` 容量 >= n_clusters
- `cluster_labels` 包含有效的聚类索引
- 表达矩阵必须是有效的 CSR 格式
- 随机种子确保可重现性

**复杂度分析**

- **时间**：O(n_bootstrap * n_cells * log(n_cells)) - 每次自举的聚类
- **空间**：O(n_cells) 每线程辅助空间

**示例**

```cpp
scl::Array<Real> stability_scores(n_clusters);

scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells,
    stability_scores,
    100,  // n_bootstrap
    42    // 种子
);

// stability_scores[c] 包含聚类 c 的稳定性分数
// 更高的分数（接近 1.0）表示更稳定的聚类
```

---

## 配置

默认参数在 `scl::kernel::subpopulation::config` 中定义：

- `EPSILON = 1e-10`：数值容差
- `MIN_CLUSTER_SIZE = 10`：细分的最小聚类大小
- `DEFAULT_K = 5`：k-means 子聚类的默认 k
- `MAX_ITERATIONS = 100`：聚类算法的最大迭代次数
- `DEFAULT_BOOTSTRAP = 100`：默认自举迭代次数

---

## 注意事项

- 递归子聚类构建分层聚类树
- 聚类稳定性有助于识别稳健与不稳定的聚类
- 自举重采样为聚类分配提供统计置信度
- 稳定性分数可用于过滤不可靠的聚类

## 相关内容

- [聚类模块](./leiden) - 用于聚类算法
- [指标模块](./metrics) - 用于聚类质量指标
