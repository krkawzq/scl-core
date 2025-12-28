# 亚群分析

用于亚群识别的聚类细化和稳定性评估。

## 概览

亚群分析内核提供：

- **递归子聚类** - 分层聚类细化
- **聚类稳定性** - 基于 bootstrap 的稳定性评估
- **质量控制** - 识别稳健与不稳定的聚类
- **细粒度分析** - 在聚类内发现亚群

## 递归子聚类

### recursive_subclustering

在聚类内执行递归子聚类：

```cpp
#include "scl/kernel/subpopulation.hpp"

Sparse<Real, true> expression = /* ... */;      // 表达矩阵 [n_cells x n_genes]
Array<Index> cluster_labels = /* ... */;        // 初始聚类标签 [n_cells]
Index n_cells = expression.rows();
Array<Index> subcluster_labels(n_cells);        // 预分配输出

// 标准子聚类（max_depth=3, min_size=10）
scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells, subcluster_labels);

// 自定义参数
scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells, subcluster_labels,
    max_depth = 4,                              // 更深层次
    min_size = 20                               // 更大的最小聚类大小
);
```

**参数：**
- `expression`: 表达矩阵（细胞 × 基因，CSR 格式）
- `cluster_labels`: 初始聚类标签，大小 = n_cells
- `n_cells`: 细胞数量
- `subcluster_labels`: 输出子聚类标签，必须预分配，大小 = n_cells
- `max_depth`: 最大递归深度（默认：3）
- `min_size`: 分割的最小聚类大小（默认：10）

**后置条件：**
- `subcluster_labels` 包含细化的子聚类分配
- 子聚类是分层的（深度由标签编码表示）
- 当适当时，原始聚类被分割为亚群

**算法：**
递归分层聚类：
1. 对于每个聚类：
   - 如果大小 < min_size：保留为叶节点
   - 否则：应用聚类算法进行细分
   - 递归处理每个子聚类
2. 继续直到达到 max_depth 或聚类太小

**复杂度：**
- 时间：O(max_depth * n_cells * log(n_cells)) 每层
- 空间：O(n_cells) 辅助空间用于标签和工作空间

**线程安全：**
- 不安全 - 带共享状态的递归算法

**用例：**
- 细粒度细胞类型识别
- 分层聚类分析
- 亚群发现
- 注释的聚类细化

## 聚类稳定性

### cluster_stability

使用 bootstrap 重采样评估聚类稳定性：

```cpp
Sparse<Real, true> expression = /* ... */;
Array<Index> cluster_labels = /* ... */;        // 聚类标签 [n_cells]
Index n_clusters = /* 唯一聚类数量 */;
Array<Real> stability_scores(n_clusters);       // 预分配输出

// 标准稳定性评估（100 次 bootstrap 迭代）
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability_scores);

// 自定义 bootstrap 迭代次数
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability_scores,
    n_bootstrap = 200,                          // 更多迭代
    seed = 12345                                // 随机种子
);
```

**参数：**
- `expression`: 表达矩阵（细胞 × 基因，CSR 格式）
- `cluster_labels`: 聚类标签，大小 = n_cells
- `n_cells`: 细胞数量
- `stability_scores`: 输出稳定性得分，必须预分配，大小 = n_clusters
- `n_bootstrap`: Bootstrap 迭代次数（默认：100）
- `seed`: 随机种子用于可重现性（默认：42）

**后置条件：**
- `stability_scores[c]` 包含聚类 c 的稳定性得分
- 得分通常在范围 [0, 1]，更高 = 更稳定
- 稳定聚类对数据扰动具有鲁棒性

**算法：**
Bootstrap 重采样方法：
1. 对于每次 bootstrap 迭代：
   - 有放回采样细胞
   - 重新聚类采样细胞
   - 计算与原始聚类的重叠
2. 跨迭代聚合重叠得分
3. 稳定性 = 平均一致性

**复杂度：**
- 时间：O(n_bootstrap * n_cells * log(n_cells)) - 由重新聚类主导
- 空间：O(n_cells) 辅助空间用于 bootstrap 样本和标签

**线程安全：**
- 安全 - 跨 bootstrap 迭代并行化
- 每次迭代是独立的

**用例：**
- 聚类结果的质量控制
- 识别稳健与不稳定的聚类
- 指导聚类细化决策
- 验证聚类参数

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CLUSTER_SIZE = 10;
    constexpr Size DEFAULT_K = 5;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Size DEFAULT_BOOTSTRAP = 100;
}
```

**最小聚类大小：**
- 小于 `MIN_CLUSTER_SIZE` 的聚类不会被分割
- 防止过度碎片化
- 根据数据集大小调整

**Bootstrap 迭代：**
- 更多迭代 = 更可靠的稳定性得分
- 默认 100 通常足够
- 对高精度要求增加迭代次数

## 示例

### 分层子聚类

```cpp
#include "scl/kernel/subpopulation.hpp"

// 初始聚类
Sparse<Real, true> expression = /* ... */;
Array<Index> initial_labels = /* ... */;  // 来自 Leiden/Louvain

// 用子聚类细化
Index n_cells = expression.rows();
Array<Index> refined_labels(n_cells);

scl::kernel::subpopulation::recursive_subclustering(
    expression, initial_labels, n_cells, refined_labels,
    max_depth = 3, min_size = 15);

// refined_labels 现在包含分层子聚类
```

### 稳定性评估

```cpp
// 评估聚类结果的稳定性
Array<Index> cluster_labels = /* ... */;
Index n_clusters = *std::max_element(cluster_labels.begin(),
                                     cluster_labels.end()) + 1;

Array<Real> stability(n_clusters);
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability,
    n_bootstrap = 200);

// 过滤不稳定聚类
std::vector<Index> stable_clusters;
for (Index c = 0; c < n_clusters; ++c) {
    if (stability[c] > 0.7) {  // 稳定性阈值
        stable_clusters.push_back(c);
    }
}

std::cout << "发现 " << stable_clusters.size()
          << " 个稳定聚类，共 " << n_clusters << " 个\n";
```

### 组合工作流

```cpp
// 1. 初始聚类
Array<Index> initial_labels(n_cells);
// ... 执行初始聚类 ...

// 2. 评估稳定性
Index n_clusters = /* ... */;
Array<Real> stability(n_clusters);
scl::kernel::subpopulation::cluster_stability(
    expression, initial_labels, n_cells, stability);

// 3. 细化稳定聚类
Array<Index> refined_labels(n_cells);
scl::kernel::subpopulation::recursive_subclustering(
    expression, initial_labels, n_cells, refined_labels);

// 4. 按稳定性过滤
// 使用稳定性得分识别可靠的子聚类
```

## 性能考虑

### 递归子聚类

- 计算成本随深度扩展
- 大聚类需要更多计算
- 对于非常大的数据集，考虑限制 max_depth

### Bootstrap 稳定性

- 跨 bootstrap 迭代并行化
- 每次迭代需要完全重新聚类
- 总时间：O(n_bootstrap * clustering_time)
- 对于初步探索，考虑减少 n_bootstrap

---

::: tip 稳定性阈值
使用稳定性得分指导下游分析：专注于稳定聚类进行标记识别和注释，谨慎对待不稳定聚类。
:::

