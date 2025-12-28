# doublet.hpp

> scl/kernel/doublet.hpp · 单细胞数据的双联体检测内核

## 概述

本文件为单细胞 RNA-seq 数据提供高性能的双联体检测算法。它使用 k 最近邻分析在模拟双联体上实现 Scrublet 风格和 DoubletFinder 风格的方法。

**头文件**: `#include "scl/kernel/doublet.hpp"`

主要特性：
- 通过平均细胞对进行合成双联体模拟
- 基于 k-NN 的双联体评分
- 多种检测方法（Scrublet、DoubletFinder、Hybrid）
- 自动阈值估计
- 聚类感知的双联体分类

---

## 主要 API

### simulate_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="simulate_doublets" collapsed
:::

**算法说明**

通过平均随机细胞对来模拟合成双联体：

1. 对于每个双联体 d（并行）：
   - 随机选择两个不同的细胞（cell1, cell2）
   - 对于任一细胞中表达的每个基因：
     - 设置 `profile[gene] = 0.5 * value_cell1 + 0.5 * value_cell2`
   - 将配置文件存储在 `doublet_profiles[d * n_genes : (d+1) * n_genes]`
2. 每个双联体配置文件是两个随机选择的细胞的平均值
3. 对双联体进行并行处理以提高效率

**边界条件**

- **n_doublets = 0**：立即返回，不生成配置文件
- **n_cells < 2**：无法生成有意义的双联体
- **空矩阵**：所有配置文件保持为零
- **选择相同的细胞对**：仍然有效（自双联体）

**数据保证（前置条件）**

- X 必须是 CSR 格式（细胞 x 基因）
- `doublet_profiles` 必须预分配 `n_doublets * n_genes` 个元素
- `n_cells >= 2` 以进行有意义的模拟
- 随机种子确保可重现性

**复杂度分析**

- **时间**：O(n_doublets * avg_nnz_per_cell)，其中 avg_nnz 是每个细胞的平均非零元素数
- **空间**：O(n_doublets * n_genes) 用于输出配置文件

**示例**

```cpp
#include "scl/kernel/doublet.hpp"

scl::Sparse<Real, true> X = /* 表达矩阵 [n_cells x n_genes] */;
Index n_doublets = 2 * n_cells;  // 自动：2x 细胞数

scl::Array<Real> doublet_profiles(n_doublets * n_genes);

scl::kernel::doublet::simulate_doublets(
    X, n_cells, n_genes, n_doublets, 
    doublet_profiles.data(), 42  // 种子
);

// doublet_profiles 现在包含 n_doublets 个合成配置文件
```

---

### compute_knn_doublet_scores

::: source_code file="scl/kernel/doublet.hpp" symbol="compute_knn_doublet_scores" collapsed
:::

**算法说明**

通过 k-NN 对观察到的和模拟的细胞计算双联体分数：

1. 对于每个细胞 i（并行）：
   - 将细胞 i 转换为密集向量
   - 计算到以下项的平方欧几里得距离：
     - 所有观察到的细胞（不包括自身）
     - 所有模拟的双联体配置文件
   - 使用基于堆的选择找到 k 个最近邻（O(n log k)）
   - 计算是模拟双联体的邻居比例
   - 分数 = count / k（范围从 0 到 1）
2. 更高的分数表示细胞更类似于模拟的双联体

**边界条件**

- **k_neighbors > total_cells**：使用所有可用邻居
- **无模拟双联体**：所有分数为 0
- **k_neighbors = 0**：未定义行为（应该 > 0）
- **相同细胞**：可能有完美的双联体邻居

**数据保证（前置条件）**

- X 必须是 CSR 格式
- `doublet_profiles` 包含 n_doublets 个配置文件
- `doublet_scores.len >= n_cells`
- `k_neighbors > 0` 且 `k_neighbors <= n_cells + n_doublets`

**复杂度分析**

- **时间**：O(n_cells * (n_cells + n_doublets) * n_genes)
- **空间**：O(n_threads * (n_genes + n_total + k_neighbors)) 工作空间

**示例**

```cpp
Index k_neighbors = 30;
scl::Array<Real> doublet_scores(n_cells);

scl::kernel::doublet::compute_knn_doublet_scores(
    X, n_cells, n_genes,
    doublet_profiles.data(), n_doublets,
    k_neighbors, doublet_scores
);

// doublet_scores[i] = k 个最近邻中是模拟双联体的比例
// (0 = 单联体, 1 = 双联体)
```

---

### scrublet_scores

::: source_code file="scl/kernel/doublet.hpp" symbol="scrublet_scores" collapsed
:::

**算法说明**

完整的 Scrublet 风格双联体检测流程：

1. 模拟合成双联体（如果 n_simulated = 0，自动：2x n_cells）
2. 计算所有细胞的 k-NN 双联体分数
3. 返回准备用于阈值化的分数
4. 在一个调用中组合模拟和评分

**边界条件**

- **n_simulated = 0**：自动使用 2 * n_cells
- **小 n_cells**：可能邻居不足
- **空矩阵**：所有分数为 0

**数据保证（前置条件）**

- X 必须是 CSR 格式
- `scores.len >= n_cells`
- 随机种子确保可重现性

**复杂度分析**

- **时间**：O(n_cells * (n_cells + n_simulated) * n_genes)
- **空间**：O(n_simulated * n_genes) 用于双联体配置文件

**示例**

```cpp
scl::Array<Real> scores(n_cells);

scl::kernel::doublet::scrublet_scores(
    X, n_cells, n_genes, scores,
    0,      // n_simulated: 0 = 自动 (2x n_cells)
    30,     // k_neighbors
    42      // 种子
);

// scores[i] 包含细胞 i 的 Scrublet 双联体分数
```

---

### detect_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="detect_doublets" collapsed
:::

**算法说明**

完整的双联体检测流程（模拟、评分、阈值、调用）：

1. 根据方法模拟合成双联体
2. 使用所选方法计算双联体分数
3. 从预期双联体率估计阈值
4. 调用双联体：`is_doublet[i] = (scores[i] > threshold)`
5. 返回检测到的双联体总数

**边界条件**

- **expected_rate = 0**：不调用双联体
- **expected_rate = 1**：所有细胞被调用为双联体
- **Method = Hybrid**：组合多个评分信号

**数据保证（前置条件）**

- X 必须是 CSR 格式
- `scores.len >= n_cells`
- `is_doublet.len >= n_cells`
- `expected_rate` 在 (0, 1) 范围内

**复杂度分析**

- **时间**：O(n_cells * (n_cells + n_simulated) * n_genes)
- **空间**：O(n_simulated * n_genes) 用于配置文件

**示例**

```cpp
scl::Array<Real> scores(n_cells);
scl::Array<bool> is_doublet(n_cells);

Index n_detected = scl::kernel::doublet::detect_doublets(
    X, n_cells, n_genes,
    scores, is_doublet,
    scl::kernel::doublet::DoubletMethod::Scrublet,
    0.06,   // expected_doublet_rate
    30,     // k_neighbors
    42      // 种子
);

// is_doublet[i] = true 如果细胞 i 是双联体
// n_detected = 找到的双联体总数
```

---

### estimate_threshold

::: source_code file="scl/kernel/doublet.hpp" symbol="estimate_threshold" collapsed
:::

**算法说明**

从预期双联体率估计分数阈值：

1. 将分数复制到临时缓冲区
2. 使用 SIMD 优化的排序对分数进行排序（O(n log n)）
3. 找到百分位数：`index = (1 - expected_rate) * n`
4. 返回 `scores[index]` 作为阈值

**边界条件**

- **expected_rate = 0**：返回最大分数（无双联体）
- **expected_rate = 1**：返回最小分数（所有双联体）
- **空分数**：未定义行为

**数据保证（前置条件）**

- `scores.len > 0`
- `expected_rate` 在 (0, 1) 范围内

**复杂度分析**

- **时间**：O(n log n) 用于排序
- **空间**：O(n) 用于排序副本

**示例**

```cpp
Real threshold = scl::kernel::doublet::estimate_threshold(
    scores, 0.06  // 6% 预期双联体率
);

// threshold 是第 94 百分位数的分数
// 大约 6% 的细胞将具有分数 > threshold
```

---

### call_doublets

::: source_code file="scl/kernel/doublet.hpp" symbol="call_doublets" collapsed
:::

**算法说明**

根据分数阈值调用双联体：

1. 对于每个细胞 i：
   - `is_doublet[i] = (scores[i] > threshold)`
2. 计算双联体总数
3. 返回计数

**边界条件**

- **threshold = infinity**：不调用双联体
- **threshold = -infinity**：所有细胞被调用为双联体
- **空分数**：返回 0

**数据保证（前置条件）**

- `is_doublet.len >= scores.len`

**复杂度分析**

- **时间**：O(n)
- **空间**：O(1)

**示例**

```cpp
scl::Array<bool> is_doublet(n_cells);

Index n_doublets = scl::kernel::doublet::call_doublets(
    scores, threshold, is_doublet
);

// is_doublet[i] = true 如果 scores[i] > threshold
// n_doublets = true 值的计数
```

---

## 工具函数

### detect_bimodal_threshold

使用双峰分布检测阈值（直方图谷值检测）。

::: source_code file="scl/kernel/doublet.hpp" symbol="detect_bimodal_threshold" collapsed
:::

**复杂度**

- 时间：O(n + n_bins)
- 空间：O(n_bins)

---

### doublet_score_stats

计算双联体分数的统计量（均值、标准差、中位数）。

::: source_code file="scl/kernel/doublet.hpp" symbol="doublet_score_stats" collapsed
:::

**复杂度**

- 时间：O(n log n) 用于中位数
- 空间：O(n) 用于排序副本

---

### expected_doublets

根据细胞计数和比率计算预期双联体数。

::: source_code file="scl/kernel/doublet.hpp" symbol="expected_doublets" collapsed
:::

**复杂度**

- 时间：O(1)
- 空间：O(1)

---

## 注意事项

- 双联体模拟使用随机采样 - 结果随种子变化
- k-NN 计算是最昂贵的步骤 - 对于大型数据集考虑 PCA 降维
- 阈值估计假设分数遵循预期分布
- Scrublet 方法在实践中最常用

## 相关内容

- [邻居模块](./neighbors) - 用于 k-NN 图构建
- [PCA/投影](./projection) - 用于评分前的降维
