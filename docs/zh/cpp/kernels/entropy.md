# entropy.hpp

> scl/kernel/entropy.hpp · 稀疏数据分析的信息论度量

## 概述

本文件提供用于分析稀疏单细胞数据的信息论度量，包括熵、互信息和特征选择方法。所有操作都针对稀疏矩阵进行了优化，并支持并行处理。

**头文件**: `#include "scl/kernel/entropy.hpp"`

主要特性：
- Shannon 熵计算
- Kullback-Leibler 和 Jensen-Shannon 散度
- 互信息和归一化变体
- 通过 MI 和 mRMR 进行特征选择
- 连续数据的离散化方法

---

## 主要 API

### count_entropy

::: source_code file="scl/kernel/entropy.hpp" symbol="count_entropy" collapsed
:::

**算法说明**

从计数数组计算 Shannon 熵：

1. 计算总计数：`total = sum(counts)`
2. 对于每个非零计数：
   - 计算概率：`p_i = counts[i] / total`
   - 累加：`entropy -= p_i * log(p_i)`
3. 返回熵 H = -sum(p_i * log(p_i))
4. 如果 `use_log2 = true` 使用以 2 为底的对数，否则使用自然对数

**边界条件**

- **总计数 = 0**：返回 0.0
- **单个非零计数**：返回 0.0（无不确定性）
- **均匀分布**：返回最大熵 = log(n)
- **全零**：返回 0.0

**数据保证（前置条件）**

- 所有计数 >= 0
- `n > 0`

**复杂度分析**

- **时间**：O(n)
- **空间**：O(1) 辅助

**示例**

```cpp
#include "scl/kernel/entropy.hpp"

Real counts[] = {10, 20, 30, 40};
Size n = 4;

Real entropy = scl::kernel::entropy::count_entropy(counts, n, false);

// entropy = -sum((count/total) * log(count/total))
```

---

### row_entropy

::: source_code file="scl/kernel/entropy.hpp" symbol="row_entropy" collapsed
:::

**算法说明**

计算稀疏矩阵每一行的 Shannon 熵：

1. 对于每一行 i（并行）：
   - 提取第 i 行中的非零值
   - 计算行和：`row_sum = sum(row_i)`
   - 对于每个非零值：
     - 概率：`p_j = value / row_sum`
     - 累加：`entropy[i] -= p_j * log(p_j)`
2. 如果 `normalize = true`：除以最大熵（log(n_cols)）
3. 如果归一化，返回 [0, 1] 范围内的熵值

**边界条件**

- **空行**：熵 = 0.0
- **单个非零**：熵 = 0.0
- **均匀行**：最大熵
- **全零**：熵 = 0.0

**数据保证（前置条件）**

- `entropies.len >= X.rows()`
- X 必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**：O(nnz) - 与非零元素数成正比
- **空间**：O(1) 辅助每行

**示例**

```cpp
scl::Sparse<Real, true> X = /* 表达矩阵 */;
scl::Array<Real> entropies(X.rows());

scl::kernel::entropy::row_entropy(X, entropies, false, false);

// entropies[i] = 第 i 行的熵（基因表达分布）
```

---

### kl_divergence

::: source_code file="scl/kernel/entropy.hpp" symbol="kl_divergence" collapsed
:::

**算法说明**

计算两个概率分布之间的 Kullback-Leibler 散度：

1. 对于每个元素 i：
   - 如果 `p[i] > 0` 且 `q[i] > 0`：累加 `p[i] * log(p[i] / q[i])`
   - 如果 `p[i] > 0` 且 `q[i] = 0`：返回大值（无穷大）
   - 如果 `p[i] = 0`：跳过（0 * log(0) = 0）
2. 返回 KL(p || q) = sum(p_i * log(p_i / q_i))
3. 非对称：KL(p||q) != KL(q||p)

**边界条件**

- **q[i] = 0 且 p[i] > 0**：返回大值（散度未定义）
- **p = q**：返回 0.0
- **p 全零**：返回 0.0
- **q 全零且 p 不全零**：返回大值

**数据保证（前置条件）**

- `p.len == q.len`
- 两个数组都表示概率分布（和为 1.0）
- 所有值 >= 0

**复杂度分析**

- **时间**：O(n)
- **空间**：O(1) 辅助

**示例**

```cpp
scl::Array<Real> p = {0.5, 0.3, 0.2};  // 分布 1
scl::Array<Real> q = {0.4, 0.4, 0.2};  // 分布 2

Real kl = scl::kernel::entropy::kl_divergence(p, q, false);

// kl = KL(p || q) = sum(p_i * log(p_i / q_i))
```

---

### js_divergence

::: source_code file="scl/kernel/entropy.hpp" symbol="js_divergence" collapsed
:::

**算法说明**

计算两个概率分布之间的 Jensen-Shannon 散度：

1. 计算混合：`m = (p + q) / 2`
2. 计算 JS = 0.5 * KL(p || m) + 0.5 * KL(q || m)
3. 总是有限且对称：JS(p||q) = JS(q||p)
4. 有界：如果使用以 2 为底的对数，JS 在 [0, 1] 范围内

**边界条件**

- **p = q**：返回 0.0
- **p 和 q 不相交**：返回最大 JS
- **总是有限**：与 KL 不同，从不返回无穷大

**数据保证（前置条件）**

- `p.len == q.len`
- 两个数组都表示概率分布
- 所有值 >= 0

**复杂度分析**

- **时间**：O(n)
- **空间**：O(1) 辅助

**示例**

```cpp
scl::Array<Real> p = {0.5, 0.3, 0.2};
scl::Array<Real> q = {0.4, 0.4, 0.2};

Real js = scl::kernel::entropy::js_divergence(p, q, false);

// js = 0.5 * KL(p || m) + 0.5 * KL(q || m) 其中 m = (p+q)/2
// 总是有限且对称
```

---

### mutual_information

::: source_code file="scl/kernel/entropy.hpp" symbol="mutual_information" collapsed
:::

**算法说明**

从分箱数据计算互信息 I(X; Y)：

1. 计算 2D 直方图：`counts[i][j]` = 在箱 (i, j) 中的样本数
2. 计算联合熵：H(X, Y) = -sum(p_ij * log(p_ij))
3. 计算边际熵：H(X) 和 H(Y)
4. 返回 MI = H(X) + H(Y) - H(X, Y)
5. 总是 >= 0，如果 X 和 Y 独立则等于 0

**边界条件**

- **X 和 Y 独立**：MI = 0.0
- **X = Y**：MI = H(X)（最大值）
- **无样本**：返回 0.0
- **所有样本在一个箱中**：MI = 0.0

**数据保证（前置条件）**

- 所有箱索引有效：`x_binned[i] in [0, n_bins_x)`，`y_binned[i] in [0, n_bins_y)`
- `n > 0`

**复杂度分析**

- **时间**：O(n + n_bins_x * n_bins_y)
- **空间**：O(n_bins_x * n_bins_y) 辅助

**示例**

```cpp
// 首先离散化连续值
scl::Array<Index> x_binned = /* 分箱的 x 值 */;
scl::Array<Index> y_binned = /* 分箱的 y 值 */;

Real mi = scl::kernel::entropy::mutual_information(
    x_binned.data(), y_binned.data(), n,
    n_bins_x, n_bins_y, false
);

// mi = I(X; Y) = H(X) + H(Y) - H(X, Y)
// 更高的 MI 表示更强的依赖性
```

---

### normalized_mi

::: source_code file="scl/kernel/entropy.hpp" symbol="normalized_mi" collapsed
:::

**算法说明**

计算两个标记之间的归一化互信息：

1. 从标记计算互信息 I(X; Y)
2. 计算边际熵 H(X) 和 H(Y)
3. 返回 NMI = 2 * I(X; Y) / (H(X) + H(Y))
4. 值在 [0, 1] 范围内，其中 1 表示完美一致
5. 对称：NMI(X, Y) = NMI(Y, X)

**边界条件**

- **完美一致**：NMI = 1.0
- **独立标记**：NMI = 0.0
- **一个标记只有一个聚类**：NMI = 0.0（H = 0）

**数据保证（前置条件）**

- `labels1.len == labels2.len`
- 所有标记索引有效：`labels1[i] in [0, n_clusters1)`，`labels2[i] in [0, n_clusters2)`

**复杂度分析**

- **时间**：O(n + n_clusters1 * n_clusters2)
- **空间**：O(n_clusters1 * n_clusters2) 辅助

**示例**

```cpp
scl::Array<Index> labels1 = /* 第一个聚类 */;
scl::Array<Index> labels2 = /* 第二个聚类 */;

Real nmi = scl::kernel::entropy::normalized_mi(
    labels1, labels2, n_clusters1, n_clusters2
);

// nmi 在 [0, 1] 范围内，越高 = 一致性越好
```

---

### select_features_mi

::: source_code file="scl/kernel/entropy.hpp" symbol="select_features_mi" collapsed
:::

**算法说明**

使用与目标的互信息选择顶级特征：

1. 对于每个特征 f：
   - 将特征值离散化为 n_bins
   - 计算离散化特征与目标标记之间的 MI
   - 存储 MI 分数
2. 按 MI 分数对特征进行排序（降序）
3. 选择前 n_to_select 个特征
4. 返回选定的特征和所有 MI 分数

**边界条件**

- **n_to_select = 0**：返回空选择
- **n_to_select >= n_features**：返回所有特征
- **常量特征**：MI = 0.0
- **完美相关**：MI = H(target)

**数据保证（前置条件）**

- `selected_features` 容量 >= n_to_select
- `mi_scores` 容量 >= n_features
- `target` 包含有效的标记索引
- X 必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**：O(n_features * n_samples * log(nnz_per_sample))
- **空间**：O(n_samples) 辅助

**示例**

```cpp
scl::Sparse<Real, true> X = /* 特征矩阵 */;
scl::Array<Index> target = /* 目标标记 */;
Index n_to_select = 100;

scl::Array<Index> selected_features(n_to_select);
scl::Array<Real> mi_scores(n_features);

scl::kernel::entropy::select_features_mi(
    X, target, n_features, n_to_select,
    selected_features, mi_scores, 10  // n_bins
);

// selected_features 包含按 MI 排序的前 100 个特征
// mi_scores 包含所有特征的 MI 分数
```

---

### mrmr_selection

::: source_code file="scl/kernel/entropy.hpp" symbol="mrmr_selection" collapsed
:::

**算法说明**

使用最小冗余最大相关性（mRMR）选择特征：

1. 初始化：选择与目标具有最高 MI 的特征
2. 对于每个剩余选择：
   - 对于每个未选择的特征 f：
     - 计算相关性：MI(f, target)
     - 计算冗余：mean(MI(f, selected_features))
     - 分数：相关性 - 冗余
   - 选择分数最高的特征
3. 贪心选择平衡相关性和冗余
4. 按选择顺序返回选定的特征

**边界条件**

- **n_to_select = 0**：返回空选择
- **n_to_select = 1**：返回单个最佳特征
- **所有特征冗余**：可能选择少于请求的数量

**数据保证（前置条件）**

- `selected_features` 容量 >= n_to_select
- `target` 包含有效的标记索引
- X 必须是有效的 CSR 或 CSC 格式

**复杂度分析**

- **时间**：O(n_to_select * n_features * n_samples)
- **空间**：O(n_features * n_samples) 辅助

**示例**

```cpp
scl::Array<Index> selected_features(n_to_select);

scl::kernel::entropy::mrmr_selection(
    X, target, n_features, n_to_select,
    selected_features, 10  // n_bins
);

// selected_features 包含 mRMR 选择的特征
// 特征最大化相关性并最小化冗余
```

---

## 工具函数

### discretize_equal_width

将连续值离散化为等宽箱。

::: source_code file="scl/kernel/entropy.hpp" symbol="discretize_equal_width" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：O(1) 辅助

---

### discretize_equal_frequency

将连续值离散化为等频箱。

::: source_code file="scl/kernel/entropy.hpp" symbol="discretize_equal_frequency" collapsed
:::

**复杂度**

- 时间：O(n log n) 用于排序
- 空间：O(n) 辅助

---

### joint_entropy

从分箱数据计算联合熵 H(X, Y)。

::: source_code file="scl/kernel/entropy.hpp" symbol="joint_entropy" collapsed
:::

**复杂度**

- 时间：O(n + n_bins_x * n_bins_y)
- 空间：O(n_bins_x * n_bins_y) 辅助

---

### conditional_entropy

从分箱数据计算条件熵 H(Y | X)。

::: source_code file="scl/kernel/entropy.hpp" symbol="conditional_entropy" collapsed
:::

**复杂度**

- 时间：O(n + n_bins_x * n_bins_y)
- 空间：O(n_bins_x * n_bins_y) 辅助

---

### adjusted_mi

计算调整后的互信息（机会校正）。

::: source_code file="scl/kernel/entropy.hpp" symbol="adjusted_mi" collapsed
:::

**复杂度**

- 时间：O(n + n_clusters1 * n_clusters2)
- 空间：O(n_clusters1 * n_clusters2) 辅助

---

## 注意事项

- 熵需要概率分布 - 确保归一化
- 在计算 MI 之前，对连续数据进行离散化是必要的
- mRMR 比简单的 MI 更适合特征选择（减少冗余）
- 归一化 MI 对于比较不同大小的聚类很有用

## 相关内容

- [特征选择模块](./feature) - 其他特征选择方法
- [统计模块](../math/statistics) - 统计度量
