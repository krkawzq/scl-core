# 熵

用于稀疏数据分析的信息论度量，包括熵、散度和互信息。

## 概述

熵模块提供：

- **Shannon 熵** - 从计数或稀疏矩阵计算信息熵
- **KL 散度** - 分布之间的 Kullback-Leibler 散度
- **JS 散度** - Jensen-Shannon 散度（对称、有限）
- **互信息** - 变量之间共享的信息
- **特征选择** - 基于 MI 和 mRMR 的特征选择
- **离散化** - 等宽和等频分箱

## 基本熵

### count_entropy

从计数数组计算 Shannon 熵：

```cpp
#include "scl/kernel/entropy.hpp"

const Real* counts = /* ... */;  // 计数值 [n]

Real entropy = scl::kernel::entropy::count_entropy(
    counts,
    n,
    false  // use_log2 = false（使用自然对数）
);
```

**参数：**
- `counts`: 计数值 [n]
- `n`: 元素数量
- `use_log2`: 如果为 true，使用以 2 为底的对数

**返回：** 熵 H = -sum(p_i * log(p_i))

**后置条件：**
- 如果总计数为零，返回 0
- 所有计数 >= 0

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 分布熵
- 多样性度量
- 信息内容

### row_entropy

计算稀疏矩阵每行的 Shannon 熵：

```cpp
Sparse<Real, true> X = /* ... */;
Array<Real> entropies(X.rows());

scl::kernel::entropy::row_entropy(
    X,
    entropies,
    false,  // normalize = false
    false   // use_log2 = false
);
```

**参数：**
- `X`: 稀疏矩阵（CSR 或 CSC）
- `entropies`: 输出的熵值 [n_rows]
- `normalize`: 如果为 true，按最大熵归一化
- `use_log2`: 如果为 true，使用以 2 为底的对数

**后置条件：**
- `entropies[i]` 包含行 i 的熵
- 如果 normalize=true，值在 [0, 1] 范围内

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间（每行）

**使用场景：**
- 细胞表达多样性
- 特征熵分析
- 行级信息内容

## 散度度量

### kl_divergence

计算两个概率分布之间的 Kullback-Leibler 散度：

```cpp
Array<const Real> p = /* ... */;  // 第一个分布 [n]
Array<const Real> q = /* ... */;  // 第二个分布 [n]

Real kl = scl::kernel::entropy::kl_divergence(
    p,
    q,
    false  // use_log2
);
```

**参数：**
- `p`: 第一个分布 [n]
- `q`: 第二个分布 [n]
- `use_log2`: 如果为 true，使用以 2 为底的对数

**返回：** KL(p || q) = sum(p_i * log(p_i / q_i))

**后置条件：**
- 如果 q_i = 0 且 p_i > 0，返回大值
- 两个数组都表示概率分布

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 分布比较
- 模型评估
- 信息增益

### js_divergence

计算两个概率分布之间的 Jensen-Shannon 散度：

```cpp
Real js = scl::kernel::entropy::js_divergence(
    p,
    q,
    false  // use_log2
);
```

**返回：** JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)，其中 m = (p+q)/2

**后置条件：**
- 始终有限且对称
- 使用 log2 时值在 [0, 1] 范围内

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 对称散度度量
- 当 KL 散度未定义时
- 距离度量

## 离散化

### discretize_equal_width

将连续值离散化为等宽分箱：

```cpp
const Real* values = /* ... */;
Index* binned = /* 分配 n */;

scl::kernel::entropy::discretize_equal_width(
    values,
    n,
    n_bins,  // 分箱数量
    binned
);
```

**参数：**
- `values`: 连续值 [n]
- `n`: 值数量
- `n_bins`: 分箱数量
- `binned`: 输出的分箱索引 [n]

**后置条件：**
- `binned[i]` 包含分箱索引，范围 [0, n_bins-1]
- 同一分箱中的所有值具有相同范围

**复杂度：**
- 时间：O(n)
- 空间：O(1) 辅助空间

**使用场景：**
- 连续到离散转换
- 直方图计算
- 熵估计

### discretize_equal_frequency

将连续值离散化为等频分箱：

```cpp
scl::kernel::entropy::discretize_equal_frequency(
    values,
    n,
    n_bins,
    binned
);
```

**后置条件：**
- `binned[i]` 包含分箱索引，范围 [0, n_bins-1]
- 每个分箱包含大约 n/n_bins 个值

**复杂度：**
- 时间：O(n log n) 用于排序
- 空间：O(n) 辅助空间

**使用场景：**
- 基于分位数的分箱
- 对异常值鲁棒
- 等样本大小分箱

## 联合和条件熵

### histogram_2d

从分箱数据计算 2D 直方图：

```cpp
const Index* x_binned = /* ... */;
const Index* y_binned = /* ... */;
Size* counts = /* 分配 n_bins_x * n_bins_y */;

scl::kernel::entropy::histogram_2d(
    x_binned,
    y_binned,
    n,
    n_bins_x,
    n_bins_y,
    counts
);
```

**参数：**
- `x_binned`: 分箱的 x 值 [n]
- `y_binned`: 分箱的 y 值 [n]
- `n`: 样本数量
- `n_bins_x`: x 分箱数量
- `n_bins_y`: y 分箱数量
- `counts`: 输出的直方图计数 [n_bins_x * n_bins_y]

**后置条件：**
- `counts[i * n_bins_y + j]` 包含分箱 (i, j) 的计数

**复杂度：**
- 时间：O(n)
- 空间：O(n_bins_x * n_bins_y) 辅助空间

**使用场景：**
- 联合分布估计
- 2D 直方图计算
- 列联表

### joint_entropy

从分箱数据计算联合熵 H(X, Y)：

```cpp
Real h_xy = scl::kernel::entropy::joint_entropy(
    x_binned,
    y_binned,
    n,
    n_bins_x,
    n_bins_y,
    false  // use_log2
);
```

**返回：** H(X, Y) = -sum(p_ij * log(p_ij))

**复杂度：**
- 时间：O(n + n_bins_x * n_bins_y)
- 空间：O(n_bins_x * n_bins_y) 辅助空间

**使用场景：**
- 联合信息内容
- 多变量熵
- 依赖性分析

### marginal_entropy

从分箱数据计算边际熵 H(X)：

```cpp
Real h_x = scl::kernel::entropy::marginal_entropy(
    binned,
    n,
    n_bins,
    false  // use_log2
);
```

**返回：** H(X) = -sum(p_i * log(p_i))

**复杂度：**
- 时间：O(n + n_bins)
- 空间：O(n_bins) 辅助空间

**使用场景：**
- 单变量熵
- 边际信息内容
- 单变量分析

### conditional_entropy

从分箱数据计算条件熵 H(Y | X)：

```cpp
Real h_y_given_x = scl::kernel::entropy::conditional_entropy(
    x_binned,
    y_binned,
    n,
    n_bins_x,
    n_bins_y,
    false  // use_log2
);
```

**返回：** H(Y | X) = H(X, Y) - H(X)

**复杂度：**
- 时间：O(n + n_bins_x * n_bins_y)
- 空间：O(n_bins_x * n_bins_y) 辅助空间

**使用场景：**
- 条件信息
- 预测信息
- 依赖性量化

## 互信息

### mutual_information

从分箱数据计算互信息 I(X; Y)：

```cpp
Real mi = scl::kernel::entropy::mutual_information(
    x_binned,
    y_binned,
    n,
    n_bins_x,
    n_bins_y,
    false  // use_log2
);
```

**返回：** I(X; Y) = H(X) + H(Y) - H(X, Y)

**后置条件：**
- 始终 >= 0
- 如果 X 和 Y 独立，则 I(X; Y) = 0

**复杂度：**
- 时间：O(n + n_bins_x * n_bins_y)
- 空间：O(n_bins_x * n_bins_y) 辅助空间

**使用场景：**
- 特征选择
- 依赖性检测
- 信息增益

### normalized_mi

计算两个标记之间的归一化互信息：

```cpp
Array<const Index> labels1 = /* ... */;
Array<const Index> labels2 = /* ... */;

Real nmi = scl::kernel::entropy::normalized_mi(
    labels1,
    labels2,
    n_clusters1,
    n_clusters2
);
```

**返回：** NMI = 2 * I(X;Y) / (H(X) + H(Y))

**后置条件：**
- 值在 [0, 1] 范围内，其中 1 表示完全一致

**复杂度：**
- 时间：O(n + n_clusters1 * n_clusters2)
- 空间：O(n_clusters1 * n_clusters2) 辅助空间

**使用场景：**
- 聚类评估
- 标签一致性
- 共识聚类

### adjusted_mi

计算调整互信息（机会校正）：

```cpp
Real ami = scl::kernel::entropy::adjusted_mi(
    labels1,
    labels2,
    n_clusters1,
    n_clusters2
);
```

**返回：** AMI = (MI - E[MI]) / (max(H1, H2) - E[MI])

**后置条件：**
- 值在 [-1, 1] 范围内，其中 1 表示完全一致
- 针对随机机会进行校正

**复杂度：**
- 时间：O(n + n_clusters1 * n_clusters2)
- 空间：O(n_clusters1 * n_clusters2) 辅助空间

**使用场景：**
- 带机会校正的聚类评估
- 比 NMI 更鲁棒
- 当聚类数量不同时

## 特征选择

### select_features_mi

使用与目标的互信息选择顶级特征：

```cpp
Sparse<Real, true> X = /* ... */;
Array<const Index> target = /* ... */;  // 目标标签
Array<Index> selected_features(n_to_select);
Array<Real> mi_scores(n_features);

scl::kernel::entropy::select_features_mi(
    X,
    target,
    n_features,
    n_to_select,
    selected_features,
    mi_scores,
    config::DEFAULT_N_BINS  // n_bins = 10
);
```

**参数：**
- `X`: 特征矩阵（CSR 或 CSC）
- `target`: 目标标签 [n_samples]
- `n_features`: 特征总数
- `n_to_select`: 要选择的特征数量
- `selected_features`: 选择的特征索引 [n_to_select]
- `mi_scores`: 所有特征的 MI 分数 [n_features]
- `n_bins`: 用于离散化的分箱数量

**后置条件：**
- `selected_features` 包含按 MI 排序的前 n_to_select 个特征
- `mi_scores` 包含每个特征的 MI 分数

**复杂度：**
- 时间：O(n_features * n_samples * log(nnz_per_sample))
- 空间：O(n_samples) 辅助空间

**使用场景：**
- 特征选择
- 基因选择
- 降维

### mrmr_selection

使用最小冗余最大相关性（mRMR）选择特征：

```cpp
Array<Index> selected_features(n_to_select);

scl::kernel::entropy::mrmr_selection(
    X,
    target,
    n_features,
    n_to_select,
    selected_features,
    config::DEFAULT_N_BINS
);
```

**后置条件：**
- `selected_features` 包含 mRMR 选择的特征
- 特征最大化相关性并最小化冗余

**算法：**
贪心选择：
1. 选择与目标具有最高 MI 的特征
2. 对于每个剩余特征：
   - 计算相关性（与目标的 MI）
   - 计算冗余（与已选特征的平均 MI）
   - 选择 max(相关性 - 冗余) 的特征

**复杂度：**
- 时间：O(n_to_select * n_features * n_samples)
- 空间：O(n_features * n_samples) 辅助空间

**使用场景：**
- 带冗余控制的特征选择
- 当特征相关时
- 比简单 MI 排序更好

## 配置

`scl::kernel::entropy::config` 中的默认参数：

```cpp
namespace config {
    constexpr Real LOG_BASE_E = 2.718281828459045;
    constexpr Real LOG_2 = 0.693147180559945;
    constexpr Real INV_LOG_2 = 1.4426950408889634;
    constexpr Real EPSILON = 1e-15;
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr size_t PREFETCH_DISTANCE = 64;
}
```

## 性能考虑

### 并行化

- `row_entropy`: 在行上并行
- `histogram_2d`: 使用原子累加并行
- `mutual_information`: 并行操作
- `select_features_mi`: 顺序（但可以在每个特征上并行）

### 内存效率

- 预分配的输出缓冲区
- 高效的直方图计算
- 最少的临时分配

## 最佳实践

### 1. 选择适当的分箱

```cpp
// 均匀分布的等宽
scl::kernel::entropy::discretize_equal_width(values, n, n_bins, binned);

// 偏斜分布的等频
scl::kernel::entropy::discretize_equal_frequency(values, n, n_bins, binned);
```

### 2. 使用 JS 散度获得对称性

```cpp
// 当需要对称时
Real js = scl::kernel::entropy::js_divergence(p, q);

// 当方向重要时
Real kl = scl::kernel::entropy::kl_divergence(p, q);
```

### 3. 使用 mRMR 进行特征选择

```cpp
// 当特征相关时
scl::kernel::entropy::mrmr_selection(X, target, n_features, n_to_select, selected);

// 当特征独立时
scl::kernel::entropy::select_features_mi(X, target, n_features, n_to_select, selected, scores);
```

---

::: tip 对数底
对信息论度量使用 log2（比特），对一般熵使用自然对数。
:::

::: warning 离散化
分箱方法和分箱数量的选择显著影响熵估计。对偏斜数据使用等频。
:::
