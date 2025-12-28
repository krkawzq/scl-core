# hotspot.hpp

> scl/kernel/hotspot.hpp · 空间统计和热点检测内核

## 概述

本文件为空间转录组学和空间数据分析提供全面的空间自相关分析和热点检测。它实现了局部和全局 Moran's I、Getis-Ord Gi*、Geary's C，以及基于置换推断的 LISA 模式分类。

本文件提供：
- 局部和全局空间自相关统计量
- 热点和冷点检测
- LISA（局部空间关联指标）模式分类
- 空间检验的多重检验校正
- 空间权重矩阵构建工具

**头文件**: `#include "scl/kernel/hotspot.hpp"`

---

## 主要 API

### local_morans_i

::: source_code file="scl/kernel/hotspot.hpp" symbol="local_morans_i" collapsed
:::

**算法说明**

计算每个观测值的局部 Moran's I 统计量，测量局部空间自相关：

1. **标准化**：将属性值标准化为 z 分数
2. **空间滞后计算**：对每个观测值并行计算空间滞后
3. **局部 I 计算**：local_I[i] = z[i] * lag[i]
4. **置换检验**：通过置换邻居值进行统计推断

**边界条件**

- 零方差：如果 std(values) = 0，所有 local_I = 0
- 孤立观测值：无邻居的观测值 lag = 0，local_I = 0
- 空权重矩阵：如果矩阵无非零值，返回全零

**数据保证（前置条件）**

- `weights` 必须是方阵稀疏矩阵 (n x n)
- `local_I` 和 `p_values` 必须预分配 n 个元素
- `n_permutations > 0` 用于置换检验

**复杂度分析**

- **时间**：O(n * n_permutations * avg_neighbors)
- **空间**：O(n_threads * n) 用于置换缓冲区

**示例**

```cpp
#include "scl/kernel/hotspot.hpp"

Sparse<Real, true> weights = /* ... */;
Array<Real> expression(n);
Array<Real> local_I(n);
Array<Real> p_values(n);

scl::kernel::hotspot::local_morans_i(
    weights, expression.data(), n,
    local_I.data(), p_values.data(),
    999, 42
);
```

---

### getis_ord_g_star

::: source_code file="scl/kernel/hotspot.hpp" symbol="getis_ord_g_star" collapsed
:::

**算法说明**

计算 Getis-Ord Gi* 统计量用于热点检测，测量高值或低值的局部集中度。

**边界条件**

- 零权重：无邻居的观测值 g_star 未定义（设为 0）
- 常数值：如果所有值相等，所有 g_star = 0

**数据保证（前置条件）**

- `weights` 必须是方阵稀疏矩阵
- `g_star` 和 `p_values` 必须预分配

**复杂度分析**

- **时间**：O(nnz)
- **空间**：O(1) 辅助空间

**示例**

```cpp
Array<Real> g_star(n);
Array<Real> p_values(n);

scl::kernel::hotspot::getis_ord_g_star(
    weights, expression.data(), n,
    g_star.data(), p_values.data()
);
```

---

### classify_lisa_patterns

::: source_code file="scl/kernel/hotspot.hpp" symbol="classify_lisa_patterns" collapsed
:::

**算法说明**

根据标准化值和空间滞后将观测值分类为 LISA 模式类别。

**边界条件**

- 零 z 值：z = 0 的观测值分类为 NotSignificant
- 零空间滞后：lag = 0 的观测值分类为 NotSignificant

**数据保证（前置条件）**

- 所有输入数组长度 >= n
- `patterns` 必须预分配

**复杂度分析**

- **时间**：O(n)
- **空间**：O(1)

---

### global_morans_i

::: source_code file="scl/kernel/hotspot.hpp" symbol="global_morans_i" collapsed
:::

**算法说明**

计算全局 Moran's I 统计量，测量所有观测值的整体空间自相关。

**边界条件**

- 小样本：n < 3 时 I 未定义（返回 0）
- 零权重：如果 W = 0，I 未定义（返回 0）

**复杂度分析**

- **时间**：O(n * n_permutations * avg_neighbors)
- **空间**：O(n_threads * n)

---

### local_gearys_c

::: source_code file="scl/kernel/hotspot.hpp" symbol="local_gearys_c" collapsed
:::

**算法说明**

计算局部 Geary's C 统计量，基于平方差的局部空间自相关替代测量。

**复杂度分析**

- **时间**：O(n * n_permutations * avg_neighbors)
- **空间**：O(n_threads * n)

---

### global_gearys_c

::: source_code file="scl/kernel/hotspot.hpp" symbol="global_gearys_c" collapsed
:::

**算法说明**

计算全局 Geary's C 统计量，全局空间自相关的逆测量。

**复杂度分析**

- **时间**：O(nnz)
- **空间**：O(1) 辅助空间

---

## 工具函数

### identify_hotspots

从 Gi* z 分数识别统计显著的热点和冷点。

::: source_code file="scl/kernel/hotspot.hpp" symbol="identify_hotspots" collapsed
:::

**复杂度**

- 时间：O(n)
- 空间：O(1)

---

### benjamini_hochberg_correction

对多个空间检验的 p 值应用 Benjamini-Hochberg FDR 校正。

::: source_code file="scl/kernel/hotspot.hpp" symbol="benjamini_hochberg_correction" collapsed
:::

**复杂度**

- 时间：O(n log n)
- 空间：O(n)

---

### distance_band_weights

基于距离阈值构建空间权重矩阵。

::: source_code file="scl/kernel/hotspot.hpp" symbol="distance_band_weights" collapsed
:::

**复杂度**

- 时间：O(n^2)
- 空间：O(nnz)

---

### knn_weights

构建 K 最近邻空间权重矩阵。

::: source_code file="scl/kernel/hotspot.hpp" symbol="knn_weights" collapsed
:::

**复杂度**

- 时间：O(n^2 log k)
- 空间：O(n * k)

---

### bivariate_local_morans_i

计算两个变量之间的双变量局部 Moran's I。

::: source_code file="scl/kernel/hotspot.hpp" symbol="bivariate_local_morans_i" collapsed
:::

**复杂度**

- 时间：O(n * n_permutations * avg_neighbors)
- 空间：O(n_threads * n)

---

### detect_spatial_clusters

从 LISA 模式检测并标记空间聚类。

::: source_code file="scl/kernel/hotspot.hpp" symbol="detect_spatial_clusters" collapsed
:::

**复杂度**

- 时间：O(n + nnz)
- 空间：O(n)

---

### spatial_autocorrelation_summary

计算全面的空间自相关摘要统计量。

::: source_code file="scl/kernel/hotspot.hpp" symbol="spatial_autocorrelation_summary" collapsed
:::

**复杂度**

- 时间：O(n * n_permutations * avg_neighbors)
- 空间：O(n_threads * n)

---

## 注意事项

**空间权重矩阵**：
- 权重矩阵通常应行归一化以便解释
- 常见构建：距离带、KNN、逆距离
- 对称权重常见但不必需

**置换检验**：
- 默认 999 次置换在准确性和速度之间提供良好平衡
- 使用相同种子以确保可重现性

**多重检验**：
- 空间检验通常涉及许多观测值（多重检验问题）
- 使用 benjamini_hochberg_correction 控制 FDR

## 相关内容

- [Spatial Analysis](/zh/cpp/kernels/spatial) - 其他空间分析工具
- [Statistics](/zh/cpp/kernels/statistics) - 通用统计操作
