# spatial.hpp

> scl/kernel/spatial.hpp · 空间自相关统计内核

## 概述

本文件为空间转录组学和空间数据分析提供高性能的空间自相关统计。它实现了 Moran's I 和 Geary's C 统计量，具有 SIMD 优化和嵌套并行性，可高效处理大规模数据集。

本文件提供：
- Moran's I 空间自相关统计量
- Geary's C 空间自相关统计量
- 图矩阵权重和计算
- SIMD 优化计算（8 路展开）
- 大规模分析的嵌套并行性

**头文件**: `#include "scl/kernel/spatial.hpp"`

---

## 主要 API

### morans_i

::: source_code file="scl/kernel/spatial.hpp" symbol="morans_i" collapsed
:::

**算法说明**

计算每个特征的 Moran's I 空间自相关统计量：

1. **中心化**：计算均值并中心化值
2. **分子计算**：计算加权邻居和并使用 8 路展开循环
3. **分母计算**：计算平方和
4. **Moran's I 计算**：I = (N / W) * (numerator / denominator)

**边界条件**

- 零方差：方差为零的特征返回 I = 0
- 零总权重：如果 W = 0，返回 I = 0
- 空图：无边的图返回 I = 0

**数据保证（前置条件）**

- `graph` 必须是方阵
- `features.secondary_dim() == graph.primary_dim()`
- `output.len == features.primary_dim()`

**复杂度分析**

- **时间**：O(n_features * nnz_graph)
- **空间**：O(n_cells * n_threads) 用于线程局部 z 缓冲区

**示例**

```cpp
#include "scl/kernel/spatial.hpp"

Sparse<Real, true> graph = /* ... */;
Sparse<Real, true> features = /* ... */;
Array<Real> morans_i_scores(n_features);

scl::kernel::spatial::morans_i(graph, features, morans_i_scores);
```

---

### gearys_c

::: source_code file="scl/kernel/spatial.hpp" symbol="gearys_c" collapsed
:::

**算法说明**

计算每个特征的 Geary's C 空间自相关统计量：

1. **中心化**：计算均值并中心化值
2. **分子计算**：计算加权平方差和
3. **分母计算**：计算分母
4. **Geary's C 计算**：C = (N-1) * numerator / denominator

**边界条件**

- 零方差：方差为零的特征返回 C = 0
- 零总权重：如果 W = 0，返回 C = 0

**数据保证（前置条件）**

- `graph` 必须是方阵
- `features.secondary_dim() == graph.primary_dim()`

**复杂度分析**

- **时间**：O(n_features * nnz_graph)
- **空间**：O(n_cells * n_threads)

---

### weight_sum

::: source_code file="scl/kernel/spatial.hpp" symbol="weight_sum" collapsed
:::

**算法说明**

计算稀疏图中所有边权重的和。

**复杂度分析**

- **时间**：O(nnz / n_threads)
- **空间**：O(n_threads)

---

## 工具函数

### detail::compute_weighted_neighbor_sum

计算邻居值的加权和（内部辅助函数）。

::: source_code file="scl/kernel/spatial.hpp" symbol="detail::compute_weighted_neighbor_sum" collapsed
:::

**复杂度**

- 时间：O(len)
- 空间：O(1)

---

## 注意事项

**Moran's I vs Geary's C**：
- Moran's I：范围 [-1, 1]，正值表示聚类
- Geary's C：范围 [0, 2]，C < 1 表示聚类
- Geary's C 与 Moran's I 呈反相关关系

**性能**：
- SIMD 优化，8 路展开
- 多累加器模式隐藏 FP 延迟
- 嵌套并行用于大规模分析

## 相关内容

- [Hotspot Detection](/zh/cpp/kernels/hotspot) - 局部空间统计和热点检测
- [Spatial Pattern](/zh/cpp/kernels/spatial_pattern) - 空间模式检测方法

