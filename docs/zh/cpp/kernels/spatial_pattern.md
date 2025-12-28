# spatial_pattern.hpp

> scl/kernel/spatial_pattern.hpp · 空间模式检测内核（SpatialDE 风格）

## 概述

本文件为空间转录组学数据提供空间模式检测方法，用于识别空间可变基因和计算空间梯度。这些方法受 SpatialDE 及相关方法的启发，用于分析空间基因表达模式。

本文件提供：
- 空间可变基因识别
- 空间梯度计算
- 空间模式分析
- 基于距离的空间统计

**头文件**: `#include "scl/kernel/spatial_pattern.hpp"`

---

## 主要 API

### spatially_variable_genes

::: source_code file="scl/kernel/spatial_pattern.hpp" symbol="spatially_variable_genes" collapsed
:::

**算法说明**

使用空间自相关识别空间可变基因：

1. **空间核计算**：对每个基因计算基于坐标和带宽的空间核权重
2. **空间变异分数**：计算加权的空间自相关或方差
3. **并行处理**：并行处理基因以提高效率

**边界条件**

- 零带宽：如果 bandwidth = 0，返回零分数
- 相同坐标：如果所有细胞坐标相同，返回零分数
- 常数值表达：常数值表达的基因返回零分数

**数据保证（前置条件）**

- `sv_scores` 必须预分配，容量 >= n_genes
- `coordinates` 长度 >= n_cells * n_dims
- `bandwidth > 0` 以获得有意义的结果

**复杂度分析**

- **时间**：O(n_genes * n_cells^2)
- **空间**：O(n_cells) 辅助空间

**示例**

```cpp
#include "scl/kernel/spatial_pattern.hpp"

Sparse<Real, true> expression = /* ... */;
Array<Real> coordinates(n_cells * n_dims);
Array<Real> sv_scores(n_genes);

Real bandwidth = 0.3;
scl::kernel::spatial_pattern::spatially_variable_genes(
    expression, coordinates.data(), n_cells, n_genes, n_dims,
    sv_scores, bandwidth
);
```

---

### spatial_gradient

::: source_code file="scl/kernel/spatial_pattern.hpp" symbol="spatial_gradient" collapsed
:::

**算法说明**

计算基因表达的空间梯度：

1. **邻居识别**：找到带宽内的空间邻居
2. **梯度计算**：计算加权梯度指示表达变化的方向和幅度
3. **并行处理**：并行处理细胞

**边界条件**

- 无邻居：带宽内无邻居的细胞梯度为零
- 零带宽：如果 bandwidth = 0，返回零梯度
- 常数值表达：常数值表达的基因返回零梯度

**数据保证（前置条件）**

- `gradients` 必须预分配，容量 >= n_cells * n_dims
- `gene_index` 必须在 [0, n_genes) 范围内

**复杂度分析**

- **时间**：O(n_cells * n_neighbors)
- **空间**：O(n_cells) 辅助空间

**示例**

```cpp
Index gene_index = 42;
Array<Real> gradients(n_cells * n_dims);

scl::kernel::spatial_pattern::spatial_gradient(
    expression, coordinates.data(), gene_index,
    n_cells, n_dims, gradients.data()
);
```

---

## 注意事项

**空间带宽**：
- 控制分析的空间尺度
- 较小带宽：局部模式
- 较大带宽：全局模式
- 典型值：0.1 - 0.5（归一化坐标）

**性能**：
- 按基因（spatially_variable_genes）和细胞（spatial_gradient）并行化
- 距离计算为 O(n_cells^2) 每个基因，对于大型数据集可能昂贵

## 相关内容

- [Spatial Statistics](/zh/cpp/kernels/spatial) - Moran's I 和 Geary's C 统计
- [Hotspot Detection](/zh/cpp/kernels/hotspot) - 局部空间统计

