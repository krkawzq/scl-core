# tissue.hpp

> scl/kernel/tissue.hpp · 组织结构和组织分析

## 概述

本文件提供用于分析空间转录组学数据中组织结构和空间组织的函数。包括基于空间坐标的层分配和沿空间轴的区域化评分。

**头文件**: `#include "scl/kernel/tissue.hpp"`

主要特性：
- 基于空间坐标的层分配
- 沿空间轴的区域化评分计算
- 空间模式分析

---

## 主要 API

### layer_assignment

::: source_code file="scl/kernel/tissue.hpp" symbol="layer_assignment" collapsed
:::

**算法说明**

基于空间坐标将细胞分配到组织层：

1. 对于每个细胞 i：
   - 计算从细胞到组织边界/中心的距离
   - 根据距离分位数确定层
   - 分配层标签：`layer_labels[i] = layer_id` 在 [0, n_layers-1] 范围内
2. 使用等宽或等频分箱分配层
3. 对细胞使用并行处理

**边界条件**

- **n_layers = 1**：所有细胞分配到层 0
- **n_layers = 0**：未定义行为
- **所有细胞在同一位置**：所有分配到同一层
- **异常坐标**：可能影响层边界

**数据保证（前置条件）**

- `layer_labels` 容量 >= n_cells
- `coordinates` 是行主序：`coordinates[i * n_dims + j]` 是细胞 i 的维度 j
- `n_layers > 0`
- `n_dims` 匹配坐标数组维度

**复杂度分析**

- **时间**：O(n_cells * n_dims) - 每个细胞的距离计算
- **空间**：O(n_cells) 辅助空间

**示例**

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coordinates = /* 空间坐标 [n_cells * n_dims] */;
scl::Array<Index> layer_labels(n_cells);

scl::kernel::tissue::layer_assignment(
    coordinates, n_cells, n_dims,
    layer_labels, 5  // n_layers
);

// layer_labels[i] 包含细胞 i 的层 ID (0-4)
```

---

### zonation_score

::: source_code file="scl/kernel/tissue.hpp" symbol="zonation_score" collapsed
:::

**算法说明**

沿空间轴计算区域化分数：

1. 对于每个细胞 i：
   - 提取沿指定轴的空间坐标
   - 计算表达与空间位置之间的相关性
   - 分数 = 归一化相关性（Pearson 或 Spearman）
2. 区域化分数表示沿轴的方向梯度强度
3. 更高的分数表示更强的空间梯度

**边界条件**

- **axis >= n_dims**：未定义行为
- **恒定表达**：所有分数为 0
- **恒定坐标**：所有分数为 0
- **完美梯度**：分数接近 1.0

**数据保证（前置条件）**

- `scores` 容量 >= n_cells
- `axis < n_dims`
- `expression` 和 `coordinates` 具有匹配的 n_cells
- 坐标是行主序布局

**复杂度分析**

- **时间**：O(n_cells) - 相关性计算
- **空间**：O(n_cells) 辅助空间

**示例**

```cpp
const Real* expression = /* 基因表达 [n_cells] */;
scl::Array<Real> scores(n_cells);

scl::kernel::tissue::zonation_score(
    coordinates, expression, n_cells,
    0,      // axis: 0 = x轴, 1 = y轴, 2 = z轴
    scores
);

// scores[i] 包含沿指定轴的区域化分数
// 更高的分数表示更强的空间梯度
```

---

## 配置

默认参数在 `scl::kernel::tissue::config` 中定义：

- `EPSILON = 1e-10`：数值容差
- `MIN_CELLS_PER_LAYER = 5`：每层所需的最小细胞数
- `DEFAULT_N_NEIGHBORS = 15`：空间分析的默认邻居数
- `MAX_ITERATIONS = 100`：迭代方法的最大迭代次数
- `PI = 3.14159...`：数学常数

---

## 注意事项

- 层分配假设组织具有分层结构（例如，皮层层）
- 区域化分数有助于识别空间表达梯度
- 空间坐标应使用一致的单位（例如，微米）
- 可以分别分析多个轴以进行 3D 组织分析

## 相关内容

- [空间模块](./spatial) - 用于其他空间分析方法
- [空间模式模块](./spatial_pattern) - 用于模式检测
