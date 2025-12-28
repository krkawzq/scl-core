# 组织结构分析

用于空间组织的组织层分配和分区分析。

## 概览

组织结构内核提供：

- **层分配** - 根据空间坐标将细胞分配到组织层
- **分区评分** - 沿空间轴计算分区得分
- **组织组织** - 分析空间组织模式
- **结构分析** - 表征组织结构

## 层分配

### layer_assignment

根据空间坐标将细胞分配到组织层：

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coordinates = /* ... */;           // 空间坐标 [n_cells * n_dims]
Size n_cells = /* ... */;
Size n_dims = 2;                                // 通常为 2D 或 3D
Index n_layers = 5;                             // 层数

Array<Index> layer_labels(n_cells);            // 预分配输出

scl::kernel::tissue::layer_assignment(
    coordinates, n_cells, n_dims,
    layer_labels,
    n_layers);

// layer_labels[i] 包含细胞 i 的层 ID (0 到 n_layers-1)
```

**参数：**
- `coordinates`: 空间坐标，大小 = n_cells × n_dims
- `n_cells`: 细胞数量
- `n_dims`: 空间维度数
- `layer_labels`: 输出层标签，必须预分配，大小 = n_cells
- `n_layers`: 要分配的层数

**后置条件：**
- `layer_labels[i]` 包含细胞 i 的层 ID（0 到 n_layers-1）
- 根据沿主轴的空间位置分配层
- 细胞在各层之间大致均匀分布

**算法：**
- 将坐标投影到主轴
- 将轴划分为 n_layers 段
- 根据位置将细胞分配到层

**复杂度：**
- 时间：O(n_cells * n_dims) - 与细胞数量线性相关
- 空间：O(n_cells) 辅助空间用于标签

**线程安全：**
- 安全 - 跨细胞并行化
- 每个细胞独立处理

**用例：**
- 组织层识别
- 分层分析（表皮、真皮等）
- 径向组织（皮质、髓质等）
- 结构注释

## 分区评分

### zonation_score

沿空间轴计算分区得分：

```cpp
const Real* coordinates = /* ... */;
const Real* expression = /* ... */;            // 基因表达值 [n_cells]
Size n_cells = /* ... */;
Size axis = 0;                                  // X 轴 (0)、Y 轴 (1) 或 Z 轴 (2)

Array<Real> scores(n_cells);                   // 预分配输出

scl::kernel::tissue::zonation_score(
    coordinates, expression,
    n_cells, axis,
    scores);

// scores[i] 包含细胞 i 沿指定轴的分区得分
```

**参数：**
- `coordinates`: 空间坐标，大小 = n_cells × n_dims
- `expression`: 基因表达值，大小 = n_cells
- `n_cells`: 细胞数量
- `axis`: 空间轴索引（0 = X，1 = Y，2 = Z）
- `scores`: 输出分区得分，必须预分配，大小 = n_cells

**后置条件：**
- `scores[i]` 包含细胞 i 沿指定轴的分区得分
- 得分表示相对于轴位置的表达水平
- 用于识别梯度模式

**算法：**
- 将细胞投影到指定轴
- 计算作为轴位置函数的表达
- 生成分区得分（可使用平滑或分箱）

**复杂度：**
- 时间：O(n_cells) - 线性处理
- 空间：O(n_cells) 辅助空间用于得分

**线程安全：**
- 安全 - 并行化处理

**用例：**
- 门静脉-中央分区（肝脏）
- 皮质-髓质分区（肾脏）
- 顶端-基底分区（上皮）
- 梯度模式检测

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_LAYER = 5;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real PI = Real(3.14159265358979323846);
}
```

**每层最小细胞数：**
- 确保每层有足够的细胞进行统计分析
- 根据数据集大小调整

**空间轴：**
- 轴 0 = X（通常为水平）
- 轴 1 = Y（通常为垂直）
- 轴 2 = Z（深度，用于 3D 数据）

## 示例

### 基于层的分析

```cpp
#include "scl/kernel/tissue.hpp"

const Real* coords = /* ... */;  // [n_cells * 2]
Size n_cells = /* ... */;
Index n_layers = 4;  // 例如，表皮、真皮、皮下组织、肌肉

Array<Index> layers(n_cells);
scl::kernel::tissue::layer_assignment(
    coords, n_cells, 2,  // 2D 坐标
    layers,
    n_layers);

// 按层分析表达
Sparse<Real, true> expression = /* ... */;
for (Index layer = 0; layer < n_layers; ++layer) {
    // 提取此层中的细胞
    std::vector<Index> layer_cells;
    for (Index i = 0; i < n_cells; ++i) {
        if (layers[i] == layer) {
            layer_cells.push_back(i);
        }
    }
    
    // 计算层特异性统计
    // ... 分析 layer_cells 的表达 ...
}
```

### 分区分析

```cpp
// 沿 X 轴分析分区（肝脏的门静脉-中央轴）
const Real* expression = /* ... */;  // 特定基因的表达
Size axis = 0;  // X 轴

Array<Real> zonation(n_cells);
scl::kernel::tissue::zonation_score(
    coords, expression,
    n_cells, axis,
    zonation);

// 根据得分阈值识别区
std::vector<Index> zone1, zone2, zone3;  // 门静脉、中间、中央
Real threshold1 = 0.33;
Real threshold2 = 0.67;

for (Index i = 0; i < n_cells; ++i) {
    Real score = zonation[i];
    if (score < threshold1) {
        zone1.push_back(i);  // 门静脉区
    } else if (score < threshold2) {
        zone2.push_back(i);  // 中间区
    } else {
        zone3.push_back(i);  // 中央区
    }
}
```

### 多轴分区

```cpp
// 沿多个轴分析分区
const Real* gene_expr = /* ... */;

// X 轴分区
Array<Real> zonation_x(n_cells);
scl::kernel::tissue::zonation_score(coords, gene_expr, n_cells, 0, zonation_x);

// Y 轴分区
Array<Real> zonation_y(n_cells);
scl::kernel::tissue::zonation_score(coords, gene_expr, n_cells, 1, zonation_y);

// 组合分析（例如，径向模式）
Array<Real> radial_pattern(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    // 组合 X 和 Y 分区得分
    Real x = zonation_x[i];
    Real y = zonation_y[i];
    radial_pattern[i] = std::sqrt(x * x + y * y);  // 距中心的距离
}
```

---

::: tip 层 vs. 分区
使用层分配进行离散组织区域，使用分区评分进行连续梯度。层是分类的，分区是定量的。
:::

