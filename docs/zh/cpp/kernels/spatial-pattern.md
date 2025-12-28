# 空间模式检测

用于空间转录组学的空间变异分析和梯度计算（SpatialDE 风格）。

## 概览

空间模式检测内核提供：

- **空间变异基因** - 识别具有空间表达模式的基因
- **空间梯度** - 计算空间中的表达梯度
- **模式识别** - 检测空间表达域
- **SpatialDE 分析** - 统计空间变异测试

## 空间变异基因

### spatially_variable_genes

使用空间自相关识别空间变异基因：

```cpp
#include "scl/kernel/spatial_pattern.hpp"

Sparse<Real, true> expression = /* ... */;     // 表达矩阵 [n_cells x n_genes]
const Real* coordinates = /* ... */;            // 空间坐标 [n_cells * n_dims]
Index n_cells = expression.rows();
Index n_genes = expression.cols();
Size n_dims = 2;                                // 2D 或 3D

Array<Real> sv_scores(n_genes);                // 预分配输出

// 标准分析
scl::kernel::spatial_pattern::spatially_variable_genes(
    expression, coordinates,
    n_cells, n_genes, n_dims,
    sv_scores,
    bandwidth = 100.0                           // 空间带宽
);

// sv_scores[g] 包含基因 g 的空间变异得分
```

**参数：**
- `expression`: 表达矩阵（细胞 × 基因，CSR 格式）
- `coordinates`: 空间坐标，大小 = n_cells × n_dims
- `n_cells`: 细胞数量
- `n_genes`: 基因数量
- `n_dims`: 空间维度数（通常为 2 或 3）
- `sv_scores`: 输出空间变异得分，必须预分配，大小 = n_genes
- `bandwidth`: 核的空间带宽（控制邻域大小）

**后置条件：**
- `sv_scores[g]` 包含基因 g 的空间变异得分
- 更高得分表示更强的空间模式
- 得分可用于 ranking 和 filtering

**算法：**
对每个基因并行：
1. 使用核函数计算空间自相关
2. 测量与随机空间分布的偏差
3. 计算变异得分（例如，Moran's I、SpatialDE 统计量）

**复杂度：**
- 时间：O(n_genes * n_cells^2) - 细胞数量的二次方（成对比较）
- 空间：O(n_cells) 每个基因的辅助空间

**线程安全：**
- 安全 - 跨基因并行化
- 每个基因独立处理

**用例：**
- 空间转录组学分析
- 识别域特异性基因
- SpatialDE 风格分析
- 组织中的模式发现

## 空间梯度

### spatial_gradient

计算基因表达的空间梯度：

```cpp
Sparse<Real, true> expression = /* ... */;
const Real* coordinates = /* ... */;
Index gene_index = 5;                           // 要分析的基因
Index n_cells = expression.rows();
Size n_dims = 2;

Array<Real> gradients(n_cells * n_dims);       // 预分配输出

scl::kernel::spatial_pattern::spatial_gradient(
    expression, coordinates,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// gradients[i * n_dims + d] 包含细胞 i 的梯度分量 d
```

**参数：**
- `expression`: 表达矩阵（细胞 × 基因，CSR 格式）
- `coordinates`: 空间坐标，大小 = n_cells × n_dims
- `gene_index`: 要分析的基因索引
- `n_cells`: 细胞数量
- `n_dims`: 空间维度数
- `gradients`: 输出梯度向量，必须预分配，大小 = n_cells × n_dims

**后置条件：**
- `gradients[i * n_dims + d]` 包含细胞 i 的梯度分量 d
- 梯度幅度表示变化率
- 梯度方向表示表达增加的方向

**算法：**
对每个细胞并行：
1. 识别空间邻居
2. 计算表达差异
3. 使用局部线性回归或有限差分估计梯度

**复杂度：**
- 时间：O(n_cells * n_neighbors)，其中 n_neighbors = 每个细胞的平均邻居数
- 空间：O(n_cells) 辅助空间用于邻居索引

**线程安全：**
- 安全 - 跨细胞并行化
- 每个细胞独立处理

**用例：**
- 基于梯度的模式分析
- 方向性表达变化
- 形态发生素梯度检测
- 空间分化模式

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_NEIGHBORS = 3;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real BANDWIDTH_SCALE = Real(0.3);
}
```

**带宽：**
- 控制空间邻域大小
- 更小 = 局部模式，更大 = 全局模式
- 应与数据的空间分辨率匹配

**邻居数：**
- 梯度计算的邻居数
- 默认 15 通常足够
- 增加以获得更平滑的梯度

## 示例

### 识别空间变异基因

```cpp
#include "scl/kernel/spatial_pattern.hpp"

Sparse<Real, true> expression = /* ... */;
const Real* coords = /* ... */;  // [n_cells * 2] 对于 2D
Index n_cells = expression.rows();
Index n_genes = expression.cols();

Array<Real> sv_scores(n_genes);
Real bandwidth = 50.0;  // 根据组织规模调整

scl::kernel::spatial_pattern::spatially_variable_genes(
    expression, coords,
    n_cells, n_genes, 2,  // 2D 坐标
    sv_scores,
    bandwidth
);

// 按空间变异对基因排序
std::vector<std::pair<Real, Index>> ranked;
for (Index g = 0; g < n_genes; ++g) {
    ranked.push_back({sv_scores[g], g});
}
std::sort(ranked.rbegin(), ranked.rend());  // 降序排序

// 顶级空间变异基因
Index top_n = 100;
for (Index i = 0; i < top_n && i < ranked.size(); ++i) {
    std::cout << "基因 " << ranked[i].second
              << " (SV 得分: " << ranked[i].first << ")\n";
}
```

### 计算表达梯度

```cpp
Index gene_index = 10;  // 感兴趣的基因
Size n_dims = 2;

Array<Real> gradients(n_cells * n_dims);
scl::kernel::spatial_pattern::spatial_gradient(
    expression, coords,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// 计算梯度幅度
Array<Real> magnitudes(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    Real gx = gradients[i * n_dims + 0];
    Real gy = gradients[i * n_dims + 1];
    magnitudes[i] = std::sqrt(gx * gx + gy * gy);
}

// 查找具有强梯度的细胞（表达边界）
Real threshold = std::percentile(magnitudes.begin(), magnitudes.end(), 90);
for (Index i = 0; i < n_cells; ++i) {
    if (magnitudes[i] > threshold) {
        // 细胞 i 在梯度/边界区域
    }
}
```

### 梯度方向分析

```cpp
// 分析梯度方向以识别表达源/汇
Array<Real> gradients(n_cells * n_dims);
scl::kernel::spatial_pattern::spatial_gradient(
    expression, coords,
    gene_index, n_cells, n_dims,
    gradients.ptr);

// 聚类梯度方向以识别域
// （实现取决于您的聚类库）
// 或直接可视化梯度向量
```

---

::: tip 带宽选择
根据预期模式规模选择带宽：小带宽用于细粒度模式，大带宽用于广泛域。考虑使用多个带宽进行多尺度分析。
:::

