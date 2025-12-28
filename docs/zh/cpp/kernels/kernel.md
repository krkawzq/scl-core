# kernel.hpp

> scl/kernel/kernel.hpp · 核密度估计与核矩阵计算

## 概述

本文件提供核密度估计和核矩阵计算的核方法。支持多种核类型，包括高斯、Epanechnikov、余弦、线性、多项式、拉普拉斯、柯西、Sigmoid、均匀和三角核。

**头文件**: `#include "scl/kernel/kernel.hpp"`

---

## 主要 API

### kernel_density_estimation

::: source_code file="scl/kernel/kernel.hpp" symbol="kernel_density_estimation" collapsed
:::

**算法说明**

使用核函数在查询点处计算核密度估计：

1. 对于每个查询点 q：
   - 计算到所有数据点的距离
   - 应用核函数 K(||q - p|| / h)，其中 h 是带宽
   - 对核值求和并除以数据点数量进行归一化
   - 存储密度估计值

2. 在查询点上并行处理以提高效率

3. 支持多种核类型，并提供优化实现

**边界条件**

- **空数据点**: 所有查询返回零密度
- **空查询**: 立即返回，不进行计算
- **零带宽**: 限制为 MIN_BANDWIDTH (1e-10) 以避免除零
- **极大距离**: 对于紧支撑核，核值接近零

**数据保证（前置条件）**

- `points` 必须是有效指针或 nullptr（如果 n_points == 0）
- `queries` 必须是有效指针或 nullptr（如果 n_queries == 0）
- `densities` 必须具有容量 >= n_queries
- 数据布局: points[i * n_dims + j] 是点 i 的维度 j
- 查询布局: queries[i * n_dims + j] 是查询 i 的维度 j

**复杂度分析**

- **时间**: O(n_queries * n_points * n_dims) - 每个查询与所有点比较
- **空间**: O(1) 辅助空间 - 仅临时距离计算

**示例**

```cpp
#include "scl/kernel/kernel.hpp"

// 准备数据
const Real* points = /* 数据点 [n_points * n_dims] */;
const Real* queries = /* 查询点 [n_queries * n_dims] */;
Size n_points = 1000;
Size n_queries = 100;
Size n_dims = 10;

// 预分配输出
Array<Real> densities(n_queries);

// 使用高斯核计算 KDE
scl::kernel::kernel::kernel_density_estimation(
    points, queries,
    n_points, n_queries, n_dims,
    densities,
    bandwidth = 1.0,
    kernel_type = scl::kernel::kernel::KernelType::Gaussian
);

// densities[i] 现在包含查询点 i 的 KDE
for (Size i = 0; i < n_queries; ++i) {
    std::cout << "查询 " << i << " 密度: " << densities[i] << "\n";
}
```

---

### kernel_matrix

::: source_code file="scl/kernel/kernel.hpp" symbol="kernel_matrix" collapsed
:::

**算法说明**

计算两个点集之间的核矩阵 K，其中 K[i, j] = kernel(points1[i], points2[j])：

1. 对于集合 1 和集合 2 的每对点 (i, j)：
   - 计算 points1[i] 和 points2[j] 之间的距离
   - 应用带带宽的核函数
   - 将结果存储在 kernel_mat[i * n2 + j]

2. 在矩阵元素上并行处理以提高效率

3. 支持所有核类型，并提供优化的距离计算

**边界条件**

- **空点集 1**: 返回零矩阵
- **空点集 2**: 返回零矩阵
- **零带宽**: 限制为 MIN_BANDWIDTH 以避免数值问题
- **相同点集**: 计算自核矩阵（某些核是对称的）

**数据保证（前置条件）**

- `points1` 必须是有效指针或 nullptr（如果 n1 == 0）
- `points2` 必须是有效指针或 nullptr（如果 n2 == 0）
- `kernel_mat` 必须具有容量 >= n1 * n2
- 数据布局: points[i * n_dims + j] 是点 i 的维度 j
- 输出布局: kernel_mat[i * n2 + j] 是核值 K(points1[i], points2[j])

**复杂度分析**

- **时间**: O(n1 * n2 * n_dims) - 计算所有成对核值
- **空间**: O(1) 辅助空间 - 仅临时距离计算

**示例**

```cpp
#include "scl/kernel/kernel.hpp"

// 准备两个点集
const Real* points1 = /* 第一个集合 [n1 * n_dims] */;
const Real* points2 = /* 第二个集合 [n2 * n_dims] */;
Size n1 = 100;
Size n2 = 200;
Size n_dims = 10;

// 预分配核矩阵
Array<Real> kernel_mat(n1 * n2);

// 使用 Epanechnikov 核计算核矩阵
scl::kernel::kernel::kernel_matrix(
    points1, points2,
    n1, n2, n_dims,
    kernel_mat.ptr,
    bandwidth = 1.5,
    kernel_type = scl::kernel::kernel::KernelType::Epanechnikov
);

// kernel_mat[i * n2 + j] 包含 K(points1[i], points2[j])
// 用于 SVM、核 PCA 等核方法
```

---

## 核类型

支持以下核类型：

- **Gaussian**: exp(-0.5 * (||x-y||/h)^2) - 平滑，无界支撑
- **Epanechnikov**: max(0, 1 - (||x-y||/h)^2) - 紧支撑，高效
- **Cosine**: cos(π * ||x-y|| / (2*h)) - 紧支撑，平滑
- **Linear**: max(0, 1 - ||x-y||/h) - 紧支撑，简单
- **Polynomial**: (1 + x·y)^d - 多项式核（需要内积）
- **Laplacian**: exp(-||x-y||/h) - 指数衰减
- **Cauchy**: 1 / (1 + (||x-y||/h)^2) - 重尾分布
- **Sigmoid**: tanh(α * x·y + c) - 神经网络风格
- **Uniform**: 1 if ||x-y|| < h, else 0 - 阶跃函数
- **Triangular**: max(0, 1 - ||x-y||/h) - 线性衰减

---

## 配置

### 默认参数

```cpp
namespace scl::kernel::kernel::config {
    constexpr Real DEFAULT_BANDWIDTH = Real(1.0);
    constexpr Real MIN_BANDWIDTH = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real LOG_MIN = Real(1e-300);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index NYSTROM_MAX_ITER = 50;
}
```

---

## 注意事项

**带宽选择**: 根据数据规模选择带宽。带宽过小会导致过拟合（尖峰密度），过大会导致过度平滑。使用交叉验证或经验法则（如 Silverman 规则）进行最优选择。

**核选择**: 高斯核最常用于平滑密度。Epanechnikov 在均方误差方面最优，但具有紧支撑。根据应用需求选择。

**性能**: 当 n_queries >= PARALLEL_THRESHOLD (500) 或 n1 * n2 >= PARALLEL_THRESHOLD 时启用并行化。

---

## 相关内容

- [Neighbors](/zh/cpp/kernels/neighbors) - K 近邻
- [BBKNN](/zh/cpp/kernels/bbknn) - 批次平衡 KNN
