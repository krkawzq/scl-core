# 投影

用于降维的稀疏随机投影内核，具有距离保持保证。

## 概述

投影模块提供：

- **高斯投影** - 密集高斯随机投影
- **Achlioptas 投影** - 稀疏三元投影（快 3 倍）
- **稀疏投影** - 用于高维数据的非常稀疏投影
- **Count-Sketch** - 基于哈希的投影（O(nnz) 时间）
- **即时方法** - 无需存储矩阵的内存高效投影
- **Johnson-Lindenstrauss** - 距离保持的维度计算

## 投影类型

### ProjectionType 枚举

```cpp
enum class ProjectionType {
    Gaussian,       // N(0, 1/k) 条目 - 最高精度
    Achlioptas,     // {+1, 0, -1}，概率 {1/6, 2/3, 1/6} - 快 3 倍
    Sparse,         // 密度 = 1/sqrt(d) - 最适合高维
    CountSketch     // 哈希 + 符号 - O(nnz) 时间
};
```

**选择指南：**
- **Gaussian**: 最高精度，中小型数据集
- **Achlioptas**: 良好平衡，中型数据集
- **Sparse**: 高维基因组/文本数据
- **CountSketch**: 流式/在线应用

## 预计算矩阵投影

### create_gaussian_projection

创建密集高斯随机投影矩阵：

```cpp
#include "scl/kernel/projection.hpp"

auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim,   // 原始维度 d
    output_dim,  // 目标维度 k
    42           // seed
);
```

**参数：**
- `input_dim`: 原始维度 d（特征数量）
- `output_dim`: 目标维度 k（减少的特征）
- `seed`: 用于可重现性的随机种子

**返回：** 具有按 1/sqrt(k) 缩放的高斯条目的 `ProjectionMatrix<T>`

**后置条件：**
- E[||Rx - Ry||^2] = ||x - y||^2（无偏）
- 给定相同种子时确定性

**复杂度：**
- 时间：O(input_dim * output_dim) 用于生成
- 空间：O(input_dim * output_dim)

**使用场景：**
- 最高质量投影
- 当可以存储投影矩阵时
- 小到中等输入维度

### create_achlioptas_projection

创建稀疏 Achlioptas 随机投影矩阵：

```cpp
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    input_dim,
    output_dim,
    42  // seed
);
```

**优势：**
- 比高斯快 3 倍
- 2/3 的条目为零
- 与高斯相同的理论保证

**参考：**
Achlioptas, D. (2003). Database-friendly random projections.

**使用场景：**
- 速度和质量的良好平衡
- 中型数据集
- 当存储是考虑因素时

### create_sparse_projection

创建非常稀疏的随机投影矩阵：

```cpp
auto proj = scl::kernel::projection::create_sparse_projection<Real>(
    input_dim,
    output_dim,
    Real(1.0 / sqrt(input_dim)),  // density
    42  // seed
);
```

**优势：**
- 对于高维数据快 sqrt(d) 倍
- (1-density) 部分的计算被跳过
- 相同的距离保持保证

**参考：**
Li, P., Hastie, T. J., & Church, K. W. (2006). Very sparse random projections.

**使用场景：**
- 非常高维数据（d > 10000）
- 基因组/文本数据
- 当速度至关重要时

### project_with_matrix

使用预计算的投影矩阵投影稀疏矩阵：

```cpp
Sparse<Real, true> matrix = /* ... */;  // n x d
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    matrix.cols(),  // d
    output_dim      // k
);
Array<Real> output(matrix.rows() * output_dim);  // 预分配

scl::kernel::projection::project_with_matrix(
    matrix,
    proj,
    output
);
```

**参数：**
- `matrix`: CSR 稀疏矩阵 X，形状 (n_rows x n_cols)
- `proj`: 预计算的投影矩阵 R，形状 (n_cols x output_dim)
- `output`: 密集输出缓冲区 Y，大小 = n_rows * output_dim，预分配

**后置条件：**
- `output[i*k ... (i+1)*k-1]` 包含投影行 i
- Y = X * R

**算法：**
在行上并行：
1. 对于每行 i：
   - 将 output_row 初始化为零
   - 对于行 i 中的每个非零 (j, v)：
     - output_row += v * R[j, :]
   - 对于大 output_dim 使用 SIMD FMA

**复杂度：**
- 时间：O(nnz * output_dim)
- 空间：O(1) 辅助空间

**性能说明：**
- 对于 output_dim >= 64 的 SIMD 4 路展开累加
- 为长稀疏行预取投影行
- 当投影矩阵适合缓存时最佳

## 即时投影

### project_gaussian_otf

使用即时高斯随机生成投影稀疏矩阵：

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> output(matrix.rows() * output_dim);

scl::kernel::projection::project_gaussian_otf(
    matrix,
    output_dim,
    output,
    42  // seed
);
```

**优势：**
- O(1) 辅助内存（不存储投影矩阵）
- 给定相同种子时确定性
- 适合非常高维数据

**劣势：**
- 对于重复投影比预计算慢
- 更多随机数生成开销

**复杂度：**
- 时间：O(nnz * output_dim)，常数更高
- 空间：O(1) 辅助空间

**使用场景：**
- 存储矩阵不切实际的非常高维数据
- 一次性投影
- 内存受限环境

### project_achlioptas_otf

即时 Achlioptas 投影（三元：+1, 0, -1）：

```cpp
scl::kernel::projection::project_achlioptas_otf(
    matrix,
    output_dim,
    output,
    42  // seed
);
```

**优势：**
- 比高斯 OTF 更快（更简单的随机生成）
- 2/3 的随机值为零（跳过计算）

**使用场景：**
- 中等维数据
- 当内存有限时
- 高斯的更快替代方案

### project_sparse_otf

具有自定义密度的即时非常稀疏投影：

```cpp
scl::kernel::projection::project_sparse_otf(
    matrix,
    output_dim,
    output,
    Real(1.0 / sqrt(matrix.cols())),  // density
    42  // seed
);
```

**优势：**
- 最适合非常高维数据（d > 10000）
- 由于稀疏性，大部分计算被跳过

**使用场景：**
- 极高维数据
- 当速度至关重要时
- 基因组/文本应用

## Count-Sketch 投影

### project_countsketch

使用基于哈希的分桶和符号翻转的 Count-Sketch 投影：

```cpp
scl::kernel::projection::project_countsketch(
    matrix,
    output_dim,  // 桶数量
    output,
    42  // seed
);
```

**优势：**
- O(nnz) 时间（不是 O(nnz * k)）
- 无偏：E[Y^T Y] = X^T X
- 适合流式/在线学习

**劣势：**
- 对于小 k，方差高于高斯
- 碰撞效应降低精度

**复杂度：**
- 时间：O(nnz)
- 空间：O(1) 辅助空间

**参考：**
Charikar, M., Chen, K., & Farach-Colton, M. (2004). Finding frequent items in data streams.

**使用场景：**
- 流式数据
- 在线学习
- 当 O(nnz * k) 过于昂贵时

## 高级接口

### project

稀疏随机投影的统一接口：

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> output(matrix.rows() * output_dim);

scl::kernel::projection::project(
    matrix,
    output_dim,
    output,
    ProjectionType::Sparse,  // type
    42  // seed
);
```

**参数：**
- `matrix`: CSR 稀疏矩阵 X
- `output_dim`: 目标维度 k
- `output`: 密集输出缓冲区 [n * k]
- `type`: 投影类型（默认：Sparse）
- `seed`: 随机种子（默认：42）

**后置条件：**
- 距离保持：(1-eps)||x-y|| <= ||Rx-Ry|| <= (1+eps)||x-y||
- 高概率

**选择：**
- Sparse 类型使用密度 = max(1/sqrt(cols), 0.01)

**使用场景：**
- 常见用例的简单接口
- 自动方法选择
- 快速原型

## 工具函数

### compute_jl_dimension

计算 Johnson-Lindenstrauss 保证的最小目标维度：

```cpp
Size k = scl::kernel::projection::compute_jl_dimension(
    n_samples,  // 数据点数量
    Real(0.1)   // epsilon（距离失真容差）
);
```

**参数：**
- `n_samples`: 数据点数量
- `epsilon`: 最大相对距离失真（默认：0.1）

**返回：** 用于 (1 +/- epsilon) 距离保持的最小目标维度 k

**保证：**
概率 >= 1 - 1/n^2：
(1-epsilon)||x-y||^2 <= ||Rx-Ry||^2 <= (1+epsilon)||x-y||^2
对于所有对 (x, y)

**典型值：**
- n=1000, eps=0.1: k ~= 300
- n=10000, eps=0.1: k ~= 400
- n=1000, eps=0.5: k ~= 20

**参考：**
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space.

**使用场景：**
- 确定距离保持所需的维度
- 投影的质量控制
- 理论保证

## 配置

`scl::kernel::projection::config` 中的默认参数：

```cpp
namespace config {
    constexpr Size SIMD_THRESHOLD = 64;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size SMALL_OUTPUT_DIM = 32;
    constexpr Real DEFAULT_EPSILON = 0.1;
}
```

## 性能考虑

### 内存效率

- **预计算**: O(d * k) 存储，更快投影
- **即时**: O(1) 存储，更慢投影
- 根据输入维度和投影重用选择

### SIMD 优化

- 对于 output_dim >= 64 的 4 路展开累加
- 小 output_dim (< 32) 的标量路径
- 预取以提高缓存效率

## 最佳实践

### 1. 选择适当的方法

```cpp
// 最高精度
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim, output_dim
);

// 速度与质量平衡
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    input_dim, output_dim
);

// 非常高维数据
scl::kernel::projection::project_sparse_otf(
    matrix, output_dim, output, 1.0/sqrt(input_dim)
);
```

### 2. 计算 JL 维度

```cpp
// 确定所需维度
Size k = scl::kernel::projection::compute_jl_dimension(
    n_samples,
    0.1  // 10% 距离失真
);

// 使用计算的维度
scl::kernel::projection::project(matrix, k, output);
```

### 3. 对重复投影使用预计算

```cpp
// 构建一次
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim, output_dim
);

// 为多个矩阵重用
for (auto& matrix : matrices) {
    scl::kernel::projection::project_with_matrix(matrix, proj, output);
}
```

---

::: tip 方法选择
使用 Gaussian 获得最高精度，Achlioptas 获得平衡，Sparse 用于高维数据，CountSketch 用于流式应用。
:::

::: warning 内存
预计算的投影矩阵需要 O(d * k) 存储。对于非常高维数据（d > 100000），使用即时方法。
:::

