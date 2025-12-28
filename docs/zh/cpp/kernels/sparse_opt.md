# 稀疏优化

用于线性代数问题的稀疏矩阵优化操作。

## 概览

稀疏优化内核提供：

- **最小二乘求解器** - 求解稀疏最小二乘问题
- **迭代方法** - 基于收敛的求解器
- **SIMD 优化** - 向量化操作
- **并行处理** - 针对大型系统高效

## 最小二乘求解器

### sparse_least_squares

求解稀疏最小二乘问题：`min ||Ax - b||^2`

```cpp
#include "scl/kernel/sparse_opt.hpp"

Sparse<Real, true> A = /* ... */;      // 稀疏矩阵 [n_rows x n_cols]
Array<const Real> b = /* ... */;        // 右端项 [n_rows]
Array<Real> x(A.cols());                // 解向量 [n_cols]

// 标准求解器
scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, A.rows(), A.cols(),
                                              x, max_iter = 100, tol = 1e-6);

// 自定义容差
scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, A.rows(), A.cols(),
                                              x, max_iter = 200, tol = 1e-8);
```

**参数：**
- `A`: 稀疏矩阵（CSR 格式），大小 = n_rows × n_cols
- `b`: 右端项向量，大小 = n_rows
- `n_rows`: 行数
- `n_cols`: 列数
- `x`: 输出解向量，必须预分配，大小 = n_cols
- `max_iter`: 最大迭代次数（默认：100）
- `tol`: 收敛容差（默认：1e-6）

**后置条件：**
- `x` 包含 `Ax ≈ b` 的近似解
- 解最小化 `||Ax - b||^2`
- 矩阵 A 和向量 b 不变

**算法：**
迭代求解器（通常是共轭梯度或 LSQR）：
1. 初始化解向量
2. 迭代直到收敛：
   - 计算残差：`r = b - Ax`
   - 更新解：`x = x + alpha * direction`
   - 检查收敛：`||r|| < tol`
3. 在收敛或达到 max_iter 时返回

**复杂度：**
- 时间：O(max_iter * nnz) 每次迭代
- 空间：O(n_cols) 辅助空间用于解和工作空间

**线程安全：**
- 安全 - 内部使用并行化的 SpMV 操作

**用例：**
- 带稀疏设计矩阵的线性回归
- 超定系统（方程数多于未知数）
- 正则化最小二乘
- 矩阵分解问题

## 配置

### 默认参数

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
}
```

**收敛：**
- 当残差范数 < `tol` 时算法停止
- 数值稳定性阈值：`EPSILON`

**性能：**
- 对于 > `PARALLEL_THRESHOLD` 行的矩阵进行并行处理
- 对于 > `SIMD_THRESHOLD` 元素的密集操作进行 SIMD 优化

## 示例

### 线性回归

```cpp
#include "scl/kernel/sparse_opt.hpp"

// 设计矩阵（稀疏特征）
Sparse<Real, true> X = /* ... */;  // [n_samples x n_features]
Array<Real> y = /* ... */;          // 目标值 [n_samples]

// 求解: X * beta = y
Array<Real> beta(X.cols());
scl::kernel::sparse_opt::sparse_least_squares(X, y.ptr, X.rows(), X.cols(),
                                              beta, max_iter = 100, tol = 1e-6);

// beta 包含回归系数
```

### 超定系统

```cpp
// 方程数多于未知数: Ax = b，其中 A 是 [m x n]，m > n
Sparse<Real, true> A = /* ... */;  // [1000 x 100]
Array<Real> b = /* ... */;          // [1000]
Array<Real> x(100);                 // 解 [100]

scl::kernel::sparse_opt::sparse_least_squares(A, b.ptr, 1000, 100,
                                              x, max_iter = 200, tol = 1e-8);
```

## 性能考虑

### 收敛

- 对于良条件问题，通常在 10-100 次迭代中收敛
- 病态矩阵可能需要更多迭代或预处理
- 使用 `tol` 平衡精度与计算时间

### 矩阵结构

- 稀疏矩阵（低密度）的最佳性能
- 需要 CSR 格式（行主访问模式）
- 矩阵-向量乘积是主要操作

### 并行化

- SpMV 操作并行化
- 多次迭代是顺序的（每次依赖于前一次）
- 对大型稀疏系统最优

---

::: tip 收敛
对于病态系统，考虑预处理（中心化、缩放）或使用预条件子。默认容差（1e-6）对大多数应用通常足够。
:::

