# sparse_opt.hpp

> scl/kernel/sparse_opt.hpp · 稀疏矩阵优化操作

## 概述

本文件为线性代数问题提供稀疏矩阵优化操作，特别是使用迭代方法的稀疏最小二乘求解器。

本文件提供：
- 使用迭代方法的稀疏最小二乘求解器
- 基于收敛的优化
- SIMD 优化的稀疏矩阵-向量操作
- 并行处理支持

**头文件**: `#include "scl/kernel/sparse_opt.hpp"`

---

## 主要 API

### sparse_least_squares

::: source_code file="scl/kernel/sparse_opt.hpp" symbol="sparse_least_squares" collapsed
:::

**算法说明**

使用迭代方法解决稀疏最小二乘问题：最小化 ||Ax - b||^2：

1. **初始化**：初始化解向量 x（通常为零）
2. **迭代细化**：每次迭代计算残差并更新解
3. **收敛检查**：如果残差范数 < 容差则退出
4. **优化**：使用并行化 SpMV 提高效率

**边界条件**

- **空矩阵**：如果 A 没有行，返回零解
- **零右端项**：如果 b = 0，返回零解
- **奇异矩阵**：如果 A 奇异（病态），可能不收敛
- **无收敛**：在 max_iter 次迭代后返回最佳解

**数据保证（前置条件）**

- `A` 必须是有效的 CSR 稀疏矩阵
- `x` 必须预分配，容量 >= n_cols
- `b` 长度 >= n_rows
- `max_iter > 0`, `tol > 0`

**复杂度分析**

- **时间**：O(max_iter * nnz)
- **空间**：O(n_cols) 辅助空间

**示例**

```cpp
#include "scl/kernel/sparse_opt.hpp"

Sparse<Real, true> A = /* ... */;
Array<Real> b(n_rows);
Array<Real> x(n_cols);

scl::kernel::sparse_opt::sparse_least_squares(
    A, b.data(), n_rows, n_cols, x, 100, 1e-6
);
```

---

## 注意事项

**收敛**：
- 典型收敛需要 10-100 次迭代，取决于条件数
- 更小的容差需要更多迭代
- 病态矩阵可能不收敛

**性能**：
- 使用并行化 SpMV 提高效率
- SIMD 优化（如适用）
- 内存高效的稀疏存储

## 相关内容

- [Sparse Matrix Operations](/zh/cpp/kernels/sparse) - 通用稀疏矩阵工具
- [Linear Algebra](/zh/cpp/math) - 其他线性代数操作
