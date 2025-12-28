# 稀疏线性代数

高性能稀疏矩阵-向量乘法内核，采用 SIMD 优化。

## 概览

代数内核提供：

- **稀疏矩阵-向量乘法 (SpMV)** - 核心线性代数运算
- **Alpha/Beta 缩放** - 灵活的缩放和累积
- **自适应点积** - 针对不同行长度优化
- **SIMD 加速** - 向量化操作以实现最大吞吐量

## 稀疏矩阵-向量乘法

### spmv

功能完整的稀疏矩阵-向量乘法，支持 alpha 和 beta 缩放：

```cpp
#include "scl/kernel/algebra.hpp"

Sparse<Real, true> A = /* ... */;  // CSR 矩阵
Array<const Real> x = /* ... */;    // 输入向量 [secondary_dim]
Array<Real> y = /* ... */;          // 输出向量 [primary_dim]，需预分配

// 计算: y = alpha * A * x + beta * y
scl::kernel::algebra::spmv(A, x, y, alpha, beta);

// 简单乘法: y = A * x
scl::kernel::algebra::spmv(A, x, y, Real(1), Real(0));

// 累积: y += A * x
scl::kernel::algebra::spmv(A, x, y, Real(1), Real(1));
```

**计算：** `y = alpha * A * x + beta * y`

**参数：**
- `A`: 稀疏矩阵（CSR 或 CSC）
- `x`: 输入向量，大小 = `secondary_dim(A)`
- `y`: 输出向量，大小 = `primary_dim(A)`，必须预分配
- `alpha`: `A*x` 的标量乘数（默认：`T(1)`）
- `beta`: `y` 的标量乘数（默认：`T(0)`）

**性能：**
- SIMD 优化的 beta 缩放（高效处理 beta=0、beta=1 情况）
- 基于行/列长度的自适应点积策略：
  - 短行（nnz < 8）：标量循环
  - 中等行（8-64）：4 路展开循环
  - 长行（64-256）：8 路展开循环，带预取
  - 超长行（>= 256）：8 路展开，激进预取
- 跨主维度并行处理
- 连续索引检测以启用密集 SIMD 路径

**用例：**
- 迭代求解器（共轭梯度、GMRES）
- 图算法（PageRank、随机游走）
- 神经网络前向/反向传播
- 稀疏线性系统

### spmv_simple

简化的稀疏矩阵-向量乘法：`y = A * x`

```cpp
// 等价于 spmv(A, x, y, T(1), T(0))
scl::kernel::algebra::spmv_simple(A, x, y);
```

**用例：**
- 简单矩阵-向量乘积
- 需要覆盖 y 的情况（不累积）

### spmv_add

累积稀疏矩阵-向量乘积：`y += A * x`

```cpp
// 等价于 spmv(A, x, y, T(1), T(1))
scl::kernel::algebra::spmv_add(A, x, y);
```

**用例：**
- 累积来自多个矩阵的贡献
- 增量更新 y 的迭代算法
- 分块矩阵运算

### spmv_scaled

缩放稀疏矩阵-向量乘法：`y = alpha * A * x`

```cpp
// 等价于 spmv(A, x, y, alpha, T(0))
scl::kernel::algebra::spmv_scaled(A, x, y, alpha);
```

**用例：**
- 需要缩放乘积但不需累积的情况
- 缩放线性变换

## 性能优化

### 自适应点积策略

实现根据行/列长度使用不同算法：

```cpp
// 短行：标量循环（最小开销）
if (nnz < 8) {
    // 简单标量累积
}

// 中等行：4 路展开（平衡开销和并行性）
else if (nnz < 64) {
    // 4 路展开循环
}

// 长行：8 路展开 + 预取
else if (nnz < 256) {
    // 8 路展开循环，带预取
}

// 超长行：激进预取
else {
    // 8 路展开 + x 向量的激进预取
}
```

### 连续索引优化

对于具有连续索引的行，算法使用密集 SIMD 路径：

```cpp
// 检测连续索引
if (is_consecutive(indices, nnz)) {
    // 使用密集 SIMD 点积（更快）
    return dense_dot(values, x + indices[0], nnz);
}
```

此优化为结构化矩阵（例如网格图的邻接矩阵）提供显著加速。

### SIMD Beta 缩放

Beta 缩放针对常见情况优化：

```cpp
if (beta == 0) {
    // 零填充（SIMD 优化）
    zero_fill(y, n);
} else if (beta == 1) {
    // 无操作（提前返回）
    return;
} else {
    // 通用缩放（SIMD 4 路展开，带预取）
    scale_output(y, n, beta);
}
```

## 示例

### 迭代求解器

```cpp
#include "scl/kernel/algebra.hpp"

Sparse<Real, true> A = /* ... */;
Array<const Real> b = /* ... */;  // 右端项
Array<Real> x = /* ... */;        // 解向量
Array<Real> r = /* ... */;        // 残差
Array<Real> p = /* ... */;        // 搜索方向
Array<Real> Ap = /* ... */;       // A * p

// 共轭梯度迭代
for (int iter = 0; iter < max_iter; ++iter) {
    // 计算 A * p
    scl::kernel::algebra::spmv_simple(A, p, Ap);
    
    // ... 计算 alpha，更新 x, r ...
    
    // 计算残差: r = b - A * x
    scl::memory::copy(r, b);
    scl::kernel::algebra::spmv_add(A, x, r, Real(-1), Real(1));
}
```

### 多个矩阵贡献

```cpp
// 累积来自多个稀疏矩阵的贡献
Array<Real> y = /* ... */;  // 初始化为零

// y += A1 * x
scl::kernel::algebra::spmv_add(A1, x, y);

// y += A2 * x
scl::kernel::algebra::spmv_add(A2, x, y);

// y += A3 * x
scl::kernel::algebra::spmv_add(A3, x, y);
```

### 缩放变换

```cpp
// 应用缩放变换: y = 0.5 * A * x
Real scale = 0.5;
scl::kernel::algebra::spmv_scaled(A, x, y, scale);
```

## 性能考虑

### 内存布局

- **CSR 格式**：行操作的最优选择（y = A * x）
- **CSC 格式**：列操作的最优选择（y = A^T * x）
- **缓存局部性**：访问模式取决于矩阵格式

### 并行化

- 自动跨主维度并行化
- 每个线程处理独立的行/列
- 无需同步（不同的输出元素）

### 数值稳定性

- 使用标准浮点运算
- 并行执行中的累积顺序是非确定性的
- 对于非常大的矩阵，考虑使用补偿求和

---

::: tip 性能提示
对于具有连续索引模式的矩阵（例如网格图），实现会自动检测并使用优化的密集 SIMD 路径，提供显著加速。
:::

