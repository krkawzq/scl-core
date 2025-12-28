# log1p.hpp

> scl/kernel/log1p.hpp · 带 SIMD 优化的对数变换核

## 概述

本文件为稀疏矩阵提供高效的对数变换操作。所有操作都经过 SIMD 加速，在行上并行化，并就地修改矩阵，无需内存分配。

**头文件**: `#include "scl/kernel/log1p.hpp"`

---

## 主要 API

### log1p_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="log1p_inplace" collapsed
:::

**算法说明**

对稀疏矩阵中的所有非零值应用 log(1 + x) 变换：

1. 对于每行并行处理：
   - 使用 4 路 SIMD 展开和预取加载值
   - 对值向量应用 SIMD Log1p 操作
   - 将变换后的值存储回去
   - 使用标量操作处理尾部元素

2. 使用数值稳定的 log1p 实现以确保接近零时的精度

3. 零值保持为零（稀疏格式中不存储）

**边界条件**

- **空矩阵**: 立即返回，不修改
- **值 < -1**: 结果为 NaN（log1p 域要求）
- **值 = -1**: 结果为 -infinity
- **非常小的值**: 对于接近零的 x，比 log(1+x) 更准确
- **大值**: 对于大 x，log1p(x) ≈ log(x)

**数据保证（前置条件）**

- 矩阵值必须 >= -1（log1p 域要求）
- 对于表达数据：值应为非负计数
- 矩阵必须是有效的 CSR 或 CSC 格式
- 矩阵值必须可变

**复杂度分析**

- **时间**: O(nnz) - 处理每个非零元素一次
- **空间**: O(1) 辅助空间 - 仅 SIMD 寄存器和临时变量

**示例**

```cpp
#include "scl/kernel/log1p.hpp"

// 加载或创建稀疏矩阵
Sparse<Real, true> expression = /* ... */;  // CSR 格式

// 就地应用 log1p 变换
scl::kernel::log1p::log1p_inplace(expression);

// 所有非零值 v 现在是 log(1 + v)
// 矩阵结构（indices, indptr）不变
// 准备进行下游分析（PCA、聚类等）

// 标准预处理流程
scl::kernel::normalize::normalize_total_inplace(expression, 1e4);
scl::kernel::log1p::log1p_inplace(expression);
// 现在可以用于 PCA、聚类等
```

---

### log2p1_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="log2p1_inplace" collapsed
:::

**算法说明**

对稀疏矩阵中的所有非零值应用 log2(1 + x) 变换：

1. 对于每行并行处理：
   - 使用 4 路 SIMD 展开和预取加载值
   - 应用 SIMD Log1p 操作
   - 乘以 INV_LN2 (1/ln(2)) 转换为以 2 为底
   - 存储变换后的值
   - 使用标量操作处理尾部元素

2. 计算为 log(1+x) * (1/ln(2)) 以提高效率

3. 以 2 为底的对数在信息论应用中很常见

**边界条件**

- **空矩阵**: 立即返回
- **值 < -1**: 结果为 NaN
- **值 = -1**: 结果为 -infinity
- **非常小的值**: 对于接近零的 x，比 log2(1+x) 更准确

**数据保证（前置条件）**

- 矩阵值必须 >= -1
- 对于表达数据：值应为非负计数
- 矩阵必须是有效的 CSR 或 CSC 格式
- 矩阵值必须可变

**复杂度分析**

- **时间**: O(nnz) - 处理每个非零元素一次
- **空间**: O(1) 辅助空间 - 仅 SIMD 寄存器

**示例**

```cpp
#include "scl/kernel/log1p.hpp"

Sparse<Real, true> matrix = /* ... */;

// 应用以 2 为底的对数变换
scl::kernel::log1p::log2p1_inplace(matrix);

// 所有非零值 v 现在是 log2(1 + v)
// 用于信息论度量（熵、互信息）
```

---

### expm1_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="expm1_inplace" collapsed
:::

**算法说明**

对稀疏矩阵中的所有非零值应用 exp(x) - 1 变换：

1. 对于每行并行处理：
   - 使用 4 路 SIMD 展开和预取加载值
   - 对值向量应用 SIMD Expm1 操作
   - 将变换后的值存储回去
   - 使用标量操作处理尾部元素

2. 使用数值稳定的 expm1 实现以确保接近零时的精度

3. log1p 的逆：expm1(log1p(x)) = x

**边界条件**

- **空矩阵**: 立即返回
- **非常大的值**: 可能溢出到无穷大
- **非常小的值**: 对于接近零的 x，比 exp(x)-1 更准确
- **值 = 0**: 结果为 0（exp(0) - 1 = 0）
- **负大值**: 结果为 -1（exp(-inf) - 1 = -1）

**数据保证（前置条件）**

- 值应在合理范围内以避免溢出
- 通常用于反转 log1p 变换
- 矩阵必须是有效的 CSR 或 CSC 格式
- 矩阵值必须可变

**复杂度分析**

- **时间**: O(nnz) - 处理每个非零元素一次
- **空间**: O(1) 辅助空间 - 仅 SIMD 寄存器

**示例**

```cpp
#include "scl/kernel/log1p.hpp"

Sparse<Real, true> matrix = /* ... */;

// 应用 log1p 变换
scl::kernel::log1p::log1p_inplace(matrix);

// ... 对对数变换的数据执行分析 ...

// 反转变换回原始尺度
scl::kernel::log1p::expm1_inplace(matrix);

// 矩阵值现在回到原始尺度（近似）
// expm1(log1p(x)) = x（对于小 x 精确）
```

---

## 配置

### 默认参数

```cpp
namespace scl::kernel::log1p::config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;  // 1/ln(2)
    constexpr double LN2 = 0.6931471805599453;          // ln(2)
}
```

---

## 注意事项

**数值稳定性**: log1p 和 expm1 对于接近零的小值比 log(1+x) 和 exp(x)-1 更准确。这对于许多值为小计数的表达数据至关重要。

**性能**: SIMD 加速在现代 CPU 上提供 4-8 倍加速。并行化随 CPU 核心数线性扩展。所有操作都是就地执行，零分配。

**矩阵格式**: 为获得最佳性能，对按行操作使用 CSR 格式（行主序）。实现针对 CSR 进行了优化，但也适用于 CSC。

**使用场景**:
- **log1p**: 单细胞 RNA-seq 数据的标准预处理
- **log2p1**: 信息论度量（熵、互信息）
- **expm1**: 反转对数变换，转换回计数尺度

---

## 相关内容

- [Normalize](/zh/cpp/kernels/normalize) - 对数变换前的归一化
- [Softmax](/zh/cpp/kernels/softmax) - Softmax 归一化
