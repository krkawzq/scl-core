# 重采样

使用快速 RNG 进行下采样和随机变换的重采样操作。

## 概述

`resample` 模块为单细胞数据提供高效的重采样操作：

- **下采样**: 将每个细胞的总计数减少到目标值
- **二项式重采样**: 以固定概率重采样计数
- **泊松重采样**: 使用缩放泊松分布重采样计数

所有操作都：
- 基于快速 RNG（xoshiro256++）
- 按行并行化，每个线程有独立的 RNG 状态
- 原地修改（无内存分配）
- 给定种子时具有确定性

## 函数

### downsample

使用二项式采样将每行下采样到目标总计数。

```cpp
#include "scl/kernel/resample.hpp"

Sparse<Real, true> matrix = /* 表达矩阵 */;

// 将每个细胞下采样到 10,000 总计数
scl::kernel::resample::downsample(matrix, 10000.0, 42);
```

**参数:**
- `matrix` [in,out] - 表达矩阵，原地修改
- `target_sum` [in] - 每行的目标总计数
- `seed` [in] - 用于可重现性的随机种子（默认: 42）

**前置条件:**
- `target_sum > 0`
- 矩阵值必须可变

**后置条件:**
- 每行的总计数约等于 target_sum
- 矩阵结构（indices, indptr）不变
- 给定种子时采样是确定性的

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按行并行化，每个线程有独立的 RNG 状态

**算法:**
对每行并行处理：
1. 计算当前总计数
2. 如果当前 <= 目标，跳过
3. 对每个非零元素：
   a. 计算概率 = remaining_target / remaining_total
   b. 采样二项式(count, probability)
   c. 更新值和剩余计数

### downsample_variable

使用二项式采样将每行下采样到可变目标计数。

```cpp
Array<Real> target_counts(n_cells);
// 为每个细胞设置不同的目标
for (Index i = 0; i < n_cells; ++i) {
    target_counts.ptr[i] = compute_target_for_cell(i);
}

scl::kernel::resample::downsample_variable(matrix, target_counts, 42);
```

**参数:**
- `matrix` [in,out] - 表达矩阵，原地修改
- `target_counts` [in] - 每行的目标计数 [n_rows]
- `seed` [in] - 用于可重现性的随机种子（默认: 42）

**前置条件:**
- `target_counts.len >= matrix.rows()`
- 所有 `target_counts[i] > 0`
- 矩阵值必须可变

**后置条件:**
- 行 i 的总计数约等于 `target_counts[i]`
- 矩阵结构（indices, indptr）不变
- 给定种子时采样是确定性的

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按行并行化，每个线程有独立的 RNG 状态

### binomial_resample

使用固定概率的二项式分布重采样每个计数值。

```cpp
// 以 50% 概率重采样
scl::kernel::resample::binomial_resample(matrix, 0.5, 42);
```

**参数:**
- `matrix` [in,out] - 表达矩阵，原地修改
- `p` [in] - 二项式采样的成功概率
- `seed` [in] - 用于可重现性的随机种子（默认: 42）

**前置条件:**
- `p` 在 [0, 1] 范围内
- 矩阵值必须可变

**后置条件:**
- 每个值被替换为 binomial(value, p)
- 矩阵结构（indices, indptr）不变
- 给定种子时采样是确定性的

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按行并行化，每个线程有独立的 RNG 状态

**算法:**
对每行并行处理：
- 对每个非零元素：
  1. 采样 binomial(count, p)
  2. 用采样计数替换值

### poisson_resample

使用缩放均值的泊松分布重采样每个计数值。

```cpp
// 使用 lambda = 0.8 重采样（减少 20% 计数）
scl::kernel::resample::poisson_resample(matrix, 0.8, 42);
```

**参数:**
- `matrix` [in,out] - 表达矩阵，原地修改
- `lambda` [in] - 泊松均值的缩放因子（均值 = count * lambda）
- `seed` [in] - 用于可重现性的随机种子（默认: 42）

**前置条件:**
- `lambda > 0`
- 矩阵值必须可变

**后置条件:**
- 每个值被替换为 Poisson(value * lambda)
- 矩阵结构（indices, indptr）不变
- 给定种子时采样是确定性的

**复杂度:**
- 时间: O(nnz)
- 空间: O(1) 辅助空间

**线程安全:** 安全 - 按行并行化，每个线程有独立的 RNG 状态

**算法:**
对每行并行处理：
- 对每个非零元素：
  1. 计算均值 = count * lambda
  2. 采样 Poisson(mean)
  3. 用采样计数替换值

## 配置

```cpp
namespace scl::kernel::resample::config {
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## 随机数生成器

模块使用快速的 xoshiro256++ RNG：
- **线程安全**: 每个并行工作线程有独立的 RNG 状态
- **确定性**: 相同种子产生相同序列
- **快速**: 针对高吞吐量采样优化

## 使用场景

### 标准下采样

```cpp
// 将所有细胞下采样到相同深度
Sparse<Real, true> expression = /* ... */;
scl::kernel::resample::downsample(expression, 10000.0, 42);
```

### 可变深度下采样

```cpp
// 根据细胞类型下采样到不同深度
Array<Real> targets(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    if (cell_types[i] == "T_cell") {
        targets.ptr[i] = 5000.0;
    } else {
        targets.ptr[i] = 10000.0;
    }
}
scl::kernel::resample::downsample_variable(expression, targets, 42);
```

### 随机变换

```cpp
// 应用二项式重采样进行噪声注入
scl::kernel::resample::binomial_resample(expression, 0.9, 42);

// 应用泊松重采样进行计数缩放
scl::kernel::resample::poisson_resample(expression, 0.8, 42);
```

### 可重现分析

```cpp
// 使用相同种子以确保可重现性
uint64_t seed = 42;

// 下采样
scl::kernel::resample::downsample(matrix1, 10000.0, seed);

// 处理...

// 使用相同种子下采样另一个数据集
scl::kernel::resample::downsample(matrix2, 10000.0, seed);
```

## 性能

- **快速 RNG**: xoshiro256++ 比标准库 RNG 快 2-3 倍
- **并行化**: 随 CPU 核心数线性扩展
- **零分配**: 所有操作都是原地进行
- **确定性**: 相同种子产生可重现的结果

---

::: tip 可重现性
始终指定种子以获得可重现的结果。相同种子和相同输入产生相同的输出。
:::

