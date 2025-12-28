# 特征统计

稀疏矩阵的特征级统计计算，采用 SIMD 优化。

## 概览

特征统计内核提供：

- **均值和方差** - 计算考虑隐式零的矩
- **裁剪矩** - 带逐行裁剪的统计
- **检出率** - 非零条目比例
- **离散度** - 方差-均值比
- **SIMD 加速** - 融合的求和与平方和计算

## 标准矩

### standard_moments

计算每个主维度的均值和方差，考虑隐式零：

```cpp
#include "scl/kernel/feature.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<Real> means = /* ... */;  // 预分配 [primary_dim]
Array<Real> vars = /* ... */;   // 预分配 [primary_dim]

// 使用总体方差计算（ddof=0）
scl::kernel::feature::standard_moments(matrix, means, vars, 0);

// 使用样本方差计算（ddof=1）
scl::kernel::feature::standard_moments(matrix, means, vars, 1);
```

**参数：**
- `matrix`: 稀疏矩阵（CSR 或 CSC）
- `out_means`: 均值缓冲区，大小 = `primary_dim`
- `out_vars`: 方差缓冲区，大小 = `primary_dim`
- `ddof`: 自由度增量（0 为总体，1 为样本方差）

**后置条件：**
- `out_means[i] = sum(row_i) / secondary_dim`
- `out_vars[i] = var(row_i)`，使用指定的 `ddof`，限制为 >= 0
- 空行的均值为 0，方差为 0

**算法：**
- 对每个主索引并行：
  1. 使用 SIMD 计算融合的求和与平方和（4 路展开）
  2. 计算均值 = sum / N（其中 N = secondary_dim）
  3. 计算方差 = (sumsq - sum * mean) / (N - ddof)
  4. 将方差限制为非负值

**用例：**
- 质量控制指标
- 特征标准化
- 基于方差的特征选择
- 统计分析

### clipped_moments

计算带逐行最大值裁剪的均值和方差：

```cpp
Array<const Real> clip_vals = /* ... */;  // 逐行裁剪阈值 [primary_dim]
Array<Real> means = /* ... */;
Array<Real> vars = /* ... */;

// 计算裁剪矩（使用 ddof=1）
scl::kernel::feature::clipped_moments(matrix, clip_vals, means, vars);
```

**参数：**
- `matrix`: 稀疏矩阵
- `clip_vals`: 逐行裁剪阈值，大小 = `primary_dim`
- `out_means`: 输出均值缓冲区
- `out_vars`: 输出方差缓冲区

**后置条件：**
- 值在统计前裁剪为 `min(value, clip_vals[i])`
- 使用 ddof=1 计算方差
- 使用 SIMD min 操作进行高效裁剪

**用例：**
- 稳健统计（抗异常值）
- 带阈值上限的值分析
- 带阈值质量控制

## 检出率

### detection_rate

计算每个主维度的非零条目比例：

```cpp
Array<Real> rates = /* ... */;  // 预分配 [primary_dim]

scl::kernel::feature::detection_rate(matrix, rates);
```

**参数：**
- `matrix`: 稀疏矩阵
- `out_rates`: 检出率缓冲区

**后置条件：**
- `out_rates[i] = nnz_in_row_i / secondary_dim`
- 值在 [0, 1] 范围内
- 空行的比率为 0

**用例：**
- 基因表达检出率
- 特征稀疏性分析
- 质量过滤

## 离散度

### dispersion

计算每个特征的离散度指数（方差 / 均值）：

```cpp
Array<const Real> means = /* ... */;      // 预计算的均值
Array<const Real> vars = /* ... */;       // 预计算的方差
Array<Real> dispersion = /* ... */;       // 预分配输出

scl::kernel::feature::dispersion(means, vars, dispersion);
```

**参数：**
- `means`: 预计算的均值
- `vars`: 预计算的方差
- `out_dispersion`: 预分配输出缓冲区

**后置条件：**
- `out_dispersion[i] = vars[i] / means[i]` 如果 `means[i] > epsilon`
- `out_dispersion[i] = 0` 如果 `means[i] <= epsilon`
- 使用 epsilon = 1e-12 避免除零

**算法：**
- SIMD 4 路展开，带掩码除法
- 为 mean > epsilon 创建掩码
- 使用掩码选择计算除法

**用例：**
- 高变异性基因检测
- 特征选择（高离散度 = 信息丰富）
- 质量指标

## 示例

### 质量控制流程

```cpp
#include "scl/kernel/feature.hpp"

Sparse<Real, true> counts = /* ... */;

// 计算统计
Array<Real> means(counts.rows());
Array<Real> vars(counts.rows());
Array<Real> detection_rates(counts.rows());

scl::kernel::feature::standard_moments(counts, means, vars, 1);
scl::kernel::feature::detection_rate(counts, detection_rates);

// 过滤细胞
for (Index i = 0; i < counts.rows(); ++i) {
    if (means[i] < min_mean || detection_rates[i] < min_rate) {
        // 过滤掉细胞 i
    }
}
```

### 高变异性基因检测

```cpp
// 计算特征统计
Array<Real> gene_means(counts.cols());
Array<Real> gene_vars(counts.cols());
Array<Real> dispersion(counts.cols());

// 注意：对于基因级统计，转置或使用 CSC 格式
Sparse<Real, false> counts_T = /* ... */;  // CSC（基因为行）

scl::kernel::feature::standard_moments(counts_T, gene_means, gene_vars, 1);
scl::kernel::feature::dispersion(gene_means, gene_vars, dispersion);

// 选择高变异性基因
std::vector<Index> hvg_indices;
for (Index i = 0; i < counts.cols(); ++i) {
    if (dispersion[i] > threshold && gene_means[i] > min_mean) {
        hvg_indices.push_back(i);
    }
}
```

### 带裁剪的稳健统计

```cpp
// 定义裁剪阈值（例如，每个细胞的第 99 百分位）
Array<Real> clip_vals(counts.rows());
// ... 从百分位数计算 clip_vals ...

Array<Real> robust_means(counts.rows());
Array<Real> robust_vars(counts.rows());

scl::kernel::feature::clipped_moments(counts, clip_vals, robust_means, robust_vars);
```

## 性能考虑

### SIMD 优化

- **融合计算**：一起计算求和与平方和（单次遍历）
- **4 路展开**：多个累加器以隐藏延迟
- **掩码操作**：高效处理边缘情况（零均值、裁剪）

### 并行化

- 所有操作跨主维度并行化
- 每个线程处理独立的行/列
- 无共享可变状态

### 数值稳定性

- 方差限制为非负
- 除零保护（epsilon 阈值）
- 标准浮点运算

---

::: tip 并行计算
所有特征统计操作默认并行化。对于大型矩阵，确保足够的线程数以获得最佳性能。
:::

