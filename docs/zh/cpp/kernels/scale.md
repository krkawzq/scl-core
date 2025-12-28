# 缩放操作

稀疏矩阵的原地缩放、平移和标准化，采用 SIMD 优化。

## 概览

缩放内核提供：

- **标准化** - (x - mean) / std，可选裁剪
- **行缩放** - 将每行乘以缩放因子
- **行平移** - 向每行添加偏移量
- **自适应 SIMD** - 针对不同行长度优化
- **零中心控制** - 灵活的标准化模式

## 标准化

### standardize

原地标准化稀疏矩阵值：`(x - mean) / std`，可选裁剪和零中心控制：

```cpp
#include "scl/kernel/scale.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<const Real> means = /* ... */;   // 逐行均值 [primary_dim]
Array<const Real> stds = /* ... */;    // 逐行标准差 [primary_dim]

// 带零中心和裁剪的标准化
Real max_value = 10.0;  // 裁剪到 [-10, 10]
bool zero_center = true;
scl::kernel::scale::standardize(matrix, means, stds, max_value, zero_center);

// 不带裁剪的标准化
scl::kernel::scale::standardize(matrix, means, stds, Real(0), true);

// 仅缩放（不零中心）
scl::kernel::scale::standardize(matrix, means, stds, Real(0), false);
```

**参数：**
- `matrix`: 要标准化的稀疏矩阵（原地修改）
- `means`: 每个主维度的均值
- `stds`: 每个主维度的标准差
- `max_value`: 裁剪阈值（0 禁用裁剪）
- `zero_center`: 是否在缩放前减去均值

**后置条件：**
- 每个值变换为：`(v - mean) / std`（如果 zero_center）或 `v / std`（如果不）
- 如果 `max_value > 0`，结果裁剪到 `[-max_value, max_value]`
- `std = 0` 的行保持不变（跳过）

**算法：**
基于行长度使用 3 层自适应策略：
- **短行（< 16）**：标量循环
- **中等行（16-128）**：4 路 SIMD 展开
- **长行（>= 128）**：8 路 SIMD 展开，带预取

分支条件（zero_center、do_clip）在内部循环外提升以提高效率。

**用例：**
- Z-score 标准化
- 机器学习的特征缩放
- 数据预处理流程
- 带异常值裁剪的标准化

## 行缩放

### scale_rows

将每个主维度乘以对应的缩放因子：

```cpp
Array<const Real> scales = /* ... */;  // 逐行缩放因子 [primary_dim]

scl::kernel::scale::scale_rows(matrix, scales);
```

**参数：**
- `matrix`: 要缩放的稀疏矩阵（原地修改）
- `scales`: 逐行缩放因子，大小 = `primary_dim`

**后置条件：**
- 行 `i` 中的每个值乘以 `scales[i]`
- `scales[i] == 1` 的行保持不变（提前退出优化）

**算法：**
- 对每个主索引并行：
  1. 如果 scale == 1 则跳过（优化）
  2. 使用 SIMD 4 路展开，带预取进行缩放

**用例：**
- 单位范数缩放
- 每个细胞的标准化因子
- 批次校正

## 行平移

### shift_rows

向每个主维度添加常量偏移量：

```cpp
Array<const Real> offsets = /* ... */;  // 逐行偏移量 [primary_dim]

scl::kernel::scale::shift_rows(matrix, offsets);
```

**参数：**
- `matrix`: 要平移的稀疏矩阵（原地修改）
- `offsets`: 要添加的逐行偏移量，大小 = `primary_dim`

**后置条件：**
- 行 `i` 中的每个值增加 `offsets[i]`
- `offsets[i] == 0` 的行保持不变（提前退出优化）

**算法：**
- 对每个主索引并行：
  1. 如果 offset == 0 则跳过（优化）
  2. 使用 SIMD 4 路展开，带预取进行加法

**数值说明：**
- 仅修改存储的（非零）值
- 隐式零保持为零
- 要平移包括零在内的所有值，必须将矩阵密集化

**用例：**
- 均值中心化（计算均值后）
- 基线校正
- 偏移调整

## 示例

### Z-Score 标准化流程

```cpp
#include "scl/kernel/scale.hpp"
#include "scl/kernel/feature.hpp"

Sparse<Real, true> counts = /* ... */;

// 步骤 1：计算统计
Array<Real> means(counts.rows());
Array<Real> vars(counts.rows());
scl::kernel::feature::standard_moments(counts, means, vars, 1);

// 步骤 2：计算标准差
Array<Real> stds(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    stds[i] = std::sqrt(vars[i]);
}

// 步骤 3：标准化（零中心和缩放）
scl::kernel::scale::standardize(counts, means, stds, Real(0), true);
```

### 带裁剪的标准化

```cpp
// 带异常值裁剪的标准化
Real clip_threshold = 10.0;  // 裁剪到 10 个标准差
scl::kernel::scale::standardize(counts, means, stds, clip_threshold, true);
```

### 每个细胞总计数标准化

```cpp
// 计算每个细胞的总计数
Array<Real> totals(counts.rows());
scl::kernel::sparse::primary_sums(counts, totals);

// 计算缩放因子（标准化到 10,000）
Real target_sum = 10000.0;
Array<Real> scales(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    scales[i] = (totals[i] > Real(0)) ? (target_sum / totals[i]) : Real(1);
}

// 缩放行
scl::kernel::scale::scale_rows(counts, scales);
```

### 均值中心化

```cpp
// 计算均值
Array<Real> means(counts.rows());
Array<Real> vars(counts.rows());
scl::kernel::feature::standard_moments(counts, means, vars, 1);

// 取负均值用于平移
Array<Real> offsets(counts.rows());
for (Index i = 0; i < counts.rows(); ++i) {
    offsets[i] = -means[i];
}

// 平移行（围绕零中心化）
scl::kernel::scale::shift_rows(counts, offsets);
```

## 性能考虑

### 自适应策略

标准化函数根据行长度使用不同算法：

- **短行**：使用标量循环的最小开销
- **中等行**：平衡开销和并行性（4 路 SIMD）
- **长行**：使用激进预取的 maximum 并行性（8 路 SIMD）

### 提前退出优化

- `scale_rows`：跳过 scale == 1 的行
- `shift_rows`：跳过 offset == 0 的行
- `standardize`：跳过 std == 0 的行

### SIMD 操作

- 使用标准差的倒数（`inv_sigma = 1/std`）将除法替换为乘法
- 向量化 min/max 用于裁剪操作
- 4 路和 8 路展开以获得更好的 ILP（指令级并行）

### 并行化

- 所有操作跨主维度并行化
- 每个线程处理独立的行
- 无需同步

---

::: tip 性能提示
对于标准化，预计算均值和标准差一次，并为具有相同结构的多个矩阵重用它们。
:::

