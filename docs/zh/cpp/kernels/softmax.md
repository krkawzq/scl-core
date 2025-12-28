# Softmax

带 SIMD 优化和温度缩放的 Softmax 归一化操作。

## 概述

Softmax 操作提供：

- **原地归一化** - 将值转换为概率分布
- **温度缩放** - 控制分布的尖锐程度
- **Log-softmax** - 数值稳定的对数概率
- **稀疏矩阵支持** - 稀疏矩阵的行级 softmax

## 密集数组操作

### softmax_inplace

对密集数组原地应用 softmax 归一化。

```cpp
#include "scl/kernel/softmax.hpp"

Real values[100];
Size len = 100;

// 标准 softmax
scl::kernel::softmax::softmax_inplace(values, len);

// 带温度缩放
scl::kernel::softmax::softmax_inplace(values, len, 0.5);
```

**参数：**
- `vals` [in,out] - 值数组指针，原地修改
- `len` [in] - 数组长度
- `temperature` [in] - 可选温度参数（值越大越均匀）

**后置条件：**
- 所有值在 [0, 1] 范围内且和为 1.0
- 对于 temperature > 0: softmax(x / temperature)
- 对于 temperature <= 0: 在最大值处为 one-hot

**算法：**
根据数组长度的 3 层自适应策略：
1. 短数组 (< 16): 标量循环
2. 中等数组 (< 128): 4 路 SIMD 展开，带预取
3. 长数组 (>= 128): 8 路 SIMD 展开，8 个累加器用于指令级并行

步骤：
1. 找到最大值以保持数值稳定性
2. 同时计算 exp(x - max) 和求和
3. 通过除以和来归一化每个元素

**复杂度：**
- 时间: O(n)
- 空间: O(1) 辅助空间

**线程安全：**
对不同数组安全，对同一数组不安全

**数值说明：**
- 最大值减法防止 exp() 溢出
- 如果和为零，返回均匀分布

### log_softmax_inplace

对密集数组原地应用 log-softmax。

```cpp
Real values[100];
Size len = 100;

// 标准 log-softmax
scl::kernel::softmax::log_softmax_inplace(values, len);

// 带温度缩放
scl::kernel::softmax::log_softmax_inplace(values, len, 0.5);
```

**参数：**
- `vals` [in,out] - 值数组指针，原地修改
- `len` [in] - 数组长度
- `temperature` [in] - 可选温度参数

**后置条件：**
- 所有值 <= 0（对数概率）
- exp(vals) 和为 1.0
- 对于 temperature > 0: log_softmax(x / temperature)
- 对于 temperature <= 0: 最大值处为 0，其他为 -inf

**算法：**
log_softmax(x) = x - max - log(sum(exp(x - max)))

3 层自适应策略：
1. 找到最大值
2. 使用 SIMD 计算 sum(exp(x - max))
3. 从每个元素中减去 (max + log(sum))

**复杂度：**
- 时间: O(n)
- 空间: O(1) 辅助空间

**数值说明：**
- 比 log(softmax(x)) 数值更稳定
- 避免计算显式概率

## 稀疏矩阵操作

### softmax_inplace (稀疏)

对稀疏矩阵按行原地应用 softmax。

```cpp
#include "scl/core/sparse.hpp"
#include "scl/kernel/softmax.hpp"

Sparse<Real, true> matrix = /* ... */;

// 标准 softmax
scl::kernel::softmax::softmax_inplace(matrix);

// 带温度缩放
scl::kernel::softmax::softmax_inplace(matrix, 0.5);
```

**参数：**
- `matrix` [in,out] - 稀疏矩阵（CSR 或 CSC），值原地修改
- `temperature` [in] - 可选温度参数

**后置条件：**
- 每行和为 1.0（仅考虑非零元素）
- 矩阵结构（索引、指针）不变
- 空行不变

**算法：**
对每行并行：
- 对非零值应用 3 层自适应 softmax

**复杂度：**
- 时间: O(nnz)
- 空间: 每个线程 O(1) 辅助空间

**线程安全：**
安全 - 按行并行，无共享可变状态

### log_softmax_inplace (稀疏)

对稀疏矩阵按行原地应用 log-softmax。

```cpp
Sparse<Real, true> matrix = /* ... */;

// 标准 log-softmax
scl::kernel::softmax::log_softmax_inplace(matrix);

// 带温度缩放
scl::kernel::softmax::log_softmax_inplace(matrix, 0.5);
```

**参数：**
- `matrix` [in,out] - 稀疏矩阵，值原地修改
- `temperature` [in] - 可选温度参数

**后置条件：**
- 所有值 <= 0（对数概率）
- 矩阵结构不变

**复杂度：**
- 时间: O(nnz)
- 空间: 每个线程 O(1) 辅助空间

**线程安全：**
安全 - 按行并行

## 使用场景

### 概率分布

将原始分数转换为概率分布：

```cpp
Real scores[10] = {3.0, 1.0, 4.0, 1.5, 2.0, 0.5, 2.5, 1.0, 3.5, 0.0};
scl::kernel::softmax::softmax_inplace(scores, 10);
// scores 现在和为 1.0
```

### 温度缩放

控制分布的尖锐程度：

```cpp
// 尖锐分布（低温度）
scl::kernel::softmax::softmax_inplace(values, len, 0.1);

// 均匀分布（高温度）
scl::kernel::softmax::softmax_inplace(values, len, 10.0);
```

### 对数概率

在对数空间计算中的数值稳定性：

```cpp
scl::kernel::softmax::log_softmax_inplace(logits, len);
// 用于交叉熵损失: -sum(y * log_softmax)
```

### 稀疏矩阵归一化

归一化稀疏矩阵的每一行：

```cpp
Sparse<Real, true> expression_matrix = /* ... */;
scl::kernel::softmax::softmax_inplace(expression_matrix);
// 每行现在表示一个概率分布
```

## 性能

### SIMD 优化

所有操作使用 SIMD 优化的 exp 和 sum 操作：
- 中等数组使用 4 路展开
- 大数组使用 8 路展开和 8 个累加器
- 预取以提高缓存效率

### 并行化

稀疏矩阵操作按行并行：
- 自动工作分配
- 线程本地累加器
- 无同步开销

## 参见

- [归一化](/zh/cpp/kernels/normalization) - 其他归一化操作
- [稀疏工具](/zh/cpp/kernels/sparse-tools) - 稀疏矩阵工具

