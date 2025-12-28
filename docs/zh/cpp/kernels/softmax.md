# softmax.hpp

> scl/kernel/softmax.hpp · Softmax 操作

## 概述

本文件为密集数组和稀疏矩阵提供高性能的 softmax 和 log-softmax 操作。所有操作都是就地操作以提高效率，并根据输入大小使用自适应 SIMD 策略。支持温度缩放以控制分布的锐度。

**头文件**: `#include "scl/kernel/softmax.hpp"`

---

## 主要 API

### softmax_inplace (密集数组)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**算法说明**

使用 3 层自适应策略对密集数组应用就地 softmax 归一化：

1. **短数组 (< 16)**: 标量循环以最小化开销
2. **中等数组 (< 128)**: 4 路 SIMD 展开加预取
3. **长数组 (>= 128)**: 8 路 SIMD 展开，8 个累加器用于指令级并行

步骤：
1. 查找最大值以保持数值稳定性：`max_val = max(vals)`
2. 使用 SIMD 同时计算 `exp(x - max)` 和和
3. 归一化：`vals[i] = exp(vals[i] - max) / sum`

**边界条件**

- **空数组 (len=0)**: 无操作，立即返回
- **全零**: 返回均匀分布（每个元素 1/len）
- **所有值相同**: 返回均匀分布
- **非常大的值**: 最大值减法防止 exp() 溢出
- **和为零**: 返回均匀分布以避免除以零

**数据保证（前置条件）**

- 如果 len > 0，`vals` 必须是有效指针
- `len >= 0`
- 数组内存可写

**复杂度分析**

- **时间**: O(n) - 单次遍历，带 SIMD 加速
- **空间**: O(1) 辅助空间 - 仅需要累加器

**示例**

```cpp
#include "scl/kernel/softmax.hpp"

Real* values = /* 值数组 */;
Size len = /* 数组长度 */;

scl::kernel::softmax::softmax_inplace(values, len);

// values[i] 现在在 [0, 1] 中且 sum(values) == 1.0
```

---

### softmax_inplace (带温度的密集数组)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**算法说明**

应用带温度缩放的就地 softmax：

1. 缩放所有值：`vals[i] = vals[i] / temperature`
2. 对缩放后的值应用标准 softmax
3. 温度 > 0：产生更软的分布（温度越高 = 越均匀）
4. 温度 <= 0：在最大值处产生 one-hot

**边界条件**

- **温度 > 1**: 更软的分布，更均匀
- **温度 < 1**: 更尖锐的分布，更峰值
- **温度 = 1**: 标准 softmax
- **温度 <= 0**: 在最大值处进行 one-hot 编码
- **温度 = 0**: 避免除以零，视为 <= 0

**数据保证（前置条件）**

- 如果 len > 0，`vals` 必须是有效指针
- `len >= 0`
- 数组内存可写

**复杂度分析**

- **时间**: O(n) - 缩放加 softmax
- **空间**: O(1) 辅助空间

**示例**

```cpp
Real* values = /* 值数组 */;
Size len = /* 数组长度 */;
Real temperature = 0.5;  // 更尖锐的分布

scl::kernel::softmax::softmax_inplace(values, len, temperature);

// values 现在表示温度缩放的 softmax 分布
```

---

### log_softmax_inplace (密集数组)

::: source_code file="scl/kernel/softmax.hpp" symbol="log_softmax_inplace" collapsed
:::

**算法说明**

对密集数组应用就地 log-softmax：

1. 查找最大值：`max_val = max(vals)`
2. 计算指数和：使用 SIMD 计算 `sum_exp = sum(exp(vals[i] - max))`
3. 计算对数-和：`log_sum = log(sum_exp)`
4. 更新值：`vals[i] = vals[i] - max - log_sum`

公式：`log_softmax(x) = x - max - log(sum(exp(x - max)))`

**边界条件**

- **空数组**: 无操作
- **全零**: 返回均匀对数概率（log(1/len)）
- **所有值相同**: 返回均匀对数概率
- **非常大的值**: 最大值减法防止溢出
- **和为零**: 返回均匀对数概率

**数据保证（前置条件）**

- 如果 len > 0，`vals` 必须是有效指针
- `len >= 0`
- 数组内存可写

**复杂度分析**

- **时间**: O(n) - 单次遍历，带 SIMD
- **空间**: O(1) 辅助空间

**示例**

```cpp
Real* values = /* 值数组 */;
Size len = /* 数组长度 */;

scl::kernel::softmax::log_softmax_inplace(values, len);

// values[i] <= 0（对数概率）
// exp(values) 和为 1.0
```

---

### softmax_inplace (稀疏矩阵)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**算法说明**

对稀疏矩阵按行应用就地 softmax：

1. 并行处理每行：
   - 提取行中的非零值
   - 仅对非零值应用 3 层自适应 softmax
   - 就地更新值
2. 矩阵结构（索引、指针）不变
3. 空行保持不变（无非零可归一化）

**边界条件**

- **空行**: 不变（无非零）
- **每行单个非零**: 归一化后变为 1.0
- **行中全零**: 保持不变
- **非常稀疏的行**: 高效处理非零较少的行

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式（CSR 或 CSC）
- 矩阵值必须可变
- 矩阵结构有效

**复杂度分析**

- **时间**: O(nnz) - 每个非零处理一次
- **空间**: O(1) 辅助空间 - 每个线程仅累加器

**示例**

```cpp
Sparse<Real, true> matrix = /* 稀疏矩阵，CSR */;

scl::kernel::softmax::softmax_inplace(matrix);

// 每行现在和为 1.0（仅考虑非零）
// 矩阵结构不变
```

---

### log_softmax_inplace (稀疏矩阵)

::: source_code file="scl/kernel/softmax.hpp" symbol="log_softmax_inplace" collapsed
:::

**算法说明**

对稀疏矩阵按行应用就地 log-softmax：

1. 并行处理每行：
   - 提取非零值
   - 计算 log-softmax：`log_softmax(x) = x - max - log(sum(exp(x - max)))`
   - 就地更新值
2. 矩阵结构不变
3. 所有值变为 <= 0（对数概率）

**边界条件**

- **空行**: 不变
- **单个非零**: 变为 0.0（log(1.0) = 0）
- **全零**: 保持不变
- **稀疏行**: 高效处理非零较少的行

**数据保证（前置条件）**

- 矩阵是有效的稀疏格式
- 矩阵值必须可变

**复杂度分析**

- **时间**: O(nnz) - 每个非零处理一次
- **空间**: O(1) 辅助空间 - 每个线程

**示例**

```cpp
Sparse<Real, true> matrix = /* 稀疏矩阵 */;

scl::kernel::softmax::log_softmax_inplace(matrix);

// 所有值 <= 0（对数概率）
// 每行的 exp(values) 和为 1.0
```

---

## 数值说明

### 稳定性

- **最大值减法**: 通过减去最大值防止 exp() 溢出
- **Log-softmax**: 对于大值比 log(softmax(x)) 更数值稳定
- **均匀回退**: 如果和为零，返回均匀分布

### 温度缩放

- **温度 > 1**: 更软的分布，减少峰值
- **温度 < 1**: 更尖锐的分布，增加峰值
- **温度 = 1**: 标准 softmax
- **温度 <= 0**: One-hot 编码（硬最大值）

---

## 相关内容

- [归一化模块](./normalize) - 其他归一化操作
- [稀疏矩阵](../core/sparse) - 稀疏矩阵操作
