# vectorize.hpp

> scl/core/vectorize.hpp · SIMD 优化的向量化数组操作

## 概述

本文件提供使用 Google Highway 对数组视图进行高性能 SIMD 优化的操作。所有操作使用激进的展开（2-4 路）并自动处理标量尾部。

主要特性：
- 零开销抽象（编译时处理）
- 自动 SIMD 向量化（架构无关）
- 激进展开（2-4 路以实现最大 ILP）
- 自动尾部处理

**头文件**: `#include "scl/core/vectorize.hpp"`

---

## 主要 API

### sum

使用 SIMD 优化的归约计算所有元素的和。

::: source_code file="scl/core/vectorize.hpp" symbol="sum" collapsed
:::

**算法说明**

使用 4 路展开的 SIMD 累加计算和：
1. 使用 4 路展开的 SIMD 循环处理大部分
2. 使用 SumOfLanes 进行水平归约
3. 对标量尾部进行剩余处理

**边界条件**

- **空数组**：返回 T(0)
- **NaN/Inf**：通过和传播

**数据保证（前置条件）**

- span 必须是有效的 Array 视图

**复杂度分析**

- **时间**：O(N)
- **空间**：O(1)

**示例**

```cpp
#include "scl/core/vectorize.hpp"

Array<const Real> data = ...;
Real total = scl::vectorize::sum(data);
```

---

### dot

计算两个向量的点积。

::: source_code file="scl/core/vectorize.hpp" symbol="dot" collapsed
:::

**算法说明**

计算点积：sum(a[i] * b[i])，使用 MulAdd (FMA) 以获得最佳性能。

**边界条件**

- **空数组**：返回 T(0)
- **大小不匹配**：未定义行为（调用者必须确保 a.len == b.len）

**数据保证（前置条件）**

- a.len == b.len
- 两个数组必须有效

**复杂度分析**

- **时间**：O(N)
- **空间**：O(1)

**示例**

```cpp
Array<const Real> a = ...;
Array<const Real> b = ...;
Real result = scl::vectorize::dot(a, b);
```

---

### norm

计算向量的 L2 范数（欧氏范数）。

::: source_code file="scl/core/vectorize.hpp" symbol="norm" collapsed
:::

**复杂度**：O(N) 时间，O(1) 空间

---

### add / mul / sub / div

逐元素算术运算。

**复杂度**：O(N) 时间，O(1) 空间

---

### count

使用 SIMD 计算值的出现次数。

**复杂度**：O(N) 时间，O(1) 空间

---

### find

查找值的首次出现。

**复杂度**：最坏情况 O(N) 时间，通常早期退出更好

## 相关内容

- [SIMD](./simd) - 底层 SIMD 抽象层
- [类型系统](./types) - 用于视图的 Array<T> 类型

