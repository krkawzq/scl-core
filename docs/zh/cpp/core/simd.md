# simd.hpp

> scl/core/simd.hpp · 使用 Google Highway 的架构无关 SIMD 抽象层

## 概述

本文件通过包装 Google Highway，为不同硬件架构（AVX2、AVX-512、NEON 等）的 SIMD 操作提供统一接口。所有 Highway 函数都导入到 `scl::simd` 命名空间中，提供架构可移植性，且无运行时开销。

主要特性：
- 架构无关的 SIMD 操作（x86、ARM 等）
- 零运行时开销（所有抽象在编译时处理）
- 类型安全的基于标签的分发
- 自动选择最佳向量宽度
- 直接访问 Highway 函数

**头文件**: `#include "scl/core/simd.hpp"`

---

## 主要 API

### Tag

`scl::Real` 类型的主要 SIMD 描述符标签。

**定义**：`using Tag = ScalableTag<scl::Real>;`

**算法说明**

Tag 是 Highway 的 ScalableTag 的类型别名，它自动为当前硬件和 scl::Real 类型（float 或 double）选择最佳向量宽度。

标签用于类型安全的 SIMD 分发 - 所有 SIMD 操作都需要标签参数来指定向量类型和宽度。

**边界条件**

- **标量模式**：如果定义了 SCL_ONLY_SCALAR，SIMD 被禁用并使用标量后备
- **类型不匹配**：如果标签与向量类型不匹配，编译时错误

**数据保证（前置条件）**

- 必须定义 scl::Real（float 或 double）
- 必须在整个 SIMD 操作中一致使用 Tag

**复杂度分析**

- **运行时**：O(1) - 仅编译时类型，无运行时成本
- **向量宽度**：取决于硬件（float 通常 4-16 个元素，double 通常 2-8 个）

**示例**

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;

// 创建标签（constexpr，零运行时成本）
const s::Tag d;

// 加载数据
Real* data = ...;
auto v = s::Load(d, data);  // 加载 Real 值向量

// 执行操作
auto v2 = s::Mul(v, s::Set(d, 2.0));  // 乘以 2.0

// 存储结果
s::Store(v2, d, output);
```

---

### IndexTag

`scl::Index` 类型的 SIMD 描述符标签。

**定义**：`using IndexTag = ScalableTag<scl::Index>;`

用于向量化索引操作、gather/scatter 和整数算术。

**示例**

```cpp
const s::IndexTag idx_d;
Index* indices = ...;
auto idx_vec = s::Load(idx_d, indices);
```

---

### ReinterpretTag

匹配 scl::Real 大小的无符号整数的 SIMD 描述符标签。

**定义**：`using ReinterpretTag = RebindToUnsigned<Tag>;`

用于无需类型转换的浮点数据位操作。

**示例**

```cpp
const s::ReinterpretTag uint_d;
auto uint_vec = s::BitCast(uint_d, float_vec);  // 重新解释为无符号
```

---

## Highway 函数

所有 Highway SIMD 函数都导入到 `scl::simd` 命名空间。关键函数包括：

### 内存操作

- `Load(d, ptr)` - 加载对齐向量
- `LoadU(d, ptr)` - 加载非对齐向量
- `Store(vec, d, ptr)` - 存储对齐向量
- `StoreU(vec, d, ptr)` - 存储非对齐向量

### 算术操作

- `Add(a, b)` - 加法
- `Sub(a, b)` - 减法
- `Mul(a, b)` - 乘法
- `Div(a, b)` - 除法

### 比较和选择

- `Min(a, b)` - 最小值
- `Max(a, b)` - 最大值
- `Abs(v)` - 绝对值
- `IfThenElse(mask, true_val, false_val)` - 条件选择

### 初始化

- `Set(d, value)` - 广播标量值
- `Zero(d)` - 零向量
- `Iota(d, start)` - 序列向量

### 逻辑操作

- `And(a, b)`, `Or(a, b)`, `Xor(a, b)`, `Not(a)` - 位操作

**注意**：请参阅 Google Highway 文档以获取完整函数列表。

---

## 工具函数

### lanes

获取 scl::Real 的 SIMD 向量中的元素（通道）数。

**示例**

```cpp
const s::Tag d;
size_t num_lanes = s::Lanes(d);  // 例如，AVX2 float 为 8，AVX-512 float 为 16
```

**复杂度**：O(1) - 编译时常量

---

## 设计原则

### 零运行时开销

所有 SIMD 抽象仅在编译时：
- 标签是类型，不是值（编译时分发）
- Using 指令导入函数（无包装开销）
- Highway 优化直接应用

### 架构可移植性

相同代码可在不同架构上运行：
- x86：AVX2、AVX-512
- ARM：NEON
- WebAssembly：SIMD128
- SIMD 不可用时使用标量后备

### 类型安全

基于标签的分发防止类型不匹配：
- 标签必须匹配向量元素类型
- 不匹配时编译时错误
- 无需运行时类型检查

## 配置

### 禁用 SIMD

定义 `SCL_ONLY_SCALAR` 以禁用 SIMD 并使用标量后备：

```cpp
#define SCL_ONLY_SCALAR
// 所有 SIMD 操作变为标量循环
```

## 性能注意事项

- **向量宽度**：根据硬件自动选择（通常 4-16 倍加速）
- **对齐**：对齐加载/存储比非对齐更快
- **循环展开**：将 SIMD 与循环展开结合以获得最佳性能
- **内存带宽**：SIMD 在计算受限而非内存受限时最有效

## 相关内容

- [内存管理](./memory) - SIMD 缓冲区的对齐分配
- [向量化](./vectorize) - 高级向量化操作
- [Google Highway 文档](https://github.com/google/highway) - 完整的 SIMD API 参考

