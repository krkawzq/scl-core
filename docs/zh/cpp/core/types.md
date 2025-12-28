# type.hpp

> scl/core/type.hpp · 具有编译时配置的统一类型系统

## 概述

本文件定义 SCL-Core 的基础类型系统，提供编译时可配置的精度和索引类型。类型系统使单个代码库能够支持多种精度级别（float32/float64/float16）和索引大小（int16/int32/int64），且无运行时开销。

主要特性：
- 通过预处理器宏进行编译时类型选择
- 零开销类型别名（无运行时成本）
- 类型安全的数组视图（Array<T>）
- 用于容器互操作的通用概念（ArrayLike, CSRLike）

**头文件**: `#include "scl/core/type.hpp"`

---

## 主要 API

### Real

用于数值计算的主要浮点类型，在编译时配置。

::: source_code file="scl/core/type.hpp" symbol="Real" collapsed
:::

**算法说明**

Real 是通过三个预处理器宏之一在编译时选择的类型别名：
- `SCL_USE_FLOAT32`: Real = float (32 位 IEEE 754)
- `SCL_USE_FLOAT64`: Real = double (64 位 IEEE 754)
- `SCL_USE_FLOAT16`: Real = _Float16 (16 位 IEEE 754-2008)

选择在编译时强制执行：必须定义且仅定义一个宏，否则编译错误。精度是库范围的，不能在同一构建中混合使用。

**边界条件**

- **未定义宏**：编译错误（必须定义且仅定义一个）
- **定义多个宏**：编译错误（定义冲突）
- **float16 兼容性**：需要 GCC >= 12 或 Clang >= 15

**数据保证（前置条件）**

- 必须定义且仅定义 SCL_USE_FLOAT32、SCL_USE_FLOAT64、SCL_USE_FLOAT16 之一
- 精度在整个库构建中保持一致

**复杂度分析**

- **时间**：O(1) - 编译时选择，零运行时开销
- **空间**：O(1) - 类型别名，无存储开销

**示例**

```cpp
#include "scl/core/type.hpp"

// 编译时使用: -DSCL_USE_FLOAT32
Real x = 3.14f;  // Real 是 float

// 编译时使用: -DSCL_USE_FLOAT64
Real y = 3.14;   // Real 是 double

// 编译时可用的元数据
constexpr int dtype_code = DTYPE_CODE;         // 0, 1, 或 2
constexpr const char* dtype_name = DTYPE_NAME; // "float32", "float64", 或 "float16"
```

---

### Index

用于数组索引和维度的有符号整数类型，在编译时配置。

::: source_code file="scl/core/type.hpp" symbol="Index" collapsed
:::

**算法说明**

Index 是通过三个预处理器宏之一在编译时选择的有符号整数类型：
- `SCL_USE_INT16`: Index = int16_t (16 位有符号，支持高达 32K x 32K 矩阵)
- `SCL_USE_INT32`: Index = int32_t (32 位有符号，标准选择，支持 2B x 2B 矩阵)
- `SCL_USE_INT64`: Index = int64_t (64 位有符号，用于非常大的矩阵)

使用有符号整数（而非无符号）允许：
- 负索引用于反向迭代
- 更简单的循环边界检查（i >= 0）
- 与 BLAS/LAPACK 约定兼容
- 防止无符号下溢错误

**边界条件**

- **未定义宏**：编译错误（必须定义且仅定义一个）
- **定义多个宏**：编译错误（定义冲突）
- **溢出**：如果值超过类型范围，Index 算术可能溢出

**数据保证（前置条件）**

- 必须定义且仅定义 SCL_USE_INT16、SCL_USE_INT32、SCL_USE_INT64 之一
- 用于数组索引的 Index 值应为非负（尽管有符号类型允许负值用于特殊目的）

**复杂度分析**

- **时间**：O(1) - 编译时选择，零运行时开销
- **空间**：较小的索引减少内存流量并提高缓存性能（特别是对于稀疏矩阵索引数组）

**示例**

```cpp
#include "scl/core/type.hpp"

// 编译时使用: -DSCL_USE_INT32
Index i = 1000;      // Index 是 int32_t
Index j = -1;        // 允许（可用于反向迭代）

// 元数据
constexpr int index_code = INDEX_DTYPE_CODE;         // 0, 1, 或 2
constexpr const char* index_name = INDEX_DTYPE_NAME; // "int16", "int32", 或 "int64"

// 用于数组索引
Array<Real> data(n);
for (Index idx = 0; idx < n; ++idx) {
    Real value = data[idx];
}
```

---

### Array<T>

轻量级、非拥有的连续一维数组视图。零开销 POD 类型，可平凡复制。

::: source_code file="scl/core/type.hpp" symbol="Array" collapsed
:::

**算法说明**

Array<T> 是包含指针和长度的结构：
- `ptr`: 指向第一个元素的指针（T*）
- `len`: 元素数量（Size）

设计理念：
- **非拥有**：无析构函数，无分配，无释放
- **零开销**：可平凡复制的 POD 类型（在 64 位系统上 16 字节）
- **常量正确**：Array<T>（可变）vs Array<const T>（不可变）
- **公共成员**：直接访问以用于性能关键代码
- **可复制**：视图的浅拷贝，而非底层数据

该结构提供标准的类似容器的方法（size, empty, begin, end, operator[]），全部强制内联以实现零函数调用开销。

**边界条件**

- **空数组**：ptr = nullptr, len = 0
- **非零长度的空指针**：未定义行为（调用者必须确保有效性）
- **越界访问**：调试版本断言，发布版本未定义行为
- **生命周期**：Array 视图不得比底层数据存活更久

**数据保证（前置条件）**

- 如果 len > 0，ptr 必须指向至少 len 个元素的有效数组
- 如果 len == 0，ptr 可以是 nullptr 或任何值（忽略）
- 底层数据生命周期必须超过 Array 视图生命周期

**复杂度分析**

- **时间**：所有操作 O(1)（所有方法都是强制内联的）
- **空间**：sizeof(Array<T>) = sizeof(T*) + sizeof(Size) = 在 64 位系统上 16 字节

**示例**

```cpp
#include "scl/core/type.hpp"

// 从指针和大小创建
Real* data = new Real[100];
Array<Real> arr(data, 100);

// 访问元素
Real x = arr[0];
arr[5] = 3.14;

// 迭代
for (Index i = 0; i < arr.size(); ++i) {
    arr[i] *= 2.0;
}

// 常量视图
Array<const Real> const_view = arr;  // 隐式转换
// const_view[0] = 1.0;  // 错误：无法修改常量视图

// 空数组
Array<Real> empty;  // ptr = nullptr, len = 0
```

---

### Size

用于内存大小和字节计数的无符号整数类型。

**定义**：`using Size = std::size_t;`

**使用指南**
- 用于内存分配大小
- 用于字节计数和缓冲区长度
- 不用于数组索引（使用 Index 代替）
- 不用于数组元素的循环计数器

**示例**

```cpp
Size buffer_size = 1024 * sizeof(Real);  // 内存大小（字节）
Real* data = new Real[100];
Size n_bytes = 100 * sizeof(Real);       // 字节大小

// 从 Index 转换为 Size（确保非负）
Index idx = 100;
Size sz = static_cast<Size>(idx);  // 仅当 idx >= 0 时
```

---

### Byte

用于原始内存操作的无符号 8 位整数。

**定义**：`using Byte = std::uint8_t;`

**使用指南**
- 用于原始内存缓冲区
- 用于序列化/反序列化
- 用于字节级 I/O 操作

**示例**

```cpp
Byte* raw_buffer = new Byte[1024];
// 用于原始内存操作
```

---

### Pointer

用于非连续存储和 C-ABI 边界的通用无类型指针。

**定义**：`using Pointer = void*;`

**使用指南**
- 用于跨 C-ABI 的类型擦除指针
- 用于通用内存句柄
- 在解引用前始终转换为适当类型

**警告**：空指针绕过类型安全。仅在 API 边界处使用，其中类型擦除是必要的（例如，Python 绑定）。

---

## 工具概念

### ArrayLike<A>

具有随机访问的一维类似数组容器的概念。

::: source_code file="scl/core/type.hpp" symbol="ArrayLike" collapsed
:::

**要求**

- `value_type`：元素类型的成员类型别名
- `size()`：返回元素数量（可转换为 Size）
- `operator[](Index)`：随机访问元素
- `begin()`：返回指向第一个元素的迭代器
- `end()`：返回指向最后一个元素之后的迭代器

**满足类型**

- `scl::Array<T>`
- `std::vector<T>`
- `std::array<T, N>`
- `std::deque<T>`
- `std::span<T>` (C++20)
- 具有兼容接口的自定义容器

**示例**

```cpp
template <ArrayLike A>
void process(const A& arr) {
    for (Index i = 0; i < arr.size(); ++i) {
        do_something(arr[i]);
    }
}

// 适用于任何 ArrayLike 类型
Array<Real> arr = ...;
std::vector<Real> vec = ...;
process(arr);  // OK
process(vec);  // OK
```

---

### CSRLike<M>

CSR（压缩稀疏行）稀疏矩阵的概念。

::: source_code file="scl/core/type.hpp" symbol="CSRLike" collapsed
:::

**要求**

- `ValueType`：元素类型别名
- `Tag`：必须是 `TagSparse<true>`
- `is_csr`：静态 constexpr bool == true
- `rows()`, `cols()`, `nnz()`：矩阵维度
- `primary_values(i)`, `primary_indices(i)`, `primary_length(i)`：行访问

使通用算法能够与任何类似 CSR 的稀疏矩阵类型一起工作。

---

### TagSparse<IsCSR>

用于稀疏矩阵格式选择的标签类型。

::: source_code file="scl/core/type.hpp" symbol="TagSparse" collapsed
:::

**模板参数**

- `IsCSR` [bool]：true 表示 CSR（行主序），false 表示 CSC（列主序）

**静态成员**

- `is_csr`：constexpr bool，等于 IsCSR
- `is_csc`：constexpr bool，等于 !IsCSR

使稀疏矩阵的编译时分发和类型特征成为可能。

---

## 类型配置

### 编译时选择

类型通过预处理器宏在编译时配置：

```cpp
// 在 CMakeLists.txt 或 config.hpp 中
#define SCL_USE_FLOAT32  // 或 FLOAT64, FLOAT16
#define SCL_USE_INT32    // 或 INT16, INT64
```

这使以下成为可能：
- 单个代码库用于多种精度
- 类型选择的零运行时开销
- 最佳代码生成

### 元数据

运行时可访问的元数据：

```cpp
constexpr int dtype_code = DTYPE_CODE;         // 0 (float32), 1 (float64), 2 (float16)
constexpr const char* dtype_name = DTYPE_NAME; // "float32", "float64", 或 "float16"

constexpr int index_code = INDEX_DTYPE_CODE;         // 0 (int16), 1 (int32), 2 (int64)
constexpr const char* index_name = INDEX_DTYPE_NAME; // "int16", "int32", 或 "int64"
```

## 设计原则

### 零依赖

核心类型仅依赖于：
- C++17 标准库
- 标准头文件：`<cstddef>`, `<cstdint>`, `<type_traits>`, `<concepts>`

### 零运行时开销

- 类型别名（Real, Index）是编译时选择
- Array<T> 是可平凡复制的 POD 类型
- 所有方法都是强制内联的
- 概念是编译时检查（零运行时成本）

### 显式资源管理

Array<T> 不拥有内存：
- 无析构函数
- 无分配/释放
- 底层数据生命周期必须超过 Array 视图生命周期

## 相关内容

- [Sparse Matrix](./sparse) - 使用 Array<T> 进行稀疏矩阵存储
- [Memory Management](./memory) - 分配函数返回与 Array<T> 一起使用的指针

