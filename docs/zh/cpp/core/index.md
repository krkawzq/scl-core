# 核心模块

`scl/core/` 目录包含 SCL-Core 的基础：所有其他模块依赖的基本类型、数据结构和工具。

## 概览

核心模块提供：

- **类型系统** - 可配置的精度和索引类型
- **稀疏矩阵** - 高性能非连续存储
- **SIMD 抽象** - 通过 Highway 实现可移植的向量化
- **内存管理** - 对齐分配和 Registry 跟踪
- **错误处理** - 断言和异常
- **向量化** - 通用向量化操作

## 模块列表

| 模块 | 用途 | 关键类型 |
|------|------|----------|
| [类型系统](/zh/cpp/core/types) | 基本类型 | `Real`, `Index`, `Size` |
| [稀疏矩阵](/zh/cpp/core/sparse) | 稀疏矩阵基础设施 | `Sparse<T, IsCSR>` |
| [注册表](/zh/cpp/core/registry) | 内存生命周期跟踪 | `Registry`, `BufferID` |
| [SIMD](/zh/cpp/core/simd) | SIMD 抽象 | `Tag`, `Vec`, SIMD 操作 |
| [错误处理](/zh/cpp/core/error) | 断言和异常 | `SCL_ASSERT`, `SCL_CHECK_*` |
| [内存](/zh/cpp/core/memory) | 对齐分配 | `aligned_alloc`, `aligned_free` |
| [向量化](/zh/cpp/core/vectorize) | 向量化操作 | `dot`, `norm`, `sum` |

## 依赖图

```
┌─────────────────────────────────────────┐
│          其他模块                        │
│    (threading, kernel, math, etc.)      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          sparse.hpp                     │
│     (稀疏矩阵操作)                       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    registry.hpp + memory.hpp            │
│  (内存管理和跟踪)                        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    simd.hpp + vectorize.hpp             │
│      (SIMD 和向量化)                    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      error.hpp + macros.hpp             │
│     (错误处理和宏)                       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           type.hpp                      │
│       (基本类型)                         │
└─────────────────────────────────────────┘
```

## 快速开始

### 基本类型

```cpp
#include "scl/core/type.hpp"

using namespace scl;

// 可配置精度
Real x = 3.14;  // float, double, or _Float16

// 有符号索引
Index i = -1;   // int16_t, int32_t, or int64_t

// 大小和字节计数
Size n = 1000;  // size_t
```

### 稀疏矩阵

```cpp
#include "scl/core/sparse.hpp"

// 创建 CSR 矩阵
auto matrix = scl::Sparse<Real, true>::create(
    rows, cols, nnz_per_row);

// 访问行
auto vals = matrix.primary_values(i);
auto idxs = matrix.primary_indices(i);
Index len = matrix.primary_length(i);

// 迭代
for (Index j = 0; j < len; ++j) {
    Real value = vals.ptr[j];
    Index col = idxs.ptr[j];
}
```

### SIMD 操作

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;
const s::Tag d;

// 加载和处理
auto v = s::Load(d, data);
auto result = s::Mul(v, s::Set(d, 2.0));
s::Store(result, d, output);
```

### 内存管理

```cpp
#include "scl/core/registry.hpp"
#include "scl/core/memory.hpp"

// 对齐分配
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// 注册以跟踪
auto& reg = scl::get_registry();
reg.register_ptr(data, 1000 * sizeof(Real), 
                 AllocType::AlignedAlloc);

// 清理
reg.unregister_ptr(data);
```

## 设计原则

### 1. 零依赖

核心模块仅依赖于：
- C++17 标准库
- Google Highway (用于 SIMD)

无其他外部依赖。

### 2. 尽可能使用 Header-Only

大多数核心工具都是 header-only，以便：
- 易于集成
- 更好的内联
- 减少链接时间

### 3. 编译时配置

类型在编译时配置：

```cpp
// 在 CMakeLists.txt 或 config.hpp 中
#define SCL_USE_FLOAT32  // 或 FLOAT64, FLOAT16
#define SCL_USE_INT32    // 或 INT16, INT64
```

这使您能够：
- 单一代码库支持多种精度
- 零运行时开销进行类型选择
- 最优代码生成

### 4. 显式资源管理

无隐藏分配或隐式成本：

```cpp
// 错误：隐藏分配
std::vector<Real> temp;  // 分配！

// 正确：显式分配
Real* temp = reg.new_array<Real>(n);
// ... 使用 temp ...
reg.unregister_ptr(temp);
```

## 常见模式

### RAII 清理

```cpp
class RegistryGuard {
    void* ptr_;
public:
    explicit RegistryGuard(void* ptr) : ptr_(ptr) {}
    ~RegistryGuard() {
        if (ptr_) scl::get_registry().unregister_ptr(ptr_);
    }
    void* release() {
        void* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
};

// 使用
auto* data = reg.new_array<Real>(1000);
RegistryGuard guard(data);
// 作用域退出时自动清理
```

### 模板约束

```cpp
// CSR-like 类型的概念
template <typename T>
concept CSRLike = requires(T t, Index i) {
    { t.rows() } -> std::convertible_to<Index>;
    { t.cols() } -> std::convertible_to<Index>;
    { t.primary_values(i) };
    { t.primary_indices(i) };
};

// 在函数中使用
template <CSRLike MatrixT>
void process(const MatrixT& matrix) {
    // 适用于任何 CSR-like 类型
}
```

### 错误处理

```cpp
// 调试断言 (Release 构建中编译掉)
SCL_ASSERT(i >= 0 && i < n, "Index out of bounds");

// 运行时检查 (始终启用)
SCL_CHECK_ARG(data != nullptr, "Null pointer");
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
```

## 性能提示

### 1. 对批量操作使用 SIMD

```cpp
// 标量循环
for (size_t i = 0; i < n; ++i) {
    output[i] = input[i] * 2.0;
}

// SIMD 循环 (快 2-4 倍)
namespace s = scl::simd;
const s::Tag d;
const size_t lanes = s::Lanes(d);

for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, input + i);
    auto result = s::Mul(v, s::Set(d, 2.0));
    s::Store(result, d, output + i);
}
```

### 2. 为 SIMD 对齐内存

```cpp
// 为 SIMD 对齐分配
Real* data = scl::memory::aligned_alloc<Real>(n, 64);

// 更快的 SIMD 加载/存储
auto v = s::Load(d, data);  // 对齐加载
```

### 3. 最小化 Registry 查找

```cpp
// 错误：在热循环中查找
for (size_t i = 0; i < n; ++i) {
    if (reg.is_registered(ptr)) {  // 昂贵！
        // ...
    }
}

// 正确：检查一次
bool is_reg = reg.is_registered(ptr);
for (size_t i = 0; i < n; ++i) {
    if (is_reg) {
        // ...
    }
}
```

## 下一步

详细了解每个核心模块：

- [类型系统](/zh/cpp/core/types) - 类型系统和配置
- [稀疏矩阵](/zh/cpp/core/sparse) - 稀疏矩阵基础设施
- [注册表](/zh/cpp/core/registry) - 内存生命周期跟踪
- [SIMD](/zh/cpp/core/simd) - SIMD 抽象
- [错误处理](/zh/cpp/core/error) - 断言和异常
- [内存](/zh/cpp/core/memory) - 对齐分配
- [向量化](/zh/cpp/core/vectorize) - 向量化操作

---

::: tip 基础优先
理解核心模块对于使用 SCL-Core 至关重要。在探索更高级模块之前从这里开始。
:::

