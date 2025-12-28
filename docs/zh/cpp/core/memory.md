# memory.hpp

> scl/core/memory.hpp · 为高性能计算优化的底层内存操作

## 概述

本文件提供为高性能计算优化的底层内存操作，支持 SIMD 加速。包括对齐内存分配、初始化（填充/清零）、数据移动（复制）和缓存优化工具。

主要特性：
- 用于 SIMD 操作的对齐内存分配
- SIMD 加速的填充和清零操作
- 具有重叠处理的快速复制操作
- 对齐缓冲区的 RAII 包装器
- 缓存优化提示（预取）

**头文件**: `#include "scl/core/memory.hpp"`

---

## 主要 API

### aligned_alloc

为基本类型分配对齐内存。

::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc" collapsed
:::

**算法说明**

分配对齐到指定边界的内存（默认 64 字节，用于缓存行对齐）：

1. 检查 count == 0，如果是则返回 nullptr
2. 计算所需总字节数：count * sizeof(T)
3. 使用特定平台的对齐分配：
   - C++17+：对齐 operator new
   - POSIX：posix_memalign
   - Windows：_aligned_malloc
4. 将分配的内存初始化为零
5. 返回对齐指针，或在失败时返回 nullptr

默认对齐 64 字节匹配 AVX-512 缓存行大小，确保最佳 SIMD 性能。

**边界条件**

- **count == 0**：返回 nullptr（不分配）
- **分配失败**：返回 nullptr（调用者必须检查）
- **溢出**：检查 count * sizeof(T) 溢出，返回 nullptr
- **无效对齐**：必须是 2 的幂且 >= sizeof(void*)

**数据保证（前置条件）**

- `alignment` 必须是 2 的幂
- `alignment >= sizeof(void*)`（平台最小值）
- `count * sizeof(T)` 不得溢出
- T 必须是可平凡构造的

**复杂度分析**

- **时间**：O(1) - 单次分配调用
- **空间**：O(count * sizeof(T)) - 分配的内存

**示例**

```cpp
#include "scl/core/memory.hpp"

// 为 SIMD 操作分配对齐缓冲区
Real* data = scl::memory::aligned_alloc<Real>(
    1000,    // 元素数量
    64       // 字节对齐（默认：64）
);

if (data == nullptr) {
    // 处理分配失败
    return;
}

// 使用数据进行 SIMD 操作...
// 内存已初始化为零

// 使用 aligned_free 释放（必须使用 aligned_free，而不是 free()）
scl::memory::aligned_free(data, 64);
```

---

### aligned_free

释放由 aligned_alloc 分配的内存。

::: source_code file="scl/core/memory.hpp" symbol="aligned_free" collapsed
:::

**算法说明**

使用特定平台的释放函数释放由 aligned_alloc 分配的内存：
- C++17+：对齐 operator delete
- POSIX：free
- Windows：_aligned_free

对齐参数必须与 aligned_alloc 中使用的值匹配。

**边界条件**

- **ptr == nullptr**：安全调用，不执行任何操作
- **双重释放**：未定义行为（调用者必须跟踪所有权）
- **对齐不匹配**：未定义行为（必须与 aligned_alloc 匹配）

**数据保证（前置条件）**

- ptr 必须为 nullptr 或由 aligned_alloc 分配
- alignment 必须与 aligned_alloc 中使用的对齐匹配

**复杂度分析**

- **时间**：O(1) - 单次释放调用
- **空间**：O(1)

**示例**

```cpp
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// 使用数据...

// 使用匹配的对齐释放
scl::memory::aligned_free(data, 64);
// data 现在无效（悬空指针）
```

---

### AlignedBuffer

对齐内存分配的 RAII 包装器。

::: source_code file="scl/core/memory.hpp" symbol="AlignedBuffer" collapsed
:::

**算法说明**

管理对齐内存分配的 RAII 包装器：
- 构造函数：使用 aligned_alloc 分配对齐内存
- 析构函数：使用 aligned_free 自动释放内存
- 可移动：支持移动构造和赋值
- 不可复制：复制操作已删除

提供安全的异常处理 - 如果发生异常，析构函数自动释放内存。

**边界条件**

- **分配失败**：缓冲区无效（operator bool 返回 false）
- **从移动**：源缓冲区变为无效（空）
- **双重移动**：第二次移动无效

**数据保证（前置条件）**

- 对齐必须是 2 的幂且 >= sizeof(void*)
- T 必须是可平凡构造的

**复杂度分析**

- **构造**：O(1) - 单次分配
- **析构**：O(1) - 单次释放
- **访问**：O(1) - 指针解引用

**示例**

```cpp
#include "scl/core/memory.hpp"

{
    // RAII 缓冲区 - 在作用域退出时自动释放
    scl::memory::AlignedBuffer<Real> buffer(1000, 64);
    
    if (buffer) {  // 检查分配成功
        Real* data = buffer.get();
        Array<Real> span = buffer.span();
        
        // 使用数据...
        span[0] = 1.0;
        span[1] = 2.0;
        
        // 在作用域退出时自动清理
    }
}
```

---

### fill

使用 SIMD 加速填充内存值。

::: source_code file="scl/core/memory.hpp" symbol="fill" collapsed
:::

**算法说明**

使用 SIMD 优化用指定值填充内存范围：

1. 创建带有广播值的 SIMD 向量
2. 使用 4 路展开的 SIMD 循环处理大部分数据
3. 使用 SIMD 循环处理剩余部分
4. 使用标量循环处理尾部元素

自动使用编译器选择的特定平台 SIMD 指令（AVX2/AVX-512/NEON）。

**边界条件**

- **空 span**：立即返回（无操作）
- **空指针**：如果 span.len == 0 则安全，否则未定义行为
- **大值**：所有元素设置为相同值（包括 NaN）

**数据保证（前置条件）**

- span.ptr 必须有效或为 nullptr（如果 span.len == 0）
- 内存必须可写

**复杂度分析**

- **时间**：O(n / lanes)，其中 lanes 是 SIMD 宽度（通常比标量快 4-16 倍）
- **空间**：O(1) 辅助空间

**示例**

```cpp
#include "scl/core/memory.hpp"

Real* data = new Real[1000];
Array<Real> span(data, 1000);

// 填充值 1.5
scl::memory::fill(span, Real(1.5));

// span 中的所有元素现在都是 1.5
```

---

### zero

高效地将内存清零。

::: source_code file="scl/core/memory.hpp" symbol="zero" collapsed
:::

**算法说明**

高效地清零内存：
- 对于平凡类型：使用优化的 memset（最快）
- 对于非平凡类型：使用 fill(span, T(0))

为常见的零化基本类型情况提供最佳性能。

**边界条件**

- **空 span**：立即返回
- **空指针**：如果 span.len == 0 则安全
- **非平凡类型**：使用 fill，可能调用构造函数

**数据保证（前置条件）**

- span.ptr 必须有效或为 nullptr（如果 span.len == 0）

**复杂度分析**

- **时间**：平凡类型 O(n * sizeof(T))（memset），非平凡类型 O(n / lanes)（SIMD fill）
- **空间**：O(1) 辅助空间

**示例**

```cpp
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);
Array<Real> span(data, 1000);

// 清零内存
scl::memory::zero(span);

// 所有元素现在都是 0.0
```

---

### copy_fast

快速复制，假设无重叠（使用 memcpy 语义）。

::: source_code file="scl/core/memory.hpp" symbol="copy_fast" collapsed
:::

**算法说明**

为不重叠范围优化的快速内存复制：
- 对可平凡复制类型使用 memcpy
- 编译器可以使用 __restrict__ 语义进行优化
- 无重叠检查（最快选项）

**边界条件**

- **重叠范围**：未定义行为（使用 copy() 代替）
- **大小不匹配**：未定义行为（src.len 必须 == dst.len）
- **空范围**：立即返回

**数据保证（前置条件）**

- src.len == dst.len
- src 和 dst 不得重叠
- 两个指针必须有效

**复杂度分析**

- **时间**：O(n * sizeof(T)) - 通常非常快（memcpy）
- **空间**：O(1) 辅助空间

**示例**

```cpp
Real* src = new Real[1000];
Real* dst = scl::memory::aligned_alloc<Real>(1000, 64);

Array<const Real> src_span(src, 1000);
Array<Real> dst_span(dst, 1000);

// 快速复制（无重叠检查）
scl::memory::copy_fast(src_span, dst_span);

// dst 现在包含 src 的副本
```

---

## 工具函数

### copy

具有重叠处理的复制（使用 memmove 语义）。对重叠范围安全，但比 copy_fast 慢。

**复杂度**：O(n * sizeof(T))

### prefetch_read / prefetch_write

用于优化的缓存预取提示。

**复杂度**：O(1) - 单次预取指令

## 平台支持

- **C++17+**：使用对齐 operator new/delete
- **POSIX**：使用 posix_memalign 和 free
- **Windows**：使用 _aligned_malloc 和 _aligned_free

## 性能注意事项

- **对齐分配**：64 字节对齐匹配 AVX-512 缓存行大小
- **SIMD 操作**：自动使用最佳可用指令（AVX2/AVX-512/NEON）
- **copy_fast**：在不可能重叠时最快，使用 copy() 以确保安全

## 相关内容

- [类型系统](./types) - 用于内存视图的 Array<T> 类型
- [SIMD](./simd) - SIMD 抽象层
- [注册表](./registry) - 用于分配缓冲区的内存跟踪

