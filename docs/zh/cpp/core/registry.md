# registry.hpp

> scl/core/registry.hpp · 具有引用计数的统一高性能内存注册表

## 概述

本文件提供 Registry 类，一个线程安全的内存注册表系统，用于跟踪分配的内存并提供自动清理。它支持简单指针跟踪（refcount=1）和引用计数缓冲区（多个别名）。

主要特性：
- 线程安全的内存跟踪
- 缓冲区别名的引用计数
- 取消注册时自动清理
- 分片设计以减少锁竞争
- 用于异常安全的 RAII 守卫

**头文件**: `#include "scl/core/registry.hpp"`

---

## 主要 API

### Registry

具有引用计数的线程安全内存注册表。

::: source_code file="scl/core/registry.hpp" symbol="Registry" collapsed
:::

**算法说明**

Registry 是一个分片哈希表，用于跟踪内存分配：
- **分片设计**：通过基于哈希的分片减少锁竞争
- **简单指针**：使用 refcount=1 跟踪，在取消注册时立即清理
- **引用计数缓冲区**：支持多个别名（BufferID），当 refcount 达到 0 时清理
- **原子操作**：线程安全的计数器和引用计数

注册表内部使用具有条带锁的 ConcurrentFlatMap 以实现高性能并发访问。

**边界条件**

- **双重注册**：覆盖先前的注册（如果未先取消注册可能会泄漏）
- **取消注册不存在的**：返回 false，无操作
- **并发访问**：所有操作都是线程安全的
- **内存耗尽**：如果哈希表重新哈希失败，注册表本身可能失败

**数据保证（前置条件）**

- 指针在注册时必须有效（非空）
- 分配类型必须匹配实际分配方法
- 如果 AllocType::Custom，自定义删除器必须有效

**复杂度分析**

- **时间**：register/unregister/is_registered 平均情况 O(1)（哈希表）
- **空间**：每个简单指针约 32 字节，每个引用计数缓冲区约 48 字节

**示例**

```cpp
#include "scl/core/registry.hpp"

auto& reg = scl::get_registry();

// 注册简单指针
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// 使用数据...

// 取消注册并清理
reg.unregister_ptr(data);  // 自动调用 delete[]

// 注册引用计数缓冲区
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr1 = reg.get_buffer(id);  // 获取别名
void* ptr2 = reg.get_buffer(id);  // 获取另一个别名

// 取消注册别名
reg.unregister_buffer(id);  // 递减 refcount
reg.unregister_buffer(id);  // Refcount = 0，在这里进行清理
```

---

### register_ptr

注册一个简单指针进行跟踪（refcount = 1）。

::: source_code file="scl/core/registry.hpp" symbol="register_ptr" collapsed
:::

**算法说明**

在注册表中注册指针：
1. 哈希指针地址以确定分片
2. 获取分片锁
3. 将指针记录插入哈希表
4. 更新统计信息（原子递增）

指针将在 unregister_ptr() 时根据 AllocType 使用适当的删除器自动释放。

**边界条件**

- **已注册**：覆盖先前的注册（可能泄漏）
- **空指针**：允许但无用（清理是无操作）

**数据保证（前置条件）**

- ptr 必须使用与 AllocType 匹配的方法分配
- 如果 AllocType::Custom，custom_deleter 必须有效

**复杂度分析**

- **时间**：平均情况 O(1)
- **空间**：O(1) - 在哈希表中存储元数据

**示例**

```cpp
auto& reg = scl::get_registry();

// 注册使用 new[] 分配的数组
Real* arr = new Real[1000];
reg.register_ptr(arr, 1000 * sizeof(Real), AllocType::ArrayNew);

// 注册对齐分配
Real* aligned = scl::memory::aligned_alloc<Real>(1000, 64);
reg.register_ptr(aligned, 1000 * sizeof(Real), AllocType::AlignedAlloc);
```

---

### unregister_ptr

取消注册指针并释放内存。

::: source_code file="scl/core/registry.hpp" symbol="unregister_ptr" collapsed
:::

**算法说明**

取消注册指针并释放内存：
1. 哈希指针地址以找到分片
2. 获取分片锁
3. 在哈希表中找到指针
4. 使用适当的删除器释放内存
5. 从哈希表中移除
6. 更新统计信息

**边界条件**

- **未注册**：返回 false，无操作
- **双重取消注册**：第二次调用返回 false
- **空指针**：安全，返回 false

**数据保证（前置条件）**

- ptr 必须已注册（或为 nullptr，在这种情况下返回 false）

**复杂度分析**

- **时间**：平均情况 O(1)
- **空间**：O(1)

**示例**

```cpp
auto& reg = scl::get_registry();
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// 使用数据...

// 取消注册并清理
bool success = reg.unregister_ptr(data);  // 返回 true，调用 delete[]
// data 现在无效
```

---

### new_buffer

创建新的引用计数缓冲区。

::: source_code file="scl/core/registry.hpp" symbol="new_buffer" collapsed
:::

**算法说明**

分配内存并注册为引用计数缓冲区：
1. 根据 AllocType 分配内存
2. 创建 BufferID（唯一标识符）
3. 在哈希表中注册，refcount = 1
4. 返回 BufferID

可以使用相同的 BufferID 通过 get_buffer() 获取多个别名。只有当 refcount 达到 0 时才释放缓冲区。

**边界条件**

- **分配失败**：返回无效 BufferID (0)，使用 is_valid_buffer_id() 检查
- **零大小**：可能返回无效 BufferID

**数据保证（前置条件）**

- size > 0（通常）
- AllocType 必须有效

**复杂度分析**

- **时间**：平均情况 O(1)（哈希表插入）+ 分配时间
- **空间**：分配的内存 O(size) + 元数据 O(1)

**示例**

```cpp
auto& reg = scl::get_registry();

// 创建缓冲区
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
if (!reg.is_valid_buffer_id(id)) {
    // 处理分配失败
    return;
}

// 获取别名
void* ptr1 = reg.get_buffer(id);
void* ptr2 = reg.get_buffer(id);  // 相同内存，refcount = 3
```

---

### get_buffer

获取引用计数缓冲区的指针别名（递增 refcount）。

::: source_code file="scl/core/registry.hpp" symbol="get_buffer" collapsed
:::

**算法说明**

返回指向缓冲区的指针并递增引用计数：
1. 哈希 BufferID 以找到分片
2. 获取分片锁
3. 在哈希表中找到缓冲区
4. 递增原子 refcount
5. 返回指针

**边界条件**

- **无效 BufferID**：返回 nullptr
- **已释放**：返回 nullptr（安全处理竞争条件）

**数据保证（前置条件）**

- BufferID 必须有效（从 new_buffer 获得）

**复杂度分析**

- **时间**：平均情况 O(1)
- **空间**：O(1)

**示例**

```cpp
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr = reg.get_buffer(id);  // Refcount = 2（1 来自 new_buffer + 1 来自 get_buffer）
```

---

### unregister_buffer

取消注册缓冲区别名（递减 refcount，当 refcount = 0 时释放）。

::: source_code file="scl/core/registry.hpp" symbol="unregister_buffer" collapsed
:::

**算法说明**

递减引用计数，当计数达到 0 时释放缓冲区：
1. 哈希 BufferID 以找到分片
2. 获取分片锁
3. 在哈希表中找到缓冲区
4. 递减原子 refcount
5. 如果 refcount == 0：释放内存并从表中移除
6. 否则：仅递减计数器

**边界条件**

- **无效 BufferID**：返回 false
- **双重取消注册**：安全，refcount 不能低于 0
- **从多个线程取消注册**：线程安全，原子操作

**数据保证（前置条件）**

- BufferID 必须有效

**复杂度分析**

- **时间**：平均情况 O(1)
- **空间**：O(1)

**示例**

```cpp
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr1 = reg.get_buffer(id);  // Refcount = 2
void* ptr2 = reg.get_buffer(id);  // Refcount = 3

reg.unregister_buffer(id);  // Refcount = 2
reg.unregister_buffer(id);  // Refcount = 1
reg.unregister_buffer(id);  // Refcount = 0，在这里释放内存
```

---

## 工具类

### RegistryGuard

用于自动取消注册的 RAII 守卫。

::: source_code file="scl/core/registry.hpp" symbol="RegistryGuard" collapsed
:::

**算法说明**

在作用域退出时自动取消注册指针的 RAII 包装器：
- 构造函数：存储指针（不注册）
- 析构函数：如果仍持有指针，调用 unregister_ptr()
- release()：阻止自动取消注册

适用于即使发生异常也必须进行取消注册的异常安全代码。

**示例**

```cpp
auto& reg = scl::get_registry();
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

{
    RegistryGuard guard(data);
    // 使用数据...
    // 在作用域退出时自动清理（即使抛出异常）
}
```

---

### get_registry

获取全局 Registry 实例的引用。

**返回**：单例 Registry 实例的引用

**线程安全**：安全 - 单例初始化是线程安全的

**示例**

```cpp
auto& reg = scl::get_registry();
reg.register_ptr(ptr, size, AllocType::ArrayNew);
```

---

## 类型别名

```cpp
using BufferID = std::uint64_t;  // 引用计数缓冲区的唯一标识符
using HandlerRegistry = Registry;  // 遗留别名
```

## 枚举：AllocType

```cpp
enum class AllocType {
    ArrayNew,      // new[] / delete[]
    ScalarNew,     // new / delete
    AlignedAlloc,  // aligned_alloc / aligned_free
    Custom         // 自定义删除器
};
```

## 设计说明

### 分片架构

Registry 使用基于哈希的分片来减少锁竞争：
- 多个分片（通常 16-64 个）
- 指针哈希确定分片
- 每个分片都有自己的锁
- 通过 num_shards 因子减少竞争

### 引用计数

引用计数缓冲区支持多个别名：
- 每个 get_buffer() 递增 refcount
- 每个 unregister_buffer() 递减 refcount
- 当 refcount 达到 0 时释放缓冲区
- 使用原子操作实现线程安全

### 线程安全

所有公共方法都是线程安全的：
- 通过条带锁进行内部同步
- 原子引用计数
- 支持并发读取
- 写入操作在每个分片上互斥

## 相关内容

- [内存管理](./memory) - 返回用于注册的指针的分配函数
- [Sparse Matrix](./sparse) - 使用 Registry 进行元数据数组跟踪

