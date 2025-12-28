# macros.hpp

> scl/core/macros.hpp · 编译器抽象和优化提示

## 概述

本文件提供用于平台检测、优化提示和断言的编译器宏。这些宏抽象了特定于编译器的功能，并提供可移植的抽象。

主要特性：
- 平台检测（Windows、POSIX、Linux、macOS）
- 优化提示（强制内联、无丢弃、限制）
- 断言宏（调试和发布构建）
- 编译器抽象层

**头文件**: `#include "scl/core/macros.hpp"`

---

## 主要 API

### SCL_FORCE_INLINE

强制函数内联。

**用法**: `SCL_FORCE_INLINE void function() { ... }`

**平台**: 映射到 `__forceinline` (MSVC) 或 `__attribute__((always_inline))` (GCC/Clang)

---

### SCL_NODISCARD

如果函数返回值被忽略则警告。

**用法**: `SCL_NODISCARD bool check_validity();`

**平台**: 映射到 `[[nodiscard]]` (C++17) 或特定于编译器的属性

---

### SCL_ASSERT

调试断言（在发布构建中编译掉）。

**用法**: `SCL_ASSERT(condition, "message");`

**行为**：
- 调试构建：如果条件为 false 则终止
- 发布构建：无操作（编译掉）

---

### SCL_CHECK_ARG / SCL_CHECK_DIM

始终执行的运行时检查。

**用法**：
```cpp
SCL_CHECK_ARG(ptr != nullptr, "Null pointer");
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
```

**行为**：如果条件为 false 则抛出异常

---

## 平台检测

- `SCL_PLATFORM_WINDOWS` - Windows 平台
- `SCL_PLATFORM_POSIX` - POSIX 兼容系统
- `SCL_PLATFORM_LINUX` - Linux
- `SCL_PLATFORM_MACOS` - macOS

## 相关内容

- [错误处理](./error) - 由 CHECK 宏抛出的异常类型

