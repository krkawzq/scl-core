# error.hpp

> scl/core/error.hpp · SCL Core 异常系统

## 概述

本文件提供 SCL-Core 的异常系统，具有与 C-ABI 兼容的错误代码，用于 Python 集成。所有异常都继承自基础 Exception 类。

主要特性：
- 具有错误代码的统一异常接口
- C-ABI 兼容（用于 Python 绑定）
- 人类可读的错误消息
- 类型安全的异常层次结构

**头文件**: `#include "scl/core/error.hpp"`

---

## 主要 API

### Exception

所有 SCL 异常的基类。

::: source_code file="scl/core/error.hpp" symbol="Exception" collapsed
:::

**算法说明**

提供以下功能的基础异常类：
- 错误代码（ErrorCode 枚举）用于 C-ABI 兼容性
- 人类可读的消息字符串
- 标准异常接口（继承自 std::exception）

所有 SCL 异常都继承自此类。

**边界条件**

- **空消息**：有效但不提供信息
- **无效错误代码**：未定义行为

**数据保证（前置条件）**

- 错误代码必须是有效的 ErrorCode 值
- 消息应该具有描述性

**复杂度分析**

- **时间**：构造 O(1)
- **空间**：O(n)，其中 n 是消息长度

**示例**

```cpp
#include "scl/core/error.hpp"

throw scl::DimensionError("Matrix dimensions must match", 
                         rows, expected_rows);
```

---

### ErrorCode

异常类型的数字标识符。

```cpp
enum class ErrorCode {
    OK = 0,
    UNKNOWN = 1,
    INTERNAL_ERROR = 2,
    INVALID_ARGUMENT = 10,
    DIMENSION_MISMATCH = 11,
    DOMAIN_ERROR = 12,
    TYPE_ERROR = 20,
    IO_ERROR = 30,
    UNREGISTERED_POINTER = 35,
    NOT_IMPLEMENTED = 40
};
```

---

## 异常类型

### DimensionError

因维度不匹配而抛出。

### ValueError

因无效参数值而抛出。

### DomainError

因数学域违反而抛出。

### TypeError

因数据类型不匹配而抛出。

## 相关内容

- [Macros](./macros) - 断言宏（SCL_ASSERT, SCL_CHECK_*）

