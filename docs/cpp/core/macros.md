# macros.hpp

> scl/core/macros.hpp · Compiler abstractions and optimization hints

## Overview

This file provides compiler macros for platform detection, optimization hints, and assertions. These macros abstract away compiler-specific features and provide portable abstractions.

Key features:
- Platform detection (Windows, POSIX, Linux, macOS)
- Optimization hints (force inline, no discard, restrict)
- Assertion macros (debug and release builds)
- Compiler abstraction layer

**Header**: `#include "scl/core/macros.hpp"`

---

## Main APIs

### SCL_FORCE_INLINE

Force function to be inlined.

**Usage**: `SCL_FORCE_INLINE void function() { ... }`

**Platform**: Maps to `__forceinline` (MSVC) or `__attribute__((always_inline))` (GCC/Clang)

---

### SCL_NODISCARD

Warn if function return value is ignored.

**Usage**: `SCL_NODISCARD bool check_validity();`

**Platform**: Maps to `[[nodiscard]]` (C++17) or compiler-specific attribute

---

### SCL_ASSERT

Debug assertion (compiled out in release builds).

**Usage**: `SCL_ASSERT(condition, "message");`

**Behavior**: 
- Debug builds: Terminates if condition is false
- Release builds: No-op (compiled out)

---

### SCL_CHECK_ARG / SCL_CHECK_DIM

Runtime checks that always execute.

**Usage**: 
```cpp
SCL_CHECK_ARG(ptr != nullptr, "Null pointer");
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
```

**Behavior**: Throws exception if condition is false

---

## Platform Detection

- `SCL_PLATFORM_WINDOWS` - Windows platform
- `SCL_PLATFORM_POSIX` - POSIX-compliant systems
- `SCL_PLATFORM_LINUX` - Linux
- `SCL_PLATFORM_MACOS` - macOS

## See Also

- [Error Handling](./error) - Exception types thrown by CHECK macros
