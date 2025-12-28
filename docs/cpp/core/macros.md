# Compiler Macros

Compiler abstractions and optimization hints for platform detection and code generation.

## Overview

Macros provide:

- **Platform Detection** - Identify operating system and architecture
- **Compiler Hints** - Force inline, no return, restrict pointers
- **Optimization Directives** - Prefetch, likely/unlikely branches
- **Feature Detection** - SIMD availability, threading backend

## Platform Detection

### Operating System

```cpp
SCL_PLATFORM_WINDOWS   // 1 if Windows, 0 otherwise
SCL_PLATFORM_POSIX     // 1 if POSIX-compliant (Linux, macOS, BSD)
SCL_PLATFORM_UNIX      // 1 if Unix-like system
SCL_PLATFORM_LINUX     // 1 if Linux
SCL_PLATFORM_MACOS     // 1 if macOS
```

**Usage:**
```cpp
#if SCL_PLATFORM_WINDOWS
    // Windows-specific code
#elif SCL_PLATFORM_LINUX
    // Linux-specific code
#elif SCL_PLATFORM_MACOS
    // macOS-specific code
#endif
```

### Architecture

```cpp
SCL_ARCH_X86           // x86/x86_64 architecture
SCL_ARCH_ARM           // ARM architecture
SCL_ARCH_AARCH64       // ARM64/AArch64
```

## Compiler Hints

### SCL_FORCE_INLINE

Force function inlining:

```cpp
SCL_FORCE_INLINE Real fast_compute(Real x) {
    return x * 2.0;
}
```

**Purpose:**
- Override compiler inlining decisions
- Use for hot-path functions that must be inlined

**Platform Behavior:**
- GCC/Clang: `__attribute__((always_inline))`
- MSVC: `__forceinline`

### SCL_NOINLINE

Prevent function inlining:

```cpp
SCL_NOINLINE void slow_function() {
    // Large function body
}
```

**Purpose:**
- Reduce code size for large functions
- Prevent inlining of debugging/logging functions

### SCL_NORETURN

Indicate function never returns:

```cpp
SCL_NORETURN void fatal_error(const char* msg) {
    std::cerr << msg << std::endl;
    std::abort();
}
```

**Purpose:**
- Helps compiler optimize (eliminate unreachable code)
- Improves static analysis

### SCL_RESTRICT

Restrict pointer aliasing (C99 restrict):

```cpp
void compute(Real* SCL_RESTRICT output,
             const Real* SCL_RESTRICT input,
             size_t n) {
    // Compiler assumes output and input don't alias
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0;
    }
}
```

**Purpose:**
- Enable aggressive compiler optimizations
- Allows vectorization that would be unsafe with aliasing

**Platform Behavior:**
- C++: `__restrict__` (GCC/Clang) or `__restrict` (MSVC)

### SCL_NODISCARD

Warn if return value is ignored:

```cpp
SCL_NODISCARD bool validate_input(const char* data);
```

**Usage:**
```cpp
validate_input(data);  // Warning: return value ignored
bool ok = validate_input(data);  // OK
```

**Purpose:**
- Prevent accidental ignoring of error codes
- Enforce checking return values

## Optimization Directives

### SCL_LIKELY / SCL_UNLIKELY

Branch prediction hints:

```cpp
if (SCL_LIKELY(n > 0)) {
    // Fast path (most common case)
} else {
    // Slow path (rare case)
}

if (SCL_UNLIKELY(error_condition)) {
    // Error handling (rare)
}
```

**Purpose:**
- Help CPU branch predictor
- Minor performance improvement in hot loops

**Platform Behavior:**
- GCC/Clang: `__builtin_expect(condition, 1/0)`
- MSVC: No-op (ignored)

### SCL_PREFETCH

Prefetch memory into cache:

```cpp
for (size_t i = 0; i < n; ++i) {
    SCL_PREFETCH(&data[i + 8]);  // Prefetch 8 elements ahead
    process(data[i]);
}
```

**Purpose:**
- Reduce memory latency
- Hide cache miss stalls

**Platform Behavior:**
- GCC/Clang: `__builtin_prefetch(ptr, rw, locality)`
- MSVC: `_mm_prefetch` (intrinsic)

## Feature Detection

### SIMD Configuration

```cpp
SCL_ONLY_SCALAR        // Disable SIMD, use scalar fallback
```

**Purpose:**
- Debugging (scalar code is easier to debug)
- Compatibility testing
- Fallback for unsupported architectures

### Threading Backend

```cpp
SCL_THREADING_OPENMP   // Use OpenMP backend
SCL_THREADING_TBB      // Use Intel TBB backend
SCL_THREADING_BS       // Use BS::thread_pool backend
SCL_THREADING_SERIAL   // Single-threaded (no parallelization)
```

**Configuration:**
- Set in CMake or config.hpp
- Determines parallel processing backend

## Debug Build Configuration

### SCL_DEBUG

Enable debug assertions:

```cpp
#ifdef SCL_DEBUG
    SCL_ASSERT(condition, "Message");
#endif
```

**Purpose:**
- Enable additional validation in debug builds
- Compile out in release builds for performance

### SCL_ENABLE_ASSERTIONS

Force enable assertions even in release:

```cpp
#define SCL_ENABLE_ASSERTIONS 1
```

**Use Cases:**
- Testing and validation
- Production debugging

## Alignment Macros

### SCL_ALIGNAS

Specify alignment:

```cpp
struct SCL_ALIGNAS(64) AlignedData {
    Real values[16];
};
```

**Platform Behavior:**
- C++11: `alignas(alignment)`
- Fallback: Compiler-specific attributes

### SCL_ALIGNED

Alignment attribute for variables:

```cpp
SCL_ALIGNED(64) Real buffer[1000];
```

**Purpose:**
- Ensure SIMD-friendly alignment
- Improve cache performance

## Attribute Helpers

### SCL_DEPRECATED

Mark function as deprecated:

```cpp
SCL_DEPRECATED("Use new_function instead")
void old_function();
```

**Purpose:**
- Warn users of deprecated APIs
- Maintain backward compatibility during migration

### SCL_MAYBE_UNUSED

Suppress unused variable warnings:

```cpp
SCL_MAYBE_UNUSED static constexpr int DEBUG_FLAG = 0;
```

**Use Cases:**
- Conditional compilation variables
- Debug-only variables

## Common Patterns

### Platform-Specific Code

```cpp
#if SCL_PLATFORM_WINDOWS
    #include <windows.h>
#elif SCL_PLATFORM_POSIX
    #include <unistd.h>
#endif
```

### Force Inline Hot Paths

```cpp
SCL_FORCE_INLINE Real dot_product(const Real* a, const Real* b, size_t n) {
    Real sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

### Branch Prediction

```cpp
void process_array(const Real* data, size_t n) {
    if (SCL_LIKELY(n > 0)) {
        // Optimize for non-empty case
        for (size_t i = 0; i < n; ++i) {
            process(data[i]);
        }
    }
}
```

---

::: tip Use Sparingly
Macros should be used judiciously. Prefer C++ language features (constexpr, inline, etc.) when possible. Use macros only for platform-specific or compiler-specific features.
:::

