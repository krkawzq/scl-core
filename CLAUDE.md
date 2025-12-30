# AGENT.md - AI Developer Guide for scl-core

本节新增要求：  
函数定义时必须遵循以下格式规范：

1. `template`（如有模板）需单独一行。
2. `[[nodiscard]]`、`SCL_FORCE_INLINE`、`constexpr`等标记须独占一行，紧随 `template`（如有）之后。
3. 函数实现必须以 `auto` 或 `void` 开头，返回类型需使用 `-> T`（Trailing Return Type）指定，即 `auto f(...) -> T` 或 `void f(...) -> void`。
4. 示例（√）：

    ```cpp
    /// @brief 示例函数
    /// @tparam T 类型参数
    /// @param[in] x 输入值
    /// @return 值加一
    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    auto add_one(T x) -> T {
        return x + 1;
    }
    ```

---

This document defines the coding standards and protocols for the `scl-core` project.

**Core Mission**: Build a high-performance biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.

---

## 1. C++ Standards

### 1.1 Language Standard

- **C++20** (ISO/IEC 14882:2020)
- **Compilers**: GCC 11+, Clang 14+, MSVC 19.29+

**Compile Flags**:  
Release/Debug builds should use appropriate C++20 and optimization/sanitizer flags.

### 1.2 Modern C++20 Features (Required)

- Use concepts for type constraints (prefer over SFINAE)
- Use `std::span` for non-owning views (replace `T* + size`)
- Use `constexpr`/`consteval` for compile-time computation as much as possible
- Use attribute syntax directly (e.g., `[[nodiscard]]`, `[[likely]]`, `[[unlikely]]`, `[[no_unique_address]]`)
- Always use trailing return type: `auto f() -> T` (required for all non-void functions)
- Use `std::source_location` for error reporting (replace `__FILE__`, `__LINE__`)
- Use designated initializers for config structs

### 1.3 Namespaces

- `scl::core`: Core types, utilities, error handling
- `scl::memory`: Memory operations (unified memory module)
- `scl::kernel`: Computational kernels
- `scl::math`: Mathematical functions
- `scl::simd`: SIMD abstractions
- `scl::threading`: Parallelization layer
- `scl::binding`: C-ABI interface

---

## 2. Code Style

### 2.1 File Structure

Every header should start with `#pragma once`.  
Sections should be commented for clarity.  
Implementation is organized under the appropriate `scl::module` namespace.

### 2.2 Formatting Rules

- Indentation: 4 spaces
- Line length: under 100 chars (max 120)
- Brace style: Same line
- Include order: Platform headers → STL → SCL headers

### 2.3 Function Declaration Style

- Each of the following must be in its own line (顺序为 template → 属性/修饰符 → 定义)：
    - `template` 声明单独一行（如有）
    - 所有属性/修饰符（如 `[[nodiscard]]`、`SCL_FORCE_INLINE`、`constexpr` 等）每个独占一行
    - 函数定义以 `auto` 或 `void` 开头，必须使用 trailing return type（-> Type），不得遗漏，即 `auto f(...) -> T`
- Batch small single-line accessors when possible.

### 2.4 Variable Naming

- Local variables: `snake_case`
- Member variables: trailing underscore (e.g. `data_`)
- Constants: `UPPER_SNAKE`
- Template params: `PascalCase`
- Concepts: `PascalCase` ending with able/like (e.g. `Arithmetic`, `SpanLike`)

---

## 3. Platform Macros (scl/core/platform.hpp)

### 3.1 Compiler Detection

- Use macros such as `SCL_COMPILER_GCC`, `SCL_COMPILER_CLANG`, `SCL_COMPILER_MSVC`, `SCL_COMPILER_GCC_LIKE` for conditional compilation.

### 3.2 Architecture Detection

- Use macros such as `SCL_ARCH_X86_64`, `SCL_ARCH_ARM64`, `SCL_ARCH_SSE4`, `SCL_ARCH_AVX2`, `SCL_ARCH_AVX512`, `SCL_ARCH_NEON` for architecture-specific code.

### 3.3 Function Attributes

- Use macros such as `SCL_FORCE_INLINE`, `SCL_NOINLINE`, `SCL_RESTRICT`, `SCL_ASSUME(expr)`, `SCL_PREFETCH(addr, rw, locality)`, `SCL_ALIGNED(n)` to control code generation, inlining, aliasing and prefetching.

### 3.4 Deprecated Macros

- Do NOT use ancient macro wrappers for C++ attributes. Use C++20 attributes directly (e.g. use [[nodiscard]] instead of `SCL_NODISCARD`).

---

## 4. Memory Module (scl/mem/)

### 4.1 Core Principle

- All memory operations must use the `scl::memory` module, NOT `std::` or platform-specific APIs directly.

### 4.2 Memory Module API (to be defined)

- Should cover copy, async copy, memset, fill, aligned alloc/free, and RAII aligned buffer abstractions.

---

## 5. Error Handling

### 5.1 Check Macros Hierarchy

| Macro               | Phase      | Failure         | Use Case                       |
|---------------------|------------|-----------------|--------------------------------|
| `static_assert`     | Compile    | Compilation     | C++ standard checks            |
| `SCL_STATIC_CHECK`  | Compile    | Compilation     | Type/precision validation      |
| `SCL_CHECK`         | Runtime    | Throws          | Argument validation            |
| `SCL_DEBUG_ASSERT`  | Debug only | Terminate       | Internal invariants (debug)    |

### 5.2 Usage Rules

- Compile-time checks: Validate types, precision bounds, shapes/statics, etc. with static_assert or `SCL_STATIC_CHECK`.
- Runtime checks: Validate inputs, sizes, user error, with `SCL_CHECK` (throws)
- Debug assertions: For invariants that should never fail in valid execution, only trigger in debug builds.

### 5.3 Exception Types

- The design should define a base `Error` exception and derived exceptions such as `DimensionError`, `ValueError`, `MemoryError`, `NotImplementedError` for typical categories.

---

## 6. Config System

### 6.1 Config Base Template

- All operator configs MUST inherit from `ConfigBase<Derived>` which provides `validate()` (throws if invalid) and `is_valid()` (returns bool, noexcept).

### 6.2 Config Definition Pattern

- Configs define their members, validation rules in `validate_impl()`, and logic in `is_valid_impl()`.

### 6.3 Usage in Operators

- Operators should accept a config argument (with default constructed value) and invoke `.validate()` at entry.

---

## 7. Operator Optimization Guidelines

### 7.1 Performance Hierarchy

All hot-path functions must optimize in the following order:

1. Algorithm selection (lowest complexity)
2. Memory access pattern (cache-friendly)
3. SIMD vectorization
4. Loop unrolling
5. Branch prediction hints (`[[likely]]`/`[[unlikely]]`)
6. Prefetching
7. Register pressure minimization

### 7.2 Required Optimizations for Hot Paths

- Follow the above order and always annotate performance-critical routines.
- Use platform prefetch and aliasing hints as needed.

### 7.3 Accessor Functions

- All accessors (e.g. `data()`, `rows()`, `cols()`, index operators) must be marked with strong inlining (`SCL_FORCE_INLINE`).

---

## 8. Documentation Standard (Doxygen)

### 8.1 Comment Style

- Use `///` and @ tags for all public APIs and important structures.
- Documentation goes in source headers; CI will auto-generate docs.

### 8.2 Required Tags

Keep the following minimum documentation:

- `@brief`        One-line description (mandatory)
- `@tparam`        If templated, describe the template param
- `@param[in/out]`   Parameter names, indicate direction
- `@return`       If it returns a value, describe it
- `@throws`       If can throw, what exceptions
- `@pre`         Preconditions
- `@post`        Postconditions
- `@note`        Important info, e.g. complexity, thread-safety
- `@warning`      Pitfalls, sharp edges

### 8.3 Documentation Examples

- Use the above tags for each API; see template blocks for guidance.
- Document logic, thresholds, assumptions directly above the critical code sections.

---

## 9. Boundary and Precision Handling

### 9.1 Numeric Precision Validation

- Use compile-time checks for floating-point types and precision in numeric kernels, with `SCL_STATIC_CHECK`.

### 9.2 Boundary Handling Pattern

- Main loops should process in vector-sized chunks; remainder handled with scalar logic.
- Always guard against empty input and document surprise paths.

### 9.3 Numeric Stability Patterns

- Use numerically stable approaches (e.g., log-sum-exp max-shifting) and document the reasoning inline.

---

## 10. Type Aliases (scl/core/types.hpp)

- Core types should be strongly-typed using aliases, e.g., `Index`, `Size`, `Real`, and span/dimension structs.
- E.g., use `scl::Index`, `scl::Size`, and explicit `Dim2`, `Dim3`, `Dim4` structures for tensor shapes.

---

## 11. Development Checklist

### Code Quality
- Ensure code compiles with strict C++20 warnings and errors enabled.
- Use C++20 attributes directly.
- All memory ops via `scl::memory`
- Platform macros from `scl/core/platform.hpp`
- Trailing return type, attributes on own line

### Error Handling
- Compile-time checks for type/precision
- Runtime checks with throws for arguments
- Debug assertions only for invariants
- No use of runtime `assert()` in production

### Performance
- All accessors marked with force inline
- Use `[[likely]]`/`[[unlikely]]` on branches
- Prefetch hot loops, unroll critical kernels
- Use restrict on non-aliased pointers

### Config System
- All configs inherit `ConfigBase`
- Validation functions as specified

### Documentation
- Doxygen with required tags on all public APIs and critical logic

---

## 12. Quick Reference

### 12.1 Check Macro Summary

- `SCL_STATIC_CHECK`: Compile-time check (type/precision)
- `SCL_CHECK`: Runtime argument validation (throws)
- `SCL_DEBUG_ASSERT`: Debug-only, for internal invariants

### 12.2 Platform Macro Summary

- `SCL_FORCE_INLINE`: Force inline  
- `SCL_NOINLINE`: Prevent inline  
- `SCL_RESTRICT`: Mark pointer as no-alias  
- `SCL_ASSUME(expr)`: Optimizer hint  
- `SCL_PREFETCH(addr, rw, locality)`: Cache prefetch  
- `SCL_ALIGNED(n)`: Alignment specifier  

### 12.3 Function Declaration Template

- Provide a doxygen comment with all required tags.
- `template` on its own line (if needed)
- All attribute/qualifier specifiers (`[[nodiscard]]`, `SCL_FORCE_INLINE`, `constexpr` etc.) each on their own line, directly after `template` (if any)
- Function implementation begins with `auto` or `void`, and always uses trailing return type (`-> T`)

    ```cpp
    /// @brief Adds 1 to value
    /// @tparam T Value type
    /// @param[in] x Input value
    /// @return x + 1
    template<typename T>
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    auto add_one(T x) -> T {
        return x + 1;
    }
    ```

