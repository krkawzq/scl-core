# scl-core Development and Coding Standard

This document defines the unified coding and development standard for all contributors to the `scl-core` project, ensuring consistency, reliability, and high-performance C++20 code suitable for cross-language integration.

---

## 1. Project Mission

Deliver a high-performance, zero-overhead C++ kernel library for biological operators, with a stable C-ABI and Python interoperability.

---

## 2. C++ Language and Compiler Rules

- **Required Standard:** C++20 (ISO/IEC 14882:2020)
- **Supported Compilers:** GCC 11+, Clang 14+, MSVC 19.29+
- **Build Flags:** Use C++20 features, enable all warnings, and apply appropriate optimizations/sanitizers for release and debug builds.

---

## 3. Modern C++20 Practices

- Prefer C++20 concepts for type constraints (avoid SFINAE).
- Use `std::span` for non-owning data views (do not use raw pointers and size pairs).
- Leverage `constexpr` and `consteval` for compile-time computation wherever possible.
- Use C++20 attribute syntax directly (e.g. `[[nodiscard]]`, `[[likely]]`, `[[no_unique_address]]`).
- Use `std::source_location` for error reporting instead of legacy macros.
- Favor designated initializers for configuration structures.

---

## 4. Namespace Layout

- `scl::core` — Core types/utilities/error.
- `scl::memory` — Unified memory operations.
- `scl::kernel` — Computational kernels.
- `scl::math` — Mathematical primitives.
- `scl::simd` — SIMD abstraction helpers.
- `scl::threading` — Parallelization.
- `scl::binding` — C-ABI and cross-language interfaces.

---

## 5. Code Style Guide

### 5.1 File Structure

- Every header must begin with `#pragma once`.
- Divide content using clear section comments.
- Place all implementation under its designated `scl::module` namespace.

### 5.2 Formatting

- Indentation: 4 spaces.
- Line length: < 100 chars (absolute max 120 chars).
- Brace style: Same line.
- Include order: System/platform headers, then STL, then scl headers.

### 5.3 Variable & Naming Conventions

- Local variables: `snake_case`
- Member variables: end with underscore (`member_`)
- Constants: `UPPER_SNAKE`
- Template parameters and concepts: `PascalCase` (concepts should end with -able or -like)

---

## 6. Function Declaration and Formatting Convention

**Mandatory Format for All Functions:**

1. Place `template` (if present) on its own line.
2. Place each attribute or qualifier (`[[nodiscard]]`, `SCL_FORCE_INLINE`, `constexpr`, etc.) on its own line, immediately after the template line if present.
3. Function return types:
   - If the function returns a value: use `auto func(...) -> T` (trailing return type)
   - If the function returns nothing: use `void func(...)` (do **not** use `auto func(...) -> void`)
4. Batch trivial single-line accessors where practical.

**Examples:**

```cpp
// With return value
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto add_one(T x) -> T {
    return x + 1;
}

// Without return value
template<typename T>
SCL_FORCE_INLINE
void fill(std::span<T> dest, T value) {
    for (auto& elem : dest) elem = value;
}

// Incorrect (do not use):
// auto fill(std::span<T> dest, T value) -> void { ... }
```

---

## 7. Macro & Attribute Policy

### 7.1 Compiler/Platform/Architecture Detection

- Use `SCL_CONFIG_COMPILER_GCC`, `SCL_CONFIG_COMPILER_CLANG`, `SCL_CONFIG_COMPILER_MSVC`, `SCL_CONFIG_COMPILER_GCC_LIKE` as appropriate.
- Use `SCL_CONFIG_PLATFORM_WINDOWS`, `SCL_CONFIG_PLATFORM_MACOS`, `SCL_CONFIG_PLATFORM_LINUX`.
- Use `SCL_CONFIG_ARCH_X86_64`, `SCL_CONFIG_ARCH_ARM64`, `SCL_CONFIG_ARCH_64BIT`.
- Use `SCL_CONFIG_SIMD_*` macros for SIMD extension detection.

### 7.2 Attributes Macros

- Inlining control: `SCL_FORCE_INLINE`, `SCL_NOINLINE`.
- Aliasing: `SCL_RESTRICT`.
- Optimization hints: `SCL_ASSUME(expr)`, `SCL_PREFETCH(addr, rw, locality)`, `SCL_ALIGNED(n)`.
- Never wrap C++20 attributes in macro aliases (e.g. always use `[[nodiscard]]`, never `SCL_NODISCARD`).

---

## 8. Memory Management

- All memory actions (allocation, free, memcpy, fill) must be implemented via `scl::memory`. Do **not** call `std::` or OS APIs directly.
- RAII aligned buffers and bulk operations must be provided.

---

## 9. Error Handling and Assertions

### Check hierarchy:

| Macro               | Phase      | On Failure    | When to Use                 |
|---------------------|------------|---------------|-----------------------------|
| `static_assert`     | Compile    | Compilation   | Standard C++ checks         |
| `SCL_STATIC_CHECK`  | Compile    | Compilation   | Types/precision validation  |
| `SCL_CHECK`         | Runtime    | Throws        | Argument validation         |
| `SCL_DEBUG_ASSERT`  | Debug only | Terminate     | Internal invariants         |

- Use compile-time checks for types, precision, shapes.
- Use runtime checks for input validation (throwing).
- Use debug assertions for internal invariants only (never in production builds).
- Never use the `assert()` macro in production code.

- Define exception hierarchy: base `Error`, subclass `DimensionError`, `ValueError`, `MemoryError`, `NotImplementedError`.

---

## 10. Operator Config Patterns

- All operator configs must derive from `ConfigBase<Derived>`, which provides `.validate()` (throws) and `.is_valid()` (returns `bool` and is noexcept).
- Define config members and validation logic in `validate_impl()`/`is_valid_impl()`.
- Operators must take a config argument (default-constructed), and must call `.validate()` at entry.

---

## 11. Operator Performance Guidelines

- Optimizations must follow this order:
    1. Algorithm selection (choose lowest-complexity method)
    2. Memory access pattern (ensure cache-friendliness)
    3. SIMD vectorization
    4. Loop unrolling
    5. Branch prediction hints (`[[likely]]` / `[[unlikely]]`)
    6. Prefetching
    7. Register pressure minimization

- For all performance-critical kernels:
    - Apply above optimizations in this order
    - Use platform prefetch/aliasing hints as needed
    - All accessors (like `data()`, `rows()`, `cols()`, operator[]) must have `SCL_FORCE_INLINE`

---

## 12. Documentation Standard (Doxygen)

- All public APIs/structs must have `///` comments with Doxygen tags.
- Minimum tags:  
    - `@brief`        One-line summary  
    - `@tparam`       If templated, explain parameter  
    - `@param[in/out]`    Name, direction  
    - `@return`           Return value meaning  
    - `@throws`           Exceptions it may throw  
    - `@pre`              Preconditions  
    - `@post`             Postconditions  
    - `@note`             Special notes (e.g. complexity, thread-safety)  
    - `@warning`          Pitfalls

- See code sample below for layout.

---

## 13. Boundary and Numeric Precision Handling

- Use `SCL_STATIC_CHECK` for validating numeric types and precisions at compile time.
- Loops must process vectorizable chunks first, tail remainder with scalar code.
- Always handle empty input robustly.
- Use numerically stable strategies (e.g. log-sum-exp with shifting) with explanatory comments.

---

## 14. Type Aliases

- Strongly typedef all core types: e.g. `Index`, `Size`, `Real`, plus explicit span/dimension structs.
- Always use `scl::Index`, `scl::Size`, and fixed-size `Dim2` / `Dim3` / `Dim4` for tensor geometry.

---

## 15. Development Checklist

### Code Quality

- All warnings and errors enabled
- Use C++20 attributes directly
- All memory ops via `scl::memory`
- Use macros from `scl/config.hpp` for env/platform/compiler detection
- Function declaration and attributes on separate lines

### Error Handling

- Compile-time checks for types/precision
- Runtime SCL_CHECK throws for arguments
- Debug assertions for invariants only

### Performance

- Accessors: `SCL_FORCE_INLINE`
- Use `[[likely]]`/`[[unlikely]]` for branching hints
- Prefetch critical loops, unroll where impactful
- Use restrict for non-aliased pointers

### Config

- All configs inherit `ConfigBase`
- Correct validation functions

### Documentation

- Doxygen with all required tags, above each public API and all critical logic

---

## 16. Quick Reference

### Assertion Macros

- `SCL_STATIC_CHECK` — Compile time (types, precision)
- `SCL_CHECK` — Runtime (throws on argument error)
- `SCL_DEBUG_ASSERT` — Debug only (for invariants)

### Env/Platform Macros

- **Compiler:** `SCL_CONFIG_COMPILER_GCC`, `SCL_CONFIG_COMPILER_CLANG`, `SCL_CONFIG_COMPILER_MSVC`, `SCL_CONFIG_COMPILER_GCC_LIKE`
- **Platform:** `SCL_CONFIG_PLATFORM_WINDOWS`, `SCL_CONFIG_PLATFORM_MACOS`, `SCL_CONFIG_PLATFORM_LINUX`
- **Architecture:** `SCL_CONFIG_ARCH_X86_64`, `SCL_CONFIG_ARCH_ARM64`, `SCL_CONFIG_ARCH_64BIT`
- **SIMD:** `SCL_CONFIG_SIMD_AVX512`, `SCL_CONFIG_SIMD_AVX2`, `SCL_CONFIG_SIMD_AVX`, `SCL_CONFIG_SIMD_SSE4_2`, `SCL_CONFIG_SIMD_SSE4_1`, `SCL_CONFIG_SIMD_SSE3`, `SCL_CONFIG_SIMD_SSE2`, `SCL_CONFIG_SIMD_NEON`

### Function Qualifiers

- `SCL_FORCE_INLINE`: Always inline  
- `SCL_NOINLINE`: Never inline  
- `SCL_RESTRICT`: Mark pointer as non-aliasing  
- `SCL_ASSUME(expr)`: Compiler optimization hint  
- `SCL_PREFETCH(addr, rw, locality)`, `SCL_ALIGNED(n)`: Optimization for prefetch and alignment

---

## 17. Unified Function Declaration Template

All critical functions must use this style:

```cpp
/// @brief Adds 1 to the given value.
/// @tparam T The numeric type.
/// @param[in] x Input value.
/// @return x + 1.
/// @note constexpr and force-inlined.
template<typename T>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto add_one(T x) -> T {
    return x + 1;
}
```

- When returning `void`, use `void` directly (never `auto ... -> void`).
- Place each modifier/attribute (`template`, `[[nodiscard]]`, `SCL_FORCE_INLINE`, `constexpr`) on its own line, in that order.

---

