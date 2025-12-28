# AGENT.md - AI Developer Guide for scl-core

This document defines the coding standards and protocols for the `scl-core` project.

**Core Mission**: Build a high-performance biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.

---

## 1. Documentation Standard

### 1.1 Documentation Location

- `xxx.hpp`: Implementation files with minimal inline comments
- `docs/`: Comprehensive API documentation in Markdown format

### 1.2 Implementation Files

Keep inline comments minimal:
- One-line function purpose (optional if self-explanatory)
- Non-obvious algorithm tricks or optimizations
- Warnings about subtle behavior

### 1.3 API Documentation Format

All API documentation in `docs/` must include these sections (in order):

| Section | Required | Description |
|---------|----------|-------------|
| SUMMARY | Yes | One-line description |
| PARAMETERS | Yes | Each parameter with [in], [out], or [in,out] |
| PRECONDITIONS | Yes | Requirements before calling |
| POSTCONDITIONS | Yes | Guarantees after execution |
| MUTABILITY | If applicable | INPLACE, CONST, or ALLOCATES |
| ALGORITHM | If non-trivial | Step-by-step description |
| COMPLEXITY | Yes | Time and space complexity |
| THREAD SAFETY | Yes | Safe, Unsafe, or conditional |
| THROWS | If applicable | Exceptions and conditions |
| NUMERICAL NOTES | If applicable | Precision, stability, edge cases |

Function signatures in documentation must match implementation exactly. Use code blocks for signatures with inline parameter comments.

**IMPORTANT**: API documentation must NOT contain examples. Describe contracts and behavior precisely.

### 1.4 Workflow Requirement

**CRITICAL**: After any file modification, update corresponding documentation in `docs/`.

Checklist:
1. Implementation in `.hpp` is correct
2. Documentation exists in `docs/` directory
3. Function signatures match exactly
4. All required sections are present
5. MUTABILITY is correct for in-place operations

### 1.5 Language and Formatting

- **Language**: English only
- **Format**: Plain text only (NO Markdown/LaTeX syntax in code comments)
- **No examples** in API documentation

---

## 2. C++ Standards

### 2.1 Language Standard

- **C++20** (ISO/IEC 14882:2020)
- **Compilers**: GCC 11+, Clang 14+, MSVC 19.29+

**Compile Flags**:
```bash
# Release: -std=c++20 -O3 -DNDEBUG -march=native -ffast-math -flto
# Debug:   -std=c++20 -O0 -g -fsanitize=address,undefined
```

### 2.2 Required C++20 Features

- Concepts (prefer over SFINAE)
- std::span (replace raw pointer + size)
- constexpr (maximize compile-time computation)
- [[likely]]/[[unlikely]] (hot paths)
- std::bit_cast (type punning for POD)
- noexcept (especially move operations)

### 2.3 Code Style

**Namespaces**:
- `scl::core`: Core types and utilities
- `scl::kernel`: Computational kernels
- `scl::math`: Mathematical functions
- `scl::threading`: Parallelization layer
- `scl::binding`: C-ABI interface

**Formatting**:
- `#pragma once` for header guards
- Section headers: `// =============================================================================`
- Indentation: 4 spaces
- Line length: < 100 chars (max 120)
- Opening brace on same line

**Attributes**:
- `SCL_FORCE_INLINE` for hot-path functions
- `SCL_NODISCARD` for return values that must be checked
- `SCL_RESTRICT` for non-aliased pointers in hot loops
- `constexpr` for compile-time constants

**Error Handling**:
- `SCL_ASSERT` for internal invariants (debug-only)
- `SCL_CHECK_ARG` for user input validation
- `SCL_CHECK_DIM` for dimension mismatches
- Prefer error codes over exceptions in hot paths

---

## 3. Clang-Tidy Compliance

### 3.1 Configuration

Project uses `.clang-tidy` at repository root. Key disabled checks:
- `modernize-use-trailing-return-type`
- `readability-identifier-length`
- `readability-magic-numbers`
- `cppcoreguidelines-pro-bounds-pointer-arithmetic`
- `cppcoreguidelines-owning-memory`

### 3.2 Mandatory Checks

Never disable without justification:
- `bugprone-*`
- `clang-analyzer-*`
- `cert-*`
- `cppcoreguidelines-init-variables`
- `cppcoreguidelines-slicing`
- `modernize-use-nullptr`
- `modernize-use-override`
- `performance-move-const-arg`

---

## 4. Performance Exceptions (NOLINT)

When performance conflicts with clang-tidy rules, document exceptions.

### 4.1 NOLINT Format

```cpp
// PERFORMANCE: [brief reason]
// [detailed explanation if needed]
// NOLINTNEXTLINE(check-name)
code_line;
```

### 4.2 Common Exceptions

**Allowed with documentation**:
- Pointer arithmetic in hot loops (with benchmark)
- Reinterpret cast for SIMD (with SIMD target and speedup)
- Magic numbers in algorithms (with mathematical reference)
- Uninitialized variables (with justification)
- Raw loops over ranges (with benchmark)
- Owning raw pointers in C-ABI layer (with ownership model)
- Short variable names in math code (with algorithm reference)

### 4.3 Prohibited Suppressions

Never suppress without human review:
- `bugprone-use-after-move`
- `clang-analyzer-core.NullDereference`
- `clang-analyzer-core.UndefinedBinaryOperatorResult`
- `cppcoreguidelines-slicing`
- `bugprone-undefined-memory-manipulation`
- `cert-err58-cpp`

### 4.4 Performance Documentation Template

```cpp
// =============================================================================
// PERFORMANCE EXCEPTION: [Short Title]
// =============================================================================
// Rule Suppressed: [check-name]
// Reason: [Why this rule cannot be followed]
// Alternative Considered: [What standard-compliant approach was tried]
// Benchmark: [Performance comparison]
// Safety: [Why this is still safe]
// =============================================================================
```

---

## 5. Modern C++ Patterns

### 5.1 RAII

Use RAII wrappers for all resources. Mark move operations `noexcept`.

### 5.2 Strong Types

Use strong types to prevent parameter confusion:
```cpp
struct RowIndex { Index value; };
struct ColIndex { Index value; };
```

### 5.3 constexpr

Maximize compile-time computation. Use `consteval` for lookup tables.

### 5.4 Structured Bindings

Use structured bindings for tuple-like returns. Use designated initializers for clarity.

---

## 6. Development Checklist

### Code Quality
- [ ] Compiles with `-std=c++20 -Wall -Wextra -Werror`
- [ ] Passes `clang-tidy` with project configuration
- [ ] All NOLINT suppressions documented
- [ ] Performance exceptions include benchmark data
- [ ] Modern C++20 features used appropriately

### Documentation
- [ ] Minimal inline comments in implementation
- [ ] Documentation exists in `docs/`
- [ ] Function signatures match exactly
- [ ] All required sections present
- [ ] MUTABILITY specified for state-changing functions
- [ ] No examples in API documentation

### Testing
- [ ] Unit tests cover all public functions
- [ ] Edge cases tested
- [ ] Thread safety tests for parallel functions
- [ ] Sanitizer builds pass (ASan, UBSan, TSan)

---

## 7. Quick Reference

### 7.1 Macros

```cpp
#define SCL_FORCE_INLINE [[gnu::always_inline]] inline
#define SCL_NODISCARD [[nodiscard]]
#define SCL_RESTRICT __restrict__
#define SCL_LIKELY(x) [[likely]] (x)
#define SCL_UNLIKELY(x) [[unlikely]] (x)
#define SCL_ASSERT(cond) assert(cond)  // debug-only
#define SCL_CHECK_ARG(cond, msg) /* throws std::invalid_argument */
#define SCL_CHECK_DIM(cond, msg) /* throws DimensionError */
```

### 7.2 Type Aliases

```cpp
namespace scl {
    using Index = std::int64_t;
    using Real = double;
    template <typename T> using Span = std::span<T>;
    template <typename T> using MutableSpan = std::span<T>;
    template <typename T> using ConstSpan = std::span<const T>;
}
```

### 7.3 Clang-Tidy Quick Reference

| Check | Default | Notes |
|-------|---------|-------|
| `cppcoreguidelines-pro-bounds-pointer-arithmetic` | OFF | Allow with NOLINT |
| `readability-magic-numbers` | OFF | Allow with documentation |
| `readability-identifier-length` | OFF | Allow math notation |
| `cppcoreguidelines-owning-memory` | OFF | Allow in C-ABI layer |
| `bugprone-*` | ON | Never disable |
| `clang-analyzer-*` | ON | Never disable |
