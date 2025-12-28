# Core Modules

The `scl/core/` directory contains the foundation of SCL-Core: fundamental types, data structures, and utilities that all other modules depend on.

## Overview

Core modules provide:

- **Type System** - Configurable precision and index types
- **Sparse Matrices** - High-performance discontiguous storage
- **SIMD Abstraction** - Portable vectorization via Highway
- **Memory Management** - Aligned allocation and Registry tracking
- **Error Handling** - Assertions and exceptions
- **Vectorization** - Common vectorized operations

## Files

| File | Description | Main APIs |
|------|-------------|-----------|
| [type.hpp](./types) | Type system | Real, Index, Size, Array, Span |
| [sparse.hpp](./sparse) | Sparse matrix | Sparse class, CSR/CSC format |
| [registry.hpp](./registry) | Memory registry | Registry, BufferID, reference counting |
| [simd.hpp](./simd) | SIMD abstraction | Tag, Vec, SIMD operations |
| [memory.hpp](./memory) | Memory management | aligned_alloc, aligned_free |
| [vectorize.hpp](./vectorize) | Vectorized ops | dot, norm, sum, vectorized functions |
| [sort.hpp](./sort) | Sorting | sort, sort_key_value |
| [argsort.hpp](./argsort) | Argument sorting | argsort_inplace, argsort_indirect |
| [error.hpp](./error) | Error handling | SCL_ASSERT, SCL_CHECK_*, exceptions |
| [macros.hpp](./macros) | Macros | Platform detection, optimization hints |

## Dependency Graph

```
┌─────────────────────────────────────────┐
│          Other Modules                  │
│    (threading, kernel, math, etc.)      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          sparse.hpp                     │
│     (Sparse matrix operations)          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    registry.hpp + memory.hpp            │
│  (Memory management and tracking)       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    simd.hpp + vectorize.hpp             │
│      (SIMD and vectorization)           │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      error.hpp + macros.hpp             │
│     (Error handling and macros)         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           type.hpp                      │
│       (Fundamental types)               │
└─────────────────────────────────────────┘
```

## Quick Start

### Basic Types

```cpp
#include "scl/core/type.hpp"

using namespace scl;

// Configurable precision
Real x = 3.14;  // float, double, or _Float16

// Signed indexing
Index i = -1;   // int16_t, int32_t, or int64_t

// Sizes and byte counts
Size n = 1000;  // size_t
```

### Sparse Matrix

```cpp
#include "scl/core/sparse.hpp"

// Create CSR matrix
auto matrix = scl::Sparse<Real, true>::create(
    rows, cols, nnz_per_row);

// Access row
auto vals = matrix.primary_values(i);
auto idxs = matrix.primary_indices(i);
Index len = matrix.primary_length(i);

// Iterate
for (Index j = 0; j < len; ++j) {
    Real value = vals.ptr[j];
    Index col = idxs.ptr[j];
}
```

### SIMD Operations

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;
const s::Tag d;

// Load and process
auto v = s::Load(d, data);
auto result = s::Mul(v, s::Set(d, 2.0));
s::Store(result, d, output);
```

### Memory Management

```cpp
#include "scl/core/registry.hpp"
#include "scl/core/memory.hpp"

// Aligned allocation
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// Register for tracking
auto& reg = scl::get_registry();
reg.register_ptr(data, 1000 * sizeof(Real), 
                 AllocType::AlignedAlloc);

// Cleanup
reg.unregister_ptr(data);
```

## Design Principles

### 1. Zero Dependencies

Core modules depend only on:
- C++17 standard library
- Google Highway (for SIMD)

No other external dependencies.

### 2. Header-Only When Possible

Most core utilities are header-only for:
- Easy integration
- Better inlining
- Reduced link time

### 3. Compile-Time Configuration

Types are configured at compile time:

```cpp
// In CMakeLists.txt or config.hpp
#define SCL_USE_FLOAT32  // or FLOAT64, FLOAT16
#define SCL_USE_INT32    // or INT16, INT64
```

This enables:
- Single codebase for multiple precisions
- Zero runtime overhead for type selection
- Optimal code generation

### 4. Explicit Resource Management

No hidden allocations or implicit costs:

```cpp
// BAD: Hidden allocation
std::vector<Real> temp;  // Allocates!

// GOOD: Explicit allocation
Real* temp = reg.new_array<Real>(n);
// ... use temp ...
reg.unregister_ptr(temp);
```

## Common Patterns

### RAII for Cleanup

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

// Usage
auto* data = reg.new_array<Real>(1000);
RegistryGuard guard(data);
// Automatic cleanup on scope exit
```

### Template Constraints

```cpp
// Concept for CSR-like types
template <typename T>
concept CSRLike = requires(T t, Index i) {
    { t.rows() } -> std::convertible_to<Index>;
    { t.cols() } -> std::convertible_to<Index>;
    { t.primary_values(i) };
    { t.primary_indices(i) };
};

// Use in function
template <CSRLike MatrixT>
void process(const MatrixT& matrix) {
    // Works with any CSR-like type
}
```

### Error Handling

```cpp
// Debug assertions (compiled out in release)
SCL_ASSERT(i >= 0 && i < n, "Index out of bounds");

// Runtime checks (always enabled)
SCL_CHECK_ARG(data != nullptr, "Null pointer");
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
```

## Performance Tips

### 1. Use SIMD for Bulk Operations

```cpp
// Scalar loop
for (size_t i = 0; i < n; ++i) {
    output[i] = input[i] * 2.0;
}

// SIMD loop (2-4x faster)
namespace s = scl::simd;
const s::Tag d;
const size_t lanes = s::Lanes(d);

for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, input + i);
    auto result = s::Mul(v, s::Set(d, 2.0));
    s::Store(result, d, output + i);
}
```

### 2. Align Memory for SIMD

```cpp
// Aligned allocation for SIMD
Real* data = scl::memory::aligned_alloc<Real>(n, 64);

// Faster SIMD loads/stores
auto v = s::Load(d, data);  // Aligned load
```

### 3. Minimize Registry Lookups

```cpp
// BAD: Lookup in hot loop
for (size_t i = 0; i < n; ++i) {
    if (reg.is_registered(ptr)) {  // Expensive!
        // ...
    }
}

// GOOD: Check once
bool is_reg = reg.is_registered(ptr);
for (size_t i = 0; i < n; ++i) {
    if (is_reg) {
        // ...
    }
}
```

## Next Steps

Explore each core module in detail:

- [Types](/cpp/core/types) - Type system and configuration
- [Sparse Matrix](/cpp/core/sparse) - Sparse matrix infrastructure
- [Registry](/cpp/core/registry) - Memory lifetime tracking
- [SIMD](/cpp/core/simd) - SIMD abstraction
- [Memory](/cpp/core/memory) - Aligned allocation
- [Vectorize](/cpp/core/vectorize) - Vectorized operations
- [Sort](/cpp/core/sort) - High-performance sorting
- [Argsort](/cpp/core/argsort) - Argument sorting (indices)
- [Error Handling](/cpp/core/error) - Assertions and exceptions
- [Macros](/cpp/core/macros) - Compiler macros and platform detection

---

::: tip Foundation First
Understanding the core modules is essential for working with SCL-Core. Start here before exploring higher-level modules.
:::

