---
title: Core Types
description: Type system, Array views, and fundamental types
---

# Core Types

SCL-Core provides a unified type system designed for zero-overhead performance and maximum flexibility.

## Fundamental Types

### Real

The floating-point type used throughout SCL-Core. Configurable at compile time:

```cpp
namespace scl {
    using Real = float;   // if SCL_PRECISION == 0 (default)
    using Real = double;  // if SCL_PRECISION == 1
    using Real = _Float16; // if SCL_PRECISION == 2
}
```

**Configuration**:
- `SCL_PRECISION=0`: `float32` (default, 4 bytes)
- `SCL_PRECISION=1`: `float64` (8 bytes, higher precision)
- `SCL_PRECISION=2`: `float16` (2 bytes, requires GCC ≥12 or Clang ≥15)

**Usage**:
```cpp
Real value = 3.14159;
Real* data = ...;
Array<Real> view = {data, n};
```

### Index

The integer index type for array indices and matrix dimensions:

```cpp
namespace scl {
    using Index = std::int16_t;  // if SCL_INDEX_PRECISION == 0
    using Index = std::int32_t;  // if SCL_INDEX_PRECISION == 1
    using Index = std::int64_t;  // if SCL_INDEX_PRECISION == 2 (default)
}
```

**Configuration**:
- `SCL_INDEX_PRECISION=0`: `int16` (max 32K elements, minimal memory)
- `SCL_INDEX_PRECISION=1`: `int32` (max 2B elements, standard)
- `SCL_INDEX_PRECISION=2`: `int64` (max 9E18 elements, NumPy-compatible, default)

**Usage**:
```cpp
Index n_rows = 1000;
Index idx = 42;
for (Index i = 0; i < n_rows; ++i) {
    // ...
}
```

### Size

Unsigned size type for array lengths and counts:

```cpp
namespace scl {
    using Size = std::size_t;
}
```

**Usage**:
```cpp
Size n = 1000;
Array<Real> data = {ptr, n};
```

## Array View

`Array<T>` is a zero-overhead view type similar to `std::span`, providing safe array access without ownership:

```cpp
template <typename T>
struct Array {
    T* ptr;
    Size len;
    
    // Constructors
    constexpr Array() noexcept;
    constexpr Array(T* p, Size s) noexcept;
    
    // Element access
    constexpr T& operator[](Index i) const noexcept;
    constexpr T& front() const noexcept;
    constexpr T& back() const noexcept;
    
    // Capacity
    constexpr T* data() const noexcept;
    constexpr Size size() const noexcept;
    constexpr bool empty() const noexcept;
    
    // Iterators
    constexpr T* begin() const noexcept;
    constexpr T* end() const noexcept;
    
    // Subviews
    constexpr Array<T> subspan(Index offset, Size count) const noexcept;
    constexpr Array<T> first(Size count) const noexcept;
    constexpr Array<T> last(Size count) const noexcept;
};
```

### Properties

- **Zero Overhead**: Compiles to raw pointer + size
- **No Ownership**: Does not manage memory lifetime
- **Bounds Checking**: Optional in debug builds (`NDEBUG` not defined)
- **STL Compatible**: Provides iterators and standard container interface

### Usage Examples

```cpp
// Create from raw pointer
Real* data = ...;
Size n = 1000;
Array<Real> view = {data, n};

// Access elements
Real value = view[0];
Real first = view.front();
Real last = view.back();

// Iterate
for (Real& val : view) {
    val *= 2.0;
}

// Subviews
Array<Real> first_half = view.first(n / 2);
Array<Real> second_half = view.last(n / 2);
Array<Real> middle = view.subspan(100, 200);

// Convert to std::span
std::span<Real> span = view.as_span();
```

### Const Correctness

```cpp
// Mutable view
Array<Real> mutable_view = {data, n};
mutable_view[0] = 1.0;  // OK

// Const view
Array<const Real> const_view = {data, n};
// const_view[0] = 1.0;  // Error: cannot modify

// Conversion from mutable to const
Array<const Real> const_from_mutable = mutable_view;  // OK
```

### ArrayLike Concept

SCL-Core defines an `ArrayLike` concept for generic programming:

```cpp
template <typename A>
concept ArrayLike = requires(const A& a, Index i) {
    typename A::value_type;
    { a.size() } -> std::convertible_to<Size>;
    { a[i] } -> std::convertible_to<const typename A::value_type&>;
    { a.begin() };
    { a.end() };
};
```

**Usage**:
```cpp
template <ArrayLike Container>
void process(Container& data) {
    for (auto& val : data) {
        // Process value
    }
}

// Works with Array, std::vector, std::span, etc.
Array<Real> arr = {ptr, n};
std::vector<Real> vec(n);
process(arr);  // OK
process(vec);  // OK
```

## Type Traits and Utilities

### Compile-Time Constants

```cpp
namespace scl {
    // Precision codes
    constexpr int DTYPE_CODE = ...;        // 0, 1, or 2
    constexpr const char* DTYPE_NAME = ...; // "float32", "float64", or "float16"
    
    // Index precision codes
    constexpr int INDEX_DTYPE_CODE = ...;        // 0, 1, or 2
    constexpr const char* INDEX_DTYPE_NAME = ...; // "int16", "int32", or "int64"
}
```

### Type Checking

```cpp
// Check if type is Real
static_assert(std::is_same_v<T, Real>);

// Check if type is Index
static_assert(std::is_same_v<T, Index>);

// Check if Array is POD (trivially copyable)
static_assert(std::is_trivially_copyable_v<Array<Real>>);
static_assert(std::is_standard_layout_v<Array<Real>>);
```

## Best Practices

### 1. Prefer Array Views Over Raw Pointers

```cpp
// Good: Type-safe, bounds-checked
void process(Array<Real> data) {
    for (Size i = 0; i < data.size(); ++i) {
        data[i] *= 2.0;
    }
}

// Avoid: Raw pointers
void process(Real* data, Size n) {  // Less safe
    // ...
}
```

### 2. Use Const Views When Possible

```cpp
// Good: Clear intent, prevents accidental modification
void compute_sum(Array<const Real> data) {
    Real sum = 0;
    for (auto val : data) {
        sum += val;
    }
    return sum;
}
```

### 3. Leverage Subviews for Partial Processing

```cpp
// Process in chunks
void process_chunks(Array<Real> data, Size chunk_size) {
    for (Size offset = 0; offset < data.size(); offset += chunk_size) {
        Size remaining = data.size() - offset;
        Size current_chunk = std::min(chunk_size, remaining);
        Array<Real> chunk = data.subspan(offset, current_chunk);
        process_chunk(chunk);
    }
}
```

### 4. Combine with STL Algorithms

```cpp
Array<Real> data = {ptr, n};

// Use with STL algorithms
std::sort(data.begin(), data.end());
auto it = std::find(data.begin(), data.end(), target);
Real sum = std::accumulate(data.begin(), data.end(), Real(0));
```

## Performance Considerations

### Zero Overhead

`Array<T>` has zero runtime overhead:

```cpp
// This code:
Array<Real> view = {ptr, n};
Real val = view[i];

// Compiles to (with optimizations):
Real val = ptr[i];
```

### Cache-Friendly Access

```cpp
// Sequential access is optimal
for (Size i = 0; i < data.size(); ++i) {
    process(data[i]);
}

// Random access may cause cache misses
for (Index idx : random_indices) {
    process(data[idx]);  // May be slower
}
```

### Alignment

For SIMD operations, ensure data is aligned:

```cpp
// Aligned allocation
auto buffer = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
Array<Real> view = {buffer.get(), n};

// Use with SIMD
namespace s = scl::simd;
const s::SimdTag d;
auto v = s::Load(d, view.data());  // Requires alignment
```

## Related Documentation

- [Sparse Matrices](./sparse.md) - Sparse matrix types
- [Memory Management](./memory.md) - Allocation and lifetime management
- [SIMD](./simd.md) - Vectorization support
