# Vectorized Operations

High-performance SIMD-optimized operations on array views using Google Highway.

## Overview

Vectorized operations provide:

- **Zero-Overhead Abstractions** - All abstractions compile away
- **Automatic SIMD Vectorization** - Architecture-agnostic
- **Aggressive Unrolling** - 2-4 way unrolling for maximum ILP
- **Automatic Tail Handling** - Scalar tail handling for remainder elements

## Design Purpose

Provides zero-overhead abstractions for common array operations with automatic SIMD vectorization. All operations use aggressive unrolling (2-4 way) and handle scalar tails automatically.

## Performance Characteristics

- 4-way unrolled SIMD loops for maximum ILP
- Automatic scalar tail handling
- Architecture-agnostic (AVX2, AVX-512, NEON, etc.)
- Zero runtime overhead (all abstractions compile away)

**Thread Safety:**
- All operations are thread-safe (pure functions, no shared state)

## Reduction Operations

### sum

Compute sum of all elements:

```cpp
#include "scl/core/vectorize.hpp"

Array<const Real> span = /* ... */;
Real total = scl::vectorize::sum(span);
```

**Returns:**
- Sum of all elements. Returns T(0) for empty arrays.

**Algorithm:**
1. 4-way unrolled SIMD accumulation
2. Horizontal reduction using SumOfLanes
3. Scalar tail handling

**Complexity:**
- Time: O(N)
- Space: O(1)

**Performance:**
- Approximately 4-8x faster than naive loop on modern CPUs

### product

Compute product of all elements:

```cpp
Real prod = scl::vectorize::product(span);
```

**Returns:**
- Product of all elements. Returns T(1) for empty arrays.

**Performance:**
- Uses 2-way unrolling for product accumulation

### dot

Compute dot product of two vectors:

```cpp
Array<const Real> a = /* ... */;
Array<const Real> b = /* ... */;

Real result = scl::vectorize::dot(a, b);
```

**PRECONDITIONS:**
- `a.len == b.len`

**Returns:**
- Dot product: sum(a[i] * b[i])

**Performance:**
- Uses MulAdd (FMA) for optimal performance

## Search Operations

### find

Find first occurrence of value:

```cpp
size_t idx = scl::vectorize::find(span, value);
```

**Returns:**
- Index of first match, or span.len if not found

**Performance:**
- Uses SIMD comparison with early exit on match

### count

Count occurrences of value:

```cpp
size_t num = scl::vectorize::count(span, value);
```

**Returns:**
- Number of elements equal to value

**Performance:**
- Uses CountTrue on comparison mask for efficient counting

### contains

Check if array contains value:

```cpp
bool found = scl::vectorize::contains(span, value);
```

**Returns:**
- True if value found, false otherwise

## Min/Max Operations

### min_element

Find index of minimum element:

```cpp
size_t idx = scl::vectorize::min_element(span);
```

**PRECONDITIONS:**
- `span.len > 0`

**Returns:**
- Index of minimum element

**Performance:**
- Uses MinOfLanes for horizontal reduction

### max_element

Find index of maximum element:

```cpp
size_t idx = scl::vectorize::max_element(span);
```

### minmax

Find both minimum and maximum in single pass:

```cpp
auto [min_val, max_val] = scl::vectorize::minmax(span);
```

**Returns:**
- Pair of (min_value, max_value)

**Performance:**
- More efficient than calling min_element and max_element separately

## Transform Operations

### transform_inplace

Apply unary operation in-place:

```cpp
Array<Real> span = /* ... */;
scl::vectorize::transform_inplace(span, [](Real x) { return x * 2.0; });
```

**Parameters:**
- `span` - Array to transform
- `op` - Unary operation: T -> T

**MUTABILITY:**
- INPLACE - modifies span

### transform (unary)

Apply unary operation: dst[i] = op(src[i]):

```cpp
Array<const Real> src = /* ... */;
Array<Real> dst = /* ... */;

scl::vectorize::transform(src, dst, [](Real x) { return x * 2.0; });
```

**PRECONDITIONS:**
- `src.len == dst.len`

### transform (binary)

Apply binary operation: dst[i] = op(a[i], b[i]):

```cpp
scl::vectorize::transform(a, b, dst, [](Real x, Real y) { return x + y; });
```

**PRECONDITIONS:**
- `a.len == b.len == dst.len`

### scale

Scale all elements: dst[i] = src[i] * scale_factor:

```cpp
Real factor = 2.0;
scl::vectorize::scale(src, dst, factor);
```

**Performance:**
- Uses 4-way unrolled SIMD multiplication

### scale_inplace

Scale in-place: span[i] *= scale_factor:

```cpp
scl::vectorize::scale_inplace(span, 2.0);
```

### add_scalar

Add scalar to all elements: dst[i] = src[i] + value:

```cpp
Real offset = 1.0;
scl::vectorize::add_scalar(src, dst, offset);
```

### add

Element-wise addition: dst[i] = a[i] + b[i]:

```cpp
scl::vectorize::add(a, b, dst);
```

### sub

Element-wise subtraction: dst[i] = a[i] - b[i]:

```cpp
scl::vectorize::sub(a, b, dst);
```

## Norm Operations

### norm_l2

Compute L2 norm (Euclidean norm):

```cpp
Real nrm = scl::vectorize::norm_l2(span);
```

**Returns:**
- sqrt(sum(span[i]^2))

**Performance:**
- Uses sum_squared + sqrt (optimized)

### norm_l2_squared

Compute squared L2 norm:

```cpp
Real nrm_sq = scl::vectorize::norm_l2_squared(span);
```

**Returns:**
- sum(span[i]^2) (no sqrt, faster)

### norm_l1

Compute L1 norm (Manhattan norm):

```cpp
Real nrm = scl::vectorize::norm_l1(span);
```

**Returns:**
- sum(|span[i]|)

---

::: tip Zero-Overhead Abstraction
All vectorized operations compile to optimal machine code with no abstraction overhead. Use them liberally for performance-critical code.
:::

