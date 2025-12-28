# Memory Primitives

Low-level memory operations optimized for high-performance computing with SIMD acceleration.

## Overview

Memory primitives provide:

- **Aligned Allocation** - SIMD-optimized memory allocation
- **Initialization** - Efficient fill and zero operations
- **Data Movement** - Fast copy operations with overlap handling
- **Prefetch Utilities** - Cache optimization hints
- **Memory Comparison** - Fast equality and comparison checks
- **Swap Operations** - Element swapping utilities

## Aligned Memory Allocation

### aligned_alloc

Allocate aligned memory for primitive types:

```cpp
#include "scl/core/memory.hpp"

Real* data = scl::memory::aligned_alloc<Real>(
    1000,      // Number of elements
    64         // Alignment in bytes (default: 64)
);

// Use data...

// Free with aligned_free
scl::memory::aligned_free(data, 64);
```

**Parameters:**
- `T` [template] - Element type (must be trivially constructible)
- `count` - Number of elements to allocate
- `alignment` - Alignment in bytes (default: 64)

**Returns:**
- Aligned pointer to allocated memory, or nullptr on failure
- Memory is zero-initialized

**PRECONDITIONS:**
- `alignment` must be a power of 2
- `alignment >= sizeof(void*)`
- `count * sizeof(T)` must not overflow

**POSTCONDITIONS:**
- If count == 0: returns nullptr
- If allocation succeeds: returns aligned pointer to zero-initialized memory
- If allocation fails: returns nullptr

**Complexity:**
- Time: O(1)
- Space: O(count * sizeof(T))

**Thread Safety:**
- Safe - each call allocates independent memory

**Platform Support:**
- C++17+: Uses aligned operator new
- POSIX: Uses posix_memalign
- Windows: Uses _aligned_malloc

**IMPORTANT:**
- Memory MUST be freed with `aligned_free`, NOT `free()` or `delete[]`
- Alignment value must match in `aligned_free`

**Use Cases:**
- SIMD workspaces (require 16/32/64-byte alignment)
- Cache-line optimization (64-byte alignment)
- Large temporary buffers in kernels

### aligned_free

Free memory allocated with `aligned_alloc`:

```cpp
scl::memory::aligned_free(data, 64);
```

**Parameters:**
- `T` [template] - Element type
- `ptr` - Pointer to free (nullptr is safe)
- `alignment` - Alignment value used in allocation (default: 64)

**PRECONDITIONS:**
- `ptr` must be nullptr or allocated with `aligned_alloc`
- `alignment` must match the alignment used in `aligned_alloc`

**WARNING:**
- Using regular `free()` or `delete[]` on aligned memory may crash
- Alignment value must match the value used in `aligned_alloc`

### AlignedBuffer

RAII wrapper for aligned memory:

```cpp
{
    scl::memory::AlignedBuffer<Real> buffer(1000, 64);
    
    if (buffer) {  // Check allocation success
        Real* data = buffer.get();
        // Use data...
        // Automatic cleanup on scope exit
    }
}
```

**Methods:**
- `T* get()` - Returns raw pointer to buffer
- `size_t size()` - Returns number of elements
- `T& operator[](size_t i)` - Access element at index i (no bounds checking)
- `Array<T> span()` - Returns Array view of buffer
- `explicit operator bool()` - Checks if buffer allocation succeeded

**MOVABLE:** Yes - supports move construction and assignment
**COPYABLE:** No - copy operations are deleted

**Use Case:**
- Automatic cleanup of temporary buffers in exception-safe code

## Initialization

### fill

Fill memory with a value using SIMD acceleration:

```cpp
Array<Real> span = /* ... */;
scl::memory::fill(span, Real(1.5));
```

**Parameters:**
- `T` [template] - Element type
- `span` - Memory range to fill
- `value` - Value to fill with

**MUTABILITY:**
- INPLACE - modifies span contents

**Algorithm:**
1. Create SIMD vector with broadcasted value
2. 4-way unrolled SIMD loop for bulk of data
3. Remainder SIMD loop
4. Scalar loop for tail elements

**Complexity:**
- Time: O(n / lanes) where lanes is SIMD width
- Space: O(1) auxiliary

**Performance:**
- Automatically uses best SIMD instructions (AVX2/AVX-512/NEON)
- 4-way unrolling for better instruction throughput
- Optimal for arrays larger than ~64 elements

### zero

Zero out memory efficiently:

```cpp
Array<Real> span = /* ... */;
scl::memory::zero(span);
```

**Parameters:**
- `T` [template] - Element type
- `span` - Memory range to zero

**MUTABILITY:**
- INPLACE - modifies span contents

**Performance:**
- Specialized fast path using SIMD zero vectors
- Faster than `fill(span, T(0))` for zero initialization

## Data Movement

### copy

Copy memory with overlap handling (memmove-like):

```cpp
Array<const Real> src = /* ... */;
Array<Real> dst = /* ... */;

scl::memory::copy(src, dst);
```

**Parameters:**
- `T` [template] - Element type
- `src` - Source memory range
- `dst` - Destination memory range

**PRECONDITIONS:**
- `src.len == dst.len`

**POSTCONDITIONS:**
- `dst` contains copy of `src`
- Handles overlapping regions correctly

**Safety Level:**
- **Safe** - Handles overlaps (memmove behavior)

**Algorithm:**
- For overlapping regions: reverse copy if dst < src
- For non-overlapping: uses fast copy path
- SIMD-optimized for large arrays

### copy_fast

Fast copy assuming NO overlap (memcpy-like):

```cpp
scl::memory::copy_fast(src, dst);
```

**Safety Level:**
- **Fast** - Assumes NO overlap (undefined behavior if violated)
- Faster than `copy` when overlap is impossible

**WARNING:**
- Undefined behavior if src and dst overlap

### copy_stream

Non-temporal copy (bypasses cache):

```cpp
scl::memory::copy_stream(src, dst);
```

**Safety Level:**
- **Stream** - Bypasses cache (non-temporal), assumes NO overlap

**Use Cases:**
- Large buffers that won't be reused soon
- One-time data movement to avoid cache pollution

## Prefetch Utilities

### prefetch_read

Prefetch memory for reading:

```cpp
scl::memory::prefetch_read(ptr, distance = 0);
```

**Parameters:**
- `ptr` - Memory address to prefetch
- `distance` - Prefetch distance (0 = immediate, higher = further ahead)

**Use Cases:**
- Prefetch next iteration's data in loops
- Reduce memory latency in hot paths

## Memory Comparison

### equal

Check if two memory ranges are equal:

```cpp
Array<const Real> a = /* ... */;
Array<const Real> b = /* ... */;

bool is_equal = scl::memory::equal(a, b);
```

**Parameters:**
- `T` [template] - Element type
- `a` - First memory range
- `b` - Second memory range

**Returns:**
- True if all elements are equal, false otherwise

**PRECONDITIONS:**
- `a.len == b.len`

**Performance:**
- SIMD-optimized comparison
- Early exit on first mismatch

## Swap Operations

### swap

Swap two elements:

```cpp
Real a = 1.0, b = 2.0;
scl::memory::swap(a, b);
// a == 2.0, b == 1.0
```

**Parameters:**
- `T` [template] - Element type
- `a` - First element (modified)
- `b` - Second element (modified)

**MUTABILITY:**
- INPLACE - modifies both arguments

---

::: tip Alignment Requirements
For optimal SIMD performance, allocate buffers with 64-byte alignment (matches AVX-512 cache line size). Always use `aligned_free` to match `aligned_alloc`.
:::

