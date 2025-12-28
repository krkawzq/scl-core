# memory.hpp

> scl/core/memory.hpp · Low-level memory operations optimized for high-performance computing

## Overview

This file provides low-level memory operations optimized for high-performance computing with SIMD acceleration. It includes aligned memory allocation, initialization (fill/zero), data movement (copy), and cache optimization utilities.

Key features:
- Aligned memory allocation for SIMD operations
- SIMD-accelerated fill and zero operations
- Fast copy operations with overlap handling
- RAII wrapper for aligned buffers
- Cache optimization hints (prefetch)

**Header**: `#include "scl/core/memory.hpp"`

---

## Main APIs

### aligned_alloc

Allocate aligned memory for primitive types.

::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc" collapsed
:::

**Algorithm Description**

Allocates memory aligned to a specified boundary (default 64 bytes for cache-line alignment):

1. Check if count == 0, return nullptr if so
2. Calculate total bytes needed: count * sizeof(T)
3. Use platform-specific aligned allocation:
   - C++17+: aligned operator new
   - POSIX: posix_memalign
   - Windows: _aligned_malloc
4. Zero-initialize the allocated memory
5. Return aligned pointer, or nullptr on failure

The default alignment of 64 bytes matches AVX-512 cache line size and ensures optimal SIMD performance.

**Edge Cases**

- **count == 0**: Returns nullptr (no allocation)
- **Allocation failure**: Returns nullptr (caller must check)
- **Overflow**: count * sizeof(T) overflow checked, returns nullptr
- **Invalid alignment**: Must be power of 2 and >= sizeof(void*)

**Data Guarantees (Preconditions)**

- `alignment` must be a power of 2
- `alignment >= sizeof(void*)` (platform minimum)
- `count * sizeof(T)` must not overflow
- T must be trivially constructible

**Complexity Analysis**

- **Time**: O(1) - single allocation call
- **Space**: O(count * sizeof(T)) - allocated memory

**Example**

```cpp
#include "scl/core/memory.hpp"

// Allocate aligned buffer for SIMD operations
Real* data = scl::memory::aligned_alloc<Real>(
    1000,    // Number of elements
    64       // Alignment in bytes (default: 64)
);

if (data == nullptr) {
    // Handle allocation failure
    return;
}

// Use data for SIMD operations...
// Memory is zero-initialized

// Free with aligned_free (MUST use aligned_free, not free())
scl::memory::aligned_free(data, 64);
```

---

### aligned_free

Free memory allocated with aligned_alloc.

::: source_code file="scl/core/memory.hpp" symbol="aligned_free" collapsed
:::

**Algorithm Description**

Frees memory allocated by aligned_alloc using platform-specific deallocation:
- C++17+: aligned operator delete
- POSIX: free
- Windows: _aligned_free

The alignment parameter must match the value used in aligned_alloc.

**Edge Cases**

- **ptr == nullptr**: Safe to call, does nothing
- **Double free**: Undefined behavior (caller must track ownership)
- **Mismatched alignment**: Undefined behavior (must match aligned_alloc)

**Data Guarantees (Preconditions)**

- ptr must be nullptr or allocated with aligned_alloc
- alignment must match the alignment used in aligned_alloc

**Complexity Analysis**

- **Time**: O(1) - single deallocation call
- **Space**: O(1)

**Example**

```cpp
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// Use data...

// Free with matching alignment
scl::memory::aligned_free(data, 64);
// data is now invalid (dangling pointer)
```

---

### AlignedBuffer

RAII wrapper for aligned memory allocation.

::: source_code file="scl/core/memory.hpp" symbol="AlignedBuffer" collapsed
:::

**Algorithm Description**

RAII wrapper that manages aligned memory allocation:
- Constructor: Allocates aligned memory using aligned_alloc
- Destructor: Automatically frees memory using aligned_free
- Movable: Supports move construction and assignment
- Non-copyable: Copy operations are deleted

Provides safe exception handling - if an exception occurs, destructor automatically frees memory.

**Edge Cases**

- **Allocation failure**: Buffer is invalid (operator bool returns false)
- **Move from**: Source buffer becomes invalid (empty)
- **Double move**: Second move has no effect

**Data Guarantees (Preconditions)**

- Alignment must be power of 2 and >= sizeof(void*)
- T must be trivially constructible

**Complexity Analysis**

- **Construction**: O(1) - single allocation
- **Destruction**: O(1) - single deallocation
- **Access**: O(1) - pointer dereference

**Example**

```cpp
#include "scl/core/memory.hpp"

{
    // RAII buffer - automatically freed on scope exit
    scl::memory::AlignedBuffer<Real> buffer(1000, 64);
    
    if (buffer) {  // Check allocation success
        Real* data = buffer.get();
        Array<Real> span = buffer.span();
        
        // Use data...
        span[0] = 1.0;
        span[1] = 2.0;
        
        // Automatic cleanup on scope exit
    }
}
```

---

### fill

Fill memory with a value using SIMD acceleration.

::: source_code file="scl/core/memory.hpp" symbol="fill" collapsed
:::

**Algorithm Description**

Fills a memory range with a specified value using SIMD optimization:

1. Create SIMD vector with broadcasted value
2. Process bulk of data with 4-way unrolled SIMD loop
3. Process remainder with SIMD loop
4. Process tail elements with scalar loop

Uses platform-specific SIMD instructions (AVX2/AVX-512/NEON) automatically selected by compiler.

**Edge Cases**

- **Empty span**: Returns immediately (no operation)
- **Null pointer**: Safe if span.len == 0, undefined behavior otherwise
- **Large values**: All elements set to same value (including NaNs)

**Data Guarantees (Preconditions)**

- span.ptr must be valid or nullptr (if span.len == 0)
- Memory must be writable

**Complexity Analysis**

- **Time**: O(n / lanes) where lanes is SIMD width (typically 4-16x faster than scalar)
- **Space**: O(1) auxiliary

**Example**

```cpp
#include "scl/core/memory.hpp"

Real* data = new Real[1000];
Array<Real> span(data, 1000);

// Fill with value 1.5
scl::memory::fill(span, Real(1.5));

// All elements in span are now 1.5
```

---

### zero

Zero out memory efficiently.

::: source_code file="scl/core/memory.hpp" symbol="zero" collapsed
:::

**Algorithm Description**

Efficiently zeros memory:
- For trivial types: Uses optimized memset (fastest)
- For non-trivial types: Uses fill(span, T(0))

Provides optimal performance for common case of zeroing primitive types.

**Edge Cases**

- **Empty span**: Returns immediately
- **Null pointer**: Safe if span.len == 0
- **Non-trivial types**: Uses fill, may call constructors

**Data Guarantees (Preconditions)**

- span.ptr must be valid or nullptr (if span.len == 0)

**Complexity Analysis**

- **Time**: O(n * sizeof(T)) for trivial types (memset), O(n / lanes) for non-trivial (SIMD fill)
- **Space**: O(1) auxiliary

**Example**

```cpp
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);
Array<Real> span(data, 1000);

// Zero out memory
scl::memory::zero(span);

// All elements are now 0.0
```

---

### copy_fast

Fast copy assuming NO overlap (uses memcpy semantics).

::: source_code file="scl/core/memory.hpp" symbol="copy_fast" collapsed
:::

**Algorithm Description**

Fast memory copy optimized for non-overlapping ranges:
- Uses memcpy for trivially copyable types
- Compiler can optimize with __restrict__ semantics
- No overlap checking (fastest option)

**Edge Cases**

- **Overlapping ranges**: Undefined behavior (use copy() instead)
- **Size mismatch**: Undefined behavior (src.len must == dst.len)
- **Empty ranges**: Returns immediately

**Data Guarantees (Preconditions)**

- src.len == dst.len
- src and dst must NOT overlap
- Both pointers must be valid

**Complexity Analysis**

- **Time**: O(n * sizeof(T)) - typically very fast (memcpy)
- **Space**: O(1) auxiliary

**Example**

```cpp
Real* src = new Real[1000];
Real* dst = scl::memory::aligned_alloc<Real>(1000, 64);

Array<const Real> src_span(src, 1000);
Array<Real> dst_span(dst, 1000);

// Fast copy (no overlap check)
scl::memory::copy_fast(src_span, dst_span);

// dst now contains copy of src
```

---

## Utility Functions

### copy

Copy with overlap handling (uses memmove semantics). Safe for overlapping ranges but slower than copy_fast.

**Complexity**: O(n * sizeof(T))

### prefetch_read / prefetch_write

Cache prefetch hints for optimization.

**Complexity**: O(1) - single prefetch instruction

## Platform Support

- **C++17+**: Uses aligned operator new/delete
- **POSIX**: Uses posix_memalign and free
- **Windows**: Uses _aligned_malloc and _aligned_free

## Performance Notes

- **Aligned allocation**: 64-byte alignment matches AVX-512 cache line size
- **SIMD operations**: Automatically use best available instructions (AVX2/AVX-512/NEON)
- **copy_fast**: Fastest when overlap is impossible, use copy() for safety

## See Also

- [Type System](./types) - Array<T> type used for memory views
- [SIMD](./simd) - SIMD abstraction layer
- [Registry](./registry) - Memory tracking for allocated buffers
