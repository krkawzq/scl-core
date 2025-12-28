# type.hpp

> scl/core/type.hpp · Unified type system with compile-time configuration

## Overview

This file defines the fundamental type system for SCL-Core, providing compile-time configurable precision and index types. The type system enables a single codebase to support multiple precision levels (float32/float64/float16) and index sizes (int16/int32/int64) without runtime overhead.

Key features:
- Compile-time type selection via preprocessor macros
- Zero-overhead type aliases (no runtime cost)
- Type-safe array views (Array<T>)
- Generic concepts for container interoperability (ArrayLike, CSRLike)

**Header**: `#include "scl/core/type.hpp"`

---

## Main APIs

### Real

Primary floating-point type for numerical computation, configured at compile time.

::: source_code file="scl/core/type.hpp" symbol="Real" collapsed
:::

**Algorithm Description**

Real is a type alias selected at compile time via one of three preprocessor macros:
- `SCL_USE_FLOAT32`: Real = float (32-bit IEEE 754)
- `SCL_USE_FLOAT64`: Real = double (64-bit IEEE 754)
- `SCL_USE_FLOAT16`: Real = _Float16 (16-bit IEEE 754-2008)

The selection is enforced at compile time: exactly one macro must be defined, otherwise compilation error. The precision is library-wide and cannot be mixed within a single build.

**Edge Cases**

- **No macro defined**: Compilation error (exactly one must be defined)
- **Multiple macros defined**: Compilation error (conflicting definitions)
- **float16 compatibility**: Requires GCC >= 12 or Clang >= 15

**Data Guarantees (Preconditions)**

- One and only one of SCL_USE_FLOAT32, SCL_USE_FLOAT64, SCL_USE_FLOAT16 must be defined
- Precision is consistent across entire library build

**Complexity Analysis**

- **Time**: O(1) - compile-time selection, zero runtime overhead
- **Space**: O(1) - type alias, no storage overhead

**Example**

```cpp
#include "scl/core/type.hpp"

// Compile with: -DSCL_USE_FLOAT32
Real x = 3.14f;  // Real is float

// Compile with: -DSCL_USE_FLOAT64
Real y = 3.14;   // Real is double

// Metadata available at compile time
constexpr int dtype_code = DTYPE_CODE;         // 0, 1, or 2
constexpr const char* dtype_name = DTYPE_NAME; // "float32", "float64", or "float16"
```

---

### Index

Signed integer type for array indexing and dimensions, configured at compile time.

::: source_code file="scl/core/type.hpp" symbol="Index" collapsed
:::

**Algorithm Description**

Index is a signed integer type selected at compile time via one of three preprocessor macros:
- `SCL_USE_INT16`: Index = int16_t (16-bit signed, supports up to 32K x 32K matrices)
- `SCL_USE_INT32`: Index = int32_t (32-bit signed, standard choice, supports 2B x 2B matrices)
- `SCL_USE_INT64`: Index = int64_t (64-bit signed, for very large matrices)

The use of signed integers (instead of unsigned) allows:
- Negative indices for reverse iteration
- Simpler loop bounds checking (i >= 0)
- Compatibility with BLAS/LAPACK conventions
- Prevention of unsigned underflow bugs

**Edge Cases**

- **No macro defined**: Compilation error (exactly one must be defined)
- **Multiple macros defined**: Compilation error (conflicting definitions)
- **Overflow**: Index arithmetic can overflow if values exceed type range

**Data Guarantees (Preconditions)**

- One and only one of SCL_USE_INT16, SCL_USE_INT32, SCL_USE_INT64 must be defined
- Index values should be non-negative for array indexing (though signed type allows negative values for special purposes)

**Complexity Analysis**

- **Time**: O(1) - compile-time selection, zero runtime overhead
- **Space**: Smaller indices reduce memory traffic and improve cache performance (especially for sparse matrix index arrays)

**Example**

```cpp
#include "scl/core/type.hpp"

// Compile with: -DSCL_USE_INT32
Index i = 1000;      // Index is int32_t
Index j = -1;        // Allowed (can be used for reverse iteration)

// Metadata
constexpr int index_code = INDEX_DTYPE_CODE;         // 0, 1, or 2
constexpr const char* index_name = INDEX_DTYPE_NAME; // "int16", "int32", or "int64"

// Use for array indexing
Array<Real> data(n);
for (Index idx = 0; idx < n; ++idx) {
    Real value = data[idx];
}
```

---

### Array<T>

Lightweight, non-owning view of a contiguous 1D array. Zero-overhead POD type with trivial copy.

::: source_code file="scl/core/type.hpp" symbol="Array" collapsed
:::

**Algorithm Description**

Array<T> is a struct containing a pointer and length:
- `ptr`: Pointer to first element (T*)
- `len`: Number of elements (Size)

Design philosophy:
- **Non-owning**: No destructor, no allocation, no deallocation
- **Zero-overhead**: POD type with trivial copy (16 bytes on 64-bit systems)
- **Const-correct**: Array<T> (mutable) vs Array<const T> (immutable)
- **Public members**: Direct access for performance-critical code
- **Copyable**: Shallow copy of view, not the underlying data

The struct provides standard container-like methods (size, empty, begin, end, operator[]) all force-inlined for zero function call overhead.

**Edge Cases**

- **Empty array**: ptr = nullptr, len = 0
- **Null pointer with non-zero length**: Undefined behavior (caller must ensure validity)
- **Out-of-bounds access**: Debug builds assert, release builds undefined behavior
- **Lifetime**: Array view must not outlive the underlying data

**Data Guarantees (Preconditions)**

- If len > 0, ptr must point to valid array of at least len elements
- If len == 0, ptr can be nullptr or any value (ignored)
- Underlying data lifetime must exceed Array view lifetime

**Complexity Analysis**

- **Time**: O(1) for all operations (all methods are force-inlined)
- **Space**: sizeof(Array<T>) = sizeof(T*) + sizeof(Size) = 16 bytes on 64-bit systems

**Example**

```cpp
#include "scl/core/type.hpp"

// Create from pointer and size
Real* data = new Real[100];
Array<Real> arr(data, 100);

// Access elements
Real x = arr[0];
arr[5] = 3.14;

// Iterate
for (Index i = 0; i < arr.size(); ++i) {
    arr[i] *= 2.0;
}

// Const view
Array<const Real> const_view = arr;  // Implicit conversion
// const_view[0] = 1.0;  // Error: cannot modify const view

// Empty array
Array<Real> empty;  // ptr = nullptr, len = 0
```

---

### Size

Unsigned integer type for memory sizes and byte counts.

**Definition**: `using Size = std::size_t;`

**Usage Guidelines**
- Use for memory allocation sizes
- Use for byte counts and buffer lengths
- Do NOT use for array indexing (use Index instead)
- Do NOT use for loop counters over array elements

**Example**

```cpp
Size buffer_size = 1024 * sizeof(Real);  // Memory size in bytes
Real* data = new Real[100];
Size n_bytes = 100 * sizeof(Real);       // Size in bytes

// Conversion from Index to Size (ensure non-negative)
Index idx = 100;
Size sz = static_cast<Size>(idx);  // Only if idx >= 0
```

---

### Byte

Unsigned 8-bit integer for raw memory operations.

**Definition**: `using Byte = std::uint8_t;`

**Usage Guidelines**
- Use for raw memory buffers
- Use for serialization/deserialization
- Use for byte-level I/O operations

**Example**

```cpp
Byte* raw_buffer = new Byte[1024];
// Use for raw memory operations
```

---

### Pointer

Generic untyped pointer for discontiguous storage and C-ABI boundaries.

**Definition**: `using Pointer = void*;`

**Usage Guidelines**
- Use for type-erased pointers across C-ABI
- Use for generic memory handles
- Always cast to proper type before dereferencing

**Warning**: Void pointers bypass type safety. Use only at API boundaries where type erasure is necessary (e.g., Python bindings).

---

## Utility Concepts

### ArrayLike<A>

Concept for 1D array-like containers with random access.

::: source_code file="scl/core/type.hpp" symbol="ArrayLike" collapsed
:::

**Requirements**

- `value_type`: Member type alias for element type
- `size()`: Returns number of elements (convertible to Size)
- `operator[](Index)`: Random access to elements
- `begin()`: Returns iterator to first element
- `end()`: Returns iterator to one-past-last element

**Satisfied By**

- `scl::Array<T>`
- `std::vector<T>`
- `std::array<T, N>`
- `std::deque<T>`
- `std::span<T>` (C++20)
- Custom containers with compatible interface

**Example**

```cpp
template <ArrayLike A>
void process(const A& arr) {
    for (Index i = 0; i < arr.size(); ++i) {
        do_something(arr[i]);
    }
}

// Works with any ArrayLike type
Array<Real> arr = ...;
std::vector<Real> vec = ...;
process(arr);  // OK
process(vec);  // OK
```

---

### CSRLike<M>

Concept for CSR (Compressed Sparse Row) sparse matrices.

::: source_code file="scl/core/type.hpp" symbol="CSRLike" collapsed
:::

**Requirements**

- `ValueType`: Element type alias
- `Tag`: Must be `TagSparse<true>`
- `is_csr`: Static constexpr bool == true
- `rows()`, `cols()`, `nnz()`: Matrix dimensions
- `primary_values(i)`, `primary_indices(i)`, `primary_length(i)`: Row access

Enables generic algorithms that work with any CSR-like sparse matrix type.

---

### TagSparse<IsCSR>

Tag type for sparse matrix format selection.

::: source_code file="scl/core/type.hpp" symbol="TagSparse" collapsed
:::

**Template Parameters**

- `IsCSR` [bool]: true for CSR (row-major), false for CSC (column-major)

**Static Members**

- `is_csr`: constexpr bool, equals IsCSR
- `is_csc`: constexpr bool, equals !IsCSR

Enables compile-time dispatch and type traits for sparse matrices.

---

## Type Configuration

### Compile-Time Selection

Types are configured at compile time via preprocessor macros:

```cpp
// In CMakeLists.txt or config.hpp
#define SCL_USE_FLOAT32  // or FLOAT64, FLOAT16
#define SCL_USE_INT32    // or INT16, INT64
```

This enables:
- Single codebase for multiple precisions
- Zero runtime overhead for type selection
- Optimal code generation

### Metadata

Runtime-accessible metadata:

```cpp
constexpr int dtype_code = DTYPE_CODE;         // 0 (float32), 1 (float64), 2 (float16)
constexpr const char* dtype_name = DTYPE_NAME; // "float32", "float64", or "float16"

constexpr int index_code = INDEX_DTYPE_CODE;         // 0 (int16), 1 (int32), 2 (int64)
constexpr const char* index_name = INDEX_DTYPE_NAME; // "int16", "int32", or "int64"
```

## Design Principles

### Zero Dependencies

Core types depend only on:
- C++17 standard library
- Standard headers: `<cstddef>`, `<cstdint>`, `<type_traits>`, `<concepts>`

### Zero Runtime Overhead

- Type aliases (Real, Index) are compile-time selections
- Array<T> is a POD type with trivial copy
- All methods are force-inlined
- Concepts are compile-time checks (zero runtime cost)

### Explicit Resource Management

Array<T> does NOT own memory:
- No destructor
- No allocation/deallocation
- Lifetime of underlying data must exceed Array view lifetime

## See Also

- [Sparse Matrix](./sparse) - Uses Array<T> for sparse matrix storage
- [Memory Management](./memory) - Allocation functions return pointers used with Array<T>
