# simd.hpp

> scl/core/simd.hpp · Architecture-agnostic SIMD abstraction layer using Google Highway

## Overview

This file provides a unified interface for SIMD operations across different hardware architectures (AVX2, AVX-512, NEON, etc.) by wrapping Google Highway. All Highway functions are imported into the `scl::simd` namespace, providing architecture portability with zero runtime overhead.

Key features:
- Architecture-agnostic SIMD operations (x86, ARM, etc.)
- Zero runtime overhead (all abstractions compile away)
- Type-safe tag-based dispatch
- Automatic selection of optimal vector width
- Direct access to Highway functions

**Header**: `#include "scl/core/simd.hpp"`

---

## Main APIs

### Tag

Primary SIMD descriptor tag for `scl::Real` type.

**Definition**: `using Tag = ScalableTag<scl::Real>;`

**Algorithm Description**

Tag is a type alias for Highway's ScalableTag, which automatically selects the optimal vector width for the current hardware and scl::Real type (float or double).

The tag is used for type-safe SIMD dispatch - all SIMD operations require a tag parameter to specify the vector type and width.

**Edge Cases**

- **Scalar mode**: If SCL_ONLY_SCALAR is defined, SIMD is disabled and scalar fallback is used
- **Type mismatch**: Compile-time error if tag doesn't match vector type

**Data Guarantees (Preconditions)**

- scl::Real must be defined (float or double)
- Tag must be used consistently throughout SIMD operations

**Complexity Analysis**

- **Runtime**: O(1) - compile-time type only, no runtime cost
- **Vector width**: Depends on hardware (typically 4-16 elements for float, 2-8 for double)

**Example**

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;

// Create tag (constexpr, zero runtime cost)
const s::Tag d;

// Load data
Real* data = ...;
auto v = s::Load(d, data);  // Loads vector of Real values

// Perform operations
auto v2 = s::Mul(v, s::Set(d, 2.0));  // Multiply by 2.0

// Store result
s::Store(v2, d, output);
```

---

### IndexTag

SIMD descriptor tag for `scl::Index` type.

**Definition**: `using IndexTag = ScalableTag<scl::Index>;`

Used for vectorized index operations, gather/scatter, and integer arithmetic.

**Example**

```cpp
const s::IndexTag idx_d;
Index* indices = ...;
auto idx_vec = s::Load(idx_d, indices);
```

---

### ReinterpretTag

SIMD descriptor tag for unsigned integers matching scl::Real size.

**Definition**: `using ReinterpretTag = RebindToUnsigned<Tag>;`

Used for bitwise operations on floating-point data without type conversion.

**Example**

```cpp
const s::ReinterpretTag uint_d;
auto uint_vec = s::BitCast(uint_d, float_vec);  // Reinterpret as unsigned
```

---

## Highway Functions

All Highway SIMD functions are imported into `scl::simd` namespace. Key functions include:

### Memory Operations

- `Load(d, ptr)` - Load aligned vector
- `LoadU(d, ptr)` - Load unaligned vector
- `Store(vec, d, ptr)` - Store aligned vector
- `StoreU(vec, d, ptr)` - Store unaligned vector

### Arithmetic Operations

- `Add(a, b)` - Addition
- `Sub(a, b)` - Subtraction
- `Mul(a, b)` - Multiplication
- `Div(a, b)` - Division

### Comparison and Selection

- `Min(a, b)` - Minimum
- `Max(a, b)` - Maximum
- `Abs(v)` - Absolute value
- `IfThenElse(mask, true_val, false_val)` - Conditional selection

### Initialization

- `Set(d, value)` - Broadcast scalar value
- `Zero(d)` - Zero vector
- `Iota(d, start)` - Sequence vector

### Logical Operations

- `And(a, b)`, `Or(a, b)`, `Xor(a, b)`, `Not(a)` - Bitwise operations

**Note**: See Google Highway documentation for complete function list.

---

## Utility Functions

### lanes

Get the number of elements (lanes) in a SIMD vector for scl::Real.

**Example**

```cpp
const s::Tag d;
size_t num_lanes = s::Lanes(d);  // e.g., 8 for AVX2 float, 16 for AVX-512 float
```

**Complexity**: O(1) - compile-time constant

---

## Design Principles

### Zero Runtime Overhead

All SIMD abstractions are compile-time only:
- Tags are types, not values (compile-time dispatch)
- Using directive imports functions (no wrapper overhead)
- Highway optimizations apply directly

### Architecture Portability

Same code runs on different architectures:
- x86: AVX2, AVX-512
- ARM: NEON
- WebAssembly: SIMD128
- Scalar fallback when SIMD unavailable

### Type Safety

Tag-based dispatch prevents type mismatches:
- Tag must match vector element type
- Compile-time errors for mismatches
- No runtime type checks needed

## Configuration

### Disable SIMD

Define `SCL_ONLY_SCALAR` to disable SIMD and use scalar fallback:

```cpp
#define SCL_ONLY_SCALAR
// All SIMD operations become scalar loops
```

## Performance Notes

- **Vector width**: Automatically selected based on hardware (4-16x speedup typical)
- **Alignment**: Aligned loads/stores are faster than unaligned
- **Loop unrolling**: Combine SIMD with loop unrolling for maximum performance
- **Memory bandwidth**: SIMD is most effective when compute-bound, not memory-bound

## See Also

- [Memory Management](./memory) - Aligned allocation for SIMD buffers
- [Vectorize](./vectorize) - High-level vectorized operations
- [Google Highway Documentation](https://github.com/google/highway) - Complete SIMD API reference
