# SIMD Abstraction

Architecture-agnostic SIMD abstraction layer using Google Highway for portable vectorization.

## Overview

SIMD abstraction provides:

- **Portable Vectorization** - Same code runs on x86, ARM, etc.
- **Type Safety** - Tag-based dispatch prevents type mismatches
- **Zero Overhead** - All abstractions compile away
- **Automatic Optimization** - Uses best vector width for hardware

## Design Principles

- Zero runtime overhead: all abstractions compile away
- Type safety: tag-based dispatch prevents type mismatches
- Architecture portability: same code runs on x86, ARM, etc.
- Scalability: automatically uses best vector width for hardware

## Configuration

- `SCL_ONLY_SCALAR`: Disable SIMD, use scalar fallback only
- `HWY_COMPILE_ONLY_SCALAR`: Propagated from SCL_ONLY_SCALAR

## Namespace Injection

All Highway functions are imported into `scl::simd` namespace:

```cpp
namespace s = scl::simd;
const s::Tag d;

auto v1 = s::Load(d, ptr);        // Instead of hwy::Load
auto v2 = s::Add(v1, v1);         // Instead of hwy::Add
s::Store(v2, d, output);          // Instead of hwy::Store
```

## Type Aliases

### Tag

Primary SIMD descriptor tag for `scl::Real` type:

```cpp
namespace s = scl::simd;
const s::Tag d;

// Tag automatically selects optimal vector width
// For float on AVX2: 8 lanes
// For double on AVX2: 4 lanes
```

**Definition:**
```cpp
using Tag = ScalableTag<scl::Real>;
```

**When to Use:**
- Processing floating-point arrays
- Mathematical operations on `scl::Real` data
- Default choice for most SCL kernels

### IndexTag

SIMD descriptor tag for `scl::Index` type:

```cpp
using IndexTag = ScalableTag<scl::Index>;
```

**When to Use:**
- Vectorized index calculations
- Gather/scatter operations with 64-bit indices
- Integer arithmetic on `scl::Index` arrays

### ReinterpretTag

SIMD descriptor tag for unsigned integers matching `scl::Real` size:

```cpp
using ReinterpretTag = RebindToUnsigned<Tag>;
```

**When to Use:**
- Bitwise masking of floating-point values
- Sign bit manipulation
- Fast absolute value (clear sign bit)
- NaN/Inf detection via bit patterns

## Helper Functions

### lanes

Get the number of elements (lanes) in a SIMD vector:

```cpp
namespace s = scl::simd;
size_t num_lanes = s::lanes();

// For scl::Real = float on AVX2: returns 8
// For scl::Real = double on AVX2: returns 4
```

**Returns:**
- Number of lanes (typically 4, 8, 16, etc. for floats on modern CPUs)

**Complexity:**
- Time: O(1) - compile-time constant
- Zero runtime cost - inlined to a compile-time constant

**Usage Notes:**
- Useful for loop bounds and buffer allocation
- Value varies by architecture (AVX2 vs AVX-512 vs NEON)

### GetSimdTag

Get the appropriate SIMD tag for a given type T:

```cpp
auto tag = scl::simd::GetSimdTag<Real>();      // Returns Tag
auto idx_tag = scl::simd::GetSimdTag<Index>(); // Returns IndexTag
```

**Returns:**
- `Tag` if T == scl::Real
- `IndexTag` if T == scl::Index
- `ScalableTag<T>` otherwise

## Common Operations

### Load and Store

```cpp
namespace s = scl::simd;
const s::Tag d;

// Load from memory
auto v = s::Load(d, float_ptr);              // Aligned load
auto v_u = s::LoadU(d, float_ptr);           // Unaligned load

// Store to memory
s::Store(v, d, output_ptr);                  // Aligned store
s::StoreU(v, d, output_ptr);                 // Unaligned store
```

### Arithmetic Operations

```cpp
auto v1 = s::Load(d, a);
auto v2 = s::Load(d, b);

auto v_add = s::Add(v1, v2);                 // Addition
auto v_sub = s::Sub(v1, v2);                 // Subtraction
auto v_mul = s::Mul(v1, v2);                 // Multiplication
auto v_div = s::Div(v1, v2);                 // Division

auto v_fma = s::MulAdd(v1, v2, v3);          // Fused multiply-add
```

### Comparison Operations

```cpp
auto mask = s::Gt(v1, v2);                   // Greater than
auto v_min = s::Min(v1, v2);                 // Minimum
auto v_max = s::Max(v1, v2);                 // Maximum
auto v_abs = s::Abs(v1);                     // Absolute value
```

### Logical Operations

```cpp
auto v_and = s::And(v1, v2);                 // Bitwise AND
auto v_or = s::Or(v1, v2);                   // Bitwise OR
auto v_xor = s::Xor(v1, v2);                 // Bitwise XOR
auto v_not = s::Not(v1);                     // Bitwise NOT
```

### Vector Initialization

```cpp
auto v_zero = s::Zero(d);                    // Zero vector
auto v_one = s::Set(d, 1.0);                 // Broadcast scalar value
auto v_iota = s::Iota(d, 0);                 // [0, 1, 2, 3, ...]
```

### Conditional Selection

```cpp
auto result = s::IfThenElse(mask, v_true, v_false);
// result[i] = mask[i] ? v_true[i] : v_false[i]
```

### Reduction Operations

```cpp
auto sum = s::SumOfLanes(d, v);              // Sum all lanes
auto max_val = s::MaxOfLanes(d, v);          // Maximum of all lanes
auto min_val = s::MinOfLanes(d, v);          // Minimum of all lanes
```

## Typical Usage Pattern

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;
const s::Tag d;
const size_t lanes = s::Lanes(d);

// Process array in SIMD chunks
for (size_t i = 0; i < n; i += lanes) {
    // Load
    auto v = s::LoadU(d, input + i);
    
    // Process (example: multiply by 2)
    auto result = s::Mul(v, s::Set(d, 2.0));
    
    // Store
    s::StoreU(result, d, output + i);
}

// Handle remainder with scalar loop
for (size_t i = (n / lanes) * lanes; i < n; ++i) {
    output[i] = input[i] * 2.0;
}
```

## Performance Tips

### 1. Use Aligned Loads/Stores When Possible

```cpp
// GOOD: Aligned allocation + aligned operations
Real* data = scl::memory::aligned_alloc<Real>(n, 64);
auto v = s::Load(d, data);  // Aligned load (faster)

// OK: Unaligned operations (slightly slower)
auto v = s::LoadU(d, data);  // Unaligned load
```

### 2. Minimize Gather/Scatter

```cpp
// BAD: Gather (slow)
auto v = s::Gather(d, base_ptr, indices);

// GOOD: Contiguous access when possible
auto v = s::Load(d, base_ptr + offset);
```

### 3. Use FMA When Available

```cpp
// GOOD: Fused multiply-add (single instruction)
auto result = s::MulAdd(a, b, c);  // a * b + c

// OK: Separate operations (two instructions)
auto result = s::Add(s::Mul(a, b), c);
```

---

::: tip Portable SIMD
The same code works across different architectures. Highway automatically selects the best SIMD instructions for your hardware (AVX2, AVX-512, NEON, etc.).
:::

