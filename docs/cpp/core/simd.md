---
title: SIMD and Vectorization
description: SIMD abstraction using Google Highway
---

# SIMD and Vectorization

SCL-Core provides SIMD (Single Instruction, Multiple Data) support via the Google Highway library, enabling vectorized operations for maximum performance.

## Overview

SIMD allows processing multiple data elements in parallel using CPU vector instructions (SSE, AVX, AVX-512). SCL-Core abstracts SIMD operations through Highway, providing a portable interface across different CPU architectures.

## Basic Usage

### SIMD Namespace

```cpp
#include "scl/core/simd.hpp"

namespace s = scl::simd;
```

### SIMD Tags

SIMD operations require a tag that specifies the data type and vector width:

```cpp
// Type-based tag selection
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;  // Tag instance

// Or use predefined tags
const s::RealTag d;      // For Real type
const s::IndexTag d_idx; // For Index type
```

### Basic Operations

```cpp
namespace s = scl::simd;
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;

// Load data
auto v = s::Load(d, data_ptr);

// Arithmetic operations
auto v_sum = s::Add(v1, v2);
auto v_prod = s::Mul(v1, v2);
auto v_diff = s::Sub(v1, v2);
auto v_div = s::Div(v1, v2);

// Store results
s::Store(v_sum, d, output_ptr);
```

## Vectorized Loops

### Basic Loop Pattern

```cpp
namespace s = scl::simd;
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;
const size_t lanes = s::Lanes(d);

Size i = 0;
for (; i + lanes <= n; i += lanes) {
    // Load vector
    auto v = s::Load(d, data + i);
    
    // Process vector
    auto v_result = s::Mul(v, s::Set(d, scale));
    
    // Store result
    s::Store(v_result, d, output + i);
}

// Scalar remainder
for (; i < n; ++i) {
    output[i] = data[i] * scale;
}
```

### Multiple Accumulators

For reduction operations, use multiple accumulators to hide latency:

```cpp
namespace s = scl::simd;
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;
const size_t lanes = s::Lanes(d);

auto v_sum0 = s::Zero(d);
auto v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d);
auto v_sum3 = s::Zero(d);

Size i = 0;
for (; i + 4 * lanes <= n; i += 4 * lanes) {
    auto v0 = s::Load(d, data + i + 0 * lanes);
    auto v1 = s::Load(d, data + i + 1 * lanes);
    auto v2 = s::Load(d, data + i + 2 * lanes);
    auto v3 = s::Load(d, data + i + 3 * lanes);
    
    v_sum0 = s::Add(v_sum0, v0);
    v_sum1 = s::Add(v_sum1, v1);
    v_sum2 = s::Add(v_sum2, v2);
    v_sum3 = s::Add(v_sum3, v3);
}

// Combine accumulators
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

// Scalar remainder
for (; i < n; ++i) {
    sum += data[i];
}
```

## Common Operations

### Element-Wise Operations

```cpp
namespace s = scl::simd;
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;
const size_t lanes = s::Lanes(d);

Size i = 0;
for (; i + lanes <= n; i += lanes) {
    auto v = s::Load(d, data + i);
    
    // Scale
    auto v_scaled = s::Mul(v, s::Set(d, scale));
    
    // Add constant
    auto v_added = s::Add(v, s::Set(d, constant));
    
    // Square
    auto v_squared = s::Mul(v, v);
    
    // Square root
    auto v_sqrt = s::Sqrt(d, v);
    
    // Exponential
    auto v_exp = s::Exp(d, v);
    
    s::Store(v_scaled, d, output + i);
}
```

### Reduction Operations

```cpp
// Sum reduction
Real compute_sum(const Real* data, Size n) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    
    auto v_sum = s::Zero(d);
    Size i = 0;
    
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, data + i);
        v_sum = s::Add(v_sum, v);
    }
    
    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    
    // Scalar remainder
    for (; i < n; ++i) {
        sum += data[i];
    }
    
    return sum;
}
```

### Dot Product

```cpp
Real dot_product(const Real* a, const Real* b, Size n) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    
    auto v_sum = s::Zero(d);
    Size i = 0;
    
    for (; i + lanes <= n; i += lanes) {
        auto va = s::Load(d, a + i);
        auto vb = s::Load(d, b + i);
        v_sum = s::MulAdd(va, vb, v_sum);  // Fused multiply-add
    }
    
    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    
    // Scalar remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}
```

## Alignment Requirements

### Aligned Memory

SIMD operations require aligned memory for optimal performance:

```cpp
// Allocate aligned memory
auto buffer = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
Array<Real> data = {buffer.get(), n};

// Use with SIMD
namespace s = scl::simd;
using SimdTag = s::SimdTagFor<Real>;
const SimdTag d;
auto v = s::Load(d, data.data());  // Requires alignment
```

### Unaligned Load/Store

For unaligned data, use `LoadU` and `StoreU`:

```cpp
// Unaligned load
auto v = s::LoadU(d, unaligned_ptr);

// Unaligned store
s::StoreU(v, d, unaligned_ptr);
```

**Note**: Unaligned operations are slower than aligned operations.

## Masked Operations

### Conditional Operations

```cpp
// Create mask
auto mask = s::Lt(d, v, s::Set(d, threshold));

// Conditional operations
auto v_result = s::IfThenElse(mask, s::Set(d, 1.0), s::Set(d, 0.0));
```

### Masked Load/Store

```cpp
// Load with mask
auto mask = s::Lt(d, indices, s::Set(d, max_idx));
auto v = s::MaskedLoad(mask, d, data_ptr);

// Store with mask
s::MaskedStore(v, mask, d, output_ptr);
```

## Advanced Patterns

### Horizontal Reductions

```cpp
// Sum all lanes
Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

// Maximum across lanes
Real max_val = s::GetLane(s::MaxOfLanes(d, v_max));

// Minimum across lanes
Real min_val = s::GetLane(s::MinOfLanes(d, v_min));
```

### Broadcast Operations

```cpp
// Broadcast scalar to vector
auto v_scalar = s::Set(d, value);

// Broadcast from array
auto v_broadcast = s::LoadDup128(d, array_ptr);
```

### Swizzle and Shuffle

```cpp
// Interleave
auto v_interleaved = s::InterleaveLower(d, v1, v2);

// Shuffle lanes
auto v_shuffled = s::Shuffle2301(d, v);
```

## Performance Tips

### 1. Ensure Alignment

```cpp
// Always align data used with SIMD
auto buffer = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
```

### 2. Use Multiple Accumulators

```cpp
// Good: Multiple accumulators hide latency
auto v_sum0 = s::Zero(d);
auto v_sum1 = s::Zero(d);
// ...

// Avoid: Single accumulator
auto v_sum = s::Zero(d);  // May cause pipeline stalls
```

### 3. Minimize Scalar Remainder

```cpp
// Process in chunks to minimize remainder
constexpr Size CHUNK_SIZE = 256;
for (Size offset = 0; offset < n; offset += CHUNK_SIZE) {
    Size chunk_size = std::min(CHUNK_SIZE, n - offset);
    process_chunk(data + offset, chunk_size);
}
```

### 4. Prefetch for Indirect Access

```cpp
// Prefetch for indirect memory access
SCL_PREFETCH_READ(&data[indices[i + PREFETCH_DISTANCE]], 0);
```

## Disabling SIMD

For debugging or compatibility:

```cpp
// Compile with scalar-only mode
#define SCL_ONLY_SCALAR
// Or
#define HWY_COMPILE_ONLY_SCALAR
```

## Architecture Support

Highway automatically selects the best SIMD instructions available:

- **AVX-512**: 64-byte vectors (8 doubles, 16 floats)
- **AVX2/AVX**: 32-byte vectors (4 doubles, 8 floats)
- **SSE2**: 16-byte vectors (2 doubles, 4 floats)
- **Scalar**: Fallback when SIMD unavailable

## Related Documentation

- [Core Types](./types.md) - Array views and types
- [Memory Management](./memory.md) - Aligned allocation
- [Kernels](../kernels/) - SIMD-optimized kernels
