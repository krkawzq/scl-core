# Type System

SCL-Core uses a unified type system with compile-time configuration for precision and index types.

## Overview

The type system provides:

- **`Real`** - Configurable floating-point type (float32/float64/float16)
- **`Index`** - Signed integer for indexing (int16/int32/int64)
- **`Size`** - Unsigned integer for sizes and byte counts

## Fundamental Types

### Real - Floating-Point Type

Primary floating-point type for numerical computation.

```cpp
using Real = /* float | double | _Float16 */;
```

**Configuration:**

Selected at compile-time via one of:
- `SCL_USE_FLOAT32`: `Real = float` (32-bit IEEE 754)
- `SCL_USE_FLOAT64`: `Real = double` (64-bit IEEE 754)
- `SCL_USE_FLOAT16`: `Real = _Float16` (16-bit IEEE 754-2008)

**Metadata:**

```cpp
constexpr int DTYPE_CODE;         // 0, 1, or 2
constexpr const char* DTYPE_NAME; // "float32", "float64", or "float16"
```

**Precision Requirements:**

| Type | Bits | Mantissa | Exponent | Range |
|------|------|----------|----------|-------|
| float32 | 32 | 23 | 8 | ±3.4e38 |
| float64 | 64 | 52 | 11 | ±1.7e308 |
| float16 | 16 | 10 | 5 | ±65504 |

**Usage Guidelines:**

- Use `Real` for all numerical computations
- Do NOT hardcode `float` or `double` in library code
- Precision is library-wide, cannot mix within single build

**Performance:**

- **float32**: Fastest on most hardware, 2x throughput vs float64 on SIMD
- **float64**: Slower but better numerical stability
- **float16**: 4x throughput on modern GPUs, limited CPU support

**Example:**

```cpp
#include "scl/core/type.hpp"

using namespace scl;

Real x = 3.14;
Real y = 2.71;
Real z = x * y;

std::cout << "Using " << DTYPE_NAME << " precision\n";
```

### Index - Signed Integer for Indexing

Signed integer type for array indexing and dimensions.

```cpp
using Index = /* int16_t | int32_t | int64_t */;
```

**Configuration:**

Selected at compile-time via one of:
- `SCL_USE_INT16`: `Index = int16_t` (16-bit signed)
- `SCL_USE_INT32`: `Index = int32_t` (32-bit signed)
- `SCL_USE_INT64`: `Index = int64_t` (64-bit signed)

**Metadata:**

```cpp
constexpr int INDEX_DTYPE_CODE;         // 0, 1, or 2
constexpr const char* INDEX_DTYPE_NAME; // "int16", "int32", or "int64"
```

**Rationale for Signed:**

- Allows negative indices for reverse iteration
- Simplifies loop bounds checking (`i >= 0`)
- Compatible with BLAS/LAPACK conventions
- Prevents unsigned underflow bugs in loops

**Size Guidelines:**

| Type | Range | Max Matrix Size | Use Case |
|------|-------|-----------------|----------|
| int16 | ±32K | 32K × 32K | Small matrices, saves memory |
| int32 | ±2B | 2B × 2B | Standard choice |
| int64 | ±9E18 | Very large | Huge matrices, increases memory |

**Usage Guidelines:**

- Use `Index` for all array indexing, dimensions, loop counters
- Do NOT use `int` or `size_t` for indexing
- Use `Size` (unsigned) only for memory/byte sizes

**Performance:**

Smaller indices reduce memory traffic and improve cache performance, but must be large enough for your data.

**Example:**

```cpp
#include "scl/core/type.hpp"

using namespace scl;

Index rows = 1000;
Index cols = 500;

for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
        // Process element (i, j)
    }
}
```

### Size - Unsigned Integer for Sizes

Unsigned integer type for sizes and byte counts.

```cpp
using Size = size_t;
```

**Usage Guidelines:**

- Use `Size` for memory sizes, byte counts, capacities
- Use `Size` for loop counters that never go negative
- Do NOT use `Size` for array indexing (use `Index`)

**Example:**

```cpp
#include "scl/core/type.hpp"

using namespace scl;

Size num_bytes = 1024 * 1024;  // 1 MB
Size capacity = 10000;

Real* data = new Real[capacity];
Size byte_size = capacity * sizeof(Real);
```

## Type Traits

### Numeric Limits

```cpp
#include <limits>

// Real limits
constexpr Real real_min = std::numeric_limits<Real>::min();
constexpr Real real_max = std::numeric_limits<Real>::max();
constexpr Real real_epsilon = std::numeric_limits<Real>::epsilon();

// Index limits
constexpr Index index_min = std::numeric_limits<Index>::min();
constexpr Index index_max = std::numeric_limits<Index>::max();
```

### Type Queries

```cpp
#include <type_traits>

// Check if Real is float
constexpr bool is_float32 = std::is_same_v<Real, float>;

// Check if Index is int32
constexpr bool is_int32 = std::is_same_v<Index, int32_t>;

// Size of types
constexpr size_t real_size = sizeof(Real);
constexpr size_t index_size = sizeof(Index);
```

## Compile-Time Configuration

### CMake Configuration

```cmake
# In CMakeLists.txt

# Set Real type
option(SCL_USE_FLOAT32 "Use float32 for Real" ON)
option(SCL_USE_FLOAT64 "Use float64 for Real" OFF)
option(SCL_USE_FLOAT16 "Use float16 for Real" OFF)

# Set Index type
option(SCL_USE_INT16 "Use int16 for Index" OFF)
option(SCL_USE_INT32 "Use int32 for Index" ON)
option(SCL_USE_INT64 "Use int64 for Index" OFF)

# Generate config.hpp
configure_file(
    "${CMAKE_SOURCE_DIR}/scl/config.hpp.in"
    "${CMAKE_BINARY_DIR}/scl/config.hpp"
)
```

### Manual Configuration

```cpp
// In scl/config.hpp

// Real type (exactly one must be defined)
#define SCL_USE_FLOAT32
// #define SCL_USE_FLOAT64
// #define SCL_USE_FLOAT16

// Index type (exactly one must be defined)
// #define SCL_USE_INT16
#define SCL_USE_INT32
// #define SCL_USE_INT64
```

## Type Conversion

### Safe Conversions

```cpp
// Index to Size (always safe)
Index i = 100;
Size s = static_cast<Size>(i);

// Size to Index (check bounds)
Size s = 1000;
if (s <= static_cast<Size>(std::numeric_limits<Index>::max())) {
    Index i = static_cast<Index>(s);
}

// Real to Index (truncation)
Real x = 3.14;
Index i = static_cast<Index>(x);  // i = 3
```

### Unsafe Conversions

```cpp
// BAD: Size to Index without check
Size s = 3000000000;  // 3 billion
Index i = static_cast<Index>(s);  // Overflow if Index = int32_t!

// BAD: Mixing signed and unsigned in comparisons
Index i = -1;
Size s = 10;
if (i < s) {  // WARNING: i converted to Size, becomes huge!
    // ...
}
```

## Best Practices

### 1. Use Correct Type for Purpose

```cpp
// GOOD
Index row = 0;           // Array index
Size capacity = 1000;    // Memory size
Real value = 3.14;       // Numerical value

// BAD
int row = 0;             // Use Index
unsigned capacity = 1000; // Use Size
float value = 3.14;      // Use Real
```

### 2. Avoid Mixing Signed and Unsigned

```cpp
// BAD: Mixing signed and unsigned
Index i = -1;
Size n = 10;
if (i < n) {  // i converted to Size, comparison wrong!
    // ...
}

// GOOD: Use same signedness
Index i = -1;
Index n = 10;
if (i < n) {  // Correct comparison
    // ...
}
```

### 3. Check Bounds for Conversions

```cpp
// BAD: Unchecked conversion
Size large = 5000000000;
Index i = static_cast<Index>(large);  // Overflow!

// GOOD: Check bounds
Size large = 5000000000;
if (large <= static_cast<Size>(std::numeric_limits<Index>::max())) {
    Index i = static_cast<Index>(large);
} else {
    // Handle error
}
```

### 4. Use Type Aliases Consistently

```cpp
// GOOD: Use type aliases
void process(const Real* data, Index n);

// BAD: Hardcode types
void process(const float* data, int n);
```

## Type Selection Guidelines

### Choosing Real Type

**Use float32 when:**
- Performance is critical
- Memory bandwidth is limited
- Numerical precision is adequate
- SIMD throughput matters

**Use float64 when:**
- High numerical precision required
- Accumulating many values
- Iterative algorithms (error accumulation)
- Scientific computing standards

**Use float16 when:**
- GPU acceleration available
- Memory extremely limited
- Reduced precision acceptable
- Inference (not training)

### Choosing Index Type

**Use int16 when:**
- Matrices < 32K × 32K
- Memory is very limited
- Cache performance critical
- Embedded systems

**Use int32 when:**
- Standard use case
- Matrices < 2B × 2B
- Good balance of range and memory

**Use int64 when:**
- Very large matrices
- Future-proofing
- Memory not a concern
- 64-bit pointers anyway

## Debugging Type Issues

### Print Type Information

```cpp
#include <iostream>
#include <typeinfo>

std::cout << "Real type: " << typeid(Real).name() << "\n";
std::cout << "Real size: " << sizeof(Real) << " bytes\n";
std::cout << "Index type: " << typeid(Index).name() << "\n";
std::cout << "Index size: " << sizeof(Index) << " bytes\n";
```

### Compile-Time Assertions

```cpp
// Ensure Real is float or double
static_assert(std::is_floating_point_v<Real>, 
              "Real must be floating-point type");

// Ensure Index is signed
static_assert(std::is_signed_v<Index>, 
              "Index must be signed integer type");

// Ensure Size is unsigned
static_assert(std::is_unsigned_v<Size>, 
              "Size must be unsigned integer type");
```

---

::: tip Type Consistency
Always use `Real`, `Index`, and `Size` instead of hardcoded types. This ensures your code works with any configuration.
:::

