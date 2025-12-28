# Documentation Standard

SCL-Core uses a dual-file documentation system to separate implementation from API documentation.

## Dual-File System

### `.hpp` Files - Implementation

Implementation files contain minimal inline comments:

```cpp
// scl/kernel/normalize.hpp
#pragma once

#include "scl/kernel/normalize.h"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

namespace scl::kernel::normalize {

template <CSRLike MatrixT>
void normalize_rows_inplace(MatrixT& matrix, NormMode mode, Real eps) {
    // Use Kahan summation for numerical stability
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        auto vals = matrix.primary_values(i);
        const Index len = matrix.primary_length(i);
        
        if (len == 0) return;
        
        // Compute norm
        Real norm = compute_norm(vals.ptr, len, mode);
        
        // Normalize if norm > epsilon
        if (norm > eps) {
            const Real inv_norm = Real(1) / norm;
            for (Index j = 0; j < len; ++j) {
                vals.ptr[j] *= inv_norm;
            }
        }
    });
}

} // namespace scl::kernel::normalize
```

**Guidelines:**
- One-line function purpose (optional if self-explanatory)
- Non-obvious algorithm tricks or optimizations
- Warning about subtle behavior
- Keep code clean and readable

### `.h` Files - API Documentation

Documentation files use C++ syntax with comprehensive block comments:

```cpp
// scl/kernel/normalize.h
#pragma once

namespace scl::kernel::normalize {

/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_rows_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Normalize each row of a CSR matrix to unit norm in-place.
 *
 * PARAMETERS:
 *     matrix   [in,out] Mutable CSR matrix, modified in-place
 *     mode     [in]     Norm type for normalization
 *     epsilon  [in]     Small constant to prevent division by zero
 *
 * PRECONDITIONS:
 *     - matrix must be valid CSR format
 *     - matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each row has unit norm under specified mode (if original norm > epsilon)
 *     - Rows with norm <= epsilon are unchanged
 *     - Matrix structure (indices, indptr) unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row i in parallel:
 *         1. Compute norm of row i
 *         2. If norm > epsilon: divide each element by norm
 *         3. Otherwise: leave row unchanged
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * NUMERICAL NOTES:
 *     - Uses epsilon to handle zero/near-zero rows gracefully
 *     - L2 norm uses compensated summation for accuracy
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows, no shared mutable state
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type
    Real epsilon = 1e-12           // Zero-norm threshold
);

} // namespace scl::kernel::normalize
```

**Key Points:**
- `.h` extension for syntax highlighting (not compiled)
- Block comments (`/* */`) for documentation
- Actual function declarations with inline comments
- Inline comments visible in quick queries

## Documentation Sections

Each function documentation block MUST include these sections (in order):

| Section | Required | Description |
|---------|----------|-------------|
| SUMMARY | Yes | One-line description of purpose |
| PARAMETERS | Yes | Each parameter with [in], [out], or [in,out] tag |
| PRECONDITIONS | Yes | Requirements that must be true before calling |
| POSTCONDITIONS | Yes | Guarantees after successful execution |
| MUTABILITY | If applicable | INPLACE, CONST, or ALLOCATES |
| ALGORITHM | If non-trivial | Step-by-step algorithm description |
| COMPLEXITY | Yes | Time and space complexity |
| THREAD SAFETY | Yes | Safe, Unsafe, or conditional |
| THROWS | If applicable | Exceptions and conditions |
| NUMERICAL NOTES | If applicable | Precision, stability, edge cases |

### SUMMARY

One-line description of what the function does:

```cpp
/* -----------------------------------------------------------------------------
 * FUNCTION: row_norms
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the norm of each row in a CSR sparse matrix.
```

### PARAMETERS

List each parameter with direction tag:

```cpp
 * PARAMETERS:
 *     matrix   [in]  CSR sparse matrix, shape (n_rows, n_cols)
 *     mode     [in]  Norm type: L1, L2, Max, or Sum
 *     output   [out] Pre-allocated buffer, size = n_rows
```

**Direction Tags:**
- `[in]` - Input parameter, not modified
- `[out]` - Output parameter, written to
- `[in,out]` - Modified in-place

### PRECONDITIONS

Requirements that must be true before calling:

```cpp
 * PRECONDITIONS:
 *     - output.size() == matrix.rows()
 *     - matrix must be valid CSR format (sorted indices, no duplicates)
```

### POSTCONDITIONS

Guarantees after successful execution:

```cpp
 * POSTCONDITIONS:
 *     - output[i] contains the norm of row i
 *     - matrix is unchanged
```

### MUTABILITY

For functions that modify state:

```cpp
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
```

**Values:**
- `INPLACE` - Modifies input in-place
- `CONST` - Does not modify inputs
- `ALLOCATES` - Allocates new memory

### ALGORITHM

Step-by-step algorithm description:

```cpp
 * ALGORITHM:
 *     For each row i in parallel:
 *         1. Iterate over non-zero elements in row i
 *         2. Accumulate according to mode:
 *            - L1:  sum(|x_j|)
 *            - L2:  sqrt(sum(x_j^2))
 *            - Max: max(|x_j|)
 *            - Sum: sum(x_j)
 *         3. Write result to output[i]
```

### COMPLEXITY

Time and space complexity:

```cpp
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
```

### THREAD SAFETY

Thread safety guarantees:

```cpp
 * THREAD SAFETY:
 *     Safe - parallelized over rows, no shared mutable state
```

**Values:**
- `Safe` - Thread-safe, can be called concurrently
- `Unsafe` - Not thread-safe, requires external synchronization
- `Conditional` - Thread-safe under specific conditions (describe)

### THROWS

Exceptions that may be thrown:

```cpp
 * THROWS:
 *     DimensionError - if output.size() != matrix.rows()
 *     std::bad_alloc - if memory allocation fails
```

### NUMERICAL NOTES

Precision, stability, edge cases:

```cpp
 * NUMERICAL NOTES:
 *     - Uses epsilon to handle zero/near-zero rows gracefully
 *     - L2 norm uses compensated summation for accuracy
 *     - Max norm returns 0 for empty rows
```

## Inline Comments in Declarations

After the documentation block, include the actual function declaration with inline comments:

```cpp
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type
    Real epsilon = 1e-12           // Zero-norm threshold
);
```

**Guidelines:**
- Brief inline comment for each parameter
- Align comments for readability
- Keep concise (visible in quick queries)

## Quick Query Command

To extract only function signatures (skip documentation):

```bash
sed '/^\/\*/,/\*\/$/d' scl/kernel/normalize.h
```

This allows rapid interface inspection without reading full documentation.

## Language and Formatting Rules

### Plain Text Only

**CRITICAL RULE:** Code comments and documentation MUST use plain text ONLY.

**NO Markdown Syntax:**
- Do NOT use `**bold**` or `*italic*`
- Do NOT use backticks for inline code
- Do NOT use `# headers`
- Do NOT use `- lists` or `* lists`
- Do NOT use code fences

**NO LaTeX Syntax:**
- Do NOT use `$...$` or `$$...$$`
- Do NOT use `\frac{}{}`, `\sum`, `\int`
- Write formulas in plain ASCII: `norm = sqrt(sum(x_i^2))`

**NO Examples in API Docs:**
- Do NOT include usage examples
- Do NOT include sample code
- Describe contracts and behavior precisely instead

### Good vs Bad Examples

**BAD - Markdown:**

```cpp
/* SUMMARY:
 *     Computes the **L2 norm** using the formula:
 *     
 *     $$\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$
 *     
 *     Example:
 *     ```cpp
 *     Real norm = compute_norm(data, n, NormMode::L2);
 *     ```
```

**GOOD - Plain Text:**

```cpp
/* SUMMARY:
 *     Computes the L2 norm using the formula: norm = sqrt(sum(x_i^2))
 *     
 *     For numerical stability, uses Kahan summation when computing
 *     the sum of squares.
```

## Workflow Requirement

**CRITICAL:** After completing any file modification task, you MUST update the corresponding `.h` file.

### Checklist

Before marking task complete:

- [ ] Implementation in `.hpp` file is correct
- [ ] Corresponding `.h` file exists
- [ ] `.h` contains actual function declarations (not just documentation)
- [ ] All function signatures in `.h` match implementation exactly
- [ ] Block comments document all PRECONDITIONS and POSTCONDITIONS
- [ ] Inline comments in declarations are concise and informative
- [ ] MUTABILITY section is correct for any in-place operations
- [ ] Quick query command works: `sed '/^\/\*/,/\*\/$/d' file.h` shows clean signatures
- [ ] No Markdown or LaTeX in any comments
- [ ] No examples in API documentation

## Template for New Functions

### Implementation File (`.hpp`)

```cpp
// scl/module/feature.hpp
#pragma once

#include "scl/module/feature.h"
#include "scl/core/type.hpp"

namespace scl::module::feature {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // Brief comment about algorithm if non-obvious
    for (size_t i = 0; i < n; ++i) {
        output[i] = process(input[i]);
    }
}

} // namespace scl::module::feature
```

### Documentation File (`.h`)

```cpp
// scl/module/feature.h
#pragma once

namespace scl::module::feature {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Brief one-line description.
 *
 * PARAMETERS:
 *     input  [in]  Input array, size n
 *     output [out] Output array, size n (pre-allocated)
 *     n      [in]  Array size
 *
 * PRECONDITIONS:
 *     - input and output must be valid pointers
 *     - output must have space for n elements
 *
 * POSTCONDITIONS:
 *     - output[i] contains processed value of input[i]
 *     - input is unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T>
void my_function(
    const T* input,    // Input array
    T* output,         // Output array
    size_t n           // Array size
);

} // namespace scl::module::feature
```

## Documentation for Different Constructs

### Enums

```cpp
/* -----------------------------------------------------------------------------
 * ENUM: NormMode
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Norm type for normalization operations.
 *
 * VALUES:
 *     L1   - Sum of absolute values: sum(|x_i|)
 *     L2   - Euclidean norm: sqrt(sum(x_i^2))
 *     Max  - Maximum absolute value: max(|x_i|)
 *     Sum  - Simple sum (signed): sum(x_i)
 * -------------------------------------------------------------------------- */
enum class NormMode {
    L1,    // sum(|x_i|)
    L2,    // sqrt(sum(x_i^2))
    Max,   // max(|x_i|)
    Sum    // sum(x_i) - signed
};
```

### Structs

```cpp
/* -----------------------------------------------------------------------------
 * STRUCT: ContiguousArraysT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Contiguous CSR/CSC arrays with registry-managed memory.
 *
 * FIELDS:
 *     data        - Values array (registry registered)
 *     indices     - Column/row indices (registry registered)
 *     indptr      - Row/column offsets (registry registered)
 *     nnz         - Number of non-zeros
 *     primary_dim - Number of rows (CSR) or columns (CSC)
 *
 * LIFETIME:
 *     All pointers are registered with scl::Registry.
 *     Caller must unregister to free memory.
 * -------------------------------------------------------------------------- */
template <typename T>
struct ContiguousArraysT {
    T* data;             // registry registered values array
    Index* indices;      // registry registered indices array
    Index* indptr;       // registry registered offset array
    Index nnz;
    Index primary_dim;
};
```

---

::: tip Consistency
Consistent documentation makes the codebase easier to navigate and maintain. Follow these standards for all public APIs.
:::

