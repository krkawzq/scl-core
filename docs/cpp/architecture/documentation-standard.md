# Documentation Standard

SCL-Core uses a dual-file documentation system that strictly separates implementation from specification. This separation keeps code readable while providing comprehensive API documentation.

## Philosophy

**Clean Code + Comprehensive Docs = Maintainable Codebase**

- Implementation files (.hpp) should be readable without extensive comments
- API documentation files (.h) should be complete specifications
- Documentation is machine-parsable and human-readable
- Quick queries should extract interfaces without noise

## Dual-File System

### Implementation Files (.hpp)

Implementation files contain actual code with minimal inline comments. The code itself should be self-documenting through clear naming and structure.

**File Structure:**

```cpp
// =============================================================================
// FILE: scl/kernel/normalize.hpp
// BRIEF: Row/column normalization kernels
// =============================================================================
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
        
        Real norm = compute_norm(vals.ptr, len, mode);
        
        if (norm > eps) {
            const Real inv_norm = Real(1) / norm;
            for (Index j = 0; j < len; ++j) {
                vals.ptr[j] *= inv_norm;
            }
        }
    });
}

template <CSRLike MatrixT>
void row_norms(const MatrixT& matrix, NormMode mode, MutableSpan<Real> output) {
    // Parallel computation of row norms
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        auto vals = matrix.primary_values(i);
        const Index len = matrix.primary_length(i);
        output[i] = (len > 0) ? compute_norm(vals.ptr, len, mode) : Real(0);
    });
}

} // namespace scl::kernel::normalize
```

**Comment Guidelines:**

Only include comments for:

1. **Non-obvious algorithmic choices:**
   ```cpp
   // Use Kahan summation for numerical stability
   ```

2. **Performance-critical decisions:**
   ```cpp
   // 4-way unrolling for FMA pipeline utilization
   ```

3. **Subtle correctness issues:**
   ```cpp
   // Must check length > 0 to avoid division by zero
   ```

**Do NOT include:**
- Function purpose (obvious from name and signature)
- Parameter descriptions (documented in .h file)
- Algorithm explanation (belongs in .h file)
- Usage examples (not in implementation)

### API Documentation Files (.h)

Documentation files use C++ syntax with comprehensive block comments. These files are NOT compiled - the .h extension enables syntax highlighting while avoiding template instantiation issues.

**Why .h extension?**
- Enables C++ syntax highlighting in all editors
- Avoids accidental inclusion in builds
- Distinguishes documentation from implementation
- Convention: .h = docs only, .hpp = implementation

**File Structure:**

```cpp
// =============================================================================
// FILE: scl/kernel/normalize.h
// BRIEF: API reference for normalization kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
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
 *     - epsilon > 0
 *
 * POSTCONDITIONS:
 *     - Each row with norm > epsilon has unit norm under specified mode
 *     - Rows with norm <= epsilon are unchanged
 *     - Matrix structure (indices, indptr) unchanged
 *     - No memory allocation
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row i in parallel:
 *         1. Extract row values and length
 *         2. Skip if length == 0
 *         3. Compute norm using specified mode:
 *            - L1:  sum(|x_j|)
 *            - L2:  sqrt(sum(x_j^2))
 *            - Max: max(|x_j|)
 *         4. If norm > epsilon:
 *            - Compute inv_norm = 1 / norm
 *            - Multiply each element by inv_norm
 *         5. Otherwise: leave row unchanged
 *
 * COMPLEXITY:
 *     Time:  O(nnz) where nnz = number of non-zero elements
 *     Space: O(1) auxiliary
 *
 * NUMERICAL NOTES:
 *     - Uses epsilon to handle zero/near-zero rows gracefully
 *     - L2 norm uses compensated summation for improved accuracy
 *     - Division by norm is transformed to multiplication by reciprocal
 *     - Empty rows are handled without branching in main loop
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with no shared mutable state
 *
 * PERFORMANCE:
 *     - Automatically parallelizes for large matrices
 *     - SIMD-optimized norm computation
 *     - Cache-friendly row-wise access pattern
 *     - Typical throughput: 5-10 GB/s memory bandwidth
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type: L1, L2, or Max
    Real epsilon = 1e-12           // Zero-norm threshold (default: 1e-12)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: row_norms
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the norm of each row in a CSR sparse matrix.
 *
 * PARAMETERS:
 *     matrix   [in]  CSR sparse matrix, shape (n_rows, n_cols)
 *     mode     [in]  Norm type: L1, L2, Max, or Sum
 *     output   [out] Pre-allocated buffer, size = n_rows
 *
 * PRECONDITIONS:
 *     - matrix must be valid CSR format (sorted indices, no duplicates)
 *     - output.size() == matrix.rows()
 *     - output must be writable
 *
 * POSTCONDITIONS:
 *     - output[i] contains the norm of row i
 *     - output[i] == 0 for empty rows
 *     - matrix is unchanged (const operation)
 *
 * ALGORITHM:
 *     For each row i in parallel:
 *         1. Extract row values and length
 *         2. If length == 0: output[i] = 0
 *         3. Otherwise: output[i] = compute_norm(values, length, mode)
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - each thread writes to disjoint output locations
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void row_norms(
    const MatrixT& matrix,         // CSR matrix input (read-only)
    NormMode mode,                 // Norm type: L1, L2, Max, or Sum
    MutableSpan<Real> output       // Output buffer [n_rows]
);

/* -----------------------------------------------------------------------------
 * ENUM: NormMode
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Norm type for normalization operations.
 *
 * VALUES:
 *     L1   - Manhattan norm: sum(|x_i|)
 *     L2   - Euclidean norm: sqrt(sum(x_i^2))
 *     Max  - Maximum absolute value: max(|x_i|)
 *     Sum  - Simple sum (signed): sum(x_i)
 *
 * NOTES:
 *     - L1: Robust to outliers, encourages sparsity
 *     - L2: Standard Euclidean distance, sensitive to outliers
 *     - Max: Extremely robust, useful for stability analysis
 *     - Sum: Not a true norm (not always positive), use for centering
 * -------------------------------------------------------------------------- */
enum class NormMode {
    L1,    // sum(|x_i|)
    L2,    // sqrt(sum(x_i^2))
    Max,   // max(|x_i|)
    Sum    // sum(x_i) - signed, not a true norm
};

} // namespace scl::kernel::normalize
```

## Documentation Sections

Each function documentation block MUST include these sections in order:

### Required Sections

| Section | Purpose | Always Required |
|---------|---------|-----------------|
| SUMMARY | One-line purpose statement | Yes |
| PARAMETERS | Parameter list with direction tags | Yes |
| PRECONDITIONS | Requirements before calling | Yes |
| POSTCONDITIONS | Guarantees after execution | Yes |
| COMPLEXITY | Time and space analysis | Yes |
| THREAD SAFETY | Concurrency guarantees | Yes |

### Conditional Sections

| Section | Purpose | Required When |
|---------|---------|---------------|
| MUTABILITY | State modification type | Function modifies input |
| ALGORITHM | Step-by-step description | Non-trivial algorithm |
| THROWS | Exception specifications | Function can throw |
| NUMERICAL NOTES | Precision/stability notes | Numerical computation |
| PERFORMANCE | Performance characteristics | Performance-critical |
| RELATED | Related functions | Part of function family |

## Section Specifications

### SUMMARY

Single-line description of function purpose. Should answer "What does this function do?"

```cpp
/* -----------------------------------------------------------------------------
 * FUNCTION: compute_pca
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute principal component analysis on sparse data matrix.
```

**Guidelines:**
- Start with verb (Compute, Calculate, Normalize, etc.)
- Be specific (not "Process data" but "Normalize rows to unit L2 norm")
- Mention key constraints (sparse, in-place, parallel, etc.)
- Keep to one line

### PARAMETERS

List each parameter with direction tag and description:

```cpp
 * PARAMETERS:
 *     matrix   [in]     Input sparse matrix, shape (n_samples, n_features)
 *     n_comps  [in]     Number of components to compute (1 <= n_comps <= min(n_samples, n_features))
 *     output   [out]    Pre-allocated output buffer, shape (n_samples, n_comps)
 *     scratch  [in,out] Workspace buffer, modified during computation
```

**Direction Tags:**
- `[in]` - Input parameter, read-only, not modified
- `[out]` - Output parameter, write-only, caller must allocate
- `[in,out]` - Modified in-place, must be valid on entry

**Guidelines:**
- Align tags and descriptions for readability
- Include dimensions for arrays (shape, size)
- Specify range constraints (0 < x < 1, etc.)
- Mention allocation requirements (pre-allocated, etc.)

### PRECONDITIONS

Requirements that MUST be true before calling. Violation indicates caller error.

```cpp
 * PRECONDITIONS:
 *     - matrix must be valid CSR format (sorted indices, no duplicates)
 *     - output.size() == matrix.rows() * n_comps
 *     - scratch.size() >= required_workspace_size(matrix, n_comps)
 *     - n_comps > 0 and n_comps <= min(matrix.rows(), matrix.cols())
 *     - matrix must contain at least n_comps linearly independent rows
```

**Guidelines:**
- List all validity requirements
- Include size/dimension constraints
- Specify format requirements (sorted, unique, etc.)
- Mention mathematical constraints (positive definite, etc.)
- Order from most to least important

### POSTCONDITIONS

Guarantees that ARE true after successful execution.

```cpp
 * POSTCONDITIONS:
 *     - output contains first n_comps principal components
 *     - Components are orthonormal (dot(output[i], output[j]) = delta_ij)
 *     - Components are sorted by decreasing explained variance
 *     - matrix is unchanged (const operation)
 *     - scratch may be modified but remains valid for reuse
```

**Guidelines:**
- State what is computed/written
- Specify invariants maintained
- Mention side effects on inputs
- Include mathematical properties when relevant

### MUTABILITY

Classify how function affects program state:

```cpp
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly, no allocation
```

**Values:**
- `CONST` - No modification, purely computational
- `INPLACE` - Modifies input(s) in-place
- `ALLOCATES` - Allocates new memory (specify what)
- `MIXED` - Some inputs modified, some const (specify which)

### ALGORITHM

Step-by-step description of algorithm. Include enough detail to understand complexity and correctness.

```cpp
 * ALGORITHM:
 *     Phase 1: Compute row-wise means
 *         For each row i in parallel:
 *             1. Sum all non-zero elements
 *             2. Divide by number of non-zeros
 *             3. Store in means[i]
 *     
 *     Phase 2: Center the matrix
 *         For each row i in parallel:
 *             For each non-zero element (i, j):
 *                 matrix[i,j] -= means[i]
 *     
 *     Phase 3: Compute covariance matrix
 *         result = matrix^T * matrix / (n_samples - 1)
 *     
 *     Phase 4: Eigen-decomposition
 *         Compute eigenvectors of covariance matrix using power iteration
```

**Guidelines:**
- Break into logical phases
- Specify parallelization strategy
- Mention key optimization techniques
- Include termination conditions for iterative algorithms

### COMPLEXITY

Time and space complexity using Big-O notation:

```cpp
 * COMPLEXITY:
 *     Time:  O(nnz * n_comps * n_iter) where n_iter is convergence iterations
 *     Space: O(n_features * n_comps) auxiliary
```

**Guidelines:**
- Express in terms of input dimensions
- Define variables used (nnz, n_features, etc.)
- Specify best/average/worst case if significantly different
- Include auxiliary space (excludes input/output)

### THREAD SAFETY

Thread safety guarantees for concurrent execution:

```cpp
 * THREAD SAFETY:
 *     Safe - parallelized internally with per-thread workspaces
```

**Values:**
- `Safe` - Can be called concurrently from multiple threads
- `Unsafe` - Requires external synchronization
- `Conditional` - Safe under specific conditions (describe them)

**Examples:**

```cpp
 * THREAD SAFETY:
 *     Conditional - safe if different threads operate on different matrices
```

```cpp
 * THREAD SAFETY:
 *     Unsafe - modifies global registry, requires external locking
```

### THROWS

Exceptions that may be thrown:

```cpp
 * THROWS:
 *     DimensionError - if output.size() != matrix.rows() * n_comps
 *     ConvergenceError - if algorithm fails to converge within max_iter iterations
 *     std::bad_alloc - if workspace allocation fails
```

**Guidelines:**
- List exception type and condition
- Order by likelihood (most common first)
- Include standard exceptions (bad_alloc, etc.)

### NUMERICAL NOTES

Precision, stability, and edge case handling:

```cpp
 * NUMERICAL NOTES:
 *     - Uses compensated summation for improved accuracy with large n
 *     - Relative error bounded by O(epsilon * sqrt(n)) for L2 norm
 *     - Division by near-zero norm replaced by zero to avoid infinities
 *     - Empty rows yield norm = 0 without error
 *     - Subnormal numbers are flushed to zero for performance
```

**Guidelines:**
- Mention numerical algorithms used (Kahan sum, etc.)
- Specify error bounds when known
- Describe handling of edge cases (zero, infinity, NaN)
- Note any loss of precision

### PERFORMANCE

Expected performance characteristics:

```cpp
 * PERFORMANCE:
 *     - Automatically parallelizes for matrices with >10,000 rows
 *     - SIMD-optimized with 4-way accumulator pattern
 *     - Typical throughput: 8-15 GB/s memory bandwidth on modern CPUs
 *     - Scales linearly with number of cores up to memory bandwidth limit
 *     - Cache-friendly: processes rows sequentially
```

**Guidelines:**
- Specify parallelization behavior
- Mention SIMD optimizations
- Give throughput numbers when available
- Describe scaling characteristics

## Inline Comments in Declarations

After the documentation block, include the actual function declaration with aligned inline comments:

```cpp
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type: L1, L2, or Max
    Real epsilon = 1e-12           // Zero-norm threshold (default: 1e-12)
);
```

**Guidelines:**
- Brief comment for each parameter (5-10 words)
- Align comments vertically for readability
- Include default value in comment
- Mention key constraints (read-only, pre-allocated, etc.)

These inline comments are visible in quick queries and IDE tooltips.

## Quick Query Command

Extract only function signatures without documentation:

```bash
sed '/^\/\*/,/\*\/$/d' scl/kernel/normalize.h
```

**Output:**

```cpp
#pragma once

namespace scl::kernel::normalize {

template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,
    NormMode mode,
    Real epsilon = 1e-12
);

template <CSRLike MatrixT>
void row_norms(
    const MatrixT& matrix,
    NormMode mode,
    MutableSpan<Real> output
);

enum class NormMode {
    L1, L2, Max, Sum
};

} // namespace scl::kernel::normalize
```

This allows rapid interface inspection for API users.

## Language and Formatting Rules

### Strict Plain Text Only

**CRITICAL RULE:** Documentation MUST use plain text only. No markup languages.

**Forbidden Syntax:**

❌ **No Markdown:**
```cpp
/* SUMMARY:
 *     Computes the **L2 norm** using `sqrt(sum(x^2))`.
 *     
 *     - First step
 *     - Second step
 */
```

❌ **No LaTeX:**
```cpp
/* SUMMARY:
 *     Computes the norm: $\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$
 */
```

❌ **No Examples:**
```cpp
/* SUMMARY:
 *     Normalizes rows.
 *     
 *     Example:
 *         normalize_rows_inplace(matrix, NormMode::L2);
 */
```

**Correct Plain Text:**

✅ **Good:**
```cpp
/* SUMMARY:
 *     Computes the L2 norm using formula: norm = sqrt(sum(x_i^2))
 *     
 * ALGORITHM:
 *     1. Compute sum of squares using compensated summation
 *     2. Take square root of result
 *     3. Handle zero case: return 0 if sum < epsilon^2
 */
```

**Rationale:**
- Markdown breaks structured parsing tools
- LaTeX is unreadable in plain editors
- Examples become outdated and add maintenance burden
- Plain text is universal and unambiguous

### Formulas in Plain Text

Use ASCII notation for mathematical expressions:

| Concept | Plain Text |
|---------|------------|
| L1 norm | `sum(\|x_i\|)` |
| L2 norm | `sqrt(sum(x_i^2))` |
| Dot product | `sum(a_i * b_i)` |
| Matrix multiply | `C[i,j] = sum_k A[i,k] * B[k,j]` |
| Summation | `sum_{i=1}^{n} x_i` |
| Argmax | `argmax_i f(x_i)` |

## Documentation for Different Constructs

### Functions

Standard template shown above.

### Enums

```cpp
/* -----------------------------------------------------------------------------
 * ENUM: AllocType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Memory allocation type for proper cleanup.
 *
 * VALUES:
 *     ArrayNew      - Allocated with new[], cleanup with delete[]
 *     ScalarNew     - Allocated with new, cleanup with delete
 *     AlignedAlloc  - Allocated with aligned_alloc, cleanup with free
 *     Custom        - Custom deleter function provided
 *
 * USAGE:
 *     Passed to Registry to specify how memory should be freed.
 * -------------------------------------------------------------------------- */
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → free
    Custom         // User-provided deleter
};
```

### Structs/Classes

```cpp
/* -----------------------------------------------------------------------------
 * STRUCT: ContiguousArraysT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Contiguous CSR/CSC arrays with registry-managed memory.
 *
 * FIELDS:
 *     data        - Non-zero values, size = nnz
 *     indices     - Column/row indices, size = nnz
 *     indptr      - Row/column offsets, size = primary_dim + 1
 *     nnz         - Number of non-zero elements
 *     primary_dim - Number of rows (CSR) or columns (CSC)
 *
 * INVARIANTS:
 *     - indptr[0] == 0
 *     - indptr[primary_dim] == nnz
 *     - indptr is monotonically increasing
 *     - All pointers registered with scl::Registry
 *
 * LIFETIME:
 *     Caller must unregister pointers to free memory.
 * -------------------------------------------------------------------------- */
template <typename T>
struct ContiguousArraysT {
    T* data;             // Values array [nnz]
    Index* indices;      // Indices array [nnz]
    Index* indptr;       // Offset array [primary_dim + 1]
    Index nnz;           // Number of non-zeros
    Index primary_dim;   // Rows (CSR) or columns (CSC)
};
```

### Type Aliases

```cpp
/* -----------------------------------------------------------------------------
 * ALIAS: Real
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Floating-point type for numerical computations.
 *
 * CONFIGURATION:
 *     - SCL_REAL_FLOAT32: float (32-bit, default)
 *     - SCL_REAL_FLOAT64: double (64-bit)
 *     - SCL_REAL_FLOAT16: _Float16 (16-bit, experimental)
 *
 * NOTES:
 *     Type is configured at compile time. All arithmetic uses this type.
 * -------------------------------------------------------------------------- */
using Real = /* float | double | _Float16 */;
```

## Workflow Requirements

**CRITICAL:** After modifying any .hpp file, update the corresponding .h file.

### Checklist

Before marking task complete:

- [ ] Implementation in .hpp is correct and compiles
- [ ] Code uses minimal inline comments
- [ ] Corresponding .h file exists
- [ ] .h contains actual function declarations (not just docs)
- [ ] All function signatures in .h match implementation exactly
- [ ] Inline comments in .h declarations are concise and aligned
- [ ] Block comments include all required sections
- [ ] PRECONDITIONS document all input requirements
- [ ] POSTCONDITIONS document all guarantees
- [ ] MUTABILITY is specified correctly
- [ ] No Markdown, LaTeX, or examples in documentation
- [ ] Quick query works: `sed '/^\/\*/,/\*\/$/d' file.h` shows clean signatures

## Template for New Functions

### Implementation (.hpp)

```cpp
// scl/module/feature.hpp
#pragma once

#include "scl/module/feature.h"
#include "scl/core/type.hpp"

namespace scl::module::feature {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // Brief comment only if algorithm is non-obvious
    for (size_t i = 0; i < n; ++i) {
        output[i] = process(input[i]);
    }
}

} // namespace scl::module::feature
```

### Documentation (.h)

```cpp
// scl/module/feature.h
#pragma once

namespace scl::module::feature {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Brief one-line description of what function does.
 *
 * PARAMETERS:
 *     input  [in]  Input array, size n
 *     output [out] Output array, size n (pre-allocated)
 *     n      [in]  Array size
 *
 * PRECONDITIONS:
 *     - input and output are valid pointers
 *     - output has space for n elements
 *     - n > 0
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
    const T* input,    // Input array [n]
    T* output,         // Output array [n] (pre-allocated)
    size_t n           // Array size
);

} // namespace scl::module::feature
```

---

::: tip Documentation as Contract
API documentation is a contract between implementation and users. Preconditions define what users must ensure, postconditions define what implementation guarantees. This contract enables reasoning about correctness without reading implementation.
:::
