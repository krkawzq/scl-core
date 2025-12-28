# AGENT.md - AI Developer Guide for scl-core

This document serves as the authoritative guide for AI agents working on the `scl-core` project. It defines the architectural philosophy, coding standards, and collaborative protocols required to maintain the project's high-performance nature.

**Core Mission**: To build a high-performance, biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.

---

## 1. The Human-AI Collaboration Protocol

We strictly enforce a "Human-in-the-Loop" architecture. Code is not just logic; it is a collaborative artifact.

### 1.1 Documentation Standard

**Documentation Location**:

This project uses a centralized documentation approach:
- `xxx.hpp`: Implementation with minimal inline comments
- `docs/`: Comprehensive API documentation in Markdown format

**Implementation Files (.hpp) - Minimal Comments**:

Implementation files should contain only brief, essential comments:
- One-line function purpose (optional if self-explanatory)
- Non-obvious algorithm tricks or optimizations
- Warning about subtle behavior

Keep implementation clean and readable. All detailed documentation belongs in the `docs/` directory.

```cpp
// GOOD - minimal inline comment
template <CSRLike MatrixT>
void normalize_rows(MatrixT& matrix, NormMode mode, Real eps) {
    // Use Kahan summation for numerical stability
    parallel_for(0, matrix.rows(), [&](Index i) {
        // ... implementation
    });
}

// BAD - too verbose for implementation file
/// @brief Normalizes each row of a sparse matrix to unit norm.
/// @detailed This function computes the norm of each row and divides
/// each element by the norm. Supports L1, L2, Max normalization...
/// (This belongs in docs/)
```

**API Documentation in docs/ - Comprehensive Reference**:

All API documentation is stored in the `docs/` directory using Markdown format. The documentation structure mirrors the codebase organization:
- `docs/cpp/kernels/`: Documentation for kernel modules
- `docs/cpp/core/`: Documentation for core utilities
- `docs/cpp/mmap/`: Documentation for memory-mapped operations
- `docs/api/`: API reference documentation

**Structure**: Each documentation file should contain:
- Function signatures with C++ syntax
- Comprehensive API documentation using the format specified below
- Clear organization matching the codebase structure

### 1.2 API Documentation Format

API documentation in `docs/` uses Markdown format with code blocks for function signatures:

```markdown
# Normalize Kernels

## row_norms

**SUMMARY:**
Compute the norm of each row in a CSR sparse matrix.

**SIGNATURE:**
```cpp
template <CSRLike MatrixT>
void row_norms(
    const MatrixT& matrix,        // CSR matrix input
    NormMode mode,                 // L1, L2, Max, or Sum
    MutableSpan<Real> output       // Output buffer [n_rows]
);
```

**PARAMETERS:**
- matrix [in]  CSR sparse matrix, shape (n_rows, n_cols)
- mode   [in]  Norm type: L1, L2, Max, or Sum
- output [out] Pre-allocated buffer, size = n_rows

**PRECONDITIONS:**
- output.size() == matrix.rows()
- matrix must be valid CSR format (sorted indices, no duplicates)

**POSTCONDITIONS:**
- output[i] contains the norm of row i
- matrix is unchanged

**ALGORITHM:**
For each row i in parallel:
1. Iterate over non-zero elements in row i
2. Accumulate according to mode:
   - L1:  sum(|x_j|)
   - L2:  sqrt(sum(x_j^2))
   - Max: max(|x_j|)
   - Sum: sum(x_j)
3. Write result to output[i]

**COMPLEXITY:**
- Time:  O(nnz)
- Space: O(1) auxiliary

**THREAD SAFETY:**
Safe - parallelized over rows, no shared mutable state

**THROWS:**
DimensionError - if output.size() != matrix.rows()

---

## normalize_rows_inplace

**SUMMARY:**
Normalize each row of a CSR matrix to unit norm in-place.

**SIGNATURE:**
```cpp
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type
    Real epsilon = 1e-12           // Zero-norm threshold
);
```

**PARAMETERS:**
- matrix  [in,out] Mutable CSR matrix, modified in-place
- mode    [in]     Norm type for normalization
- epsilon [in]     Small constant to prevent division by zero

**PRECONDITIONS:**
- matrix must be valid CSR format
- matrix values must be mutable

**POSTCONDITIONS:**
- Each row has unit norm under specified mode (if original norm > epsilon)
- Rows with norm <= epsilon are unchanged
- Matrix structure (indices, indptr) unchanged

**MUTABILITY:**
INPLACE - modifies matrix.values() directly

**ALGORITHM:**
For each row i in parallel:
1. Compute norm of row i
2. If norm > epsilon: divide each element by norm
3. Otherwise: leave row unchanged

**COMPLEXITY:**
- Time:  O(nnz)
- Space: O(1) auxiliary

**NUMERICAL NOTES:**
- Uses epsilon to handle zero/near-zero rows gracefully
- L2 norm uses compensated summation for accuracy

---

## NormMode

**VALUES:**
- L1   - Sum of absolute values: sum(|x_i|)
- L2   - Euclidean norm: sqrt(sum(x_i^2))
- Max  - Maximum absolute value: max(|x_i|)
- Sum  - Simple sum (signed): sum(x_i)

**SIGNATURE:**
```cpp
enum class NormMode {
    L1,    // sum(|x_i|)
    L2,    // sqrt(sum(x_i^2))
    Max,   // max(|x_i|)
    Sum    // sum(x_i) - signed
};
```

### 1.3 API Documentation Sections

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

After the documentation block, include the actual function declaration with inline comments:
- Each parameter should have a brief inline comment explaining its purpose
- Use alignment for readability
- Keep inline comments concise (visible in quick queries)

**IMPORTANT**: API documentation must NOT contain examples. Use precise descriptions of contracts, invariants, and algorithms instead. Examples can introduce ambiguity and become outdated.

### 1.4 Workflow Requirement

**CRITICAL**: After completing any file modification task, you MUST update the corresponding documentation in `docs/` to reflect the changes.

Checklist before marking task complete:
1. Implementation in `.hpp` file is correct
2. Corresponding documentation file exists in `docs/` directory
3. Documentation contains actual function signatures in code blocks
4. All function signatures in documentation match implementation exactly
5. Documentation includes all PRECONDITIONS and POSTCONDITIONS
6. Function signatures are clearly formatted with inline comments
7. MUTABILITY section is correct for any in-place operations
8. Documentation structure matches codebase organization

### 1.5 Language and Formatting Rules

**Language**: English only. All comments and documentation must be in English.

**Format**: Plain Text Only (STRICTLY No Markdown/LaTeX)

**CRITICAL RULE**: Code comments and documentation MUST use plain text ONLY.

* **NO Markdown Syntax**:
  * Do NOT use `**bold**` or `*italic*`
  * Do NOT use backticks for inline code
  * Do NOT use `# headers`
  * Do NOT use `- lists` or `* lists`
  * Do NOT use code fences

* **NO LaTeX Syntax**:
  * Do NOT use `$...$` or `$$...$$`
  * Do NOT use `\frac{}{}`, `\sum`, `\int`
  * Write formulas in plain ASCII: `norm = sqrt(sum(x_i^2))`

* **NO Examples in API Docs**:
  * Do NOT include usage examples
  * Do NOT include sample code
  * Describe contracts and behavior precisely instead

---

## 2. C++ Kernel Architecture: "Zero-Overhead"

The internal C++ layer (`scl/`) is designed for maximum throughput and minimal latency.

### 2.1 Code Style Standards

**Namespace Organization**:

* `scl::core`: Core types, utilities, and infrastructure
* `scl::kernel`: Computational kernels (normalize, mwu, neighbors, etc.)
* `scl::math`: Mathematical functions (regression, stats, etc.)
* `scl::utils`: Utility functions (matrix operations, etc.)
* `scl::threading`: Parallelization abstraction layer
* `scl::binding`: C-ABI interface for Python bindings

**Code Formatting**:

* Use `#pragma once` for header guards
* Section headers use `// =============================================================================`
* Indentation: 4 spaces (no tabs)
* Line length: Prefer < 100 characters, max 120
* Braces: Opening brace on same line

**Attributes and Qualifiers**:

* Use `SCL_FORCE_INLINE` for hot-path functions
* Use `SCL_NODISCARD` for return values that must be checked
* Use `constexpr` for compile-time constants
* Use `noexcept` where appropriate
* Use `SCL_RESTRICT` for non-aliased pointers in hot loops

**Error Handling**:

* Use `SCL_ASSERT` for internal invariants
* Use `SCL_CHECK_ARG` for user input validation
* Use `SCL_CHECK_DIM` for dimension mismatches

## 3. Development Checklist

When completing any task, verify:

- [ ] Implementation is correct and compiles
- [ ] Code uses minimal inline comments
- [ ] Corresponding documentation file is created/updated in `docs/`
- [ ] Documentation contains actual function signatures in code blocks
- [ ] All function signatures in documentation match implementation exactly
- [ ] Function signatures are clearly formatted with inline comments
- [ ] PRECONDITIONS document all input requirements
- [ ] POSTCONDITIONS document all guarantees
- [ ] MUTABILITY is specified for state-changing functions
- [ ] Documentation structure matches codebase organization
- [ ] No examples in API documentation (use precise contract descriptions)
