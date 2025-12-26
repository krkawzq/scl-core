# AGENT.md - AI Developer Guide for scl-core

This document serves as the authoritative guide for AI agents working on the `scl-core` project. It defines the architectural philosophy, coding standards, and collaborative protocols required to maintain the project's high-performance nature.

**Core Mission**: To build a high-performance, biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.

---

## 1. The Human-AI Collaboration Protocol

We strictly enforce a "Human-in-the-Loop" architecture. Code is not just logic; it is a collaborative artifact.

### 1.1 Documentation Standard

**Style**: Doxygen Triple Slash (`///`)

All public APIs, classes, functions, and important internal logic must use Doxygen-style triple-slash comments (`///`). Single-line comments (`//`) are acceptable for brief inline notes.

**Format**: Plain Text Only (STRICTLY No Markdown/LaTeX)

**CRITICAL RULE**: Code comments and documentation MUST use plain text ONLY. Any use of Markdown or LaTeX syntax is STRICTLY FORBIDDEN.

* **NO Markdown Syntax**:
  * Do NOT use `**bold**` or `*italic*` - use UPPERCASE or "quotes" for emphasis
  * Do NOT use `` `code` `` - just write code directly
  * Do NOT use `# headers` - use section labels like "Section:" or separators
  * Do NOT use `- lists` or `* lists` - use numbered lists with plain text (1., 2., 3.) or indent with spaces
  * Do NOT use ```` code blocks ```` - describe code in plain text or show it directly
  * Do NOT use `[links](url)` - write URLs directly

* **NO LaTeX Syntax**:
  * Do NOT use `$...$` or `$$...$$` for mathematical formulas
  * Do NOT use `\frac{}{}`, `\sum`, `\int`, or any LaTeX commands
  * Write formulas in plain text using ASCII characters:
    * Good: `sigma(z)_i = e^(z_i) / sum(e^(z_j))`
    * Good: `gradient = (y - y_pred) / n`
    * Good: `distance = sqrt(sum((x_i - y_i)^2))`
    * Bad: `$\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$`

* **NO Special Tags**:
  * Do NOT include `[Owner: Human]` or `[Owner: AI]` markers in comments
  * Do NOT use `@human`, `@ai`, or similar ownership annotations

**Comment Structure**:

```cpp
// =============================================================================
/// @file filename.hpp
/// @brief Brief one-line description
///
/// Detailed description paragraph explaining the purpose, design, and usage.
///
/// Key Concepts:
///
/// 1. Concept Name: Description
///    - Detail point 1
///    - Detail point 2
///
/// 2. Another Concept: Description
///
/// Use Cases:
///
/// - Use case 1: Description
/// - Use case 2: Description
///
/// Performance:
///
/// - Time complexity: O(...)
/// - Space complexity: O(...)
/// - Throughput: ... operations/sec
// =============================================================================
```

**Function Documentation**:

```cpp
/// @brief One-line summary of function purpose.
///
/// Detailed description of what the function does, its behavior, and any
/// important implementation details.
///
/// @tparam T Template parameter description
/// @param param1 Parameter description
/// @param param2 Parameter description
/// @return Return value description
/// @throws ExceptionType When this condition occurs
```

**Language**: English only. All comments must be in English.

---

## 2. C++ Kernel Architecture: "Zero-Overhead"

The internal C++ layer (`scl/`) is designed for maximum throughput and minimal latency.

### 2.1 Code Style Standards

**Namespace Organization**:

* `scl::core`: Core types, utilities, and infrastructure (type.hpp, matrix.hpp, error.hpp, macros.hpp, memory.hpp, etc.)
* `scl::kernel`: Computational kernels (normalize, mwu, neighbors, etc.)
* `scl::math`: Mathematical functions (regression, stats, etc.)
* `scl::utils`: Utility functions (matrix operations, etc.)
* `scl::threading`: Parallelization abstraction layer
* `scl::binding`: C-ABI interface for Python bindings

**Naming Conventions**:

* **Macros**: All macros prefixed with `SCL_` (e.g., `SCL_FORCE_INLINE`, `SCL_CHECK_DIM`, `SCL_ASSERT`)
* **Types**: PascalCase for classes/structs (e.g., `MemHandle`, `Span`, `CSRMatrix`)
* **Functions**: snake_case (e.g., `log1p_inplace`, `softmax`, `normalize`)
* **Constants**: UPPER_SNAKE_CASE (e.g., `SCL_ALIGNMENT`)
* **Template Parameters**: PascalCase (e.g., `MatrixT`, `ValueType`)

**Code Formatting**:

* Use `#pragma once` for header guards
* Section headers use `// =============================================================================`
* Indentation: 4 spaces (no tabs)
* Line length: Prefer < 100 characters, max 120
* Braces: Opening brace on same line for functions/classes, new line for namespaces

**Attributes and Qualifiers**:

* Use `SCL_FORCE_INLINE` for hot-path functions
* Use `[[nodiscard]]` or `SCL_NODISCARD` for functions whose return values must be checked
* Use `constexpr` for compile-time constants and simple functions
* Use `noexcept` where appropriate (especially move constructors/assignments)
* Use `SCL_RESTRICT` for non-aliased pointers in hot loops

**Error Handling**:

* Use `SCL_ASSERT` for internal invariants (always active, throws `InternalError`)
* Use `SCL_CHECK_ARG` for user input validation (throws `ValueError`)
* Use `SCL_CHECK_DIM` for dimension mismatches (throws `DimensionError`)
* All exceptions inherit from `scl::Exception` with appropriate `ErrorCode`

### 2.2 Memory Management Philosophy

* **No Hidden Allocations**: Kernel functions must operate on pre-allocated memory. Dynamic allocation (heap) inside a compute kernel is **strictly forbidden**.
* **Span-Based API**: Use `Span<T>` and `MutableSpan<T>` instead of raw pointers where possible. These provide bounds checking in debug mode and zero overhead in release.
* **Raw Pointers in Hot Paths**: For maximum performance in inner loops, raw pointers with `SCL_RESTRICT` are acceptable.
* **Container Policy**: Standard Library containers (`std::vector`, `std::map`, `std::string`) are permitted only in:
  * Cold paths (initialization, configuration)
  * Thread-local workspaces (reused across iterations)
  * Non-performance-critical utilities

**Memory Alignment**:

* SIMD operations require aligned memory (typically 64 bytes for AVX-512)
* Use `scl::memory::aligned_alloc<T>()` for aligned buffers
* Use `SCL_ALIGN_AS(N)` for stack-allocated aligned data
* Use `SCL_ASSUME_ALIGNED(ptr, N)` to hint compiler about alignment

### 2.3 Type System and Abstractions

**Core Types**:

* `Real`: Floating-point type (float or double, configured at compile time)
* `Index`: Signed 64-bit integer for array indexing (int64_t)
* `Size`: Unsigned size type (size_t)
* `Byte`: Unsigned 8-bit integer (uint8_t)

**Span Types**:

* `Span<T>`: Non-owning, read-only view of contiguous memory
* `MutableSpan<T>`: Non-owning, mutable view of contiguous memory
* Zero overhead: Compiles to raw pointer + size in release builds

**Matrix Concepts**:

* `CSRLike<M>`: Concept for Compressed Sparse Row matrices
* `CSCLike<M>`: Concept for Compressed Sparse Column matrices
* `DenseLike<M>`: Concept for dense matrices
* `SparseLike<M>`: Union of CSR and CSC concepts

All matrix types must provide:
* `ValueType`: Element type
* `Tag`: Type tag (TagCSR, TagCSC, TagDense)
* `rows`, `cols`: Dimensions
* Row/column accessors: `row_values(i)`, `row_indices(i)`, `row_length(i)` (CSR) or `col_values(j)`, `col_indices(j)`, `col_length(j)` (CSC)

### 2.4 Parallelism: "Backend Agnostic"

* **Abstraction Layer**: Direct usage of `omp.h`, `tbb.h`, or `pthread` in kernels is **prohibited**.
* **Unified Interface**: All parallel loops must use `scl::threading::parallel_for(start, end, lambda)`.
* **Backend Selection**: Build system selects backend (OpenMP, TBB, BS::thread_pool, Serial) via CMake.
* **Thread Safety**: Kernels must be thread-safe. Use thread-local storage for workspaces.

**Parallelization Pattern**:

```cpp
scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
    // Thread-local workspace (if needed)
    thread_local std::vector<T> workspace;
    
    // Process row i
    auto vals = matrix.row_values(static_cast<Index>(i));
    // ... computation ...
});
```

### 2.5 SIMD Usage

* Use `scl::simd` namespace (wraps Google Highway)
* Auto-vectorization preferred: Write clean loops that compiler can vectorize
* Explicit SIMD only when necessary for performance
* Use `SCL_PREFETCH_READ`/`SCL_PREFETCH_WRITE` for memory-bound operations

---

## 3. Binding Strategy: "The C-ABI Firewall"

We reject heavy binding generators (Pybind11/Nanobind) in favor of a manual, stable, and lightweight C-ABI.

### 3.1 The `extern "C"` Interface
* **Stability**: All exported functions must use `extern "C"` linkage to prevent C++ name mangling.
* **Return Type Protocol**: Functions must not return C++ objects or throw exceptions across the boundary.
    * **Success**: Return `nullptr`.
    * **Failure**: Return a pointer to an Error Instance (see Section 3.2).

### 3.2 Exception Containment
* **The Barrier**: Every C-ABI wrapper must contain a top-level `try-catch` block.
* **Error Registry**: C++ exceptions are caught and converted into a generic "Error Instance" (struct with code & message) managed by a global registry.
    * *Mechanism*: Python receives a pointer to this error instance, looks up the corresponding Python exception type in a shared table, and raises it.

---

## 4. Observability: "Pooled Telemetry"

Progress tracking and status reporting must be **asynchronous** and **non-intrusive**.

### 4.1 Decoupled Progress System
* **Concept**: Operators report progress to a side-channel system, independent of the main computation flow.
* **Mechanism**:
    * **Pooling**: Progress slots are managed by a global, pre-allocated pool to avoid allocation during execution.
    * **Macro Access**: Operators request a progress buffer via a dedicated macro (e.g., `SCL_GET_PROGRESS`).
* **Concurrency**:
    * Writing to progress counters must be **lock-free** (e.g., atomic increment or loose non-atomic increment).
    * Precision is secondary to performance. A slightly inaccurate progress bar is better than a stalled computation kernel.

---

## 5. Kernel Design Philosophy

### 5.1 Operator Design Principles

**Functional and Stateless**:

* Kernels are pure functions: No hidden state, no side effects (except output buffers)
* All inputs passed as parameters (no global state)
* Thread-safe by design

**Generic and Type-Agnostic**:

* Use templates with concepts (`CSRLike`, `CSCLike`) for matrix types
* Support both Standard and Virtual matrices through unified interface
* Tag dispatch for compile-time specialization

**Performance-First**:

* Zero-allocation hot paths: Pre-allocate all buffers
* SIMD-friendly: Structure loops for auto-vectorization
* Cache-aware: Process data in cache-friendly chunks
* Bandwidth-optimized: Minimize memory traffic

**Example Kernel Structure**:

```cpp
namespace scl::kernel::module_name {

namespace detail {
    // Internal helpers (not exported)
    template <typename T>
    SCL_FORCE_INLINE void helper_function(...) {
        // Implementation
    }
} // namespace detail

// Public API
template <CSRLike MatrixT>
void public_function(
    const MatrixT& matrix,
    MutableSpan<Real> output
) {
    SCL_CHECK_DIM(output.size == expected_size, "Size mismatch");
    
    scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
        // Process row i
        auto vals = matrix.row_values(static_cast<Index>(i));
        // ... computation ...
    });
}

} // namespace scl::kernel::module_name
```

### 5.2 Input/Output Conventions

**Input Parameters**:

* Matrix inputs: Pass by const reference (`const MatrixT&`)
* Scalar inputs: Pass by value (small types) or const reference (large types)
* Configuration: Use `Span<const T>` for arrays, simple types for scalars

**Output Parameters**:

* Use `MutableSpan<T>` for output buffers (pre-allocated by caller)
* Never allocate output memory inside kernel
* Validate output size matches expected dimensions

**Error Handling**:

* Validate inputs at function entry (use `SCL_CHECK_*` macros)
* Throw appropriate exceptions (`ValueError`, `DimensionError`, etc.)
* Never return error codes (use exceptions)

### 5.3 Performance Optimization Guidelines

**Hot Path Optimization**:

* Use `SCL_FORCE_INLINE` for small, frequently-called functions
* Use `SCL_LIKELY`/`SCL_UNLIKELY` for branch prediction hints
* Avoid virtual functions, use tag dispatch or concepts
* Minimize indirection (prefer direct memory access)

**Memory Optimization**:

* Reuse thread-local workspaces across iterations
* Use streaming stores (`scl::simd::Stream`) for write-only data
* Prefetch data for random access patterns
* Align buffers for SIMD operations

**Parallelization Strategy**:

* Row-level or column-level parallelism (not element-level)
* Chunked parallelism for better load balancing
* Avoid false sharing (pad thread-local data if needed)

## 6. Framework Structure

### 6.1 Directory Organization

```
scl/
├── core/           # Core infrastructure
│   ├── type.hpp    # Fundamental types (Real, Index, Span, etc.)
│   ├── matrix.hpp  # Matrix concepts and base types
│   ├── error.hpp   # Exception system
│   ├── macros.hpp  # Compiler macros and attributes
│   ├── memory.hpp  # Memory allocation primitives
│   ├── simd.hpp    # SIMD abstraction (Highway wrapper)
│   ├── lifetime.hpp # RAII memory handles
│   └── ...
├── kernel/         # Computational kernels
│   ├── normalize.hpp
│   ├── mwu.hpp
│   ├── neighbors.hpp
│   ├── log1p.hpp
│   ├── softmax.hpp
│   └── ...
├── math/           # Mathematical functions
│   ├── regression.hpp
│   ├── stats.hpp
│   └── ...
├── utils/          # Utility functions
│   └── matrix.hpp
├── threading/      # Parallelization abstraction
│   ├── parallel_for.hpp
│   └── scheduler.hpp
└── binding/        # C-ABI interface
    └── c_api.cpp
```

### 6.2 Module Dependencies

**Core Layer** (no dependencies on kernel/math/utils):
* `scl/core/*`: Fundamental types and utilities
* Can be used by any other module

**Kernel Layer** (depends on core):
* `scl/kernel/*`: Computational operators
* Uses core types, matrix concepts, threading, SIMD

**Math Layer** (depends on core):
* `scl/math/*`: Mathematical algorithms
* May use kernel functions for building blocks

**Utils Layer** (depends on core, may use kernel):
* `scl/utils/*`: Helper functions
* Often wraps kernel functions with convenience APIs

**Binding Layer** (depends on all):
* `scl/binding/*`: C-ABI exports
* Wraps kernel/math functions for Python/other languages

### 6.3 Include Order

Standard include order within a file:

1. `#pragma once`
2. Project headers (grouped by layer):
   * Core headers (`scl/core/*`)
   * Kernel headers (`scl/kernel/*`)
   * Math headers (`scl/math/*`)
   * Utils headers (`scl/utils/*`)
3. Standard library headers (`<vector>`, `<algorithm>`, etc.)
4. System headers (if needed)

Example:

```cpp
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <algorithm>
```

## 7. Development Mindset

When generating code for `scl-core`, assume the role of a **Systems Engineer**:

1.  **Trust the Compiler**: Write clean, loop-friendly code that auto-vectorizes well. Use explicit SIMD only when necessary.
2.  **Respect the Boundary**: Keep C++ pure; keep Python high-level. The C-ABI layer is the only bridge.
3.  **Defensive on Interfaces, Aggressive on Internals**: Be strict about pointer validity at the API entry point, but assume valid data inside the hot loops to avoid redundant checks.
4.  **Zero Overhead Abstractions**: Use concepts, templates, and tag dispatch to provide type safety without runtime cost.
5.  **Performance First**: Optimize for throughput and latency. Profile before optimizing, but design with performance in mind.

---
