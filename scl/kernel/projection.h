// =============================================================================
// FILE: scl/kernel/projection.h
// BRIEF: API reference for sparse random projection kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.h"
#include "scl/core/sparse.h"

namespace scl::kernel::projection {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

/* -----------------------------------------------------------------------------
 * NAMESPACE: config
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Configuration constants for projection performance tuning.
 *
 * CONSTANTS:
 *     SIMD_THRESHOLD     - Min output_dim for SIMD accumulation (64)
 *     PREFETCH_DISTANCE  - Cache line prefetch distance (16 elements)
 *     SMALL_OUTPUT_DIM   - Threshold for scalar path (32)
 *     DEFAULT_EPSILON    - Default distance distortion for JL dimension (0.1)
 *
 * PERFORMANCE TUNING:
 *     - Large output_dim: SIMD 4-way unrolled accumulation
 *     - Small output_dim: scalar loop to avoid SIMD overhead
 * -------------------------------------------------------------------------- */

namespace config {
    constexpr Size SIMD_THRESHOLD = 64;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size SMALL_OUTPUT_DIM = 32;
    constexpr Real DEFAULT_EPSILON = Real(0.1);
}

// =============================================================================
// SECTION 2: Projection Types
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: ProjectionType
 * -----------------------------------------------------------------------------
 * VALUES:
 *     Gaussian    - Dense Gaussian projection N(0, 1/k)
 *                   Best quality, highest memory usage
 *     Achlioptas  - Sparse {+1, 0, -1} with prob {1/6, 2/3, 1/6}
 *                   Good quality, 3x faster than Gaussian
 *     Sparse      - Very sparse with density 1/sqrt(d)
 *                   Best for high-dimensional data, sqrt(d)x faster
 *     CountSketch - Hash-based sign flips into buckets
 *                   O(nnz) time, unbiased estimator
 *
 * SELECTION GUIDE:
 *     - Gaussian: highest accuracy, small-medium datasets
 *     - Achlioptas: good balance, medium datasets
 *     - Sparse: high-dimensional genomic/text data
 *     - CountSketch: streaming/online applications
 * -------------------------------------------------------------------------- */
enum class ProjectionType {
    Gaussian,       // N(0, 1/k) entries
    Achlioptas,     // {+1, 0, -1} with prob {1/6, 2/3, 1/6}
    Sparse,         // Density = 1/sqrt(d)
    CountSketch     // Hash + sign
};

// =============================================================================
// SECTION 3: Projection Matrix Structure
// =============================================================================

/* -----------------------------------------------------------------------------
 * STRUCT: ProjectionMatrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Pre-computed dense projection matrix R of shape (input_dim x output_dim).
 *     Stored in row-major order for cache-efficient access during projection.
 *
 * MEMBERS:
 *     data       - Flattened matrix data [input_dim * output_dim]
 *     input_dim  - Original dimension d
 *     output_dim - Target dimension k
 *     owns_data  - Whether destructor should free data
 *
 * USAGE:
 *     Use create_*_projection() factory functions to construct.
 *     Use project_with_matrix() to apply projection.
 *
 * MEMORY:
 *     Requires O(d * k) storage. For large d, prefer on-the-fly methods.
 *
 * THREAD SAFETY:
 *     Read-only after construction. Safe for concurrent projection calls.
 * -------------------------------------------------------------------------- */
template <typename T>
struct ProjectionMatrix {
    T* data;           // Flattened d x k matrix (row-major)
    Size input_dim;    // d
    Size output_dim;   // k
    bool owns_data;    // True if destructor should free

    const T* row(Size col_idx) const noexcept;
    bool valid() const noexcept;
};

// =============================================================================
// SECTION 4: Factory Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: create_gaussian_projection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create a dense Gaussian random projection matrix.
 *
 * COMPUTES:
 *     R[i,j] ~ N(0, 1/output_dim)
 *
 * PARAMETERS:
 *     input_dim  [in] Original dimension d (number of features)
 *     output_dim [in] Target dimension k (reduced features)
 *     seed       [in] Random seed for reproducibility (default: 42)
 *
 * RETURNS:
 *     ProjectionMatrix<T> with Gaussian entries scaled by 1/sqrt(k)
 *
 * PRECONDITIONS:
 *     - input_dim > 0
 *     - output_dim > 0
 *
 * POSTCONDITIONS:
 *     - result.data allocated with input_dim * output_dim elements
 *     - result.owns_data = true
 *     - E[||Rx - Ry||^2] = ||x - y||^2 (unbiased)
 *
 * COMPLEXITY:
 *     Time:  O(input_dim * output_dim) for generation
 *     Space: O(input_dim * output_dim)
 *
 * NUMERICAL NOTES:
 *     Uses Splitmix64 PRNG with Box-Muller transform for Gaussian samples.
 *     Deterministic given same seed.
 * -------------------------------------------------------------------------- */
template <typename T>
ProjectionMatrix<T> create_gaussian_projection(
    Size input_dim,    // Original dimension
    Size output_dim,   // Target dimension
    uint64_t seed = 42 // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: create_achlioptas_projection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create a sparse Achlioptas random projection matrix.
 *
 * COMPUTES:
 *     R[i,j] = sqrt(3/k) * X, where X ~ {+1, 0, -1} with prob {1/6, 2/3, 1/6}
 *
 * PARAMETERS:
 *     input_dim  [in] Original dimension d
 *     output_dim [in] Target dimension k
 *     seed       [in] Random seed (default: 42)
 *
 * RETURNS:
 *     ProjectionMatrix<T> with sparse ternary entries
 *
 * ADVANTAGES:
 *     - 3x faster to compute than Gaussian
 *     - 2/3 of entries are zero
 *     - Same theoretical guarantees as Gaussian
 *
 * REFERENCE:
 *     Achlioptas, D. (2003). Database-friendly random projections.
 * -------------------------------------------------------------------------- */
template <typename T>
ProjectionMatrix<T> create_achlioptas_projection(
    Size input_dim,    // Original dimension
    Size output_dim,   // Target dimension
    uint64_t seed = 42 // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: create_sparse_projection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Create a very sparse random projection matrix.
 *
 * COMPUTES:
 *     R[i,j] = sqrt(1/(k*density)) * X
 *     where X ~ {+1, 0, -1} with prob {density/2, 1-density, density/2}
 *
 * PARAMETERS:
 *     input_dim  [in] Original dimension d
 *     output_dim [in] Target dimension k
 *     density    [in] Expected fraction of non-zeros (typically 1/sqrt(d))
 *     seed       [in] Random seed (default: 42)
 *
 * RETURNS:
 *     ProjectionMatrix<T> with very sparse entries
 *
 * ADVANTAGES:
 *     - sqrt(d)x faster for high-dimensional data
 *     - (1-density) fraction of computation skipped
 *     - Same distance preservation guarantees
 *
 * REFERENCE:
 *     Li, P., Hastie, T. J., & Church, K. W. (2006). Very sparse random
 *     projections.
 * -------------------------------------------------------------------------- */
template <typename T>
ProjectionMatrix<T> create_sparse_projection(
    Size input_dim,    // Original dimension
    Size output_dim,   // Target dimension
    Real density,      // Expected sparsity (e.g., 1/sqrt(d))
    uint64_t seed = 42 // Random seed
);

// =============================================================================
// SECTION 5: Pre-computed Matrix Projection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: project_with_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Project sparse matrix X using pre-computed projection matrix R.
 *
 * COMPUTES:
 *     Y = X * R
 *     where X is (n x d) sparse, R is (d x k) dense, Y is (n x k) dense
 *
 * PARAMETERS:
 *     matrix [in]  CSR sparse matrix X, shape (n_rows x n_cols)
 *     proj   [in]  Pre-computed projection matrix R, shape (n_cols x output_dim)
 *     output [out] Dense output buffer Y, size = n_rows * output_dim, PRE-ALLOCATED
 *
 * PRECONDITIONS:
 *     - matrix must be CSR format (IsCSR = true)
 *     - proj.input_dim == matrix.cols()
 *     - output.len >= matrix.rows() * proj.output_dim
 *
 * POSTCONDITIONS:
 *     - output[i*k ... (i+1)*k-1] contains projected row i
 *     - matrix and proj are unchanged
 *
 * MUTABILITY:
 *     WRITES to output buffer
 *
 * ALGORITHM:
 *     Parallel over rows:
 *     1. For each row i:
 *        a. Initialize output_row to zero
 *        b. For each non-zero (j, v) in row i:
 *           - output_row += v * R[j, :]
 *        c. Use SIMD FMA for large output_dim
 *
 * COMPLEXITY:
 *     Time:  O(nnz * output_dim)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows, no shared mutable state
 *
 * PERFORMANCE NOTES:
 *     - SIMD 4-way unrolled accumulation for output_dim >= 64
 *     - Prefetching projection rows for long sparse rows
 *     - Best when projection matrix fits in cache
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project_with_matrix(
    const Sparse<T, IsCSR>& matrix,  // CSR input matrix (n x d)
    const ProjectionMatrix<T>& proj, // Projection matrix (d x k)
    Array<T> output                   // Output buffer [n * k], PRE-ALLOCATED
);

// =============================================================================
// SECTION 6: On-the-Fly Projection (Memory Efficient)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: project_gaussian_otf
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Project sparse matrix using on-the-fly Gaussian random generation.
 *     Memory efficient: no explicit projection matrix stored.
 *
 * COMPUTES:
 *     Y[i, :] = sum_j X[i,j] * R[j, :], where R is generated on-demand
 *
 * PARAMETERS:
 *     matrix     [in]  CSR sparse matrix X
 *     output_dim [in]  Target dimension k
 *     output     [out] Dense output buffer, size = n_rows * output_dim
 *     seed       [in]  Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - matrix must be CSR format
 *     - output.len >= matrix.rows() * output_dim
 *
 * ADVANTAGES:
 *     - O(1) auxiliary memory (no projection matrix storage)
 *     - Deterministic given same seed
 *     - Good for very high-dimensional data
 *
 * DISADVANTAGES:
 *     - Slower than pre-computed for repeated projections
 *     - More random number generation overhead
 *
 * COMPLEXITY:
 *     Time:  O(nnz * output_dim) with higher constant
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project_gaussian_otf(
    const Sparse<T, IsCSR>& matrix, // CSR input matrix
    Size output_dim,                 // Target dimension
    Array<T> output,                 // Output buffer [n * k]
    uint64_t seed = 42               // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: project_achlioptas_otf
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     On-the-fly Achlioptas projection (ternary: +1, 0, -1).
 *
 * PARAMETERS:
 *     matrix     [in]  CSR sparse matrix X
 *     output_dim [in]  Target dimension k
 *     output     [out] Dense output buffer
 *     seed       [in]  Random seed
 *
 * ADVANTAGES:
 *     - Faster than Gaussian OTF (simpler random generation)
 *     - 2/3 of random values are zero (skip computation)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project_achlioptas_otf(
    const Sparse<T, IsCSR>& matrix, // CSR input matrix
    Size output_dim,                 // Target dimension
    Array<T> output,                 // Output buffer [n * k]
    uint64_t seed = 42               // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: project_sparse_otf
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     On-the-fly very sparse projection with custom density.
 *
 * PARAMETERS:
 *     matrix     [in]  CSR sparse matrix X
 *     output_dim [in]  Target dimension k
 *     output     [out] Dense output buffer
 *     density    [in]  Expected density of projection (e.g., 1/sqrt(d))
 *     seed       [in]  Random seed
 *
 * ADVANTAGES:
 *     - Best for very high-dimensional data (d > 10000)
 *     - Most computation skipped due to sparsity
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project_sparse_otf(
    const Sparse<T, IsCSR>& matrix, // CSR input matrix
    Size output_dim,                 // Target dimension
    Array<T> output,                 // Output buffer [n * k]
    Real density,                    // Projection density
    uint64_t seed = 42               // Random seed
);

// =============================================================================
// SECTION 7: Count-Sketch Projection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: project_countsketch
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count-Sketch projection using hash-based bucketing and sign flips.
 *
 * COMPUTES:
 *     For each feature j, hash to bucket h(j) with sign s(j).
 *     Y[i, h(j)] += s(j) * X[i, j]
 *
 * PARAMETERS:
 *     matrix     [in]  CSR sparse matrix X
 *     output_dim [in]  Number of buckets k
 *     output     [out] Dense output buffer
 *     seed       [in]  Random seed for hash functions
 *
 * ADVANTAGES:
 *     - O(nnz) time (not O(nnz * k))
 *     - Unbiased: E[Y^T Y] = X^T X
 *     - Good for streaming/online learning
 *
 * DISADVANTAGES:
 *     - Higher variance than Gaussian for small k
 *     - Collision effects reduce accuracy
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * REFERENCE:
 *     Charikar, M., Chen, K., & Farach-Colton, M. (2004). Finding frequent
 *     items in data streams.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project_countsketch(
    const Sparse<T, IsCSR>& matrix, // CSR input matrix
    Size output_dim,                 // Number of buckets
    Array<T> output,                 // Output buffer [n * k]
    uint64_t seed = 42               // Random seed
);

// =============================================================================
// SECTION 8: High-Level Interface
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: project
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unified interface for sparse random projection.
 *     Automatically selects on-the-fly method based on type.
 *
 * PARAMETERS:
 *     matrix     [in]  CSR sparse matrix X
 *     output_dim [in]  Target dimension k
 *     output     [out] Dense output buffer [n * k]
 *     type       [in]  Projection type (default: Sparse)
 *     seed       [in]  Random seed (default: 42)
 *
 * PRECONDITIONS:
 *     - matrix must be CSR format
 *     - output.len >= matrix.rows() * output_dim
 *
 * POSTCONDITIONS:
 *     - output contains projected data
 *     - Distance preservation: (1-eps)||x-y|| <= ||Rx-Ry|| <= (1+eps)||x-y||
 *       with high probability
 *
 * SELECTION:
 *     - Sparse type uses density = max(1/sqrt(cols), 0.01)
 *
 * THREAD SAFETY:
 *     Safe - parallelized internally
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void project(
    const Sparse<T, IsCSR>& matrix, // CSR input matrix
    Size output_dim,                 // Target dimension
    Array<T> output,                 // Output buffer [n * k]
    ProjectionType type = ProjectionType::Sparse,
    uint64_t seed = 42               // Random seed
);

// =============================================================================
// SECTION 9: Utility Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_jl_dimension
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute minimum target dimension for Johnson-Lindenstrauss guarantee.
 *
 * COMPUTES:
 *     k >= 4 * ln(n) / (epsilon^2/2 - epsilon^3/3)
 *
 * PARAMETERS:
 *     n_samples [in] Number of data points
 *     epsilon   [in] Maximum relative distance distortion (default: 0.1)
 *
 * RETURNS:
 *     Minimum target dimension k for (1 +/- epsilon) distance preservation
 *
 * GUARANTEE:
 *     With probability >= 1 - 1/n^2:
 *     (1-epsilon)||x-y||^2 <= ||Rx-Ry||^2 <= (1+epsilon)||x-y||^2
 *     for all pairs (x, y)
 *
 * TYPICAL VALUES:
 *     n=1000,  eps=0.1: k ~= 300
 *     n=10000, eps=0.1: k ~= 400
 *     n=1000,  eps=0.5: k ~= 20
 *
 * REFERENCE:
 *     Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
 *     mappings into a Hilbert space.
 * -------------------------------------------------------------------------- */
Size compute_jl_dimension(
    Size n_samples,                    // Number of data points
    Real epsilon = config::DEFAULT_EPSILON  // Distance distortion tolerance
);

} // namespace scl::kernel::projection
