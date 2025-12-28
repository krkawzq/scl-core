// =============================================================================
// FILE: scl/kernel/hotspot.h
// BRIEF: API reference for spatial statistics and hotspot detection kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::hotspot {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_PERMUTATIONS = 999;
    constexpr Real DEFAULT_ALPHA = Real(0.05);
    constexpr Size PARALLEL_THRESHOLD = 32;
}

// =============================================================================
// Enumerations
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: LISAPattern
 * -----------------------------------------------------------------------------
 * VALUES:
 *     NotSignificant - p-value > alpha, no significant pattern
 *     HighHigh       - High value surrounded by high values (hot spot)
 *     LowLow         - Low value surrounded by low values (cold spot)
 *     HighLow        - High value surrounded by low values (spatial outlier)
 *     LowHigh        - Low value surrounded by high values (spatial outlier)
 * -------------------------------------------------------------------------- */
enum class LISAPattern : int8_t {
    NotSignificant = 0,
    HighHigh = 1,
    LowLow = 2,
    HighLow = 3,
    LowHigh = 4
};

// =============================================================================
// Core Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: local_morans_i
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Local Moran's I statistic for each observation.
 *
 * PARAMETERS:
 *     weights        [in]  Spatial weight matrix (CSR/CSC sparse)
 *     values         [in]  Attribute values array
 *     n              [in]  Number of observations
 *     local_I        [out] Local Moran's I values [n]
 *     p_values       [out] Pseudo p-values from permutation test [n]
 *     n_permutations [in]  Number of permutations (default: 999)
 *     seed           [in]  Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - local_I and p_values must be pre-allocated with n elements
 *     - n_permutations > 0
 *
 * POSTCONDITIONS:
 *     - local_I[i] = z[i] * sum_j(w_ij * z[j]) where z is standardized values
 *     - p_values[i] = proportion of permuted I >= observed I
 *     - Positive I indicates spatial clustering, negative indicates dispersion
 *
 * ALGORITHM:
 *     For each observation i in parallel:
 *         1. Compute spatial lag: lag[i] = sum_j(w_ij * z[j])
 *         2. Compute local I: I[i] = z[i] * lag[i]
 *         3. Permutation test (if n_permutations > 0):
 *            - Shuffle neighbors and recompute I
 *            - Count extreme values for pseudo p-value
 *
 * COMPLEXITY:
 *     Time:  O(n * n_permutations * avg_neighbors)
 *     Space: O(n_threads * n) for permutation buffers
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local RNG and buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void local_morans_i(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix (n x n)
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* local_I,                     // Output: Local Moran's I [n]
    Real* p_values,                    // Output: Pseudo p-values [n]
    Index n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: getis_ord_g_star
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Getis-Ord Gi* statistic for hotspot detection.
 *
 * PARAMETERS:
 *     weights      [in]  Spatial weight matrix (CSR/CSC sparse)
 *     values       [in]  Attribute values array
 *     n            [in]  Number of observations
 *     g_star       [out] Gi* z-scores [n]
 *     p_values     [out] P-values [n]
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - g_star and p_values must be pre-allocated with n elements
 *     - values should be non-negative for meaningful interpretation
 *
 * POSTCONDITIONS:
 *     - g_star[i] = z-score for local concentration
 *     - p_values[i] = two-tailed p-value under normal assumption
 *     - High positive g_star indicates hot spot (high value cluster)
 *     - High negative g_star indicates cold spot (low value cluster)
 *
 * ALGORITHM:
 *     For each observation i in parallel:
 *         1. Compute weighted local sum: local_sum = sum_j(w_ij * x[j])
 *         2. Compute expected value under null
 *         3. Compute variance under null
 *         4. Gi* = (local_sum - expected) / sqrt(variance)
 *         5. p-value = 2 * (1 - normal_cdf(|Gi*|))
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over observations
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void getis_ord_g_star(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix (n x n)
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* g_star,                      // Output: Gi* z-scores [n]
    Real* p_values                     // Output: P-values [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: classify_lisa_patterns
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Classify observations into LISA pattern categories.
 *
 * PARAMETERS:
 *     z_values   [in]  Standardized attribute values [n]
 *     spatial_lag [in]  Spatial lag of standardized values [n]
 *     p_values   [in]  Significance p-values [n]
 *     n          [in]  Number of observations
 *     patterns   [out] LISA pattern classification [n]
 *     alpha      [in]  Significance level (default: 0.05)
 *
 * PRECONDITIONS:
 *     - All input arrays must have length >= n
 *     - patterns must be pre-allocated with n elements
 *
 * POSTCONDITIONS:
 *     - patterns[i] = LISAPattern based on quadrant and significance:
 *       - HighHigh if z > 0 and lag > 0 and p < alpha
 *       - LowLow if z < 0 and lag < 0 and p < alpha
 *       - HighLow if z > 0 and lag < 0 and p < alpha
 *       - LowHigh if z < 0 and lag > 0 and p < alpha
 *       - NotSignificant otherwise
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void classify_lisa_patterns(
    const Real* z_values,              // Standardized values [n]
    const Real* spatial_lag,           // Spatial lag [n]
    const Real* p_values,              // P-values [n]
    Index n,                           // Number of observations
    LISAPattern* patterns,             // Output: LISA patterns [n]
    Real alpha = config::DEFAULT_ALPHA
);

/* -----------------------------------------------------------------------------
 * FUNCTION: identify_hotspots
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify statistically significant hot spots and cold spots.
 *
 * PARAMETERS:
 *     g_star       [in]  Gi* z-scores [n]
 *     p_values     [in]  P-values [n]
 *     n            [in]  Number of observations
 *     is_hotspot   [out] Boolean: true if hot spot [n]
 *     is_coldspot  [out] Boolean: true if cold spot [n]
 *     alpha        [in]  Significance level
 *
 * PRECONDITIONS:
 *     - All arrays must have length >= n
 *     - is_hotspot and is_coldspot must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - is_hotspot[i] = true if p < alpha and g_star > 0
 *     - is_coldspot[i] = true if p < alpha and g_star < 0
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void identify_hotspots(
    const Real* g_star,                // Gi* z-scores [n]
    const Real* p_values,              // P-values [n]
    Index n,                           // Number of observations
    bool* is_hotspot,                  // Output: Hot spot flags [n]
    bool* is_coldspot,                 // Output: Cold spot flags [n]
    Real alpha = config::DEFAULT_ALPHA
);

/* -----------------------------------------------------------------------------
 * FUNCTION: local_gearys_c
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Local Geary's C statistic for each observation.
 *
 * PARAMETERS:
 *     weights        [in]  Spatial weight matrix (CSR/CSC sparse)
 *     values         [in]  Attribute values array
 *     n              [in]  Number of observations
 *     local_C        [out] Local Geary's C values [n]
 *     p_values       [out] Pseudo p-values [n]
 *     n_permutations [in]  Number of permutations
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - local_C and p_values must be pre-allocated with n elements
 *
 * POSTCONDITIONS:
 *     - local_C[i] = sum_j(w_ij * (x[i] - x[j])^2) / variance
 *     - p_values[i] from permutation test
 *     - Small C indicates positive spatial association (clustering)
 *     - Large C indicates negative spatial association (dispersion)
 *
 * COMPLEXITY:
 *     Time:  O(n * n_permutations * avg_neighbors)
 *     Space: O(n_threads * n) for permutation buffers
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void local_gearys_c(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix (n x n)
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* local_C,                     // Output: Local Geary's C [n]
    Real* p_values,                    // Output: Pseudo p-values [n]
    Index n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: global_morans_i
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Global Moran's I statistic with inference.
 *
 * PARAMETERS:
 *     weights        [in]  Spatial weight matrix (CSR/CSC sparse)
 *     values         [in]  Attribute values array
 *     n              [in]  Number of observations
 *     global_I       [out] Global Moran's I value (single scalar)
 *     expected_I     [out] Expected I under null hypothesis
 *     variance_I     [out] Variance of I under null
 *     z_score        [out] Z-score for significance
 *     p_value        [out] P-value (two-tailed, permutation-based)
 *     n_permutations [in]  Number of permutations
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - All output pointers must be valid
 *
 * POSTCONDITIONS:
 *     - global_I = overall spatial autocorrelation measure
 *     - I > 0 indicates positive spatial autocorrelation (clustering)
 *     - I < 0 indicates negative spatial autocorrelation (dispersion)
 *     - I ~ 0 indicates random spatial pattern
 *
 * COMPLEXITY:
 *     Time:  O(n * n_permutations * avg_neighbors)
 *     Space: O(n_threads * n)
 *
 * THREAD SAFETY:
 *     Safe - parallelized permutation tests
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void global_morans_i(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* global_I,                    // Output: Global Moran's I
    Real* expected_I,                  // Output: Expected I
    Real* variance_I,                  // Output: Variance of I
    Real* z_score,                     // Output: Z-score
    Real* p_value,                     // Output: P-value
    Index n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: global_gearys_c
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Global Geary's C statistic with inference.
 *
 * PARAMETERS:
 *     weights      [in]  Spatial weight matrix (CSR/CSC sparse)
 *     values       [in]  Attribute values array
 *     n            [in]  Number of observations
 *     global_C     [out] Global Geary's C value
 *     expected_C   [out] Expected C under null (= 1)
 *     variance_C   [out] Variance of C under null
 *     z_score      [out] Z-score for significance
 *     p_value      [out] P-value (two-tailed)
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - All output pointers must be valid
 *
 * POSTCONDITIONS:
 *     - global_C = overall spatial autocorrelation (inverse of Moran's I)
 *     - C < 1 indicates positive spatial autocorrelation
 *     - C > 1 indicates negative spatial autocorrelation
 *     - C ~ 1 indicates random spatial pattern
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void global_gearys_c(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* global_C,                    // Output: Global Geary's C
    Real* expected_C,                  // Output: Expected C
    Real* variance_C,                  // Output: Variance of C
    Real* z_score,                     // Output: Z-score
    Real* p_value                      // Output: P-value
);

// =============================================================================
// Multiple Testing Correction
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: benjamini_hochberg_correction
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Benjamini-Hochberg FDR correction to p-values.
 *
 * PARAMETERS:
 *     p_values  [in]  Original p-values array
 *     n         [in]  Number of p-values
 *     q_values  [out] Adjusted p-values (q-values)
 *
 * PRECONDITIONS:
 *     - p_values.len >= n
 *     - q_values.len >= n
 *     - All p-values should be in [0, 1]
 *
 * POSTCONDITIONS:
 *     - q_values[i] = adjusted p-value controlling FDR
 *     - q_values are monotonic: q[sorted_i] >= q[sorted_i+1]
 *     - q_values clamped to [0, 1]
 *
 * ALGORITHM:
 *     1. Sort p-values ascending with VQSort O(n log n)
 *     2. Compute adjusted values: q[i] = p[i] * n / rank[i]
 *     3. Backward pass enforcing monotonicity
 *     4. Map back to original indices
 *
 * COMPLEXITY:
 *     Time:  O(n log n) using VQSort
 *     Space: O(n) for sorted arrays
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void benjamini_hochberg_correction(
    Array<const Real> p_values,        // Original p-values [n]
    Index n,                            // Number of p-values
    Array<Real> q_values                // Output: Adjusted p-values [n]
);

// =============================================================================
// Spatial Weight Construction
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: distance_band_weights
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Construct spatial weight matrix based on distance threshold.
 *
 * PARAMETERS:
 *     coordinates        [in]  Point coordinates (n x 2, row-major)
 *     n                  [in]  Number of points
 *     threshold_distance [in]  Distance threshold for neighbors
 *     row_ptrs           [out] CSR row pointers [n+1]
 *     col_indices        [out] CSR column indices [nnz]
 *     weights            [out] CSR values (all 1.0) [nnz]
 *     nnz                [out] Number of non-zeros
 *
 * PRECONDITIONS:
 *     - coordinates contains 2D points: [x0, y0, x1, y1, ...]
 *     - Output arrays must be pre-allocated to max possible size
 *     - threshold_distance > 0
 *
 * POSTCONDITIONS:
 *     - w_ij = 1 if distance(i, j) <= threshold_distance and i != j
 *     - w_ij = 0 otherwise
 *     - Matrix is symmetric
 *
 * COMPLEXITY:
 *     Time:  O(n^2) brute-force distance computation
 *     Space: O(nnz) for output
 *
 * THREAD SAFETY:
 *     Safe - parallelized with two-pass construction
 * -------------------------------------------------------------------------- */
void distance_band_weights(
    const Real* coordinates,           // Coordinates [n * 2]
    Index n,                           // Number of points
    Real threshold_distance,           // Distance threshold
    Index* row_ptrs,                   // Output: CSR row pointers [n+1]
    Index* col_indices,                // Output: CSR column indices
    Real* weights,                     // Output: CSR values
    Index& nnz                         // Output: Number of non-zeros
);

/* -----------------------------------------------------------------------------
 * FUNCTION: knn_weights
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Construct K-nearest neighbors spatial weight matrix.
 *
 * PARAMETERS:
 *     coordinates [in]  Point coordinates (n x 2, row-major)
 *     n           [in]  Number of points
 *     k           [in]  Number of nearest neighbors
 *     row_ptrs    [out] CSR row pointers [n+1]
 *     col_indices [out] CSR column indices [n*k]
 *     weights     [out] CSR values [n*k]
 *
 * PRECONDITIONS:
 *     - coordinates contains 2D points
 *     - k < n
 *     - Output arrays pre-allocated for n*k elements
 *
 * POSTCONDITIONS:
 *     - Each row has exactly k non-zeros (k nearest neighbors)
 *     - weights can be 1 (binary) or inverse distance
 *     - Matrix may not be symmetric (i in knn(j) not implies j in knn(i))
 *
 * COMPLEXITY:
 *     Time:  O(n^2 log k) using heap-based selection
 *     Space: O(n * k) for output
 *
 * THREAD SAFETY:
 *     Safe - parallelized over observations
 * -------------------------------------------------------------------------- */
void knn_weights(
    const Real* coordinates,           // Coordinates [n * 2]
    Index n,                           // Number of points
    Index k,                           // Number of neighbors
    Index* row_ptrs,                   // Output: CSR row pointers [n+1]
    Index* col_indices,                // Output: CSR column indices [n*k]
    Real* weights                      // Output: CSR values [n*k]
);

// =============================================================================
// Auxiliary Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: bivariate_local_morans_i
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute bivariate Local Moran's I between two variables.
 *
 * PARAMETERS:
 *     weights        [in]  Spatial weight matrix
 *     values_x       [in]  First attribute values [n]
 *     values_y       [in]  Second attribute values [n]
 *     n              [in]  Number of observations
 *     local_I        [out] Bivariate Local Moran's I [n]
 *     p_values       [out] Pseudo p-values [n]
 *     n_permutations [in]  Number of permutations
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - All arrays must have length >= n
 *
 * POSTCONDITIONS:
 *     - local_I[i] = z_x[i] * lag_y[i] where lag_y = W * z_y
 *     - Measures local correlation between x and spatially lagged y
 *
 * COMPLEXITY:
 *     Time:  O(n * n_permutations * avg_neighbors)
 *     Space: O(n_threads * n)
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void bivariate_local_morans_i(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix
    const Real* values_x,              // First variable [n]
    const Real* values_y,              // Second variable [n]
    Index n,                           // Number of observations
    Real* local_I,                     // Output: Bivariate Local I [n]
    Real* p_values,                    // Output: Pseudo p-values [n]
    Index n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_spatial_clusters
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect and label spatial clusters from LISA patterns.
 *
 * PARAMETERS:
 *     weights    [in]  Spatial weight matrix
 *     patterns   [in]  LISA pattern classification [n]
 *     n          [in]  Number of observations
 *     cluster_id [out] Cluster labels [n], -1 for non-clustered
 *
 * RETURNS:
 *     Number of clusters found
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - patterns from classify_lisa_patterns
 *     - cluster_id must be pre-allocated with n elements
 *
 * POSTCONDITIONS:
 *     - Connected significant observations get same cluster_id
 *     - Non-significant observations have cluster_id = -1
 *     - Cluster IDs are contiguous starting from 0
 *
 * ALGORITHM:
 *     1. Build adjacency graph from significant observations
 *     2. Find connected components using BFS
 *     3. Assign cluster labels
 *
 * COMPLEXITY:
 *     Time:  O(n + nnz)
 *     Space: O(n) for BFS queue
 *
 * THREAD SAFETY:
 *     Safe - sequential BFS with parallel-safe data structures
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index detect_spatial_clusters(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix
    const LISAPattern* patterns,       // LISA patterns [n]
    Index n,                           // Number of observations
    Index* cluster_id                  // Output: Cluster labels [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: spatial_autocorrelation_summary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute comprehensive spatial autocorrelation summary statistics.
 *
 * PARAMETERS:
 *     weights    [in]  Spatial weight matrix
 *     values     [in]  Attribute values [n]
 *     n          [in]  Number of observations
 *     morans_i   [out] Global Moran's I
 *     gearys_c   [out] Global Geary's C
 *     n_hotspots [out] Count of significant hot spots
 *     n_coldspots [out] Count of significant cold spots
 *     n_high_low [out] Count of high-low outliers
 *     n_low_high [out] Count of low-high outliers
 *
 * PRECONDITIONS:
 *     - weights must be square (n x n) sparse matrix
 *     - All output pointers must be valid
 *
 * POSTCONDITIONS:
 *     - Provides complete spatial pattern summary
 *     - All statistics computed with permutation inference
 *
 * COMPLEXITY:
 *     Time:  O(n * n_permutations * avg_neighbors)
 *     Space: O(n_threads * n)
 *
 * THREAD SAFETY:
 *     Safe - all internal operations are thread-safe
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spatial_autocorrelation_summary(
    const Sparse<T, IsCSR>& weights,  // Spatial weight matrix
    const Real* values,                // Attribute values [n]
    Index n,                           // Number of observations
    Real* morans_i,                    // Output: Global Moran's I
    Real* gearys_c,                    // Output: Global Geary's C
    Index* n_hotspots,                 // Output: Hot spot count
    Index* n_coldspots,                // Output: Cold spot count
    Index* n_high_low,                 // Output: High-Low outlier count
    Index* n_low_high                  // Output: Low-High outlier count
);

} // namespace scl::kernel::hotspot
