// =============================================================================
// FILE: scl/kernel/resample.h
// BRIEF: API reference for resampling operations with fast RNG
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::resample {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// Transform Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: downsample
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Downsample each row to a target total count using binomial sampling.
 *
 * PARAMETERS:
 *     matrix     [in,out] Expression matrix, modified in-place
 *     target_sum [in]     Target total count per row
 *     seed       [in]     Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - target_sum > 0
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each row has total count approximately equal to target_sum
 *     - Matrix structure (indices, indptr) unchanged
 *     - Sampling is deterministic given seed
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. Compute current total count
 *         2. If current <= target, skip
 *         3. For each non-zero element:
 *            a. Compute probability = remaining_target / remaining_total
 *            b. Sample binomial(count, probability)
 *            c. Update value and remaining counts
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with independent RNG states
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void downsample(
    Sparse<T, IsCSR>& matrix,               // Expression matrix, modified in-place
    Real target_sum,                        // Target total count per row
    uint64_t seed = 42                      // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: downsample_variable
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Downsample each row to a variable target count using binomial sampling.
 *
 * PARAMETERS:
 *     matrix        [in,out] Expression matrix, modified in-place
 *     target_counts [in]     Target count for each row [n_rows]
 *     seed          [in]      Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - target_counts.len >= matrix.rows()
 *     - All target_counts[i] > 0
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Row i has total count approximately equal to target_counts[i]
 *     - Matrix structure (indices, indptr) unchanged
 *     - Sampling is deterministic given seed
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. Compute current total count
 *         2. If current <= target, skip
 *         3. For each non-zero element:
 *            a. Compute probability = remaining_target / remaining_total
 *            b. Sample binomial(count, probability)
 *            c. Update value and remaining counts
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with independent RNG states
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void downsample_variable(
    Sparse<T, IsCSR>& matrix,               // Expression matrix, modified in-place
    Array<const Real> target_counts,        // Target counts per row [n_rows]
    uint64_t seed = 42                      // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: binomial_resample
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Resample each count value using binomial distribution with fixed probability.
 *
 * PARAMETERS:
 *     matrix [in,out] Expression matrix, modified in-place
 *     p      [in]     Success probability for binomial sampling
 *     seed   [in]     Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - p in [0, 1]
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each value is replaced by binomial(value, p)
 *     - Matrix structure (indices, indptr) unchanged
 *     - Sampling is deterministic given seed
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         For each non-zero element:
 *             1. Sample binomial(count, p)
 *             2. Replace value with sampled count
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with independent RNG states
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void binomial_resample(
    Sparse<T, IsCSR>& matrix,               // Expression matrix, modified in-place
    Real p,                                  // Binomial probability
    uint64_t seed = 42                      // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: poisson_resample
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Resample each count value using Poisson distribution with scaled mean.
 *
 * PARAMETERS:
 *     matrix [in,out] Expression matrix, modified in-place
 *     lambda [in]     Scaling factor for Poisson mean (mean = count * lambda)
 *     seed   [in]     Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - lambda > 0
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each value is replaced by Poisson(value * lambda)
 *     - Matrix structure (indices, indptr) unchanged
 *     - Sampling is deterministic given seed
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         For each non-zero element:
 *             1. Compute mean = count * lambda
 *             2. Sample Poisson(mean)
 *             3. Replace value with sampled count
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows with independent RNG states
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void poisson_resample(
    Sparse<T, IsCSR>& matrix,               // Expression matrix, modified in-place
    Real lambda,                             // Poisson scaling factor
    uint64_t seed = 42                      // Random seed
);

} // namespace scl::kernel::resample

