// =============================================================================
// FILE: scl/kernel/stat/effect_size.h
// BRIEF: API reference for effect size computation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::effect_size {

/* -----------------------------------------------------------------------------
 * ENUM: EffectSizeType
 * -----------------------------------------------------------------------------
 * VALUES:
 *     CohensD     - Cohen's d: (mean2 - mean1) / pooled_sd
 *     HedgesG     - Hedges' g: bias-corrected Cohen's d
 *     GlassDelta  - Glass' delta: (mean2 - mean1) / sd1 (control SD)
 *     CLES        - Common Language Effect Size (from AUROC)
 * -------------------------------------------------------------------------- */
enum class EffectSizeType {
    CohensD,
    HedgesG,
    GlassDelta,
    CLES
};

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_cohens_d
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Cohen's d effect size.
 *
 * FORMULA:
 *     d = (mean2 - mean1) / sqrt(pooled_variance)
 *     pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
 *
 * RETURNS:
 *     Cohen's d, or 0 if pooled_sd < SIGMA_MIN
 * -------------------------------------------------------------------------- */
Real compute_cohens_d(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_hedges_g
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Hedges' g (bias-corrected Cohen's d).
 *
 * FORMULA:
 *     g = d * J
 *     J = 1 - 3 / (4*df - 1), where df = n1 + n2 - 2
 *
 * NOTE:
 *     Hedges' correction reduces small-sample bias in Cohen's d.
 * -------------------------------------------------------------------------- */
Real compute_hedges_g(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_glass_delta
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Glass' delta effect size.
 *
 * FORMULA:
 *     delta = (mean2 - mean1) / sd1
 *
 * NOTE:
 *     Uses only control group (group 1) standard deviation.
 *     Useful when treatment affects variance.
 * -------------------------------------------------------------------------- */
Real compute_glass_delta(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: auroc_to_cles
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert AUROC to Common Language Effect Size.
 *
 * INTERPRETATION:
 *     CLES = probability that a randomly selected value from group 2
 *     exceeds a randomly selected value from group 1.
 *
 * NOTE:
 *     AUROC and CLES are mathematically equivalent.
 * -------------------------------------------------------------------------- */
Real auroc_to_cles(Real auroc);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_effect_size
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic effect size computation dispatcher.
 *
 * PARAMETERS:
 *     mean1, var1, n1  [in]  Group 1 statistics
 *     mean2, var2, n2  [in]  Group 2 statistics
 *     type             [in]  Effect size type to compute
 *
 * RETURNS:
 *     Computed effect size based on type
 *
 * NOTE:
 *     For CLES, returns 0.5 (requires AUROC, not directly computable).
 * -------------------------------------------------------------------------- */
Real compute_effect_size(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2,
    EffectSizeType type
);

/* -----------------------------------------------------------------------------
 * FUNCTION: effect_size
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute effect size for each feature in a sparse matrix.
 *
 * PARAMETERS:
 *     matrix           [in]  Sparse matrix (features x samples)
 *     group_ids        [in]  Binary group assignment (0 or 1)
 *     out_effect_size  [out] Effect sizes, size = n_features
 *     type             [in]  Effect size type (default: CohensD)
 *
 * PRECONDITIONS:
 *     - Both groups have at least one member
 *     - out_effect_size.len >= matrix.primary_dim()
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Partition values by group with moment accumulation
 *     2. Compute mean and variance for each group
 *     3. Apply effect size formula based on type
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature)
 *     Space: O(threads * max_row_length)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void effect_size(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_effect_size,
    EffectSizeType type = EffectSizeType::CohensD
);

/* -----------------------------------------------------------------------------
 * FUNCTION: ttest_with_effect_size
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Combined t-test and effect size computation in single pass.
 *
 * PARAMETERS:
 *     matrix           [in]  Sparse matrix (features x samples)
 *     group_ids        [in]  Binary group assignment (0 or 1)
 *     out_t_stats      [out] T-statistics
 *     out_p_values     [out] P-values
 *     out_log2_fc      [out] Log2 fold changes
 *     out_effect_size  [out] Effect sizes
 *     es_type          [in]  Effect size type (default: CohensD)
 *     use_welch        [in]  Use Welch's t-test (default: true)
 *
 * NOTE:
 *     More efficient than calling ttest() and effect_size() separately,
 *     as partitioning and variance computation are done once.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void ttest_with_effect_size(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    Array<Real> out_effect_size,
    EffectSizeType es_type = EffectSizeType::CohensD,
    bool use_welch = true
);

} // namespace scl::kernel::stat::effect_size
