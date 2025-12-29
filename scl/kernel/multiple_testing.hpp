#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/multiple_testing.hpp
// BRIEF: Multiple testing correction methods
//
// APPLICATIONS:
// - FDR control (Benjamini-Hochberg, Storey)
// - P-value adjustment (Bonferroni)
// - Q-value estimation
// - Local FDR estimation
// =============================================================================

namespace scl::kernel::multiple_testing {

namespace config {
    constexpr Real DEFAULT_FDR_LEVEL = Real(0.05);
    constexpr Real DEFAULT_LAMBDA = Real(0.5);
    constexpr Real MIN_PVALUE = Real(1e-300);
    constexpr Real MAX_PVALUE = Real(1.0);
    constexpr Size SPLINE_KNOTS = 10;
    constexpr Size MIN_TESTS_FOR_STOREY = 100;
}

namespace detail {

// Sort indices by p-values (ascending) using SIMD-optimized VQSort
SCL_FORCE_INLINE void sort_indices_by_pvalue(
    Array<const Real> p_values,
    Index* indices
) {
    const Size n = p_values.len;
    if (n == 0) return;

    // Copy p-values to temporary array for sorting
    // PERFORMANCE: RAII memory management with unique_ptr
    auto pvals_copy_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* pvals_copy = pvals_copy_ptr.get();
    scl::memory::copy_fast(p_values, Array<Real>(pvals_copy, n));

    // Initialize indices
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - i is bounded by array size and fits in Index
        indices[static_cast<Index>(i)] = static_cast<Index>(i);
    }

    // Use SIMD-optimized sort_pairs (VQSort - 2-5x faster than std::sort)
    scl::sort::sort_pairs(
        Array<Real>(pvals_copy, n),
        Array<Index>(indices, n)
    );

    // unique_ptr automatically frees memory when going out of scope
}

// Clamp p-value to valid range
SCL_FORCE_INLINE Real clamp_pvalue(Real p) {
    if (p < config::MIN_PVALUE) return config::MIN_PVALUE;
    if (p > config::MAX_PVALUE) return config::MAX_PVALUE;
    return p;
}

// Estimate pi0 (proportion of true nulls) using Storey's method
SCL_FORCE_INLINE Real estimate_pi0(
    Array<const Real> p_values,
    Real lambda
) {
    const Size n = p_values.len;
    if (n == 0) return Real(1.0);

    Size count_above = 0;
    for (Size i = 0; i < n; ++i) {
        if (p_values.ptr[i] > lambda) {
            ++count_above;
        }
    }

    Real pi0 = static_cast<Real>(count_above) / (static_cast<Real>(n) * (Real(1.0) - lambda));
    return scl::algo::min2(pi0, Real(1.0));
}

// Estimate pi0 with bootstrap for Storey q-value
SCL_FORCE_INLINE Real estimate_pi0_bootstrap(
    Array<const Real> p_values,
    Real* lambda_grid,
    Size n_lambda
) {
    const Size n = p_values.len;
    if (n < config::MIN_TESTS_FOR_STOREY) {
        return Real(1.0);
    }

    // Compute pi0 estimates at each lambda
    // PERFORMANCE: RAII memory management with unique_ptr
    auto pi0_estimates_ptr = scl::memory::aligned_alloc<Real>(n_lambda, SCL_ALIGNMENT);
    Real* pi0_estimates = pi0_estimates_ptr.get();

    for (Size l = 0; l < n_lambda; ++l) {
        Size count_above = 0;
        for (Size i = 0; i < n; ++i) {
            if (p_values.ptr[i] > lambda_grid[l]) {
                ++count_above;
            }
        }
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - l is bounded by n_lambda and fits in Index
        pi0_estimates[static_cast<Index>(l)] = static_cast<Real>(count_above) /
                          (static_cast<Real>(n) * (Real(1.0) - lambda_grid[l]));
    }

    // Use spline smoothing (simplified: take minimum pi0 estimate)
    Real min_pi0 = pi0_estimates[0];
    for (Size l = 1; l < n_lambda; ++l) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        if (pi0_estimates[static_cast<Index>(l)] < min_pi0) {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            min_pi0 = pi0_estimates[static_cast<Index>(l)];
        }
    }

    // unique_ptr automatically frees memory when going out of scope

    Real clamped = scl::algo::max2(min_pi0, Real(0.0));
    return scl::algo::min2(clamped, Real(1.0));
}

// Kernel density estimation for local FDR
SCL_FORCE_INLINE Real gaussian_kernel(Real x, Real bandwidth) {
    Real z = x / bandwidth;
    return std::exp(-Real(0.5) * z * z) / (bandwidth * std::sqrt(Real(2.0) * Real(3.14159265358979323846)));
}

SCL_FORCE_INLINE void kde_estimate(
    Array<const Real> data,
    Real* grid,
    Real* density,
    Size n_grid,
    Real bandwidth
) {
    const Size n = data.len;
    const Real norm_factor = Real(1.0) / static_cast<Real>(n);

    for (Size g = 0; g < n_grid; ++g) {
        density[g] = Real(0.0);
        for (Size i = 0; i < n; ++i) {
            density[g] += gaussian_kernel(grid[g] - data.ptr[i], bandwidth);
        }
        density[g] *= norm_factor;
    }
}

// Silverman's rule of thumb for bandwidth selection
SCL_FORCE_INLINE Real silverman_bandwidth(Array<const Real> data) {
    const Size n = data.len;
    if (n < 2) return Real(0.1);

    // Compute mean and std
    Real sum = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        sum += data.ptr[i];
    }
    Real mean = sum / static_cast<Real>(n);

    Real var_sum = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        Real diff = data.ptr[i] - mean;
        var_sum += diff * diff;
    }
    Real std_dev = std::sqrt(var_sum / static_cast<Real>(n - 1));

    // Silverman's rule
    return Real(1.06) * std_dev * std::pow(static_cast<Real>(n), Real(-0.2));
}

// Transform p-values to z-scores
SCL_FORCE_INLINE Real pvalue_to_zscore(Real p) {
    // Probit transformation (inverse normal CDF approximation)
    Real p_clamped = clamp_pvalue(p);
    if (p_clamped <= Real(0.5)) {
        Real t = std::sqrt(-Real(2.0) * std::log(p_clamped));
        Real c0 = Real(2.515517);
        Real c1 = Real(0.802853);
        Real c2 = Real(0.010328);
        Real d1 = Real(1.432788);
        Real d2 = Real(0.189269);
        Real d3 = Real(0.001308);
        return t - (c0 + c1 * t + c2 * t * t) / (Real(1.0) + d1 * t + d2 * t * t + d3 * t * t * t);
    } else {
        Real t = std::sqrt(-Real(2.0) * std::log(Real(1.0) - p_clamped));
        Real c0 = Real(2.515517);
        Real c1 = Real(0.802853);
        Real c2 = Real(0.010328);
        Real d1 = Real(1.432788);
        Real d2 = Real(0.189269);
        Real d3 = Real(0.001308);
        return -(t - (c0 + c1 * t + c2 * t * t) / (Real(1.0) + d1 * t + d2 * t * t + d3 * t * t * t));
    }
}

} // namespace detail

// =============================================================================
// Benjamini-Hochberg FDR correction
// =============================================================================

void benjamini_hochberg(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values,
    Real /* fdr_level */
) {
    SCL_CHECK_DIM(p_values.len == adjusted_p_values.len,
        "p_values and adjusted_p_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // Allocate working memory
    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto sorted_pvalues_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto adjusted_sorted_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = sorted_indices_ptr.get();
    Real* sorted_pvalues = sorted_pvalues_ptr.get();
    Real* adjusted_sorted = adjusted_sorted_ptr.get();

    // Sort indices by p-value
    detail::sort_indices_by_pvalue(p_values, sorted_indices);

    // Get sorted p-values
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - i and sorted_indices[i] are bounded and fit in Index
        sorted_pvalues[static_cast<Index>(i)] = p_values.ptr[sorted_indices[static_cast<Index>(i)]];
    }

    // BH adjustment: p_adj[i] = p[i] * n / rank
    // Then enforce monotonicity from the end
    const Real n_real = static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        Real rank = static_cast<Real>(i + 1);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_sorted[static_cast<Index>(i)] = sorted_pvalues[static_cast<Index>(i)] * n_real / rank;
    }

    // Enforce monotonicity: cumulative minimum from right to left
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
    adjusted_sorted[static_cast<Index>(n - 1)] = scl::algo::min2(adjusted_sorted[static_cast<Index>(n - 1)], Real(1.0));
    for (Size i = n - 1; i > 0; --i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_idx = static_cast<Index>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_minus_one_idx = static_cast<Index>(i - 1);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], adjusted_sorted[i_idx]);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], Real(1.0));
    }

    // Map back to original order
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_p_values.ptr[sorted_indices[static_cast<Index>(i)]] = adjusted_sorted[static_cast<Index>(i)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Bonferroni correction
// =============================================================================

void bonferroni(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    SCL_CHECK_DIM(p_values.len == adjusted_p_values.len,
        "p_values and adjusted_p_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    const Real n_real = static_cast<Real>(n);

    // Copy p-values to adjusted array and scale by n
    for (Size i = 0; i < n; ++i) {
        adjusted_p_values.ptr[i] = p_values.ptr[i] * n_real;
    }

    // Clamp all values to [0, 1.0] using SIMD
    scl::vectorize::clamp_max(adjusted_p_values, Real(1.0));
}

// =============================================================================
// Storey q-value estimation
// =============================================================================

void storey_qvalue(
    Array<const Real> p_values,
    Array<Real> q_values,
    Real lambda
) {
    SCL_CHECK_DIM(p_values.len == q_values.len,
        "p_values and q_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // Estimate pi0 (proportion of true nulls)
    Real pi0 = Real(1.0);
    if (n >= config::MIN_TESTS_FOR_STOREY) {
        // Use bootstrap method with lambda grid
        constexpr Size n_lambda = 20;
        // PERFORMANCE: Small fixed-size array for lambda grid
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        Real lambda_grid[n_lambda];
        for (Size l = 0; l < n_lambda; ++l) {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            // PERFORMANCE: Safe narrowing - l is bounded by n_lambda and fits in Index
            lambda_grid[static_cast<Index>(l)] = Real(0.05) + static_cast<Real>(l) * Real(0.05);
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
        pi0 = detail::estimate_pi0_bootstrap(p_values, lambda_grid, n_lambda);
    } else {
        pi0 = detail::estimate_pi0(p_values, lambda);
    }

    // Allocate working memory
    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto sorted_pvalues_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto q_sorted_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = sorted_indices_ptr.get();
    Real* sorted_pvalues = sorted_pvalues_ptr.get();
    Real* q_sorted = q_sorted_ptr.get();

    // Sort indices by p-value
    detail::sort_indices_by_pvalue(p_values, sorted_indices);

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        sorted_pvalues[static_cast<Index>(i)] = p_values.ptr[sorted_indices[static_cast<Index>(i)]];
    }

    // Compute q-values: q[i] = pi0 * n * p[i] / rank
    const Real n_real = static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        Real rank = static_cast<Real>(i + 1);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        q_sorted[static_cast<Index>(i)] = pi0 * n_real * sorted_pvalues[static_cast<Index>(i)] / rank;
    }

    // Enforce monotonicity from right to left
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
    q_sorted[static_cast<Index>(n - 1)] = scl::algo::min2(q_sorted[static_cast<Index>(n - 1)], Real(1.0));
    for (Size i = n - 1; i > 0; --i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_idx = static_cast<Index>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_minus_one_idx = static_cast<Index>(i - 1);
        q_sorted[i_minus_one_idx] = scl::algo::min2(q_sorted[i_minus_one_idx], q_sorted[i_idx]);
        q_sorted[i_minus_one_idx] = scl::algo::min2(q_sorted[i_minus_one_idx], Real(1.0));
    }

    // Map back to original order
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        q_values.ptr[sorted_indices[static_cast<Index>(i)]] = q_sorted[static_cast<Index>(i)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Local FDR estimation
// =============================================================================

void local_fdr(
    Array<const Real> p_values,
    Array<Real> lfdr
) {
    SCL_CHECK_DIM(p_values.len == lfdr.len,
        "p_values and lfdr must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // Transform p-values to z-scores
    // PERFORMANCE: RAII memory management with unique_ptr
    auto z_scores_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* z_scores = z_scores_ptr.get();
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        z_scores[static_cast<Index>(i)] = detail::pvalue_to_zscore(p_values.ptr[i]);
    }

    // Find z-score range
    Real z_min = z_scores[0], z_max = z_scores[0];
    for (Size i = 1; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        if (z_scores[static_cast<Index>(i)] < z_min) z_min = z_scores[static_cast<Index>(i)];
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        if (z_scores[static_cast<Index>(i)] > z_max) z_max = z_scores[static_cast<Index>(i)];
    }

    // Compute bandwidth using Silverman's rule
    Array<const Real> z_array(z_scores, n);
    Real bandwidth = detail::silverman_bandwidth(z_array);
    bandwidth = scl::algo::max2(bandwidth, Real(0.1));

    // Create evaluation grid
    // PERFORMANCE: RAII memory management with unique_ptr
    constexpr Size n_grid = 200;
    auto grid_ptr = scl::memory::aligned_alloc<Real>(n_grid, SCL_ALIGNMENT);
    auto f_density_ptr = scl::memory::aligned_alloc<Real>(n_grid, SCL_ALIGNMENT);
    auto f0_density_ptr = scl::memory::aligned_alloc<Real>(n_grid, SCL_ALIGNMENT);
    Real* grid = grid_ptr.get();
    Real* f_density = f_density_ptr.get();
    Real* f0_density = f0_density_ptr.get();

    Real grid_step = (z_max - z_min + Real(2.0) * bandwidth) / static_cast<Real>(n_grid - 1);
    for (Size g = 0; g < n_grid; ++g) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        grid[static_cast<Index>(g)] = z_min - bandwidth + static_cast<Real>(g) * grid_step;
    }

    // Estimate f(z) using KDE
    detail::kde_estimate(z_array, grid, f_density, n_grid, bandwidth);

    // f0(z) is standard normal density
    for (Size g = 0; g < n_grid; ++g) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        f0_density[static_cast<Index>(g)] = std::exp(-Real(0.5) * grid[static_cast<Index>(g)] * grid[static_cast<Index>(g)]) / std::sqrt(Real(2.0) * Real(3.14159265358979323846));
    }

    // Estimate pi0
    Real pi0 = detail::estimate_pi0(p_values, config::DEFAULT_LAMBDA);

    // Compute local FDR for each test by interpolation
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real z = z_scores[static_cast<Index>(i)];

        // Find grid position
        Size g_low = 0;
        for (Size g = 0; g < n_grid - 1; ++g) {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            if (grid[static_cast<Index>(g + 1)] > z) {
                g_low = g;
                break;
            }
            g_low = g;
        }

        // Linear interpolation
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real t = (z - grid[static_cast<Index>(g_low)]) / grid_step;
        t = scl::algo::max2(Real(0.0), scl::algo::min2(Real(1.0), t));

        Size g_high = scl::algo::min2(g_low + 1, n_grid - 1);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real f_z = f_density[static_cast<Index>(g_low)] * (Real(1.0) - t) + f_density[static_cast<Index>(g_high)] * t;
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real f0_z = f0_density[static_cast<Index>(g_low)] * (Real(1.0) - t) + f0_density[static_cast<Index>(g_high)] * t;

        // local FDR = pi0 * f0(z) / f(z)
        if (f_z > Real(1e-10)) {
            lfdr.ptr[i] = scl::algo::min2(pi0 * f0_z / f_z, Real(1.0));
        } else {
            lfdr.ptr[i] = Real(1.0);
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Empirical FDR (permutation-based)
// =============================================================================

void empirical_fdr(
    Array<const Real> observed_scores,
    const std::vector<Array<Real>>& permuted_scores,
    Array<Real> fdr
) {
    SCL_CHECK_DIM(observed_scores.len == fdr.len,
        "observed_scores and fdr must have same length");

    const Size n = observed_scores.len;
    const Size n_perms = permuted_scores.size();

    if (n == 0 || n_perms == 0) return;

    // Sort observed scores (descending, assuming higher = more significant)
    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_obs_indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto scores_copy_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_obs_indices = sorted_obs_indices_ptr.get();
    Real* scores_copy = scores_copy_ptr.get();

    // Copy scores and initialize indices
    scl::memory::copy_fast(observed_scores, Array<Real>(scores_copy, n));
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        sorted_obs_indices[static_cast<Index>(i)] = static_cast<Index>(i);
    }

    // Use SIMD-optimized sort_pairs_descending (VQSort - 2-5x faster than std::sort)
    scl::sort::sort_pairs_descending(
        Array<Real>(scores_copy, n),
        Array<Index>(sorted_obs_indices, n)
    );

    // Count permutation discoveries at each threshold
    // PERFORMANCE: RAII memory management with unique_ptr
    auto fdr_at_rank_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* fdr_at_rank = fdr_at_rank_ptr.get();

    for (Size r = 0; r < n; ++r) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        Real threshold = scores_copy[static_cast<Index>(r)];

        // Count observed discoveries at this threshold
        Size obs_discoveries = r + 1;

        // Count null discoveries across all permutations
        Size null_discoveries = 0;
        for (Size p = 0; p < n_perms; ++p) {
            const Array<Real>& perm = permuted_scores[p];
            for (Size i = 0; i < perm.len; ++i) {
                if (perm.ptr[i] >= threshold) {
                    ++null_discoveries;
                }
            }
        }

        // Empirical FDR = (null_discoveries / n_perms) / obs_discoveries
        Real expected_null = static_cast<Real>(null_discoveries) / static_cast<Real>(n_perms);
        Real fdr_est = expected_null / static_cast<Real>(obs_discoveries);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        fdr_at_rank[static_cast<Index>(r)] = scl::algo::min2(fdr_est, Real(1.0));
    }

    // Enforce monotonicity (cumulative minimum from right to left)
    for (Size r = n - 1; r > 0; --r) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto r_idx = static_cast<Index>(r);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto r_minus_one_idx = static_cast<Index>(r - 1);
        fdr_at_rank[r_minus_one_idx] = scl::algo::min2(fdr_at_rank[r_minus_one_idx], fdr_at_rank[r_idx]);
    }

    // Map back to original order
    for (Size r = 0; r < n; ++r) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        fdr.ptr[sorted_obs_indices[static_cast<Index>(r)]] = fdr_at_rank[static_cast<Index>(r)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Additional utility functions
// =============================================================================

// Benjamini-Yekutieli FDR (for arbitrary dependency)
void benjamini_yekutieli(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    SCL_CHECK_DIM(p_values.len == adjusted_p_values.len,
        "p_values and adjusted_p_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // Compute c(n) = sum(1/i) for i = 1 to n
    Real c_n = Real(0.0);
    for (Size i = 1; i <= n; ++i) {
        c_n += Real(1.0) / static_cast<Real>(i);
    }

    auto sorted_indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto sorted_pvalues_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto adjusted_sorted_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = sorted_indices_ptr.release();
    Real* sorted_pvalues = sorted_pvalues_ptr.release();
    Real* adjusted_sorted = adjusted_sorted_ptr.release();

    detail::sort_indices_by_pvalue(p_values, sorted_indices);

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        sorted_pvalues[static_cast<Index>(i)] = p_values.ptr[sorted_indices[static_cast<Index>(i)]];
    }

    // BY adjustment: p_adj[i] = p[i] * n * c(n) / rank
    const Real n_real = static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        Real rank = static_cast<Real>(i + 1);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_sorted[static_cast<Index>(i)] = sorted_pvalues[static_cast<Index>(i)] * n_real * c_n / rank;
    }

    // Enforce monotonicity
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
    adjusted_sorted[static_cast<Index>(n - 1)] = scl::algo::min2(adjusted_sorted[static_cast<Index>(n - 1)], Real(1.0));
    for (Size i = n - 1; i > 0; --i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_idx = static_cast<Index>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_minus_one_idx = static_cast<Index>(i - 1);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], adjusted_sorted[i_idx]);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], Real(1.0));
    }

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_p_values.ptr[sorted_indices[static_cast<Index>(i)]] = adjusted_sorted[static_cast<Index>(i)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// Holm-Bonferroni step-down procedure
void holm_bonferroni(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    SCL_CHECK_DIM(p_values.len == adjusted_p_values.len,
        "p_values and adjusted_p_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_indices_ptr2 = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto sorted_pvalues_ptr2 = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto adjusted_sorted_ptr2 = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = sorted_indices_ptr2.get();
    Real* sorted_pvalues = sorted_pvalues_ptr2.get();
    Real* adjusted_sorted = adjusted_sorted_ptr2.get();

    detail::sort_indices_by_pvalue(p_values, sorted_indices);

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        sorted_pvalues[static_cast<Index>(i)] = p_values.ptr[sorted_indices[static_cast<Index>(i)]];
    }

    // Holm adjustment: p_adj[i] = p[i] * (n - rank + 1)
    const Real n_real = static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        Real multiplier = n_real - static_cast<Real>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_sorted[static_cast<Index>(i)] = sorted_pvalues[static_cast<Index>(i)] * multiplier;
    }

    // Enforce monotonicity (cumulative maximum from left to right)
    for (Size i = 1; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_idx = static_cast<Index>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_minus_one_idx = static_cast<Index>(i - 1);
        adjusted_sorted[i_idx] = scl::algo::max2(adjusted_sorted[i_idx], adjusted_sorted[i_minus_one_idx]);
    }

    // Cap at 1.0
    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_sorted[static_cast<Index>(i)] = scl::algo::min2(adjusted_sorted[static_cast<Index>(i)], Real(1.0));
    }

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_p_values.ptr[sorted_indices[static_cast<Index>(i)]] = adjusted_sorted[static_cast<Index>(i)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// Hochberg step-up procedure
void hochberg(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    SCL_CHECK_DIM(p_values.len == adjusted_p_values.len,
        "p_values and adjusted_p_values must have same length");

    const Size n = p_values.len;
    if (n == 0) return;

    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_indices_ptr3 = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    auto sorted_pvalues_ptr3 = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto adjusted_sorted_ptr3 = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* sorted_indices = sorted_indices_ptr3.get();
    Real* sorted_pvalues = sorted_pvalues_ptr3.get();
    Real* adjusted_sorted = adjusted_sorted_ptr3.get();

    detail::sort_indices_by_pvalue(p_values, sorted_indices);

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        sorted_pvalues[static_cast<Index>(i)] = p_values.ptr[sorted_indices[static_cast<Index>(i)]];
    }

    // Hochberg adjustment: p_adj[i] = p[i] * (n - rank + 1)
    const Real n_real = static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        Real multiplier = n_real - static_cast<Real>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_sorted[static_cast<Index>(i)] = sorted_pvalues[static_cast<Index>(i)] * multiplier;
    }

    // Enforce monotonicity (cumulative minimum from right to left)
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
    adjusted_sorted[static_cast<Index>(n - 1)] = scl::algo::min2(adjusted_sorted[static_cast<Index>(n - 1)], Real(1.0));
    for (Size i = n - 1; i > 0; --i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_idx = static_cast<Index>(i);
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        const auto i_minus_one_idx = static_cast<Index>(i - 1);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], adjusted_sorted[i_idx]);
        adjusted_sorted[i_minus_one_idx] = scl::algo::min2(adjusted_sorted[i_minus_one_idx], Real(1.0));
    }

    for (Size i = 0; i < n; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        adjusted_p_values.ptr[sorted_indices[static_cast<Index>(i)]] = adjusted_sorted[static_cast<Index>(i)];
    }

    // unique_ptr automatically frees memory when going out of scope
}

// Count significant tests at given threshold
Size count_significant(
    Array<const Real> p_values,
    Real threshold
) {
    Size count = 0;
    for (Size i = 0; i < p_values.len; ++i) {
        if (p_values.ptr[i] <= threshold) {
            ++count;
        }
    }
    return count;
}

// Get indices of significant tests
void significant_indices(
    Array<const Real> p_values,
    Real threshold,
    Index* out_indices,
    Size& out_count
) {
    out_count = 0;
    for (Size i = 0; i < p_values.len; ++i) {
        if (p_values.ptr[i] <= threshold) {
            out_indices[out_count++] = static_cast<Index>(i);
        }
    }
}

// Compute negative log10 p-values (for visualization)
void neglog10_pvalues(
    Array<const Real> p_values,
    Array<Real> neglog_p
) {
    SCL_CHECK_DIM(p_values.len == neglog_p.len,
        "p_values and neglog_p must have same length");

    const Real log10_e = Real(0.43429448190325182765);

    for (Size i = 0; i < p_values.len; ++i) {
        Real p = detail::clamp_pvalue(p_values.ptr[i]);
        neglog_p.ptr[i] = -std::log(p) * log10_e;
    }
}

// Fisher's method for combining p-values
Real fisher_combine(Array<const Real> p_values) {
    if (p_values.len == 0) return Real(1.0);

    Real chi2_stat = Real(0.0);
    for (Size i = 0; i < p_values.len; ++i) {
        Real p = detail::clamp_pvalue(p_values.ptr[i]);
        chi2_stat -= Real(2.0) * std::log(p);
    }

    // Degrees of freedom = 2 * n
    // Return the test statistic (p-value computation requires chi2 CDF)
    return chi2_stat;
}

// Stouffer's method for combining z-scores
Real stouffer_combine(
    Array<const Real> p_values,
    Array<const Real> weights
) {
    if (p_values.len == 0) return Real(0.0);

    SCL_CHECK_DIM(weights.len == 0 || weights.len == p_values.len,
        "weights must be empty or same length as p_values");

    Real z_sum = Real(0.0);
    Real weight_sum = Real(0.0);

    for (Size i = 0; i < p_values.len; ++i) {
        Real z = detail::pvalue_to_zscore(p_values.ptr[i]);
        Real w = (weights.len > 0) ? weights.ptr[i] : Real(1.0);
        z_sum += w * z;
        weight_sum += w * w;
    }

    if (weight_sum > Real(0.0)) {
        return z_sum / std::sqrt(weight_sum);
    }
    return Real(0.0);
}

} // namespace scl::kernel::multiple_testing
