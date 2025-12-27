#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstdint>

// =============================================================================
// FILE: scl/kernel/permutation.hpp
// BRIEF: Permutation testing framework for nonparametric inference
// =============================================================================

namespace scl::kernel::permutation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Size MIN_PERMUTATIONS = 100;
    constexpr Size MAX_PERMUTATIONS = 100000;
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Fast PRNG (Xoshiro256++)
// =============================================================================

namespace detail {

class FastRNG {
public:
    using result_type = uint64_t;

    explicit FastRNG(uint64_t seed) noexcept {
        uint64_t s = seed;
        for (int i = 0; i < 4; ++i) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            state_[i] = z ^ (z >> 31);
        }
    }

    SCL_FORCE_INLINE uint64_t operator()() noexcept {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);

        return result;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>((*this)() >> 11) * Real(0x1.0p-53);
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>((*this)() % static_cast<uint64_t>(n));
    }

private:
    uint64_t state_[4];

    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
};

// =============================================================================
// Fisher-Yates Shuffle
// =============================================================================

SCL_FORCE_INLINE void shuffle_indices(Index* indices, Size n, FastRNG& rng) noexcept {
    for (Size i = n - 1; i > 0; --i) {
        Size j = rng.bounded(i + 1);
        Index tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// =============================================================================
// P-value Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_one_sided_pvalue(
    Real observed,
    const Real* null_dist,
    Size n_perm,
    bool greater
) noexcept {
    Size count = 0;

    if (greater) {
        for (Size i = 0; i < n_perm; ++i) {
            if (null_dist[i] >= observed) ++count;
        }
    } else {
        for (Size i = 0; i < n_perm; ++i) {
            if (null_dist[i] <= observed) ++count;
        }
    }

    return static_cast<Real>(count + 1) / static_cast<Real>(n_perm + 1);
}

SCL_FORCE_INLINE Real compute_two_sided_pvalue(
    Real observed,
    const Real* null_dist,
    Size n_perm
) noexcept {
    Real abs_obs = (observed >= Real(0)) ? observed : -observed;
    Size count = 0;

    for (Size i = 0; i < n_perm; ++i) {
        Real abs_null = (null_dist[i] >= Real(0)) ? null_dist[i] : -null_dist[i];
        if (abs_null >= abs_obs) ++count;
    }

    return static_cast<Real>(count + 1) / static_cast<Real>(n_perm + 1);
}

// =============================================================================
// Group Statistics for Permutation Tests
// =============================================================================

template <typename T>
SCL_FORCE_INLINE Real compute_mean_diff(
    const T* values,
    const Index* indices,
    Size n,
    const Index* group_mask,
    Size n_group1
) noexcept {
    Real sum1 = Real(0);
    Real sum2 = Real(0);
    Size count1 = 0;
    Size count2 = 0;

    for (Size i = 0; i < n; ++i) {
        Index idx = indices[i];
        Real v = static_cast<Real>(values[i]);

        if (group_mask[idx] == 0) {
            sum1 += v;
            ++count1;
        } else {
            sum2 += v;
            ++count2;
        }
    }

    Real mean1 = (count1 > 0) ? sum1 / static_cast<Real>(count1) : Real(0);
    Real mean2 = (count2 > 0) ? sum2 / static_cast<Real>(count2) : Real(0);

    return mean1 - mean2;
}

} // namespace detail

// =============================================================================
// Generic Permutation Test
// =============================================================================

template <typename StatFunc>
Real permutation_test(
    StatFunc&& compute_statistic,
    Array<Index> labels,
    Real observed_statistic,
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS,
    bool two_sided = true,
    uint64_t seed = 42
) {
    const Size n = labels.len;
    if (n == 0) return Real(1);

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    Real* null_dist = scl::memory::aligned_alloc<Real>(n_permutations, SCL_ALIGNMENT);
    Index* perm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    for (Size i = 0; i < n; ++i) {
        perm[i] = labels[i];
    }

    detail::FastRNG rng(seed);

    for (Size p = 0; p < n_permutations; ++p) {
        detail::shuffle_indices(perm, n, rng);
        null_dist[p] = compute_statistic(Array<const Index>(perm, n));
    }

    Real p_value;
    if (two_sided) {
        p_value = detail::compute_two_sided_pvalue(observed_statistic, null_dist, n_permutations);
    } else {
        p_value = detail::compute_one_sided_pvalue(observed_statistic, null_dist, n_permutations, true);
    }

    scl::memory::aligned_free(perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(null_dist, SCL_ALIGNMENT);

    return p_value;
}

// =============================================================================
// Permutation Test for Correlation
// =============================================================================

inline Real permutation_correlation_test(
    Array<const Real> x,
    Array<const Real> y,
    Real observed_correlation,
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
) {
    const Size n = x.len;
    SCL_CHECK_DIM(y.len == n, "Permutation: x and y must have same length");

    if (n < 3) return Real(1);

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    Real* null_dist = scl::memory::aligned_alloc<Real>(n_permutations, SCL_ALIGNMENT);
    Index* perm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* y_perm = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Size i = 0; i < n; ++i) {
        perm[i] = static_cast<Index>(i);
    }

    // Precompute x statistics
    Real sum_x = Real(0), sum_x2 = Real(0);
    for (Size i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_x2 += x[i] * x[i];
    }
    Real mean_x = sum_x / static_cast<Real>(n);
    Real std_x = std::sqrt(sum_x2 / static_cast<Real>(n) - mean_x * mean_x);

    detail::FastRNG rng(seed);

    for (Size p = 0; p < n_permutations; ++p) {
        detail::shuffle_indices(perm, n, rng);

        for (Size i = 0; i < n; ++i) {
            y_perm[i] = y[perm[i]];
        }

        Real sum_y = Real(0), sum_y2 = Real(0), sum_xy = Real(0);
        for (Size i = 0; i < n; ++i) {
            sum_y += y_perm[i];
            sum_y2 += y_perm[i] * y_perm[i];
            sum_xy += x[i] * y_perm[i];
        }

        Real mean_y = sum_y / static_cast<Real>(n);
        Real std_y = std::sqrt(sum_y2 / static_cast<Real>(n) - mean_y * mean_y);
        Real cov = sum_xy / static_cast<Real>(n) - mean_x * mean_y;

        Real corr = (std_x > Real(1e-15) && std_y > Real(1e-15))
                    ? cov / (std_x * std_y)
                    : Real(0);

        null_dist[p] = corr;
    }

    Real p_value = detail::compute_two_sided_pvalue(observed_correlation, null_dist, n_permutations);

    scl::memory::aligned_free(y_perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(null_dist, SCL_ALIGNMENT);

    return p_value;
}

// =============================================================================
// FDR Correction: Benjamini-Hochberg
// =============================================================================

inline void fdr_correction_bh(
    Array<const Real> p_values,
    Array<Real> q_values
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(q_values.len >= n, "FDR: output buffer too small");

    if (n == 0) return;

    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    scl::argsort::argsort(p_values, Array<Index>(order, n));

    Real cummin = Real(1);
    for (Size i = n; i > 0; --i) {
        Size idx = static_cast<Size>(order[i - 1]);
        Real adjusted = p_values[idx] * static_cast<Real>(n) / static_cast<Real>(i);
        adjusted = scl::algo::min2(adjusted, Real(1));
        cummin = scl::algo::min2(cummin, adjusted);
        q_values[idx] = cummin;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

// =============================================================================
// FDR Correction: Benjamini-Yekutieli (for dependent tests)
// =============================================================================

inline void fdr_correction_by(
    Array<const Real> p_values,
    Array<Real> q_values
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(q_values.len >= n, "FDR: output buffer too small");

    if (n == 0) return;

    // Compute harmonic sum: sum(1/i) for i = 1..n
    Real cn = Real(0);
    for (Size i = 1; i <= n; ++i) {
        cn += Real(1) / static_cast<Real>(i);
    }

    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    scl::argsort::argsort(p_values, Array<Index>(order, n));

    Real cummin = Real(1);
    for (Size i = n; i > 0; --i) {
        Size idx = static_cast<Size>(order[i - 1]);
        Real adjusted = p_values[idx] * cn * static_cast<Real>(n) / static_cast<Real>(i);
        adjusted = scl::algo::min2(adjusted, Real(1));
        cummin = scl::algo::min2(cummin, adjusted);
        q_values[idx] = cummin;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

// =============================================================================
// Bonferroni Correction
// =============================================================================

inline void bonferroni_correction(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(adjusted_p_values.len >= n, "Bonferroni: output buffer too small");

    for (Size i = 0; i < n; ++i) {
        adjusted_p_values[i] = scl::algo::min2(p_values[i] * static_cast<Real>(n), Real(1));
    }
}

// =============================================================================
// Holm-Bonferroni Step-Down Correction
// =============================================================================

inline void holm_correction(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(adjusted_p_values.len >= n, "Holm: output buffer too small");

    if (n == 0) return;

    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    scl::argsort::argsort(p_values, Array<Index>(order, n));

    Real cummax = Real(0);
    for (Size i = 0; i < n; ++i) {
        Size idx = static_cast<Size>(order[i]);
        Real adjusted = p_values[idx] * static_cast<Real>(n - i);
        adjusted = scl::algo::min2(adjusted, Real(1));
        cummax = scl::algo::max2(cummax, adjusted);
        adjusted_p_values[idx] = cummax;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

// =============================================================================
// Multiple Comparison Utilities
// =============================================================================

inline Size count_significant(
    Array<const Real> p_values,
    Real alpha = Real(0.05)
) {
    Size count = 0;
    for (Size i = 0; i < p_values.len; ++i) {
        if (p_values[i] < alpha) ++count;
    }
    return count;
}

inline void get_significant_indices(
    Array<const Real> p_values,
    Real alpha,
    Array<Index> indices,
    Size& n_significant
) {
    n_significant = 0;
    for (Size i = 0; i < p_values.len; ++i) {
        if (p_values[i] < alpha) {
            if (n_significant < indices.len) {
                indices[n_significant] = static_cast<Index>(i);
            }
            ++n_significant;
        }
    }
}

// =============================================================================
// Parallel Permutation Test for Multiple Features
// =============================================================================

template <typename T, bool IsCSR>
void batch_permutation_test(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> group_labels,
    Size n_permutations,
    Array<Real> p_values,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "batch_permutation_test requires CSR format");

    const Index n_rows = matrix.rows();
    const Index n_cols = matrix.cols();

    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n_rows),
                  "Permutation: p_values buffer too small");
    SCL_CHECK_DIM(group_labels.len >= static_cast<Size>(n_cols),
                  "Permutation: group_labels size mismatch");

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> null_pool;
    null_pool.init(n_threads, n_permutations);

    scl::threading::WorkspacePool<Index> perm_pool;
    perm_pool.init(n_threads, static_cast<Size>(n_cols));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        if (len == 0) {
            p_values[i] = Real(1);
            return;
        }

        Real* null_dist = null_pool.get(thread_rank);
        Index* perm = perm_pool.get(thread_rank);

        for (Size j = 0; j < static_cast<Size>(n_cols); ++j) {
            perm[j] = group_labels[j];
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        // Compute observed statistic
        Real obs = detail::compute_mean_diff(
            values.ptr, indices.ptr, static_cast<Size>(len),
            group_labels.ptr, 0
        );

        // Thread-specific RNG
        detail::FastRNG rng(seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL));

        for (Size p = 0; p < n_permutations; ++p) {
            detail::shuffle_indices(perm, static_cast<Size>(n_cols), rng);

            null_dist[p] = detail::compute_mean_diff(
                values.ptr, indices.ptr, static_cast<Size>(len),
                perm, 0
            );
        }

        p_values[i] = detail::compute_two_sided_pvalue(obs, null_dist, n_permutations);
    });
}

} // namespace scl::kernel::permutation
