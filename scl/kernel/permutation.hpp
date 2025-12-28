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
//
// OPTIMIZATIONS vs ORIGINAL:
//   1. SIMD-accelerated p-value computation (AVX2/AVX fallback)
//   2. Branchless comparisons for better pipelining
//   3. Parallel permutation_correlation_test (was sequential)
//   4. Adaptive early termination for clearly significant/non-significant
//   5. Lemire's nearly divisionless bounded random (2-3x faster)
//   6. Loop unrolling (4x) in hot paths
//   7. Fused statistics computation (single-pass)
//   8. Jump function for independent parallel RNG streams
//   9. Kahan summation in BY correction for numerical stability
//  10. Precomputed x-statistics in correlation test
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
    
    // Adaptive early stopping
    constexpr Size EARLY_CHECK_INTERVAL = 100;
    constexpr Real EARLY_STOP_ALPHA = Real(0.001);
    constexpr Real EARLY_STOP_BETA = Real(0.5);
}

// =============================================================================
// Fast PRNG (Xoshiro256++) with jump() for parallel streams
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

    // Lemire's nearly divisionless method - avoids expensive modulo
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        uint64_t x = (*this)();
        __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
        uint64_t l = static_cast<uint64_t>(m);
        if (l < n) {
            uint64_t t = -n % n;
            while (l < t) {
                x = (*this)();
                m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
                l = static_cast<uint64_t>(m);
            }
        }
        return static_cast<Size>(m >> 64);
    }

    // Jump 2^128 steps - essential for parallel independent streams
    void jump() noexcept {
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
        };

        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= state_[0]; s1 ^= state_[1];
                    s2 ^= state_[2]; s3 ^= state_[3];
                }
                (*this)();
            }
        }

        state_[0] = s0; state_[1] = s1;
        state_[2] = s2; state_[3] = s3;
    }

private:
    alignas(32) uint64_t state_[4];

    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
};

// =============================================================================
// Fisher-Yates Shuffle - 4x unrolled
// =============================================================================

SCL_FORCE_INLINE void shuffle_indices(Index* indices, Size n, FastRNG& rng) noexcept {
    Size i = n - 1;
    for (; i >= 4; i -= 4) {
        Size j0 = rng.bounded(i + 1), j1 = rng.bounded(i);
        Size j2 = rng.bounded(i - 1), j3 = rng.bounded(i - 2);
        
        Index t0 = indices[i];     indices[i] = indices[j0];     indices[j0] = t0;
        Index t1 = indices[i-1];   indices[i-1] = indices[j1];   indices[j1] = t1;
        Index t2 = indices[i-2];   indices[i-2] = indices[j2];   indices[j2] = t2;
        Index t3 = indices[i-3];   indices[i-3] = indices[j3];   indices[j3] = t3;
    }

    for (; i > 0; --i) {
        Size j = rng.bounded(i + 1);
        Index tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// =============================================================================
// SIMD P-value Counting (using AVX when available for best performance)
// =============================================================================

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>

SCL_FORCE_INLINE Size count_geq_simd(const Real* data, Size n, Real thresh) noexcept {
    Size count = 0;
    const __m256d tv = _mm256_set1_pd(thresh);
    Size i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        count += __builtin_popcount(_mm256_movemask_pd(_mm256_cmp_pd(v, tv, _CMP_GE_OQ)));
    }

    for (; i < n; ++i) {
        count += (data[i] >= thresh);
    }

    return count;
}

SCL_FORCE_INLINE Size count_abs_geq_simd(const Real* data, Size n, Real thresh) noexcept {
    Size count = 0;
    const __m256d tv = _mm256_set1_pd(thresh);
    const __m256d sign = _mm256_set1_pd(-0.0);
    Size i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_andnot_pd(sign, _mm256_loadu_pd(data + i));
        count += __builtin_popcount(_mm256_movemask_pd(_mm256_cmp_pd(v, tv, _CMP_GE_OQ)));
    }

    for (; i < n; ++i) {
        Real av = (data[i] >= 0) ? data[i] : -data[i];
        count += (av >= thresh);
    }

    return count;
}

#else  // Scalar fallback with unrolling

SCL_FORCE_INLINE Size count_geq_simd(const Real* data, Size n, Real thresh) noexcept {
    Size count = 0;
    Size i = 0;

    for (; i + 4 <= n; i += 4) {
        count += (data[i] >= thresh) + (data[i+1] >= thresh) +
                 (data[i+2] >= thresh) + (data[i+3] >= thresh);
    }

    for (; i < n; ++i) {
        count += (data[i] >= thresh);
    }

    return count;
}

SCL_FORCE_INLINE Size count_abs_geq_simd(const Real* data, Size n, Real thresh) noexcept {
    Size count = 0;
    Size i = 0;

    for (; i + 4 <= n; i += 4) {
        Real a0 = (data[i] >= 0) ? data[i] : -data[i];
        Real a1 = (data[i+1] >= 0) ? data[i+1] : -data[i+1];
        Real a2 = (data[i+2] >= 0) ? data[i+2] : -data[i+2];
        Real a3 = (data[i+3] >= 0) ? data[i+3] : -data[i+3];
        count += (a0 >= thresh) + (a1 >= thresh) + (a2 >= thresh) + (a3 >= thresh);
    }

    for (; i < n; ++i) {
        Real av = (data[i] >= 0) ? data[i] : -data[i];
        count += (av >= thresh);
    }

    return count;
}

#endif

SCL_FORCE_INLINE Real compute_one_sided_pvalue(
    Real observed,
    const Real* null_dist,
    Size n_perm,
    bool greater
) noexcept {
    Size count = 0;

    if (greater) {
        count = count_geq_simd(null_dist, n_perm, observed);
    } else {
        // Count values <= observed
        for (Size i = 0; i < n_perm; ++i) {
            count += (null_dist[i] <= observed);
        }
    }

    return static_cast<Real>(count + 1) / static_cast<Real>(n_perm + 1);
}

SCL_FORCE_INLINE Real compute_two_sided_pvalue(
    Real observed,
    const Real* null_dist,
    Size n_perm
) noexcept {
    Real abs_obs = (observed >= 0) ? observed : -observed;
    Size count = count_abs_geq_simd(null_dist, n_perm, abs_obs);
    return static_cast<Real>(count + 1) / static_cast<Real>(n_perm + 1);
}

// =============================================================================
// Branchless Group Statistics
// =============================================================================

template <typename T>
SCL_FORCE_INLINE Real compute_mean_diff(
    const T* values,
    const Index* indices,
    Size n,
    const Index* group_mask,
    Size n_group1
) noexcept {
    Real sum1 = 0;
    Real sum2 = 0;
    Size count1 = 0;
    Size count2 = 0;
    
    Size i = 0;
    for (; i + 4 <= n; i += 4) {
        Real v0 = static_cast<Real>(values[i]);
        Real v1 = static_cast<Real>(values[i+1]);
        Real v2 = static_cast<Real>(values[i+2]);
        Real v3 = static_cast<Real>(values[i+3]);
        
        int g0 = (group_mask[indices[i]] == 0);
        int g1 = (group_mask[indices[i+1]] == 0);
        int g2 = (group_mask[indices[i+2]] == 0);
        int g3 = (group_mask[indices[i+3]] == 0);
        
        sum1 += g0*v0 + g1*v1 + g2*v2 + g3*v3;
        sum2 += (1-g0)*v0 + (1-g1)*v1 + (1-g2)*v2 + (1-g3)*v3;
        count1 += g0 + g1 + g2 + g3;
        count2 += (1-g0) + (1-g1) + (1-g2) + (1-g3);
    }

    for (; i < n; ++i) {
        Real v = static_cast<Real>(values[i]);
        int g = (group_mask[indices[i]] == 0);
        sum1 += g * v;
        sum2 += (1 - g) * v;
        count1 += g;
        count2 += (1 - g);
    }
    
    Real mean1 = (count1 > 0) ? sum1 / static_cast<Real>(count1) : Real(0);
    Real mean2 = (count2 > 0) ? sum2 / static_cast<Real>(count2) : Real(0);
    return mean1 - mean2;
}

// =============================================================================
// Fast Correlation with Precomputed X Stats
// =============================================================================

SCL_FORCE_INLINE Real compute_correlation_fast(
    const Real* x,
    const Real* y_perm,
    Size n,
    Real mean_x,
    Real std_x
) noexcept {
    Real sum_y = 0;
    Real sum_y2 = 0;
    Real sum_xy = 0;
    
    Size i = 0;
    for (; i + 4 <= n; i += 4) {
        Real y0 = y_perm[i];
        Real y1 = y_perm[i+1];
        Real y2 = y_perm[i+2];
        Real y3 = y_perm[i+3];
        sum_y += y0 + y1 + y2 + y3;
        sum_y2 += y0*y0 + y1*y1 + y2*y2 + y3*y3;
        sum_xy += x[i]*y0 + x[i+1]*y1 + x[i+2]*y2 + x[i+3]*y3;
    }

    for (; i < n; ++i) {
        sum_y += y_perm[i];
        sum_y2 += y_perm[i] * y_perm[i];
        sum_xy += x[i] * y_perm[i];
    }
    
    Real inv_n = Real(1) / static_cast<Real>(n);
    Real mean_y = sum_y * inv_n;
    Real std_y = std::sqrt(sum_y2 * inv_n - mean_y * mean_y);
    
    if (std_x > Real(1e-15) && std_y > Real(1e-15)) {
        return (sum_xy * inv_n - mean_x * mean_y) / (std_x * std_y);
    }

    return Real(0);
}

} // namespace detail

// =============================================================================
// Generic Permutation Test with Adaptive Early Stopping
// =============================================================================

template <typename StatFunc>
Real permutation_test(
    StatFunc&& compute_statistic,
    Array<Index> labels,
    Real observed_statistic,
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS,
    bool two_sided = true,
    uint64_t seed = 42,
    bool enable_early_stop = true
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
    Real abs_obs = (observed_statistic >= 0) ? observed_statistic : -observed_statistic;
    Size actual_perms = n_permutations;

    for (Size p = 0; p < n_permutations; ++p) {
        detail::shuffle_indices(perm, n, rng);
        null_dist[p] = compute_statistic(Array<const Index>(perm, n));
        
        // Early stopping check
        if (enable_early_stop && ((p + 1) % config::EARLY_CHECK_INTERVAL == 0)) {
            Size checked = p + 1;
            Size extreme = two_sided 
                ? detail::count_abs_geq_simd(null_dist, checked, abs_obs)
                : detail::count_geq_simd(null_dist, checked, observed_statistic);
            
            Real pval = static_cast<Real>(extreme + 1) / static_cast<Real>(checked + 1);
            if (pval < config::EARLY_STOP_ALPHA || pval > config::EARLY_STOP_BETA) {
                actual_perms = checked;
                break;
            }
        }
    }

    Real p_value = two_sided 
        ? detail::compute_two_sided_pvalue(observed_statistic, null_dist, actual_perms)
        : detail::compute_one_sided_pvalue(observed_statistic, null_dist, actual_perms, true);

    scl::memory::aligned_free(perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(null_dist, SCL_ALIGNMENT);

    return p_value;
}

// =============================================================================
// Parallel Correlation Test (NEW: was sequential in original)
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

    // Precompute x statistics once
    Real sum_x = 0;
    Real sum_x2 = 0;
    for (Size i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_x2 += x[i] * x[i];
    }
    Real mean_x = sum_x / static_cast<Real>(n);
    Real std_x = std::sqrt(sum_x2 / static_cast<Real>(n) - mean_x * mean_x);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    
    // Parallel path for large n_permutations
    if (n_permutations >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        Real* null_dist = scl::memory::aligned_alloc<Real>(n_permutations, SCL_ALIGNMENT);
        const Size perms_per_thread = (n_permutations + n_threads - 1) / n_threads;
        
        scl::threading::WorkspacePool<Index> perm_pool;
        scl::threading::WorkspacePool<Real> y_pool;
        perm_pool.init(n_threads, n);
        y_pool.init(n_threads, n);
        
        scl::threading::parallel_for(Size(0), n_threads, [&](size_t t, size_t) {
            Index* perm = perm_pool.get(t);
            Real* y_perm = y_pool.get(t);
            for (Size i = 0; i < n; ++i) {
                perm[i] = static_cast<Index>(i);
            }
            
            detail::FastRNG rng(seed);
            for (size_t j = 0; j < t; ++j) {
                rng.jump();
            }
            
            Size start = t * perms_per_thread;
            Size end = scl::algo::min2(start + perms_per_thread, n_permutations);
            
            for (Size p = start; p < end; ++p) {
                detail::shuffle_indices(perm, n, rng);
                for (Size i = 0; i < n; ++i) {
                    y_perm[i] = y[perm[i]];
                }
                null_dist[p] = detail::compute_correlation_fast(x.ptr, y_perm, n, mean_x, std_x);
            }
        });
        
        Real p_value = detail::compute_two_sided_pvalue(observed_correlation, null_dist, n_permutations);
        scl::memory::aligned_free(null_dist, SCL_ALIGNMENT);
        return p_value;
    }
    
    // Sequential path
    Real* null_dist = scl::memory::aligned_alloc<Real>(n_permutations, SCL_ALIGNMENT);
    Index* perm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* y_perm = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Size i = 0; i < n; ++i) {
        perm[i] = static_cast<Index>(i);
    }

    detail::FastRNG rng(seed);

    for (Size p = 0; p < n_permutations; ++p) {
        detail::shuffle_indices(perm, n, rng);
        for (Size i = 0; i < n; ++i) {
            y_perm[i] = y[perm[i]];
        }
        null_dist[p] = detail::compute_correlation_fast(x.ptr, y_perm, n, mean_x, std_x);
    }

    Real p_value = detail::compute_two_sided_pvalue(observed_correlation, null_dist, n_permutations);

    scl::memory::aligned_free(y_perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm, SCL_ALIGNMENT);
    scl::memory::aligned_free(null_dist, SCL_ALIGNMENT);

    return p_value;
}

// =============================================================================
// FDR Corrections
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
        Real adj = scl::algo::min2(p_values[idx] * static_cast<Real>(n) / static_cast<Real>(i), Real(1));
        cummin = scl::algo::min2(cummin, adj);
        q_values[idx] = cummin;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

inline void fdr_correction_by(
    Array<const Real> p_values,
    Array<Real> q_values
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(q_values.len >= n, "FDR: output buffer too small");

    if (n == 0) return;

    // Kahan summation for harmonic sum
    Real cn = 0;
    Real c = 0;
    for (Size i = 1; i <= n; ++i) {
        Real y = Real(1) / static_cast<Real>(i) - c;
        Real t = cn + y;
        c = (t - cn) - y;
        cn = t;
    }

    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    scl::argsort::argsort(p_values, Array<Index>(order, n));

    Real cn_n = cn * static_cast<Real>(n);
    Real cummin = Real(1);
    for (Size i = n; i > 0; --i) {
        Size idx = static_cast<Size>(order[i - 1]);
        Real adj = scl::algo::min2(p_values[idx] * cn_n / static_cast<Real>(i), Real(1));
        cummin = scl::algo::min2(cummin, adj);
        q_values[idx] = cummin;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

// =============================================================================
// Bonferroni (SIMD-optimized)
// =============================================================================

inline void bonferroni_correction(
    Array<const Real> p_values,
    Array<Real> adjusted
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(adjusted.len >= n, "Bonferroni: output buffer too small");

    const Real n_real = static_cast<Real>(n);

#if defined(__AVX2__) || defined(__AVX__)
    const __m256d nv = _mm256_set1_pd(n_real);
    const __m256d one = _mm256_set1_pd(1.0);
    Size i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d p = _mm256_loadu_pd(p_values.ptr + i);
        _mm256_storeu_pd(adjusted.ptr + i, _mm256_min_pd(_mm256_mul_pd(p, nv), one));
    }

    for (; i < n; ++i) {
        adjusted[i] = scl::algo::min2(p_values[i] * n_real, Real(1));
    }
#else
    for (Size i = 0; i < n; ++i) {
        adjusted[i] = scl::algo::min2(p_values[i] * n_real, Real(1));
    }
#endif
}

// =============================================================================
// Holm-Bonferroni
// =============================================================================

inline void holm_correction(
    Array<const Real> p_values,
    Array<Real> adjusted
) {
    const Size n = p_values.len;
    SCL_CHECK_DIM(adjusted.len >= n, "Holm: output buffer too small");

    if (n == 0) return;

    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    scl::argsort::argsort(p_values, Array<Index>(order, n));

    Real cummax = 0;
    for (Size i = 0; i < n; ++i) {
        Size idx = static_cast<Size>(order[i]);
        Real adj = scl::algo::min2(p_values[idx] * static_cast<Real>(n - i), Real(1));
        cummax = scl::algo::max2(cummax, adj);
        adjusted[idx] = cummax;
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
}

// =============================================================================
// Utilities (SIMD-optimized)
// =============================================================================

inline Size count_significant(
    Array<const Real> p_values,
    Real alpha = Real(0.05)
) {
    Size count = 0;

#if defined(__AVX2__) || defined(__AVX__)
    const __m256d av = _mm256_set1_pd(alpha);
    Size i = 0;

    for (; i + 4 <= p_values.len; i += 4) {
        __m256d p = _mm256_loadu_pd(p_values.ptr + i);
        count += __builtin_popcount(_mm256_movemask_pd(_mm256_cmp_pd(p, av, _CMP_LT_OQ)));
    }

    for (; i < p_values.len; ++i) {
        count += (p_values[i] < alpha);
    }
#else
    for (Size i = 0; i < p_values.len; ++i) {
        count += (p_values[i] < alpha);
    }
#endif

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
        if (p_values[i] < alpha && n_significant < indices.len) {
            indices[n_significant++] = static_cast<Index>(i);
        }
    }
}

// =============================================================================
// Parallel Batch Test
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

    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n_rows), "p_values buffer too small");
    SCL_CHECK_DIM(group_labels.len >= static_cast<Size>(n_cols), "group_labels size mismatch");

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> null_pool;
    scl::threading::WorkspacePool<Index> perm_pool;
    null_pool.init(n_threads, n_permutations);
    perm_pool.init(n_threads, static_cast<Size>(n_cols));

    Size total_group0 = 0;
    for (Size j = 0; j < static_cast<Size>(n_cols); ++j) {
        total_group0 += (group_labels[j] == 0);
    }

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i, size_t rank) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        if (len == 0) {
            p_values[i] = Real(1);
            return;
        }

        Real* null_dist = null_pool.get(rank);
        Index* perm = perm_pool.get(rank);

        for (Size j = 0; j < static_cast<Size>(n_cols); ++j) {
            perm[j] = group_labels[j];
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        Real obs = detail::compute_mean_diff(
            values.ptr, indices.ptr, static_cast<Size>(len),
            group_labels.ptr, total_group0
        );

        detail::FastRNG rng(seed);
        for (size_t j = 0; j < rank; ++j) {
            rng.jump();
        }
        for (size_t j = 0; j < (i % 16); ++j) {
            rng();
        }

        for (Size p = 0; p < n_permutations; ++p) {
            detail::shuffle_indices(perm, static_cast<Size>(n_cols), rng);
            null_dist[p] = detail::compute_mean_diff(
                values.ptr, indices.ptr, 
                static_cast<Size>(len), perm, total_group0
            );
        }

        p_values[i] = detail::compute_two_sided_pvalue(obs, null_dist, n_permutations);
    });
}

} // namespace scl::kernel::permutation
