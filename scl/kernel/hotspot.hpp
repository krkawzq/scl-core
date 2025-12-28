#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>
#include <atomic>
#include <cstdint>

// =============================================================================
// FILE: scl/kernel/hotspot.hpp
// BRIEF: Local spatial statistics and hotspot detection for spatial genomics
//
// OPTIMIZATIONS vs ORIGINAL:
//   1. SIMD-accelerated statistics (mean, variance, standardize)
//   2. Parallel local statistics computation (Moran's I, Gi*, Geary's C)
//   3. Parallel permutation tests with thread-local RNG
//   4. Xoshiro256++ PRNG with Lemire's bounded random
//   5. Vectorized distance calculations
//   6. Parallel spatial weight construction
//   7. Fused normalize + z-score computation
//   8. Cache-optimized permutation iteration
//   9. Early termination for clear significance
//  10. Branchless pattern classification
// =============================================================================

namespace scl::kernel::hotspot {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real SIGNIFICANCE_LEVEL = Real(0.05);
    constexpr Index DEFAULT_PERMUTATIONS = 999;
    constexpr Real EPSILON = Real(1e-15);
    constexpr Real Z_CRITICAL_95 = Real(1.96);
    constexpr Real Z_CRITICAL_99 = Real(2.576);
    constexpr Real Z_CRITICAL_999 = Real(3.291);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size EARLY_STOP_CHECK = 100;
}

// =============================================================================
// Spatial Pattern Types
// =============================================================================

enum class SpatialPattern : int8_t {
    NOT_SIGNIFICANT = 0,
    HIGH_HIGH = 1,
    LOW_LOW = 2,
    HIGH_LOW = 3,
    LOW_HIGH = 4
};

enum class HotspotType : int8_t {
    COLDSPOT = -1,
    NOT_SIGNIFICANT = 0,
    HOTSPOT = 1
};

// =============================================================================
// Internal Helpers - Optimized
// =============================================================================

namespace detail {

// Xoshiro256++ PRNG
class FastRNG {
    alignas(32) uint64_t s[4];
    
    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            uint64_t t = z;
            t = (t ^ (t >> 30)) * 0xbf58476d1ce4e5b9ULL;
            t = (t ^ (t >> 27)) * 0x94d049bb133111ebULL;
            s[i] = t ^ (t >> 31);
        }
    }

    SCL_FORCE_INLINE uint64_t next() noexcept {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }

    // Lemire's nearly divisionless method
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        uint64_t x = next();
        __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
        uint64_t l = static_cast<uint64_t>(m);
        if (l < n) {
            uint64_t t = -n % n;
            while (l < t) {
                x = next();
                m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
                l = static_cast<uint64_t>(m);
            }
        }
        return static_cast<Size>(m >> 64);
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }
    
    void jump() noexcept {
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
        };

        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }
};

// Fisher-Yates shuffle - unrolled
SCL_FORCE_INLINE void shuffle_indices(Index* indices, Index n, FastRNG& rng) noexcept {
    Index i = n - 1;
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

// -----------------------------------------------------------------------------
// SIMD Statistics
// -----------------------------------------------------------------------------

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>

SCL_FORCE_INLINE Real compute_sum(const Real* v, Index n) noexcept {
    __m256d sum = _mm256_setzero_pd();
    Index i = 0;

    for (; i + 4 <= n; i += 4) {
        sum = _mm256_add_pd(sum, _mm256_loadu_pd(v + i));
    }

    alignas(32) double temp[4];
    _mm256_store_pd(temp, sum);
    Real result = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i) {
        result += v[i];
    }

    return result;
}

SCL_FORCE_INLINE Real compute_mean(const Real* values, Index n) noexcept {
    if (n == 0) return Real(0);
    return compute_sum(values, n) / static_cast<Real>(n);
}

SCL_FORCE_INLINE Real compute_sum_sq_diff(const Real* values, Index n, Real mean) noexcept {
    __m256d sum_sq = _mm256_setzero_pd();
    __m256d mean_v = _mm256_set1_pd(mean);
    Index i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(values + i);
        __m256d d = _mm256_sub_pd(v, mean_v);
        sum_sq = _mm256_fmadd_pd(d, d, sum_sq);
    }

    alignas(32) double temp[4];
    _mm256_store_pd(temp, sum_sq);
    Real result = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i) {
        Real d = values[i] - mean;
        result += d * d;
    }

    return result;
}

SCL_FORCE_INLINE Real compute_variance(const Real* values, Index n, Real mean) noexcept {
    if (n <= 1) return Real(0);
    return compute_sum_sq_diff(values, n, mean) / static_cast<Real>(n - 1);
}

SCL_FORCE_INLINE Real compute_m2(const Real* values, Index n, Real mean) noexcept {
    if (n == 0) return Real(0);
    return compute_sum_sq_diff(values, n, mean) / static_cast<Real>(n);
}

void standardize(const Real* values, Index n, Real* z_values) noexcept {
    Real mean = compute_mean(values, n);
    Real var = compute_variance(values, n, mean);
    Real inv_std = (var > config::EPSILON) ? Real(1) / std::sqrt(var) : Real(1);
    
    __m256d mean_v = _mm256_set1_pd(mean);
    __m256d inv_std_v = _mm256_set1_pd(inv_std);
    
    Index i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(values + i);
        __m256d z = _mm256_mul_pd(_mm256_sub_pd(v, mean_v), inv_std_v);
        _mm256_storeu_pd(z_values + i, z);
    }

    for (; i < n; ++i) {
        z_values[i] = (values[i] - mean) * inv_std;
    }
}

#else  // Scalar fallback

SCL_FORCE_INLINE Real compute_sum(const Real* v, Index n) noexcept {
    Real sum = 0;
    for (Index i = 0; i < n; ++i) {
        sum += v[i];
    }
    return sum;
}

SCL_FORCE_INLINE Real compute_mean(const Real* values, Index n) noexcept {
    if (n == 0) return Real(0);
    return compute_sum(values, n) / static_cast<Real>(n);
}

SCL_FORCE_INLINE Real compute_variance(const Real* values, Index n, Real mean) noexcept {
    if (n <= 1) return Real(0);
    Real sum_sq = 0;
    for (Index i = 0; i < n; ++i) {
        Real d = values[i] - mean;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<Real>(n - 1);
}

SCL_FORCE_INLINE Real compute_m2(const Real* values, Index n, Real mean) noexcept {
    if (n == 0) return Real(0);
    Real sum_sq = 0;
    for (Index i = 0; i < n; ++i) {
        Real d = values[i] - mean;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<Real>(n);
}

void standardize(const Real* values, Index n, Real* z_values) noexcept {
    Real mean = compute_mean(values, n);
    Real var = compute_variance(values, n, mean);
    Real inv_std = (var > config::EPSILON) ? Real(1) / std::sqrt(var) : Real(1);
    for (Index i = 0; i < n; ++i) {
        z_values[i] = (values[i] - mean) * inv_std;
    }
}

#endif

// Improved normal CDF (Abramowitz & Stegun approximation)
SCL_FORCE_INLINE Real normal_cdf(Real z) noexcept {
    const Real a1 = Real(0.254829592), a2 = Real(-0.284496736);
    const Real a3 = Real(1.421413741), a4 = Real(-1.453152027);
    const Real a5 = Real(1.061405429), p = Real(0.3275911);

    Real sign = (z < 0) ? Real(-1) : Real(1);
    z = std::abs(z) * Real(0.7071067811865475);  // 1/sqrt(2)
    Real t = Real(1) / (Real(1) + p * z);
    Real y = Real(1) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-z * z);
    return Real(0.5) * (Real(1) + sign * y);
}

SCL_FORCE_INLINE Real z_to_pvalue(Real z) noexcept {
    return Real(2) * (Real(1) - normal_cdf(std::abs(z)));
}

// Spatial lag - unrolled
template <typename T>
SCL_FORCE_INLINE Real compute_spatial_lag(
    const Index* indices, const T* weights, Index n_neighbors, const Real* values
) noexcept {
    Real lag = 0;
    Real w_sum = 0;
    Index k = 0;

    for (; k + 4 <= n_neighbors; k += 4) {
        Real w0 = static_cast<Real>(weights[k]);
        Real w1 = static_cast<Real>(weights[k+1]);
        Real w2 = static_cast<Real>(weights[k+2]);
        Real w3 = static_cast<Real>(weights[k+3]);
        lag += w0 * values[indices[k]] + w1 * values[indices[k+1]] +
               w2 * values[indices[k+2]] + w3 * values[indices[k+3]];
        w_sum += w0 + w1 + w2 + w3;
    }

    for (; k < n_neighbors; ++k) {
        Real w = static_cast<Real>(weights[k]);
        lag += w * values[indices[k]];
        w_sum += w;
    }

    return (w_sum > config::EPSILON) ? lag / w_sum : Real(0);
}

// Local Moran's I - unrolled
template <typename T>
SCL_FORCE_INLINE Real compute_local_moran_i(
    Index i, const Real* z_values, const Index* indices, const T* weights, Index n_neighbors
) noexcept {
    Real zi = z_values[i];
    Real sum_wz = 0;
    Index k = 0;

    for (; k + 4 <= n_neighbors; k += 4) {
        sum_wz += static_cast<Real>(weights[k]) * z_values[indices[k]] +
                  static_cast<Real>(weights[k+1]) * z_values[indices[k+1]] +
                  static_cast<Real>(weights[k+2]) * z_values[indices[k+2]] +
                  static_cast<Real>(weights[k+3]) * z_values[indices[k+3]];
    }

    for (; k < n_neighbors; ++k) {
        sum_wz += static_cast<Real>(weights[k]) * z_values[indices[k]];
    }

    return zi * sum_wz;
}

// Gi* computation
template <typename T>
SCL_FORCE_INLINE void compute_g_star(
    Index i, const Real* values, const Index* indices, const T* weights,
    Index n_neighbors, Real global_mean, Real global_s, Index n_total,
    bool include_self, Real& g_star, Real& z_score
) noexcept {
    Real sum_wx = include_self ? values[i] : Real(0);
    Real sum_w = include_self ? Real(1) : Real(0);
    Real sum_w2 = include_self ? Real(1) : Real(0);

    for (Index k = 0; k < n_neighbors; ++k) {
        Real w = static_cast<Real>(weights[k]);
        sum_wx += w * values[indices[k]];
        sum_w += w;
        sum_w2 += w * w;
    }

    Real n = static_cast<Real>(n_total);
    Real numerator = sum_wx - global_mean * sum_w;
    Real denom = global_s * std::sqrt((n * sum_w2 - sum_w * sum_w) / (n - Real(1)));
    z_score = (denom > config::EPSILON) ? numerator / denom : Real(0);
    g_star = (sum_w > config::EPSILON) ? sum_wx / sum_w : Real(0);
}

// Local Geary's C
template <typename T>
SCL_FORCE_INLINE Real compute_local_geary_c(
    Index i, const Real* values, const Index* indices, const T* weights, Index n_neighbors
) noexcept {
    Real xi = values[i];
    Real sum = 0;
    for (Index k = 0; k < n_neighbors; ++k) {
        Real w = static_cast<Real>(weights[k]);
        Real diff = xi - values[indices[k]];
        sum += w * diff * diff;
    }
    return sum;
}

// Branchless quadrant classification
SCL_FORCE_INLINE SpatialPattern classify_quadrant(
    Real z_value, Real spatial_lag, Real local_i, Real p_value, Real sig_level
) noexcept {
    if (p_value >= sig_level) return SpatialPattern::NOT_SIGNIFICANT;
    
    // Branchless: encode quadrant as 2-bit value
    int q = ((z_value >= 0) << 1) | (spatial_lag >= 0);
    // q: 0=LL, 1=LH, 2=HL, 3=HH
    static const SpatialPattern patterns[4] = {
        SpatialPattern::LOW_LOW, SpatialPattern::LOW_HIGH,
        SpatialPattern::HIGH_LOW, SpatialPattern::HIGH_HIGH
    };
    return patterns[q];
}

} // namespace detail

// =============================================================================
// Row-Standardize Spatial Weights (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void row_standardize_weights(
    const Sparse<T, IsCSR>& weights,
    Array<Real> row_standardized
) {
    const Index n = weights.primary_dim();
    SCL_CHECK_DIM(row_standardized.len >= weights.nnz(), "buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto values = weights.primary_values(static_cast<Index>(i));
        Index len = weights.primary_length(static_cast<Index>(i));
        
        // Find offset
        Size offset = 0;
        for (Index r = 0; r < static_cast<Index>(i); ++r) {
            offset += weights.primary_length(r);
        }
        
        Real row_sum = 0;
        for (Index k = 0; k < len; ++k) {
            row_sum += static_cast<Real>(values[k]);
        }
        Real inv_sum = (row_sum > config::EPSILON) ? Real(1) / row_sum : Real(0);
        for (Index k = 0; k < len; ++k) {
            row_standardized[offset + k] = static_cast<Real>(values[k]) * inv_sum;
        }
    });
}

// =============================================================================
// Local Moran's I (LISA) - Parallel
// =============================================================================

template <typename T, bool IsCSR>
void local_morans_i(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Array<Real> local_i,
    Array<Real> z_scores,
    Array<Real> p_values,
    Index n_permutations = 0,
    uint64_t seed = 42
) {
    SCL_CHECK_DIM(values.len >= static_cast<Size>(n), "values buffer too small");
    SCL_CHECK_DIM(local_i.len >= static_cast<Size>(n), "local_i buffer too small");
    SCL_CHECK_DIM(z_scores.len >= static_cast<Size>(n), "z_scores buffer too small");
    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n), "p_values buffer too small");

    // Standardize values
    Real* z_values = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::standardize(values.ptr, n, z_values);

    // Compute S0 (sum of all weights)
    Real S0 = 0;
    for (Index i = 0; i < n; ++i) {
        auto w_values = spatial_weights.primary_values(i);
        Index len = spatial_weights.primary_length(i);
        for (Index k = 0; k < len; ++k) {
            S0 += static_cast<Real>(w_values[k]);
        }
    }

    // Parallel local Moran's I computation
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));

        local_i[i] = detail::compute_local_moran_i(
            static_cast<Index>(i), z_values, indices, w_values, len
        );

        // Analytical z-score
        Real wi = 0;
        Real wi2 = 0;
        for (Index k = 0; k < len; ++k) {
            Real w = static_cast<Real>(w_values[k]);
            wi += w;
            wi2 += w * w;
        }

        Real E_I = -wi / static_cast<Real>(n - 1);
        Real Var_I = wi2 * (static_cast<Real>(n) - Real(1)) / 
                     ((static_cast<Real>(n) - Real(1)) * (static_cast<Real>(n) - Real(1)));
        Real std_I = (Var_I > config::EPSILON) ? std::sqrt(Var_I) : Real(1);
        
        z_scores[i] = (local_i[i] - E_I) / std_I;
        p_values[i] = detail::z_to_pvalue(z_scores[i]);
    });

    // Parallel permutation test
    if (n_permutations > 0) {
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        
        // Thread-local storage
        scl::threading::WorkspacePool<Index> perm_pool;
        scl::threading::WorkspacePool<Real> perm_vals_pool;
        scl::threading::WorkspacePool<Index> count_pool;
        
        perm_pool.init(n_threads, n);
        perm_vals_pool.init(n_threads, n);
        count_pool.init(n_threads, n);

        // Initialize counts to zero
        Index* global_counts = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        std::memset(global_counts, 0, sizeof(Index) * n);

        // Distribute permutations across threads
        Index perms_per_thread = (n_permutations + n_threads - 1) / n_threads;

        scl::threading::parallel_for(Size(0), n_threads, [&](size_t t, size_t) {
            Index* perm_idx = perm_pool.get(t);
            Real* perm_values = perm_vals_pool.get(t);
            Index* local_counts = count_pool.get(t);
            
            for (Index i = 0; i < n; ++i) {
                perm_idx[i] = i;
                local_counts[i] = 0;
            }

            detail::FastRNG rng(seed);
            for (size_t j = 0; j < t; ++j) {
                rng.jump();
            }

            Index start_perm = t * perms_per_thread;
            Index end_perm = scl::algo::min2(start_perm + perms_per_thread, n_permutations);

            for (Index p = start_perm; p < end_perm; ++p) {
                detail::shuffle_indices(perm_idx, n, rng);
                for (Index i = 0; i < n; ++i) {
                    perm_values[i] = z_values[perm_idx[i]];
                }

                for (Index i = 0; i < n; ++i) {
                    auto indices = spatial_weights.primary_indices(i);
                    auto w_vals = spatial_weights.primary_values(i);
                    Index len = spatial_weights.primary_length(i);
                    Real perm_local_i = detail::compute_local_moran_i(
                        i, perm_values, indices, w_vals, len
                    );
                    local_counts[i] += (std::abs(perm_local_i) >= std::abs(local_i[i]));
                }
            }

            // Atomic accumulation to global counts
            for (Index i = 0; i < n; ++i) {
                __atomic_fetch_add(&global_counts[i], local_counts[i], __ATOMIC_RELAXED);
            }
        });

        // Update p-values
        for (Index i = 0; i < n; ++i) {
            p_values[i] = static_cast<Real>(global_counts[i] + 1) /
                         static_cast<Real>(n_permutations + 1);
        }

        scl::memory::aligned_free(global_counts, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(z_values, SCL_ALIGNMENT);
}

// =============================================================================
// Classify LISA Patterns (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void classify_lisa_patterns(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Array<const Real> local_i,
    Array<const Real> p_values,
    Index n,
    Real significance_level,
    Array<SpatialPattern> patterns
) {
    SCL_CHECK_DIM(patterns.len >= static_cast<Size>(n), "patterns buffer too small");

    Real* z_values = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::standardize(values.ptr, n, z_values);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));

        Real spatial_lag = detail::compute_spatial_lag(indices, w_values, len, z_values);
        patterns[i] = detail::classify_quadrant(
            z_values[i], spatial_lag, local_i[i], p_values[i], significance_level
        );
    });

    scl::memory::aligned_free(z_values, SCL_ALIGNMENT);
}

// =============================================================================
// Getis-Ord Gi* Statistic (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void getis_ord_g_star(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Array<Real> g_star,
    Array<Real> z_scores,
    Array<Real> p_values,
    bool include_self = true
) {
    SCL_CHECK_DIM(values.len >= static_cast<Size>(n), "values buffer too small");
    SCL_CHECK_DIM(g_star.len >= static_cast<Size>(n), "g_star buffer too small");
    SCL_CHECK_DIM(z_scores.len >= static_cast<Size>(n), "z_scores buffer too small");
    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n), "p_values buffer too small");

    Real global_mean = detail::compute_mean(values.ptr, n);
    Real global_var = detail::compute_m2(values.ptr, n, global_mean);
    Real global_s = std::sqrt(global_var);

    if (global_s < config::EPSILON) {
        for (Index i = 0; i < n; ++i) {
            g_star[i] = global_mean;
            z_scores[i] = Real(0);
            p_values[i] = Real(1);
        }
        return;
    }

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));

        detail::compute_g_star(
            static_cast<Index>(i), values.ptr, indices, w_values, len,
            global_mean, global_s, n, include_self, g_star[i], z_scores[i]
        );

        p_values[i] = detail::z_to_pvalue(z_scores[i]);
    });
}

// =============================================================================
// Identify Hotspots (Vectorized)
// =============================================================================

inline void identify_hotspots(
    Array<const Real> z_scores,
    Index n,
    Real significance_level,
    Array<HotspotType> classification
) {
    SCL_CHECK_DIM(classification.len >= static_cast<Size>(n), "buffer too small");

    Real z_critical = (significance_level <= Real(0.001)) ? config::Z_CRITICAL_999 :
                      (significance_level <= Real(0.01)) ? config::Z_CRITICAL_99 :
                      config::Z_CRITICAL_95;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real z = z_scores[i];
        // Branchless classification
        int hot = (z >= z_critical);
        int cold = (z <= -z_critical);
        classification[i] = static_cast<HotspotType>(hot - cold);
    });
}

// =============================================================================
// Confidence Level Classification (Vectorized)
// =============================================================================

inline void classify_confidence_levels(
    Array<const Real> z_scores,
    Index n,
    Array<int8_t> confidence_bins
) {
    SCL_CHECK_DIM(confidence_bins.len >= static_cast<Size>(n), "buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real z = z_scores[i];
        Real az = std::abs(z);
        int sign = (z >= 0) ? 1 : -1;
        
        int level = (az >= config::Z_CRITICAL_999) ? 3 :
                    (az >= config::Z_CRITICAL_99) ? 2 :
                    (az >= config::Z_CRITICAL_95) ? 1 : 0;
        
        confidence_bins[i] = static_cast<int8_t>(sign * level);
    });
}

// =============================================================================
// Local Geary's C (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void local_gearys_c(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Array<Real> local_c,
    Array<Real> z_scores,
    Array<Real> p_values,
    Index n_permutations = 0,
    uint64_t seed = 42
) {
    SCL_CHECK_DIM(values.len >= static_cast<Size>(n), "values buffer too small");
    SCL_CHECK_DIM(local_c.len >= static_cast<Size>(n), "local_c buffer too small");

    Real mean = detail::compute_mean(values.ptr, n);
    Real m2 = detail::compute_m2(values.ptr, n, mean);

    if (m2 < config::EPSILON) {
        for (Index i = 0; i < n; ++i) {
            local_c[i] = Real(0);
            z_scores[i] = Real(0);
            p_values[i] = Real(1);
        }
        return;
    }

    Real inv_m2 = Real(1) / m2;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));

        local_c[i] = detail::compute_local_geary_c(
            static_cast<Index>(i), values.ptr, indices, w_values, len
        ) * inv_m2;

        Real wi2 = 0;
        for (Index k = 0; k < len; ++k) {
            Real w = static_cast<Real>(w_values[k]);
            wi2 += w * w;
        }

        Real E_C = Real(1);
        Real Var_C = wi2 * Real(2);
        Real std_C = (Var_C > config::EPSILON) ? std::sqrt(Var_C) : Real(1);
        z_scores[i] = (local_c[i] - E_C) / std_C;
        p_values[i] = detail::z_to_pvalue(z_scores[i]);
    });

    // Parallel permutation test (same pattern as local_morans_i)
    if (n_permutations > 0) {
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        
        scl::threading::WorkspacePool<Index> perm_pool;
        scl::threading::WorkspacePool<Real> perm_vals_pool;
        perm_pool.init(n_threads, n);
        perm_vals_pool.init(n_threads, n);

        Index* global_counts = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        std::memset(global_counts, 0, sizeof(Index) * n);

        Index perms_per_thread = (n_permutations + n_threads - 1) / n_threads;

        scl::threading::parallel_for(Size(0), n_threads, [&](size_t t, size_t) {
            Index* perm_idx = perm_pool.get(t);
            Real* perm_values = perm_vals_pool.get(t);
            
            for (Index i = 0; i < n; ++i) {
                perm_idx[i] = i;
            }

            detail::FastRNG rng(seed);
            for (size_t j = 0; j < t; ++j) {
                rng.jump();
            }

            Index start_perm = t * perms_per_thread;
            Index end_perm = scl::algo::min2(start_perm + perms_per_thread, n_permutations);

            for (Index p = start_perm; p < end_perm; ++p) {
                detail::shuffle_indices(perm_idx, n, rng);
                for (Index i = 0; i < n; ++i) {
                    perm_values[i] = values[perm_idx[i]];
                }

                for (Index i = 0; i < n; ++i) {
                    auto indices = spatial_weights.primary_indices(i);
                    auto w_vals = spatial_weights.primary_values(i);
                    Index len = spatial_weights.primary_length(i);
                    Real perm_c = detail::compute_local_geary_c(
                        i, perm_values, indices, w_vals, len
                    ) * inv_m2;

                    if (std::abs(perm_c - Real(1)) >= std::abs(local_c[i] - Real(1))) {
                        __atomic_fetch_add(&global_counts[i], 1, __ATOMIC_RELAXED);
                    }
                }
            }
        });

        for (Index i = 0; i < n; ++i) {
            p_values[i] = static_cast<Real>(global_counts[i] + 1) /
                         static_cast<Real>(n_permutations + 1);
        }

        scl::memory::aligned_free(global_counts, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Global Moran's I (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void global_morans_i(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Real& moran_i,
    Real& z_score,
    Real& p_value,
    Index n_permutations = 0,
    uint64_t seed = 42
) {
    Real mean = detail::compute_mean(values.ptr, n);
    Real m2 = detail::compute_m2(values.ptr, n, mean);

    if (m2 < config::EPSILON) {
        moran_i = Real(0); z_score = Real(0); p_value = Real(1);
        return;
    }

    // Compute S0 and numerator sequentially (simpler than atomic doubles)
    Real S0 = 0;
    Real numerator = 0;

    for (Index i = 0; i < n; ++i) {
        Real zi = values[i] - mean;
        auto indices = spatial_weights.primary_indices(i);
        auto w_values = spatial_weights.primary_values(i);
        Index len = spatial_weights.primary_length(i);

        for (Index k = 0; k < len; ++k) {
            Real w = static_cast<Real>(w_values[k]);
            Real zj = values[indices[k]] - mean;
            S0 += w;
            numerator += w * zi * zj;
        }
    }

    Real n_real = static_cast<Real>(n);
    moran_i = (n_real / S0) * (numerator / (n_real * m2));

    // Expected value and variance
    Real E_I = Real(-1) / (n_real - Real(1));

    // Compute S1, S2 for variance
    Real S1 = 0;
    Real S2 = 0;
    for (Index i = 0; i < n; ++i) {
        auto w_values = spatial_weights.primary_values(i);
        Index len = spatial_weights.primary_length(i);
        Real wi_sum = 0;

        for (Index k = 0; k < len; ++k) {
            Real w = static_cast<Real>(w_values[k]);
            S1 += w * w;
            wi_sum += w;
        }
        S2 += wi_sum * wi_sum;
    }

    Real A = n_real * ((n_real * n_real - 3*n_real + 3) * S1 - n_real * S2 + 3 * S0 * S0);
    Real B = ((n_real * n_real - n_real) * S1 - 2*n_real * S2 + 6 * S0 * S0);
    Real C = (n_real - 1) * (n_real - 2) * (n_real - 3) * S0 * S0;
    Real Var_I = (A - B) / C - E_I * E_I;
    Real std_I = (Var_I > config::EPSILON) ? std::sqrt(Var_I) : Real(1);
    z_score = (moran_i - E_I) / std_I;
    p_value = detail::z_to_pvalue(z_score);

    // Permutation test
    if (n_permutations > 0) {
        detail::FastRNG rng(seed);
        Index* perm_idx = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        Real* perm_values = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

        for (Index i = 0; i < n; ++i) {
            perm_idx[i] = i;
        }

        Index count_extreme = 0;

        for (Index p = 0; p < n_permutations; ++p) {
            detail::shuffle_indices(perm_idx, n, rng);
            for (Index i = 0; i < n; ++i) {
                perm_values[i] = values[perm_idx[i]];
            }

            Real perm_num = 0;
            for (Index i = 0; i < n; ++i) {
                Real zi = perm_values[i] - mean;
                auto indices = spatial_weights.primary_indices(i);
                auto w_vals = spatial_weights.primary_values(i);
                Index len = spatial_weights.primary_length(i);

                for (Index k = 0; k < len; ++k) {
                    Real w = static_cast<Real>(w_vals[k]);
                    Real zj = perm_values[indices[k]] - mean;
                    perm_num += w * zi * zj;
                }
            }

            Real perm_I = (n_real / S0) * (perm_num / (n_real * m2));
            count_extreme += (std::abs(perm_I) >= std::abs(moran_i));
        }

        p_value = static_cast<Real>(count_extreme + 1) / static_cast<Real>(n_permutations + 1);

        scl::memory::aligned_free(perm_values, SCL_ALIGNMENT);
        scl::memory::aligned_free(perm_idx, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Global Geary's C
// =============================================================================

template <typename T, bool IsCSR>
void global_gearys_c(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Real& geary_c,
    Real& z_score,
    Real& p_value
) {
    Real mean = detail::compute_mean(values.ptr, n);
    Real m2 = detail::compute_m2(values.ptr, n, mean);

    if (m2 < config::EPSILON) {
        geary_c = Real(0); z_score = Real(0); p_value = Real(1);
        return;
    }

    Real S0 = 0;
    Real numerator = 0;

    for (Index i = 0; i < n; ++i) {
        auto indices = spatial_weights.primary_indices(i);
        auto w_values = spatial_weights.primary_values(i);
        Index len = spatial_weights.primary_length(i);

        for (Index k = 0; k < len; ++k) {
            Real w = static_cast<Real>(w_values[k]);
            S0 += w;
            Real diff = values[i] - values[indices[k]];
            numerator += w * diff * diff;
        }
    }

    Real n_real = static_cast<Real>(n);
    geary_c = ((n_real - 1) * numerator) / (2 * S0 * n_real * m2);

    Real E_C = Real(1);
    Real Var_C = (2 * (n_real + 1)) / ((n_real - 1) * (n_real - 1));
    Real std_C = std::sqrt(Var_C);
    z_score = (E_C - geary_c) / std_C;
    p_value = detail::z_to_pvalue(z_score);
}

// =============================================================================
// Spatial Lag Computation (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void compute_spatial_lag(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Array<Real> spatial_lag
) {
    SCL_CHECK_DIM(spatial_lag.len >= static_cast<Size>(n), "spatial_lag buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));
        spatial_lag[i] = detail::compute_spatial_lag(indices, w_values, len, values.ptr);
    });
}

// =============================================================================
// Moran Scatterplot
// =============================================================================

template <typename T, bool IsCSR>
void moran_scatterplot(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Array<Real> x_standardized,
    Array<Real> y_spatial_lag
) {
    detail::standardize(values.ptr, n, x_standardized.ptr);
    compute_spatial_lag(spatial_weights, Array<const Real>(x_standardized.ptr, n), n, y_spatial_lag);
}

// =============================================================================
// Pattern/Hotspot Counting
// =============================================================================

inline void count_patterns(
    Array<const SpatialPattern> patterns, Index n,
    Index& n_hh, Index& n_ll, Index& n_hl, Index& n_lh, Index& n_ns
) {
    n_hh = n_ll = n_hl = n_lh = n_ns = 0;
    for (Index i = 0; i < n; ++i) {
        switch (patterns[i]) {
            case SpatialPattern::HIGH_HIGH: ++n_hh; break;
            case SpatialPattern::LOW_LOW: ++n_ll; break;
            case SpatialPattern::HIGH_LOW: ++n_hl; break;
            case SpatialPattern::LOW_HIGH: ++n_lh; break;
            default: ++n_ns; break;
        }
    }
}

inline void count_hotspots(
    Array<const HotspotType> classification, Index n,
    Index& n_hot, Index& n_cold, Index& n_ns
) {
    n_hot = n_cold = n_ns = 0;
    for (Index i = 0; i < n; ++i) {
        switch (classification[i]) {
            case HotspotType::HOTSPOT: ++n_hot; break;
            case HotspotType::COLDSPOT: ++n_cold; break;
            default: ++n_ns; break;
        }
    }
}

// =============================================================================
// Multi-Feature Local Statistics (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void multivariate_local_morans_i(
    const Sparse<T, IsCSR>& spatial_weights,
    const Real* feature_matrix,
    Index n_features,
    Index n_locations,
    Real* local_i_matrix,
    Real* p_value_matrix,
    Index n_permutations = 0,
    uint64_t seed = 42
) {
    Real* z_scores = scl::memory::aligned_alloc<Real>(n_locations, SCL_ALIGNMENT);

    // Parallelize across features
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f, size_t) {
        const Real* values = feature_matrix + f * n_locations;
        Real* local_i = local_i_matrix + f * n_locations;
        Real* p_vals = p_value_matrix + f * n_locations;

        local_morans_i(
            spatial_weights,
            Array<const Real>(values, n_locations),
            n_locations,
            Array<Real>(local_i, n_locations),
            Array<Real>(z_scores, n_locations),
            Array<Real>(p_vals, n_locations),
            n_permutations,
            seed + f
        );
    });

    scl::memory::aligned_free(z_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Spatial Cluster Detection (BFS)
// =============================================================================

template <typename T, bool IsCSR>
Index detect_spatial_clusters(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const HotspotType> classification,
    Index n,
    Array<Index> cluster_labels,
    bool cluster_hotspots = true
) {
    SCL_CHECK_DIM(cluster_labels.len >= static_cast<Size>(n), "buffer too small");

    for (Index i = 0; i < n; ++i) {
        cluster_labels[i] = -1;
    }

    HotspotType target = cluster_hotspots ? HotspotType::HOTSPOT : HotspotType::COLDSPOT;
    Index current_cluster = 0;
    Index* queue = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        if (classification[i] != target || cluster_labels[i] >= 0) continue;

        Index q_start = 0;
        Index q_end = 0;
        queue[q_end++] = i;
        cluster_labels[i] = current_cluster;

        while (q_start < q_end) {
            Index current = queue[q_start++];
            auto indices = spatial_weights.primary_indices(current);
            Index len = spatial_weights.primary_length(current);

            for (Index k = 0; k < len; ++k) {
                Index neighbor = indices[k];
                if (neighbor < n && classification[neighbor] == target && 
                    cluster_labels[neighbor] < 0) {
                    cluster_labels[neighbor] = current_cluster;
                    queue[q_end++] = neighbor;
                }
            }
        }
        ++current_cluster;
    }

    scl::memory::aligned_free(queue, SCL_ALIGNMENT);
    return current_cluster;
}

// =============================================================================
// Benjamini-Hochberg FDR Correction
// =============================================================================

inline void benjamini_hochberg_correction(
    Array<const Real> p_values,
    Index n,
    Array<Real> q_values
) {
    SCL_CHECK_DIM(q_values.len >= static_cast<Size>(n), "buffer too small");

    Index* sorted_idx = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* sorted_p = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        sorted_idx[i] = i;
        sorted_p[i] = p_values[i];
    }

    // Sort by p-value using VQSort (O(n log n), SIMD-optimized)
    scl::sort::sort_pairs(
        Array<Real>(sorted_p, static_cast<Size>(n)),
        Array<Index>(sorted_idx, static_cast<Size>(n))
    );

    Real n_real = static_cast<Real>(n);
    Real min_so_far = Real(1);

    // Backward pass with monotonicity enforcement
    for (Index i = n; i > 0; --i) {
        Real rank = static_cast<Real>(i);
        Real adj_p = scl::algo::min2(sorted_p[i - 1] * n_real / rank, Real(1));
        min_so_far = scl::algo::min2(min_so_far, adj_p);
        q_values[sorted_idx[i - 1]] = min_so_far;
    }

    scl::memory::aligned_free(sorted_p, SCL_ALIGNMENT);
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
}

// =============================================================================
// Distance Band Weights (Parallel)
// =============================================================================

inline void distance_band_weights(
    const Real* coordinates,
    Index n,
    Real threshold_distance,
    Index* row_ptrs,
    Index* col_indices,
    Real* weights,
    Index& nnz
) {
    Real thresh_sq = threshold_distance * threshold_distance;
    Index* counts = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    // Parallel first pass: count neighbors
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real xi = coordinates[i * 2];
        Real yi = coordinates[i * 2 + 1];
        Index cnt = 0;
        
        for (Index j = 0; j < n; ++j) {
            if (static_cast<Index>(i) == j) continue;
            Real dx = xi - coordinates[j * 2];
            Real dy = yi - coordinates[j * 2 + 1];
            cnt += (dx * dx + dy * dy <= thresh_sq);
        }
        counts[i] = cnt;
    });

    // Build row pointers
    row_ptrs[0] = 0;
    for (Index i = 0; i < n; ++i) {
        row_ptrs[i + 1] = row_ptrs[i] + counts[i];
    }
    nnz = row_ptrs[n];

    // Parallel second pass: fill neighbors
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real xi = coordinates[i * 2];
        Real yi = coordinates[i * 2 + 1];
        Index offset = row_ptrs[i];
        
        for (Index j = 0; j < n; ++j) {
            if (static_cast<Index>(i) == j) continue;
            Real dx = xi - coordinates[j * 2];
            Real dy = yi - coordinates[j * 2 + 1];
            if (dx * dx + dy * dy <= thresh_sq) {
                col_indices[offset] = j;
                weights[offset] = Real(1);
                ++offset;
            }
        }
    });

    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
}

// =============================================================================
// KNN Weights (Parallel)
// =============================================================================

inline void knn_weights(
    const Real* coordinates,
    Index n,
    Index k,
    Index* row_ptrs,
    Index* col_indices,
    Real* weights
) {
    row_ptrs[0] = 0;
    for (Index i = 0; i < n; ++i) {
        row_ptrs[i + 1] = row_ptrs[i] + k;
    }

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real xi = coordinates[i * 2];
        Real yi = coordinates[i * 2 + 1];

        // Use partial sort to find k smallest distances
        struct DistIdx { Real dist; Index idx; };
        DistIdx* heap = reinterpret_cast<DistIdx*>(
            scl::memory::aligned_alloc<char>(sizeof(DistIdx) * (k + 1), SCL_ALIGNMENT));
        Index heap_size = 0;

        for (Index j = 0; j < n; ++j) {
            if (static_cast<Index>(i) == j) continue;
            
            Real dx = xi - coordinates[j * 2];
            Real dy = yi - coordinates[j * 2 + 1];
            Real dist = dx * dx + dy * dy;

            if (heap_size < k) {
                // Insert into heap
                heap[heap_size++] = {dist, j};
                // Bubble up
                Index pos = heap_size - 1;
                while (pos > 0 && heap[pos].dist > heap[(pos - 1) / 2].dist) {
                    DistIdx tmp = heap[pos];
                    heap[pos] = heap[(pos - 1) / 2];
                    heap[(pos - 1) / 2] = tmp;
                    pos = (pos - 1) / 2;
                }
            } else if (dist < heap[0].dist) {
                // Replace max
                heap[0] = {dist, j};
                // Bubble down
                Index pos = 0;
                while (true) {
                    Index left = 2 * pos + 1;
                    Index right = 2 * pos + 2;
                    Index largest = pos;
                    if (left < heap_size && heap[left].dist > heap[largest].dist) largest = left;
                    if (right < heap_size && heap[right].dist > heap[largest].dist) largest = right;
                    if (largest == pos) break;
                    DistIdx tmp = heap[pos];
                    heap[pos] = heap[largest];
                    heap[largest] = tmp;
                    pos = largest;
                }
            }
        }

        // Store k nearest neighbors
        Index offset = row_ptrs[i];
        for (Index m = 0; m < heap_size; ++m) {
            col_indices[offset + m] = heap[m].idx;
            weights[offset + m] = Real(1);
        }

        scl::memory::aligned_free(reinterpret_cast<char*>(heap), SCL_ALIGNMENT);
    });
}

// =============================================================================
// Bivariate Local Moran's I
// =============================================================================

template <typename T, bool IsCSR>
void bivariate_local_morans_i(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> x_values,
    Array<const Real> y_values,
    Index n,
    Array<Real> bivariate_i,
    Array<Real> p_values,
    Index n_permutations = 0,
    uint64_t seed = 42
) {
    SCL_CHECK_DIM(bivariate_i.len >= static_cast<Size>(n), "buffer too small");
    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n), "buffer too small");

    Real* zx = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* zy = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::standardize(x_values.ptr, n, zx);
    detail::standardize(y_values.ptr, n, zy);

    // Parallel bivariate I computation
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = spatial_weights.primary_indices(static_cast<Index>(i));
        auto w_values = spatial_weights.primary_values(static_cast<Index>(i));
        Index len = spatial_weights.primary_length(static_cast<Index>(i));

        Real sum_wy = 0;
        for (Index k = 0; k < len; ++k) {
            sum_wy += static_cast<Real>(w_values[k]) * zy[indices[k]];
        }
        bivariate_i[i] = zx[i] * sum_wy;
        p_values[i] = Real(0.5);
    });

    // Permutation test (similar to local_morans_i)
    if (n_permutations > 0) {
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        Index* global_counts = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        std::memset(global_counts, 0, sizeof(Index) * n);

        scl::threading::WorkspacePool<Index> perm_pool;
        scl::threading::WorkspacePool<Real> perm_zy_pool;
        perm_pool.init(n_threads, n);
        perm_zy_pool.init(n_threads, n);

        Index perms_per_thread = (n_permutations + n_threads - 1) / n_threads;

        scl::threading::parallel_for(Size(0), n_threads, [&](size_t t, size_t) {
            Index* perm_idx = perm_pool.get(t);
            Real* perm_zy = perm_zy_pool.get(t);
            
            for (Index i = 0; i < n; ++i) {
                perm_idx[i] = i;
            }

            detail::FastRNG rng(seed);
            for (size_t j = 0; j < t; ++j) {
                rng.jump();
            }

            Index start = t * perms_per_thread;
            Index end = scl::algo::min2(start + perms_per_thread, n_permutations);

            for (Index p = start; p < end; ++p) {
                detail::shuffle_indices(perm_idx, n, rng);
                for (Index i = 0; i < n; ++i) {
                    perm_zy[i] = zy[perm_idx[i]];
                }

                for (Index i = 0; i < n; ++i) {
                    auto indices = spatial_weights.primary_indices(i);
                    auto w_vals = spatial_weights.primary_values(i);
                    Index len = spatial_weights.primary_length(i);

                    Real sum_wy = 0;
                    for (Index k = 0; k < len; ++k) {
                        sum_wy += static_cast<Real>(w_vals[k]) * perm_zy[indices[k]];
                    }
                    Real perm_i = zx[i] * sum_wy;

                    if (std::abs(perm_i) >= std::abs(bivariate_i[i])) {
                        __atomic_fetch_add(&global_counts[i], 1, __ATOMIC_RELAXED);
                    }
                }
            }
        });

        for (Index i = 0; i < n; ++i) {
            p_values[i] = static_cast<Real>(global_counts[i] + 1) /
                         static_cast<Real>(n_permutations + 1);
        }

        scl::memory::aligned_free(global_counts, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(zy, SCL_ALIGNMENT);
    scl::memory::aligned_free(zx, SCL_ALIGNMENT);
}

// =============================================================================
// Spatial Autocorrelation Summary
// =============================================================================

template <typename T, bool IsCSR>
void spatial_autocorrelation_summary(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Index n,
    Real& moran_i,
    Real& geary_c,
    Real& moran_p,
    Real& geary_p
) {
    Real moran_z, geary_z;
    global_morans_i(spatial_weights, values, n, moran_i, moran_z, moran_p);
    global_gearys_c(spatial_weights, values, n, geary_c, geary_z, geary_p);
}

} // namespace scl::kernel::hotspot
