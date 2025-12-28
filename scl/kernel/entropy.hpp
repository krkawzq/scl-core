#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>
#include <atomic>

// =============================================================================
// FILE: scl/kernel/entropy.hpp
// BRIEF: Information theory measures for sparse data analysis
// =============================================================================

namespace scl::kernel::entropy {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real LOG_BASE_E = Real(2.718281828459045);
    constexpr Real LOG_2 = Real(0.693147180559945);
    constexpr Real INV_LOG_2 = Real(1.4426950408889634);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr size_t PREFETCH_DISTANCE = 64;
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// SIMD Operations - Use vectorize::sum for consistency
SCL_HOT SCL_FORCE_INLINE Real sum_simd(const Real* SCL_RESTRICT x, Size n) noexcept {
    return scl::vectorize::sum(Array<const Real>(x, n));
}

// Safe log functions
SCL_FORCE_INLINE Real safe_log(Real x) noexcept {
    return (x > config::EPSILON) ? std::log(x) : Real(0);
}

SCL_FORCE_INLINE Real safe_log2(Real x) noexcept {
    return (x > config::EPSILON) ? std::log(x) * config::INV_LOG_2 : Real(0);
}

// =============================================================================
// OPTIMIZATION #1: Vectorized p*log(p) computation using SIMD Log
// =============================================================================

SCL_HOT Real plogp_sum(const Real* probs, Size n, bool use_log2) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    Real H = Real(0);
    const Real scale = use_log2 ? config::INV_LOG_2 : Real(1);
    
    // Use SIMD for the main loop when n is large enough
    if (SCL_LIKELY(n >= lanes * 2)) {
        auto v_eps = s::Set(d, config::EPSILON);
        auto v_zero = s::Zero(d);
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        Size k = 0;
        
        // 2-way SIMD unroll for better ILP
        for (; k + 2 * lanes <= n; k += 2 * lanes) {
            auto p0 = s::Load(d, probs + k);
            auto p1 = s::Load(d, probs + k + lanes);
            
            // Mask for p > epsilon
            auto mask0 = s::Gt(p0, v_eps);
            auto mask1 = s::Gt(p1, v_eps);
            
            // Safe p for log (avoid log(0))
            auto safe_p0 = s::IfThenElse(mask0, p0, s::Set(d, Real(1)));
            auto safe_p1 = s::IfThenElse(mask1, p1, s::Set(d, Real(1)));
            
            // SIMD log - using scl::simd::Log
            auto log_p0 = s::Log(d, safe_p0);
            auto log_p1 = s::Log(d, safe_p1);
            
            // p * log(p), masked to zero where p <= epsilon
            auto plogp0 = s::IfThenElse(mask0, s::Mul(p0, log_p0), v_zero);
            auto plogp1 = s::IfThenElse(mask1, s::Mul(p1, log_p1), v_zero);
            
            v_sum0 = s::Add(v_sum0, plogp0);
            v_sum1 = s::Add(v_sum1, plogp1);
        }
        
        // Single SIMD lane cleanup
        for (; k + lanes <= n; k += lanes) {
            auto p = s::Load(d, probs + k);
            auto mask = s::Gt(p, v_eps);
            auto safe_p = s::IfThenElse(mask, p, s::Set(d, Real(1)));
            auto log_p = s::Log(d, safe_p);
            auto plogp = s::IfThenElse(mask, s::Mul(p, log_p), v_zero);
            v_sum0 = s::Add(v_sum0, plogp);
        }
        
        H = s::GetLane(s::SumOfLanes(d, s::Add(v_sum0, v_sum1)));
        
        // Scalar cleanup
        for (; k < n; ++k) {
            if (SCL_LIKELY(probs[k] > config::EPSILON)) {
                H += probs[k] * std::log(probs[k]);
            }
        }
    } else {
        // Scalar path with 4-way unrolling for small n
        Size k = 0;
        Real H0 = Real(0), H1 = Real(0), H2 = Real(0), H3 = Real(0);
        
        for (; k + 4 <= n; k += 4) {
            Real p0 = probs[k], p1 = probs[k+1], p2 = probs[k+2], p3 = probs[k+3];
            if (SCL_LIKELY(p0 > config::EPSILON)) H0 += p0 * std::log(p0);
            if (SCL_LIKELY(p1 > config::EPSILON)) H1 += p1 * std::log(p1);
            if (SCL_LIKELY(p2 > config::EPSILON)) H2 += p2 * std::log(p2);
            if (SCL_LIKELY(p3 > config::EPSILON)) H3 += p3 * std::log(p3);
        }
        
        H = H0 + H1 + H2 + H3;
        for (; k < n; ++k) {
            if (SCL_LIKELY(probs[k] > config::EPSILON)) {
                H += probs[k] * std::log(probs[k]);
            }
        }
    }
    
    return -H * scale;
}

// Fast PRNG (Xoshiro128+)
struct alignas(16) FastRNG {
    uint32_t s[4];
    
    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s[i] = static_cast<uint32_t>(z >> 32);
        }
    }
    
    SCL_FORCE_INLINE uint32_t next() noexcept {
        uint32_t t = s[3];
        uint32_t const x = s[0];
        s[3] = s[2];
        s[2] = s[1];
        s[1] = x;
        t ^= t >> 11;
        t ^= t << 8;
        s[0] = t ^ x ^ (x << 19);
        return s[0];
    }
    
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint32_t>(n));
    }
};

// Optimized Histogram with Thread-Local Accumulation
SCL_HOT void histogram_parallel(
    const Index* binned,
    Size n,
    Index n_bins,
    Size* counts
) {
    scl::algo::zero(counts, static_cast<Size>(n_bins));
    if (SCL_UNLIKELY(n < config::PARALLEL_THRESHOLD)) {
        for (Size i = 0; i < n; ++i) {
            Index bin = binned[i];
            if (SCL_LIKELY(bin >= 0 && bin < n_bins)) {
                ++counts[bin];
            }
        }
        return;
    }
    
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    Size** thread_counts = static_cast<Size**>(
        scl::memory::aligned_alloc<Size*>(n_threads, SCL_ALIGNMENT));
    for (size_t t = 0; t < n_threads; ++t) {
        thread_counts[t] = scl::memory::aligned_alloc<Size>(n_bins, SCL_ALIGNMENT);
        scl::algo::zero(thread_counts[t], static_cast<Size>(n_bins));
    }
    
    scl::threading::parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
        Index bin = binned[i];
        if (SCL_LIKELY(bin >= 0 && bin < n_bins)) {
            ++thread_counts[thread_rank][bin];
        }
    });
    
    // Reduce
    for (size_t t = 0; t < n_threads; ++t) {
        for (Index b = 0; b < n_bins; ++b) {
            counts[b] += thread_counts[t][b];
        }
        scl::memory::aligned_free(thread_counts[t], SCL_ALIGNMENT);
    }
    scl::memory::aligned_free(thread_counts, SCL_ALIGNMENT);
}

// 2D histogram with thread-local accumulation
SCL_HOT void histogram_2d_parallel(
    const Index* x_binned,
    const Index* y_binned,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    Size* counts
) {
    const Size total_bins = static_cast<Size>(n_bins_x) * n_bins_y;
    scl::algo::zero(counts, total_bins);
    
    if (SCL_UNLIKELY(n < config::PARALLEL_THRESHOLD)) {
        for (Size i = 0; i < n; ++i) {
            Index bx = x_binned[i];
            Index by = y_binned[i];
            if (SCL_LIKELY(bx >= 0 && bx < n_bins_x && by >= 0 && by < n_bins_y)) {
                ++counts[static_cast<Size>(bx) * n_bins_y + by];
            }
        }
        return;
    }
    
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    Size** thread_counts = static_cast<Size**>(
        scl::memory::aligned_alloc<Size*>(n_threads, SCL_ALIGNMENT));
    for (size_t t = 0; t < n_threads; ++t) {
        thread_counts[t] = scl::memory::aligned_alloc<Size>(total_bins, SCL_ALIGNMENT);
        scl::algo::zero(thread_counts[t], total_bins);
    }
    
    scl::threading::parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
        Index bx = x_binned[i];
        Index by = y_binned[i];
        if (SCL_LIKELY(bx >= 0 && bx < n_bins_x && by >= 0 && by < n_bins_y)) {
            ++thread_counts[thread_rank][static_cast<Size>(bx) * n_bins_y + by];
        }
    });
    
    for (size_t t = 0; t < n_threads; ++t) {
        for (Size i = 0; i < total_bins; ++i) {
            counts[i] += thread_counts[t][i];
        }
        scl::memory::aligned_free(thread_counts[t], SCL_ALIGNMENT);
    }
    scl::memory::aligned_free(thread_counts, SCL_ALIGNMENT);
}

// Optimized Discretization with SIMD min/max
template <typename T>
SCL_HOT void discretize_equal_width_parallel(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
) {
    if (SCL_UNLIKELY(n == 0 || n_bins == 0)) return;
    
    // Parallel min/max using vectorize
    T min_val = values[0];
    T max_val = values[0];
    if (SCL_LIKELY(n >= config::PARALLEL_THRESHOLD)) {
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        T* thread_mins = scl::memory::aligned_alloc<T>(n_threads, SCL_ALIGNMENT);
        T* thread_maxs = scl::memory::aligned_alloc<T>(n_threads, SCL_ALIGNMENT);
        for (size_t t = 0; t < n_threads; ++t) {
            thread_mins[t] = values[0];
            thread_maxs[t] = values[0];
        }
        
        scl::threading::parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
            thread_mins[thread_rank] = scl::algo::min2(thread_mins[thread_rank], values[i]);
            thread_maxs[thread_rank] = scl::algo::max2(thread_maxs[thread_rank], values[i]);
        });
        
        for (size_t t = 0; t < n_threads; ++t) {
            min_val = scl::algo::min2(min_val, thread_mins[t]);
            max_val = scl::algo::max2(max_val, thread_maxs[t]);
        }
        
        scl::memory::aligned_free(thread_maxs, SCL_ALIGNMENT);
        scl::memory::aligned_free(thread_mins, SCL_ALIGNMENT);
    } else {
        // Use SIMD for sequential min/max when T is Real
        if constexpr (std::is_same_v<T, Real>) {
            auto [minmax_min, minmax_max] = scl::vectorize::minmax(Array<const Real>(values, n));
            min_val = minmax_min;
            max_val = minmax_max;
        } else {
            for (Size i = 1; i < n; ++i) {
                min_val = scl::algo::min2(min_val, values[i]);
                max_val = scl::algo::max2(max_val, values[i]);
            }
        }
    }
    
    Real range = static_cast<Real>(max_val - min_val);
    if (SCL_UNLIKELY(range < config::EPSILON)) {
        scl::algo::zero(binned, n);
        return;
    }
    
    Real inv_bin_width = static_cast<Real>(n_bins) / range;
    Real min_real = static_cast<Real>(min_val);
    
    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        Real offset = static_cast<Real>(values[i]) - min_real;
        Index bin = static_cast<Index>(offset * inv_bin_width);
        binned[i] = scl::algo::min2(bin, n_bins - 1);
    });
}

// Shell sort for equal-frequency discretization
template <typename T>
SCL_HOT void argsort_shell(const T* values, Index* indices, Size n) {
    for (Size i = 0; i < n; ++i) {
        indices[i] = static_cast<Index>(i);
    }
    
    // Shell sort with Ciura gaps
    Size gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
    for (Size gap : gaps) {
        if (SCL_UNLIKELY(gap >= n)) continue;
        for (Size i = gap; i < n; ++i) {
            Index idx = indices[i];
            T val = values[idx];
            Size j = i;
            while (j >= gap && values[indices[j - gap]] > val) {
                indices[j] = indices[j - gap];
                j -= gap;
            }
            indices[j] = idx;
        }
    }
}

template <typename T>
SCL_HOT void discretize_equal_frequency(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
) {
    if (SCL_UNLIKELY(n == 0 || n_bins == 0)) return;
    
    Index* sorted_idx = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    argsort_shell(values, sorted_idx, n);
    
    Size items_per_bin = (n + n_bins - 1) / n_bins;
    
    // Parallel bin assignment
    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        Index original_idx = sorted_idx[i];
        Index bin = static_cast<Index>(i / items_per_bin);
        binned[original_idx] = scl::algo::min2(bin, n_bins - 1);
    });
    
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
}

// =============================================================================
// OPTIMIZATION #3: Entropy from Counts with 4-way unrolling
// =============================================================================

SCL_HOT Real entropy_from_counts(
    const Size* counts,
    Index n_bins,
    Size total,
    bool use_log2
) noexcept {
    if (SCL_UNLIKELY(total == 0)) return Real(0);
    
    Real inv_total = Real(1) / static_cast<Real>(total);
    Real log_total = std::log(static_cast<Real>(total));
    
    // 4-way unrolled sum of c*log(c)
    Real sum0 = Real(0), sum1 = Real(0), sum2 = Real(0), sum3 = Real(0);
    Index b = 0;
    
    for (; b + 4 <= n_bins; b += 4) {
        if (SCL_LIKELY(counts[b] > 0)) {
            Real c = static_cast<Real>(counts[b]);
            sum0 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[b+1] > 0)) {
            Real c = static_cast<Real>(counts[b+1]);
            sum1 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[b+2] > 0)) {
            Real c = static_cast<Real>(counts[b+2]);
            sum2 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[b+3] > 0)) {
            Real c = static_cast<Real>(counts[b+3]);
            sum3 += c * std::log(c);
        }
    }
    
    Real sum_clogc = sum0 + sum1 + sum2 + sum3;
    
    // Cleanup
    for (; b < n_bins; ++b) {
        if (SCL_LIKELY(counts[b] > 0)) {
            Real c = static_cast<Real>(counts[b]);
            sum_clogc += c * std::log(c);
        }
    }
    
    Real H = log_total - sum_clogc * inv_total;
    return use_log2 ? H * config::INV_LOG_2 : H;
}

} // namespace detail

// =============================================================================
// Discrete Entropy (Optimized)
// =============================================================================

inline Real discrete_entropy(
    Array<const Real> probabilities,
    bool use_log2 = false
) {
    return detail::plogp_sum(probabilities.ptr, probabilities.len, use_log2);
}

// =============================================================================
// Entropy from Count Data (Optimized with 4-way unroll)
// =============================================================================

template <typename T>
Real count_entropy(
    const T* counts,
    Size n,
    bool use_log2 = false
) {
    // 4-way unrolled total computation
    T total = T(0);
    Size i = 0;
    T t0 = T(0), t1 = T(0), t2 = T(0), t3 = T(0);
    
    for (; i + 4 <= n; i += 4) {
        t0 += counts[i];
        t1 += counts[i+1];
        t2 += counts[i+2];
        t3 += counts[i+3];
    }
    total = t0 + t1 + t2 + t3;
    
    for (; i < n; ++i) {
        total += counts[i];
    }
    
    if (SCL_UNLIKELY(total == T(0))) return Real(0);
    
    Real total_real = static_cast<Real>(total);
    Real log_total = std::log(total_real);
    
    // 4-way unrolled c*log(c) computation
    Real sum0 = Real(0), sum1 = Real(0), sum2 = Real(0), sum3 = Real(0);
    i = 0;
    
    for (; i + 4 <= n; i += 4) {
        if (SCL_LIKELY(counts[i] > T(0))) {
            Real c = static_cast<Real>(counts[i]);
            sum0 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[i+1] > T(0))) {
            Real c = static_cast<Real>(counts[i+1]);
            sum1 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[i+2] > T(0))) {
            Real c = static_cast<Real>(counts[i+2]);
            sum2 += c * std::log(c);
        }
        if (SCL_LIKELY(counts[i+3] > T(0))) {
            Real c = static_cast<Real>(counts[i+3]);
            sum3 += c * std::log(c);
        }
    }
    
    Real sum_clogc = sum0 + sum1 + sum2 + sum3;
    for (; i < n; ++i) {
        if (SCL_LIKELY(counts[i] > T(0))) {
            Real c = static_cast<Real>(counts[i]);
            sum_clogc += c * std::log(c);
        }
    }
    
    Real H = log_total - sum_clogc / total_real;
    return use_log2 ? H * config::INV_LOG_2 : H;
}

// =============================================================================
// Sparse Row Entropy (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void row_entropy(
    const Sparse<T, IsCSR>& X,
    Array<Real> entropies,
    bool normalize = false,
    bool use_log2 = false
) {
    const Index n = X.rows();
    
    SCL_CHECK_DIM(entropies.len >= static_cast<Size>(n),
                  "Entropy: output buffer too small");
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i) {
        auto values = X.row_values_unsafe(static_cast<Index>(i));
        const Index len = X.row_length_unsafe(static_cast<Index>(i));
        
        if (SCL_UNLIKELY(len == 0)) {
            entropies[i] = Real(0);
            return;
        }
        
        // Compute sum with unrolling
        Real sum = Real(0);
        Index k = 0;
        Real s0 = Real(0), s1 = Real(0), s2 = Real(0), s3 = Real(0);
        
        for (; k + 4 <= len; k += 4) {
            s0 += static_cast<Real>(values[k]);
            s1 += static_cast<Real>(values[k+1]);
            s2 += static_cast<Real>(values[k+2]);
            s3 += static_cast<Real>(values[k+3]);
        }
        sum = s0 + s1 + s2 + s3;
        
        for (; k < len; ++k) {
            sum += static_cast<Real>(values[k]);
        }
        
        if (SCL_UNLIKELY(sum < config::EPSILON)) {
            entropies[i] = Real(0);
            return;
        }
        
        // Compute entropy: H = log(sum) - sum(v*log(v))/sum
        Real log_sum = std::log(sum);
        Real sum_vlogv = Real(0);
        Real v0 = Real(0), v1 = Real(0), v2 = Real(0), v3 = Real(0);
        k = 0;
        
        for (; k + 4 <= len; k += 4) {
            Real val0 = static_cast<Real>(values[k]);
            Real val1 = static_cast<Real>(values[k+1]);
            Real val2 = static_cast<Real>(values[k+2]);
            Real val3 = static_cast<Real>(values[k+3]);
            
            if (SCL_LIKELY(val0 > config::EPSILON)) v0 += val0 * std::log(val0);
            if (SCL_LIKELY(val1 > config::EPSILON)) v1 += val1 * std::log(val1);
            if (SCL_LIKELY(val2 > config::EPSILON)) v2 += val2 * std::log(val2);
            if (SCL_LIKELY(val3 > config::EPSILON)) v3 += val3 * std::log(val3);
        }
        
        sum_vlogv = v0 + v1 + v2 + v3;
        for (; k < len; ++k) {
            Real v = static_cast<Real>(values[k]);
            if (SCL_LIKELY(v > config::EPSILON)) {
                sum_vlogv += v * std::log(v);
            }
        }
        
        Real H = log_sum - sum_vlogv / sum;
        if (use_log2) H *= config::INV_LOG_2;
        
        if (normalize && len > 1) {
            Real max_H = std::log(static_cast<Real>(len));
            if (use_log2) max_H *= config::INV_LOG_2;
            H = (max_H > config::EPSILON) ? H / max_H : Real(0);
        }
        
        entropies[i] = H;
    });
}

// =============================================================================
// KL Divergence (Optimized with SIMD-friendly structure)
// =============================================================================

inline Real kl_divergence(
    Array<const Real> p,
    Array<const Real> q,
    bool use_log2 = false
) {
    SCL_CHECK_DIM(p.len == q.len, "Entropy: array size mismatch");
    
    Real kl = Real(0);
    Real kl0 = Real(0), kl1 = Real(0), kl2 = Real(0), kl3 = Real(0);
    Size k = 0;
    
    for (; k + 4 <= p.len; k += 4) {
        if (SCL_LIKELY(p[k] > config::EPSILON)) {
            kl0 += SCL_LIKELY(q[k] > config::EPSILON) 
                   ? p[k] * std::log(p[k] / q[k])
                   : Real(1e10);
        }
        if (SCL_LIKELY(p[k+1] > config::EPSILON)) {
            kl1 += SCL_LIKELY(q[k+1] > config::EPSILON)
                   ? p[k+1] * std::log(p[k+1] / q[k+1])
                   : Real(1e10);
        }
        if (SCL_LIKELY(p[k+2] > config::EPSILON)) {
            kl2 += SCL_LIKELY(q[k+2] > config::EPSILON)
                   ? p[k+2] * std::log(p[k+2] / q[k+2])
                   : Real(1e10);
        }
        if (SCL_LIKELY(p[k+3] > config::EPSILON)) {
            kl3 += SCL_LIKELY(q[k+3] > config::EPSILON)
                   ? p[k+3] * std::log(p[k+3] / q[k+3])
                   : Real(1e10);
        }
    }
    
    kl = kl0 + kl1 + kl2 + kl3;
    for (; k < p.len; ++k) {
        if (SCL_LIKELY(p[k] > config::EPSILON)) {
            kl += SCL_LIKELY(q[k] > config::EPSILON)
                  ? p[k] * std::log(p[k] / q[k])
                  : Real(1e10);
        }
    }
    
    return use_log2 ? kl * config::INV_LOG_2 : kl;
}

// =============================================================================
// OPTIMIZATION #2: JS Divergence with 4-way unrolling
// =============================================================================

inline Real js_divergence(
    Array<const Real> p,
    Array<const Real> q,
    bool use_log2 = false
) {
    SCL_CHECK_DIM(p.len == q.len, "Entropy: array size mismatch");
    
    Real js0 = Real(0), js1 = Real(0), js2 = Real(0), js3 = Real(0);
    Size i = 0;
    
    for (; i + 4 <= p.len; i += 4) {
        // Unrolled: compute m = (p + q) / 2 and accumulate JS terms
        Real m0 = (p[i] + q[i]) * Real(0.5);
        Real m1 = (p[i+1] + q[i+1]) * Real(0.5);
        Real m2 = (p[i+2] + q[i+2]) * Real(0.5);
        Real m3 = (p[i+3] + q[i+3]) * Real(0.5);
        
        if (SCL_LIKELY(m0 > config::EPSILON)) {
            if (SCL_LIKELY(p[i] > config::EPSILON)) js0 += p[i] * std::log(p[i] / m0);
            if (SCL_LIKELY(q[i] > config::EPSILON)) js0 += q[i] * std::log(q[i] / m0);
        }
        if (SCL_LIKELY(m1 > config::EPSILON)) {
            if (SCL_LIKELY(p[i+1] > config::EPSILON)) js1 += p[i+1] * std::log(p[i+1] / m1);
            if (SCL_LIKELY(q[i+1] > config::EPSILON)) js1 += q[i+1] * std::log(q[i+1] / m1);
        }
        if (SCL_LIKELY(m2 > config::EPSILON)) {
            if (SCL_LIKELY(p[i+2] > config::EPSILON)) js2 += p[i+2] * std::log(p[i+2] / m2);
            if (SCL_LIKELY(q[i+2] > config::EPSILON)) js2 += q[i+2] * std::log(q[i+2] / m2);
        }
        if (SCL_LIKELY(m3 > config::EPSILON)) {
            if (SCL_LIKELY(p[i+3] > config::EPSILON)) js3 += p[i+3] * std::log(p[i+3] / m3);
            if (SCL_LIKELY(q[i+3] > config::EPSILON)) js3 += q[i+3] * std::log(q[i+3] / m3);
        }
    }
    
    Real js = js0 + js1 + js2 + js3;
    
    // Scalar cleanup
    for (; i < p.len; ++i) {
        Real m = (p[i] + q[i]) * Real(0.5);
        if (SCL_LIKELY(m > config::EPSILON)) {
            if (SCL_LIKELY(p[i] > config::EPSILON)) {
                js += p[i] * std::log(p[i] / m);
            }
            if (SCL_LIKELY(q[i] > config::EPSILON)) {
                js += q[i] * std::log(q[i] / m);
            }
        }
    }
    
    js *= Real(0.5);
    return use_log2 ? js * config::INV_LOG_2 : js;
}

// =============================================================================
// Symmetric KL Divergence (Optimized)
// =============================================================================

inline Real symmetric_kl(
    Array<const Real> p,
    Array<const Real> q,
    bool use_log2 = false
) {
    SCL_CHECK_DIM(p.len == q.len, "Entropy: array size mismatch");
    
    Real kl_pq0 = Real(0), kl_pq1 = Real(0), kl_pq2 = Real(0), kl_pq3 = Real(0);
    Real kl_qp0 = Real(0), kl_qp1 = Real(0), kl_qp2 = Real(0), kl_qp3 = Real(0);
    Size i = 0;
    
    for (; i + 4 <= p.len; i += 4) {
        if (SCL_LIKELY(p[i] > config::EPSILON && q[i] > config::EPSILON)) {
            Real log_ratio = std::log(p[i] / q[i]);
            kl_pq0 += p[i] * log_ratio;
            kl_qp0 -= q[i] * log_ratio;
        }
        if (SCL_LIKELY(p[i+1] > config::EPSILON && q[i+1] > config::EPSILON)) {
            Real log_ratio = std::log(p[i+1] / q[i+1]);
            kl_pq1 += p[i+1] * log_ratio;
            kl_qp1 -= q[i+1] * log_ratio;
        }
        if (SCL_LIKELY(p[i+2] > config::EPSILON && q[i+2] > config::EPSILON)) {
            Real log_ratio = std::log(p[i+2] / q[i+2]);
            kl_pq2 += p[i+2] * log_ratio;
            kl_qp2 -= q[i+2] * log_ratio;
        }
        if (SCL_LIKELY(p[i+3] > config::EPSILON && q[i+3] > config::EPSILON)) {
            Real log_ratio = std::log(p[i+3] / q[i+3]);
            kl_pq3 += p[i+3] * log_ratio;
            kl_qp3 -= q[i+3] * log_ratio;
        }
    }
    
    Real kl_pq = kl_pq0 + kl_pq1 + kl_pq2 + kl_pq3;
    Real kl_qp = kl_qp0 + kl_qp1 + kl_qp2 + kl_qp3;
    
    for (; i < p.len; ++i) {
        if (SCL_LIKELY(p[i] > config::EPSILON && q[i] > config::EPSILON)) {
            Real log_ratio = std::log(p[i] / q[i]);
            kl_pq += p[i] * log_ratio;
            kl_qp -= q[i] * log_ratio;
        }
    }
    
    Real result = (kl_pq + kl_qp) * Real(0.5);
    return use_log2 ? result * config::INV_LOG_2 : result;
}

// =============================================================================
// Equal-Width Discretization (Parallel)
// =============================================================================

template <typename T>
void discretize_equal_width(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
) {
    detail::discretize_equal_width_parallel(values, n, n_bins, binned);
}

// =============================================================================
// Equal-Frequency Discretization
// =============================================================================

template <typename T>
void discretize_equal_frequency(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
) {
    detail::discretize_equal_frequency(values, n, n_bins, binned);
}

// =============================================================================
// 2D Histogram (Parallel)
// =============================================================================

inline void histogram_2d(
    const Index* x_binned,
    const Index* y_binned,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    Size* counts
) {
    detail::histogram_2d_parallel(x_binned, y_binned, n, n_bins_x, n_bins_y, counts);
}

// =============================================================================
// Joint Entropy H(X, Y)
// =============================================================================

inline Real joint_entropy(
    const Index* x_binned,
    const Index* y_binned,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    bool use_log2 = false
) {
    Size total_bins = static_cast<Size>(n_bins_x) * n_bins_y;
    Size* counts = scl::memory::aligned_alloc<Size>(total_bins, SCL_ALIGNMENT);
    detail::histogram_2d_parallel(x_binned, y_binned, n, n_bins_x, n_bins_y, counts);
    Real H = detail::entropy_from_counts(counts, static_cast<Index>(total_bins), n, use_log2);
    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
    return H;
}

// =============================================================================
// Marginal Entropy from Binned Data
// =============================================================================

inline Real marginal_entropy(
    const Index* binned,
    Size n,
    Index n_bins,
    bool use_log2 = false
) {
    Size* counts = scl::memory::aligned_alloc<Size>(n_bins, SCL_ALIGNMENT);
    detail::histogram_parallel(binned, n, n_bins, counts);
    Real H = detail::entropy_from_counts(counts, n_bins, n, use_log2);
    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
    return H;
}

// =============================================================================
// Conditional Entropy H(Y | X)
// =============================================================================

inline Real conditional_entropy(
    const Index* x_binned,
    const Index* y_binned,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    bool use_log2 = false
) {
    // H(Y|X) = H(X,Y) - H(X)
    Real H_xy = joint_entropy(x_binned, y_binned, n, n_bins_x, n_bins_y, use_log2);
    Real H_x = marginal_entropy(x_binned, n, n_bins_x, use_log2);
    return H_xy - H_x;
}

// =============================================================================
// Mutual Information I(X; Y) = H(X) + H(Y) - H(X, Y)
// =============================================================================

inline Real mutual_information(
    const Index* x_binned,
    const Index* y_binned,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    bool use_log2 = false
) {
    Real H_x = marginal_entropy(x_binned, n, n_bins_x, use_log2);
    Real H_y = marginal_entropy(y_binned, n, n_bins_y, use_log2);
    Real H_xy = joint_entropy(x_binned, y_binned, n, n_bins_x, n_bins_y, use_log2);

    return scl::algo::max2(H_x + H_y - H_xy, Real(0));
}

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

inline Real normalized_mi(
    Array<const Index> labels1,
    Array<const Index> labels2,
    Index n_clusters1,
    Index n_clusters2
) {
    SCL_CHECK_DIM(labels1.len == labels2.len, "Entropy: label arrays must have same length");

    Size n = labels1.len;
    if (n == 0) return Real(0);

    Real MI = mutual_information(
        labels1.ptr, labels2.ptr, n, n_clusters1, n_clusters2, true
    );

    Real H1 = marginal_entropy(labels1.ptr, n, n_clusters1, true);
    Real H2 = marginal_entropy(labels2.ptr, n, n_clusters2, true);

    if (H1 < config::EPSILON || H2 < config::EPSILON) return Real(0);

    // NMI = 2 * I(X;Y) / (H(X) + H(Y))
    return Real(2) * MI / (H1 + H2);
}

// =============================================================================
// Adjusted Mutual Information (AMI)
// =============================================================================

inline Real adjusted_mi(
    Array<const Index> labels1,
    Array<const Index> labels2,
    Index n_clusters1,
    Index n_clusters2
) {
    SCL_CHECK_DIM(labels1.len == labels2.len, "Entropy: label arrays must have same length");

    Size n = labels1.len;
    if (n == 0) return Real(0);

    // Build contingency table
    Size table_size = static_cast<Size>(n_clusters1) * static_cast<Size>(n_clusters2);
    Size* contingency = scl::memory::aligned_alloc<Size>(table_size, SCL_ALIGNMENT);
    scl::algo::zero(contingency, table_size);

    for (Size i = 0; i < n; ++i) {
        Index c1 = labels1[i];
        Index c2 = labels2[i];
        if (c1 >= 0 && c1 < n_clusters1 && c2 >= 0 && c2 < n_clusters2) {
            ++contingency[static_cast<Size>(c1) * n_clusters2 + c2];
        }
    }

    // Compute row and column sums
    Size* row_sums = scl::memory::aligned_alloc<Size>(n_clusters1, SCL_ALIGNMENT);
    Size* col_sums = scl::memory::aligned_alloc<Size>(n_clusters2, SCL_ALIGNMENT);
    scl::algo::zero(row_sums, static_cast<Size>(n_clusters1));
    scl::algo::zero(col_sums, static_cast<Size>(n_clusters2));

    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size count = contingency[static_cast<Size>(i) * n_clusters2 + j];
            row_sums[i] += count;
            col_sums[j] += count;
        }
    }

    // Compute MI
    Real MI = Real(0);
    Real n_real = static_cast<Real>(n);
    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size nij = contingency[static_cast<Size>(i) * n_clusters2 + j];
            if (nij > 0 && row_sums[i] > 0 && col_sums[j] > 0) {
                Real p_ij = static_cast<Real>(nij) / n_real;
                Real p_i = static_cast<Real>(row_sums[i]) / n_real;
                Real p_j = static_cast<Real>(col_sums[j]) / n_real;
                MI += p_ij * std::log(p_ij / (p_i * p_j));
            }
        }
    }

    // Compute expected MI (simplified approximation)
    Real E_MI = Real(0);
    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            if (row_sums[i] > 0 && col_sums[j] > 0) {
                Real expected = static_cast<Real>(row_sums[i]) *
                               static_cast<Real>(col_sums[j]) / n_real;
                expected /= n_real;
                if (expected > config::EPSILON) {
                    E_MI += expected * std::log(expected);
                }
            }
        }
    }

    // Compute entropies
    Real H1 = marginal_entropy(labels1.ptr, n, n_clusters1, false);
    Real H2 = marginal_entropy(labels2.ptr, n, n_clusters2, false);

    scl::memory::aligned_free(col_sums, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
    scl::memory::aligned_free(contingency, SCL_ALIGNMENT);

    // AMI = (MI - E[MI]) / (max(H1, H2) - E[MI])
    Real max_H = scl::algo::max2(H1, H2);
    Real denominator = max_H - E_MI;

    if (denominator < config::EPSILON) return Real(0);
    return (MI - E_MI) / denominator;
}

// =============================================================================
// Feature Selection by Mutual Information
// =============================================================================

template <typename T, bool IsCSR>
void select_features_mi(
    const Sparse<T, IsCSR>& X,
    Array<const Index> target,
    Index n_features,
    Index n_to_select,
    Array<Index> selected_features,
    Array<Real> mi_scores,
    Index n_bins = config::DEFAULT_N_BINS
) {
    const Index n_samples = IsCSR ? X.rows() : X.cols();

    SCL_CHECK_DIM(target.len >= static_cast<Size>(n_samples),
                  "Entropy: target length mismatch");
    SCL_CHECK_DIM(mi_scores.len >= static_cast<Size>(n_features),
                  "Entropy: mi_scores buffer too small");
    SCL_CHECK_DIM(selected_features.len >= static_cast<Size>(n_to_select),
                  "Entropy: selected_features buffer too small");

    // Find number of target classes
    Index n_classes = 0;
    for (Size i = 0; i < target.len; ++i) {
        n_classes = scl::algo::max2(n_classes, target[i] + 1);
    }

    // Compute MI for each feature
    Index* feature_binned = scl::memory::aligned_alloc<Index>(n_samples, SCL_ALIGNMENT);
    Real* feature_values = scl::memory::aligned_alloc<Real>(n_samples, SCL_ALIGNMENT);

    for (Index f = 0; f < n_features; ++f) {
        // Extract feature values
        scl::algo::zero(feature_values, static_cast<Size>(n_samples));

        if (IsCSR) {
            for (Index row = 0; row < n_samples; ++row) {
                auto indices = X.row_indices_unsafe(row);
                auto values = X.row_values_unsafe(row);
                const Index len = X.row_length_unsafe(row);

                for (Index k = 0; k < len; ++k) {
                    if (indices[k] == f) {
                        feature_values[row] = static_cast<Real>(values[k]);
                        break;
                    }
                }
            }
        } else {
            auto indices = X.col_indices_unsafe(f);
            auto values = X.col_values_unsafe(f);
            const Index len = X.col_length_unsafe(f);

            for (Index k = 0; k < len; ++k) {
                Index row = indices[k];
                if (row < n_samples) {
                    feature_values[row] = static_cast<Real>(values[k]);
                }
            }
        }

        // Discretize feature
        discretize_equal_width(feature_values, static_cast<Size>(n_samples), n_bins, feature_binned);

        // Compute MI with target
        mi_scores[f] = mutual_information(
            feature_binned, target.ptr, static_cast<Size>(n_samples),
            n_bins, n_classes, true
        );
    }

    scl::memory::aligned_free(feature_values, SCL_ALIGNMENT);
    scl::memory::aligned_free(feature_binned, SCL_ALIGNMENT);

    // Select top features using partial_sort for O(n log k) instead of O(n*k) insertion sort
    Index* sorted_idx = scl::memory::aligned_alloc<Index>(n_features, SCL_ALIGNMENT);
    for (Index f = 0; f < n_features; ++f) {
        sorted_idx[f] = f;
    }
    
    // Use partial_sort with custom comparator for descending order (largest MI first)
    Index k = scl::algo::min2(n_to_select, n_features);
    scl::algo::partial_sort(sorted_idx, static_cast<Size>(n_features), static_cast<Size>(k),
        [&mi_scores](Index a, Index b) {
            return mi_scores[a] > mi_scores[b];  // Descending order
        });
    
    // Copy top features
    for (Index i = 0; i < k; ++i) {
        selected_features[i] = sorted_idx[i];
    }
    
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
}

// =============================================================================
// mRMR Feature Selection
// =============================================================================

template <typename T, bool IsCSR>
void mrmr_selection(
    const Sparse<T, IsCSR>& X,
    Array<const Index> target,
    Index n_features,
    Index n_to_select,
    Array<Index> selected_features,
    Index n_bins = config::DEFAULT_N_BINS
) {
    const Index n_samples = IsCSR ? X.rows() : X.cols();

    SCL_CHECK_DIM(selected_features.len >= static_cast<Size>(n_to_select),
                  "Entropy: selected_features buffer too small");

    // Find number of target classes
    Index n_classes = 0;
    for (Size i = 0; i < target.len; ++i) {
        n_classes = scl::algo::max2(n_classes, target[i] + 1);
    }

    // Allocate workspace
    Real* relevance = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
    Real* redundancy = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
    bool* selected = reinterpret_cast<bool*>(scl::memory::aligned_alloc<char>(n_features, SCL_ALIGNMENT));
    Index** binned_features = scl::memory::aligned_alloc<Index*>(n_features, SCL_ALIGNMENT);

    std::memset(selected, 0, n_features);
    scl::algo::zero(redundancy, static_cast<Size>(n_features));

    // Pre-discretize all features
    Real* temp_values = scl::memory::aligned_alloc<Real>(n_samples, SCL_ALIGNMENT);

    for (Index f = 0; f < n_features; ++f) {
        binned_features[f] = scl::memory::aligned_alloc<Index>(n_samples, SCL_ALIGNMENT);
        scl::algo::zero(temp_values, static_cast<Size>(n_samples));

        if (IsCSR) {
            for (Index row = 0; row < n_samples; ++row) {
                auto indices = X.row_indices_unsafe(row);
                auto values = X.row_values_unsafe(row);
                const Index len = X.row_length_unsafe(row);

                for (Index k = 0; k < len; ++k) {
                    if (indices[k] == f) {
                        temp_values[row] = static_cast<Real>(values[k]);
                        break;
                    }
                }
            }
        } else {
            auto indices = X.col_indices_unsafe(f);
            auto values = X.col_values_unsafe(f);
            const Index len = X.col_length_unsafe(f);

            for (Index k = 0; k < len; ++k) {
                Index row = indices[k];
                if (row < n_samples) {
                    temp_values[row] = static_cast<Real>(values[k]);
                }
            }
        }

        discretize_equal_width(temp_values, static_cast<Size>(n_samples), n_bins, binned_features[f]);

        // Compute relevance (MI with target)
        relevance[f] = mutual_information(
            binned_features[f], target.ptr, static_cast<Size>(n_samples),
            n_bins, n_classes, true
        );
    }

    scl::memory::aligned_free(temp_values, SCL_ALIGNMENT);

    // Greedy selection
    Index n_selected = 0;
    while (n_selected < n_to_select && n_selected < n_features) {
        // Find best feature: max(relevance - redundancy)
        Index best_f = -1;
        Real best_score = -Real(1e30);

        for (Index f = 0; f < n_features; ++f) {
            if (selected[f]) continue;

            Real score = relevance[f];
            if (n_selected > 0) {
                score -= redundancy[f] / static_cast<Real>(n_selected);
            }

            if (score > best_score) {
                best_score = score;
                best_f = f;
            }
        }

        if (best_f < 0) break;

        // Add to selected set
        selected_features[n_selected++] = best_f;
        selected[best_f] = true;

        // Update redundancy for remaining features
        for (Index f = 0; f < n_features; ++f) {
            if (selected[f]) continue;

            Real mi_ff = mutual_information(
                binned_features[f], binned_features[best_f],
                static_cast<Size>(n_samples), n_bins, n_bins, true
            );
            redundancy[f] += mi_ff;
        }
    }

    // Cleanup
    for (Index f = 0; f < n_features; ++f) {
        scl::memory::aligned_free(binned_features[f], SCL_ALIGNMENT);
    }
    scl::memory::aligned_free(binned_features, SCL_ALIGNMENT);
    scl::memory::aligned_free(reinterpret_cast<char*>(selected), SCL_ALIGNMENT);
    scl::memory::aligned_free(redundancy, SCL_ALIGNMENT);
    scl::memory::aligned_free(relevance, SCL_ALIGNMENT);
}

// =============================================================================
// Cross-Entropy Loss (Optimized with 4-way unrolling)
// =============================================================================

inline Real cross_entropy(
    Array<const Real> true_probs,
    Array<const Real> pred_probs
) {
    SCL_CHECK_DIM(true_probs.len == pred_probs.len, "Entropy: array size mismatch");
    
    Real ce0 = Real(0), ce1 = Real(0), ce2 = Real(0), ce3 = Real(0);
    Size i = 0;
    
    for (; i + 4 <= true_probs.len; i += 4) {
        if (SCL_LIKELY(true_probs[i] > config::EPSILON)) {
            ce0 -= true_probs[i] * detail::safe_log(pred_probs[i]);
        }
        if (SCL_LIKELY(true_probs[i+1] > config::EPSILON)) {
            ce1 -= true_probs[i+1] * detail::safe_log(pred_probs[i+1]);
        }
        if (SCL_LIKELY(true_probs[i+2] > config::EPSILON)) {
            ce2 -= true_probs[i+2] * detail::safe_log(pred_probs[i+2]);
        }
        if (SCL_LIKELY(true_probs[i+3] > config::EPSILON)) {
            ce3 -= true_probs[i+3] * detail::safe_log(pred_probs[i+3]);
        }
    }
    
    Real ce = ce0 + ce1 + ce2 + ce3;
    for (; i < true_probs.len; ++i) {
        if (SCL_LIKELY(true_probs[i] > config::EPSILON)) {
            ce -= true_probs[i] * detail::safe_log(pred_probs[i]);
        }
    }
    
    return ce;
}

// =============================================================================
// Perplexity (exp of cross-entropy)
// =============================================================================

inline Real perplexity(
    Array<const Real> true_probs,
    Array<const Real> pred_probs
) {
    Real ce = cross_entropy(true_probs, pred_probs);
    return std::exp(ce);
}

// =============================================================================
// Information Gain
// =============================================================================

inline Real information_gain(
    const Index* feature_binned,
    const Index* target,
    Size n,
    Index n_feature_bins,
    Index n_target_classes
) {
    // IG(T, F) = H(T) - H(T|F) = I(T; F)
    return mutual_information(
        feature_binned, target, n, n_feature_bins, n_target_classes, true
    );
}

// =============================================================================
// Gini Impurity (SIMD optimized)
// =============================================================================

inline Real gini_impurity(
    Array<const Real> probabilities
) {
    // Use vectorize for sum of squares when large enough
    if (SCL_LIKELY(probabilities.len >= config::SIMD_THRESHOLD)) {
        Real sum_sq = scl::vectorize::sum_squared(probabilities);
        return Real(1) - sum_sq;
    }
    
    // Scalar path with 4-way unrolling for small arrays
    Real s0 = Real(0), s1 = Real(0), s2 = Real(0), s3 = Real(0);
    Size k = 0;
    
    for (; k + 4 <= probabilities.len; k += 4) {
        s0 += probabilities[k] * probabilities[k];
        s1 += probabilities[k+1] * probabilities[k+1];
        s2 += probabilities[k+2] * probabilities[k+2];
        s3 += probabilities[k+3] * probabilities[k+3];
    }
    
    Real sum_sq = s0 + s1 + s2 + s3;
    for (; k < probabilities.len; ++k) {
        sum_sq += probabilities[k] * probabilities[k];
    }
    
    return Real(1) - sum_sq;
}

} // namespace scl::kernel::entropy
