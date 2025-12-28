#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <cstdint>

// =============================================================================
// FILE: scl/kernel/resample.hpp
// BRIEF: Resampling operations with fast RNG
// =============================================================================

namespace scl::kernel::resample {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
}

namespace detail {

template <typename T>
SCL_FORCE_INLINE SCL_HOT Real sum_simd_4way(const T* SCL_RESTRICT vals, Size len) {
    if constexpr (std::is_same_v<T, Real>) {
        return scl::vectorize::sum(Array<const Real>(vals, len));
    } else {
        // For non-Real types, still use vectorize::sum but convert results
        // Multi-accumulator pattern with cast-as-you-go approach
        Real sum0 = Real(0), sum1 = Real(0);
        Real sum2 = Real(0), sum3 = Real(0);
        Size k = 0;

        // 8-way unroll with 4 independent accumulators for ILP
        for (; k + 8 <= len; k += 8) {
            sum0 += static_cast<Real>(vals[k + 0]);
            sum1 += static_cast<Real>(vals[k + 1]);
            sum2 += static_cast<Real>(vals[k + 2]);
            sum3 += static_cast<Real>(vals[k + 3]);
            sum0 += static_cast<Real>(vals[k + 4]);
            sum1 += static_cast<Real>(vals[k + 5]);
            sum2 += static_cast<Real>(vals[k + 6]);
            sum3 += static_cast<Real>(vals[k + 7]);
        }

        Real sum = sum0 + sum1 + sum2 + sum3;

        // Scalar cleanup
        for (; k < len; ++k) {
            sum += static_cast<Real>(vals[k]);
        }

        return sum;
    }
}

class FastRNG {
public:
    using result_type = uint64_t;
    
    explicit FastRNG(uint64_t seed) {
        uint64_t s = seed;
        for (int i = 0; i < 4; ++i) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            state_[i] = z ^ (z >> 31);
        }
    }
    
    SCL_FORCE_INLINE uint64_t operator()() {
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
    
    SCL_FORCE_INLINE Real uniform() {
        return static_cast<Real>((*this)() >> 11) * 0x1.0p-53;
    }
    
private:
    uint64_t state_[4];
    
    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

class FastBinomial {
public:
    template <typename RNG>
    SCL_FORCE_INLINE static Index sample(RNG& rng, Index n, Real p) {
        if (n <= 0 || p <= Real(0)) return 0;
        if (p >= Real(1)) return n;
        
        bool flip = false;
        if (p > Real(0.5)) {
            p = Real(1) - p;
            flip = true;
        }
        
        Index result;
        if (n <= 10) {
            result = sample_direct(rng, n, p);
        } else if (n <= 100) {
            result = sample_inversion(rng, n, p);
        } else {
            result = sample_normal_approx(rng, n, p);
        }
        
        return flip ? (n - result) : result;
    }
    
private:
    template <typename RNG>
    SCL_FORCE_INLINE static Index sample_direct(RNG& rng, Index n, Real p) {
        Index count = 0;
        for (Index i = 0; i < n; ++i) {
            if (rng.uniform() < p) ++count;
        }
        return count;
    }
    
    template <typename RNG>
    static Index sample_inversion(RNG& rng, Index n, Real p) {
        Real q = Real(1) - p;
        Real s = p / q;
        Real a = static_cast<Real>(n + 1) * s;
        Real r = std::pow(q, static_cast<Real>(n));
        Real u = rng.uniform();
        
        Index x = 0;
        while (u > r && x < n) {
            u -= r;
            ++x;
            r *= (a / static_cast<Real>(x) - s);
        }
        return x;
    }
    
    template <typename RNG>
    static Index sample_normal_approx(RNG& rng, Index n, Real p) {
        Real np = static_cast<Real>(n) * p;
        Real sigma = std::sqrt(np * (Real(1) - p));
        
        Real u1 = rng.uniform();
        Real u2 = rng.uniform();
        if (u1 < Real(1e-10)) u1 = Real(1e-10);
        Real z = std::sqrt(Real(-2) * std::log(u1)) * std::cos(Real(2) * Real(M_PI) * u2);
        
        Real x = np + sigma * z + Real(0.5);
        return static_cast<Index>(std::max(Real(0), std::min(static_cast<Real>(n), std::floor(x))));
    }
};

class FastPoisson {
public:
    template <typename RNG>
    static Index sample(RNG& rng, Real lambda) {
        if (lambda <= Real(0)) return 0;
        
        if (lambda < Real(10)) {
            return sample_inversion(rng, lambda);
        } else {
            return sample_normal_approx(rng, lambda);
        }
    }
    
private:
    template <typename RNG>
    static Index sample_inversion(RNG& rng, Real lambda) {
        Real L = std::exp(-lambda);
        Real p = Real(1);
        Index k = 0;
        
        do {
            ++k;
            p *= rng.uniform();
        } while (p > L);
        
        return k - 1;
    }
    
    template <typename RNG>
    static Index sample_normal_approx(RNG& rng, Real lambda) {
        Real sigma = std::sqrt(lambda);
        
        Real u1 = rng.uniform();
        Real u2 = rng.uniform();
        if (u1 < Real(1e-10)) u1 = Real(1e-10);
        Real z = std::sqrt(Real(-2) * std::log(u1)) * std::cos(Real(2) * Real(M_PI) * u2);
        
        Real x = lambda + sigma * z + Real(0.5);
        return static_cast<Index>(std::max(Real(0), std::floor(x)));
    }
};

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void downsample(
    Sparse<T, IsCSR>& matrix,
    Real target_sum,
    uint64_t seed
) {
    const Index primary_dim = matrix.primary_dim();
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;
        
        auto values = matrix.primary_values(idx);
        
        Real current_sum = detail::sum_simd_4way(values.ptr, len_sz);
        
        if (SCL_UNLIKELY(current_sum <= target_sum || current_sum <= Real(0))) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target_sum;

        for (Size k = 0; k < len_sz && remaining_target > Real(0); ++k) {
            // Prefetch ahead for better cache behavior
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&values.ptr[k + config::PREFETCH_DISTANCE], 0);
            }

            Real count = static_cast<Real>(values[k]);
            if (SCL_UNLIKELY(count <= Real(0))) continue;

            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);

            values.ptr[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

template <typename T, bool IsCSR>
void downsample_variable(
    Sparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    const Index primary_dim = matrix.primary_dim();
    
    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(primary_dim), "Target counts size mismatch");
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;
        
        auto values = matrix.primary_values(idx);
        
        Real current_sum = detail::sum_simd_4way(values.ptr, len_sz);
        Real target = target_counts[p];
        
        if (SCL_UNLIKELY(current_sum <= target || current_sum <= Real(0))) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target;

        for (Size k = 0; k < len_sz && remaining_target > Real(0); ++k) {
            // Prefetch ahead for better cache behavior
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&values.ptr[k + config::PREFETCH_DISTANCE], 0);
            }

            Real count = static_cast<Real>(values[k]);
            if (SCL_UNLIKELY(count <= Real(0))) continue;

            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);

            values.ptr[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

template <typename T, bool IsCSR>
void binomial_resample(
    Sparse<T, IsCSR>& matrix,
    Real p,
    uint64_t seed
) {
    const Index primary_dim = matrix.primary_dim();
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        const Index idx = static_cast<Index>(row);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;
        
        auto values = matrix.primary_values(idx);
        
        for (Size k = 0; k < len_sz; ++k) {
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&values.ptr[k + config::PREFETCH_DISTANCE], 0);
            }
            Index count = static_cast<Index>(values[k]);
            if (SCL_LIKELY(count > 0)) {
                values.ptr[k] = static_cast<T>(detail::FastBinomial::sample(rng, count, p));
            }
        }
    });
}

template <typename T, bool IsCSR>
void poisson_resample(
    Sparse<T, IsCSR>& matrix,
    Real lambda,
    uint64_t seed
) {
    const Index primary_dim = matrix.primary_dim();
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        const Index idx = static_cast<Index>(row);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;
        
        auto values = matrix.primary_values(idx);
        
        for (Size k = 0; k < len_sz; ++k) {
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&values.ptr[k + config::PREFETCH_DISTANCE], 0);
            }
            Real count = static_cast<Real>(values[k]);
            if (SCL_LIKELY(count > Real(0))) {
                values.ptr[k] = static_cast<T>(detail::FastPoisson::sample(rng, count * lambda));
            }
        }
    });
}

} // namespace scl::kernel::resample

