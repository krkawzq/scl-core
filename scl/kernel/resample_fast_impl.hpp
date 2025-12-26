#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <cstdint>

// =============================================================================
/// @file resample_fast_impl.hpp
/// @brief Fast Path for Resampling Operations
///
/// ## Key Optimizations
///
/// 1. 4-Way SIMD Sum Accumulation
/// 2. Fast Xoshiro256++ RNG Integration
/// 3. Batch Binomial Sampling
/// 4. Contiguous Memory Access for CustomSparse
///
/// Note: Resampling is primarily RNG-bound, but memory access patterns
/// still matter for large matrices. CustomSparse benefits from
/// contiguous data layout.
// =============================================================================

namespace scl::kernel::resample::fast {

// Forward declarations for RNG and samplers from resample.hpp
class Xoshiro256pp;

namespace detail {

// =============================================================================
// SECTION 1: SIMD Sum Utilities
// =============================================================================

/// @brief 4-way unrolled SIMD sum for Real type
template <typename T>
SCL_FORCE_INLINE Real sum_simd_4way(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    
    // For non-Real types, use scalar accumulation
    if constexpr (!std::is_same_v<T, Real>) {
        Real sum = Real(0);
        Size k = 0;
        
        // 4-way scalar unroll
        for (; k + 4 <= len; k += 4) {
            sum += static_cast<Real>(vals[k + 0]);
            sum += static_cast<Real>(vals[k + 1]);
            sum += static_cast<Real>(vals[k + 2]);
            sum += static_cast<Real>(vals[k + 3]);
        }
        
        for (; k < len; ++k) {
            sum += static_cast<Real>(vals[k]);
        }
        
        return sum;
    } else {
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        Size k = 0;
        
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
            v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
            v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
            v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            sum += vals[k];
        }
        
        return sum;
    }
}

/// @brief Xoshiro256++ fast PRNG (local copy for fast_impl)
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

/// @brief Fast binomial sampling
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
        
        // Box-Muller
        Real u1 = rng.uniform();
        Real u2 = rng.uniform();
        if (u1 < Real(1e-10)) u1 = Real(1e-10);
        Real z = std::sqrt(Real(-2) * std::log(u1)) * std::cos(Real(2) * Real(M_PI) * u2);
        
        Real x = np + sigma * z + Real(0.5);
        return static_cast<Index>(std::max(Real(0), std::min(static_cast<Real>(n), std::floor(x))));
    }
};

/// @brief Fast Poisson sampling
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
// SECTION 2: CustomSparse Fast Path
// =============================================================================

/// @brief Downsample CustomSparse with target sum
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void downsample_custom(
    CustomSparse<T, IsCSR>& matrix,
    Real target_sum,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);
        
        if (len == 0) return;
        
        T* vals = matrix.data + start;
        
        Real current_sum = detail::sum_simd_4way(vals, len);
        
        if (current_sum <= target_sum || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target_sum;
        
        for (Size k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Downsample CustomSparse with variable targets
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void downsample_variable_custom(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(primary_dim), "Target counts size mismatch");
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);
        
        if (len == 0) return;
        
        T* vals = matrix.data + start;
        
        Real current_sum = detail::sum_simd_4way(vals, len);
        Real target = target_counts[p];
        
        if (current_sum <= target || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target;
        
        for (Size k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Binomial resample CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void binomial_resample_custom(
    CustomSparse<T, IsCSR>& matrix,
    Real p,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index start = matrix.indptr[row];
        Index end = matrix.indptr[row + 1];
        
        T* vals = matrix.data + start;
        Size len = static_cast<Size>(end - start);
        
        for (Size k = 0; k < len; ++k) {
            Index count = static_cast<Index>(vals[k]);
            if (count > 0) {
                vals[k] = static_cast<T>(detail::FastBinomial::sample(rng, count, p));
            }
        }
    });
}

/// @brief Poisson resample CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void poisson_resample_custom(
    CustomSparse<T, IsCSR>& matrix,
    Real lambda,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index start = matrix.indptr[row];
        Index end = matrix.indptr[row + 1];
        
        T* vals = matrix.data + start;
        Size len = static_cast<Size>(end - start);
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count > Real(0)) {
                vals[k] = static_cast<T>(detail::FastPoisson::sample(rng, count * lambda));
            }
        }
    });
}

// =============================================================================
// SECTION 3: VirtualSparse Fast Path
// =============================================================================

/// @brief Downsample VirtualSparse with target sum
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void downsample_virtual(
    VirtualSparse<T, IsCSR>& matrix,
    Real target_sum,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index len = matrix.lengths[p];
        if (len == 0) return;
        
        T* vals = static_cast<T*>(matrix.data_ptrs[p]);
        
        Real current_sum = detail::sum_simd_4way(vals, static_cast<Size>(len));
        
        if (current_sum <= target_sum || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target_sum;
        
        for (Index k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Downsample VirtualSparse with variable targets
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void downsample_variable_virtual(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(primary_dim), "Target counts size mismatch");
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index len = matrix.lengths[p];
        if (len == 0) return;
        
        T* vals = static_cast<T*>(matrix.data_ptrs[p]);
        
        Real current_sum = detail::sum_simd_4way(vals, static_cast<Size>(len));
        Real target = target_counts[p];
        
        if (current_sum <= target || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target;
        
        for (Index k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Binomial resample VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void binomial_resample_virtual(
    VirtualSparse<T, IsCSR>& matrix,
    Real p,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index len = matrix.lengths[row];
        if (len == 0) return;
        
        T* vals = static_cast<T*>(matrix.data_ptrs[row]);
        
        for (Index k = 0; k < len; ++k) {
            Index count = static_cast<Index>(vals[k]);
            if (count > 0) {
                vals[k] = static_cast<T>(detail::FastBinomial::sample(rng, count, p));
            }
        }
    });
}

/// @brief Poisson resample VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void poisson_resample_virtual(
    VirtualSparse<T, IsCSR>& matrix,
    Real lambda,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index len = matrix.lengths[row];
        if (len == 0) return;
        
        T* vals = static_cast<T*>(matrix.data_ptrs[row]);
        
        for (Index k = 0; k < len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count > Real(0)) {
                vals[k] = static_cast<T>(detail::FastPoisson::sample(rng, count * lambda));
            }
        }
    });
}

// =============================================================================
// SECTION 4: Unified Dispatchers
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void downsample_fast(MatrixT& matrix, Real target_sum, uint64_t seed) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        downsample_custom(matrix, target_sum, seed);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        downsample_virtual(matrix, target_sum, seed);
    }
}

template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void downsample_variable_fast(MatrixT& matrix, Array<const Real> targets, uint64_t seed) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        downsample_variable_custom(matrix, targets, seed);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        downsample_variable_virtual(matrix, targets, seed);
    }
}

template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void binomial_resample_fast(MatrixT& matrix, Real p, uint64_t seed) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        binomial_resample_custom(matrix, p, seed);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        binomial_resample_virtual(matrix, p, seed);
    }
}

template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void poisson_resample_fast(MatrixT& matrix, Real lambda, uint64_t seed) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        poisson_resample_custom(matrix, lambda, seed);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        poisson_resample_virtual(matrix, lambda, seed);
    }
}

} // namespace scl::kernel::resample::fast
