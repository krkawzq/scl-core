#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/resample_fast_impl.hpp"
#include "scl/kernel/resample_mapped_impl.hpp"

#include <cmath>
#include <cstdint>
#include <algorithm>

// =============================================================================
/// @file resample.hpp
/// @brief High-Performance Count Resampling for Data Augmentation
///
/// ## Supported Operations
///
/// 1. Downsample: Sample M counts from N (M < N) - Multinomial
/// 2. Binomial resample: c' ~ Binomial(c, p)
/// 3. Poisson resample: c' ~ Poisson(c * lambda)
///
/// ## Key Optimizations
///
/// 1. Xoshiro256++ Fast RNG
///    - 3-4x faster than std::mt19937_64
///    - Excellent statistical quality (BigCrush, PractRand)
///
/// 2. Adaptive Binomial Sampling
///    - n <= 10: Direct Bernoulli trials
///    - n <= 100: Inversion method
///    - n > 100: Normal approximation
///
/// 3. 4-Way SIMD Sum Accumulation
///    - Better instruction-level parallelism
///
/// 4. Backend Dispatch
///    - CustomSparseLike -> resample_fast_impl.hpp
///    - VirtualSparseLike -> resample_fast_impl.hpp
///    - MappedSparseLike -> resample_mapped_impl.hpp
///
/// ## Use Cases
///
/// - Data augmentation for deep learning
/// - Depth normalization
/// - Robustness testing
///
/// Performance: O(nnz) with low constant factor
// =============================================================================

namespace scl::kernel::resample {

// =============================================================================
// SECTION 1: Fast RNG (Xoshiro256++)
// =============================================================================

/// @brief Xoshiro256++ fast PRNG
///
/// 3-4x faster than std::mt19937_64 with excellent statistical quality.
/// Passes BigCrush and PractRand tests.
class Xoshiro256pp {
public:
    using result_type = uint64_t;

    explicit Xoshiro256pp(uint64_t seed = 0) {
        // SplitMix64 for seeding
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

    /// @brief Generate uniform Real in [0, 1)
    SCL_FORCE_INLINE Real uniform() {
        // Use upper 53 bits for double precision
        return static_cast<Real>((*this)() >> 11) * 0x1.0p-53;
    }

    /// @brief Generate uniform Real in [0, max)
    SCL_FORCE_INLINE Real uniform(Real max) {
        return uniform() * max;
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

private:
    uint64_t state_[4];

    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// =============================================================================
// SECTION 2: Efficient Sampling Algorithms
// =============================================================================

namespace detail {

/// @brief Binomial sampling using different algorithms based on parameters
class BinomialSampler {
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

    template <typename RNG>
    static Real sample_normal(RNG& rng) {
        Real u1 = rng.uniform();
        Real u2 = rng.uniform();
        if (u1 < Real(1e-10)) u1 = Real(1e-10);
        return std::sqrt(Real(-2) * std::log(u1)) * std::cos(Real(2) * Real(M_PI) * u2);
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
        Real z = sample_normal(rng);
        Real x = np + sigma * z + Real(0.5);
        return static_cast<Index>(std::max(Real(0), std::min(static_cast<Real>(n), std::floor(x))));
    }
};

/// @brief Poisson sampling
class PoissonSampler {
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
        Real z = BinomialSampler::sample_normal(rng);
        Real x = lambda + sigma * z + Real(0.5);
        return static_cast<Index>(std::max(Real(0), std::floor(x)));
    }
};

/// @brief 4-way scalar sum (for any type)
template <typename T>
SCL_FORCE_INLINE Real sum_4way(const T* vals, Size len) {
    Real sum = Real(0);
    Size k = 0;
    
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
}

} // namespace detail

// =============================================================================
// SECTION 3: Generic Implementations (Fallback)
// =============================================================================

namespace generic {

/// @brief Downsample counts to target sum
template <typename MatrixT>
    requires AnySparse<MatrixT>
void downsample_counts_impl(
    MatrixT& matrix,
    Real target_sum,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Xoshiro256pp rng(seed + p * 0x9e3779b97f4a7c15ULL);

        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        const Size len = vals.len;

        if (len == 0) return;

        Real current_sum = detail::sum_4way(vals.ptr, len);

        if (current_sum <= target_sum || current_sum <= Real(0)) return;

        Real remaining_total = current_sum;
        Real remaining_target = target_sum;

        for (Size k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;

            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::BinomialSampler::sample(rng, static_cast<Index>(count), p_select);

            vals[k] = static_cast<typename MatrixT::ValueType>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Downsample with variable targets
template <typename MatrixT>
    requires AnySparse<MatrixT>
void downsample_variable_impl(
    MatrixT& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(primary_dim), "Target counts size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Xoshiro256pp rng(seed + p * 0x9e3779b97f4a7c15ULL);

        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        const Size len = vals.len;

        if (len == 0) return;

        Real current_sum = detail::sum_4way(vals.ptr, len);
        Real target = target_counts[p];

        if (current_sum <= target || current_sum <= Real(0)) return;

        Real remaining_total = current_sum;
        Real remaining_target = target;

        for (Size k = 0; k < len && remaining_target > Real(0); ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count <= Real(0)) continue;

            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::BinomialSampler::sample(rng, static_cast<Index>(count), p_select);

            vals[k] = static_cast<typename MatrixT::ValueType>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
}

/// @brief Binomial resampling
template <typename MatrixT>
    requires AnySparse<MatrixT>
void binomial_resample_impl(
    MatrixT& matrix,
    Real p,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        Xoshiro256pp rng(seed + row * 0x9e3779b97f4a7c15ULL);

        auto vals = scl::primary_values(matrix, static_cast<Index>(row));

        for (Size k = 0; k < vals.len; ++k) {
            Index count = static_cast<Index>(vals[k]);
            if (count > 0) {
                vals[k] = static_cast<typename MatrixT::ValueType>(
                    detail::BinomialSampler::sample(rng, count, p));
            }
        }
    });
}

/// @brief Poisson resampling
template <typename MatrixT>
    requires AnySparse<MatrixT>
void poisson_resample_impl(
    MatrixT& matrix,
    Real lambda,
    uint64_t seed
) {
    const Index primary_dim = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t row) {
        Xoshiro256pp rng(seed + row * 0x9e3779b97f4a7c15ULL);

        auto vals = scl::primary_values(matrix, static_cast<Index>(row));

        for (Size k = 0; k < vals.len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count > Real(0)) {
                vals[k] = static_cast<typename MatrixT::ValueType>(
                    detail::PoissonSampler::sample(rng, count * lambda));
            }
        }
    });
}

} // namespace generic

// =============================================================================
// SECTION 4: Public API with Backend Dispatch
// =============================================================================

/// @brief Downsample counts to target sum
///
/// Uses multinomial sampling with efficient Binomial subroutine.
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param target_sum Target sum for each row
/// @param seed Random seed
template <typename MatrixT>
    requires AnySparse<MatrixT>
void downsample_counts(
    MatrixT& matrix,
    Real target_sum,
    uint64_t seed = 42
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::downsample_fast<MatrixT, IsCSR>(matrix, target_sum, seed);
    } else {
        generic::downsample_counts_impl(matrix, target_sum, seed);
    }
}

/// @brief Downsample with per-row target counts
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param target_counts Target sum for each row [size = primary_dim]
/// @param seed Random seed
template <typename MatrixT>
    requires AnySparse<MatrixT>
void downsample_counts_variable(
    MatrixT& matrix,
    Array<const Real> target_counts,
    uint64_t seed = 42
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::downsample_variable_fast<MatrixT, IsCSR>(matrix, target_counts, seed);
    } else {
        generic::downsample_variable_impl(matrix, target_counts, seed);
    }
}

/// @brief Binomial resampling: c' ~ Binomial(c, p)
///
/// Each count is resampled independently.
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param p Success probability [0, 1]
/// @param seed Random seed
template <typename MatrixT>
    requires AnySparse<MatrixT>
void binomial_resample(
    MatrixT& matrix,
    Real p,
    uint64_t seed = 42
) {
    SCL_CHECK_ARG(p >= Real(0) && p <= Real(1), "Binomial resample: p must be in [0, 1]");

    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::binomial_resample_fast<MatrixT, IsCSR>(matrix, p, seed);
    } else {
        generic::binomial_resample_impl(matrix, p, seed);
    }
}

/// @brief Poisson resampling: c' ~ Poisson(c * lambda)
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param lambda Rate multiplier (>= 0)
/// @param seed Random seed
template <typename MatrixT>
    requires AnySparse<MatrixT>
void poisson_resample(
    MatrixT& matrix,
    Real lambda,
    uint64_t seed = 42
) {
    SCL_CHECK_ARG(lambda >= Real(0), "Poisson resample: lambda must be >= 0");

    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::poisson_resample_fast<MatrixT, IsCSR>(matrix, lambda, seed);
    } else {
        generic::poisson_resample_impl(matrix, lambda, seed);
    }
}

// =============================================================================
// SECTION 5: Mapped Matrix API (Returns OwnedSparse)
// =============================================================================

/// @brief Downsample mapped matrix (returns new OwnedSparse)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, std::is_same_v<typename MatrixT::Tag, TagCSR>>
auto downsample_counts_mapped(
    const MatrixT& matrix,
    Real target_sum,
    uint64_t seed = 42
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    return mapped::downsample_mapped_dispatch<MatrixT, IsCSR>(matrix, target_sum, seed);
}

/// @brief Downsample mapped matrix with variable targets (returns new OwnedSparse)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, std::is_same_v<typename MatrixT::Tag, TagCSR>>
auto downsample_counts_variable_mapped(
    const MatrixT& matrix,
    Array<const Real> target_counts,
    uint64_t seed = 42
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    return mapped::downsample_variable_mapped_dispatch<MatrixT, IsCSR>(matrix, target_counts, seed);
}

/// @brief Binomial resample mapped matrix (returns new OwnedSparse)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, std::is_same_v<typename MatrixT::Tag, TagCSR>>
auto binomial_resample_mapped(
    const MatrixT& matrix,
    Real p,
    uint64_t seed = 42
) {
    SCL_CHECK_ARG(p >= Real(0) && p <= Real(1), "Binomial resample: p must be in [0, 1]");
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    return mapped::binomial_resample_mapped_dispatch<MatrixT, IsCSR>(matrix, p, seed);
}

/// @brief Poisson resample mapped matrix (returns new OwnedSparse)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, std::is_same_v<typename MatrixT::Tag, TagCSR>>
auto poisson_resample_mapped(
    const MatrixT& matrix,
    Real lambda,
    uint64_t seed = 42
) {
    SCL_CHECK_ARG(lambda >= Real(0), "Poisson resample: lambda must be >= 0");
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    return mapped::poisson_resample_mapped_dispatch<MatrixT, IsCSR>(matrix, lambda, seed);
}

} // namespace scl::kernel::resample
