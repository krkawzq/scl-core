#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>

// =============================================================================
/// @file resample_mapped_impl.hpp
/// @brief Resampling for Memory-Mapped Sparse Matrices
///
/// ## Design
///
/// Mapped matrices are READ-ONLY. For resampling:
/// 1. Materialize to OwnedSparse
/// 2. Apply sampling in-place on owned data
/// 3. Return OwnedSparse
///
/// ## Key Optimizations
///
/// 1. Fused Read + Sample (avoid intermediate buffer)
/// 2. Fast Xoshiro256++ RNG
/// 3. 4-Way SIMD Sum Accumulation
/// 4. Prefetch Hints for Sequential Access
// =============================================================================

namespace scl::kernel::resample::mapped {

// =============================================================================
// SECTION 1: Fast RNG and Samplers
// =============================================================================

namespace detail {

/// @brief Xoshiro256++ fast PRNG
class FastRNG {
public:
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

/// @brief 4-way sum
template <typename T>
SCL_FORCE_INLINE Real sum_4way(const T* SCL_RESTRICT vals, Size len) {
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
// SECTION 2: MappedCustomSparse
// =============================================================================

/// @brief Downsample with target sum
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Real target_sum,
    uint64_t seed
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    // Allocate owned storage
    scl::io::OwnedSparse<T, IsCSR> owned(matrix.rows, matrix.cols, nnz);

    // Copy structure
    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    // Fused copy + sample
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);

        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        const T* src = matrix.data() + start;
        T* dst = owned.data.data() + start;
        
        // Compute sum
        Real current_sum = detail::sum_4way(src, len);
        
        if (current_sum <= target_sum || current_sum <= Real(0)) {
            // Just copy
            std::copy(src, src + len, dst);
            return;
        }
        
        // Multinomial sampling
        Real remaining_total = current_sum;
        Real remaining_target = target_sum;
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(src[k]);
            
            if (count <= Real(0) || remaining_target <= Real(0)) {
                dst[k] = static_cast<T>(0);
                continue;
            }
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            dst[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
    
    return owned;
}

/// @brief Downsample with variable targets
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_variable_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(n_primary), "Target counts size mismatch");

    scl::io::OwnedSparse<T, IsCSR> owned(matrix.rows, matrix.cols, nnz);

    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);

        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        const T* src = matrix.data() + start;
        T* dst = owned.data.data() + start;
        
        Real current_sum = detail::sum_4way(src, len);
        Real target = target_counts[p];
        
        if (current_sum <= target || current_sum <= Real(0)) {
            std::copy(src, src + len, dst);
            return;
        }
        
        Real remaining_total = current_sum;
        Real remaining_target = target;
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(src[k]);
            
            if (count <= Real(0) || remaining_target <= Real(0)) {
                dst[k] = static_cast<T>(0);
                continue;
            }
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            dst[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
    
    return owned;
}

/// @brief Binomial resample
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> binomial_resample_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Real p,
    uint64_t seed
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    scl::io::OwnedSparse<T, IsCSR> owned(matrix.rows, matrix.cols, nnz);

    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);

        Index start = matrix.indptr()[row];
        Index end = matrix.indptr()[row + 1];
        Size len = static_cast<Size>(end - start);

        const T* src = matrix.data() + start;
        T* dst = owned.data.data() + start;

        for (Size k = 0; k < len; ++k) {
            Index count = static_cast<Index>(src[k]);
            dst[k] = (count > 0) ? static_cast<T>(detail::FastBinomial::sample(rng, count, p)) : T(0);
        }
    });

    return owned;
}

/// @brief Poisson resample
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> poisson_resample_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Real lambda,
    uint64_t seed
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    scl::io::OwnedSparse<T, IsCSR> owned(matrix.rows, matrix.cols, nnz);

    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);

        Index start = matrix.indptr()[row];
        Index end = matrix.indptr()[row + 1];
        Size len = static_cast<Size>(end - start);

        const T* src = matrix.data() + start;
        T* dst = owned.data.data() + start;

        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(src[k]);
            dst[k] = (count > Real(0)) ? static_cast<T>(detail::FastPoisson::sample(rng, count * lambda)) : T(0);
        }
    });

    return owned;
}

// =============================================================================
// SECTION 3: MappedVirtualSparse
// =============================================================================

/// @brief Downsample MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Real target_sum,
    uint64_t seed
) {
    auto owned = matrix.materialize();
    
    const Index n_primary = scl::primary_size(owned);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Size len = static_cast<Size>(end - start);
        
        if (len == 0) return;
        
        T* vals = owned.data.data() + start;
        
        Real current_sum = detail::sum_4way(vals, len);
        
        if (current_sum <= target_sum || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target_sum;
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            
            if (count <= Real(0) || remaining_target <= Real(0)) {
                vals[k] = T(0);
                continue;
            }
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
    
    return owned;
}

/// @brief Downsample MappedVirtualSparse with variable targets
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_variable_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    auto owned = matrix.materialize();
    
    const Index n_primary = scl::primary_size(owned);
    
    SCL_CHECK_DIM(target_counts.len >= static_cast<Size>(n_primary), "Target counts size mismatch");
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        detail::FastRNG rng(seed + p * 0x9e3779b97f4a7c15ULL);
        
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Size len = static_cast<Size>(end - start);
        
        if (len == 0) return;
        
        T* vals = owned.data.data() + start;
        
        Real current_sum = detail::sum_4way(vals, len);
        Real target = target_counts[p];
        
        if (current_sum <= target || current_sum <= Real(0)) return;
        
        Real remaining_total = current_sum;
        Real remaining_target = target;
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            
            if (count <= Real(0) || remaining_target <= Real(0)) {
                vals[k] = T(0);
                continue;
            }
            
            Real p_select = remaining_target / remaining_total;
            Index sampled = detail::FastBinomial::sample(rng, static_cast<Index>(count), p_select);
            
            vals[k] = static_cast<T>(sampled);
            remaining_target -= static_cast<Real>(sampled);
            remaining_total -= count;
        }
    });
    
    return owned;
}

/// @brief Binomial resample MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> binomial_resample_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Real p,
    uint64_t seed
) {
    auto owned = matrix.materialize();
    
    const Index n_primary = scl::primary_size(owned);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index start = owned.indptr[row];
        Index end = owned.indptr[row + 1];
        
        T* vals = owned.data.data() + start;
        Size len = static_cast<Size>(end - start);
        
        for (Size k = 0; k < len; ++k) {
            Index count = static_cast<Index>(vals[k]);
            if (count > 0) {
                vals[k] = static_cast<T>(detail::FastBinomial::sample(rng, count, p));
            }
        }
    });
    
    return owned;
}

/// @brief Poisson resample MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> poisson_resample_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Real lambda,
    uint64_t seed
) {
    auto owned = matrix.materialize();
    
    const Index n_primary = scl::primary_size(owned);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t row) {
        detail::FastRNG rng(seed + row * 0x9e3779b97f4a7c15ULL);
        
        Index start = owned.indptr[row];
        Index end = owned.indptr[row + 1];
        
        T* vals = owned.data.data() + start;
        Size len = static_cast<Size>(end - start);
        
        for (Size k = 0; k < len; ++k) {
            Real count = static_cast<Real>(vals[k]);
            if (count > Real(0)) {
                vals[k] = static_cast<T>(detail::FastPoisson::sample(rng, count * lambda));
            }
        }
    });
    
    return owned;
}

// =============================================================================
// SECTION 4: Unified Dispatchers
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto downsample_mapped_dispatch(
    const MatrixT& matrix,
    Real target_sum,
    uint64_t seed
) {
    return downsample_mapped(matrix, target_sum, seed);
}

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto downsample_variable_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    return downsample_variable_mapped(matrix, target_counts, seed);
}

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto binomial_resample_mapped_dispatch(
    const MatrixT& matrix,
    Real p,
    uint64_t seed
) {
    return binomial_resample_mapped(matrix, p, seed);
}

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto poisson_resample_mapped_dispatch(
    const MatrixT& matrix,
    Real lambda,
    uint64_t seed
) {
    return poisson_resample_mapped(matrix, lambda, seed);
}

} // namespace scl::kernel::resample::mapped
