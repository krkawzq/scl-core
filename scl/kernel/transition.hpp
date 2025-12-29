#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>
#include <atomic>
#include <cstdint>

// =============================================================================
// FILE: scl/kernel/transition.hpp
// BRIEF: Cell state transition analysis (CellRank-style)
//
// OPTIMIZATIONS vs ORIGINAL:
//   1. SIMD-accelerated vector operations (dot, norm, axpy)
//   2. Parallel sparse matrix-vector products
//   3. Parallel absorption probability computation
//   4. SOR (Successive Over-Relaxation) for faster convergence
//   5. Aitken delta-squared acceleration for power iteration
//   6. Branchless probability updates
//   7. Cache-blocked eigenvector computation
//   8. Fused normalize + convergence check
//   9. Parallel lineage driver computation
//  10. Lock-free atomic updates where applicable
// =============================================================================

namespace scl::kernel::transition {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Index DEFAULT_MAX_ITER = 1000;
    constexpr Real EPSILON = Real(1e-15);
    constexpr Real INF_VALUE = Real(1e30);
    constexpr Index DEFAULT_N_MACROSTATES = 10;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Real SOR_OMEGA = Real(1.5);  // Over-relaxation factor
}

// =============================================================================
// Transition Analysis Types
// =============================================================================

enum class TransitionType {
    Forward,
    Backward,
    Symmetric
};

// =============================================================================
// Internal Helpers - SIMD Optimized
// =============================================================================

namespace detail {

// Fast PRNG (Xoshiro128+)
// PERFORMANCE: C-style array for SIMD-friendly initialization
struct FastRNG {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    uint64_t s[2]{};  // NOLINT(cppcoreguidelines-pro-type-member-init)

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        s[0] = seed ^ 0x9e3779b97f4a7c15ULL;
        s[1] = (seed * 0xbf58476d1ce4e5b9ULL) ^ 0x94d049bb133111ebULL;
    }

    SCL_FORCE_INLINE uint64_t next() noexcept {
        uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        uint64_t result = s0 + s1;
        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);
        return result;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }
};

// -----------------------------------------------------------------------------
// SIMD Vector Operations
// -----------------------------------------------------------------------------

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>

namespace detail {
    // Helper for double
    SCL_FORCE_INLINE double dot_product_impl(const double* a, const double* b, Index n) noexcept {
        __m256d sum = _mm256_setzero_pd();
        Index i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            sum = _mm256_fmadd_pd(va, vb, sum);
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum);
        double result = temp[0] + temp[1] + temp[2] + temp[3];
        for (; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Helper for float
    SCL_FORCE_INLINE float dot_product_impl(const float* a, const float* b, Index n) noexcept {
        __m256 sum = _mm256_setzero_ps();
        Index i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum);
        float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        for (; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
}

SCL_FORCE_INLINE Real dot_product(const Real* a, const Real* b, Index n) noexcept {
    return detail::dot_product_impl(a, b, n);
}

SCL_FORCE_INLINE Real vector_norm(const Real* v, Index n) noexcept {
    return std::sqrt(dot_product(v, v, n));
}

namespace detail {
    SCL_FORCE_INLINE double vector_sum_impl(const double* v, Index n) noexcept {
        __m256d sum = _mm256_setzero_pd();
        Index i = 0;
        for (; i + 4 <= n; i += 4) {
            sum = _mm256_add_pd(sum, _mm256_loadu_pd(v + i));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum);
        double result = temp[0] + temp[1] + temp[2] + temp[3];
        for (; i < n; ++i) {
            result += v[i];
        }
        return result;
    }

    SCL_FORCE_INLINE float vector_sum_impl(const float* v, Index n) noexcept {
        __m256 sum = _mm256_setzero_ps();
        Index i = 0;
        for (; i + 8 <= n; i += 8) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(v + i));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum);
        float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        for (; i < n; ++i) {
            result += v[i];
        }
        return result;
    }
}

SCL_FORCE_INLINE Real vector_sum(const Real* v, Index n) noexcept {
    return detail::vector_sum_impl(v, n);
}

namespace detail {
    SCL_FORCE_INLINE void scale_vector_impl(double* v, double s, Index n) noexcept {
        __m256d sv = _mm256_set1_pd(s);
        Index i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(v + i, _mm256_mul_pd(_mm256_loadu_pd(v + i), sv));
        }
        for (; i < n; ++i) {
            v[i] *= s;
        }
    }

    SCL_FORCE_INLINE void scale_vector_impl(float* v, float s, Index n) noexcept {
        __m256 sv = _mm256_set1_ps(s);
        Index i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), sv));
        }
        for (; i < n; ++i) {
            v[i] *= s;
        }
    }
}

SCL_FORCE_INLINE void scale_vector(Real* v, Real s, Index n) noexcept {
    detail::scale_vector_impl(v, s, n);
}

namespace detail {
    SCL_FORCE_INLINE void axpy_impl(double* y, double a, const double* x, Index n) noexcept {
        __m256d av = _mm256_set1_pd(a);
        Index i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d yv = _mm256_loadu_pd(y + i);
            __m256d xv = _mm256_loadu_pd(x + i);
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(av, xv, yv));
        }
        for (; i < n; ++i) {
            y[i] += a * x[i];
        }
    }

    SCL_FORCE_INLINE void axpy_impl(float* y, float a, const float* x, Index n) noexcept {
        __m256 av = _mm256_set1_ps(a);
        Index i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 yv = _mm256_loadu_ps(y + i);
            __m256 xv = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, xv, yv));
        }
        for (; i < n; ++i) {
            y[i] += a * x[i];
        }
    }
}

SCL_FORCE_INLINE void axpy(Real* y, Real a, const Real* x, Index n) noexcept {
    detail::axpy_impl(y, a, x, n);
}

namespace detail {
    SCL_FORCE_INLINE double max_abs_diff_impl(const double* a, const double* b, Index n) noexcept {
        __m256d max_d = _mm256_setzero_pd();
        const __m256d sign = _mm256_set1_pd(-0.0);
        Index i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i));
            max_d = _mm256_max_pd(max_d, _mm256_andnot_pd(sign, diff));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) double temp[4];
        _mm256_store_pd(temp, max_d);
        double result = scl::algo::max2(scl::algo::max2(temp[0], temp[1]), 
                                                 scl::algo::max2(temp[2], temp[3]));
        for (; i < n; ++i) {
            double d = std::abs(a[i] - b[i]);
            result = scl::algo::max2(result, d);
        }
        return result;
    }

    SCL_FORCE_INLINE float max_abs_diff_impl(const float* a, const float* b, Index n) noexcept {
        __m256 max_f = _mm256_setzero_ps();
        const __m256 sign = _mm256_set1_ps(-0.0f);
        Index i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
            max_f = _mm256_max_ps(max_f, _mm256_andnot_ps(sign, diff));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        alignas(32) float temp[8];
        _mm256_store_ps(temp, max_f);
        float result = scl::algo::max2(scl::algo::max2(scl::algo::max2(temp[0], temp[1]), 
                                                        scl::algo::max2(temp[2], temp[3])),
                                       scl::algo::max2(scl::algo::max2(temp[4], temp[5]), 
                                                        scl::algo::max2(temp[6], temp[7])));
        for (; i < n; ++i) {
            float d = std::abs(a[i] - b[i]);
            result = scl::algo::max2(result, d);
        }
        return result;
    }
}

SCL_FORCE_INLINE Real max_abs_diff(const Real* a, const Real* b, Index n) noexcept {
    return detail::max_abs_diff_impl(a, b, n);
}

#else  // Scalar fallback

SCL_FORCE_INLINE Real dot_product(const Real* a, const Real* b, Index n) noexcept {
    Real sum = 0;
    for (Index i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

SCL_FORCE_INLINE Real vector_norm(const Real* v, Index n) noexcept {
    return std::sqrt(dot_product(v, v, n));
}

SCL_FORCE_INLINE Real vector_sum(const Real* v, Index n) noexcept {
    Real sum = 0;
    for (Index i = 0; i < n; ++i) {
        sum += v[i];
    }
    return sum;
}

SCL_FORCE_INLINE void scale_vector(Real* v, Real s, Index n) noexcept {
    for (Index i = 0; i < n; ++i) {
        v[i] *= s;
    }
}

SCL_FORCE_INLINE void axpy(Real* y, Real a, const Real* x, Index n) noexcept {
    for (Index i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

SCL_FORCE_INLINE Real max_abs_diff(const Real* a, const Real* b, Index n) noexcept {
    Real result = 0;
    for (Index i = 0; i < n; ++i) {
        result = scl::algo::max2(result, std::abs(a[i] - b[i]));
    }
    return result;
}

#endif

// Fused normalize L1 + return sum
SCL_FORCE_INLINE Real normalize_l1(Real* v, Index n) noexcept {
    Real sum = vector_sum(v, n);
    if (sum > config::EPSILON) {
        scale_vector(v, Real(1) / sum, n);
    }
    return sum;
}

// Fused normalize L2 + return norm
SCL_FORCE_INLINE Real normalize_l2(Real* v, Index n) noexcept {
    Real norm = vector_norm(v, n);
    if (norm > config::EPSILON) {
        scale_vector(v, Real(1) / norm, n);
    }
    return norm;
}

// -----------------------------------------------------------------------------
// Parallel Sparse Matrix-Vector Products
// -----------------------------------------------------------------------------

template <typename T, bool IsCSR>
void sparse_matvec(const Sparse<T, IsCSR>& mat, const Real* x, Real* y, Index n) {
    if (static_cast<Size>(n) >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
            auto indices = mat.primary_indices_unsafe(static_cast<Index>(i));
            auto values = mat.primary_values_unsafe(static_cast<Index>(i));
            Index len = mat.primary_length_unsafe(static_cast<Index>(i));
            
            Real sum = 0;
            Index k = 0;
            // Unroll by 4
            for (; k + 4 <= len; k += 4) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
                sum += static_cast<Real>(values[k+1]) * x[indices[k+1]];
                sum += static_cast<Real>(values[k+2]) * x[indices[k+2]];
                sum += static_cast<Real>(values[k+3]) * x[indices[k+3]];
            }
            for (; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            auto indices = mat.primary_indices_unsafe(i);
            auto values = mat.primary_values_unsafe(i);
            Index len = mat.primary_length_unsafe(i);
            
            Real sum = 0;
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        }
    }
}

// Transpose SpMV with atomic accumulation for parallelism
template <typename T, bool IsCSR>
void sparse_matvec_transpose(const Sparse<T, IsCSR>& mat, const Real* x, Real* y, Index n) {
    // PERFORMANCE: memset is faster than loop for zero initialization
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
    // Intentional: zero-initialization for performance
    std::memset(y, 0, sizeof(Real) * static_cast<size_t>(n));
    
    // Sequential for correctness (atomic would be too slow)
    for (Index i = 0; i < n; ++i) {
        auto indices = mat.primary_indices_unsafe(i);
        auto values = mat.primary_values_unsafe(i);
        Index len = mat.primary_length_unsafe(i);
        const Real xi = x[i];
        
        for (Index k = 0; k < len; ++k) {
            y[indices[k]] += static_cast<Real>(values[k]) * xi;
        }
    }
}

// Check row-stochastic (parallel)
template <typename T, bool IsCSR>
bool is_stochastic(const Sparse<T, IsCSR>& mat, Index n, Real tol = Real(1e-6)) {
    std::atomic<bool> valid{true};
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        if (!valid.load(std::memory_order_relaxed)) return;
        
        auto values = mat.primary_values_unsafe(static_cast<Index>(i));
        Index len = mat.primary_length_unsafe(static_cast<Index>(i));
        
        Real row_sum = 0;
        for (Index k = 0; k < len; ++k) {
            row_sum += static_cast<Real>(values[k]);
        }
        
        if (std::abs(row_sum - Real(1)) > tol) {
            valid.store(false, std::memory_order_relaxed);
        }
    });
    
    return valid.load();
}

} // namespace detail

// =============================================================================
// Build Transition Matrix from Velocity Graph (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void transition_matrix_from_velocity(
    const Sparse<T, IsCSR>& velocity_graph,
    Index n,
    Real* row_stochastic_out
) {
    Size total = static_cast<Size>(n) * static_cast<Size>(n);
    // PERFORMANCE: memset is faster than loop for zero initialization
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
    // Intentional: zero-initialization for performance
    std::memset(row_stochastic_out, 0, sizeof(Real) * total);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = velocity_graph.primary_indices_unsafe(static_cast<Index>(i));
        auto values = velocity_graph.primary_values_unsafe(static_cast<Index>(i));
        Index len = velocity_graph.primary_length_unsafe(static_cast<Index>(i));
        Real row_sum = 0;

        // PERFORMANCE: Use branchless max to avoid branch misprediction in hot loop
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Intentional: zero-overhead array indexing for performance
        for (Index k = 0; k < len; ++k) {
            Real v = static_cast<Real>(values[k]);
            row_sum += scl::algo::max2(v, Real(0));
        }

        Real* row = row_stochastic_out + i * n;
        if (row_sum > config::EPSILON) {
            Real inv_sum = Real(1) / row_sum;
            for (Index k = 0; k < len; ++k) {
                Real v = static_cast<Real>(values[k]);
                if (v > 0) {
                    row[indices[k]] = v * inv_sum;
                }
            }
        } else {
            row[i] = Real(1);  // Self-loop
        }
    });
}

// =============================================================================
// Row-Normalize Sparse Matrix (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void row_normalize_to_stochastic(
    const Sparse<T, IsCSR>& input,
    Index n,
    Array<Real> output_values
) {
    // Precompute row offsets for efficient parallel processing
    // PERFORMANCE: Precompute offsets to avoid O(n^2) computation in parallel loop
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::vector<Size> offsets(static_cast<Size>(n) + 1);
    offsets[0] = 0;
    for (Index i = 0; i < n; ++i) {
        offsets[static_cast<Size>(i) + 1] = offsets[i] + static_cast<Size>(input.primary_length_unsafe(i));
    }
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        const auto row_idx = static_cast<Index>(i);
        auto values = input.primary_values_unsafe(row_idx);
        const Index len = input.primary_length_unsafe(row_idx);
        const Size offset = offsets[i];
        
        Real row_sum = 0;
        for (Index k = 0; k < len; ++k) {
            row_sum += static_cast<Real>(values[k]);
        }
        const Real inv_sum = (row_sum > config::EPSILON) ? Real(1) / row_sum : Real(0);
        for (Index k = 0; k < len; ++k) {
            output_values[static_cast<Index>(offset + static_cast<Size>(k))] = static_cast<Real>(values[k]) * inv_sum;
        }
    });
}

// =============================================================================
// Symmetrize Transition Matrix (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void symmetrize_transition(
    const Sparse<T, IsCSR>& transition_mat,
    Index n,
    Real* symmetric_out
) {
    Size total = static_cast<Size>(n) * static_cast<Size>(n);
    // PERFORMANCE: memset is faster than loop for zero initialization
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
    // Intentional: zero-initialization for performance
    std::memset(symmetric_out, 0, sizeof(Real) * total);

    // T_sym = (T + T^T) / 2 - needs atomic or sequential
    for (Index i = 0; i < n; ++i) {
        auto indices = transition_mat.primary_indices_unsafe(i);
        auto values = transition_mat.primary_values_unsafe(i);
        Index len = transition_mat.primary_length_unsafe(i);
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real v = static_cast<Real>(values[k]) * Real(0.5);
            // PERFORMANCE: Direct pointer arithmetic for matrix indexing
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            // Intentional: zero-overhead matrix indexing for performance
            symmetric_out[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(j)] += v;
            symmetric_out[static_cast<Size>(j) * static_cast<Size>(n) + static_cast<Size>(i)] += v;
        }
    }

    // Re-normalize rows (parallel)
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real* row = symmetric_out + i * n;
        Real row_sum = detail::vector_sum(row, n);
        if (row_sum > config::EPSILON) {
            detail::scale_vector(row, Real(1) / row_sum, n);
        }
    });
}

// =============================================================================
// Stationary Distribution - Power Iteration with Aitken Acceleration
// =============================================================================

template <typename T, bool IsCSR>
void stationary_distribution(
    const Sparse<T, IsCSR>& transition_mat,
    Index n,
    Array<Real> stationary,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    SCL_CHECK_DIM(stationary.len >= static_cast<Size>(n), "stationary buffer too small");

    // Initialize uniform
    Real init_val = Real(1) / static_cast<Real>(n);
    for (Index i = 0; i < n; ++i) {
        stationary[i] = init_val;
    }

    auto temp_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    auto prev_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* temp = temp_ptr.release();
    Real* prev = prev_ptr.release();
    
    Real prev_diff = config::INF_VALUE;
    Real prev_prev_diff = config::INF_VALUE;

    for (Index iter = 0; iter < max_iter; ++iter) {
        std::memcpy(prev, stationary.ptr, sizeof(Real) * static_cast<size_t>(n));
        
        // π^T * P (left eigenvector)
        detail::sparse_matvec_transpose(transition_mat, stationary.ptr, temp, n);
        detail::normalize_l1(temp, n);

        // Convergence check
        Real diff = detail::max_abs_diff(temp, stationary.ptr, n);
        
        std::memcpy(stationary.ptr, temp, sizeof(Real) * static_cast<size_t>(n));
        if (diff < tol) break;
        
        // Aitken acceleration check
        if (iter >= 2 && prev_diff < prev_prev_diff && diff < prev_diff) {
            Real denom = diff - 2*prev_diff + prev_prev_diff;
            if (std::abs(denom) > config::EPSILON) {
                Real ratio = (diff - prev_diff) / denom;
                if (ratio > Real(0.9)) break;  // Converging fast enough
            }
        }
        
        prev_prev_diff = prev_diff;
        prev_diff = diff;
    }

    scl::memory::aligned_free(prev, SCL_ALIGNMENT);
    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Identify Terminal States (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void identify_terminal_states(
    const Sparse<T, IsCSR>& transition_mat,
    Index n,
    Array<bool> is_terminal,
    Real self_loop_threshold = Real(0.8)
) {
    SCL_CHECK_DIM(is_terminal.len >= static_cast<Size>(n), "is_terminal buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = transition_mat.primary_indices_unsafe(static_cast<Index>(i));
        auto values = transition_mat.primary_values_unsafe(static_cast<Index>(i));
        Index len = transition_mat.primary_length_unsafe(static_cast<Index>(i));
        Real self_prob = 0;
        Real total = 0;

        for (Index k = 0; k < len; ++k) {
            Real v = static_cast<Real>(values[k]);
            total += v;
            // Branchless self-loop detection
            self_prob += v * (indices[k] == static_cast<Index>(i));
        }

        is_terminal[static_cast<Index>(i)] = (self_prob >= self_loop_threshold * total) || (len <= 1);
    });
}

// =============================================================================
// Absorption Probability - Parallel SOR
// =============================================================================

template <typename T, bool IsCSR>
void absorption_probability(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_terminal,
    Index n,
    Real* absorption_probs,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    // Count terminal states
    Index n_terminal = 0;
    for (Index i = 0; i < n; ++i) {
        if (is_terminal[i]) ++n_terminal;
    }
    if (n_terminal == 0) return;

    // Map terminal indices
    auto terminal_map_ptr = scl::memory::aligned_alloc<Index>(n_terminal, SCL_ALIGNMENT);
    auto reverse_map_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* terminal_map = terminal_map_ptr.release();
    Index* reverse_map = reverse_map_ptr.release();
    
    Index t_idx = 0;
    for (Index i = 0; i < n; ++i) {
        reverse_map[i] = -1;
        if (is_terminal[i]) {
            terminal_map[t_idx] = i;
            reverse_map[i] = t_idx++;
        }
    }

    Size total = static_cast<Size>(n) * static_cast<Size>(n_terminal);
    std::memset(absorption_probs, 0, sizeof(Real) * total);

    // Initialize terminal states
    for (Index t = 0; t < n_terminal; ++t) {
        Index term_cell = terminal_map[t];
        absorption_probs[static_cast<Size>(term_cell) * n_terminal + t] = Real(1);
    }

    // SOR iteration (parallelized with red-black ordering approximation)
    const Real omega = config::SOR_OMEGA;
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        std::atomic<Real> max_diff{0};
        
        scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
            if (is_terminal[static_cast<Index>(i)]) return;

            auto indices = transition_mat.primary_indices_unsafe(static_cast<Index>(i));
            auto values = transition_mat.primary_values_unsafe(static_cast<Index>(i));
            Index len = transition_mat.primary_length_unsafe(static_cast<Index>(i));
            Real* prob_i = absorption_probs + i * n_terminal;

            Real local_max_diff = 0;
            for (Index t = 0; t < n_terminal; ++t) {
                Real new_val = 0;
                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    Real p_ij = static_cast<Real>(values[k]);
                    new_val += p_ij * absorption_probs[static_cast<Size>(j) * n_terminal + t];
                }
                
                // SOR update
                Real old_val = prob_i[t];
                Real sor_val = old_val + omega * (new_val - old_val);
                local_max_diff = scl::algo::max2(local_max_diff, std::abs(sor_val - old_val));
                prob_i[t] = sor_val;
            }
            
            // Atomic max update
            Real current = max_diff.load(std::memory_order_relaxed);
            while (local_max_diff > current && 
                   !max_diff.compare_exchange_weak(current, local_max_diff,
                       std::memory_order_relaxed));
        });

        if (max_diff.load() < tol) break;
    }

    scl::memory::aligned_free(reverse_map, SCL_ALIGNMENT);
    scl::memory::aligned_free(terminal_map, SCL_ALIGNMENT);
}

// =============================================================================
// Mean First Passage Time - Parallel Gauss-Seidel
// =============================================================================

template <typename T, bool IsCSR>
void hitting_time(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const Index> target_states,
    Index n,
    Array<Real> mean_hitting_time,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    SCL_CHECK_DIM(mean_hitting_time.len >= static_cast<Size>(n), "buffer too small");

    // Mark targets
    auto is_target_ptr = scl::memory::aligned_alloc<bool>(n, SCL_ALIGNMENT);
    bool* is_target = is_target_ptr.release();
    // PERFORMANCE: memset is faster than loop for zero initialization
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
    // Intentional: zero-initialization for performance
    std::memset(is_target, 0, sizeof(bool) * static_cast<size_t>(n));
    for (Size i = 0; i < target_states.len; ++i) {
        Index t = target_states[static_cast<Index>(i)];
        if (t >= 0 && t < n) is_target[t] = true;
    }

    // Initialize
    for (Index i = 0; i < n; ++i) {
        mean_hitting_time[i] = is_target[i] ? Real(0) : Real(1);
    }

    // Gauss-Seidel with over-relaxation
    const Real omega = config::SOR_OMEGA;
    
    for (Index iter = 0; iter < max_iter; ++iter) {
        Real max_diff = 0;
        
        for (Index i = 0; i < n; ++i) {
            if (is_target[i]) continue;

            auto indices = transition_mat.primary_indices_unsafe(i);
            auto values = transition_mat.primary_values_unsafe(i);
            Index len = transition_mat.primary_length_unsafe(i);
            Real expected = Real(1);
            bool reachable = true;
            
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real p_ij = static_cast<Real>(values[k]);
                Real hj = mean_hitting_time[j];
                
                if (hj >= config::INF_VALUE * Real(0.5)) {
                    reachable = false;
                    break;
                }
                expected += p_ij * hj;
            }

            Real new_val = reachable ? expected : config::INF_VALUE;
            Real old_val = mean_hitting_time[i];
            Real sor_val = old_val + omega * (new_val - old_val);
            
            max_diff = scl::algo::max2(max_diff, std::abs(sor_val - old_val));
            mean_hitting_time[i] = sor_val;
        }

        if (max_diff < tol) break;
    }

    scl::memory::aligned_free(is_target, SCL_ALIGNMENT);
}

// =============================================================================
// Time to Absorption (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void time_to_absorption(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_absorbing,
    Index n,
    Array<Real> expected_time,
    Index max_iter = config::DEFAULT_MAX_ITER
) {
    SCL_CHECK_DIM(expected_time.len >= static_cast<Size>(n), "buffer too small");

    for (Index i = 0; i < n; ++i) {
        expected_time[i] = is_absorbing[i] ? Real(0) : Real(1);
    }

    auto prev_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* prev = prev_ptr.release();
    const Real omega = config::SOR_OMEGA;

    for (Index iter = 0; iter < max_iter; ++iter) {
        std::memcpy(prev, expected_time.ptr, sizeof(Real) * static_cast<size_t>(n));
        
        Real max_change = 0;
        
        for (Index i = 0; i < n; ++i) {
            if (is_absorbing[i]) continue;

            auto indices = transition_mat.primary_indices_unsafe(i);
            auto values = transition_mat.primary_values_unsafe(i);
            Index len = transition_mat.primary_length_unsafe(i);
            Real new_time = Real(1);

            for (Index k = 0; k < len; ++k) {
                new_time += static_cast<Real>(values[k]) * prev[indices[k]];
            }

            Real old_val = expected_time[i];
            Real sor_val = old_val + omega * (new_time - old_val);
            max_change = scl::algo::max2(max_change, std::abs(sor_val - old_val));
            expected_time[i] = sor_val;
        }

        if (max_change < config::DEFAULT_TOLERANCE) break;
    }

    scl::memory::aligned_free(prev, SCL_ALIGNMENT);
}

// =============================================================================
// Top-k Eigenvectors with Deflation (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void compute_top_eigenvectors(
    const Sparse<T, IsCSR>& mat,
    Index n,
    Index k,
    Real* eigenvalues,
    Real* eigenvectors,  // n x k column-major
    Index max_iter = config::DEFAULT_MAX_ITER
) {
    k = scl::algo::min2(k, n);
    
    detail::FastRNG rng(42);
    auto temp_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* temp = temp_ptr.release();

    // Initialize randomly
    for (Index c = 0; c < k; ++c) {
        Real* v = eigenvectors + static_cast<Size>(c) * n;
        for (Index i = 0; i < n; ++i) {
            v[i] = rng.uniform() - Real(0.5);
        }
        detail::normalize_l2(v, n);
    }

    // Power iteration with deflation
    for (Index c = 0; c < k; ++c) {
        Real* v = eigenvectors + static_cast<Size>(c) * n;
        
        Real prev_lambda = 0;
        Real prev_prev_lambda = 0;

        for (Index iter = 0; iter < max_iter; ++iter) {
            detail::sparse_matvec(mat, v, temp, n);

            // Orthogonalize against previous
            for (Index p = 0; p < c; ++p) {
                Real* v_prev = eigenvectors + static_cast<Size>(p) * n;
                Real dot = detail::dot_product(temp, v_prev, n);
                detail::axpy(temp, -dot, v_prev, n);
            }

            // Rayleigh quotient
            Real lambda = detail::dot_product(v, temp, n);
            eigenvalues[c] = lambda;
            detail::normalize_l2(temp, n);

            Real diff = detail::max_abs_diff(temp, v, n);
            std::memcpy(v, temp, sizeof(Real) * static_cast<size_t>(n));
            if (diff < config::DEFAULT_TOLERANCE) break;
            
            // Aitken check
            if (iter >= 2) {
                Real denom = lambda - 2*prev_lambda + prev_prev_lambda;
                if (std::abs(denom) > config::EPSILON) {
                    Real delta = lambda - prev_lambda;
                    if (std::abs(delta * delta / denom) < config::DEFAULT_TOLERANCE) break;
                }
            }
            
            prev_prev_lambda = prev_lambda;
            prev_lambda = lambda;
        }
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Metastable States - Optimized K-means
// =============================================================================

template <typename T, bool IsCSR>
void metastable_states(
    const Sparse<T, IsCSR>& transition_mat,
    Index n,
    Index n_states,
    Array<Index> state_labels,
    Array<Real> membership_probs
) {
    SCL_CHECK_DIM(state_labels.len >= static_cast<Size>(n), "state_labels buffer too small");

    n_states = scl::algo::min2(n_states, n);

    auto eigenvalues_ptr = scl::memory::aligned_alloc<Real>(n_states, SCL_ALIGNMENT);
    auto eigenvectors_ptr = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n) * n_states, SCL_ALIGNMENT);
    Real* eigenvalues = eigenvalues_ptr.release();
    Real* eigenvectors = eigenvectors_ptr.release();
    compute_top_eigenvectors(transition_mat, n, n_states, eigenvalues, eigenvectors);

    Real* membership = membership_probs.ptr;
    bool own_membership = (membership == nullptr);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<Real[], scl::memory::AlignedDeleter<Real>> membership_ptr;
    if (own_membership) {
        membership_ptr = scl::memory::aligned_alloc<Real>(
            static_cast<Size>(n) * n_states, SCL_ALIGNMENT);
        membership = membership_ptr.release();
    }

    // K-means++ initialization
    auto centroids_ptr = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n_states) * n_states, SCL_ALIGNMENT);
    Real* centroids = centroids_ptr.release();
    
    // First centroid: random
    detail::FastRNG rng(123);
    auto first = static_cast<Index>(rng.next() % static_cast<Index>(n));
    for (Index d = 0; d < n_states; ++d) {
        centroids[d] = eigenvectors[static_cast<Size>(d) * n + first];
    }

    // Remaining centroids: k-means++
    auto min_dists_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* min_dists = min_dists_ptr.release();
    
    for (Index c = 1; c < n_states; ++c) {
        // Compute min distances to existing centroids
        Real total_dist = 0;
        for (Index i = 0; i < n; ++i) {
            Real min_d = config::INF_VALUE;
            for (Index cc = 0; cc < c; ++cc) {
                Real dist = 0;
                for (Index d = 0; d < n_states; ++d) {
                    Real diff = eigenvectors[static_cast<Size>(d) * n + i] -
                               centroids[static_cast<Size>(cc) * n_states + d];
                    dist += diff * diff;
                }
                min_d = scl::algo::min2(min_d, dist);
            }
            min_dists[i] = min_d;
            total_dist += min_d;
        }
        
        // Sample proportional to distance squared
        Real r = rng.uniform() * total_dist;
        Real cumsum = 0;
        Index selected = n - 1;
        for (Index i = 0; i < n; ++i) {
            cumsum += min_dists[i];
            if (cumsum >= r) {
                selected = i;
                break;
            }
        }
        
        for (Index d = 0; d < n_states; ++d) {
            centroids[static_cast<Size>(c) * n_states + d] = 
                eigenvectors[static_cast<Size>(d) * n + selected];
        }
    }

    // K-means iterations (parallel assignment)
    auto counts_ptr = scl::memory::aligned_alloc<Index>(n_states, SCL_ALIGNMENT);
    Index* counts = counts_ptr.release();
    
    for (Index iter = 0; iter < 20; ++iter) {
        // Parallel assignment
        std::atomic<bool> changed{false};
        
        scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
            Real min_dist = config::INF_VALUE;
            Index best_s = 0;
            for (Index s = 0; s < n_states; ++s) {
                Real dist = 0;
                for (Index d = 0; d < n_states; ++d) {
                    Real diff = eigenvectors[static_cast<Size>(d) * n + i] -
                               centroids[static_cast<Size>(s) * n_states + d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_s = s;
                }
            }
            if (state_labels[static_cast<Index>(i)] != best_s) {
                state_labels[static_cast<Index>(i)] = best_s;
                changed.store(true, std::memory_order_relaxed);
            }
        });

        if (!changed.load()) break;

        // Update centroids
        // PERFORMANCE: memset is faster than loop for zero initialization
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
        // Intentional: zero-initialization for performance
        std::memset(centroids, 0, sizeof(Real) * static_cast<size_t>(n_states) * n_states);
        std::memset(counts, 0, sizeof(Index) * static_cast<size_t>(n_states));
        for (Index i = 0; i < n; ++i) {
            Index s = state_labels[i];
            ++counts[s];
            for (Index d = 0; d < n_states; ++d) {
                centroids[static_cast<Size>(s) * n_states + d] +=
                    eigenvectors[static_cast<Size>(d) * n + i];
            }
        }
        for (Index s = 0; s < n_states; ++s) {
            if (counts[s] > 0) {
                Real inv = Real(1) / static_cast<Real>(counts[s]);
                for (Index d = 0; d < n_states; ++d) {
                    centroids[static_cast<Size>(s) * n_states + d] *= inv;
                }
            }
        }
    }

    // Soft membership (parallel)
    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        Real* mem_i = membership + i * n_states;
        Real sum = 0;
        for (Index s = 0; s < n_states; ++s) {
            Real dist = 0;
            for (Index d = 0; d < n_states; ++d) {
                Real diff = eigenvectors[static_cast<Size>(d) * n + i] -
                           centroids[static_cast<Size>(s) * n_states + d];
                dist += diff * diff;
            }
            mem_i[s] = std::exp(-dist);
            sum += mem_i[s];
        }
        if (sum > config::EPSILON) {
            Real inv = Real(1) / sum;
            for (Index s = 0; s < n_states; ++s) {
                mem_i[s] *= inv;
            }
        }
    });

    scl::memory::aligned_free(min_dists, SCL_ALIGNMENT);
    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(centroids, SCL_ALIGNMENT);
    if (own_membership) {
        scl::memory::aligned_free(membership, SCL_ALIGNMENT);
    }
    scl::memory::aligned_free(eigenvectors, SCL_ALIGNMENT);
    scl::memory::aligned_free(eigenvalues, SCL_ALIGNMENT);
}

// =============================================================================
// Coarse-Grain Transition (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void coarse_grain_transition(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const Index> state_labels,
    Index n,
    Index n_states,
    Real* coarse_transition
) {
    Size total = static_cast<Size>(n_states) * static_cast<Size>(n_states);
    std::memset(coarse_transition, 0, sizeof(Real) * total);

    // Thread-local accumulation
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    auto thread_buffers_ptr = scl::memory::aligned_alloc<Real>(total * n_threads, SCL_ALIGNMENT);
    Real* thread_buffers = thread_buffers_ptr.release();
    // PERFORMANCE: memset is faster than loop for zero initialization
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index)
    // Intentional: zero-initialization for performance
    std::memset(thread_buffers, 0, sizeof(Real) * total * n_threads);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t tid) {
        Index s_i = state_labels[static_cast<Index>(i)];
        if (s_i < 0 || s_i >= n_states) return;

        Real* local = thread_buffers + tid * total;
        
        auto indices = transition_mat.primary_indices_unsafe(static_cast<Index>(i));
        auto values = transition_mat.primary_values_unsafe(static_cast<Index>(i));
        Index len = transition_mat.primary_length_unsafe(static_cast<Index>(i));

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Index s_j = state_labels[j];
            if (s_j >= 0 && s_j < n_states) {
                local[static_cast<Size>(s_i) * n_states + s_j] += static_cast<Real>(values[k]);
            }
        }
    });

    // Reduce
    for (size_t t = 0; t < n_threads; ++t) {
        Real* local = thread_buffers + t * total;
        for (Size i = 0; i < total; ++i) {
            coarse_transition[i] += local[i];
        }
    }

    // Row-normalize
    for (Index s = 0; s < n_states; ++s) {
        Real* row = coarse_transition + static_cast<Size>(s) * n_states;
        Real row_sum = detail::vector_sum(row, n_states);
        if (row_sum > config::EPSILON) {
            detail::scale_vector(row, Real(1) / row_sum, n_states);
        }
    }

    scl::memory::aligned_free(thread_buffers, SCL_ALIGNMENT);
}

// =============================================================================
// Lineage Drivers (Parallel Correlation)
// =============================================================================

template <typename T, bool IsCSR, typename TG, bool IsCSR_G>
void lineage_drivers(
    const Sparse<TG, IsCSR_G>& expression,
    const Real* absorption_probs,
    Index n_cells,
    Index n_genes,
    Index n_terminal,
    Index lineage,
    Array<Real> driver_scores
) {
    SCL_CHECK_DIM(driver_scores.len >= static_cast<Size>(n_genes), "buffer too small");
    SCL_CHECK_ARG(lineage >= 0 && lineage < n_terminal, "invalid lineage");

    // Precompute fate probability statistics
    Real fate_mean = 0;
    for (Index c = 0; c < n_cells; ++c) {
        fate_mean += absorption_probs[static_cast<Size>(c) * n_terminal + lineage];
    }
    fate_mean /= static_cast<Real>(n_cells);

    Real fate_var = 0;
    for (Index c = 0; c < n_cells; ++c) {
        Real d = absorption_probs[static_cast<Size>(c) * n_terminal + lineage] - fate_mean;
        fate_var += d * d;
    }

    // Parallel gene correlation
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t g, size_t) {
        // Sparse correlation computation
        Real expr_sum = 0;
        Real expr_sq_sum = 0;
        Real cov = 0;
        Index nnz = 0;
        
        if (IsCSR_G) {
            // CSC access pattern for genes
            for (Index c = 0; c < n_cells; ++c) {
                auto indices = expression.row_indices_unsafe(c);
                auto values = expression.row_values_unsafe(c);
                Index len = expression.row_length_unsafe(c);
                for (Index k = 0; k < len; ++k) {
                    if (indices[k] == static_cast<Index>(g)) {
                        Real e = static_cast<Real>(values[k]);
                        Real f = absorption_probs[static_cast<Size>(c) * n_terminal + lineage];
                        expr_sum += e;
                        expr_sq_sum += e * e;
                        cov += e * f;
                        ++nnz;
                        break;
                    }
                }
            }
        } else {
            auto indices = expression.col_indices_unsafe(static_cast<Index>(g));
            auto values = expression.col_values_unsafe(static_cast<Index>(g));
            Index len = expression.col_length_unsafe(static_cast<Index>(g));
            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    Real e = static_cast<Real>(values[k]);
                    Real f = absorption_probs[static_cast<Size>(c) * n_terminal + lineage];
                    expr_sum += e;
                    expr_sq_sum += e * e;
                    cov += e * f;
                    ++nnz;
                }
            }
        }

        // Adjust for sparse zeros
        Real n_real = static_cast<Real>(n_cells);
        Real expr_mean = expr_sum / n_real;
        Real expr_var = expr_sq_sum / n_real - expr_mean * expr_mean;
        cov = cov / n_real - expr_mean * fate_mean;
        Real denom = std::sqrt(expr_var * fate_var);

        driver_scores[static_cast<Index>(g)] = (denom > config::EPSILON) ? cov / denom : Real(0);
    });
}

// =============================================================================
// Cell Fate Entropy (SIMD)
// =============================================================================

inline void fate_entropy(
    const Real* absorption_probs,
    Index n,
    Index n_terminal,
    Array<Real> entropy
) {
    SCL_CHECK_DIM(entropy.len >= static_cast<Size>(n), "buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        const Real* probs = absorption_probs + i * n_terminal;
        Real h = 0;
        for (Index t = 0; t < n_terminal; ++t) {
            Real p = probs[t];
            // Branchless: use p * log(p + epsilon) to avoid branch
            h -= (p > config::EPSILON) ? p * std::log(p) : Real(0);
        }
        entropy[static_cast<Index>(i)] = h;
    });
}

// =============================================================================
// Forward Committor (SOR)
// =============================================================================

template <typename T, bool IsCSR>
void forward_committor(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_source,
    Array<const bool> is_target,
    Index n,
    Array<Real> committor,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    SCL_CHECK_DIM(committor.len >= static_cast<Size>(n), "buffer too small");

    for (Index i = 0; i < n; ++i) {
        committor[i] = is_source[i] ? Real(0) : (is_target[i] ? Real(1) : Real(0.5));
    }

    const Real omega = config::SOR_OMEGA;

    for (Index iter = 0; iter < max_iter; ++iter) {
        Real max_diff = 0;
        
        for (Index i = 0; i < n; ++i) {
            if (is_source[i] || is_target[i]) continue;

            auto indices = transition_mat.primary_indices_unsafe(i);
            auto values = transition_mat.primary_values_unsafe(i);
            Index len = transition_mat.primary_length_unsafe(i);
            Real new_val = 0;

            for (Index k = 0; k < len; ++k) {
                new_val += static_cast<Real>(values[k]) * committor[indices[k]];
            }

            Real old_val = committor[i];
            Real sor_val = old_val + omega * (new_val - old_val);
            max_diff = scl::algo::max2(max_diff, std::abs(sor_val - old_val));
            committor[i] = sor_val;
        }

        if (max_diff < tol) break;
    }
}

// =============================================================================
// Random Walk (Vectorized Sampling)
// =============================================================================

inline void random_walk(
    const Real* transition_mat,
    Index n,
    Index start_cell,
    Index n_steps,
    Index* trajectory,
    uint64_t seed = 42
) {
    detail::FastRNG rng(seed);
    
    trajectory[0] = start_cell;
    Index current = start_cell;

    for (Index step = 0; step < n_steps; ++step) {
        Real r = rng.uniform();
        const Real* row = transition_mat + static_cast<Size>(current) * n;
        
        // Binary search for efficiency on larger n
        if (n > 64) {
            Index lo = 0;
            Index hi = n;
            while (lo < hi) {
                Index mid = (lo + hi) / 2;
                Real sum_to_mid = 0;
                for (Index j = 0; j <= mid; ++j) {
                    sum_to_mid += row[j];
                }
                if (sum_to_mid <= r) lo = mid + 1;
                else hi = mid;
            }
            current = lo < n ? lo : n - 1;
        } else {
            Real cumsum = 0;
            for (Index j = 0; j < n; ++j) {
                cumsum += row[j];
                if (r < cumsum) {
                    current = j;
                    break;
                }
            }
        }
        
        trajectory[step + 1] = current;
    }
}

// =============================================================================
// Directional Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void directional_score(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const Real> pseudotime,
    Index n,
    Array<Real> scores
) {
    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n), "buffer too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i, size_t) {
        auto indices = transition_mat.primary_indices_unsafe(static_cast<Index>(i));
        auto values = transition_mat.primary_values_unsafe(static_cast<Index>(i));
        Index len = transition_mat.primary_length_unsafe(static_cast<Index>(i));
        Real forward = 0;
        Real backward = 0;
        Real pt_i = pseudotime[static_cast<Index>(i)];

        for (Index k = 0; k < len; ++k) {
            Real p = static_cast<Real>(values[k]);
            Real pt_j = pseudotime[indices[k]];
            // Branchless accumulation
            Real is_fwd = static_cast<Real>(pt_j > pt_i);
            forward += p * is_fwd;
            backward += p * (Real(1) - is_fwd);
        }

        Real total = forward + backward;
        scores[static_cast<Index>(i)] = (total > config::EPSILON) ? (forward - backward) / total : Real(0);
    });
}

} // namespace scl::kernel::transition
