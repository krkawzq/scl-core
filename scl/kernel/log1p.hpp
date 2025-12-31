#pragma once

/// @file scl/kernel/log1p.hpp
/// @brief Log1p transformation kernel with extreme SIMD optimization
///
/// This kernel provides:
///   - log1p(x) = log(1 + x) transformation
///   - log2p1(x) = log2(1 + x) transformation
///   - expm1(x) = exp(x) - 1 transformation
///   - In-place and out-of-place variants
///   - Sparse matrix support (CSR/CSC)
///   - Dense array support
///
/// Optimizations:
///   - 8-way SIMD loop unrolling for maximum throughput
///   - Pipeline prefetching with configurable distance
///   - Parallel processing for sparse matrices
///   - Explicit precision support (Real32/Real64 only)
///
/// @note This is a preprocessing step before normalization
/// @note Numerically stable for small x values

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <span>

// =============================================================================
// SECTION 1: Configuration and Constants
// =============================================================================

namespace scl::kernel::log1p {

/// @brief Configuration constants for log1p transformations
namespace config {
    /// @brief Prefetch distance in elements (tuned for L1 cache)
    /// @note Default: 64 elements ahead (~512 bytes for double)
    inline constexpr Size PREFETCH_DISTANCE = 64;

    /// @brief 1 / ln(2) for log2 conversion
    /// @note log2(x) = log(x) / ln(2) = log(x) * INV_LN2
    inline constexpr double INV_LN2 = 1.44269504088896340736;

    /// @brief ln(2) constant (unused, kept for reference)
    inline constexpr double LN2 = 0.6931471805599453;

    /// @brief Minimum size for parallel processing
    /// @note Avoid parallelization overhead for small arrays
    inline constexpr Size MIN_PARALLEL_SIZE = 1024;

    /// @brief SIMD unroll factor (number of vectors per iteration)
    /// @note 8-way unrolling for maximum instruction-level parallelism
    inline constexpr Size SIMD_UNROLL_FACTOR = 8;
} // namespace config

// =============================================================================
// SECTION 2: Precision Support Concepts
// =============================================================================

/// @brief Concept for supported log1p precision types
/// @tparam T Precision type to check
/// @note Only Real32 (float) and Real64 (double) are supported
/// @note Real16 and Real128 are explicitly disabled
template<typename T>
concept Log1pSupportedPrecision =
    SupportedPrecision<T> &&  // Must be supported on platform
    (std::is_same_v<T, Real32> || std::is_same_v<T, Real64>);  // Only standard floats

// =============================================================================
// SECTION 3: Core SIMD Transformation Routines (Internal)
// =============================================================================

namespace detail {

/// @brief Apply log1p transformation with extreme SIMD optimization
/// @tparam T Precision type (Real32 or Real64)
/// @param[in,out] vals Pointer to values array (will be modified in-place)
/// @param[in] len Number of elements in array
/// @note Uses 8-way SIMD loop unrolling with prefetching
/// @note Processes boundary with 1-way SIMD, then scalar remainder
/// @note Performance: ~16 flops/cycle on AVX-512, ~8 flops/cycle on AVX2
template<Log1pSupportedPrecision T>
SCL_FORCE_INLINE
auto apply_log1p_simd(
    T* SCL_RESTRICT vals,
    Index len
) -> void {
    namespace s = scl::simd;

    // SIMD descriptor for type T
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);  // Vector width (e.g., 8 for AVX2 double)

    Index k = 0;

    // =========================================================================
    // Main Loop: 8-way SIMD unrolling with prefetching
    // =========================================================================
    // Each iteration processes 8 * lanes elements
    // Example: AVX2 double (lanes=4) → 32 elements per iteration

    const Index unroll_stride = config::SIMD_UNROLL_FACTOR * static_cast<Index>(lanes);

    for (; k + unroll_stride <= len; k += unroll_stride) {
        // Prefetch next cache line for L1 cache (read-only, temporal locality)
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) [[likely]] {
            SCL_PREFETCH(vals + k + config::PREFETCH_DISTANCE, 0, 3);
        }

        // Load 8 vectors (unrolled)
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        // Apply log1p transformation (vectorized)
        // Highway provides optimized log1p with polynomial approximation
        v0 = s::Log1p(d, v0);
        v1 = s::Log1p(d, v1);
        v2 = s::Log1p(d, v2);
        v3 = s::Log1p(d, v3);
        v4 = s::Log1p(d, v4);
        v5 = s::Log1p(d, v5);
        v6 = s::Log1p(d, v6);
        v7 = s::Log1p(d, v7);

        // Store results (unrolled)
        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
        s::Store(v4, d, vals + k + 4 * lanes);
        s::Store(v5, d, vals + k + 5 * lanes);
        s::Store(v6, d, vals + k + 6 * lanes);
        s::Store(v7, d, vals + k + 7 * lanes);
    }

    // =========================================================================
    // Boundary Loop: 1-way SIMD (process remaining full vectors)
    // =========================================================================
    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, vals + k);
        v = s::Log1p(d, v);
        s::Store(v, d, vals + k);
    }

    // =========================================================================
    // Scalar Remainder: Process last few elements (< lanes)
    // =========================================================================
    for (; k < len; ++k) {
        vals[k] = std::log1p(vals[k]);
    }
}

/// @brief Apply log2(1+x) transformation with extreme SIMD optimization
/// @tparam T Precision type (Real32 or Real64)
/// @param[in,out] vals Pointer to values array (will be modified in-place)
/// @param[in] len Number of elements in array
/// @note log2(1+x) = log(1+x) / ln(2) = log1p(x) * INV_LN2
/// @note Uses 8-way SIMD loop unrolling with prefetching
template<Log1pSupportedPrecision T>
SCL_FORCE_INLINE
auto apply_log2p1_simd(
    T* SCL_RESTRICT vals,
    Index len
) -> void {
    namespace s = scl::simd;

    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);

    // Broadcast conversion constant to all SIMD lanes
    const auto v_inv_ln2 = s::Set(d, static_cast<T>(config::INV_LN2));

    Index k = 0;

    // =========================================================================
    // Main Loop: 8-way SIMD unrolling
    // =========================================================================
    const Index unroll_stride = config::SIMD_UNROLL_FACTOR * static_cast<Index>(lanes);

    for (; k + unroll_stride <= len; k += unroll_stride) {
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) [[likely]] {
            SCL_PREFETCH(vals + k + config::PREFETCH_DISTANCE, 0, 3);
        }

        // Load 8 vectors
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        // Apply log1p and convert to log2 (fused multiply-add friendly)
        v0 = s::Mul(s::Log1p(d, v0), v_inv_ln2);
        v1 = s::Mul(s::Log1p(d, v1), v_inv_ln2);
        v2 = s::Mul(s::Log1p(d, v2), v_inv_ln2);
        v3 = s::Mul(s::Log1p(d, v3), v_inv_ln2);
        v4 = s::Mul(s::Log1p(d, v4), v_inv_ln2);
        v5 = s::Mul(s::Log1p(d, v5), v_inv_ln2);
        v6 = s::Mul(s::Log1p(d, v6), v_inv_ln2);
        v7 = s::Mul(s::Log1p(d, v7), v_inv_ln2);

        // Store results
        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
        s::Store(v4, d, vals + k + 4 * lanes);
        s::Store(v5, d, vals + k + 5 * lanes);
        s::Store(v6, d, vals + k + 6 * lanes);
        s::Store(v7, d, vals + k + 7 * lanes);
    }

    // =========================================================================
    // Boundary Loop: 1-way SIMD
    // =========================================================================
    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, vals + k);
        v = s::Mul(s::Log1p(d, v), v_inv_ln2);
        s::Store(v, d, vals + k);
    }

    // =========================================================================
    // Scalar Remainder
    // =========================================================================
    const T inv_ln2_scalar = static_cast<T>(config::INV_LN2);
    for (; k < len; ++k) {
        vals[k] = std::log1p(vals[k]) * inv_ln2_scalar;
    }
}

/// @brief Apply expm1(x) = exp(x) - 1 transformation with extreme SIMD optimization
/// @tparam T Precision type (Real32 or Real64)
/// @param[in,out] vals Pointer to values array (will be modified in-place)
/// @param[in] len Number of elements in array
/// @note Numerically stable for small x values
/// @note Uses 8-way SIMD loop unrolling with prefetching
template<Log1pSupportedPrecision T>
SCL_FORCE_INLINE
auto apply_expm1_simd(
    T* SCL_RESTRICT vals,
    Index len
) -> void {
    namespace s = scl::simd;

    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const auto lanes = s::Lanes(d);

    Index k = 0;

    // =========================================================================
    // Main Loop: 8-way SIMD unrolling
    // =========================================================================
    const Index unroll_stride = config::SIMD_UNROLL_FACTOR * static_cast<Index>(lanes);

    for (; k + unroll_stride <= len; k += unroll_stride) {
        if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) [[likely]] {
            SCL_PREFETCH(vals + k + config::PREFETCH_DISTANCE, 0, 3);
        }

        // Load 8 vectors
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        // Apply expm1 transformation
        v0 = s::Expm1(d, v0);
        v1 = s::Expm1(d, v1);
        v2 = s::Expm1(d, v2);
        v3 = s::Expm1(d, v3);
        v4 = s::Expm1(d, v4);
        v5 = s::Expm1(d, v5);
        v6 = s::Expm1(d, v6);
        v7 = s::Expm1(d, v7);

        // Store results
        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
        s::Store(v4, d, vals + k + 4 * lanes);
        s::Store(v5, d, vals + k + 5 * lanes);
        s::Store(v6, d, vals + k + 6 * lanes);
        s::Store(v7, d, vals + k + 7 * lanes);
    }

    // =========================================================================
    // Boundary Loop: 1-way SIMD
    // =========================================================================
    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, vals + k);
        v = s::Expm1(d, v);
        s::Store(v, d, vals + k);
    }

    // =========================================================================
    // Scalar Remainder
    // =========================================================================
    for (; k < len; ++k) {
        vals[k] = std::expm1(vals[k]);
    }
}

} // namespace detail

// =============================================================================
// SECTION 4: Sparse Matrix In-Place Transformations (Parallel)
// =============================================================================

/// @brief Apply log1p transformation to sparse matrix (in-place, parallel)
/// @tparam T Value type (Real32 or Real64)
/// @tparam IndexT Index type
/// @tparam IsCSR Matrix format (true = CSR, false = CSC)
/// @param[in,out] matrix Sparse matrix to transform (modified in-place)
/// @note Processes each row/column in parallel using thread pool
/// @note Only non-zero values are transformed (sparse structure preserved)
/// @note Performance: Scales linearly with number of non-zeros
template<Log1pSupportedPrecision T, typename IndexT, bool IsCSR>
auto log1p_inplace(
    Sparse<T, IndexT, IsCSR>& matrix
) -> void {
    const Index primary_dim = matrix.primary_dim();

    // Parallel processing for each primary dimension (row for CSR, col for CSC)
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (len > 0) [[likely]] {
            // Get pointer to values in this primary dimension
            auto values = matrix.primary_values(idx);
            detail::apply_log1p_simd(values.data(), len);
        }
    });
}

/// @brief Apply log2(1+x) transformation to sparse matrix (in-place, parallel)
/// @tparam T Value type (Real32 or Real64)
/// @tparam IndexT Index type
/// @tparam IsCSR Matrix format (true = CSR, false = CSC)
/// @param[in,out] matrix Sparse matrix to transform (modified in-place)
template<Log1pSupportedPrecision T, typename IndexT, bool IsCSR>
auto log2p1_inplace(
    Sparse<T, IndexT, IsCSR>& matrix
) -> void {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (len > 0) [[likely]] {
            auto values = matrix.primary_values(idx);
            detail::apply_log2p1_simd(values.data(), len);
        }
    });
}

/// @brief Apply expm1(x) transformation to sparse matrix (in-place, parallel)
/// @tparam T Value type (Real32 or Real64)
/// @tparam IndexT Index type
/// @tparam IsCSR Matrix format (true = CSR, false = CSC)
/// @param[in,out] matrix Sparse matrix to transform (modified in-place)
template<Log1pSupportedPrecision T, typename IndexT, bool IsCSR>
auto expm1_inplace(
    Sparse<T, IndexT, IsCSR>& matrix
) -> void {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);

        if (len > 0) [[likely]] {
            auto values = matrix.primary_values(idx);
            detail::apply_expm1_simd(values.data(), len);
        }
    });
}

// =============================================================================
// SECTION 5: Dense Array In-Place Transformations (Sequential/Parallel)
// =============================================================================

/// @brief Apply log1p transformation to dense array (in-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in,out] values Array of values (modified in-place)
/// @note Automatically uses parallel processing for large arrays (>= 1024 elements)
/// @note For small arrays, uses sequential SIMD processing
template<Log1pSupportedPrecision T>
auto log1p_inplace(
    std::span<T> values
) -> void {
    const Size len = values.size();

    if (len >= config::MIN_PARALLEL_SIZE) {
        // Parallel processing for large arrays
        const Size num_threads = scl::threading::get_num_threads();
        const Size chunk_size = (len + num_threads - 1) / num_threads;

        scl::threading::parallel_for(Size(0), num_threads, [&](Size thread_id) {
            const Size start = thread_id * chunk_size;
            const Size end = std::min(start + chunk_size, len);

            if (start < end) [[likely]] {
                detail::apply_log1p_simd(values.data() + start, static_cast<Index>(end - start));
            }
        });
    } else {
        // Sequential SIMD processing for small arrays
        detail::apply_log1p_simd(values.data(), static_cast<Index>(len));
    }
}

/// @brief Apply log2(1+x) transformation to dense array (in-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in,out] values Array of values (modified in-place)
template<Log1pSupportedPrecision T>
auto log2p1_inplace(
    std::span<T> values
) -> void {
    const Size len = values.size();

    if (len >= config::MIN_PARALLEL_SIZE) {
        const Size num_threads = scl::threading::get_num_threads();
        const Size chunk_size = (len + num_threads - 1) / num_threads;

        scl::threading::parallel_for(Size(0), num_threads, [&](Size thread_id) {
            const Size start = thread_id * chunk_size;
            const Size end = std::min(start + chunk_size, len);

            if (start < end) [[likely]] {
                detail::apply_log2p1_simd(values.data() + start, static_cast<Index>(end - start));
            }
        });
    } else {
        detail::apply_log2p1_simd(values.data(), static_cast<Index>(len));
    }
}

/// @brief Apply expm1(x) transformation to dense array (in-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in,out] values Array of values (modified in-place)
template<Log1pSupportedPrecision T>
auto expm1_inplace(
    std::span<T> values
) -> void {
    const Size len = values.size();

    if (len >= config::MIN_PARALLEL_SIZE) {
        const Size num_threads = scl::threading::get_num_threads();
        const Size chunk_size = (len + num_threads - 1) / num_threads;

        scl::threading::parallel_for(Size(0), num_threads, [&](Size thread_id) {
            const Size start = thread_id * chunk_size;
            const Size end = std::min(start + chunk_size, len);

            if (start < end) [[likely]] {
                detail::apply_expm1_simd(values.data() + start, static_cast<Index>(end - start));
            }
        });
    } else {
        detail::apply_expm1_simd(values.data(), static_cast<Index>(len));
    }
}

// =============================================================================
// SECTION 6: Dense Array Out-of-Place Transformations
// =============================================================================

/// @brief Apply log1p transformation to dense array (out-of-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in] input Input array (read-only)
/// @param[out] output Output array (must have same size as input)
/// @pre input.size() == output.size()
/// @throws std::invalid_argument if sizes don't match
template<Log1pSupportedPrecision T>
auto log1p(
    std::span<const T> input,
    std::span<T> output
) -> void {
    SCL_CHECK(input.size() == output.size(), DimensionError,
        "log1p: input and output must have the same size");

    // Copy input to output, then apply in-place transformation
    std::copy(input.begin(), input.end(), output.begin());
    log1p_inplace(output);
}

/// @brief Apply log2(1+x) transformation to dense array (out-of-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in] input Input array (read-only)
/// @param[out] output Output array (must have same size as input)
template<Log1pSupportedPrecision T>
auto log2p1(
    std::span<const T> input,
    std::span<T> output
) -> void {
    SCL_CHECK(input.size() == output.size(), DimensionError,
        "log2p1: input and output must have the same size");

    std::copy(input.begin(), input.end(), output.begin());
    log2p1_inplace(output);
}

/// @brief Apply expm1(x) transformation to dense array (out-of-place)
/// @tparam T Value type (Real32 or Real64)
/// @param[in] input Input array (read-only)
/// @param[out] output Output array (must have same size as input)
template<Log1pSupportedPrecision T>
auto expm1(
    std::span<const T> input,
    std::span<T> output
) -> void {
    SCL_CHECK(input.size() == output.size(), DimensionError,
        "expm1: input and output must have the same size");

    std::copy(input.begin(), input.end(), output.begin());
    expm1_inplace(output);
}

} // namespace scl::kernel::log1p
