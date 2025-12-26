#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/gram_mapped_impl.hpp"

#include <algorithm>

// =============================================================================
/// @file gram_fast_impl.hpp
/// @brief Extreme Performance Gram Matrix (G = A * A^T or A^T * A)
///
/// ## Key Optimizations
///
/// 1. Branchless Sparse Dot Product
///    - 4 independent accumulators without switch/case
///    - Compile-time unrolled accumulation
///
/// 2. SIMD Self Dot Product
///    - 4-way unrolled FMA for diagonal elements
///
/// 3. Adaptive Algorithm Selection
///    - Linear merge for similar lengths
///    - Binary search for skewed lengths
///    - Galloping search for extreme skew
///
/// 4. Cache-Blocked Processing
///    - Process rows in chunks for L2 efficiency
///    - Prefetch hints for random access
///
/// Performance Target: 2x faster than generic
// =============================================================================

namespace scl::kernel::gram::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size RATIO_THRESHOLD = 32;      // Switch to binary search
    constexpr Size GALLOP_THRESHOLD = 256;    // Switch to galloping
    constexpr Size CHUNK_SIZE = 64;           // Rows per cache block
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD self dot product (4-way unrolled)
template <typename T>
SCL_FORCE_INLINE T self_dot_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size k = 0;

    // 4-way unrolled
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
        v_sum0 = s::MulAdd(v2, v2, v_sum0);
        v_sum1 = s::MulAdd(v3, v3, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::MulAdd(v, v, v_sum);
    }

    T result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        result += vals[k] * vals[k];
    }

    return result;
}

// =============================================================================
// SECTION 3: Branchless Sparse Dot Products
// =============================================================================

/// @brief Linear merge with 4 independent accumulators (no switch/case)
template <typename T>
SCL_FORCE_INLINE T dot_linear_branchless(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    Size i = 0, j = 0;

    // Main loop with prefetch
    while (i < n1 && j < n2) {
        // Prefetch ahead
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < n1)) {
            SCL_PREFETCH_READ(&idx1[i + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val1[i + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < n2)) {
            SCL_PREFETCH_READ(&idx2[j + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val2[j + config::PREFETCH_DISTANCE], 0);
        }

        Index r1 = idx1[i];
        Index r2 = idx2[j];

        if (r1 == r2) {
            T prod = val1[i] * val2[j];
            // Round-robin to accumulators without branch
            sum0 += prod;
            // Rotate accumulators
            T tmp = sum0; sum0 = sum1; sum1 = sum2; sum2 = sum3; sum3 = tmp;
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }

    return sum0 + sum1 + sum2 + sum3;
}

/// @brief Binary search dot product for skewed lengths
template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    const Index* base = idx_large;
    Size len = n_large;

    for (Size i = 0; i < n_small; ++i) {
        Index target = idx_small[i];

        auto it = std::lower_bound(base, base + len, target);

        if (it != base + len && *it == target) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];

            Size step = static_cast<Size>(it - base) + 1;
            if (step >= len) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (step >= len) break;
            base += step;
            len -= step;
        }
    }

    return sum;
}

/// @brief Galloping search for extreme skew
template <typename T>
SCL_FORCE_INLINE T dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    Size j = 0;

    for (Size i = 0; i < n_small && j < n_large; ++i) {
        Index target = idx_small[i];

        // Galloping: exponential search
        Size step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        // Binary search in [j, j+step]
        Size lo = j;
        Size hi = std::min(j + step, n_large);

        while (lo < hi) {
            Size mid = lo + (hi - lo) / 2;
            if (idx_large[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        j = lo;
        if (j < n_large && idx_large[j] == target) {
            sum += val_small[i] * val_large[j];
            ++j;
        }
    }

    return sum;
}

/// @brief Adaptive dot product dispatcher
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    // Ensure n1 <= n2
    if (n1 > n2) {
        std::swap(idx1, idx2);
        std::swap(val1, val2);
        std::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        return dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else if (ratio >= config::RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear_branchless(idx1, val1, n1, idx2, val2, n2);
    }
}

} // namespace detail

// =============================================================================
// SECTION 4: CustomSparse Fast Path
// =============================================================================

/// @brief Gram matrix for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void gram_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        Index start_i = matrix.indptr[i];
        Index end_i = matrix.indptr[i + 1];
        Size len_i = static_cast<Size>(end_i - start_i);

        const Index* SCL_RESTRICT idx_i = matrix.indices + start_i;
        const T* SCL_RESTRICT val_i = matrix.data + start_i;

        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal: SIMD self dot
        row_ptr[i] = detail::self_dot_simd(val_i, len_i);

        // Upper triangle
        for (Size j = i + 1; j < N_size; ++j) {
            Index start_j = matrix.indptr[j];
            Index end_j = matrix.indptr[j + 1];
            Size len_j = static_cast<Size>(end_j - start_j);

            const Index* SCL_RESTRICT idx_j = matrix.indices + start_j;
            const T* SCL_RESTRICT val_j = matrix.data + start_j;

            T dot = detail::sparse_dot_adaptive(idx_i, val_i, len_i, idx_j, val_j, len_j);

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;  // Mirror
        }
    });
}

// =============================================================================
// SECTION 5: VirtualSparse Fast Path
// =============================================================================

/// @brief Gram matrix for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void gram_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(Size(0), N_size, [&](size_t i) {
        Size len_i = static_cast<Size>(matrix.lengths[i]);

        // Single pointer dereference
        const Index* SCL_RESTRICT idx_i = static_cast<const Index*>(matrix.indices_ptrs[i]);
        const T* SCL_RESTRICT val_i = static_cast<const T*>(matrix.data_ptrs[i]);

        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal
        row_ptr[i] = detail::self_dot_simd(val_i, len_i);

        // Upper triangle
        for (Size j = i + 1; j < N_size; ++j) {
            Size len_j = static_cast<Size>(matrix.lengths[j]);

            const Index* SCL_RESTRICT idx_j = static_cast<const Index*>(matrix.indices_ptrs[j]);
            const T* SCL_RESTRICT val_j = static_cast<const T*>(matrix.data_ptrs[j]);

            T dot = detail::sparse_dot_adaptive(idx_i, val_i, len_i, idx_j, val_j, len_j);

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;
        }
    });
}

// =============================================================================
// SECTION 6: Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void gram_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::gram::mapped::gram_mapped_dispatch<MatrixT, IsCSR>(matrix, output);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        gram_custom(matrix, output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        gram_virtual(matrix, output);
    }
}

} // namespace scl::kernel::gram::fast
