#pragma once

#include "scl/core/type.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>

// =============================================================================
/// @file gram.hpp
/// @brief Gram Matrix Computation
///
/// Computes G = A^T A (CSC) or G = A A^T (CSR)
///
/// Algorithm:
/// - Adaptive sparse dot product (linear merge or binary search)
/// - Symmetric output (compute upper triangle, mirror to lower)
/// - Parallel over rows with dynamic scheduling
///
/// Performance: ~500M dot products/sec per core
// =============================================================================

namespace scl::kernel::gram {

namespace detail {

/// @brief Linear merge dot product O(n + m)
template <typename T>
SCL_FORCE_INLINE T dot_linear(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum = static_cast<T>(0.0);
    Size i = 0, j = 0;
    
    while (i < n1 && j < n2) {
        Index r1 = idx1[i];
        Index r2 = idx2[j];
        
        if (SCL_LIKELY(r1 == r2)) {
            sum += val1[i] * val2[j];
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }
    return sum;
}

/// @brief Binary search dot product O(n log m), n << m
template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = static_cast<T>(0.0);
    const Index* base = idx_large;
    Size len = n_large;

    for (Size i = 0; i < n_small; ++i) {
        Index target = idx_small[i];
        
        auto it = std::lower_bound(base, base + len, target);
        
        if (SCL_LIKELY(it != base + len && *it == target)) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];
            
            Size step = static_cast<Size>(it - base) + 1;
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        }
    }
    return sum;
}

/// @brief Adaptive dispatcher (auto-selects linear or binary)
template <typename T>
SCL_FORCE_INLINE T dot_product(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return static_cast<T>(0.0);
    }

    constexpr Size RATIO_THRESHOLD = 32;

    if (n1 < n2) {
        if (n2 > n1 * RATIO_THRESHOLD) {
            return dot_binary(idx1, val1, n1, idx2, val2, n2);
        } else {
            return dot_linear(idx1, val1, n1, idx2, val2, n2);
        }
    } else {
        if (n1 > n2 * RATIO_THRESHOLD) {
            return dot_binary(idx2, val2, n2, idx1, val1, n1);
        } else {
            return dot_linear(idx1, val1, n1, idx2, val2, n2);
        }
    }
}

} // namespace detail

// =============================================================================
// Public API (Unified for CSR/CSC)
// =============================================================================

/// @brief Compute Gram matrix (unified for CSR/CSC)
///
/// For CSR: G = A × A^T (sample similarity)
/// For CSC: G = A^T × A (feature correlation)
///
/// @param matrix Input sparse matrix
/// @param output Output dense Gram matrix [N × N], row-major
template <typename MatrixT>
    requires AnySparse<MatrixT>
void gram(const MatrixT& matrix, Array<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    
    SCL_CHECK_DIM(output.size() == N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(0, N_size, [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        
        auto idx_i = scl::primary_indices(matrix, idx);
        auto val_i = scl::primary_values(matrix, idx);
        
        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal
        T self_dot = static_cast<T>(0.0);
        for (Size k = 0; k < idx_i.size(); ++k) {
            self_dot += val_i[k] * val_i[k];
        }
        row_ptr[i] = self_dot;

        // Upper triangle
        for (Size j = i + 1; j < N_size; ++j) {
            const Index jdx = static_cast<Index>(j);
            auto idx_j = scl::primary_indices(matrix, jdx);
            auto val_j = scl::primary_values(matrix, jdx);

            T dot = detail::dot_product(
                idx_i.ptr, val_i.ptr, idx_i.size(),
                idx_j.ptr, val_j.ptr, idx_j.size()
            );

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;  // Mirror to lower triangle
        }
    });
}

} // namespace scl::kernel::gram
