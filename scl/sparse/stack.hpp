#pragma once

/// @file scl/sparse/stack.hpp
/// @brief High-Performance Sparse Matrix Stacking Operations
///
/// This header provides:
///   - stack_rows(): Stack matrices vertically (concatenate rows)
///   - stack_cols(): Stack matrices horizontally (concatenate columns)
///   - Binary and multi-matrix variants
///
/// ## Design Overview
///
/// Stacking is split into two cases:
///
/// ### 1. Primary Dimension Stack (Zero-Copy)
///
/// - CSR row stack / CSC column stack
/// - Simply concatenate SharedSpan vectors (reference counted)
/// - O(n) vector concatenation, no data movement
/// - Result dimensions: (Σrows, max_cols) for CSR row stack
///
/// ### 2. Secondary Dimension Stack (Merge with Offset)
///
/// - CSR column stack / CSC row stack
/// - Merge each row's data from all matrices
/// - Add index offset for each matrix: matrix k gets offset Σcols[0..k-1]
/// - SIMD-optimized index offsetting
/// - Result dimensions: (rows, Σcols) for CSR column stack
/// - Requires: All matrices must have same primary dimension
///
/// ## Performance Optimizations
///
/// - **SIMD Index Offset**: Highway vectorized addition for index offsetting
/// - **Parallel Processing**: `threading::parallel_for` for row/column processing
/// - **Pre-allocation**: `threading::parallel_reduce` to compute total NNZ
/// - **Binary Specialization**: Optimized fast path for 2-matrix case
/// - **Zero-Copy Primary Stack**: No data movement, only SharedSpan concatenation
///
/// ## Example Usage
///
/// ```cpp
/// // Stack two CSR matrices vertically (concatenate rows)
/// auto result = scl::sparse::stack_rows(csr1, csr2);
///
/// // Stack multiple CSR matrices horizontally (concatenate columns)
/// std::vector<const CSR*> matrices = {&csr1, &csr2, &csr3};
/// auto result = scl::sparse::stack_cols(std::span(matrices));
/// ```

#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/threading.hpp"
#include "scl/core/error.hpp"

#include <hwy/highway.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace scl::sparse {

// =============================================================================
// SECTION 1: SIMD Index Offset Utility
// =============================================================================

namespace detail {

/// @brief Add offset to all indices using SIMD
/// @tparam IndexT Index type
/// @param[in,out] indices Indices to offset (modified in-place)
/// @param[in] offset Offset to add
/// @note Uses Highway SIMD for vectorized addition
template<typename IndexT>
SCL_FORCE_INLINE
auto add_offset_simd(std::span<IndexT> indices, IndexT offset) -> void {
    if (indices.empty() || offset == 0) return;

    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<IndexT> d;
    const Size lanes = hn::Lanes(d);
    const auto v_offset = hn::Set(d, offset);

    Size i = 0;
    const Size n = indices.size();
    IndexT* data = indices.data();

    // SIMD loop (4x unrolled for better ILP)
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        auto v0 = hn::LoadU(d, data + i + 0 * lanes);
        auto v1 = hn::LoadU(d, data + i + 1 * lanes);
        auto v2 = hn::LoadU(d, data + i + 2 * lanes);
        auto v3 = hn::LoadU(d, data + i + 3 * lanes);

        v0 = hn::Add(v0, v_offset);
        v1 = hn::Add(v1, v_offset);
        v2 = hn::Add(v2, v_offset);
        v3 = hn::Add(v3, v_offset);

        hn::StoreU(v0, d, data + i + 0 * lanes);
        hn::StoreU(v1, d, data + i + 1 * lanes);
        hn::StoreU(v2, d, data + i + 2 * lanes);
        hn::StoreU(v3, d, data + i + 3 * lanes);
    }

    // 1x SIMD loop
    for (; i + lanes <= n; i += lanes) {
        auto v = hn::LoadU(d, data + i);
        v = hn::Add(v, v_offset);
        hn::StoreU(v, d, data + i);
    }

    // Scalar remainder
    for (; i < n; ++i) {
        data[i] += offset;
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: Primary Dimension Stack (Zero-Copy)
// =============================================================================

/// @brief Stack matrices along primary dimension (zero-copy)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] matrices Matrices to stack (must all be CSR or all CSC)
/// @param[in] strategy Buffer allocation strategy for result
/// @return Stacked matrix
/// @throws scl::ValueError if matrices is empty
/// @note Primary dimension: rows for CSR, columns for CSC
/// @note Result dimensions: (Σprimary_dims, max(secondary_dims))
/// @note Zero data copy, only SharedSpan concatenation
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_primary(
    std::span<const Sparse<ValueT, IndexT, IsCSR>*> matrices,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    SCL_CHECK_ARG(matrices.size() > 0, "stack_primary: matrices is empty");

    // Early return for single matrix
    if (matrices.size() == 1) {
        return matrices[0]->clone(strategy);
    }

    // Compute result dimensions
    IndexT total_primary = 0;
    IndexT max_secondary = 0;

    for (const auto* mat : matrices) {
        SCL_CHECK_ARG(mat != nullptr, "stack_primary: null matrix pointer");
        total_primary += mat->primary_dim();
        max_secondary = std::max(max_secondary, mat->secondary_dim());
    }

    // Create result matrix with correct dimensions
    Sparse<ValueT, IndexT, IsCSR> result;
    if constexpr (IsCSR) {
        result = Sparse<ValueT, IndexT, IsCSR>(total_primary, max_secondary);
    } else {
        result = Sparse<ValueT, IndexT, IsCSR>(max_secondary, total_primary);
    }

    // Reserve capacity for concatenation
    Size total_spans = 0;
    for (const auto* mat : matrices) {
        total_spans += static_cast<Size>(mat->primary_dim());
    }
    result.values().reserve(total_spans);
    result.indices().reserve(total_spans);

    // Zero-copy concatenation: just copy SharedSpan objects (reference counting handles lifetime)
    for (const auto* mat : matrices) {
        const auto& src_values = mat->values();
        const auto& src_indices = mat->indices();

        result.values().insert(result.values().end(), src_values.begin(), src_values.end());
        result.indices().insert(result.indices().end(), src_indices.begin(), src_indices.end());
    }

    return result;
}

/// @brief Stack two matrices along primary dimension (binary specialization)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] a First matrix
/// @param[in] b Second matrix
/// @param[in] strategy Buffer allocation strategy for result
/// @return Stacked matrix
/// @note Optimized fast path for 2-matrix case
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_primary(
    const Sparse<ValueT, IndexT, IsCSR>& a,
    const Sparse<ValueT, IndexT, IsCSR>& b,
    [[maybe_unused]] SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    // Note: strategy unused for zero-copy primary stack (SharedSpan handles memory)
    // Compute result dimensions
    const IndexT total_primary = a.primary_dim() + b.primary_dim();
    const IndexT max_secondary = std::max(a.secondary_dim(), b.secondary_dim());

    // Create result matrix
    Sparse<ValueT, IndexT, IsCSR> result;
    if constexpr (IsCSR) {
        result = Sparse<ValueT, IndexT, IsCSR>(total_primary, max_secondary);
    } else {
        result = Sparse<ValueT, IndexT, IsCSR>(max_secondary, total_primary);
    }

    // Reserve exact capacity
    const Size total_spans = static_cast<Size>(a.primary_dim() + b.primary_dim());
    result.values().reserve(total_spans);
    result.indices().reserve(total_spans);

    // Zero-copy concatenation
    const auto& a_values = a.values();
    const auto& a_indices = a.indices();
    const auto& b_values = b.values();
    const auto& b_indices = b.indices();

    result.values().insert(result.values().end(), a_values.begin(), a_values.end());
    result.values().insert(result.values().end(), b_values.begin(), b_values.end());
    
    result.indices().insert(result.indices().end(), a_indices.begin(), a_indices.end());
    result.indices().insert(result.indices().end(), b_indices.begin(), b_indices.end());

    return result;
}

// =============================================================================
// SECTION 3: Secondary Dimension Stack (Merge with Offset)
// =============================================================================

/// @brief Stack matrices along secondary dimension (merge with index offset)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] matrices Matrices to stack (must all have same primary dimension)
/// @param[in] strategy Buffer allocation strategy for result
/// @return Stacked matrix
/// @throws scl::ValueError if matrices is empty or primary dimensions mismatch
/// @note Secondary dimension: columns for CSR, rows for CSC
/// @note Result dimensions: (primary_dim, Σsecondary_dims)
/// @note Uses SIMD index offsetting and parallel processing
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_secondary(
    std::span<const Sparse<ValueT, IndexT, IsCSR>*> matrices,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    SCL_CHECK_ARG(matrices.size() > 0, "stack_secondary: matrices is empty");

    // Early return for single matrix
    if (matrices.size() == 1) {
        return matrices[0]->clone(strategy);
    }

    // Validate all matrices have same primary dimension
    const IndexT primary_dim = matrices[0]->primary_dim();
    for (Size k = 1; k < matrices.size(); ++k) {
        SCL_CHECK_ARG(matrices[k] != nullptr, "stack_secondary: null matrix pointer");
        SCL_CHECK_ARG(matrices[k]->primary_dim() == primary_dim,
                      "stack_secondary: primary dimension mismatch");
    }

    // Compute secondary dimension offsets
    std::vector<IndexT> secondary_offsets(matrices.size() + 1);
    secondary_offsets[0] = 0;
    for (Size k = 0; k < matrices.size(); ++k) {
        secondary_offsets[k + 1] = secondary_offsets[k] + matrices[k]->secondary_dim();
    }
    const IndexT total_secondary = secondary_offsets[matrices.size()];

    // Compute per-row NNZ counts (parallel reduction)
    std::vector<IndexT> row_nnz(static_cast<Size>(primary_dim));
    
    threading::parallel_for(0, static_cast<threading::Index>(primary_dim), 
                           [&](threading::Index i) {
        IndexT count = 0;
        for (const auto* mat : matrices) {
            count += static_cast<IndexT>(mat->primary_length(static_cast<IndexT>(i)));
        }
        row_nnz[static_cast<Size>(i)] = count;
    });

    // Create result matrix with pre-allocation
    Sparse<ValueT, IndexT, IsCSR> result;
    if constexpr (IsCSR) {
        result = Sparse<ValueT, IndexT, IsCSR>(primary_dim, total_secondary);
    } else {
        result = Sparse<ValueT, IndexT, IsCSR>(total_secondary, primary_dim);
    }

    // Allocate spans for each row
    // TODO: Implement buffer strategy (currently fragmented for simplicity)
    for (IndexT i = 0; i < primary_dim; ++i) {
        const auto nnz = row_nnz[static_cast<Size>(i)];
        if (nnz > 0) {
            result.values()[static_cast<Size>(i)] = SharedSpan<ValueT>::allocate(static_cast<Size>(nnz));
            result.indices()[static_cast<Size>(i)] = SharedSpan<IndexT>::allocate(static_cast<Size>(nnz));
        }
    }

    // Parallel fill: merge matrices for each row
    threading::parallel_for(0, static_cast<threading::Index>(primary_dim),
                           [&](threading::Index i) {
        const auto idx = static_cast<IndexT>(i);
        auto& dst_values = result.values()[static_cast<Size>(i)];
        auto& dst_indices = result.indices()[static_cast<Size>(i)];

        if (dst_values.empty()) return;

        Size pos = 0;
        for (Size k = 0; k < matrices.size(); ++k) {
            const auto* mat = matrices[k];
            const auto src_len = mat->primary_length(idx);

            if (src_len == 0) continue;

            const auto& src_values = mat->primary_values(idx);
            const auto& src_indices = mat->primary_indices(idx);

            // Copy values
            memory::copy_fast(
                std::span<const ValueT>(src_values.data(), src_len),
                std::span<ValueT>(dst_values.data() + pos, src_len)
            );

            // Copy indices and add offset
            memory::copy_fast(
                std::span<const IndexT>(src_indices.data(), src_len),
                std::span<IndexT>(dst_indices.data() + pos, src_len)
            );

            // SIMD offset
            const IndexT offset = secondary_offsets[k];
            if (offset != 0) {
                detail::add_offset_simd(
                    std::span<IndexT>(dst_indices.data() + pos, src_len),
                    offset
                );
            }

            pos += src_len;
        }
    });

    return result;
}

/// @brief Stack two matrices along secondary dimension (binary specialization)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] a First matrix
/// @param[in] b Second matrix
/// @param[in] strategy Buffer allocation strategy for result
/// @return Stacked matrix
/// @throws scl::ValueError if primary dimensions mismatch
/// @note Optimized fast path for 2-matrix case
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_secondary(
    const Sparse<ValueT, IndexT, IsCSR>& a,
    const Sparse<ValueT, IndexT, IsCSR>& b,
    [[maybe_unused]] SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    // TODO: Implement buffer strategy (currently fragmented for simplicity)
    // Validate dimensions
    const IndexT primary_dim = a.primary_dim();
    SCL_CHECK_ARG(b.primary_dim() == primary_dim, 
                  "stack_secondary: primary dimension mismatch");

    const IndexT total_secondary = a.secondary_dim() + b.secondary_dim();
    const IndexT b_offset = a.secondary_dim();

    // Compute per-row NNZ counts
    std::vector<IndexT> row_nnz(static_cast<Size>(primary_dim));
    threading::parallel_for(0, static_cast<threading::Index>(primary_dim),
                           [&](threading::Index i) {
        const auto idx = static_cast<IndexT>(i);
        row_nnz[static_cast<Size>(i)] = 
            static_cast<IndexT>(a.primary_length(idx) + b.primary_length(idx));
    });

    // Create result matrix
    Sparse<ValueT, IndexT, IsCSR> result;
    if constexpr (IsCSR) {
        result = Sparse<ValueT, IndexT, IsCSR>(primary_dim, total_secondary);
    } else {
        result = Sparse<ValueT, IndexT, IsCSR>(total_secondary, primary_dim);
    }

    // Allocate spans
    for (IndexT i = 0; i < primary_dim; ++i) {
        const auto nnz = row_nnz[static_cast<Size>(i)];
        if (nnz > 0) {
            result.values()[static_cast<Size>(i)] = SharedSpan<ValueT>::allocate(static_cast<Size>(nnz));
            result.indices()[static_cast<Size>(i)] = SharedSpan<IndexT>::allocate(static_cast<Size>(nnz));
        }
    }

    // Parallel merge (binary case - simpler and faster)
    threading::parallel_for(0, static_cast<threading::Index>(primary_dim),
                           [&](threading::Index i) {
        const auto idx = static_cast<IndexT>(i);
        const auto a_len = a.primary_length(idx);
        const auto b_len = b.primary_length(idx);

        if (a_len == 0 && b_len == 0) return;

        auto& dst_values = result.values()[static_cast<Size>(i)];
        auto& dst_indices = result.indices()[static_cast<Size>(i)];

        // Copy from matrix a
        if (a_len > 0) {
            const auto& src_a_values = a.primary_values(idx);
            const auto& src_a_indices = a.primary_indices(idx);

            memory::copy_fast(
                std::span<const ValueT>(src_a_values.data(), a_len),
                std::span<ValueT>(dst_values.data(), a_len)
            );
            memory::copy_fast(
                std::span<const IndexT>(src_a_indices.data(), a_len),
                std::span<IndexT>(dst_indices.data(), a_len)
            );
        }

        // Copy from matrix b with offset
        if (b_len > 0) {
            const auto& src_b_values = b.primary_values(idx);
            const auto& src_b_indices = b.primary_indices(idx);

            memory::copy_fast(
                std::span<const ValueT>(src_b_values.data(), b_len),
                std::span<ValueT>(dst_values.data() + a_len, b_len)
            );
            memory::copy_fast(
                std::span<const IndexT>(src_b_indices.data(), b_len),
                std::span<IndexT>(dst_indices.data() + a_len, b_len)
            );

            // SIMD offset for b's indices
            detail::add_offset_simd(
                std::span<IndexT>(dst_indices.data() + a_len, b_len),
                b_offset
            );
        }
    });

    return result;
}

// =============================================================================
// SECTION 4: Public API (stack_rows / stack_cols)
// =============================================================================

/// @brief Stack matrices vertically (concatenate rows)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] matrices Matrices to stack
/// @param[in] strategy Buffer allocation strategy for result
/// @return Vertically stacked matrix
/// @note CSR: Primary dimension stack (zero-copy)
/// @note CSC: Secondary dimension stack (merge with offset, requires same column count)
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_rows(
    std::span<const Sparse<ValueT, IndexT, IsCSR>*> matrices,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    if constexpr (IsCSR) {
        // CSR: rows are primary dimension (zero-copy)
        return stack_primary(matrices, strategy);
    } else {
        // CSC: rows are secondary dimension (merge with offset)
        return stack_secondary(matrices, strategy);
    }
}

/// @brief Stack two matrices vertically (binary specialization)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] a First matrix
/// @param[in] b Second matrix
/// @param[in] strategy Buffer allocation strategy for result
/// @return Vertically stacked matrix
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_rows(
    const Sparse<ValueT, IndexT, IsCSR>& a,
    const Sparse<ValueT, IndexT, IsCSR>& b,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    if constexpr (IsCSR) {
        return stack_primary(a, b, strategy);
    } else {
        return stack_secondary(a, b, strategy);
    }
}

/// @brief Stack matrices horizontally (concatenate columns)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] matrices Matrices to stack
/// @param[in] strategy Buffer allocation strategy for result
/// @return Horizontally stacked matrix
/// @note CSR: Secondary dimension stack (merge with offset, requires same row count)
/// @note CSC: Primary dimension stack (zero-copy)
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_cols(
    std::span<const Sparse<ValueT, IndexT, IsCSR>*> matrices,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    if constexpr (IsCSR) {
        // CSR: columns are secondary dimension (merge with offset)
        return stack_secondary(matrices, strategy);
    } else {
        // CSC: columns are primary dimension (zero-copy)
        return stack_primary(matrices, strategy);
    }
}

/// @brief Stack two matrices horizontally (binary specialization)
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR, false for CSC
/// @param[in] a First matrix
/// @param[in] b Second matrix
/// @param[in] strategy Buffer allocation strategy for result
/// @return Horizontally stacked matrix
template<typename ValueT, typename IndexT, bool IsCSR>
[[nodiscard]]
auto stack_cols(
    const Sparse<ValueT, IndexT, IsCSR>& a,
    const Sparse<ValueT, IndexT, IsCSR>& b,
    SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy()
) -> Sparse<ValueT, IndexT, IsCSR> {
    if constexpr (IsCSR) {
        return stack_secondary(a, b, strategy);
    } else {
        return stack_primary(a, b, strategy);
    }
}

} // namespace scl::sparse

