#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cstring>

// =============================================================================
/// @file merge_mapped_impl.hpp
/// @brief Matrix Merging for Mapped Sparse Matrices
///
/// Merge operations are O(rows) index array operations, not data-intensive.
/// For Mapped matrices, we support:
/// - Building merged VirtualSparse views (zero-copy when possible)
/// - Materializing to OwnedSparse when needed
///
/// Operations:
/// - compute_merge_layout_mapped: Compute merged matrix layout
/// - materialize_merged_mapped: Materialize merged matrix
// =============================================================================

namespace scl::kernel::merge::mapped {

// =============================================================================
// Merge Layout Computation
// =============================================================================

/// @brief Compute total NNZ for merged matrix (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
Index compute_total_nnz_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2
) {
    const Index nnz1 = matrix1.indptr[scl::primary_size(matrix1)];
    const Index nnz2 = matrix2.indptr[scl::primary_size(matrix2)];
    return nnz1 + nnz2;
}

/// @brief Compute total NNZ for merged matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
Index compute_total_nnz_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix1,
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix2
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);

    Index total_nnz = 0;
    for (Index p = 0; p < primary_dim1; ++p) {
        total_nnz += matrix1.lengths[p];
    }
    for (Index p = 0; p < primary_dim2; ++p) {
        total_nnz += matrix2.lengths[p];
    }
    return total_nnz;
}

// =============================================================================
// Materialize Merged Matrix
// =============================================================================

/// @brief Materialize merged matrix from two MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> materialize_merged_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2,
    Index secondary_offset  // Offset to add to indices of matrix2
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index nnz1 = matrix1.indptr[primary_dim1];
    const Index nnz2 = matrix2.indptr[primary_dim2];
    const Index total_nnz = nnz1 + nnz2;

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(total_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_primary) + 1);

    auto out_data = data_handle.template as_span<T>();
    auto out_indices = indices_handle.template as_span<Index>();
    auto out_indptr = indptr_handle.template as_span<Index>();

    // Copy matrix1
    std::memcpy(out_data.data(), matrix1.data, nnz1 * sizeof(T));
    std::memcpy(out_indices.data(), matrix1.indices, nnz1 * sizeof(Index));
    std::memcpy(out_indptr.data(), matrix1.indptr, (primary_dim1 + 1) * sizeof(Index));

    // Copy matrix2 with offset
    std::memcpy(out_data.data() + nnz1, matrix2.data, nnz2 * sizeof(T));

    if (secondary_offset == 0) {
        std::memcpy(out_indices.data() + nnz1, matrix2.indices, nnz2 * sizeof(Index));
    } else {
        scl::threading::parallel_for(0, static_cast<size_t>(nnz2), [&](size_t k) {
            out_indices[nnz1 + k] = matrix2.indices[k] + secondary_offset;
        });
    }

    // Update indptr for matrix2 portion
    for (Index p = 0; p <= primary_dim2; ++p) {
        out_indptr[primary_dim1 + p] = matrix2.indptr[p] + nnz1;
    }

    // Build result
    scl::io::OwnedSparse<T, IsCSR> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();

    if constexpr (IsCSR) {
        result.matrix.rows = total_primary;
        result.matrix.cols = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    } else {
        result.matrix.cols = total_primary;
        result.matrix.rows = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    }

    result.matrix.data = result.data.ptr;
    result.matrix.indices = result.indices.ptr;
    result.matrix.indptr = result.indptr.ptr;

    return result;
}

/// @brief Materialize merged matrix from two MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> materialize_merged_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix1,
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix2,
    Index secondary_offset
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index total_nnz = compute_total_nnz_mapped(matrix1, matrix2);

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(total_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_primary) + 1);

    auto out_data = data_handle.template as_span<T>();
    auto out_indices = indices_handle.template as_span<Index>();
    auto out_indptr = indptr_handle.template as_span<Index>();

    // Build indptr
    out_indptr[0] = 0;
    Index pos = 0;
    for (Index p = 0; p < primary_dim1; ++p) {
        pos += matrix1.lengths[p];
        out_indptr[p + 1] = pos;
    }
    for (Index p = 0; p < primary_dim2; ++p) {
        pos += matrix2.lengths[p];
        out_indptr[primary_dim1 + p + 1] = pos;
    }

    // Copy data from matrix1
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim1), [&](size_t p) {
        Index len = matrix1.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[p];
        const T* src_data = static_cast<const T*>(matrix1.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix1.indices_ptrs[p]);

        std::memcpy(out_data.data() + dst_start, src_data, len * sizeof(T));
        std::memcpy(out_indices.data() + dst_start, src_indices, len * sizeof(Index));
    });

    // Copy data from matrix2 with offset
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim2), [&](size_t p) {
        Index len = matrix2.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[primary_dim1 + p];
        const T* src_data = static_cast<const T*>(matrix2.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix2.indices_ptrs[p]);

        std::memcpy(out_data.data() + dst_start, src_data, len * sizeof(T));

        if (secondary_offset == 0) {
            std::memcpy(out_indices.data() + dst_start, src_indices, len * sizeof(Index));
        } else {
            for (Index k = 0; k < len; ++k) {
                out_indices[dst_start + k] = src_indices[k] + secondary_offset;
            }
        }
    });

    // Build result
    scl::io::OwnedSparse<T, IsCSR> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();

    if constexpr (IsCSR) {
        result.matrix.rows = total_primary;
        result.matrix.cols = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    } else {
        result.matrix.cols = total_primary;
        result.matrix.rows = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    }

    result.matrix.data = result.data.ptr;
    result.matrix.indices = result.indices.ptr;
    result.matrix.indptr = result.indptr.ptr;

    return result;
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
Index compute_total_nnz_mapped_dispatch(
    const MatrixT& matrix1,
    const MatrixT& matrix2
) {
    return compute_total_nnz_mapped(matrix1, matrix2);
}

} // namespace scl::kernel::merge::mapped

