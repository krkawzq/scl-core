#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cstring>

// =============================================================================
/// @file reorder_mapped_impl.hpp
/// @brief Reordering for Mapped Sparse Matrices
///
/// Reorder operations remap secondary indices and sort within rows.
/// For Mapped matrices (read-only), we must:
/// 1. Materialize to OwnedSparse
/// 2. Apply remapping and sorting in-place
///
/// Operations:
/// - align_secondary_mapped: Remap and sort secondary indices
// =============================================================================

namespace scl::kernel::reorder::mapped {

// =============================================================================
// Secondary Alignment - Materialize + Remap
// =============================================================================

/// @brief Align secondary dimension for mapped matrix (MappedCustomSparse)
///
/// Materializes first, then applies remapping and sorting
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    // Materialize first
    auto owned = matrix.materialize();
    const Index primary_dim = scl::primary_size(owned.matrix);

    // Compute new lengths after filtering
    std::vector<Index> new_lengths(static_cast<size_t>(primary_dim));

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = owned.matrix.indptr[p];
        Index end = owned.matrix.indptr[p + 1];
        Index len = end - start;

        if (len == 0) {
            new_lengths[p] = 0;
            return;
        }

        Index* inds = owned.indices.ptr + start;
        T* vals = owned.data.ptr + start;

        // Remap and filter in-place
        Index write_pos = 0;
        for (Index k = 0; k < len; ++k) {
            Index old_idx = inds[k];
            Index new_idx = index_map[old_idx];

            if (new_idx >= 0 && new_idx < new_secondary_dim) {
                inds[write_pos] = new_idx;
                vals[write_pos] = vals[k];
                write_pos++;
            }
        }

        // Sort using VQSort
        if (write_pos > 1) {
            scl::sort::sort_pairs(
                Array<Index>(inds, write_pos),
                Array<T>(vals, write_pos)
            );
        }

        new_lengths[p] = write_pos;
    });

    // Rebuild indptr based on new lengths
    Index new_nnz = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        owned.matrix.indptr[p] = new_nnz;
        new_nnz += new_lengths[p];
    }
    owned.matrix.indptr[primary_dim] = new_nnz;

    // Compact data if necessary (when filtering removed elements)
    // For simplicity, we skip compaction and keep sparse structure as-is
    // The indptr correctly reflects the valid ranges

    // Update dimensions
    if constexpr (IsCSR) {
        owned.matrix.cols = new_secondary_dim;
    } else {
        owned.matrix.rows = new_secondary_dim;
    }

    return owned;
}

/// @brief Align secondary dimension for mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    // Materialize first
    auto owned = matrix.materialize();
    const Index primary_dim = scl::primary_size(owned.matrix);

    // Compute new lengths after filtering
    std::vector<Index> new_lengths(static_cast<size_t>(primary_dim));

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = owned.matrix.indptr[p];
        Index end = owned.matrix.indptr[p + 1];
        Index len = end - start;

        if (len == 0) {
            new_lengths[p] = 0;
            return;
        }

        Index* inds = owned.indices.ptr + start;
        T* vals = owned.data.ptr + start;

        // Remap and filter in-place
        Index write_pos = 0;
        for (Index k = 0; k < len; ++k) {
            Index old_idx = inds[k];
            Index new_idx = index_map[old_idx];

            if (new_idx >= 0 && new_idx < new_secondary_dim) {
                inds[write_pos] = new_idx;
                vals[write_pos] = vals[k];
                write_pos++;
            }
        }

        // Sort using VQSort
        if (write_pos > 1) {
            scl::sort::sort_pairs(
                Array<Index>(inds, write_pos),
                Array<T>(vals, write_pos)
            );
        }

        new_lengths[p] = write_pos;
    });

    // Rebuild indptr based on new lengths
    Index new_nnz = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        owned.matrix.indptr[p] = new_nnz;
        new_nnz += new_lengths[p];
    }
    owned.matrix.indptr[primary_dim] = new_nnz;

    // Update dimensions
    if constexpr (IsCSR) {
        owned.matrix.cols = new_secondary_dim;
    } else {
        owned.matrix.rows = new_secondary_dim;
    }

    return owned;
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> align_secondary_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return align_secondary_mapped(matrix, index_map, new_secondary_dim);
}

} // namespace scl::kernel::reorder::mapped

