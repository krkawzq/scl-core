#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cstring>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file reorder_mapped_impl.hpp
/// @brief Extreme Performance Reordering for Memory-Mapped Sparse Matrices
///
/// Streaming-optimized reordering for disk-backed data:
///
/// Key Optimizations:
///
/// 1. Two-Pass Strategy: Count first, allocate once, then fill
/// 2. Chunk-Based Streaming: Sequential access for page cache efficiency
/// 3. Prefetch Pipeline: Hint OS for upcoming pages
/// 4. Parallel VQSort: Ultra-fast sorting per row
/// 5. Minimal Materialization: Only copy data that passes filter
///
/// Note: Mapped matrices are read-only, so we must materialize to OwnedSparse.
// =============================================================================

namespace scl::kernel::reorder::mapped {

namespace detail {

// =============================================================================
// SECTION 1: Counting Helpers
// =============================================================================

/// @brief Count valid indices (4-way unrolled)
template <typename IndexT>
SCL_FORCE_INLINE Size count_valid_indices(
    const IndexT* SCL_RESTRICT indices,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size count = 0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        IndexT idx0 = indices[k + 0];
        IndexT idx1 = indices[k + 1];
        IndexT idx2 = indices[k + 2];
        IndexT idx3 = indices[k + 3];

        IndexT new0 = index_map[idx0];
        IndexT new1 = index_map[idx1];
        IndexT new2 = index_map[idx2];
        IndexT new3 = index_map[idx3];

        count += (new0 >= 0 && new0 < new_dim) ? 1 : 0;
        count += (new1 >= 0 && new1 < new_dim) ? 1 : 0;
        count += (new2 >= 0 && new2 < new_dim) ? 1 : 0;
        count += (new3 >= 0 && new3 < new_dim) ? 1 : 0;
    }

    for (; k < len; ++k) {
        IndexT new_idx = index_map[indices[k]];
        if (new_idx >= 0 && new_idx < new_dim) {
            ++count;
        }
    }

    return count;
}

/// @brief Filter and remap indices/values
template <typename T, typename IndexT>
SCL_FORCE_INLINE Size filter_remap_row(
    const IndexT* SCL_RESTRICT src_indices,
    const T* SCL_RESTRICT src_values,
    Size src_len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim,
    IndexT* SCL_RESTRICT dst_indices,
    T* SCL_RESTRICT dst_values
) {
    Size write_pos = 0;

    for (Size k = 0; k < src_len; ++k) {
        IndexT old_idx = src_indices[k];
        IndexT new_idx = index_map[old_idx];

        if (new_idx >= 0 && new_idx < new_dim) {
            dst_indices[write_pos] = new_idx;
            dst_values[write_pos] = src_values[k];
            ++write_pos;
        }
    }

    return write_pos;
}

} // namespace detail

// =============================================================================
// SECTION 2: MappedCustomSparse Reordering
// =============================================================================

/// @brief Align secondary dimension for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Phase 1: Count filtered NNZ per row
    std::vector<Size> row_nnz(static_cast<size_t>(primary_dim));

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (static_cast<Size>(primary_dim) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        // Prefetch next chunk
        if (chunk_id + 1 < n_chunks) {
            Index next_start = static_cast<Index>((chunk_id + 1) * CHUNK_SIZE);
            auto indices_next = scl::primary_indices(matrix, next_start);
            SCL_PREFETCH_READ(indices_next.ptr, 0);
        }

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto indices = scl::primary_indices(matrix, p);
            row_nnz[p] = detail::count_valid_indices(
                indices.ptr, indices.len,
                index_map.ptr, new_secondary_dim
            );
        });
    }

    // Phase 2: Compute prefix sums
    std::vector<Index> new_indptr(static_cast<size_t>(primary_dim) + 1);
    new_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        new_indptr[p + 1] = new_indptr[p] + static_cast<Index>(row_nnz[p]);
    }
    Size total_nnz = static_cast<Size>(new_indptr[primary_dim]);

    // Allocate output vectors
    std::vector<T> out_data(total_nnz);
    std::vector<Index> out_indices(total_nnz);

    // Phase 3: Fill with filtering and sorting
    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            Size new_len = row_nnz[p];
            if (new_len == 0) return;

            auto src_indices = scl::primary_indices(matrix, p);
            auto src_values = scl::primary_values(matrix, p);

            Index dst_start = new_indptr[p];
            Index* dst_idx = out_indices.data() + dst_start;
            T* dst_val = out_data.data() + dst_start;

            detail::filter_remap_row(
                src_indices.ptr, src_values.ptr, src_indices.len,
                index_map.ptr, new_secondary_dim,
                dst_idx, dst_val
            );

            if (new_len > 1) {
                scl::sort::sort_pairs(
                    Array<Index>(dst_idx, new_len),
                    Array<T>(dst_val, new_len)
                );
            }
        });
    }

    // Build OwnedSparse
    Index out_rows = IsCSR ? primary_dim : new_secondary_dim;
    Index out_cols = IsCSR ? new_secondary_dim : primary_dim;

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(new_indptr),
        out_rows, out_cols
    );
}

// =============================================================================
// SECTION 3: MappedVirtualSparse Reordering
// =============================================================================

/// @brief Align secondary dimension for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);

    // Phase 1: Count filtered NNZ per row
    std::vector<Size> row_nnz(static_cast<size_t>(primary_dim));

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (static_cast<Size>(primary_dim) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto indices = scl::primary_indices(matrix, p);
            row_nnz[p] = detail::count_valid_indices(
                indices.ptr, indices.len,
                index_map.ptr, new_secondary_dim
            );
        });
    }

    // Phase 2: Compute prefix sums
    std::vector<Index> new_indptr(static_cast<size_t>(primary_dim) + 1);
    new_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        new_indptr[p + 1] = new_indptr[p] + static_cast<Index>(row_nnz[p]);
    }
    Size total_nnz = static_cast<Size>(new_indptr[primary_dim]);

    // Allocate output vectors
    std::vector<T> out_data(total_nnz);
    std::vector<Index> out_indices(total_nnz);

    // Phase 3: Fill with filtering and sorting
    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            Size new_len = row_nnz[p];
            if (new_len == 0) return;

            auto src_indices = scl::primary_indices(matrix, p);
            auto src_values = scl::primary_values(matrix, p);

            Index dst_start = new_indptr[p];
            Index* dst_idx = out_indices.data() + dst_start;
            T* dst_val = out_data.data() + dst_start;

            detail::filter_remap_row(
                src_indices.ptr, src_values.ptr, src_indices.len,
                index_map.ptr, new_secondary_dim,
                dst_idx, dst_val
            );

            if (new_len > 1) {
                scl::sort::sort_pairs(
                    Array<Index>(dst_idx, new_len),
                    Array<T>(dst_val, new_len)
                );
            }
        });
    }

    // Build OwnedSparse
    Index out_rows = IsCSR ? primary_dim : new_secondary_dim;
    Index out_cols = IsCSR ? new_secondary_dim : primary_dim;

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(new_indptr),
        out_rows, out_cols
    );
}

// =============================================================================
// SECTION 4: Unified Dispatcher
// =============================================================================

/// @brief Unified dispatcher for mapped reordering
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> align_secondary_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        return align_secondary_mapped_custom(matrix, index_map, new_secondary_dim);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        return align_secondary_mapped_virtual(matrix, index_map, new_secondary_dim);
    } else {
        // Generic fallback: materialize first
        auto owned = matrix.materialize();
        const Index primary_dim = scl::primary_size(owned.view());

        // Count phase
        std::vector<Size> row_nnz(static_cast<size_t>(primary_dim));
        for (Index p = 0; p < primary_dim; ++p) {
            auto indices = scl::primary_indices(owned.view(), p);
            row_nnz[p] = detail::count_valid_indices(
                indices.ptr, indices.len,
                index_map.ptr, new_secondary_dim
            );
        }

        // Build indptr
        std::vector<Index> new_indptr(static_cast<size_t>(primary_dim) + 1);
        new_indptr[0] = 0;
        for (Index p = 0; p < primary_dim; ++p) {
            new_indptr[p + 1] = new_indptr[p] + static_cast<Index>(row_nnz[p]);
        }
        Size total_nnz = static_cast<Size>(new_indptr[primary_dim]);

        // Allocate output
        std::vector<T> out_data(total_nnz);
        std::vector<Index> out_indices(total_nnz);

        // Fill
        for (Index p = 0; p < primary_dim; ++p) {
            Size new_len = row_nnz[p];
            if (new_len == 0) continue;

            auto src_indices = scl::primary_indices(owned.view(), p);
            auto src_values = scl::primary_values(owned.view(), p);

            Index dst_start = new_indptr[p];
            Index* dst_idx = out_indices.data() + dst_start;
            T* dst_val = out_data.data() + dst_start;

            detail::filter_remap_row(
                src_indices.ptr, src_values.ptr, src_indices.len,
                index_map.ptr, new_secondary_dim,
                dst_idx, dst_val
            );

            if (new_len > 1) {
                scl::sort::sort_pairs(
                    Array<Index>(dst_idx, new_len),
                    Array<T>(dst_val, new_len)
                );
            }
        }

        Index out_rows = IsCSR ? primary_dim : new_secondary_dim;
        Index out_cols = IsCSR ? new_secondary_dim : primary_dim;

        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(new_indptr),
            out_rows, out_cols
        );
    }
}

// =============================================================================
// SECTION 5: Query Utilities
// =============================================================================

/// @brief Compute filtered NNZ without materializing
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
Size compute_filtered_nnz_mapped(
    const MatrixT& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);

    std::vector<Size> partial_sums(static_cast<size_t>(primary_dim));

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto indices = scl::primary_indices(matrix, p);
        partial_sums[p] = detail::count_valid_indices(
            indices.ptr, indices.len,
            index_map.ptr, new_secondary_dim
        );
    });

    Size total = 0;
    for (Size s : partial_sums) {
        total += s;
    }

    return total;
}

} // namespace scl::kernel::reorder::mapped
