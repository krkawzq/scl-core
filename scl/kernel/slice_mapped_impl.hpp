#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/slice.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>
#include <atomic>

// =============================================================================
/// @file slice_mapped_impl.hpp
/// @brief Mapped Backend Slicing Operations
///
/// Key insight: Mapped data supports efficient streaming reads.
/// For slicing:
/// - Primary slice: Extract selected rows/columns (read-only, sequential)
/// - Secondary filter: Filter by mask (read-only, needs inspection)
///
/// Both operations return newly allocated SliceResult.
// =============================================================================

namespace scl::kernel::slice::mapped {

// =============================================================================
// MappedCustomSparse Primary Slicing
// =============================================================================

/// @brief Inspect primary dimension slice for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
Index inspect_slice_primary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    const Index n_primary = scl::primary_size(matrix);

    if (n_keep == 0) return 0;

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // For small slices, use sequential
    if (n_keep < 1000) {
        Index total_nnz = 0;
        for (Size i = 0; i < n_keep; ++i) {
            Index idx = keep_indices[i];
            SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
            total_nnz += scl::primary_length(matrix, idx);
        }
        return total_nnz;
    }

    // For large slices, use parallel reduction
    std::atomic<Index> total_nnz{0};

    scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
        Index len = scl::primary_length(matrix, idx);
        total_nnz.fetch_add(len, std::memory_order_relaxed);
    });

    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Materialize primary dimension slice for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void materialize_slice_primary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.size();

    SCL_CHECK_DIM(out_indptr.size() >= n_keep + 1, "Output indptr too small");

    // Build indptr first (sequential)
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index len = scl::primary_length(matrix, src_idx);
        out_indptr[i + 1] = out_indptr[i] + len;
    }

    // Parallel copy data and indices
    scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
        Index src_idx = keep_indices[i];
        auto values = scl::primary_values(matrix, src_idx);
        auto indices = scl::primary_indices(matrix, src_idx);
        Index len = static_cast<Index>(values.len);

        if (len == 0) return;

        Index dst_start = out_indptr[i];

        // Bulk copy using memcpy
        if (len >= 8) {
            std::memcpy(out_data.data() + dst_start, values.ptr, len * sizeof(T));
            std::memcpy(out_indices.data() + dst_start, indices.ptr, len * sizeof(Index));
        } else {
            for (Index k = 0; k < len; ++k) {
                out_data[dst_start + k] = values.ptr[k];
                out_indices[dst_start + k] = indices.ptr[k];
            }
        }
    });
}

/// @brief Primary dimension slice for MappedCustomSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
SliceResult<T> slice_primary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();

    // Fast inspection
    Index out_nnz = inspect_slice_primary_mapped(matrix, keep_indices);

    // Allocate output arrays
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(n_keep + 1);

    // Fast materialization
    materialize_slice_primary_mapped(
        matrix,
        keep_indices,
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );

    // Build result
    SliceResult<T> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();
    result.n_primary = static_cast<Index>(n_keep);
    result.n_secondary = scl::secondary_size(matrix);
    result.nnz = out_nnz;

    return result;
}

// =============================================================================
// MappedCustomSparse Secondary Filtering
// =============================================================================

/// @brief Inspect secondary dimension filter for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
Index inspect_filter_secondary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index n_primary = scl::primary_size(matrix);

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    std::atomic<Index> total_nnz{0};

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto indices = scl::primary_indices(matrix, p);
            Index local_count = 0;

            for (Size k = 0; k < indices.len; ++k) {
                if (mask[indices.ptr[k]]) {
                    local_count++;
                }
            }

            if (local_count > 0) {
                total_nnz.fetch_add(local_count, std::memory_order_relaxed);
            }
        });
    }

    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Materialize secondary dimension filter for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void materialize_filter_secondary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Index n_primary = scl::primary_size(matrix);

    // First pass: compute output indptr (must be sequential for prefix sum)
    out_indptr[0] = 0;
    for (Index p = 0; p < n_primary; ++p) {
        auto indices = scl::primary_indices(matrix, p);
        Index count = 0;

        for (Size k = 0; k < indices.len; ++k) {
            if (mask[indices.ptr[k]]) {
                count++;
            }
        }

        out_indptr[p + 1] = out_indptr[p] + count;
    }

    // Second pass: parallel copy filtered data
    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);
        auto indices = scl::primary_indices(matrix, p);
        Index dst_pos = out_indptr[p];

        for (Size k = 0; k < values.len; ++k) {
            Index old_idx = indices.ptr[k];
            if (mask[old_idx]) {
                out_data[dst_pos] = values.ptr[k];
                out_indices[dst_pos] = new_indices[old_idx];
                dst_pos++;
            }
        }
    });
}

/// @brief Secondary dimension filter for MappedCustomSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
SliceResult<T> filter_secondary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = scl::secondary_size(matrix);
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(secondary_dim), "Mask size mismatch");

    // Build new index mapping
    auto new_indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(secondary_dim));
    auto new_indices = new_indices_handle.template as_span<Index>();

    Index new_secondary = 0;
    for (Index i = 0; i < secondary_dim; ++i) {
        if (mask[i]) {
            new_indices[i] = new_secondary++;
        } else {
            new_indices[i] = -1;
        }
    }

    // Fast inspection
    Index out_nnz = inspect_filter_secondary_mapped(matrix, mask);

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);

    // Fast materialization
    materialize_filter_secondary_mapped(
        matrix,
        mask,
        Array<const Index>(new_indices.data(), new_indices.size()),
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );

    // Build result
    SliceResult<T> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();
    result.n_primary = primary_dim;
    result.n_secondary = new_secondary;
    result.nnz = out_nnz;

    return result;
}

// =============================================================================
// MappedVirtualSparse Primary Slicing
// =============================================================================

/// @brief Inspect primary dimension slice for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
Index inspect_slice_primary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    const Index n_primary = scl::primary_size(matrix);

    if (n_keep == 0) return 0;

    if (n_keep < 1000) {
        Index total_nnz = 0;
        for (Size i = 0; i < n_keep; ++i) {
            Index idx = keep_indices[i];
            SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
            total_nnz += scl::primary_length(matrix, idx);
        }
        return total_nnz;
    }

    std::atomic<Index> total_nnz{0};

    scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
        Index len = scl::primary_length(matrix, idx);
        total_nnz.fetch_add(len, std::memory_order_relaxed);
    });

    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Materialize primary dimension slice for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void materialize_slice_primary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.size();

    SCL_CHECK_DIM(out_indptr.size() >= n_keep + 1, "Output indptr too small");

    // Build indptr first
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index len = scl::primary_length(matrix, src_idx);
        out_indptr[i + 1] = out_indptr[i] + len;
    }

    // Parallel copy
    scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
        Index src_idx = keep_indices[i];
        auto values = scl::primary_values(matrix, src_idx);
        auto indices = scl::primary_indices(matrix, src_idx);
        Index len = static_cast<Index>(values.len);

        if (len == 0) return;

        Index dst_start = out_indptr[i];

        if (len >= 8) {
            std::memcpy(out_data.data() + dst_start, values.ptr, len * sizeof(T));
            std::memcpy(out_indices.data() + dst_start, indices.ptr, len * sizeof(Index));
        } else {
            for (Index k = 0; k < len; ++k) {
                out_data[dst_start + k] = values.ptr[k];
                out_indices[dst_start + k] = indices.ptr[k];
            }
        }
    });
}

/// @brief Primary dimension slice for MappedVirtualSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
SliceResult<T> slice_primary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();

    Index out_nnz = inspect_slice_primary_mapped(matrix, keep_indices);

    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(n_keep + 1);

    materialize_slice_primary_mapped(
        matrix,
        keep_indices,
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );

    SliceResult<T> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();
    result.n_primary = static_cast<Index>(n_keep);
    result.n_secondary = scl::secondary_size(matrix);
    result.nnz = out_nnz;

    return result;
}

// =============================================================================
// MappedVirtualSparse Secondary Filtering
// =============================================================================

/// @brief Secondary dimension filter for MappedVirtualSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
SliceResult<T> filter_secondary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = scl::secondary_size(matrix);
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(secondary_dim), "Mask size mismatch");

    // Build new index mapping
    auto new_indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(secondary_dim));
    auto new_indices = new_indices_handle.template as_span<Index>();

    Index new_secondary = 0;
    for (Index i = 0; i < secondary_dim; ++i) {
        if (mask[i]) {
            new_indices[i] = new_secondary++;
        } else {
            new_indices[i] = -1;
        }
    }

    // Count total nnz
    std::atomic<Index> total_nnz{0};

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto indices = scl::primary_indices(matrix, p);
        Index local_count = 0;

        for (Size k = 0; k < indices.len; ++k) {
            if (mask[indices.ptr[k]]) {
                local_count++;
            }
        }

        if (local_count > 0) {
            total_nnz.fetch_add(local_count, std::memory_order_relaxed);
        }
    });

    Index out_nnz = total_nnz.load(std::memory_order_relaxed);

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);

    auto out_data = data_handle.template as_span<T>();
    auto out_indices = indices_handle.template as_span<Index>();
    auto out_indptr = indptr_handle.template as_span<Index>();

    // Build indptr
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        auto indices = scl::primary_indices(matrix, p);
        Index count = 0;

        for (Size k = 0; k < indices.len; ++k) {
            if (mask[indices.ptr[k]]) {
                count++;
            }
        }

        out_indptr[p + 1] = out_indptr[p] + count;
    }

    // Parallel copy
    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto values = scl::primary_values(matrix, p);
        auto indices = scl::primary_indices(matrix, p);
        Index dst_pos = out_indptr[p];

        for (Size k = 0; k < values.len; ++k) {
            Index old_idx = indices.ptr[k];
            if (mask[old_idx]) {
                out_data[dst_pos] = values.ptr[k];
                out_indices[dst_pos] = new_indices[old_idx];
                dst_pos++;
            }
        }
    });

    SliceResult<T> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();
    result.n_primary = primary_dim;
    result.n_secondary = new_secondary;
    result.nnz = out_nnz;

    return result;
}

} // namespace scl::kernel::slice::mapped
