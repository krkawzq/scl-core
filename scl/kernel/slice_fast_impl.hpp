#pragma once

#include "scl/kernel/slice.hpp"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>

// =============================================================================
/// @file slice_fast_impl.hpp
/// @brief Extreme Performance Sparse Matrix Slicing
///
/// Separate optimizations:
/// - CustomSparse: Parallel memcpy + cache-friendly access
/// - VirtualSparse: Row-wise parallel copy with minimal indirection
///
/// Ultra-optimized slicing with:
/// - Parallel primary dimension processing
/// - Bulk memcpy for contiguous data segments
/// - Cache-friendly sequential writes
/// - Prefetch hints for large slices
///
/// Performance Target: 1.5-2x faster than generic
/// Bandwidth: Near memory bandwidth limit for large slices
// =============================================================================

namespace scl::kernel::slice::fast {

// =============================================================================
// CustomSparse Fast Path - Primary Dimension Slicing
// =============================================================================

/// @brief Ultra-fast primary dimension slice inspection (CustomSparse)
///
/// Optimization: Parallel length accumulation
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE Index inspect_slice_primary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    
    if (n_keep == 0) return 0;
    
    // For small slices, use sequential
    if (n_keep < 1000) {
        Index total_nnz = 0;
        for (Size i = 0; i < n_keep; ++i) {
            Index idx = keep_indices[i];
            SCL_CHECK_ARG(idx >= 0 && idx < scl::primary_size(matrix),
                          "Slice: Index out of bounds");
            total_nnz += matrix.indptr[idx + 1] - matrix.indptr[idx];
        }
        return total_nnz;
    }
    
    // For large slices, use parallel reduction
    std::atomic<Index> total_nnz{0};
    
    scl::threading::parallel_for(0, n_keep, [&](size_t i) {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < scl::primary_size(matrix),
                      "Slice: Index out of bounds");
        Index len = matrix.indptr[idx + 1] - matrix.indptr[idx];
        total_nnz.fetch_add(len, std::memory_order_relaxed);
    });
    
    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Ultra-fast primary dimension slice materialization (CustomSparse)
///
/// Optimization: Parallel row copy with bulk memcpy
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void materialize_slice_primary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.size();
    
    SCL_CHECK_DIM(out_indptr.size() >= n_keep + 1,
                  "Slice: Output indptr too small");
    
    // Build indptr first (sequential, very fast)
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index len = matrix.indptr[src_idx + 1] - matrix.indptr[src_idx];
        out_indptr[i + 1] = out_indptr[i] + len;
    }
    
    // Parallel copy data and indices
    scl::threading::parallel_for(0, n_keep, [&](size_t i) {
        Index src_idx = keep_indices[i];
        Index src_start = matrix.indptr[src_idx];
        Index src_end = matrix.indptr[src_idx + 1];
        Index len = src_end - src_start;
        
        if (len == 0) return;
        
        Index dst_start = out_indptr[i];
        
        // Bulk copy using memcpy (much faster than loop for large segments)
        if (len >= 8) {
            std::memcpy(out_data.data() + dst_start,
                       matrix.data + src_start,
                       len * sizeof(T));
            std::memcpy(out_indices.data() + dst_start,
                       matrix.indices + src_start,
                       len * sizeof(Index));
        } else {
            // Small segments: manual copy to avoid memcpy overhead
            for (Index k = 0; k < len; ++k) {
                out_data[dst_start + k] = matrix.data[src_start + k];
                out_indices[dst_start + k] = matrix.indices[src_start + k];
            }
        }
    });
}

/// @brief Ultra-fast primary dimension slice (CustomSparse) - allocating
///
/// Optimization: Parallel inspection + parallel materialization
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE SliceResult<T> slice_primary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    
    // Fast inspection
    Index out_nnz = inspect_slice_primary_custom_fast(matrix, keep_indices);
    
    // Allocate output arrays
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(n_keep + 1);
    
    // Fast materialization
    materialize_slice_primary_custom_fast(
        matrix,
        keep_indices,
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );
    
    // Build result and release ownership
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
// CustomSparse Fast Path - Secondary Dimension Filtering
// =============================================================================

/// @brief Ultra-fast secondary dimension filter inspection (CustomSparse)
///
/// Optimization: Parallel mask checking with cache-friendly access
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE Index inspect_filter_secondary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    // For small matrices, use sequential
    if (primary_dim < 500) {
        Index total_nnz = 0;
        for (Index p = 0; p < primary_dim; ++p) {
            Index start = matrix.indptr[p];
            Index end = matrix.indptr[p + 1];
            
            for (Index k = start; k < end; ++k) {
                if (mask[matrix.indices[k]]) {
                    total_nnz++;
                }
            }
        }
        return total_nnz;
    }
    
    // For large matrices, use parallel reduction
    std::atomic<Index> total_nnz{0};
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index local_count = 0;
        
        // Count locally to reduce atomic contention
        for (Index k = start; k < end; ++k) {
            if (mask[matrix.indices[k]]) {
                local_count++;
            }
        }
        
        if (local_count > 0) {
            total_nnz.fetch_add(local_count, std::memory_order_relaxed);
        }
    });
    
    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Ultra-fast secondary dimension filter materialization (CustomSparse)
///
/// Optimization: Parallel filtering with branch-free writes
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void materialize_filter_secondary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    // First pass: compute output indptr (must be sequential for prefix sum)
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index count = 0;
        
        for (Index k = start; k < end; ++k) {
            if (mask[matrix.indices[k]]) {
                count++;
            }
        }
        
        out_indptr[p + 1] = out_indptr[p] + count;
    }
    
    // Second pass: parallel copy filtered data
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index src_start = matrix.indptr[p];
        Index src_end = matrix.indptr[p + 1];
        Index dst_pos = out_indptr[p];
        
        for (Index k = src_start; k < src_end; ++k) {
            Index old_idx = matrix.indices[k];
            if (mask[old_idx]) {
                out_data[dst_pos] = matrix.data[k];
                out_indices[dst_pos] = new_indices[old_idx];
                dst_pos++;
            }
        }
    });
}

/// @brief Ultra-fast secondary dimension filter (CustomSparse) - allocating
///
/// Optimization: Parallel inspection + parallel materialization
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE SliceResult<T> filter_secondary_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = scl::secondary_size(matrix);
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(secondary_dim),
                  "Filter: Mask size mismatch");
    
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
    Index out_nnz = inspect_filter_secondary_custom_fast(matrix, mask);
    
    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);
    
    // Fast materialization
    materialize_filter_secondary_custom_fast(
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
// VirtualSparse Fast Path - Primary Dimension Slicing
// =============================================================================

/// @brief Ultra-fast primary dimension slice inspection (VirtualSparse)
///
/// Optimization: Parallel length accumulation with minimal indirection
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE Index inspect_slice_primary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    
    if (n_keep == 0) return 0;
    
    // For small slices, use sequential
    if (n_keep < 1000) {
        Index total_nnz = 0;
        for (Size i = 0; i < n_keep; ++i) {
            Index idx = keep_indices[i];
            SCL_CHECK_ARG(idx >= 0 && idx < scl::primary_size(matrix),
                          "Slice: Index out of bounds");
            total_nnz += matrix.lengths[idx];
        }
        return total_nnz;
    }
    
    // For large slices, use parallel reduction
    std::atomic<Index> total_nnz{0};
    
    scl::threading::parallel_for(0, n_keep, [&](size_t i) {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < scl::primary_size(matrix),
                      "Slice: Index out of bounds");
        Index len = matrix.lengths[idx];
        total_nnz.fetch_add(len, std::memory_order_relaxed);
    });
    
    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Ultra-fast primary dimension slice materialization (VirtualSparse)
///
/// Optimization: Parallel row copy with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void materialize_slice_primary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.size();
    
    SCL_CHECK_DIM(out_indptr.size() >= n_keep + 1,
                  "Slice: Output indptr too small");
    
    // Build indptr first
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index len = matrix.lengths[src_idx];
        out_indptr[i + 1] = out_indptr[i] + len;
    }
    
    // Parallel copy data and indices
    scl::threading::parallel_for(0, n_keep, [&](size_t i) {
        Index src_idx = keep_indices[i];
        Index len = matrix.lengths[src_idx];
        
        if (len == 0) return;
        
        Index dst_start = out_indptr[i];
        
        // Single pointer dereference per row
        const T* SCL_RESTRICT src_data = static_cast<const T*>(matrix.data_ptrs[src_idx]);
        const Index* SCL_RESTRICT src_indices = static_cast<const Index*>(matrix.indices_ptrs[src_idx]);
        
        // Bulk copy using memcpy
        if (len >= 8) {
            std::memcpy(out_data.data() + dst_start, src_data, len * sizeof(T));
            std::memcpy(out_indices.data() + dst_start, src_indices, len * sizeof(Index));
        } else {
            // Small segments: manual copy
            for (Index k = 0; k < len; ++k) {
                out_data[dst_start + k] = src_data[k];
                out_indices[dst_start + k] = src_indices[k];
            }
        }
    });
}

/// @brief Ultra-fast primary dimension slice (VirtualSparse) - allocating
///
/// Optimization: Parallel inspection + parallel materialization
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE SliceResult<T> slice_primary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    
    // Fast inspection
    Index out_nnz = inspect_slice_primary_virtual_fast(matrix, keep_indices);
    
    // Allocate output arrays
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(n_keep + 1);
    
    // Fast materialization
    materialize_slice_primary_virtual_fast(
        matrix,
        keep_indices,
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );
    
    // Build result and release ownership
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
// VirtualSparse Fast Path - Secondary Dimension Filtering
// =============================================================================

/// @brief Ultra-fast secondary dimension filter inspection (VirtualSparse)
///
/// Optimization: Parallel mask checking with minimal indirection
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE Index inspect_filter_secondary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    // For small matrices, use sequential
    if (primary_dim < 500) {
        Index total_nnz = 0;
        for (Index p = 0; p < primary_dim; ++p) {
            Index len = matrix.lengths[p];
            const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[p]);
            
            for (Index k = 0; k < len; ++k) {
                if (mask[indices[k]]) {
                    total_nnz++;
                }
            }
        }
        return total_nnz;
    }
    
    // For large matrices, use parallel reduction
    std::atomic<Index> total_nnz{0};
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[p]);
        Index local_count = 0;
        
        for (Index k = 0; k < len; ++k) {
            if (mask[indices[k]]) {
                local_count++;
            }
        }
        
        if (local_count > 0) {
            total_nnz.fetch_add(local_count, std::memory_order_relaxed);
        }
    });
    
    return total_nnz.load(std::memory_order_relaxed);
}

/// @brief Ultra-fast secondary dimension filter materialization (VirtualSparse)
///
/// Optimization: Parallel filtering with minimal indirection
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void materialize_filter_secondary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    // First pass: compute output indptr
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index len = matrix.lengths[p];
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[p]);
        Index count = 0;
        
        for (Index k = 0; k < len; ++k) {
            if (mask[indices[k]]) {
                count++;
            }
        }
        
        out_indptr[p + 1] = out_indptr[p] + count;
    }
    
    // Second pass: parallel copy filtered data
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        const T* SCL_RESTRICT src_data = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* SCL_RESTRICT src_indices = static_cast<const Index*>(matrix.indices_ptrs[p]);
        Index dst_pos = out_indptr[p];
        
        for (Index k = 0; k < len; ++k) {
            Index old_idx = src_indices[k];
            if (mask[old_idx]) {
                out_data[dst_pos] = src_data[k];
                out_indices[dst_pos] = new_indices[old_idx];
                dst_pos++;
            }
        }
    });
}

/// @brief Ultra-fast secondary dimension filter (VirtualSparse) - allocating
///
/// Optimization: Parallel inspection + parallel materialization
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE SliceResult<T> filter_secondary_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = scl::secondary_size(matrix);
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(secondary_dim),
                  "Filter: Mask size mismatch");
    
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
    Index out_nnz = inspect_filter_secondary_virtual_fast(matrix, mask);
    
    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);
    
    // Fast materialization
    materialize_filter_secondary_virtual_fast(
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
// Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch primary slice to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE SliceResult<typename MatrixT::ValueType> slice_primary_fast(
    const MatrixT& matrix,
    Array<const Index> keep_indices
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        return slice_primary_custom_fast(matrix, keep_indices);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        return slice_primary_virtual_fast(matrix, keep_indices);
    }
}

/// @brief Auto-dispatch secondary filter to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE SliceResult<typename MatrixT::ValueType> filter_secondary_fast(
    const MatrixT& matrix,
    Array<const uint8_t> mask
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        return filter_secondary_custom_fast(matrix, mask);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        return filter_secondary_virtual_fast(matrix, mask);
    }
}

// =============================================================================
// Convenience Wrappers
// =============================================================================

/// @brief Fast row slicing for CSR matrices
template <typename T>
SCL_FORCE_INLINE SliceResult<T> slice_rows_fast(
    const CustomSparse<T, true>& matrix,
    Array<const Index> row_indices
) {
    return slice_primary_custom_fast(matrix, row_indices);
}

/// @brief Fast column slicing for CSC matrices
template <typename T>
SCL_FORCE_INLINE SliceResult<T> slice_cols_fast(
    const CustomSparse<T, false>& matrix,
    Array<const Index> col_indices
) {
    return slice_primary_custom_fast(matrix, col_indices);
}

/// @brief Fast column filtering for CSR matrices
template <typename T>
SCL_FORCE_INLINE SliceResult<T> filter_cols_fast(
    const CustomSparse<T, true>& matrix,
    Array<const uint8_t> mask
) {
    return filter_secondary_custom_fast(matrix, mask);
}

/// @brief Fast row filtering for CSC matrices
template <typename T>
SCL_FORCE_INLINE SliceResult<T> filter_rows_fast(
    const CustomSparse<T, false>& matrix,
    Array<const uint8_t> mask
) {
    return filter_secondary_custom_fast(matrix, mask);
}

} // namespace scl::kernel::slice::fast
