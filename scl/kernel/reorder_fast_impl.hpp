#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file reorder_fast_impl.hpp
/// @brief Extreme Performance Reordering
///
/// Separate optimizations:
/// - CustomSparse: Direct pointer manipulation + parallel VQSort
/// - VirtualSparse: Row-wise remapping with minimal indirection
///
/// Ultra-optimized column/row alignment with:
/// - In-place remapping with minimal copies
/// - Parallel sorting with VQSort
/// - Cache-friendly scatter pattern
///
/// Performance Target: 1.5-2x faster than generic
// =============================================================================

namespace scl::kernel::reorder::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast secondary dimension alignment (CustomSparse)
///
/// Optimization: Direct pointer manipulation + parallel VQSort
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void align_secondary_custom_fast(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_lengths.len >= static_cast<Size>(primary_dim),
                  "Output lengths too small");
    
    // Parallel remapping and sorting
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) {
            out_lengths[p] = 0;
            return;
        }
        
        Index* inds = matrix.indices + start;
        T* vals = matrix.data + start;
        
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
        
        // Sort using VQSort (ultra-fast)
        if (write_pos > 1) {
            scl::sort::sort_pairs(
                Array<Index>(inds, write_pos),
                Array<T>(vals, write_pos)
            );
        }
        
        out_lengths[p] = write_pos;
    });
    
    // Update dimensions
    if constexpr (IsCSR) {
        matrix.cols = new_secondary_dim;
    } else {
        matrix.rows = new_secondary_dim;
    }
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast secondary dimension alignment (VirtualSparse)
///
/// Optimization: Row-wise remapping with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void align_secondary_virtual_fast(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_lengths.len >= static_cast<Size>(primary_dim),
                  "Output lengths too small");
    
    // Parallel remapping and sorting
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        if (len == 0) {
            out_lengths[p] = 0;
            return;
        }
        
        // Single pointer dereference
        Index* inds = static_cast<Index*>(matrix.indices_ptrs[p]);
        T* vals = static_cast<T*>(matrix.data_ptrs[p]);
        
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
        
        // Sort using VQSort (ultra-fast)
        if (write_pos > 1) {
            scl::sort::sort_pairs(
                Array<Index>(inds, write_pos),
                Array<T>(vals, write_pos)
            );
        }
        
        out_lengths[p] = write_pos;
        matrix.lengths[p] = write_pos;
    });
    
    // Update dimensions
    if constexpr (IsCSR) {
        matrix.cols = new_secondary_dim;
    } else {
        matrix.rows = new_secondary_dim;
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void align_secondary_fast(
    MatrixT& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        align_secondary_custom_fast(matrix, index_map, out_lengths, new_secondary_dim);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        align_secondary_virtual_fast(matrix, index_map, out_lengths, new_secondary_dim);
    }
}

} // namespace scl::kernel::reorder::fast

