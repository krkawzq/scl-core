#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>

// =============================================================================
/// @file reorder.hpp
/// @brief Sparse Matrix Alignment and Reordering
///
/// Implements in-place column/row reordering for sparse matrices.
/// Used for gene alignment and feature permutation.
///
/// Performance: O(nnz log nnz_per_row), in-place
// =============================================================================

namespace scl::kernel::reorder {

/// @brief Align secondary dimension using index map (unified for CSR/CSC)
///
/// For CSR: Aligns columns (genes)
/// For CSC: Aligns rows (cells)
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param index_map Mapping from old to new indices [size = old_secondary_dim]
/// @param out_lengths Output valid lengths [size = primary_dim]
/// @param new_secondary_dim New secondary dimension size
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void align_secondary(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_lengths.size() >= static_cast<Size>(primary_dim),
                  "Reorder: Output lengths too small");
    
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
        
        // Remap and filter
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
        
        // Sort by new indices
        if (write_pos > 1) {
            scl::sort::sort_pairs(
                Array<Index>(inds, write_pos),
                Array<T>(vals, write_pos)
            );
        }
        
        out_lengths[p] = write_pos;
    });
    
    // Update matrix dimensions
    if constexpr (IsCSR) {
        matrix.cols = new_secondary_dim;
    } else {
        matrix.rows = new_secondary_dim;
    }
}

} // namespace scl::kernel::reorder
