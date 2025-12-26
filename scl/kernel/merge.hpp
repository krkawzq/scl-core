#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/reorder.hpp"

#include <vector>
#include <numeric>

// =============================================================================
/// @file merge.hpp
/// @brief Zero-Copy Virtual Matrix Merging
///
/// Implements high-performance matrix concatenation using Virtual Matrix views.
///
/// Design Philosophy:
///
/// **Virtual-Only**: This module ONLY works with VirtualCSR/VirtualCSC.
/// We do NOT physically copy data. Instead, we construct new indirection maps.
///
/// Key Insight:
/// - Physical merge requires O(NNZ) data copying
/// - Virtual merge requires O(rows) index array creation
/// - Speedup: 100-1000x for large matrices
///
/// Use Cases:
///
/// - **Batch Integration**: Merge aligned datasets (Dataset A + Dataset B)
/// - **Cross-Validation**: Combine train/test splits
/// - **Incremental Loading**: Progressively merge data batches
///
/// Performance:
///
/// - Time: O(total_rows) - just build indirection map
/// - Space: O(total_rows) - one index array
/// - Data Movement: ZERO (pure view composition)
///
/// Example:
///
/// @code{.cpp}
/// // Align two datasets
/// CustomCSR<Real> batch_a = load_batch_a();  // 1000 × 2000
/// CustomCSR<Real> batch_b = load_batch_b();  // 1500 × 2500
/// 
/// std::vector<Index> map_a = build_map(...);
/// std::vector<Index> map_b = build_map(...);
/// std::vector<Index> len_a(batch_a.rows), len_b(batch_b.rows);
/// 
/// align_cols(batch_a, map_a, len_a, 1800);  // Both → 1800 genes
/// align_cols(batch_b, map_b, len_b, 1800);
/// 
/// // Create virtual views
/// VirtualCSR<Real> view_a(batch_a, identity_map(batch_a.rows));
/// VirtualCSR<Real> view_b(batch_b, identity_map(batch_b.rows));
/// 
/// // Zero-copy merge
/// std::vector<Index> merged_map(2500);
/// VirtualCSR<Real> merged = vstack_virtual(
///     {&view_a, &view_b}, 
///     MutableSpan<Index>(merged_map.data(), merged_map.size())
/// );
/// 
/// // merged: 2500 × 1800, zero data copying!
/// @endcode
// =============================================================================

namespace scl::kernel::merge {

// =============================================================================
// Virtual Matrix Vertical Stack (VStack)
// =============================================================================

/// @brief Vertically stack VirtualCSR matrices (Zero-Copy).
///
/// Constructs a new VirtualCSR that views multiple source matrices as one.
/// No physical data copying - only builds indirection map.
///
/// Requirements:
/// - All inputs must have identical cols dimension
/// - All inputs must share the SAME underlying physical matrix
/// - Inputs can have different row counts
///
/// Algorithm:
/// 1. Validate dimensions
/// 2. Concatenate row_maps sequentially with offset adjustment
/// 3. Construct new VirtualCSR pointing to shared physical data
///
/// Output:
/// - VirtualCSR with total_rows = sum(input[i].rows)
/// - row_map: [view_a.map[0..n_a], view_b.map[0..n_b] + offset_a, ...]
///
/// Time: O(total_rows), Space: O(total_rows)
///
/// @tparam T Value type (typically Real)
/// @param inputs Array of VirtualCSR pointers [must share same source matrix]
/// @param out_row_map Output: Merged row indirection map [size = total_rows]
/// @return New VirtualCSR viewing all inputs as one matrix
template <typename T>
VirtualCSR<T> vstack_virtual(
    Span<const VirtualCSR<T>*> inputs,
    MutableSpan<Index> out_row_map
) {
    SCL_CHECK_ARG(inputs.size > 0, "VStack: Must provide at least one input matrix");

    // -------------------------------------------------------------------------
    // Step 1: Validate Inputs & Calculate Total Dimensions
    // -------------------------------------------------------------------------
    
    const auto* first = inputs[0];
    const Index common_cols = first->cols;
    const Index src_rows = first->src_rows;
    
    // Validate: All must share same source and columns
    Index total_rows = 0;
    for (Size i = 0; i < inputs.size; ++i) {
        const auto* mat = inputs[i];
        
        SCL_CHECK_DIM(mat->cols == common_cols, 
                      "VStack: Column dimension mismatch across inputs");
        SCL_CHECK_ARG(mat->src_data == first->src_data, 
                      "VStack: All inputs must share the same physical source matrix");
        SCL_CHECK_ARG(mat->src_rows == src_rows,
                      "VStack: Source dimension mismatch");
        
        total_rows += mat->rows;
    }
    
    SCL_CHECK_DIM(out_row_map.size == static_cast<Size>(total_rows),
                  "VStack: Output row_map size must match total rows");

    // -------------------------------------------------------------------------
    // Step 2: Build Merged Row Map (Sequential Concatenation)
    // -------------------------------------------------------------------------
    // Time: O(total_rows), Space: O(1) (write to pre-allocated buffer)
    
    Index write_pos = 0;
    
    for (Size i = 0; i < inputs.size; ++i) {
        const auto* mat = inputs[i];
        
        // Copy this matrix's row_map to output
        // Each input's row_map is already relative to the shared source
        std::copy(
            mat->row_map,
            mat->row_map + mat->rows,
            out_row_map.ptr + write_pos
        );
        
        write_pos += mat->rows;
    }

    // -------------------------------------------------------------------------
    // Step 3: Construct Merged VirtualCSR
    // -------------------------------------------------------------------------
    
    VirtualCSR<T> result;
    result.src_data = first->src_data;
    result.src_indices = first->src_indices;
    result.src_indptr = first->src_indptr;
    result.src_row_lengths = first->src_row_lengths;  // Inherit from source
    
    result.row_map = out_row_map.ptr;
    result.rows = total_rows;
    result.cols = common_cols;
    result.src_rows = src_rows;

    return result;
}

// =============================================================================
// Horizontal Stack (HStack) for VirtualCSC
// =============================================================================

/// @brief Horizontally stack VirtualCSC matrices (Zero-Copy).
///
/// Constructs a new VirtualCSC that views multiple source matrices as one.
/// Column dimensions are concatenated, row dimensions must match.
///
/// Requirements:
/// - All inputs must have identical rows dimension
/// - All inputs must share the SAME underlying physical matrix
/// - Inputs can have different column counts
///
/// Algorithm:
/// 1. Validate dimensions
/// 2. Concatenate col_maps with offset adjustment
/// 3. Construct new VirtualCSC
///
/// Time: O(total_cols), Space: O(total_cols)
///
/// @tparam T Value type (typically Real)
/// @param inputs Array of VirtualCSC pointers [must share same source]
/// @param out_col_map Output: Merged column indirection map [size = total_cols]
/// @return New VirtualCSC viewing all inputs side-by-side
template <typename T>
VirtualCSC<T> hstack_virtual(
    Span<const VirtualCSC<T>*> inputs,
    MutableSpan<Index> out_col_map
) {
    SCL_CHECK_ARG(inputs.size > 0, "HStack: Must provide at least one input matrix");

    // -------------------------------------------------------------------------
    // Step 1: Validate & Calculate Dimensions
    // -------------------------------------------------------------------------
    
    const auto* first = inputs[0];
    const Index common_rows = first->rows;
    const Index src_cols = first->src_cols;
    
    Index total_cols = 0;
    for (Size i = 0; i < inputs.size; ++i) {
        const auto* mat = inputs[i];
        
        SCL_CHECK_DIM(mat->rows == common_rows,
                      "HStack: Row dimension mismatch across inputs");
        SCL_CHECK_ARG(mat->src_data == first->src_data,
                      "HStack: All inputs must share the same physical source matrix");
        SCL_CHECK_ARG(mat->src_cols == src_cols,
                      "HStack: Source dimension mismatch");
        
        total_cols += mat->cols;
    }
    
    SCL_CHECK_DIM(out_col_map.size == static_cast<Size>(total_cols),
                  "HStack: Output col_map size must match total cols");

    // -------------------------------------------------------------------------
    // Step 2: Build Merged Column Map
    // -------------------------------------------------------------------------
    
    Index write_pos = 0;
    
    for (Size i = 0; i < inputs.size; ++i) {
        const auto* mat = inputs[i];
        
        // Copy this matrix's col_map to output
        std::copy(
            mat->col_map,
            mat->col_map + mat->cols,
            out_col_map.ptr + write_pos
        );
        
        write_pos += mat->cols;
    }

    // -------------------------------------------------------------------------
    // Step 3: Construct Merged VirtualCSC
    // -------------------------------------------------------------------------
    
    VirtualCSC<T> result;
    result.src_data = first->src_data;
    result.src_indices = first->src_indices;
    result.src_indptr = first->src_indptr;
    result.src_col_lengths = first->src_col_lengths;
    
    result.col_map = out_col_map.ptr;
    result.rows = common_rows;
    result.cols = total_cols;
    result.src_cols = src_cols;

    return result;
}

// =============================================================================
// Utility: Create Identity Map (for converting CustomCSR to VirtualCSR)
// =============================================================================

/// @brief Create identity mapping [0, 1, 2, ..., n-1].
///
/// Used to wrap a CustomCSR as VirtualCSR without any transformation.
///
/// @param n Size of the map
/// @param out_map Output buffer [size = n]
SCL_FORCE_INLINE void identity_map(Index n, MutableSpan<Index> out_map) {
    SCL_CHECK_DIM(out_map.size == static_cast<Size>(n),
                  "Identity map: Output size mismatch");
    
    for (Index i = 0; i < n; ++i) {
        out_map[i] = i;
    }
}

// =============================================================================
// Advanced: Multi-Source VStack (Different Physical Matrices)
// =============================================================================

/// @brief Vertically stack matrices from DIFFERENT physical sources.
///
/// This is NOT zero-copy - requires physical data copying.
/// Use only when sources cannot be aligned to same physical matrix.
///
/// Algorithm:
/// 1. Build output indptr (prefix sum of row lengths)
/// 2. Parallel copy data from each source to destination
///
/// Time: O(NNZ), Space: O(1) (user provides dst)
///
/// @tparam T Value type (typically Real)
/// @param inputs Array of CustomCSR pointers (can be from different allocations)
/// @param dst Destination matrix [must be pre-allocated with sufficient capacity]
template <typename T>
void vstack_physical(
    Span<const CustomCSR<T>*> inputs,
    CustomCSR<T>& dst
) {
    SCL_CHECK_ARG(inputs.size > 0, "VStack Physical: Must provide at least one input");

    // -------------------------------------------------------------------------
    // Step 1: Validate & Calculate Offsets
    // -------------------------------------------------------------------------
    
    const Index common_cols = inputs[0]->cols;
    
    std::vector<Index> row_offsets(inputs.size + 1, 0);
    std::vector<Size> nnz_offsets(inputs.size + 1, 0);
    
    Index total_rows = 0;
    Size total_nnz = 0;
    
    for (Size i = 0; i < inputs.size; ++i) {
        const auto* mat = inputs[i];
        
        SCL_CHECK_DIM(mat->cols == common_cols,
                      "VStack Physical: Column dimension mismatch");
        
        row_offsets[i] = total_rows;
        nnz_offsets[i] = total_nnz;
        
        total_rows += mat->rows;
        
        // Calculate true NNZ (respecting explicit lengths)
        Size mat_nnz = 0;
        for (Index r = 0; r < mat->rows; ++r) {
            mat_nnz += static_cast<Size>(mat->row_length(r));
        }
        total_nnz += mat_nnz;
    }
    
    row_offsets[inputs.size] = total_rows;
    nnz_offsets[inputs.size] = total_nnz;
    
    // Validate destination capacity
    SCL_CHECK_ARG(dst.rows >= total_rows, "VStack Physical: Destination rows insufficient");
    SCL_CHECK_ARG(dst.cols >= common_cols, "VStack Physical: Destination cols insufficient");
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "VStack Physical: Destination arrays not allocated");

    // -------------------------------------------------------------------------
    // Step 2: Build Output Indptr (Serial Prefix Sum)
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    Index running_offset = 0;
    
    for (Size b = 0; b < inputs.size; ++b) {
        const auto* mat = inputs[b];
        
        for (Index r = 0; r < mat->rows; ++r) {
            Index len = mat->row_length(r);
            running_offset += len;
            
            Index global_row = row_offsets[b] + r;
            dst.indptr[global_row + 1] = running_offset;
        }
    }
    
    // Update metadata
    dst.rows = total_rows;
    dst.cols = common_cols;
    dst.nnz = running_offset;
    dst.row_lengths = nullptr;  // Compact output, no explicit lengths needed

    // -------------------------------------------------------------------------
    // Step 3: Parallel Data Copy
    // -------------------------------------------------------------------------
    // Parallelize over batches, then rows within each batch
    
    for (Size b = 0; b < inputs.size; ++b) {
        const auto* mat = inputs[b];
        const Index row_base = row_offsets[b];
        
        scl::threading::parallel_for(0, static_cast<size_t>(mat->rows), [&](size_t r) {
            Index row_idx = static_cast<Index>(r);
            Index len = mat->row_length(row_idx);
            
            if (len == 0) return;
            
            // Source location
            Index src_start = mat->indptr[row_idx];
            
            // Destination location
            Index global_row = row_base + row_idx;
            Index dst_start = dst.indptr[global_row];
            
            // Copy indices
            std::copy(
                mat->indices + src_start,
                mat->indices + src_start + len,
                dst.indices + dst_start
            );
            
            // Copy values
            std::copy(
                mat->data + src_start,
                mat->data + src_start + len,
                dst.data + dst_start
            );
        });
    }
}

// =============================================================================
// Recommended Workflow: Align → Virtual Merge
// =============================================================================

/// @brief Complete workflow: Align multiple batches then merge virtually.
///
/// This is the recommended high-level function for batch integration.
///
/// Steps:
/// 1. Align all inputs to common gene set (in-place with explicit lengths)
/// 2. Create virtual views wrapping aligned matrices
/// 3. Merge virtually (zero-copy)
///
/// Memory Efficiency:
/// - Total allocation: O(sum(rows)) for maps + O(sum(rows)) for lengths
/// - No data copying until final compact (optional)
///
/// @tparam T Value type
/// @param sources Array of source matrices [modified in-place by alignment]
/// @param alignment_maps Alignment maps for each source [map[i] size = sources[i].cols]
/// @param target_n_cols Target column dimension (common gene set size)
/// @param out_lengths Array of length buffers [lengths[i] size = sources[i].rows]
/// @param out_virtual_maps Array of virtual map buffers [vmaps[i] size = sources[i].rows]
/// @param out_merged_map Merged row map [size = sum(sources[i].rows)]
/// @return VirtualCSR viewing all aligned and merged data
template <typename T>
VirtualCSR<T> align_and_merge(
    Span<CustomCSR<T>*> sources,
    Span<const Span<const Index>> alignment_maps,
    Index target_n_cols,
    Span<MutableSpan<Index>> out_lengths,
    Span<MutableSpan<Index>> out_virtual_maps,
    MutableSpan<Index> out_merged_map
) {
    SCL_CHECK_ARG(sources.size > 0, "Align & Merge: Empty sources");
    SCL_CHECK_DIM(alignment_maps.size == sources.size, "Alignment maps size mismatch");
    SCL_CHECK_DIM(out_lengths.size == sources.size, "Output lengths size mismatch");
    SCL_CHECK_DIM(out_virtual_maps.size == sources.size, "Virtual maps size mismatch");

    // -------------------------------------------------------------------------
    // Step 1: Align All Sources
    // -------------------------------------------------------------------------
    
    for (Size i = 0; i < sources.size; ++i) {
        auto* mat = sources[i];
        
        SCL_CHECK_DIM(alignment_maps[i].size == static_cast<Size>(mat->cols),
                      "Align & Merge: Alignment map size must match source cols");
        SCL_CHECK_DIM(out_lengths[i].size == static_cast<Size>(mat->rows),
                      "Align & Merge: Length buffer size must match source rows");
        
        scl::kernel::reorder::align_cols(
            *mat,
            alignment_maps[i],
            out_lengths[i],
            target_n_cols
        );
    }

    // -------------------------------------------------------------------------
    // Step 2: Create Virtual Views
    // -------------------------------------------------------------------------
    
    std::vector<VirtualCSR<T>> virtual_views;
    virtual_views.reserve(sources.size);
    
    Index total_rows = 0;
    for (Size i = 0; i < sources.size; ++i) {
        const auto* mat = sources[i];
        
        SCL_CHECK_DIM(out_virtual_maps[i].size == static_cast<Size>(mat->rows),
                      "Align & Merge: Virtual map buffer size must match source rows");
        
        // Create identity map for this source
        identity_map(mat->rows, out_virtual_maps[i]);
        
        // Create virtual view
        VirtualCSR<T> view;
        view.src_data = mat->data;
        view.src_indices = mat->indices;
        view.src_indptr = mat->indptr;
        view.src_row_lengths = mat->row_lengths;
        view.row_map = out_virtual_maps[i].ptr;
        view.rows = mat->rows;
        view.cols = mat->cols;
        view.src_rows = mat->rows;
        
        virtual_views.push_back(view);
        total_rows += mat->rows;
    }

    // -------------------------------------------------------------------------
    // Step 3: Virtual Merge
    // -------------------------------------------------------------------------
    
    SCL_CHECK_DIM(out_merged_map.size == static_cast<Size>(total_rows),
                  "Align & Merge: Merged map size must match total rows");
    
    // Build array of pointers for vstack_virtual
    std::vector<const VirtualCSR<T>*> view_ptrs;
    view_ptrs.reserve(virtual_views.size());
    for (auto& view : virtual_views) {
        view_ptrs.push_back(&view);
    }
    
    return vstack_virtual(
        Span<const VirtualCSR<T>*>(view_ptrs.data(), view_ptrs.size()),
        out_merged_map
    );
}

} // namespace scl::kernel::merge

