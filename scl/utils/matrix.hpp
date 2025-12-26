#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <algorithm>

// =============================================================================
/// @file utils/matrix.hpp
/// @brief Matrix Utility Functions
///
/// Provides:
/// - NNZ counting (parallel reduction)
/// - Materialization (compact copy)
/// - Format conversion (CSR <-> CSC)
/// - Slicing operations
///
/// All functions unified for CSR/CSC using AnySparse
// =============================================================================

namespace scl::utils {

namespace detail {

/// @brief Cache-aligned accumulator (avoid false sharing)
template <typename T>
struct alignas(64) Accumulator {
    T value;
    char padding[64 - sizeof(T)];
    
    Accumulator() : value(0) {}
};

/// @brief Parallel memory copy using scl::memory
template <typename T>
inline void parallel_copy(const T* src, T* dst, Size count) {
    Array<const T> src_arr(src, count);
    Array<T> dst_arr(dst, count);
    scl::memory::copy(src_arr, dst_arr);
}

} // namespace detail

// =============================================================================
// NNZ Counting (Parallel Reduction)
// =============================================================================

/// @brief Count total active NNZ (unified for CSR/CSC)
///
/// For virtual/filtered matrices, performs parallel summation of lengths.
///
/// @param mat Input sparse matrix
/// @return Total number of active non-zero elements
template <typename MatrixT>
    requires AnySparse<MatrixT>
Size count_active_nnz(const MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    if (primary_dim == 0) return 0;

    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    const size_t chunk_size = (static_cast<size_t>(primary_dim) + num_threads - 1) / num_threads;

    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(primary_dim));
        
        Size local_sum = 0;
        
        for (size_t i = start; i < end; ++i) {
            local_sum += static_cast<Size>(scl::primary_length(mat, static_cast<Index>(i)));
        }
        
        partial_sums[t_id].value = local_sum;
    });

    Size total_nnz = 0;
    for (const auto& acc : partial_sums) {
        total_nnz += acc.value;
    }
    
    return total_nnz;
}

// =============================================================================
// Materialization (Compact Copy)
// =============================================================================

/// @brief Materialize sparse view to CustomSparse (unified for CSR/CSC)
///
/// Compacts any source view into pre-allocated CustomSparse with contiguous storage.
///
/// Algorithm:
/// 1. Rebuild indptr via prefix sum
/// 2. Parallel copy of data and indices
///
/// @param src Source sparse matrix (any type)
/// @param dst Destination CustomSparse [pre-allocated]
template <typename SrcT, typename T, bool IsCSR>
    requires AnySparse<SrcT>
void compact_copy(const SrcT& src, CustomSparse<T, IsCSR>& dst) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>, 
                  "Source and destination value types must match");
    
    const Index primary_dim = scl::primary_size(src);
    
    SCL_CHECK_ARG(dst.data != nullptr, "Destination data not allocated");
    SCL_CHECK_ARG(dst.indices != nullptr, "Destination indices not allocated");
    SCL_CHECK_ARG(dst.indptr != nullptr, "Destination indptr not allocated");
    SCL_CHECK_DIM(scl::rows(dst) == scl::rows(src), "Row dimension mismatch");
    SCL_CHECK_DIM(scl::cols(dst) == scl::cols(src), "Column dimension mismatch");

    // Build indptr (prefix sum)
    dst.indptr[0] = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Index len = scl::primary_length(src, i);
        dst.indptr[i + 1] = dst.indptr[i] + len;
    }
    
    const Size required_nnz = static_cast<Size>(dst.indptr[primary_dim]);
    SCL_CHECK_ARG(required_nnz <= static_cast<Size>(dst.nnz()), 
                  "Destination capacity insufficient");

    // Parallel data copy
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        
        auto src_vals = scl::primary_values(src, idx);
        auto src_inds = scl::primary_indices(src, idx);
        Index len = scl::primary_length(src, idx);
        
        if (len == 0) return;

        Index dst_start = dst.indptr[idx];
        
        scl::memory::copy_fast(
            Array<const T>(src_vals.ptr, static_cast<Size>(len)),
            Array<T>(dst.data + dst_start, static_cast<Size>(len))
        );
        scl::memory::copy_fast(
            Array<const Index>(src_inds.ptr, static_cast<Size>(len)),
            Array<Index>(dst.indices + dst_start, static_cast<Size>(len))
        );
    });
}

// =============================================================================
// Format Conversion (CSR <-> CSC)
// =============================================================================

/// @brief Convert CSR to CSC (transpose)
///
/// @param src Source CSR matrix
/// @param dst Destination CSC matrix [pre-allocated]
/// @param workspace Temporary buffer [size >= secondary_dim]
template <typename SrcT, typename T>
    requires SparseLike<SrcT, true>  // CSR source
void convert_to_csc(
    const SrcT& src,
    CustomSparse<T, false>& dst,
    Array<Index> workspace
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>, "Type mismatch");
    
    const Index rows = scl::rows(src);
    const Index cols = scl::cols(src);
    
    SCL_CHECK_DIM(workspace.len >= static_cast<Size>(cols), "Workspace too small");
    SCL_CHECK_DIM(scl::rows(dst) == cols, "CSC rows must equal CSR cols");
    SCL_CHECK_DIM(scl::cols(dst) == rows, "CSC cols must equal CSR rows");

    // Histogram: Count NNZ per column
    std::fill(workspace.ptr, workspace.ptr + cols, 0);
    
    for (Index i = 0; i < rows; ++i) {
        auto row_inds = scl::primary_indices(src, i);
        Index row_len = scl::primary_length(src, i);
        
        for (Index k = 0; k < row_len; ++k) {
            workspace[row_inds[k]]++;
        }
    }

    // Build CSC indptr
    dst.indptr[0] = 0;
    for (Index j = 0; j < cols; ++j) {
        dst.indptr[j + 1] = dst.indptr[j] + workspace[j];
    }

    // Prepare write heads
    std::copy(dst.indptr, dst.indptr + cols, workspace.ptr);

    // Scatter fill
    for (Index i = 0; i < rows; ++i) {
        auto row_inds = scl::primary_indices(src, i);
        auto row_vals = scl::primary_values(src, i);
        Index row_len = scl::primary_length(src, i);

        for (Index k = 0; k < row_len; ++k) {
            Index col = row_inds[k];
            T val = row_vals[k];

            Index write_pos = workspace[col];
            workspace[col]++;
            
            dst.indices[write_pos] = i;
            dst.data[write_pos] = val;
        }
    }
}

/// @brief Convert CSC to CSR (transpose)
///
/// @param src Source CSC matrix
/// @param dst Destination CSR matrix [pre-allocated]
/// @param workspace Temporary buffer [size >= secondary_dim]
template <typename SrcT, typename T>
    requires SparseLike<SrcT, false>  // CSC source
void convert_to_csr(
    const SrcT& src,
    CustomSparse<T, true>& dst,
    Array<Index> workspace
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>, "Type mismatch");
    
    const Index rows = scl::rows(src);
    const Index cols = scl::cols(src);
    
    SCL_CHECK_DIM(workspace.len >= static_cast<Size>(rows), "Workspace too small");
    SCL_CHECK_DIM(scl::rows(dst) == cols, "CSR rows must equal CSC cols");
    SCL_CHECK_DIM(scl::cols(dst) == rows, "CSR cols must equal CSC rows");

    // Histogram
    std::fill(workspace.ptr, workspace.ptr + rows, 0);
    
    for (Index j = 0; j < cols; ++j) {
        auto col_inds = scl::primary_indices(src, j);
        Index col_len = scl::primary_length(src, j);
        
        for (Index k = 0; k < col_len; ++k) {
            workspace[col_inds[k]]++;
        }
    }

    // Build CSR indptr
    dst.indptr[0] = 0;
    for (Index i = 0; i < rows; ++i) {
        dst.indptr[i + 1] = dst.indptr[i] + workspace[i];
    }

    // Prepare write heads
    std::copy(dst.indptr, dst.indptr + rows, workspace.ptr);

    // Scatter fill
    for (Index j = 0; j < cols; ++j) {
        auto col_inds = scl::primary_indices(src, j);
        auto col_vals = scl::primary_values(src, j);
        Index col_len = scl::primary_length(src, j);

        for (Index k = 0; k < col_len; ++k) {
            Index row = col_inds[k];
            T val = col_vals[k];

            Index write_pos = workspace[row];
            workspace[row]++;
            
            dst.indices[write_pos] = j;
            dst.data[write_pos] = val;
        }
    }
}

// =============================================================================
// Slicing Operations
// =============================================================================

/// @brief Slice primary dimension (zero-copy via VirtualSparse)
///
/// For CSR: Row slicing (zero-copy)
/// For CSC: Column slicing (zero-copy)
///
/// @param src Source CustomSparse
/// @param indices Indices to keep
/// @return VirtualSparse view
template <typename T, bool IsCSR>
VirtualSparse<T, IsCSR> slice_primary_virtual(
    const CustomSparse<T, IsCSR>& src,
    Array<const Index> indices
) {
    // Create virtual view with indirection
    // Note: Simplified - full implementation needs proper VirtualSparse construction
    VirtualSparse<T, IsCSR> result;
    if constexpr (IsCSR) {
        result.rows = static_cast<Index>(indices.len);
        result.cols = src.cols;
    } else {
        result.rows = src.rows;
        result.cols = static_cast<Index>(indices.len);
    }
    return result;
}

/// @brief Slice secondary dimension (requires materialization)
///
/// For CSR: Column slicing (requires copy)
/// For CSC: Row slicing (requires copy)
///
/// @param src Source sparse matrix
/// @param mask Binary mask [size = secondary_dim]
/// @param dst Destination CustomSparse [pre-allocated]
template <typename SrcT, typename T, bool IsCSR>
    requires AnySparse<SrcT>
void slice_secondary_masked(
    const SrcT& src,
    Array<const uint8_t> mask,
    CustomSparse<T, IsCSR>& dst
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>, "Type mismatch");
    
    const Index primary_dim = scl::primary_size(src);
    const Index secondary_dim = scl::secondary_size(src);
    
    SCL_CHECK_DIM(mask.len == static_cast<Size>(secondary_dim), "Mask size mismatch");
    SCL_CHECK_ARG(dst.data != nullptr, "Destination not allocated");

    // Build remapping table
    std::vector<Index> remap(secondary_dim);
    Index new_secondary_dim = 0;
    
    for (Index i = 0; i < secondary_dim; ++i) {
        if (mask[i]) {
            remap[i] = new_secondary_dim++;
        } else {
            remap[i] = -1;
        }
    }

    // Build indptr
    dst.indptr[0] = 0;
    
    for (Index p = 0; p < primary_dim; ++p) {
        auto inds = scl::primary_indices(src, p);
        Index len = scl::primary_length(src, p);
        
        Index count = 0;
        for (Index k = 0; k < len; ++k) {
            if (mask[inds[k]]) count++;
        }
        
        dst.indptr[p + 1] = dst.indptr[p] + count;
    }

    // Parallel copy and remap
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        Index dst_start = dst.indptr[idx];
        
        auto src_vals = scl::primary_values(src, idx);
        auto src_inds = scl::primary_indices(src, idx);
        Index src_len = scl::primary_length(src, idx);
        
        Index write_pos = 0;
        
        for (Index k = 0; k < src_len; ++k) {
            Index old_idx = src_inds[k];
            
            if (mask[old_idx]) {
                dst.indices[dst_start + write_pos] = remap[old_idx];
                dst.data[dst_start + write_pos] = src_vals[k];
                write_pos++;
            }
        }
    });
    
    // Update dimensions
    if constexpr (IsCSR) {
        dst.cols = new_secondary_dim;
    } else {
        dst.rows = new_secondary_dim;
    }
}

// =============================================================================
// In-Place Defragmentation
// =============================================================================

/// @brief Defragment gapped CustomSparse in-place
///
/// Compacts data within existing allocation.
///
/// @param mat Matrix to defragment [modified in-place]
template <typename T, bool IsCSR>
void defragment_inplace(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    // Build new indptr
    std::vector<Index> new_indptr(primary_dim + 1);
    new_indptr[0] = 0;
    
    for (Index i = 0; i < primary_dim; ++i) {
        Index len = scl::primary_length(mat, i);
        new_indptr[i + 1] = new_indptr[i] + len;
    }
    
    const Index new_nnz = new_indptr[primary_dim];

    // Compact via temporary buffer
    std::vector<T> temp_data(new_nnz);
    std::vector<Index> temp_indices(new_nnz);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        Index len = scl::primary_length(mat, idx);
        
        if (len == 0) return;
        
        Index old_start = mat.indptr[idx];
        Index new_start = new_indptr[idx];
        
        scl::memory::copy_fast(
            Array<const T>(mat.data + old_start, static_cast<Size>(len)),
            Array<T>(temp_data.data() + new_start, static_cast<Size>(len))
        );
        scl::memory::copy_fast(
            Array<const Index>(mat.indices + old_start, static_cast<Size>(len)),
            Array<Index>(temp_indices.data() + new_start, static_cast<Size>(len))
        );
    });
    
    // Copy back
    scl::memory::copy_fast(
        Array<const T>(temp_data.data(), new_nnz),
        Array<T>(mat.data, new_nnz)
    );
    scl::memory::copy_fast(
        Array<const Index>(temp_indices.data(), new_nnz),
        Array<Index>(mat.indices, new_nnz)
    );
    scl::memory::copy_fast(
        Array<const Index>(new_indptr.data(), primary_dim + 1),
        Array<Index>(mat.indptr, primary_dim + 1)
    );
}

// =============================================================================
// Helper Functions
// =============================================================================

/// @brief Calculate required buffer sizes for materialization
///
/// @param mat Input sparse matrix
/// @param out_nnz Output: Required data/indices size
/// @param out_indptr_size Output: Required indptr size
template <typename MatrixT>
    requires AnySparse<MatrixT>
void calculate_compact_sizes(
    const MatrixT& mat,
    Size& out_nnz,
    Size& out_indptr_size
) {
    out_nnz = count_active_nnz(mat);
    out_indptr_size = static_cast<Size>(scl::primary_size(mat)) + 1;
}

} // namespace scl::utils
