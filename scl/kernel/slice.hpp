#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>
#include <numeric>
#include <cstring>

// =============================================================================
/// @file slice.hpp
/// @brief High-Performance Sparse Matrix Slicing Operations
///
/// Key Design Principles:
/// 1. Unsafe functions allocate and return new memory (caller owns)
/// 2. Returns CustomSparse (always contiguous after slicing)
/// 3. Memory management via lifetime.hpp
/// 4. Parallel execution for large matrices
///
/// Optimizations:
/// - SIMD mask operations
/// - Thread-local accumulation
/// - Cache-friendly memory access
/// - Parallel prefix sum where applicable
///
/// Memory Handoff Protocol:
/// - C++ allocates memory using scl::core::mem
/// - Returns raw pointers to caller
/// - Caller (Python) takes ownership and must free via scl_free
// =============================================================================

namespace scl::kernel::slice {

// =============================================================================
// Result Structure for Unsafe Operations
// =============================================================================

/// @brief Result of unsafe slice operation
///
/// Contains allocated memory pointers that caller must free.
/// All pointers are allocated via standard malloc.
template <typename T>
struct SliceResult {
    T* data;            ///< Allocated data array [nnz]
    Index* indices;     ///< Allocated indices array [nnz]
    Index* indptr;      ///< Allocated indptr array [n_primary + 1]
    Index n_primary;    ///< Number of primary dimension elements
    Index n_secondary;  ///< Number of secondary dimension elements
    Index nnz;          ///< Number of non-zeros

    /// @brief Convert to CustomSparse (non-owning view)
    template <bool IsCSR>
    CustomSparse<T, IsCSR> as_sparse() const {
        if constexpr (IsCSR) {
            return CustomSparse<T, IsCSR>(data, indices, indptr, n_primary, n_secondary);
        } else {
            return CustomSparse<T, IsCSR>(data, indices, indptr, n_secondary, n_primary);
        }
    }
};

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD popcount for mask (count set bits in uint8_t array)
SCL_FORCE_INLINE Size popcount_mask_simd(const uint8_t* mask, Size count) {
    Size result = 0;
    
    // 8-way unrolled scalar (compiler will auto-vectorize)
    Size k = 0;
    for (; k + 8 <= count; k += 8) {
        result += mask[k + 0];
        result += mask[k + 1];
        result += mask[k + 2];
        result += mask[k + 3];
        result += mask[k + 4];
        result += mask[k + 5];
        result += mask[k + 6];
        result += mask[k + 7];
    }
    
    for (; k < count; ++k) {
        result += mask[k];
    }
    
    return result;
}

/// @brief Count masked elements in sparse indices (SIMD-friendly)
SCL_FORCE_INLINE Index count_masked_indices(
    const Index* indices,
    Size len,
    const uint8_t* mask
) {
    Index count = 0;
    
    // 4-way unrolled for better ILP
    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        count += mask[indices[k + 0]];
        count += mask[indices[k + 1]];
        count += mask[indices[k + 2]];
        count += mask[indices[k + 3]];
    }
    
    for (; k < len; ++k) {
        count += mask[indices[k]];
    }
    
    return count;
}

/// @brief Build new index mapping from mask (returns count of set bits)
SCL_FORCE_INLINE Index build_index_mapping(
    const uint8_t* mask,
    Index* new_indices,
    Index size
) {
    Index new_idx = 0;
    for (Index i = 0; i < size; ++i) {
        if (mask[i]) {
            new_indices[i] = new_idx++;
        } else {
            new_indices[i] = -1;
        }
    }
    return new_idx;
}

/// @brief Parallel prefix sum
inline void parallel_prefix_sum(Index* arr, Size n) {
    // For small arrays, sequential is faster
    if (n < 10000) {
        for (Size i = 1; i < n; ++i) {
            arr[i] += arr[i - 1];
        }
        return;
    }
    
    // Block-based parallel prefix sum
    const Size num_threads = scl::threading::Scheduler::get_num_threads();
    const Size block_size = (n + num_threads - 1) / num_threads;
    
    // Phase 1: Local prefix sums
    std::vector<Index> block_sums(num_threads, 0);
    
    scl::threading::parallel_for(Size(0), num_threads, [&](size_t tid) {
        Size start = tid * block_size;
        Size end = std::min(start + block_size, n);
        
        if (start >= n) return;
        
        for (Size i = start + 1; i < end; ++i) {
            arr[i] += arr[i - 1];
        }
        
        if (end > start) {
            block_sums[tid] = arr[end - 1];
        }
    });
    
    // Phase 2: Scan block sums
    for (Size i = 1; i < num_threads; ++i) {
        block_sums[i] += block_sums[i - 1];
    }
    
    // Phase 3: Add block offsets
    scl::threading::parallel_for(Size(1), num_threads, [&](size_t tid) {
        Size start = tid * block_size;
        Size end = std::min(start + block_size, n);
        
        Index offset = block_sums[tid - 1];
        for (Size i = start; i < end; ++i) {
            arr[i] += offset;
        }
    });
}

} // namespace detail

// =============================================================================
// Inspection Functions (Compute Output Size)
// =============================================================================

/// @brief Compute output nnz for primary dimension slice
///
/// @param matrix Input sparse matrix
/// @param keep_indices Indices of primary dimension elements to keep
/// @return Output nnz count
template <typename MatrixT>
    requires AnySparse<MatrixT>
Index inspect_slice_primary(
    const MatrixT& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();
    const Index n_primary = scl::primary_size(matrix);
    
    if (n_keep == 0) return 0;
    
    Index total_nnz = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary,
                      "Slice: Index out of bounds");
        total_nnz += scl::primary_length(matrix, idx);
    }

    return total_nnz;
}

/// @brief Compute output nnz for secondary dimension filter (mask-based)
///
/// Uses optimized counting with SIMD-friendly loop
///
/// @param matrix Input CustomSparse matrix
/// @param mask Boolean mask for secondary dimension [size = secondary_dim]
/// @return Output nnz count
template <typename T, bool IsCSR>
Index inspect_filter_secondary(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    Index total_nnz = 0;

    for (Index p = 0; p < primary_dim; ++p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        if (len > 0) {
            total_nnz += detail::count_masked_indices(
                matrix.indices + start,
                static_cast<Size>(len),
                mask.data()
            );
        }
    }

    return total_nnz;
}

// =============================================================================
// Safe Slice Functions (Pre-allocated Output)
// =============================================================================

/// @brief Slice primary dimension into pre-allocated output
///
/// @param matrix Input sparse matrix
/// @param keep_indices Indices to keep
/// @param out_data Pre-allocated output data [out_nnz]
/// @param out_indices Pre-allocated output indices [out_nnz]
/// @param out_indptr Pre-allocated output indptr [n_keep + 1]
template <typename T, bool IsCSR>
void materialize_slice_primary(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.size();

    SCL_CHECK_DIM(out_indptr.size() >= n_keep + 1,
                  "Slice: Output indptr too small");

    // Build indptr and copy data
    Index write_pos = 0;
    out_indptr[0] = 0;

    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index src_start = matrix.indptr[src_idx];
        Index src_end = matrix.indptr[src_idx + 1];
        Index len = src_end - src_start;

        // Bulk copy using memcpy for efficiency
        if (len > 0) {
            std::memcpy(out_data.data() + write_pos,
                       matrix.data + src_start,
                       len * sizeof(T));
            std::memcpy(out_indices.data() + write_pos,
                       matrix.indices + src_start,
                       len * sizeof(Index));
        }

        write_pos += len;
        out_indptr[i + 1] = write_pos;
    }
}

/// @brief Filter secondary dimension into pre-allocated output
///
/// @param matrix Input sparse matrix
/// @param mask Boolean mask for secondary dimension
/// @param new_indices Remapped indices (old -> new, -1 = drop)
/// @param out_data Pre-allocated output data
/// @param out_indices Pre-allocated output indices
/// @param out_indptr Pre-allocated output indptr
template <typename T, bool IsCSR>
void materialize_filter_secondary(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Index primary_dim = scl::primary_size(matrix);

    Index write_pos = 0;
    out_indptr[0] = 0;

    for (Index p = 0; p < primary_dim; ++p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];

        for (Index k = start; k < end; ++k) {
            Index old_idx = matrix.indices[k];
            if (mask[old_idx]) {
                out_data[write_pos] = matrix.data[k];
                out_indices[write_pos] = new_indices[old_idx];
                write_pos++;
            }
        }

        out_indptr[p + 1] = write_pos;
    }
}

// =============================================================================
// Unsafe Slice Functions (Allocating)
// =============================================================================

/// @brief Slice primary dimension - allocates output memory
///
/// UNSAFE: Caller must free returned pointers via scl_free
///
/// @param matrix Input sparse matrix
/// @param keep_indices Indices of primary elements to keep
/// @return SliceResult with allocated memory (caller owns)
template <typename T, bool IsCSR>
SliceResult<T> slice_primary_unsafe(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.size();

    // Compute output nnz
    Index out_nnz = inspect_slice_primary(matrix, keep_indices);

    // Allocate output arrays
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(n_keep + 1);

    // Materialize slice
    materialize_slice_primary(
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

/// @brief Slice rows (CSR) - allocates output memory
///
/// Convenience wrapper for CSR row slicing.
/// UNSAFE: Caller must free returned pointers.
///
/// @param matrix Input CSR matrix
/// @param row_indices Rows to keep
/// @return SliceResult with allocated memory
template <typename T>
SliceResult<T> slice_rows_unsafe(
    const CustomSparse<T, true>& matrix,
    Array<const Index> row_indices
) {
    return slice_primary_unsafe(matrix, row_indices);
}

/// @brief Slice columns (CSC) - allocates output memory
///
/// Convenience wrapper for CSC column slicing.
/// UNSAFE: Caller must free returned pointers.
///
/// @param matrix Input CSC matrix
/// @param col_indices Columns to keep
/// @return SliceResult with allocated memory
template <typename T>
SliceResult<T> slice_cols_unsafe(
    const CustomSparse<T, false>& matrix,
    Array<const Index> col_indices
) {
    return slice_primary_unsafe(matrix, col_indices);
}

/// @brief Filter secondary dimension by mask - allocates output memory
///
/// UNSAFE: Caller must free returned pointers via scl_free
///
/// @param matrix Input sparse matrix
/// @param mask Boolean mask [size = secondary_dim]
/// @return SliceResult with allocated memory (caller owns)
template <typename T, bool IsCSR>
SliceResult<T> filter_secondary_unsafe(
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

    Index new_secondary = detail::build_index_mapping(
        mask.data(),
        new_indices.data(),
        secondary_dim
    );

    // Compute output nnz
    Index out_nnz = inspect_filter_secondary(matrix, mask);

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);

    // Materialize
    materialize_filter_secondary(
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

/// @brief Filter columns in CSR matrix by mask - allocates output
template <typename T>
SliceResult<T> filter_cols_unsafe(
    const CustomSparse<T, true>& matrix,
    Array<const uint8_t> mask
) {
    return filter_secondary_unsafe(matrix, mask);
}

/// @brief Filter rows in CSC matrix by mask - allocates output
template <typename T>
SliceResult<T> filter_rows_unsafe(
    const CustomSparse<T, false>& matrix,
    Array<const uint8_t> mask
) {
    return filter_secondary_unsafe(matrix, mask);
}

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Compute primary lengths from indptr (parallel)
///
/// @param indptr Input indptr array [n + 1]
/// @param output Output lengths array [n]
inline void compute_lengths_from_indptr(
    Array<const Index> indptr,
    Array<Index> output
) {
    SCL_CHECK_DIM(output.size() + 1 <= indptr.size(),
                  "compute_lengths: Size mismatch");

    const Size n = output.size();

    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        output[i] = indptr[i + 1] - indptr[i];
    });
}

/// @brief Build indptr from lengths (exclusive prefix sum)
///
/// @param lengths Input lengths array [n]
/// @param indptr Output indptr array [n + 1]
inline void build_indptr_from_lengths(
    Array<const Index> lengths,
    Array<Index> indptr
) {
    SCL_CHECK_DIM(indptr.size() >= lengths.size() + 1,
                  "build_indptr: Output too small");

    indptr[0] = 0;
    for (Size i = 0; i < lengths.size(); ++i) {
        indptr[i + 1] = indptr[i] + lengths[i];
    }
}

/// @brief Build mask from indices (SIMD-optimized zero-fill)
///
/// @param indices Indices to set to 1
/// @param total_size Size of output mask
/// @param mask Output mask array [total_size], zero-initialized
inline void build_mask_from_indices(
    Array<const Index> indices,
    Index total_size,
    Array<uint8_t> mask
) {
    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(total_size),
                  "build_mask: Mask too small");

    // Zero out mask using memset (fastest)
    std::memset(mask.data(), 0, static_cast<Size>(total_size));

    // Set selected indices
    for (Size i = 0; i < indices.size(); ++i) {
        Index idx = indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < total_size,
                      "build_mask: Index out of bounds");
        mask[idx] = 1;
    }
}

/// @brief Count non-zero elements in mask
inline Size count_mask(Array<const uint8_t> mask) {
    return detail::popcount_mask_simd(mask.data(), mask.size());
}

} // namespace scl::kernel::slice
