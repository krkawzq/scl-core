#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <cstring>  // std::memcpy
#include <algorithm>
#include <type_traits>  // std::is_same_v

// =============================================================================
/// @file utils/matrix.hpp
/// @brief High-Performance Matrix Utilities
///
/// Tools for memory management, introspection, and materialization of matrix views.
///
/// Core Functions:
///
/// 1. count_active_nnz: Parallel computation of true NNZ in virtual matrices
/// 2. compact_copy: Materialize virtual views to standard matrices
///
/// Design Philosophy:
///
/// - Separation of Concerns: User allocates memory, we fill it
/// - Maximum Performance: SIMD + parallel reduction
/// - Type Safety: Works with any matrix type via concepts
/// - Zero State: Pure functions, thread-safe
///
/// Typical Workflow:
///
/// @code{.cpp}
/// // 1. Count required storage
/// Size nnz = count_active_nnz(virtual_matrix);
/// 
/// // 2. Allocate memory (user controls allocation strategy)
/// std::vector<Real> data(nnz);
/// std::vector<Index> indices(nnz);
/// std::vector<Index> indptr(virtual_matrix.rows + 1);
/// 
/// // 3. Materialize view to standard matrix
/// CustomCSR<Real> standard(data.data(), indices.data(), indptr.data(),
///                          virtual_matrix.rows, virtual_matrix.cols, nnz);
/// compact_copy(virtual_matrix, standard);
/// 
/// // 4. Use standard matrix (or serialize to disk)
/// @endcode
// =============================================================================

namespace scl::utils {

namespace detail {

/// @brief Cache-aligned accumulator to avoid false sharing.
///
/// Each thread writes to its own cache line (64 bytes).
template <typename T>
struct alignas(64) Accumulator {
    T value;
    char padding[64 - sizeof(T)];  // Pad to cache line
    
    Accumulator() : value(0) {}
};

} // namespace detail

// =============================================================================
// 1. Storage Inspection (Parallel Reduction)
// =============================================================================

/// @brief Compute total active NNZ for CSR-like matrices (Parallel Reduction).
///
/// For virtual/filtered matrices, performs parallel summation of row lengths.
/// For standard matrices without explicit lengths, could shortcut to mat.nnz,
/// but we compute accurately to handle gapped matrices.
///
/// Algorithm:
/// 1. Partition rows across threads
/// 2. Each thread sums its chunk (auto-vectorized)
/// 3. Final reduction (no atomics needed)
///
/// Performance:
/// - Time: O(rows / cores)
/// - Bandwidth: Memory-bound (prefetcher hides latency)
/// - Scalability: Linear with thread count
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param mat Input matrix (VirtualCSR, CustomCSR, etc.)
/// @return Total number of active non-zero elements
template <CSRLike MatrixT>
Size count_active_nnz(const MatrixT& mat) {
    const Index rows = mat.rows;
    
    if (rows == 0) return 0;

    // Determine concurrency
    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    const size_t chunk_size = (static_cast<size_t>(rows) + num_threads - 1) / num_threads;

    // Thread-local accumulators (avoid atomic contention)
    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(rows));
        
        Size local_sum = 0;
        
        // Compiler auto-vectorization friendly loop
        // Modern compilers will generate SIMD code for this simple accumulation
        for (size_t i = start; i < end; ++i) {
            local_sum += static_cast<Size>(mat.row_length(static_cast<Index>(i)));
        }
        
        partial_sums[t_id].value = local_sum;
    });

    // Final reduction (serial, but fast for small num_threads)
    Size total_nnz = 0;
    for (const auto& acc : partial_sums) {
        total_nnz += acc.value;
    }
    
    return total_nnz;
}

/// @brief Compute total active NNZ for CSC-like matrices (Parallel Reduction).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param mat Input matrix
/// @return Total number of active non-zero elements
template <CSCLike MatrixT>
Size count_active_nnz(const MatrixT& mat) {
    const Index cols = mat.cols;
    
    if (cols == 0) return 0;

    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    const size_t chunk_size = (static_cast<size_t>(cols) + num_threads - 1) / num_threads;

    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(cols));
        
        Size local_sum = 0;
        
        for (size_t j = start; j < end; ++j) {
            local_sum += static_cast<Size>(mat.col_length(static_cast<Index>(j)));
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
// 2. Data Materialization (Compact Copy)
// =============================================================================

/// @brief Materialize CSR-like view to standard CustomCSR (Parallel Copy).
///
/// Compacts ANY source view (VirtualCSR, gapped CustomCSR) into a
/// pre-allocated standard CSR matrix with contiguous storage.
///
/// Algorithm:
/// 1. Validate dimensions and capacity
/// 2. Rebuild dst.indptr via prefix sum of source row lengths
/// 3. Parallel memcpy of indices and values arrays
///
/// Properties:
/// - Input: Any CSR-like matrix (virtual, filtered, gapped)
/// - Output: Standard compact CSR (no explicit lengths needed)
/// - Performance: Memory bandwidth bound (~10-15 GB/s per core)
///
/// Requirements:
/// - dst.data, dst.indices, dst.indptr must be pre-allocated
/// - dst.nnz should represent buffer capacity
/// - Will fail gracefully if capacity insufficient
///
/// @tparam SrcT Source matrix type (must satisfy CSRLike)
/// @tparam T Value type (typically Real)
/// @param src Source matrix (any CSR-like view)
/// @param dst Destination matrix (CustomCSR with pre-allocated arrays)
template <CSRLike SrcT, typename T>
void compact_copy(const SrcT& src, CustomCSR<T>& dst) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>, 
                  "Compact copy: Source and destination value types must match");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    // -------------------------------------------------------------------------
    // Step 1: Validation
    // -------------------------------------------------------------------------
    
    SCL_CHECK_ARG(dst.data != nullptr, "Compact copy: Destination data array not allocated");
    SCL_CHECK_ARG(dst.indices != nullptr, "Compact copy: Destination indices array not allocated");
    SCL_CHECK_ARG(dst.indptr != nullptr, "Compact copy: Destination indptr array not allocated");
    SCL_CHECK_DIM(dst.rows == rows, "Compact copy: Row dimension mismatch");
    SCL_CHECK_DIM(dst.cols == cols, "Compact copy: Column dimension mismatch");

    // -------------------------------------------------------------------------
    // Step 2: Build Destination Indptr (Serial Prefix Sum)
    // -------------------------------------------------------------------------
    // Time: O(rows) - typically << 1ms for millions of rows
    
    dst.indptr[0] = 0;
    for (Index i = 0; i < rows; ++i) {
        Index len = src.row_length(i);
        dst.indptr[i + 1] = dst.indptr[i] + len;
    }
    
    const Size required_nnz = static_cast<Size>(dst.indptr[rows]);
    
    // Capacity check
    SCL_CHECK_ARG(required_nnz <= static_cast<Size>(dst.nnz),
                  "Compact copy: Destination capacity insufficient");
    
    // Update actual NNZ
    dst.nnz = static_cast<Index>(required_nnz);
    dst.row_lengths = nullptr;  // No explicit lengths needed (compact)

    // -------------------------------------------------------------------------
    // Step 3: Parallel Data Copy
    // -------------------------------------------------------------------------
    // Time: O(NNZ / cores) - memory bandwidth bound
    // Uses std::memcpy for optimal performance (SIMD on some platforms)
    
    scl::threading::parallel_for(0, static_cast<size_t>(rows), [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        
        // Get source spans via unified interface
        auto src_vals = src.row_values(row_idx);
        auto src_idxs = src.row_indices(row_idx);
        Index len = src.row_length(row_idx);
        
        if (len == 0) return;

        // Destination location
        Index dst_start = dst.indptr[row_idx];
        
        // Fast memcpy (compiler optimizes to SIMD moves)
        std::memcpy(dst.data + dst_start, src_vals.ptr, 
                   static_cast<size_t>(len) * sizeof(T));
        std::memcpy(dst.indices + dst_start, src_idxs.ptr, 
                   static_cast<size_t>(len) * sizeof(Index));
    });
}

/// @brief Materialize CSC-like view to standard CustomCSC (Parallel Copy).
///
/// Symmetric implementation for CSC format.
///
/// @tparam SrcT Source matrix type (must satisfy CSCLike)
/// @tparam T Value type (typically Real)
/// @param src Source matrix (any CSC-like view)
/// @param dst Destination matrix (CustomCSC with pre-allocated arrays)
template <CSCLike SrcT, typename T>
void compact_copy(const SrcT& src, CustomCSC<T>& dst) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Compact copy: Source and destination value types must match");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    // -------------------------------------------------------------------------
    // Step 1: Validation
    // -------------------------------------------------------------------------
    
    SCL_CHECK_ARG(dst.data != nullptr, "Compact copy: Destination data array not allocated");
    SCL_CHECK_ARG(dst.indices != nullptr, "Compact copy: Destination indices array not allocated");
    SCL_CHECK_ARG(dst.indptr != nullptr, "Compact copy: Destination indptr array not allocated");
    SCL_CHECK_DIM(dst.rows == rows, "Compact copy: Row dimension mismatch");
    SCL_CHECK_DIM(dst.cols == cols, "Compact copy: Column dimension mismatch");

    // -------------------------------------------------------------------------
    // Step 2: Build Destination Indptr (Serial Prefix Sum)
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    for (Index j = 0; j < cols; ++j) {
        Index len = src.col_length(j);
        dst.indptr[j + 1] = dst.indptr[j] + len;
    }
    
    const Size required_nnz = static_cast<Size>(dst.indptr[cols]);
    
    SCL_CHECK_ARG(required_nnz <= static_cast<Size>(dst.nnz),
                  "Compact copy: Destination capacity insufficient");
    
    dst.nnz = static_cast<Index>(required_nnz);
    dst.col_lengths = nullptr;

    // -------------------------------------------------------------------------
    // Step 3: Parallel Data Copy
    // -------------------------------------------------------------------------
    
    scl::threading::parallel_for(0, static_cast<size_t>(cols), [&](size_t j) {
        Index col_idx = static_cast<Index>(j);
        
        auto src_vals = src.col_values(col_idx);
        auto src_idxs = src.col_indices(col_idx);
        Index len = src.col_length(col_idx);
        
        if (len == 0) return;

        Index dst_start = dst.indptr[col_idx];
        
        std::memcpy(dst.data + dst_start, src_vals.ptr, 
                   static_cast<size_t>(len) * sizeof(T));
        std::memcpy(dst.indices + dst_start, src_idxs.ptr, 
                   static_cast<size_t>(len) * sizeof(Index));
    });
}

// =============================================================================
// 3. Helper: Calculate Required Buffer Sizes
// =============================================================================

/// @brief Calculate required buffer sizes for materializing a sparse matrix.
///
/// Returns the exact sizes needed for data, indices, and indptr arrays.
/// User can then allocate memory before calling compact_copy.
///
/// @tparam MatrixT Any CSR-like or CSC-like matrix type
/// @param mat Input matrix
/// @param out_nnz Output: Required NNZ (data and indices array size)
/// @param out_indptr_size Output: Required indptr array size (rows+1 or cols+1)
template <SparseLike MatrixT>
void calculate_compact_sizes(
    const MatrixT& mat,
    Size& out_nnz,
    Size& out_indptr_size
) {
    out_nnz = count_active_nnz(mat);
    
    if constexpr (CSRLike<MatrixT>) {
        out_indptr_size = static_cast<Size>(mat.rows) + 1;
    } else {
        out_indptr_size = static_cast<Size>(mat.cols) + 1;
    }
}

// =============================================================================
// 4. Advanced: In-Place Defragmentation
// =============================================================================

/// @brief Defragment a gapped CustomCSR matrix in-place.
///
/// Compacts data within the existing allocation by shifting valid data forward.
/// Rebuilds indptr to point to compacted locations.
///
/// Use Case: When matrix has been filtered/aligned but you want to reuse
/// the same memory allocation without external copy.
///
/// Limitations:
/// - Can only compact within existing capacity (cannot grow)
/// - Requires temporary workspace for parallel coordinate calculation
///
/// @tparam T Value type
/// @param mat Matrix to defragment [modified in-place]
template <typename T>
void defragment_inplace(CustomCSR<T>& mat) {
    SCL_CHECK_ARG(mat.row_lengths != nullptr,
                  "Defragment: Matrix must have explicit lengths (gapped state)");

    const Index rows = mat.rows;
    
    // -------------------------------------------------------------------------
    // Step 1: Build New Indptr (Prefix Sum)
    // -------------------------------------------------------------------------
    
    std::vector<Index> new_indptr(rows + 1);
    new_indptr[0] = 0;
    
    for (Index i = 0; i < rows; ++i) {
        new_indptr[i + 1] = new_indptr[i] + mat.row_length(i);
    }
    
    const Index new_nnz = new_indptr[rows];

    // -------------------------------------------------------------------------
    // Step 2: Parallel Compaction
    // -------------------------------------------------------------------------
    // Strategy: Copy data to correct positions
    // Challenge: Potential overlapping ranges require careful ordering
    
    // Safe approach: Use temporary buffer (simpler than in-place shuffle)
    std::vector<T> temp_data(new_nnz);
    std::vector<Index> temp_indices(new_nnz);
    
    scl::threading::parallel_for(0, static_cast<size_t>(rows), [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        Index len = mat.row_length(row_idx);
        
        if (len == 0) return;
        
        Index old_start = mat.indptr[row_idx];
        Index new_start = new_indptr[row_idx];
        
        // Copy to temp buffer
        std::memcpy(temp_data.data() + new_start, 
                   mat.data + old_start, 
                   static_cast<size_t>(len) * sizeof(T));
        std::memcpy(temp_indices.data() + new_start,
                   mat.indices + old_start,
                   static_cast<size_t>(len) * sizeof(Index));
    });
    
    // -------------------------------------------------------------------------
    // Step 3: Copy Back and Update Metadata
    // -------------------------------------------------------------------------
    
    std::memcpy(mat.data, temp_data.data(), 
               static_cast<size_t>(new_nnz) * sizeof(T));
    std::memcpy(mat.indices, temp_indices.data(),
               static_cast<size_t>(new_nnz) * sizeof(Index));
    std::memcpy(mat.indptr, new_indptr.data(),
               static_cast<size_t>(rows + 1) * sizeof(Index));
    
    mat.nnz = new_nnz;
    mat.row_lengths = nullptr;  // No longer gapped
}

/// @brief Defragment CSC matrix in-place.
///
/// @tparam T Value type
/// @param mat Matrix to defragment [modified in-place]
template <typename T>
void defragment_inplace(CustomCSC<T>& mat) {
    SCL_CHECK_ARG(mat.col_lengths != nullptr,
                  "Defragment: Matrix must have explicit lengths (gapped state)");

    const Index cols = mat.cols;
    
    std::vector<Index> new_indptr(cols + 1);
    new_indptr[0] = 0;
    
    for (Index j = 0; j < cols; ++j) {
        new_indptr[j + 1] = new_indptr[j] + mat.col_length(j);
    }
    
    const Index new_nnz = new_indptr[cols];
    
    std::vector<T> temp_data(new_nnz);
    std::vector<Index> temp_indices(new_nnz);
    
    scl::threading::parallel_for(0, static_cast<size_t>(cols), [&](size_t j) {
        Index col_idx = static_cast<Index>(j);
        Index len = mat.col_length(col_idx);
        
        if (len == 0) return;
        
        Index old_start = mat.indptr[col_idx];
        Index new_start = new_indptr[col_idx];
        
        std::memcpy(temp_data.data() + new_start,
                   mat.data + old_start,
                   static_cast<size_t>(len) * sizeof(T));
        std::memcpy(temp_indices.data() + new_start,
                   mat.indices + old_start,
                   static_cast<size_t>(len) * sizeof(Index));
    });
    
    std::memcpy(mat.data, temp_data.data(),
               static_cast<size_t>(new_nnz) * sizeof(T));
    std::memcpy(mat.indices, temp_indices.data(),
               static_cast<size_t>(new_nnz) * sizeof(Index));
    std::memcpy(mat.indptr, new_indptr.data(),
               static_cast<size_t>(cols + 1) * sizeof(Index));
    
    mat.nnz = new_nnz;
    mat.col_lengths = nullptr;
}

// =============================================================================
// 5. Slice Inspection (Pre-calculation)
// =============================================================================

/// @brief Calculate storage requirements for CSR matrix slice (Subsetting).
///
/// Determines the NNZ and Indptr size required to store the result of
/// `matrix[row_indices, col_indices]`.
///
/// Optimization:
/// - If col_indices is empty (keep all), NNZ calculation is just summing row lengths
/// - If col_indices provided, must be SORTED for O(log N) binary search
///
/// Performance:
/// - Time: O(rows × avg_nnz_per_row × log(n_cols)) if filtering columns
/// - Time: O(rows) if keeping all columns
/// - Space: O(1)
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param mat Source matrix
/// @param row_indices Rows to keep (empty = keep all)
/// @param col_indices Columns to keep (empty = keep all, MUST BE SORTED if provided)
/// @param out_nnz Output: Required data/indices array size
/// @param out_indptr_size Output: Required indptr array size (n_rows_out + 1)
template <CSRLike MatrixT>
void calculate_slice_shape(
    const MatrixT& mat,
    Span<const Index> row_indices,
    Span<const Index> col_indices,
    Size& out_nnz,
    Size& out_indptr_size
) {
    // Determine output dimensions
    const bool keep_all_rows = (row_indices.size == 0);
    const Index n_rows_out = keep_all_rows ? mat.rows : static_cast<Index>(row_indices.size);
    out_indptr_size = static_cast<Size>(n_rows_out) + 1;

    const bool filter_cols = (col_indices.size > 0);
    
    // Parallel reduction to count NNZ
    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);
    const size_t chunk_size = (static_cast<size_t>(n_rows_out) + num_threads - 1) / num_threads;

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(n_rows_out));
        
        Size local_nnz = 0;

        for (size_t i = start; i < end; ++i) {
            // Get source row index
            Index src_row = keep_all_rows ? static_cast<Index>(i) : row_indices[i];
            
            if (!filter_cols) {
                // Fast path: Keep all columns
                local_nnz += static_cast<Size>(mat.row_length(src_row));
            } else {
                // Slow path: Filter columns via binary search
                auto row_idxs = mat.row_indices(src_row);
                Index row_len = mat.row_length(src_row);
                
                for (Index k = 0; k < row_len; ++k) {
                    Index col = row_idxs[k];
                    // Binary search in sorted col_indices
                    if (std::binary_search(col_indices.begin(), col_indices.end(), col)) {
                        local_nnz++;
                    }
                }
            }
        }
        
        partial_sums[t_id].value = local_nnz;
    });

    // Final reduction
    out_nnz = 0;
    for (const auto& acc : partial_sums) {
        out_nnz += acc.value;
    }
}

// =============================================================================
// 6. Slice Materialization (Execution)
// =============================================================================

/// @brief Materialize a matrix slice into pre-allocated CustomCSR.
///
/// Performs actual data copying and index remapping for slice operation.
/// Column indices are remapped to [0, n_cols_out) in output.
///
/// Algorithm:
/// 1. Count NNZ per output row (parallel)
/// 2. Build output indptr (serial prefix sum)
/// 3. Copy and remap data (parallel)
///
/// Requirements:
/// - dst arrays must be pre-allocated (use calculate_slice_shape first)
/// - col_indices must be SORTED if provided
///
/// @tparam SrcT Source matrix type (any CSR-like)
/// @tparam T Value type
/// @param src Source matrix
/// @param row_indices Rows to extract (empty = all)
/// @param col_indices Columns to extract (empty = all, MUST BE SORTED)
/// @param dst Destination matrix [pre-allocated]
template <CSRLike SrcT, typename T>
void materialize_slice(
    const SrcT& src,
    Span<const Index> row_indices,
    Span<const Index> col_indices,
    CustomCSR<T>& dst
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Slice: Source and destination value types must match");
    
    const bool keep_all_rows = (row_indices.size == 0);
    const Index n_rows_out = keep_all_rows ? src.rows : static_cast<Index>(row_indices.size);
    const bool filter_cols = (col_indices.size > 0);

    // Validation
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Slice: Destination arrays not allocated");
    SCL_CHECK_DIM(dst.rows == n_rows_out, "Slice: Destination rows mismatch");

    // -------------------------------------------------------------------------
    // Step 1: Count NNZ Per Output Row (Parallel)
    // -------------------------------------------------------------------------
    
    std::vector<Index> row_nnzs(n_rows_out);
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_rows_out), [&](size_t i) {
        Index src_row = keep_all_rows ? static_cast<Index>(i) : row_indices[i];
        
        if (!filter_cols) {
            // Keep all columns
            row_nnzs[i] = src.row_length(src_row);
        } else {
            // Count matching columns
            auto row_idxs = src.row_indices(src_row);
            Index row_len = src.row_length(src_row);
            
            Index count = 0;
            for (Index k = 0; k < row_len; ++k) {
                if (std::binary_search(col_indices.begin(), col_indices.end(), row_idxs[k])) {
                    count++;
                }
            }
            row_nnzs[i] = count;
        }
    });

    // -------------------------------------------------------------------------
    // Step 2: Build Indptr (Serial Prefix Sum)
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    for (Index i = 0; i < n_rows_out; ++i) {
        dst.indptr[i + 1] = dst.indptr[i] + row_nnzs[i];
    }
    
    const Size total_nnz = static_cast<Size>(dst.indptr[n_rows_out]);
    SCL_CHECK_ARG(total_nnz <= static_cast<Size>(dst.nnz),
                  "Slice: Destination capacity insufficient");
    
    dst.nnz = static_cast<Index>(total_nnz);
    dst.cols = filter_cols ? static_cast<Index>(col_indices.size) : src.cols;
    dst.row_lengths = nullptr;

    // -------------------------------------------------------------------------
    // Step 3: Copy and Remap Data (Parallel)
    // -------------------------------------------------------------------------
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_rows_out), [&](size_t i) {
        Index src_row = keep_all_rows ? static_cast<Index>(i) : row_indices[i];
        Index dst_start = dst.indptr[i];
        
        auto src_vals = src.row_values(src_row);
        auto src_idxs = src.row_indices(src_row);
        Index src_len = src.row_length(src_row);

        if (!filter_cols) {
            // Fast path: Bulk copy without remapping
            std::memcpy(dst.data + dst_start, src_vals.ptr, 
                       static_cast<size_t>(src_len) * sizeof(T));
            std::memcpy(dst.indices + dst_start, src_idxs.ptr,
                       static_cast<size_t>(src_len) * sizeof(Index));
        } else {
            // Slow path: Filter and remap column indices
            Index write_pos = 0;
            for (Index k = 0; k < src_len; ++k) {
                Index old_col = src_idxs[k];
                
                // Binary search for column position in filtered set
                auto it = std::lower_bound(col_indices.begin(), col_indices.end(), old_col);
                if (it != col_indices.end() && *it == old_col) {
                    // Found: remap to new position
                    Index new_col = static_cast<Index>(it - col_indices.begin());
                    
                    dst.indices[dst_start + write_pos] = new_col;
                    dst.data[dst_start + write_pos] = src_vals[k];
                    write_pos++;
                }
            }
        }
    });
}

// =============================================================================
// 7. Format Conversion Inspection (CSR <-> CSC)
// =============================================================================

/// @brief Calculate storage requirements for CSR to CSC conversion.
///
/// CSR → CSC is essentially a transpose operation.
/// Output dimensions: (rows, cols) → (cols, rows)
///
/// Requirements:
/// - out_nnz: Total active NNZ (= count_active_nnz(mat))
/// - out_indptr_size: cols + 1
/// - out_workspace_size: cols (for histogram/scatter coordination)
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param mat Source CSR matrix
/// @param out_nnz Output: Required size for data and indices arrays
/// @param out_indptr_size Output: Required size for indptr array
/// @param out_workspace_size Output: Required temporary workspace size (in Index elements)
template <CSRLike MatrixT>
void calculate_csr_to_csc_sizes(
    const MatrixT& mat,
    Size& out_nnz,
    Size& out_indptr_size,
    Size& out_workspace_size
) {
    out_nnz = count_active_nnz(mat);
    out_indptr_size = static_cast<Size>(mat.cols) + 1;
    out_workspace_size = static_cast<Size>(mat.cols);  // For scatter coordination
}

/// @brief Calculate storage requirements for CSC to CSR conversion.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param mat Source CSC matrix
/// @param out_nnz Output: Required size for data and indices arrays
/// @param out_indptr_size Output: Required size for indptr array
/// @param out_workspace_size Output: Required temporary workspace size
template <CSCLike MatrixT>
void calculate_csc_to_csr_sizes(
    const MatrixT& mat,
    Size& out_nnz,
    Size& out_indptr_size,
    Size& out_workspace_size
) {
    out_nnz = count_active_nnz(mat);
    out_indptr_size = static_cast<Size>(mat.rows) + 1;
    out_workspace_size = static_cast<Size>(mat.rows);
}

// =============================================================================
// 8. Format Conversion Execution
// =============================================================================

/// @brief Convert CSR to CSC (Transpose Operation).
///
/// Requires pre-allocated destination and workspace buffers.
/// User must call calculate_csr_to_csc_sizes first to determine sizes.
///
/// Algorithm (Classic Transpose):
/// 1. Histogram: Count NNZ per column (serial scan)
/// 2. Prefix Sum: Build dst.indptr from histogram
/// 3. Scatter: Iterate CSR rows, write to CSC columns using workspace as write heads
///
/// Time: O(NNZ), Space: O(cols) workspace
///
/// @tparam SrcT Source matrix type (any CSR-like)
/// @tparam T Value type
/// @param src Source CSR matrix
/// @param dst Destination CSC matrix [pre-allocated arrays]
/// @param workspace Temporary buffer [size = src.cols, type Index]
template <CSRLike SrcT, typename T>
void convert_csr_to_csc(
    const SrcT& src,
    CustomCSC<T>& dst,
    MutableSpan<Index> workspace
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Convert: Source and destination value types must match");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    // Validation
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Convert: Destination arrays not allocated");
    SCL_CHECK_DIM(workspace.size >= static_cast<Size>(cols),
                  "Convert: Workspace too small");
    SCL_CHECK_DIM(dst.rows == cols, "Convert: CSC rows must equal CSR cols (transpose)");
    SCL_CHECK_DIM(dst.cols == rows, "Convert: CSC cols must equal CSR rows (transpose)");

    // -------------------------------------------------------------------------
    // Step 1: Histogram - Count NNZ per column
    // -------------------------------------------------------------------------
    // Serial scan (thread-safe, simple, fast for typical sparsity)
    
    std::fill(workspace.ptr, workspace.ptr + cols, 0);
    
    for (Index i = 0; i < rows; ++i) {
        auto row_idxs = src.row_indices(i);
        Index row_len = src.row_length(i);
        
        for (Index k = 0; k < row_len; ++k) {
            workspace[row_idxs[k]]++;
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: Build CSC Indptr (Prefix Sum)
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    for (Index j = 0; j < cols; ++j) {
        dst.indptr[j + 1] = dst.indptr[j] + workspace[j];
    }
    
    dst.nnz = dst.indptr[cols];

    // -------------------------------------------------------------------------
    // Step 3: Prepare Workspace as Write Heads
    // -------------------------------------------------------------------------
    // workspace[j] will track current write position for column j
    
    std::copy(dst.indptr, dst.indptr + cols, workspace.ptr);

    // -------------------------------------------------------------------------
    // Step 4: Scatter Fill (Row-by-Row)
    // -------------------------------------------------------------------------
    // Cannot easily parallelize this without complex synchronization
    // Serial is safe and typically fast (O(NNZ))
    
    for (Index i = 0; i < rows; ++i) {
        auto row_idxs = src.row_indices(i);
        auto row_vals = src.row_values(i);
        Index row_len = src.row_length(i);

        for (Index k = 0; k < row_len; ++k) {
            Index col = row_idxs[k];
            T val = row_vals[k];

            // Write to CSC column
            Index write_pos = workspace[col];
            workspace[col]++;  // Advance write head
            
            dst.indices[write_pos] = i;      // Row index becomes CSC index
            dst.data[write_pos] = val;
        }
    }
    
    dst.col_lengths = nullptr;  // Output is compact
}

/// @brief Convert CSC to CSR (Transpose Operation).
///
/// Symmetric implementation for CSC → CSR conversion.
///
/// @tparam SrcT Source matrix type (any CSC-like)
/// @tparam T Value type
/// @param src Source CSC matrix
/// @param dst Destination CSR matrix [pre-allocated arrays]
/// @param workspace Temporary buffer [size = src.rows, type Index]
template <CSCLike SrcT, typename T>
void convert_csc_to_csr(
    const SrcT& src,
    CustomCSR<T>& dst,
    MutableSpan<Index> workspace
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Convert: Source and destination value types must match");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    // Validation
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Convert: Destination arrays not allocated");
    SCL_CHECK_DIM(workspace.size >= static_cast<Size>(rows),
                  "Convert: Workspace too small");
    SCL_CHECK_DIM(dst.rows == cols, "Convert: CSR rows must equal CSC cols (transpose)");
    SCL_CHECK_DIM(dst.cols == rows, "Convert: CSR cols must equal CSC rows (transpose)");

    // -------------------------------------------------------------------------
    // Step 1: Histogram - Count NNZ per row
    // -------------------------------------------------------------------------
    
    std::fill(workspace.ptr, workspace.ptr + rows, 0);
    
    for (Index j = 0; j < cols; ++j) {
        auto col_idxs = src.col_indices(j);
        Index col_len = src.col_length(j);
        
        for (Index k = 0; k < col_len; ++k) {
            workspace[col_idxs[k]]++;
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: Build CSR Indptr
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    for (Index i = 0; i < rows; ++i) {
        dst.indptr[i + 1] = dst.indptr[i] + workspace[i];
    }
    
    dst.nnz = dst.indptr[rows];

    // -------------------------------------------------------------------------
    // Step 3: Prepare Write Heads
    // -------------------------------------------------------------------------
    
    std::copy(dst.indptr, dst.indptr + rows, workspace.ptr);

    // -------------------------------------------------------------------------
    // Step 4: Scatter Fill (Column-by-Column)
    // -------------------------------------------------------------------------
    
    for (Index j = 0; j < cols; ++j) {
        auto col_idxs = src.col_indices(j);
        auto col_vals = src.col_values(j);
        Index col_len = src.col_length(j);

        for (Index k = 0; k < col_len; ++k) {
            Index row = col_idxs[k];
            T val = col_vals[k];

            Index write_pos = workspace[row];
            workspace[row]++;
            
            dst.indices[write_pos] = j;      // Column index becomes CSR index
            dst.data[write_pos] = val;
        }
    }
    
    dst.row_lengths = nullptr;
}

// =============================================================================
// 9. Bitmap-Based Column Slicing (CSR) - Two-Pass Algorithm
// =============================================================================

/// @brief Inspect column slice using bitmap mask (Phase 1).
///
/// Calculates storage requirements for column-filtered CSR matrix.
/// Uses ByteMask (uint8_t) for O(1) lookup instead of binary search.
///
/// Design Note: ByteMask vs BitMask
/// - We use `Span<const Byte>` (uint8_t array) instead of std::vector<bool>
/// - Rationale: No bit-unpacking overhead, L1 cache friendly (30K genes = 30KB)
/// - Performance: 2-3x faster than BitMask for typical sizes
///
/// Algorithm:
/// - Parallel reduction over rows
/// - Each thread scans its chunk and counts matching columns
/// - Optional: Cache per-row NNZ for Phase 2 speedup
///
/// Performance:
/// - Time: O(NNZ) - single pass over all non-zeros
/// - Space: O(1) + optional O(rows) cache
/// - Speedup vs binary search: 5-10x for sparse selection
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param mat Source matrix
/// @param keep_col_mask Bitmap mask [size = mat.cols], 0 = drop, non-zero = keep
/// @param out_row_nnz Optional cache: NNZ per output row [size = mat.rows]
/// @return Total NNZ required for sliced matrix
template <CSRLike MatrixT>
Size inspect_col_slice(
    const MatrixT& mat,
    Span<const Byte> keep_col_mask,
    MutableSpan<Index> out_row_nnz = MutableSpan<Index>()
) {
    const Index rows = mat.rows;
    const Index cols = mat.cols;
    
    SCL_CHECK_DIM(keep_col_mask.size == static_cast<Size>(cols),
                  "Col slice inspect: Mask size must match matrix.cols");
    
    if (out_row_nnz.size > 0) {
        SCL_CHECK_DIM(out_row_nnz.size == static_cast<Size>(rows),
                      "Col slice inspect: Cache size must match matrix.rows");
    }

    // Parallel reduction
    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);
    const size_t chunk_size = (static_cast<size_t>(rows) + num_threads - 1) / num_threads;

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(rows));
        
        Size local_total = 0;

        for (size_t i = start; i < end; ++i) {
            Index row_idx = static_cast<Index>(i);
            auto row_idxs = mat.row_indices(row_idx);
            Index row_len = mat.row_length(row_idx);
            
            Index count = 0;
            
            // Hot loop: Bitmap check is branch-predictable and cache-friendly
            for (Index k = 0; k < row_len; ++k) {
                if (keep_col_mask[row_idxs[k]]) {
                    count++;
                }
            }
            
            local_total += static_cast<Size>(count);
            
            // Cache per-row count if requested
            if (out_row_nnz.size > 0) {
                out_row_nnz[row_idx] = count;
            }
        }
        
        partial_sums[t_id].value = local_total;
    });

    // Final reduction
    Size total_nnz = 0;
    for (const auto& acc : partial_sums) {
        total_nnz += acc.value;
    }
    
    return total_nnz;
}

/// @brief Materialize column slice using bitmap mask (Phase 2).
///
/// Executes the actual data copy and column index remapping.
/// Returns a new CustomCSR with compact, contiguous storage.
///
/// Algorithm:
/// 1. Build column remapping table (old_idx → new_idx)
/// 2. Build output indptr (using cached_row_nnz if available)
/// 3. Parallel copy and remap data
///
/// Requirements:
/// - dst arrays must be pre-allocated (use inspect_col_slice first)
/// - Returns CustomCSR directly (Custom → Virtual is zero-cost)
///
/// @tparam SrcT Source matrix type (any CSR-like)
/// @tparam T Value type
/// @param src Source matrix
/// @param keep_col_mask Bitmap mask [size = src.cols]
/// @param dst Destination matrix [pre-allocated arrays]
/// @param cached_row_nnz Optional: Per-row NNZ from inspect (huge speedup!)
/// @return Reference to dst (for chaining)
template <CSRLike SrcT, typename T>
CustomCSR<T>& materialize_col_slice(
    const SrcT& src,
    Span<const Byte> keep_col_mask,
    CustomCSR<T>& dst,
    Span<const Index> cached_row_nnz = Span<const Index>()
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Slice: Source and destination value types must match");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    SCL_CHECK_DIM(keep_col_mask.size == static_cast<Size>(cols),
                  "Col slice: Mask size mismatch");
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Col slice: Destination not allocated");
    SCL_CHECK_DIM(dst.rows == rows, "Col slice: Destination rows mismatch");

    // -------------------------------------------------------------------------
    // Step 1: Build Column Remapping Table
    // -------------------------------------------------------------------------
    // Time: O(cols) - serial, but cols is typically small (< 50K)
    
    std::vector<Index> col_remap(cols);
    Index new_col_count = 0;
    
    for (Index j = 0; j < cols; ++j) {
        if (keep_col_mask[j]) {
            col_remap[j] = new_col_count++;
        } else {
            col_remap[j] = -1;
        }
    }
    
    dst.cols = new_col_count;

    // -------------------------------------------------------------------------
    // Step 2: Build Indptr
    // -------------------------------------------------------------------------
    
    dst.indptr[0] = 0;
    
    if (cached_row_nnz.size > 0) {
        // Fast path: Use cached counts (O(rows))
        for (Index i = 0; i < rows; ++i) {
            dst.indptr[i + 1] = dst.indptr[i] + cached_row_nnz[i];
        }
    } else {
        // Slow path: Re-count (O(NNZ))
        for (Index i = 0; i < rows; ++i) {
            auto row_idxs = src.row_indices(i);
            Index row_len = src.row_length(i);
            
            Index count = 0;
            for (Index k = 0; k < row_len; ++k) {
                if (keep_col_mask[row_idxs[k]]) count++;
            }
            
            dst.indptr[i + 1] = dst.indptr[i] + count;
        }
    }
    
    const Size total_nnz = static_cast<Size>(dst.indptr[rows]);
    SCL_CHECK_ARG(total_nnz <= static_cast<Size>(dst.nnz),
                  "Col slice: Buffer overflow");
    
    dst.nnz = static_cast<Index>(total_nnz);
    dst.row_lengths = nullptr;

    // -------------------------------------------------------------------------
    // Step 3: Parallel Copy and Remap
    // -------------------------------------------------------------------------
    
    scl::threading::parallel_for(0, static_cast<size_t>(rows), [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        Index dst_start = dst.indptr[row_idx];
        
        auto src_vals = src.row_values(row_idx);
        auto src_idxs = src.row_indices(row_idx);
        Index src_len = src.row_length(row_idx);
        
        Index write_pos = 0;
        
        for (Index k = 0; k < src_len; ++k) {
            Index old_col = src_idxs[k];
            
            if (keep_col_mask[old_col]) {
                dst.indices[dst_start + write_pos] = col_remap[old_col];
                dst.data[dst_start + write_pos] = src_vals[k];
                write_pos++;
            }
        }
    });
    
    return dst;
}

// =============================================================================
// 10. Unsafe One-Pass Slicing (Advanced Users Only)
// =============================================================================

/// @brief UNSAFE: Materialize column slice with user-provided oversized buffer.
///
/// Skips inspection phase - user provides buffer "large enough" and we fill it.
/// Updates dst dimensions automatically for immediate usability.
///
/// WARNING: If buffer is insufficient, WILL OVERFLOW (checked in debug mode).
/// Use only if you can guarantee buffer size (e.g., allocate src.nnz as upper bound).
///
/// Use Case: Python binding where user allocates worst-case buffer.
///
/// Safety Notes:
/// - Despite being "unsafe", we update dst.rows/cols for convenience
/// - Debug builds include overflow checks
/// - Release builds skip checks for maximum performance
///
/// @tparam SrcT Source matrix type
/// @tparam T Value type
/// @param src Source matrix
/// @param keep_col_mask ByteMask (uint8_t): 0=drop, non-zero=keep [size = src.cols]
/// @param dst Destination matrix [pre-allocated with oversized buffers]
/// @return Actual NNZ used (dst.nnz is updated)
template <CSRLike SrcT, typename T>
Index unsafe_materialize_col_slice(
    const SrcT& src,
    Span<const Byte> keep_col_mask,
    CustomCSR<T>& dst
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Unsafe slice: Type mismatch");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Unsafe slice: Destination arrays null");
    
    // Build remapping
    std::vector<Index> col_remap(cols);
    Index new_cols = 0;
    
    for (Index j = 0; j < cols; ++j) {
        if (keep_col_mask[j]) {
            col_remap[j] = new_cols++;
        } else {
            col_remap[j] = -1;
        }
    }
    
    // Update dst dimensions (convenience for immediate use)
    dst.rows = rows;
    dst.cols = new_cols;

    // Build indptr + copy data (serial for safety in unsafe mode)
    dst.indptr[0] = 0;
    Index global_write_pos = 0;
    
    for (Index i = 0; i < rows; ++i) {
        auto src_vals = src.row_values(i);
        auto src_idxs = src.row_indices(i);
        Index src_len = src.row_length(i);
        
        Index row_count = 0;
        
        for (Index k = 0; k < src_len; ++k) {
            Index old_col = src_idxs[k];
            
            if (keep_col_mask[old_col]) {
#if !defined(NDEBUG)
                // Safety check in debug mode only
                if (SCL_UNLIKELY(static_cast<Size>(global_write_pos) >= static_cast<Size>(dst.nnz))) {
                    SCL_CHECK_ARG(false, "Unsafe slice: Buffer overflow!");
                }
#endif
                
                dst.indices[global_write_pos] = col_remap[old_col];
                dst.data[global_write_pos] = src_vals[k];
                global_write_pos++;
                row_count++;
            }
        }
        
        dst.indptr[i + 1] = dst.indptr[i] + row_count;
    }
    
    // Update actual NNZ used
    dst.nnz = global_write_pos;
    dst.row_lengths = nullptr;
    
    return global_write_pos;
}

// =============================================================================
// 11. Complete Column Slice (Safe, Returns CustomCSR)
// =============================================================================

/// @brief Complete column slice with automatic memory management.
///
/// Two-pass algorithm wrapped in single function.
/// Returns CustomCSR directly (Custom → Virtual conversion is zero-cost).
///
/// Memory: User provides pre-allocated buffers OR uses returned CustomCSR.
///
/// Example (External Allocation):
/// @code{.cpp}
/// std::vector<Byte> mask(mat.cols, 0);
/// for (auto idx : selected) mask[idx] = 1;
/// 
/// // Pre-allocate
/// std::vector<Real> data(mat.nnz);  // Upper bound
/// std::vector<Index> indices(mat.nnz);
/// std::vector<Index> indptr(mat.rows + 1);
/// 
/// CustomCSR<Real> result(data.data(), indices.data(), indptr.data(),
///                        mat.rows, 0, mat.nnz);
/// 
/// slice_cols_safe(mat, mask, result);
/// // result.nnz now contains actual NNZ (≤ mat.nnz)
/// @endcode
///
/// @tparam SrcT Source matrix type
/// @tparam T Value type
/// @param src Source matrix
/// @param keep_col_mask Bitmap mask [size = src.cols]
/// @param dst Destination matrix [pre-allocated with sufficient capacity]
/// @return Reference to dst
template <CSRLike SrcT, typename T>
CustomCSR<T>& slice_cols_safe(
    const SrcT& src,
    Span<const Byte> keep_col_mask,
    CustomCSR<T>& dst
) {
    // Phase 1: Inspect (with caching)
    std::vector<Index> row_nnzs(src.rows);
    
    Size total_nnz = inspect_col_slice(
        src,
        keep_col_mask,
        MutableSpan<Index>(row_nnzs.data(), row_nnzs.size())
    );
    
    // Validate capacity
    SCL_CHECK_ARG(total_nnz <= static_cast<Size>(dst.nnz),
                  "Slice cols safe: Destination capacity insufficient");

    // Phase 2: Materialize (using cache)
    materialize_col_slice(
        src,
        keep_col_mask,
        dst,
        Span<const Index>(row_nnzs.data(), row_nnzs.size())
    );
    
    return dst;
}

/// @brief UNSAFE: Column slice assuming buffer is large enough.
///
/// Single-pass algorithm that skips inspection phase.
/// User must guarantee dst arrays have sufficient capacity (≥ actual NNZ).
///
/// Typical Usage: Allocate src.nnz as upper bound (acceptable waste).
///
/// Performance: Slightly faster than safe version (no double scan).
///
/// @tparam SrcT Source matrix type
/// @tparam T Value type
/// @param src Source matrix
/// @param keep_col_mask ByteMask: 0=drop, non-zero=keep
/// @param dst Destination matrix [pre-allocated with oversized buffers]
/// @return Reference to dst (dimensions and nnz updated)
template <CSRLike SrcT, typename T>
CustomCSR<T>& slice_cols_unsafe(
    const SrcT& src,
    Span<const Byte> keep_col_mask,
    CustomCSR<T>& dst
) {
    unsafe_materialize_col_slice(src, keep_col_mask, dst);
    return dst;
}

// =============================================================================
// 12. Row Slicing (CSR) - Zero-Copy via Virtual View
// =============================================================================

/// @brief Row slice for CSR (Zero-Copy via Virtual View).
///
/// Row slicing on CSR is the "free" dimension - just create VirtualCSR.
/// No data movement needed! Custom → Virtual conversion is zero-cost.
///
/// IMPORTANT: Lifetime Constraint
/// - The returned VirtualCSR holds a pointer to row_indices.data()
/// - User MUST ensure row_indices array outlives the VirtualCSR
/// - Typical pattern: Store both in same struct/scope
///
/// Example:
/// @code{.cpp}
/// std::vector<Index> rows = {0, 10, 20};  // Keep alive!
/// VirtualCSR<Real> view = slice_rows_virtual(mat, rows);
/// // Use view here...
/// // Ensure 'rows' is not destroyed before 'view'
/// @endcode
///
/// @tparam T Value type
/// @param src Source matrix (CustomCSR)
/// @param row_indices Rows to keep [can be arbitrary order]
/// @return VirtualCSR viewing the selected rows
template <typename T>
VirtualCSR<T> slice_rows_virtual(
    const CustomCSR<T>& src,
    Span<const Index> row_indices
) {
    // Simply create virtual view with row indirection
    return VirtualCSR<T>(src, row_indices);
}

// =============================================================================
// 13. Bitmap-Based Row Slicing (CSC) - Two-Pass Algorithm
// =============================================================================

/// @brief Inspect row slice on CSC using bitmap mask.
///
/// Symmetric to inspect_col_slice but for CSC format.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param mat Source matrix
/// @param keep_row_mask Bitmap mask [size = mat.rows]
/// @param out_col_nnz Optional cache [size = mat.cols]
/// @return Total NNZ required
template <CSCLike MatrixT>
Size inspect_row_slice(
    const MatrixT& mat,
    Span<const Byte> keep_row_mask,
    MutableSpan<Index> out_col_nnz = MutableSpan<Index>()
) {
    const Index rows = mat.rows;
    const Index cols = mat.cols;
    
    SCL_CHECK_DIM(keep_row_mask.size == static_cast<Size>(rows),
                  "Row slice inspect: Mask size must match matrix.rows");
    
    if (out_col_nnz.size > 0) {
        SCL_CHECK_DIM(out_col_nnz.size == static_cast<Size>(cols),
                      "Row slice inspect: Cache size must match matrix.cols");
    }

    const size_t num_threads = scl::threading::Scheduler::get_num_threads();
    std::vector<detail::Accumulator<Size>> partial_sums(num_threads);
    const size_t chunk_size = (static_cast<size_t>(cols) + num_threads - 1) / num_threads;

    scl::threading::parallel_for(0, num_threads, [&](size_t t_id) {
        size_t start = t_id * chunk_size;
        size_t end = std::min(start + chunk_size, static_cast<size_t>(cols));
        
        Size local_total = 0;

        for (size_t j = start; j < end; ++j) {
            Index col_idx = static_cast<Index>(j);
            auto col_idxs = mat.col_indices(col_idx);
            Index col_len = mat.col_length(col_idx);
            
            Index count = 0;
            
            for (Index k = 0; k < col_len; ++k) {
                if (keep_row_mask[col_idxs[k]]) {
                    count++;
                }
            }
            
            local_total += static_cast<Size>(count);
            
            if (out_col_nnz.size > 0) {
                out_col_nnz[col_idx] = count;
            }
        }
        
        partial_sums[t_id].value = local_total;
    });

    Size total_nnz = 0;
    for (const auto& acc : partial_sums) {
        total_nnz += acc.value;
    }
    
    return total_nnz;
}

/// @brief Materialize row slice on CSC.
///
/// Returns CustomCSC directly.
///
/// @tparam SrcT Source matrix type (any CSC-like)
/// @tparam T Value type
/// @param src Source matrix
/// @param keep_row_mask Bitmap mask
/// @param dst Destination matrix [pre-allocated]
/// @param cached_col_nnz Optional cache from inspect
/// @return Reference to dst
template <CSCLike SrcT, typename T>
CustomCSC<T>& materialize_row_slice(
    const SrcT& src,
    Span<const Byte> keep_row_mask,
    CustomCSC<T>& dst,
    Span<const Index> cached_col_nnz = Span<const Index>()
) {
    using SrcValueType = typename SrcT::ValueType;
    static_assert(std::is_same_v<SrcValueType, T>,
                  "Slice: Type mismatch");
    
    const Index rows = src.rows;
    const Index cols = src.cols;
    
    SCL_CHECK_DIM(keep_row_mask.size == static_cast<Size>(rows),
                  "Row slice: Mask size mismatch");
    SCL_CHECK_ARG(dst.data != nullptr && dst.indices != nullptr && dst.indptr != nullptr,
                  "Row slice: Destination not allocated");
    SCL_CHECK_DIM(dst.cols == cols, "Row slice: Destination cols mismatch");

    // Build row remapping
    std::vector<Index> row_remap(rows);
    Index new_row_count = 0;
    
    for (Index i = 0; i < rows; ++i) {
        if (keep_row_mask[i]) {
            row_remap[i] = new_row_count++;
        } else {
            row_remap[i] = -1;
        }
    }
    
    dst.rows = new_row_count;

    // Build indptr
    dst.indptr[0] = 0;
    
    if (cached_col_nnz.size > 0) {
        for (Index j = 0; j < cols; ++j) {
            dst.indptr[j + 1] = dst.indptr[j] + cached_col_nnz[j];
        }
    } else {
        for (Index j = 0; j < cols; ++j) {
            auto col_idxs = src.col_indices(j);
            Index col_len = src.col_length(j);
            
            Index count = 0;
            for (Index k = 0; k < col_len; ++k) {
                if (keep_row_mask[col_idxs[k]]) count++;
            }
            
            dst.indptr[j + 1] = dst.indptr[j] + count;
        }
    }
    
    const Size total_nnz = static_cast<Size>(dst.indptr[cols]);
    SCL_CHECK_ARG(total_nnz <= static_cast<Size>(dst.nnz),
                  "Row slice: Buffer overflow");
    
    dst.nnz = static_cast<Index>(total_nnz);
    dst.col_lengths = nullptr;

    // Parallel copy and remap
    scl::threading::parallel_for(0, static_cast<size_t>(cols), [&](size_t j) {
        Index col_idx = static_cast<Index>(j);
        Index dst_start = dst.indptr[col_idx];
        
        auto src_vals = src.col_values(col_idx);
        auto src_idxs = src.col_indices(col_idx);
        Index src_len = src.col_length(col_idx);
        
        Index write_pos = 0;
        
        for (Index k = 0; k < src_len; ++k) {
            Index old_row = src_idxs[k];
            
            if (keep_row_mask[old_row]) {
                dst.indices[dst_start + write_pos] = row_remap[old_row];
                dst.data[dst_start + write_pos] = src_vals[k];
                write_pos++;
            }
        }
    });
    
    return dst;
}

/// @brief Column slice for CSC (Zero-Copy via Virtual View).
///
/// Column slicing on CSC is the "free" dimension - just create VirtualCSC.
/// No data movement needed! Custom → Virtual conversion is zero-cost.
///
/// IMPORTANT: Lifetime Constraint
/// - The returned VirtualCSC holds a pointer to col_indices.data()
/// - User MUST ensure col_indices array outlives the VirtualCSC
/// - Typical pattern: Store both in same struct/scope
///
/// Example:
/// @code{.cpp}
/// std::vector<Index> cols = {5, 15, 25};  // Keep alive!
/// VirtualCSC<Real> view = slice_cols_virtual(mat, cols);
/// // Use view here...
/// // Ensure 'cols' is not destroyed before 'view'
/// @endcode
///
/// @tparam T Value type
/// @param src Source matrix
/// @param col_indices Columns to keep [can be arbitrary order]
/// @return VirtualCSC viewing selected columns
template <typename T>
VirtualCSC<T> slice_cols_virtual(
    const CustomCSC<T>& src,
    Span<const Index> col_indices
) {
    return VirtualCSC<T>(src, col_indices);
}

} // namespace scl::utils

