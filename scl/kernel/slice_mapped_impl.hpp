#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/slice.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file slice_mapped_impl.hpp
/// @brief High-Performance Slicing for Mapped Sparse Matrices
///
/// Key Optimizations:
/// 1. Thread-Local Accumulation: Avoid atomic contention
/// 2. Streaming Prefetch: Efficient mapped data access
/// 3. SIMD Mask Counting: Vectorized popcount
/// 4. Bulk Memory Operations: memcpy for large segments
/// 5. Adaptive Parallelism: Based on matrix size
///
/// Both operations return newly allocated SliceResult.
// =============================================================================

namespace scl::kernel::slice::mapped {

// =============================================================================
// Constants
// =============================================================================

constexpr Size PARALLEL_THRESHOLD = 512;
constexpr Size MEMCPY_THRESHOLD = 8;
constexpr Size PREFETCH_DIST = 4;

// =============================================================================
// Detail Utilities
// =============================================================================

namespace detail {

/// @brief Chunked parallel reduction (avoids atomic contention)
template <typename F>
SCL_FORCE_INLINE Index parallel_reduce(Size n, F&& get_value) {
    if (n < PARALLEL_THRESHOLD) {
        Index total = 0;
        for (Size i = 0; i < n; ++i) {
            total += get_value(i);
        }
        return total;
    }
    
    const Size num_threads = scl::threading::Scheduler::get_num_threads();
    const Size chunk_size = (n + num_threads - 1) / num_threads;
    
    std::vector<Index> partial_sums(num_threads, 0);
    
    scl::threading::parallel_for(Size(0), num_threads, [&](size_t tid) {
        Size start = tid * chunk_size;
        Size end = std::min(start + chunk_size, n);
        
        Index local_sum = 0;
        for (Size i = start; i < end; ++i) {
            local_sum += get_value(i);
        }
        partial_sums[tid] = local_sum;
    });
    
    Index total = 0;
    for (Size t = 0; t < num_threads; ++t) {
        total += partial_sums[t];
    }
    return total;
}

/// @brief SIMD-friendly mask counting for indices
SCL_FORCE_INLINE Index count_masked_fast(
    const Index* SCL_RESTRICT indices,
    Size len,
    const uint8_t* SCL_RESTRICT mask
) {
    Index count = 0;
    
    Size k = 0;
    for (; k + 8 <= len; k += 8) {
        count += mask[indices[k + 0]];
        count += mask[indices[k + 1]];
        count += mask[indices[k + 2]];
        count += mask[indices[k + 3]];
        count += mask[indices[k + 4]];
        count += mask[indices[k + 5]];
        count += mask[indices[k + 6]];
        count += mask[indices[k + 7]];
    }
    
    for (; k < len; ++k) {
        count += mask[indices[k]];
    }
    
    return count;
}

/// @brief Fast memcpy with prefetch hint
template <typename T>
SCL_FORCE_INLINE void fast_copy(
    T* SCL_RESTRICT dst,
    const T* SCL_RESTRICT src,
    Size count
) {
    if (count >= MEMCPY_THRESHOLD) {
        std::memcpy(dst, src, count * sizeof(T));
    } else {
        for (Size k = 0; k < count; ++k) {
            dst[k] = src[k];
        }
    }
}

} // namespace detail

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

    return detail::parallel_reduce(n_keep, [&](Size i) -> Index {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
        return scl::primary_length(matrix, idx);
    });
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

    // Build indptr first (sequential prefix sum)
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[i];
        Index len = scl::primary_length(matrix, src_idx);
        out_indptr[i + 1] = out_indptr[i] + len;
    }

    // Prefetch ahead
    kernel::mapped::hint_prefetch(matrix);

    // Parallel copy data and indices
    if (n_keep < PARALLEL_THRESHOLD) {
        for (Size i = 0; i < n_keep; ++i) {
            Index src_idx = keep_indices[i];
            auto values = scl::primary_values(matrix, src_idx);
            auto indices = scl::primary_indices(matrix, src_idx);
            Index len = static_cast<Index>(values.len);

            if (len == 0) continue;

            Index dst_start = out_indptr[i];
            detail::fast_copy(out_data.data() + dst_start, values.ptr, static_cast<Size>(len));
            detail::fast_copy(out_indices.data() + dst_start, indices.ptr, static_cast<Size>(len));
        }
    } else {
        scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
            Index src_idx = keep_indices[i];
            auto values = scl::primary_values(matrix, src_idx);
            auto indices = scl::primary_indices(matrix, src_idx);
            Index len = static_cast<Index>(values.len);

            if (len == 0) return;

            // Prefetch next row
            if (i + PREFETCH_DIST < n_keep) {
                Index next_idx = keep_indices[i + PREFETCH_DIST];
                auto next_vals = scl::primary_values(matrix, next_idx);
                SCL_PREFETCH_READ(next_vals.ptr, 0);
            }

            Index dst_start = out_indptr[i];
            detail::fast_copy(out_data.data() + dst_start, values.ptr, static_cast<Size>(len));
            detail::fast_copy(out_indices.data() + dst_start, indices.ptr, static_cast<Size>(len));
        });
    }
}

/// @brief Primary dimension slice for MappedCustomSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
SliceResult<T> slice_primary_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
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
    const uint8_t* mask_ptr = mask.data();

    kernel::mapped::hint_prefetch(matrix);

    return detail::parallel_reduce(static_cast<Size>(n_primary), [&](Size p) -> Index {
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        return detail::count_masked_fast(indices.ptr, indices.len, mask_ptr);
    });
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
    const uint8_t* mask_ptr = mask.data();
    const Index* new_idx_ptr = new_indices.data();

    // First pass: compute output indptr
    out_indptr[0] = 0;
    for (Index p = 0; p < n_primary; ++p) {
        auto indices = scl::primary_indices(matrix, p);
        Index count = detail::count_masked_fast(indices.ptr, indices.len, mask_ptr);
        out_indptr[p + 1] = out_indptr[p] + count;
    }

    // Second pass: parallel copy filtered data
    if (n_primary < static_cast<Index>(PARALLEL_THRESHOLD)) {
        for (Index p = 0; p < n_primary; ++p) {
            auto values = scl::primary_values(matrix, p);
            auto indices = scl::primary_indices(matrix, p);
            Index dst_pos = out_indptr[p];

            for (Size k = 0; k < values.len; ++k) {
                Index old_idx = indices.ptr[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values.ptr[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        }
    } else {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
            auto values = scl::primary_values(matrix, static_cast<Index>(p));
            auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
            Index dst_pos = out_indptr[p];

            for (Size k = 0; k < values.len; ++k) {
                Index old_idx = indices.ptr[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values.ptr[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        });
    }
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

    Index new_secondary = slice::detail::build_index_mapping(
        mask.data(),
        new_indices.data(),
        secondary_dim
    );

    Index out_nnz = inspect_filter_secondary_mapped(matrix, mask);

    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(out_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(out_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(primary_dim) + 1);

    materialize_filter_secondary_mapped(
        matrix,
        mask,
        Array<const Index>(new_indices.data(), new_indices.size()),
        data_handle.template as_span<T>(),
        indices_handle.template as_span<Index>(),
        indptr_handle.template as_span<Index>()
    );

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

    return detail::parallel_reduce(n_keep, [&](Size i) -> Index {
        Index idx = keep_indices[i];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
        return scl::primary_length(matrix, idx);
    });
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
    if (n_keep < PARALLEL_THRESHOLD) {
        for (Size i = 0; i < n_keep; ++i) {
            Index src_idx = keep_indices[i];
            auto values = scl::primary_values(matrix, src_idx);
            auto indices = scl::primary_indices(matrix, src_idx);
            Index len = static_cast<Index>(values.len);

            if (len == 0) continue;

            Index dst_start = out_indptr[i];
            detail::fast_copy(out_data.data() + dst_start, values.ptr, static_cast<Size>(len));
            detail::fast_copy(out_indices.data() + dst_start, indices.ptr, static_cast<Size>(len));
        }
    } else {
        scl::threading::parallel_for(Size(0), n_keep, [&](size_t i) {
            Index src_idx = keep_indices[i];
            auto values = scl::primary_values(matrix, src_idx);
            auto indices = scl::primary_indices(matrix, src_idx);
            Index len = static_cast<Index>(values.len);

            if (len == 0) return;

            Index dst_start = out_indptr[i];
            detail::fast_copy(out_data.data() + dst_start, values.ptr, static_cast<Size>(len));
            detail::fast_copy(out_indices.data() + dst_start, indices.ptr, static_cast<Size>(len));
        });
    }
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

/// @brief Inspect secondary dimension filter for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
Index inspect_filter_secondary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index n_primary = scl::primary_size(matrix);
    const uint8_t* mask_ptr = mask.data();

    return detail::parallel_reduce(static_cast<Size>(n_primary), [&](Size p) -> Index {
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        return detail::count_masked_fast(indices.ptr, indices.len, mask_ptr);
    });
}

/// @brief Secondary dimension filter for MappedVirtualSparse - allocating
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
SliceResult<T> filter_secondary_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = scl::secondary_size(matrix);
    const Index primary_dim = scl::primary_size(matrix);
    const uint8_t* mask_ptr = mask.data();

    SCL_CHECK_DIM(mask.size() >= static_cast<Size>(secondary_dim), "Mask size mismatch");

    // Build new index mapping
    auto new_indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(secondary_dim));
    auto new_indices = new_indices_handle.template as_span<Index>();

    Index new_secondary = slice::detail::build_index_mapping(
        mask.data(),
        new_indices.data(),
        secondary_dim
    );
    const Index* new_idx_ptr = new_indices.data();

    // Count total nnz
    Index out_nnz = inspect_filter_secondary_mapped(matrix, mask);

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
        Index count = detail::count_masked_fast(indices.ptr, indices.len, mask_ptr);
        out_indptr[p + 1] = out_indptr[p] + count;
    }

    // Parallel copy
    if (primary_dim < static_cast<Index>(PARALLEL_THRESHOLD)) {
        for (Index p = 0; p < primary_dim; ++p) {
            auto values = scl::primary_values(matrix, p);
            auto indices = scl::primary_indices(matrix, p);
            Index dst_pos = out_indptr[p];

            for (Size k = 0; k < values.len; ++k) {
                Index old_idx = indices.ptr[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values.ptr[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        }
    } else {
        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            auto values = scl::primary_values(matrix, static_cast<Index>(p));
            auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
            Index dst_pos = out_indptr[p];

            for (Size k = 0; k < values.len; ++k) {
                Index old_idx = indices.ptr[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values.ptr[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        });
    }

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

/// @brief Dispatch primary slice for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto slice_primary_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Index> keep_indices
) {
    return slice_primary_mapped(matrix, keep_indices);
}

/// @brief Dispatch secondary filter for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
auto filter_secondary_mapped_dispatch(
    const MatrixT& matrix,
    Array<const uint8_t> mask
) {
    return filter_secondary_mapped(matrix, mask);
}

} // namespace scl::kernel::slice::mapped
