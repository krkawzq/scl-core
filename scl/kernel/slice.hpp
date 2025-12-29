#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/algo.hpp"
#include "scl/kernel/sparse.hpp"  // for from_contiguous_arrays
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/scheduler.hpp"

#include <cstring>

// =============================================================================
// FILE: scl/kernel/slice.hpp
// BRIEF: Sparse matrix slicing with optimized parallel processing
// =============================================================================

namespace scl::kernel::slice {

constexpr Size PARALLEL_THRESHOLD_ROWS = 512;
constexpr Size PARALLEL_THRESHOLD_NNZ = 10000;
constexpr Size MEMCPY_THRESHOLD = 8;

namespace detail {

template <typename F>
SCL_FORCE_INLINE Index parallel_reduce_nnz(Size n, F&& get_nnz) {
    if (n < PARALLEL_THRESHOLD_ROWS) {
        Index total = 0;
        for (Size i = 0; i < n; ++i) {
            total += get_nnz(i);
        }
        return total;
    }

    const Size num_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());
    const Size chunk_size = (n + num_threads - 1) / num_threads;

    auto partial_sums_ptr = scl::memory::aligned_alloc<Index>(num_threads, SCL_ALIGNMENT);
    Index* partial_sums = partial_sums_ptr.get();
    scl::algo::zero(partial_sums, num_threads);

    scl::threading::parallel_for(Size(0), num_threads, [&](Size tid) {
        Size start = tid * chunk_size;
        Size end = scl::algo::min2(start + chunk_size, n);

        Index local_sum = 0;
        for (Size i = start; i < end; ++i) {
            local_sum += get_nnz(i);
        }
        partial_sums[tid] = local_sum;
    });

    Index total = 0;
    for (Size t = 0; t < num_threads; ++t) {
        total += partial_sums[t];
    }

    // partial_sums_ptr automatically freed
    return total;
}

SCL_FORCE_INLINE Index count_masked_fast(
    const Index* SCL_RESTRICT indices,
    Index len,
    const uint8_t* SCL_RESTRICT mask
) {
    Index count = 0;
    
    Index k = 0;
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

template <typename T>
SCL_FORCE_INLINE void fast_copy_with_prefetch(
    T* SCL_RESTRICT dst,
    const T* SCL_RESTRICT src,
    Size count
) {
    constexpr Size PREFETCH_DIST = 16;
    
    if (count >= MEMCPY_THRESHOLD) {
        if (count > PREFETCH_DIST) {
            SCL_PREFETCH_READ(src + PREFETCH_DIST, 0);
        }
        std::memcpy(dst, src, count * sizeof(T));
    } else {
        for (Size k = 0; k < count; ++k) {
            dst[k] = src[k];
        }
    }
}

template <typename T>
void parallel_bulk_copy(
    T* SCL_RESTRICT dst,
    const T* SCL_RESTRICT src,
    const Index* offsets_dst,
    const Index* offsets_src,
    Size n_segments
) {
    if (n_segments < PARALLEL_THRESHOLD_ROWS) {
        for (Size i = 0; i < n_segments; ++i) {
            Index len = offsets_dst[i + 1] - offsets_dst[i];
            if (len > 0) {
                fast_copy_with_prefetch(
                    dst + offsets_dst[i],
                    src + offsets_src[i],
                    static_cast<Size>(len)
                );
            }
        }
        return;
    }
    
    scl::threading::parallel_for(Size(0), n_segments, [&](Size i) {
        Index len = offsets_dst[i + 1] - offsets_dst[i];
        if (len > 0) {
            fast_copy_with_prefetch(
                dst + offsets_dst[i],
                src + offsets_src[i],
                static_cast<Size>(len)
            );
        }
    });
}

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

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
Index inspect_slice_primary(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.len;
    const Index n_primary = matrix.primary_dim();
    
    if (n_keep == 0) return 0;
    
    return detail::parallel_reduce_nnz(n_keep, [&](Size i) -> Index {
        Index idx = keep_indices[static_cast<Index>(i)];
        SCL_CHECK_ARG(idx >= 0 && idx < n_primary, "Slice: Index out of bounds");
        return matrix.primary_length_unsafe(idx);
    });
}

template <typename T, bool IsCSR>
void materialize_slice_primary(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Size n_keep = keep_indices.len;
    
    SCL_CHECK_DIM(out_indptr.len >= n_keep + 1, "Output indptr too small");
    
    out_indptr[0] = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index src_idx = keep_indices[static_cast<Index>(i)];
        Index len = matrix.primary_length_unsafe(src_idx);
        out_indptr[static_cast<Index>(i) + 1] = out_indptr[static_cast<Index>(i)] + len;
    }
    
    if (n_keep < PARALLEL_THRESHOLD_ROWS) {
        for (Size i = 0; i < n_keep; ++i) {
            Index src_idx = keep_indices[static_cast<Index>(i)];
            Index len = matrix.primary_length_unsafe(src_idx);
            if (len == 0) continue;
            
            Index dst_start = out_indptr[static_cast<Index>(i)];
            auto src_values = matrix.primary_values_unsafe(src_idx);
            auto src_indices = matrix.primary_indices_unsafe(src_idx);
            
            detail::fast_copy_with_prefetch(
                out_data.ptr + dst_start,
                src_values.ptr,
                static_cast<Size>(len)
            );
            
            detail::fast_copy_with_prefetch(
                out_indices.ptr + dst_start,
                src_indices.ptr,
                static_cast<Size>(len)
            );
        }
    } else {
        scl::threading::parallel_for(Size(0), n_keep, [&](Size i) {
            Index src_idx = keep_indices[static_cast<Index>(i)];
            Index len = matrix.primary_length_unsafe(src_idx);
            if (len == 0) return;
            
            Index dst_start = out_indptr[static_cast<Index>(i)];
            auto src_values = matrix.primary_values_unsafe(src_idx);
            auto src_indices = matrix.primary_indices_unsafe(src_idx);
            
            detail::fast_copy_with_prefetch(
                out_data.ptr + dst_start,
                src_values.ptr,
                static_cast<Size>(len)
            );
            
            detail::fast_copy_with_prefetch(
                out_indices.ptr + dst_start,
                src_indices.ptr,
                static_cast<Size>(len)
            );
        });
    }
}

template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_primary(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices
) {
    const Size n_keep = keep_indices.len;
    
    Index out_nnz = inspect_slice_primary(matrix, keep_indices);
    
    if (out_nnz == 0 || n_keep == 0) {
        Index new_rows = IsCSR ? static_cast<Index>(n_keep) : matrix.rows();
        Index new_cols = IsCSR ? matrix.cols() : static_cast<Index>(n_keep);
        return Sparse<T, IsCSR>::zeros(new_rows, new_cols);
    }
    
    auto data_ptr_obj = scl::memory::aligned_alloc<T>(static_cast<Size>(out_nnz), SCL_ALIGNMENT);
    auto indices_ptr_obj = scl::memory::aligned_alloc<Index>(static_cast<Size>(out_nnz), SCL_ALIGNMENT);
    auto indptr_ptr_obj = scl::memory::aligned_alloc<Index>(n_keep + 1, SCL_ALIGNMENT);
    
    if (!data_ptr_obj || !indices_ptr_obj || !indptr_ptr_obj) {
        return {};
    }
    
    T* data_ptr = data_ptr_obj.release();
    Index* indices_ptr = indices_ptr_obj.release();
    Index* indptr_ptr = indptr_ptr_obj.release();
    
    materialize_slice_primary(
        matrix,
        keep_indices,
        Array<T>(data_ptr, static_cast<Size>(out_nnz)),
        Array<Index>(indices_ptr, static_cast<Size>(out_nnz)),
        Array<Index>(indptr_ptr, n_keep + 1)
    );
    
    Index new_rows = IsCSR ? static_cast<Index>(n_keep) : matrix.rows();
    Index new_cols = IsCSR ? matrix.cols() : static_cast<Index>(n_keep);
    
    // Use from_contiguous_arrays with take_ownership to properly register
    // memory with registry for automatic lifecycle management
    return scl::kernel::sparse::from_contiguous_arrays<T, IsCSR>(
        data_ptr, indices_ptr, indptr_ptr,
        new_rows, new_cols, out_nnz,
        true  // take_ownership - register with registry
    );
}

template <typename T, bool IsCSR>
Index inspect_filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index n_primary = matrix.primary_dim();
    const uint8_t* mask_ptr = mask.ptr;
    
    return detail::parallel_reduce_nnz(static_cast<Size>(n_primary), [&](Size p) -> Index {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        if (len == 0) return 0;
        
        auto indices = matrix.primary_indices_unsafe(idx);
        return detail::count_masked_fast(indices.ptr, len, mask_ptr);
    });
}

template <typename T, bool IsCSR>
void materialize_filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask,
    Array<const Index> new_indices,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr
) {
    const Index n_primary = matrix.primary_dim();
    const uint8_t* mask_ptr = mask.ptr;
    const Index* new_idx_ptr = new_indices.ptr;
    
    out_indptr[0] = 0;
    for (Index p = 0; p < n_primary; ++p) {
        const Index idx = p;
        const Index len = matrix.primary_length_unsafe(idx);
        
        if (len == 0) {
            out_indptr[p + 1] = out_indptr[p];
            continue;
        }
        
        auto indices = matrix.primary_indices_unsafe(idx);
        Index count = detail::count_masked_fast(indices.ptr, len, mask_ptr);
        out_indptr[p + 1] = out_indptr[p] + count;
    }
    
    if (n_primary < static_cast<Index>(PARALLEL_THRESHOLD_ROWS)) {
        for (Index p = 0; p < n_primary; ++p) {
            const Index idx = p;
            const Index len = matrix.primary_length_unsafe(idx);
            if (len == 0) continue;
            
            auto values = matrix.primary_values_unsafe(idx);
            auto indices = matrix.primary_indices_unsafe(idx);
            Index dst_pos = out_indptr[p];
            
            for (Index k = 0; k < len; ++k) {
                Index old_idx = indices[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        }
    } else {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](Size p) {
            const auto idx = static_cast<Index>(p);
            const Index len = matrix.primary_length_unsafe(idx);
            if (len == 0) return;
            
            auto values = matrix.primary_values_unsafe(idx);
            auto indices = matrix.primary_indices_unsafe(idx);
            Index dst_pos = out_indptr[static_cast<Index>(p)];
            
            for (Index k = 0; k < len; ++k) {
                Index old_idx = indices[k];
                if (mask_ptr[old_idx]) {
                    out_data[dst_pos] = values[k];
                    out_indices[dst_pos] = new_idx_ptr[old_idx];
                    dst_pos++;
                }
            }
        });
    }
}

template <typename T, bool IsCSR>
Sparse<T, IsCSR> filter_secondary(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> mask
) {
    const Index secondary_dim = matrix.secondary_dim();
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(mask.len >= static_cast<Size>(secondary_dim), "Mask size mismatch");

    auto new_indices_ptr_obj = scl::memory::aligned_alloc<Index>(static_cast<Size>(secondary_dim), SCL_ALIGNMENT);
    if (!new_indices_ptr_obj) {
        Index new_rows = IsCSR ? matrix.rows() : 0;
        Index new_cols = IsCSR ? 0 : matrix.cols();
        return Sparse<T, IsCSR>::zeros(new_rows, new_cols);
    }
    Index* new_indices = new_indices_ptr_obj.release();
    
    Index new_secondary = detail::build_index_mapping(
        mask.ptr,
        new_indices,
        secondary_dim
    );

    Index out_nnz = inspect_filter_secondary(matrix, mask);

    if (out_nnz == 0) {
        scl::memory::aligned_free(new_indices, SCL_ALIGNMENT);
        Index new_rows = IsCSR ? matrix.rows() : static_cast<Index>(new_secondary);
        Index new_cols = IsCSR ? static_cast<Index>(new_secondary) : matrix.cols();
        return Sparse<T, IsCSR>::zeros(new_rows, new_cols);
    }

    auto data_ptr_obj = scl::memory::aligned_alloc<T>(static_cast<Size>(out_nnz), SCL_ALIGNMENT);
    auto indices_ptr_obj = scl::memory::aligned_alloc<Index>(static_cast<Size>(out_nnz), SCL_ALIGNMENT);
    auto indptr_ptr_obj = scl::memory::aligned_alloc<Index>(static_cast<Size>(primary_dim) + 1, SCL_ALIGNMENT);

    if (!data_ptr_obj || !indices_ptr_obj || !indptr_ptr_obj) {
        scl::memory::aligned_free(new_indices, SCL_ALIGNMENT);
        return {};
    }

    T* data_ptr = data_ptr_obj.release();
    Index* indices_ptr = indices_ptr_obj.release();
    Index* indptr_ptr = indptr_ptr_obj.release();

    materialize_filter_secondary(
        matrix,
        mask,
        Array<const Index>(new_indices, static_cast<Size>(secondary_dim)),
        Array<T>(data_ptr, static_cast<Size>(out_nnz)),
        Array<Index>(indices_ptr, static_cast<Size>(out_nnz)),
        Array<Index>(indptr_ptr, static_cast<Size>(primary_dim) + 1)
    );

    scl::memory::aligned_free(new_indices, SCL_ALIGNMENT);

    Index new_rows = IsCSR ? matrix.rows() : static_cast<Index>(new_secondary);
    Index new_cols = IsCSR ? static_cast<Index>(new_secondary) : matrix.cols();

    // Use from_contiguous_arrays with take_ownership to properly register
    // memory with registry for automatic lifecycle management
    return scl::kernel::sparse::from_contiguous_arrays<T, IsCSR>(
        data_ptr, indices_ptr, indptr_ptr,
        new_rows, new_cols, out_nnz,
        true  // take_ownership - register with registry
    );
}

} // namespace scl::kernel::slice

