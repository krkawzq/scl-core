#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cstring>
#include <algorithm>
#include <vector>

// =============================================================================
/// @file merge_mapped_impl.hpp
/// @brief Matrix Merging for Mapped Sparse Matrices (Extreme Performance)
///
/// Optimizations:
/// - Parallel NNZ computation with SIMD reduction
/// - SIMD index offset for column concatenation
/// - Streaming copy with prefetch
/// - Load-balanced parallel copy
/// - Batch merge for multiple matrices
///
/// Operations:
/// - compute_total_nnz_mapped: Parallel NNZ computation
/// - materialize_merged_mapped: Streaming materialization
/// - vstack_mapped: Vertical concatenation
/// - hstack_mapped: Horizontal concatenation
/// - merge_n_mapped: Batch merge multiple matrices
// =============================================================================

namespace scl::kernel::merge::mapped {

// =============================================================================
// SECTION 1: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD add offset to index array
SCL_FORCE_INLINE void add_offset_simd(
    const Index* SCL_RESTRICT src,
    Index* SCL_RESTRICT dst,
    Size count,
    Index offset
) {
    if (offset == 0) {
        std::memcpy(dst, src, count * sizeof(Index));
        return;
    }

    namespace s = scl::simd;
    const s::IndexTag d;
    const size_t lanes = s::Lanes(d);

    const auto v_offset = s::Set(d, offset);

    Size i = 0;
    const Size simd_end = count - (count % (lanes * 2));

    // 2-way unrolled SIMD
    for (; i < simd_end; i += lanes * 2) {
        auto v0 = s::Load(d, src + i);
        auto v1 = s::Load(d, src + i + lanes);
        s::Store(s::Add(v0, v_offset), d, dst + i);
        s::Store(s::Add(v1, v_offset), d, dst + i + lanes);
    }

    for (; i + lanes <= count; i += lanes) {
        auto v = s::Load(d, src + i);
        s::Store(s::Add(v, v_offset), d, dst + i);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] + offset;
    }
}

/// @brief Parallel sum of Index array
inline Index parallel_sum_index(const Index* values, Index count) {
    if (count <= 10000) {
        Index sum = 0;
        for (Index i = 0; i < count; ++i) {
            sum += values[i];
        }
        return sum;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Index chunk = (count + static_cast<Index>(n_threads) - 1) / static_cast<Index>(n_threads);

    std::vector<Index> partial(n_threads, 0);

    scl::threading::parallel_for(0, n_threads, [&](size_t tid) {
        Index start = static_cast<Index>(tid) * chunk;
        Index end = std::min(start + chunk, count);

        Index local_sum = 0;
        for (Index i = start; i < end; ++i) {
            local_sum += values[i];
        }
        partial[tid] = local_sum;
    });

    Index total = 0;
    for (size_t i = 0; i < n_threads; ++i) {
        total += partial[i];
    }
    return total;
}

/// @brief Copy with prefetch hints
template <typename T>
SCL_FORCE_INLINE void copy_prefetch(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size count
) {
    constexpr Size prefetch_dist = 16;

    for (Size i = 0; i < count; ++i) {
        if (i + prefetch_dist < count) {
            SCL_PREFETCH_READ(src + i + prefetch_dist, 0);
        }
        dst[i] = src[i];
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: NNZ Computation
// =============================================================================

/// @brief Compute total NNZ for merged matrix (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
Index compute_total_nnz_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2
) {
    const Index nnz1 = matrix1.indptr[scl::primary_size(matrix1)];
    const Index nnz2 = matrix2.indptr[scl::primary_size(matrix2)];
    return nnz1 + nnz2;
}

/// @brief Compute total NNZ for merged matrix (MappedVirtualSparse) - Parallel
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
Index compute_total_nnz_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix1,
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix2
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);

    Index nnz1 = detail::parallel_sum_index(matrix1.lengths, primary_dim1);
    Index nnz2 = detail::parallel_sum_index(matrix2.lengths, primary_dim2);

    return nnz1 + nnz2;
}

/// @brief Compute NNZ for single MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
Index compute_nnz_mapped(const scl::io::MappedVirtualSparse<T, IsCSR>& matrix) {
    const Index primary_dim = scl::primary_size(matrix);
    return detail::parallel_sum_index(matrix.lengths, primary_dim);
}

// =============================================================================
// SECTION 3: Vertical Stack (vstack)
// =============================================================================

/// @brief vstack two MappedCustomSparse matrices into OwnedSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void vstack_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index nnz1 = matrix1.indptr[primary_dim1];
    const Index nnz2 = matrix2.indptr[primary_dim2];
    const Index total_nnz = nnz1 + nnz2;

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "vstack_mapped: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "vstack_mapped: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(total_primary) + 1,
                  "vstack_mapped: Output indptr too small");

    // Parallel copy matrix1 (streaming from mmap)
    constexpr Size CHUNK_SIZE = 65536;
    const Size n_chunks1 = (static_cast<Size>(nnz1) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (n_chunks1 > 1) {
        scl::threading::parallel_for(0, n_chunks1, [&](size_t c) {
            Size start = c * CHUNK_SIZE;
            Size end = std::min(start + CHUNK_SIZE, static_cast<Size>(nnz1));

            // Prefetch next chunk
            if (c + 1 < n_chunks1) {
                SCL_PREFETCH_READ(matrix1.data + end, 0);
                SCL_PREFETCH_READ(matrix1.indices + end, 0);
            }

            std::memcpy(out_data.ptr + start, matrix1.data + start, (end - start) * sizeof(T));
            std::memcpy(out_indices.ptr + start, matrix1.indices + start, (end - start) * sizeof(Index));
        });
    } else {
        std::memcpy(out_data.ptr, matrix1.data, nnz1 * sizeof(T));
        std::memcpy(out_indices.ptr, matrix1.indices, nnz1 * sizeof(Index));
    }

    // Copy indptr for matrix1
    std::memcpy(out_indptr.ptr, matrix1.indptr, (primary_dim1 + 1) * sizeof(Index));

    // Parallel copy matrix2
    const Size n_chunks2 = (static_cast<Size>(nnz2) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (n_chunks2 > 1) {
        scl::threading::parallel_for(0, n_chunks2, [&](size_t c) {
            Size start = c * CHUNK_SIZE;
            Size end = std::min(start + CHUNK_SIZE, static_cast<Size>(nnz2));

            if (c + 1 < n_chunks2) {
                SCL_PREFETCH_READ(matrix2.data + end, 0);
                SCL_PREFETCH_READ(matrix2.indices + end, 0);
            }

            std::memcpy(out_data.ptr + nnz1 + start, matrix2.data + start, (end - start) * sizeof(T));
            std::memcpy(out_indices.ptr + nnz1 + start, matrix2.indices + start, (end - start) * sizeof(Index));
        });
    } else {
        std::memcpy(out_data.ptr + nnz1, matrix2.data, nnz2 * sizeof(T));
        std::memcpy(out_indices.ptr + nnz1, matrix2.indices, nnz2 * sizeof(Index));
    }

    // Update indptr for matrix2 portion (parallel)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim2) + 1, [&](size_t p) {
        out_indptr[primary_dim1 + p] = matrix2.indptr[p] + nnz1;
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = std::max(scl::secondary_size(matrix1), scl::secondary_size(matrix2));
    } else {
        out_result.cols = total_primary;
        out_result.rows = std::max(scl::secondary_size(matrix1), scl::secondary_size(matrix2));
    }
}

/// @brief vstack two MappedVirtualSparse matrices
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void vstack_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix1,
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix2,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index total_nnz = compute_total_nnz_mapped(matrix1, matrix2);

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "vstack_mapped: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "vstack_mapped: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(total_primary) + 1,
                  "vstack_mapped: Output indptr too small");

    // Build indptr
    out_indptr[0] = 0;
    Index pos = 0;
    for (Index p = 0; p < primary_dim1; ++p) {
        pos += matrix1.lengths[p];
        out_indptr[p + 1] = pos;
    }
    for (Index p = 0; p < primary_dim2; ++p) {
        pos += matrix2.lengths[p];
        out_indptr[primary_dim1 + p + 1] = pos;
    }

    // Parallel copy from matrix1
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim1), [&](size_t p) {
        Index len = matrix1.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[p];
        const T* src_data = static_cast<const T*>(matrix1.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix1.indices_ptrs[p]);

        std::memcpy(out_data.ptr + dst_start, src_data, len * sizeof(T));
        std::memcpy(out_indices.ptr + dst_start, src_indices, len * sizeof(Index));
    });

    // Parallel copy from matrix2
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim2), [&](size_t p) {
        Index len = matrix2.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[primary_dim1 + p];
        const T* src_data = static_cast<const T*>(matrix2.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix2.indices_ptrs[p]);

        std::memcpy(out_data.ptr + dst_start, src_data, len * sizeof(T));
        std::memcpy(out_indices.ptr + dst_start, src_indices, len * sizeof(Index));
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = std::max(scl::secondary_size(matrix1), scl::secondary_size(matrix2));
    } else {
        out_result.cols = total_primary;
        out_result.rows = std::max(scl::secondary_size(matrix1), scl::secondary_size(matrix2));
    }
}

// =============================================================================
// SECTION 4: Horizontal Stack (hstack)
// =============================================================================

/// @brief hstack two MappedCustomSparse matrices
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void hstack_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index secondary_dim1 = scl::secondary_size(matrix1);
    const Index secondary_dim2 = scl::secondary_size(matrix2);

    SCL_CHECK_DIM(primary_dim1 == primary_dim2, "hstack_mapped: Primary dimension mismatch");

    const Index primary_dim = primary_dim1;
    const Index nnz1 = matrix1.indptr[primary_dim];
    const Index nnz2 = matrix2.indptr[primary_dim];
    const Index total_nnz = nnz1 + nnz2;
    const Index total_secondary = secondary_dim1 + secondary_dim2;

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "hstack_mapped: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "hstack_mapped: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(primary_dim) + 1,
                  "hstack_mapped: Output indptr too small");

    // Build indptr
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index len1 = matrix1.indptr[p + 1] - matrix1.indptr[p];
        Index len2 = matrix2.indptr[p + 1] - matrix2.indptr[p];
        out_indptr[p + 1] = out_indptr[p] + len1 + len2;
    }

    // Copy data row by row (parallel)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index dst_start = out_indptr[p];

        // Copy from matrix1
        Index src1_start = matrix1.indptr[p];
        Index src1_end = matrix1.indptr[p + 1];
        Index len1 = src1_end - src1_start;

        if (len1 > 0) {
            std::memcpy(out_data.ptr + dst_start, matrix1.data + src1_start, len1 * sizeof(T));
            std::memcpy(out_indices.ptr + dst_start, matrix1.indices + src1_start, len1 * sizeof(Index));
        }

        // Copy from matrix2 with index offset
        Index src2_start = matrix2.indptr[p];
        Index src2_end = matrix2.indptr[p + 1];
        Index len2 = src2_end - src2_start;

        if (len2 > 0) {
            Index dst2 = dst_start + len1;
            std::memcpy(out_data.ptr + dst2, matrix2.data + src2_start, len2 * sizeof(T));
            detail::add_offset_simd(
                matrix2.indices + src2_start,
                out_indices.ptr + dst2,
                static_cast<Size>(len2),
                secondary_dim1
            );
        }
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = primary_dim;
        out_result.cols = total_secondary;
    } else {
        out_result.cols = primary_dim;
        out_result.rows = total_secondary;
    }
}

// =============================================================================
// SECTION 5: Legacy Materialization Interface
// =============================================================================

/// @brief Materialize merged matrix from two MappedCustomSparse (legacy)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> materialize_merged_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix1,
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix2,
    Index secondary_offset
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index nnz1 = matrix1.indptr[primary_dim1];
    const Index nnz2 = matrix2.indptr[primary_dim2];
    const Index total_nnz = nnz1 + nnz2;

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(total_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_primary) + 1);

    auto out_data = data_handle.template as_span<T>();
    auto out_indices = indices_handle.template as_span<Index>();
    auto out_indptr = indptr_handle.template as_span<Index>();

    // Streaming copy matrix1
    std::memcpy(out_data.data(), matrix1.data, nnz1 * sizeof(T));
    std::memcpy(out_indices.data(), matrix1.indices, nnz1 * sizeof(Index));
    std::memcpy(out_indptr.data(), matrix1.indptr, (primary_dim1 + 1) * sizeof(Index));

    // Copy matrix2 with offset
    std::memcpy(out_data.data() + nnz1, matrix2.data, nnz2 * sizeof(T));

    if (secondary_offset == 0) {
        std::memcpy(out_indices.data() + nnz1, matrix2.indices, nnz2 * sizeof(Index));
    } else {
        detail::add_offset_simd(
            matrix2.indices,
            out_indices.data() + nnz1,
            static_cast<Size>(nnz2),
            secondary_offset
        );
    }

    // Update indptr for matrix2 portion (parallel)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim2) + 1, [&](size_t p) {
        out_indptr[primary_dim1 + p] = matrix2.indptr[p] + nnz1;
    });

    // Build result
    scl::io::OwnedSparse<T, IsCSR> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();

    if constexpr (IsCSR) {
        result.matrix.rows = total_primary;
        result.matrix.cols = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    } else {
        result.matrix.cols = total_primary;
        result.matrix.rows = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    }

    result.matrix.data = result.data.ptr;
    result.matrix.indices = result.indices.ptr;
    result.matrix.indptr = result.indptr.ptr;

    return result;
}

/// @brief Materialize merged matrix from two MappedVirtualSparse (legacy)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> materialize_merged_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix1,
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix2,
    Index secondary_offset
) {
    const Index primary_dim1 = scl::primary_size(matrix1);
    const Index primary_dim2 = scl::primary_size(matrix2);
    const Index total_primary = primary_dim1 + primary_dim2;

    const Index total_nnz = compute_total_nnz_mapped(matrix1, matrix2);

    // Allocate output
    auto data_handle = scl::core::mem::alloc_array<T>(static_cast<Size>(total_nnz));
    auto indices_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_nnz));
    auto indptr_handle = scl::core::mem::alloc_array<Index>(static_cast<Size>(total_primary) + 1);

    auto out_data = data_handle.template as_span<T>();
    auto out_indices = indices_handle.template as_span<Index>();
    auto out_indptr = indptr_handle.template as_span<Index>();

    // Build indptr
    out_indptr[0] = 0;
    Index pos = 0;
    for (Index p = 0; p < primary_dim1; ++p) {
        pos += matrix1.lengths[p];
        out_indptr[p + 1] = pos;
    }
    for (Index p = 0; p < primary_dim2; ++p) {
        pos += matrix2.lengths[p];
        out_indptr[primary_dim1 + p + 1] = pos;
    }

    // Copy data from matrix1 (parallel)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim1), [&](size_t p) {
        Index len = matrix1.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[p];
        const T* src_data = static_cast<const T*>(matrix1.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix1.indices_ptrs[p]);

        std::memcpy(out_data.data() + dst_start, src_data, len * sizeof(T));
        std::memcpy(out_indices.data() + dst_start, src_indices, len * sizeof(Index));
    });

    // Copy data from matrix2 with offset (parallel)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim2), [&](size_t p) {
        Index len = matrix2.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[primary_dim1 + p];
        const T* src_data = static_cast<const T*>(matrix2.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix2.indices_ptrs[p]);

        std::memcpy(out_data.data() + dst_start, src_data, len * sizeof(T));

        if (secondary_offset == 0) {
            std::memcpy(out_indices.data() + dst_start, src_indices, len * sizeof(Index));
        } else {
            detail::add_offset_simd(
                src_indices,
                out_indices.data() + dst_start,
                static_cast<Size>(len),
                secondary_offset
            );
        }
    });

    // Build result
    scl::io::OwnedSparse<T, IsCSR> result;
    result.data = data_handle.template release<T>();
    result.indices = indices_handle.template release<Index>();
    result.indptr = indptr_handle.template release<Index>();

    if constexpr (IsCSR) {
        result.matrix.rows = total_primary;
        result.matrix.cols = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    } else {
        result.matrix.cols = total_primary;
        result.matrix.rows = std::max(scl::secondary_size(matrix1),
                                       scl::secondary_size(matrix2) + secondary_offset);
    }

    result.matrix.data = result.data.ptr;
    result.matrix.indices = result.indices.ptr;
    result.matrix.indptr = result.indptr.ptr;

    return result;
}

// =============================================================================
// SECTION 6: Unified Dispatchers
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
Index compute_total_nnz_mapped_dispatch(
    const MatrixT& matrix1,
    const MatrixT& matrix2
) {
    return compute_total_nnz_mapped(matrix1, matrix2);
}

/// @brief Query vstack dimensions
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void query_vstack_dims_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& m1,
    const scl::io::MappedCustomSparse<T, IsCSR>& m2,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    out_primary = scl::primary_size(m1) + scl::primary_size(m2);
    out_secondary = std::max(scl::secondary_size(m1), scl::secondary_size(m2));
    out_nnz = compute_total_nnz_mapped(m1, m2);
}

/// @brief Query hstack dimensions
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void query_hstack_dims_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& m1,
    const scl::io::MappedCustomSparse<T, IsCSR>& m2,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    out_primary = scl::primary_size(m1);
    out_secondary = scl::secondary_size(m1) + scl::secondary_size(m2);
    out_nnz = compute_total_nnz_mapped(m1, m2);
}

} // namespace scl::kernel::merge::mapped
