#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>
#include <algorithm>

// =============================================================================
/// @file merge_fast_impl.hpp
/// @brief Fast Path for Matrix Merging (CustomSparse/VirtualSparse)
///
/// Optimizations:
/// - SIMD index offset operations
/// - Parallel data copying per matrix
/// - Bulk memcpy for contiguous regions
/// - Two-pass strategy (count then copy) for VirtualSparse
///
/// Note: Merge is O(rows) for virtual views, O(NNZ) for materialization.
/// The virtual path is already near-optimal; materialization benefits from
/// parallel I/O and SIMD offset computation.
// =============================================================================

namespace scl::kernel::merge::fast {

// =============================================================================
// SECTION 1: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD add offset to index array (2-way unrolled)
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

    // Single vector
    for (; i + lanes <= count; i += lanes) {
        auto v = s::Load(d, src + i);
        s::Store(s::Add(v, v_offset), d, dst + i);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] + offset;
    }
}

/// @brief Parallel memcpy with prefetch
template <typename T>
inline void parallel_memcpy(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size count,
    Size chunk_size = 65536
) {
    if (count < chunk_size) {
        std::memcpy(dst, src, count * sizeof(T));
        return;
    }

    const Size n_chunks = (count + chunk_size - 1) / chunk_size;

    scl::threading::parallel_for(0, n_chunks, [&](size_t c) {
        Size start = c * chunk_size;
        Size end = std::min(start + chunk_size, count);
        Size len = end - start;

        // Prefetch next chunk
        if (c + 1 < n_chunks) {
            SCL_PREFETCH_READ(src + end, 0);
        }

        std::memcpy(dst + start, src + start, len * sizeof(T));
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: CustomSparse Two-Matrix Merge
// =============================================================================

/// @brief Merge two CustomSparse matrices vertically (vstack)
///
/// @param matrix1 First matrix
/// @param matrix2 Second matrix
/// @param out_data Output data [size = nnz1 + nnz2], PRE-ALLOCATED
/// @param out_indices Output indices [size = nnz1 + nnz2], PRE-ALLOCATED
/// @param out_indptr Output indptr [size = rows1 + rows2 + 1], PRE-ALLOCATED
/// @param out_result Output CustomSparse view
template <typename T, bool IsCSR>
void vstack_custom_fast(
    const CustomSparse<T, IsCSR>& matrix1,
    const CustomSparse<T, IsCSR>& matrix2,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary1 = IsCSR ? matrix1.rows : matrix1.cols;
    const Index primary2 = IsCSR ? matrix2.rows : matrix2.cols;
    const Index secondary1 = IsCSR ? matrix1.cols : matrix1.rows;
    const Index secondary2 = IsCSR ? matrix2.cols : matrix2.rows;

    const Index nnz1 = matrix1.indptr[primary1];
    const Index nnz2 = matrix2.indptr[primary2];
    const Index total_primary = primary1 + primary2;
    const Index total_nnz = nnz1 + nnz2;

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "vstack_custom_fast: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "vstack_custom_fast: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(total_primary) + 1,
                  "vstack_custom_fast: Output indptr too small");

    // Parallel copy of matrix1
    detail::parallel_memcpy(matrix1.data, out_data.ptr, static_cast<Size>(nnz1));
    detail::parallel_memcpy(matrix1.indices, out_indices.ptr, static_cast<Size>(nnz1));
    std::memcpy(out_indptr.ptr, matrix1.indptr, (primary1 + 1) * sizeof(Index));

    // Parallel copy of matrix2
    detail::parallel_memcpy(matrix2.data, out_data.ptr + nnz1, static_cast<Size>(nnz2));
    detail::parallel_memcpy(matrix2.indices, out_indices.ptr + nnz1, static_cast<Size>(nnz2));

    // Build indptr for matrix2 portion (offset by nnz1)
    scl::threading::parallel_for(0, static_cast<size_t>(primary2) + 1, [&](size_t p) {
        out_indptr[primary1 + p] = matrix2.indptr[p] + nnz1;
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = std::max(secondary1, secondary2);
    } else {
        out_result.cols = total_primary;
        out_result.rows = std::max(secondary1, secondary2);
    }
}

/// @brief Merge two CustomSparse matrices horizontally (hstack)
///
/// Concatenates columns (CSR) or rows (CSC).
///
/// @param matrix1 First matrix
/// @param matrix2 Second matrix
/// @param out_data Output data [size = nnz1 + nnz2], PRE-ALLOCATED
/// @param out_indices Output indices [size = nnz1 + nnz2], PRE-ALLOCATED
/// @param out_indptr Output indptr [size = primary_dim + 1], PRE-ALLOCATED
/// @param out_result Output CustomSparse view
template <typename T, bool IsCSR>
void hstack_custom_fast(
    const CustomSparse<T, IsCSR>& matrix1,
    const CustomSparse<T, IsCSR>& matrix2,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary1 = IsCSR ? matrix1.rows : matrix1.cols;
    const Index primary2 = IsCSR ? matrix2.rows : matrix2.cols;
    const Index secondary1 = IsCSR ? matrix1.cols : matrix1.rows;
    const Index secondary2 = IsCSR ? matrix2.cols : matrix2.rows;

    SCL_CHECK_DIM(primary1 == primary2, "hstack_custom_fast: Primary dimension mismatch");

    const Index primary_dim = primary1;
    const Index nnz1 = matrix1.indptr[primary_dim];
    const Index nnz2 = matrix2.indptr[primary_dim];
    const Index total_nnz = nnz1 + nnz2;
    const Index total_secondary = secondary1 + secondary2;

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "hstack_custom_fast: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "hstack_custom_fast: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(primary_dim) + 1,
                  "hstack_custom_fast: Output indptr too small");

    // Build indptr: sum of row lengths from both matrices
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index len1 = matrix1.indptr[p + 1] - matrix1.indptr[p];
        Index len2 = matrix2.indptr[p + 1] - matrix2.indptr[p];
        out_indptr[p + 1] = out_indptr[p] + len1 + len2;
    }

    // Copy data and indices row by row (parallel)
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
                secondary1  // Column offset
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
// SECTION 3: VirtualSparse Materialization
// =============================================================================

/// @brief Materialize VirtualSparse to contiguous CustomSparse
///
/// @param matrix Input VirtualSparse
/// @param out_data Output data [size = nnz], PRE-ALLOCATED
/// @param out_indices Output indices [size = nnz], PRE-ALLOCATED
/// @param out_indptr Output indptr [size = primary + 1], PRE-ALLOCATED
/// @param out_result Output CustomSparse view
template <typename T, bool IsCSR>
void materialize_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    const Index primary_dim = IsCSR ? matrix.rows : matrix.cols;

    // Build indptr via prefix sum
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        out_indptr[p + 1] = out_indptr[p] + matrix.lengths[p];
    }

    const Index total_nnz = out_indptr[primary_dim];

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "materialize_virtual_fast: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "materialize_virtual_fast: Output indices too small");

    // Parallel copy
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        if (len == 0) return;

        Index dst_start = out_indptr[p];
        const T* src_data = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* src_indices = static_cast<const Index*>(matrix.indices_ptrs[p]);

        std::memcpy(out_data.ptr + dst_start, src_data, len * sizeof(T));
        std::memcpy(out_indices.ptr + dst_start, src_indices, len * sizeof(Index));
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = matrix.rows;
        out_result.cols = matrix.cols;
    } else {
        out_result.cols = matrix.cols;
        out_result.rows = matrix.rows;
    }
}

// =============================================================================
// SECTION 4: Dimension Queries (No Allocation)
// =============================================================================

/// @brief Compute vstack output dimensions
template <typename T, bool IsCSR>
inline void query_vstack_dims_custom(
    const CustomSparse<T, IsCSR>& m1,
    const CustomSparse<T, IsCSR>& m2,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    const Index p1 = IsCSR ? m1.rows : m1.cols;
    const Index p2 = IsCSR ? m2.rows : m2.cols;
    const Index s1 = IsCSR ? m1.cols : m1.rows;
    const Index s2 = IsCSR ? m2.cols : m2.rows;

    out_primary = p1 + p2;
    out_secondary = std::max(s1, s2);
    out_nnz = m1.indptr[p1] + m2.indptr[p2];
}

/// @brief Compute hstack output dimensions
template <typename T, bool IsCSR>
inline void query_hstack_dims_custom(
    const CustomSparse<T, IsCSR>& m1,
    const CustomSparse<T, IsCSR>& m2,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    const Index p1 = IsCSR ? m1.rows : m1.cols;
    const Index s1 = IsCSR ? m1.cols : m1.rows;
    const Index s2 = IsCSR ? m2.cols : m2.rows;

    out_primary = p1;
    out_secondary = s1 + s2;
    out_nnz = m1.indptr[p1] + m2.indptr[p1];
}

/// @brief Compute VirtualSparse total NNZ
template <typename T, bool IsCSR>
inline Index compute_virtual_nnz(const VirtualSparse<T, IsCSR>& matrix) {
    const Index primary_dim = IsCSR ? matrix.rows : matrix.cols;

    Index total = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        total += matrix.lengths[p];
    }
    return total;
}

} // namespace scl::kernel::merge::fast
