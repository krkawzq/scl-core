#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <cstring>
#include <algorithm>

// =============================================================================
/// @file merge.hpp
/// @brief High-Performance Matrix Merging Operations
///
/// Implements matrix concatenation with multiple strategies:
/// 1. Zero-Copy Virtual Views: O(rows) pointer setup, no data movement
/// 2. Materialized Merge: O(NNZ) with SIMD-optimized copies
///
/// Key Insight:
/// - Virtual merge: O(rows) index array creation - 100-1000x faster
/// - Physical merge: O(NNZ) but provides contiguous memory for computation
///
/// Operations:
/// - vstack: Vertical concatenation (along primary axis)
/// - hstack: Horizontal concatenation (along secondary axis)
/// - merge_n: Batch merge multiple matrices
///
/// Use Cases:
/// - Batch integration
/// - Cross-validation splits
/// - Incremental loading
/// - Out-of-core processing
// =============================================================================

namespace scl::kernel::merge {

// =============================================================================
// SECTION 1: Common Utilities
// =============================================================================

namespace detail {

/// @brief SIMD add offset to index array
SCL_FORCE_INLINE void add_offset_indices(
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

/// @brief Compute total dimensions for vstack
template <typename T, bool IsCSR>
inline void compute_vstack_dims(
    Array<const VirtualSparse<T, IsCSR>*> inputs,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    out_primary = 0;
    out_secondary = 0;
    out_nnz = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;

        out_primary += mat_primary;
        out_secondary = std::max(out_secondary, mat_secondary);

        // Sum NNZ from lengths
        for (Index j = 0; j < mat_primary; ++j) {
            out_nnz += mat->lengths[j];
        }
    }
}

/// @brief Compute total dimensions for CustomSparse vstack
template <typename T, bool IsCSR>
inline void compute_vstack_dims_custom(
    Array<const CustomSparse<T, IsCSR>*> inputs,
    Index& out_primary,
    Index& out_secondary,
    Index& out_nnz
) {
    out_primary = 0;
    out_secondary = 0;
    out_nnz = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;

        out_primary += mat_primary;
        out_secondary = std::max(out_secondary, mat_secondary);
        out_nnz += mat->indptr[mat_primary];
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: Zero-Copy Virtual Stack (VirtualSparse)
// =============================================================================

/// @brief Vertically stack VirtualSparse matrices (zero-copy)
///
/// Constructs a new VirtualSparse that views multiple source matrices as one.
/// No data copying - only builds pointer indirection arrays.
///
/// Requirements:
/// - All inputs must have compatible secondary dimension
/// - Output arrays must be pre-allocated
///
/// @param inputs Array of VirtualSparse pointers
/// @param out_data_ptrs Output data pointers array [size = total_rows]
/// @param out_indices_ptrs Output indices pointers array [size = total_rows]
/// @param out_lengths Output lengths array [size = total_rows]
/// @param out_result Output VirtualSparse view
template <typename T, bool IsCSR>
void vstack_virtual(
    Array<const VirtualSparse<T, IsCSR>*> inputs,
    Array<void*> out_data_ptrs,
    Array<void*> out_indices_ptrs,
    Array<Index> out_lengths,
    VirtualSparse<T, IsCSR>& out_result
) {
    if (inputs.size() == 0) {
        out_result = VirtualSparse<T, IsCSR>{};
        return;
    }

    // Compute dimensions
    Index total_primary = 0;
    Index secondary_dim = IsCSR ? inputs[0]->cols : inputs[0]->rows;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;
        SCL_CHECK_DIM(mat_secondary == secondary_dim, "vstack: Secondary dimension mismatch");

        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        total_primary += mat_primary;
    }

    SCL_CHECK_DIM(out_data_ptrs.size() >= static_cast<Size>(total_primary),
                  "vstack: Output data_ptrs too small");
    SCL_CHECK_DIM(out_indices_ptrs.size() >= static_cast<Size>(total_primary),
                  "vstack: Output indices_ptrs too small");
    SCL_CHECK_DIM(out_lengths.size() >= static_cast<Size>(total_primary),
                  "vstack: Output lengths too small");

    // Build merged indirection (parallel per-matrix copy)
    Index offset = 0;
    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;

        // Copy pointers and lengths
        scl::threading::parallel_for(0, static_cast<size_t>(mat_primary), [&](size_t j) {
            out_data_ptrs[offset + j] = const_cast<void*>(mat->data_ptrs[j]);
            out_indices_ptrs[offset + j] = const_cast<void*>(mat->indices_ptrs[j]);
            out_lengths[offset + j] = mat->lengths[j];
        });

        offset += mat_primary;
    }

    // Setup result
    out_result.data_ptrs = out_data_ptrs.ptr;
    out_result.indices_ptrs = out_indices_ptrs.ptr;
    out_result.lengths = out_lengths.ptr;

    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = secondary_dim;
    } else {
        out_result.rows = secondary_dim;
        out_result.cols = total_primary;
    }
}

// =============================================================================
// SECTION 3: Materialized Stack (CustomSparse)
// =============================================================================

/// @brief Vertically stack CustomSparse matrices with materialization
///
/// Creates a contiguous CustomSparse by copying all data.
/// Use when contiguous memory is required for downstream operations.
///
/// @param inputs Array of CustomSparse pointers
/// @param out_data Output data array [size = total_nnz], PRE-ALLOCATED
/// @param out_indices Output indices array [size = total_nnz], PRE-ALLOCATED
/// @param out_indptr Output indptr array [size = total_rows + 1], PRE-ALLOCATED
/// @param out_result Output CustomSparse view
template <typename T, bool IsCSR>
void vstack_materialize(
    Array<const CustomSparse<T, IsCSR>*> inputs,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    if (inputs.size() == 0) {
        out_indptr[0] = 0;
        out_result = CustomSparse<T, IsCSR>{};
        return;
    }

    // Compute dimensions
    Index total_primary, secondary_dim, total_nnz;
    detail::compute_vstack_dims_custom(inputs, total_primary, secondary_dim, total_nnz);

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "vstack_materialize: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "vstack_materialize: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(total_primary) + 1,
                  "vstack_materialize: Output indptr too small");

    // Build indptr and track copy positions
    std::vector<Index> matrix_offsets(inputs.size() + 1);
    std::vector<Index> nnz_offsets(inputs.size() + 1);

    matrix_offsets[0] = 0;
    nnz_offsets[0] = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_nnz = mat->indptr[mat_primary];

        matrix_offsets[i + 1] = matrix_offsets[i] + mat_primary;
        nnz_offsets[i + 1] = nnz_offsets[i] + mat_nnz;
    }

    // Build indptr (parallel)
    out_indptr[0] = 0;
    scl::threading::parallel_for(0, inputs.size(), [&](size_t i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index row_offset = matrix_offsets[i];
        Index nnz_offset = nnz_offsets[i];

        for (Index j = 0; j < mat_primary; ++j) {
            out_indptr[row_offset + j + 1] = mat->indptr[j + 1] + nnz_offset;
        }
    });

    // Copy data and indices (parallel per matrix)
    scl::threading::parallel_for(0, inputs.size(), [&](size_t i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_nnz = mat->indptr[mat_primary];
        Index nnz_offset = nnz_offsets[i];

        // Bulk copy data
        std::memcpy(out_data.ptr + nnz_offset, mat->data, mat_nnz * sizeof(T));
        std::memcpy(out_indices.ptr + nnz_offset, mat->indices, mat_nnz * sizeof(Index));
    });

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = secondary_dim;
    } else {
        out_result.rows = secondary_dim;
        out_result.cols = total_primary;
    }
}

// =============================================================================
// SECTION 4: Horizontal Stack (Column Concatenation)
// =============================================================================

/// @brief Horizontally stack CustomSparse matrices
///
/// Concatenates matrices along the secondary axis.
/// For CSR: appends columns; For CSC: appends rows.
///
/// @param inputs Array of CustomSparse pointers
/// @param out_data Output data array [size = total_nnz], PRE-ALLOCATED
/// @param out_indices Output indices array [size = total_nnz], PRE-ALLOCATED
/// @param out_indptr Output indptr array [size = primary_dim + 1], PRE-ALLOCATED
/// @param out_result Output CustomSparse view
template <typename T, bool IsCSR>
void hstack_materialize(
    Array<const CustomSparse<T, IsCSR>*> inputs,
    Array<T> out_data,
    Array<Index> out_indices,
    Array<Index> out_indptr,
    CustomSparse<T, IsCSR>& out_result
) {
    if (inputs.size() == 0) {
        out_indptr[0] = 0;
        out_result = CustomSparse<T, IsCSR>{};
        return;
    }

    // Compute dimensions
    const Index primary_dim = IsCSR ? inputs[0]->rows : inputs[0]->cols;
    Index secondary_dim = 0;
    Index total_nnz = 0;

    // Compute secondary offsets
    std::vector<Index> secondary_offsets(inputs.size() + 1);
    secondary_offsets[0] = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;
        Index mat_nnz = mat->indptr[mat_primary];

        SCL_CHECK_DIM(mat_primary == primary_dim, "hstack: Primary dimension mismatch");

        secondary_offsets[i + 1] = secondary_offsets[i] + mat_secondary;
        total_nnz += mat_nnz;
    }
    secondary_dim = secondary_offsets[inputs.size()];

    SCL_CHECK_DIM(out_data.size() >= static_cast<Size>(total_nnz),
                  "hstack_materialize: Output data too small");
    SCL_CHECK_DIM(out_indices.size() >= static_cast<Size>(total_nnz),
                  "hstack_materialize: Output indices too small");
    SCL_CHECK_DIM(out_indptr.size() >= static_cast<Size>(primary_dim) + 1,
                  "hstack_materialize: Output indptr too small");

    // Phase 1: Compute row lengths for merged matrix
    std::vector<Index> row_lengths(static_cast<Size>(primary_dim), 0);

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        for (Index p = 0; p < primary_dim; ++p) {
            row_lengths[p] += mat->indptr[p + 1] - mat->indptr[p];
        }
    }

    // Phase 2: Build indptr via prefix sum
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        out_indptr[p + 1] = out_indptr[p] + row_lengths[p];
    }

    // Phase 3: Copy data row by row (parallel over rows)
    std::vector<Index> row_positions(static_cast<Size>(primary_dim));
    for (Index p = 0; p < primary_dim; ++p) {
        row_positions[p] = out_indptr[p];
    }

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        const Index col_offset = secondary_offsets[i];

        scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
            Index src_start = mat->indptr[p];
            Index src_end = mat->indptr[p + 1];
            Index len = src_end - src_start;
            if (len == 0) return;

            // Thread-safe position tracking via atomic would be needed for true parallel
            // For simplicity, we do sequential per-matrix here
            Index dst_pos = row_positions[p];
            row_positions[p] += len;

            // Copy data
            std::memcpy(out_data.ptr + dst_pos, mat->data + src_start, len * sizeof(T));

            // Copy indices with offset
            detail::add_offset_indices(
                mat->indices + src_start,
                out_indices.ptr + dst_pos,
                static_cast<Size>(len),
                col_offset
            );
        });
    }

    // Setup result
    out_result.data = out_data.ptr;
    out_result.indices = out_indices.ptr;
    out_result.indptr = out_indptr.ptr;

    if constexpr (IsCSR) {
        out_result.rows = primary_dim;
        out_result.cols = secondary_dim;
    } else {
        out_result.rows = secondary_dim;
        out_result.cols = primary_dim;
    }
}

// =============================================================================
// SECTION 5: Legacy Compatibility
// =============================================================================

/// @brief Legacy vstack interface (simplified)
template <typename T, bool IsCSR>
void vstack(
    Array<const VirtualSparse<T, IsCSR>*> inputs,
    Array<Index> out_row_map,
    VirtualSparse<T, IsCSR>& out_result
) {
    if (inputs.size() == 0) {
        out_result = VirtualSparse<T, IsCSR>();
        return;
    }

    // Validate dimensions
    const Index secondary_dim = IsCSR ? inputs[0]->cols : inputs[0]->rows;
    Index total_primary = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;
        SCL_CHECK_DIM(mat_secondary == secondary_dim, "Merge: Secondary dimension mismatch");

        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        total_primary += mat_primary;
    }

    SCL_CHECK_DIM(out_row_map.size() >= static_cast<Size>(total_primary),
                  "Merge: Output row map too small");

    // Build row map (identity within each block)
    Index offset = 0;
    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;

        for (Index j = 0; j < mat_primary; ++j) {
            out_row_map[offset + j] = j;
        }
        offset += mat_primary;
    }

    // Set dimensions only (data_ptrs etc need separate allocation)
    if constexpr (IsCSR) {
        out_result.rows = total_primary;
        out_result.cols = secondary_dim;
    } else {
        out_result.rows = secondary_dim;
        out_result.cols = total_primary;
    }
}

// =============================================================================
// SECTION 6: Dimension Query Utilities
// =============================================================================

/// @brief Compute merged dimensions without materializing
template <typename T, bool IsCSR>
inline void query_vstack_dims(
    Array<const CustomSparse<T, IsCSR>*> inputs,
    Index& out_rows,
    Index& out_cols,
    Index& out_nnz
) {
    Index primary = 0, secondary = 0;
    out_nnz = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;

        primary += mat_primary;
        secondary = std::max(secondary, mat_secondary);
        out_nnz += mat->indptr[mat_primary];
    }

    if constexpr (IsCSR) {
        out_rows = primary;
        out_cols = secondary;
    } else {
        out_rows = secondary;
        out_cols = primary;
    }
}

/// @brief Compute hstack dimensions
template <typename T, bool IsCSR>
inline void query_hstack_dims(
    Array<const CustomSparse<T, IsCSR>*> inputs,
    Index& out_rows,
    Index& out_cols,
    Index& out_nnz
) {
    if (inputs.size() == 0) {
        out_rows = out_cols = out_nnz = 0;
        return;
    }

    const Index primary_dim = IsCSR ? inputs[0]->rows : inputs[0]->cols;
    Index secondary = 0;
    out_nnz = 0;

    for (Size i = 0; i < inputs.size(); ++i) {
        const auto* mat = inputs[i];
        Index mat_primary = IsCSR ? mat->rows : mat->cols;
        Index mat_secondary = IsCSR ? mat->cols : mat->rows;

        SCL_CHECK_DIM(mat_primary == primary_dim, "query_hstack_dims: Primary dimension mismatch");

        secondary += mat_secondary;
        out_nnz += mat->indptr[mat_primary];
    }

    if constexpr (IsCSR) {
        out_rows = primary_dim;
        out_cols = secondary;
    } else {
        out_rows = secondary;
        out_cols = primary_dim;
    }
}

} // namespace scl::kernel::merge
