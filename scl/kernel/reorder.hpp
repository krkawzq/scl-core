#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/reorder_fast_impl.hpp"
#include "scl/kernel/reorder_mapped_impl.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <algorithm>
#include <vector>
#include <cstring>

// =============================================================================
/// @file reorder.hpp
/// @brief Sparse Matrix Alignment and Reordering
///
/// ## Operations
///
/// 1. align_secondary: Remap secondary indices using index map
///    - CSR: Aligns columns (genes) to new reference
///    - CSC: Aligns rows (cells) to new reference
///
/// 2. permute_primary: Reorder primary dimension by permutation
///    - CSR: Reorders rows (cells)
///    - CSC: Reorders columns (genes)
///
/// ## Backend Dispatch
///
/// - CustomSparse / VirtualSparse: In-place modification (fast_impl)
/// - MappedSparse: Returns OwnedSparse (mapped_impl)
///
/// ## Performance Optimizations
///
/// 1. Branchless Index Filtering
///    - Avoid branch mispredictions
///    - 10-20% speedup on random access patterns
///
/// 2. Adaptive Row Processing
///    - Short/Medium/Long row strategies
///    - Prefetch tuning per category
///
/// 3. VQSort Integration
///    - 10-20x faster than std::sort
///
/// 4. Two-Pass Strategy (Mapped)
///    - Exact allocation (no over-provisioning)
///    - Streaming-friendly access
///
/// ## Performance Targets
///
/// - In-Memory: O(nnz) remap + O(nnz log max_row_nnz) sort
/// - Mapped: Sequential streaming with minimal page faults
// =============================================================================

namespace scl::kernel::reorder {

// =============================================================================
// SECTION 1: In-Place Align Secondary (CustomSparse / VirtualSparse)
// =============================================================================

/// @brief Align secondary dimension using index map (CustomSparse)
///
/// In-place modification: remaps indices and compacts within existing storage.
/// Invalid mappings (index_map[i] < 0 or >= new_dim) are filtered out.
///
/// @tparam T Value type
/// @tparam IsCSR True for CSR, false for CSC
/// @param matrix Input sparse matrix (MODIFIED IN-PLACE)
/// @param index_map Mapping: old_idx -> new_idx (use -1 to filter out)
/// @param out_lengths Output valid lengths per row [size >= primary_dim], PRE-ALLOCATED
/// @param new_secondary_dim New secondary dimension size
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void align_secondary(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    fast::align_secondary_custom_fast(matrix, index_map, out_lengths, new_secondary_dim);
}

/// @brief Align secondary dimension using index map (VirtualSparse)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void align_secondary(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    fast::align_secondary_virtual_fast(matrix, index_map, out_lengths, new_secondary_dim);
}

// =============================================================================
// SECTION 2: Mapped Align Secondary (Returns OwnedSparse)
// =============================================================================

/// @brief Align secondary dimension for mapped matrix (MappedCustomSparse)
///
/// Materializes to OwnedSparse with filtering applied.
/// Mapped matrices are read-only, so we must create new storage.
///
/// @tparam T Value type
/// @tparam IsCSR True for CSR, false for CSC
/// @param matrix Input mapped sparse matrix (read-only)
/// @param index_map Mapping: old_idx -> new_idx (use -1 to filter out)
/// @param new_secondary_dim New secondary dimension size
/// @return OwnedSparse with filtered and reordered data
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return mapped::align_secondary_mapped_custom(matrix, index_map, new_secondary_dim);
}

/// @brief Align secondary dimension for mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> align_secondary(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return mapped::align_secondary_mapped_virtual(matrix, index_map, new_secondary_dim);
}

// =============================================================================
// SECTION 3: Unified Dispatchers
// =============================================================================

/// @brief Unified align_secondary (in-place for mutable matrices)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void align_secondary_inplace(
    MatrixT& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    fast::align_secondary_fast_dispatch<MatrixT, IsCSR>(
        matrix, index_map, out_lengths, new_secondary_dim
    );
}

/// @brief Unified align_secondary (returns OwnedSparse for mapped)
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> align_secondary_materialize(
    const MatrixT& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return mapped::align_secondary_mapped_dispatch<MatrixT, IsCSR>(
        matrix, index_map, new_secondary_dim
    );
}

// =============================================================================
// SECTION 4: Permute Primary Dimension
// =============================================================================

/// @brief Permute primary dimension (out-of-place)
///
/// Reorders rows (CSR) or columns (CSC) according to permutation.
/// Output matrix must have pre-allocated storage.
///
/// @tparam T Value type
/// @tparam IsCSR True for CSR, false for CSC
/// @param input Input sparse matrix
/// @param permutation Permutation: new_pos -> old_pos mapping
/// @param output Output sparse matrix with pre-allocated storage
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void permute_primary(
    const CustomSparse<T, IsCSR>& input,
    Array<const Index> permutation,
    CustomSparse<T, IsCSR>& output
) {
    fast::permute_primary_fast(input, permutation, output);
}

/// @brief Permute primary and return new OwnedSparse
///
/// Convenience function that allocates output automatically.
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> permute_primary_owned(
    const CustomSparse<T, IsCSR>& input,
    Array<const Index> permutation
) {
    const Index primary_dim = scl::primary_size(input);
    const Index total_nnz = input.indptr[primary_dim];

    SCL_CHECK_DIM(permutation.len == static_cast<Size>(primary_dim),
                  "Permutation size mismatch");

    // Allocate output vectors
    std::vector<T> out_data(static_cast<size_t>(total_nnz));
    std::vector<Index> out_indices(static_cast<size_t>(total_nnz));
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);

    // Build new indptr
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index old_p = permutation[p];
        Index len = input.indptr[old_p + 1] - input.indptr[old_p];
        out_indptr[p + 1] = out_indptr[p] + len;
    }

    // Copy data in parallel
    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        Index old_p = permutation[p];
        Index old_start = input.indptr[old_p];
        Index old_end = input.indptr[old_p + 1];
        Size len = static_cast<Size>(old_end - old_start);

        if (len == 0) return;

        Index new_start = out_indptr[p];

        std::memcpy(out_indices.data() + new_start, input.indices + old_start, len * sizeof(Index));
        std::memcpy(out_data.data() + new_start, input.data + old_start, len * sizeof(T));
    });

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(out_indptr),
        input.rows, input.cols
    );
}

// =============================================================================
// SECTION 5: Query Utilities
// =============================================================================

/// @brief Compute total NNZ after filtering (CustomSparse)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
Size compute_filtered_nnz(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return fast::compute_filtered_nnz(matrix, index_map, new_secondary_dim);
}

/// @brief Compute filtered NNZ for mapped matrix
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
Size compute_filtered_nnz(
    const MatrixT& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    return mapped::compute_filtered_nnz_mapped<MatrixT, IsCSR>(
        matrix, index_map, new_secondary_dim
    );
}

// =============================================================================
// SECTION 6: Index Map Builders
// =============================================================================

/// @brief Build identity index map: index_map[i] = i
///
/// Useful as starting point for modifications.
inline void build_identity_map(Array<Index> index_map, Index dim) {
    SCL_CHECK_DIM(index_map.len >= static_cast<Size>(dim),
                  "Index map too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(dim), [&](size_t i) {
        index_map[i] = static_cast<Index>(i);
    });
}

/// @brief Build subset index map
///
/// Creates mapping that keeps only indices in the subset.
/// index_map[old] = new_position if old is in subset, else -1 (filtered out).
///
/// @param old_dim Original dimension
/// @param subset Indices to keep (sorted recommended for cache efficiency)
/// @param index_map Output map [size >= old_dim], PRE-ALLOCATED
inline void build_subset_map(
    Index old_dim,
    Array<const Index> subset,
    Array<Index> index_map
) {
    SCL_CHECK_DIM(index_map.len >= static_cast<Size>(old_dim),
                  "Index map too small");

    // Initialize all to -1 (filtered out)
    scl::threading::parallel_for(Size(0), static_cast<Size>(old_dim), [&](size_t i) {
        index_map[i] = -1;
    });

    // Set valid mappings (sequential for correctness with duplicates)
    for (Size i = 0; i < subset.len; ++i) {
        Index old_idx = subset[i];
        if (old_idx >= 0 && old_idx < old_dim) {
            index_map[old_idx] = static_cast<Index>(i);
        }
    }
}

/// @brief Build permutation index map
///
/// Creates mapping from a permutation array.
/// permutation[new_pos] = old_pos implies index_map[old_pos] = new_pos.
///
/// @param permutation Permutation: new_pos -> old_pos
/// @param index_map Output map [size >= permutation.len], PRE-ALLOCATED
inline void build_permutation_map(
    Array<const Index> permutation,
    Array<Index> index_map
) {
    Size dim = permutation.len;

    SCL_CHECK_DIM(index_map.len >= dim, "Index map too small");

    scl::threading::parallel_for(Size(0), dim, [&](size_t new_pos) {
        Index old_pos = permutation[new_pos];
        if (old_pos >= 0 && old_pos < static_cast<Index>(dim)) {
            index_map[old_pos] = static_cast<Index>(new_pos);
        }
    });
}

/// @brief Build inverse permutation: inv[perm[i]] = i
///
/// @param permutation Input permutation
/// @param inverse Output inverse [size >= permutation.len], PRE-ALLOCATED
inline void build_inverse_permutation(
    Array<const Index> permutation,
    Array<Index> inverse
) {
    fast::build_inverse_permutation(permutation, inverse);
}

/// @brief Build mask-based index map
///
/// Creates mapping that keeps indices where mask[i] != 0.
/// Indices are compacted to [0, count-1].
///
/// @param mask Binary mask [size = old_dim]
/// @param index_map Output map [size >= old_dim], PRE-ALLOCATED
/// @return Number of kept indices
inline Size build_mask_map(
    Array<const uint8_t> mask,
    Array<Index> index_map
) {
    Size old_dim = mask.len;

    SCL_CHECK_DIM(index_map.len >= old_dim, "Index map too small");

    // Sequential prefix sum (dependency chain)
    Index current_idx = 0;
    for (Size i = 0; i < old_dim; ++i) {
        if (mask[i] != 0) {
            index_map[i] = current_idx++;
        } else {
            index_map[i] = -1;
        }
    }

    return static_cast<Size>(current_idx);
}

/// @brief Build filtered subset map with explicit new indices
///
/// Like build_subset_map but allows specifying target indices.
///
/// @param old_dim Original dimension
/// @param old_indices Source indices [size = n]
/// @param new_indices Target indices [size = n]
/// @param index_map Output map [size >= old_dim], PRE-ALLOCATED
inline void build_explicit_map(
    Index old_dim,
    Array<const Index> old_indices,
    Array<const Index> new_indices,
    Array<Index> index_map
) {
    SCL_CHECK_DIM(old_indices.len == new_indices.len,
                  "Old and new indices must have same length");
    SCL_CHECK_DIM(index_map.len >= static_cast<Size>(old_dim),
                  "Index map too small");

    // Initialize all to -1
    scl::threading::parallel_for(Size(0), static_cast<Size>(old_dim), [&](size_t i) {
        index_map[i] = -1;
    });

    // Set explicit mappings
    for (Size i = 0; i < old_indices.len; ++i) {
        Index old_idx = old_indices[i];
        if (old_idx >= 0 && old_idx < old_dim) {
            index_map[old_idx] = new_indices[i];
        }
    }
}

} // namespace scl::kernel::reorder
