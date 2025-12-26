#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <vector>

// =============================================================================
/// @file mwu_mapped_impl.hpp
/// @brief Mann-Whitney U Test for Mapped Sparse Matrices
///
/// MWU is primarily sort-bound (O(n log n)), not memory-bound.
/// For Mapped matrices, we stream the data to extract values for ranking.
///
/// Operations:
/// - extract_values_for_ranking_mapped: Stream data for ranking
// =============================================================================

namespace scl::kernel::mwu::mapped {

namespace detail {
constexpr Size CHUNK_SIZE = 256;
}

// =============================================================================
// Value Extraction - Streaming Implementation
// =============================================================================

/// @brief Extract values for ranking from mapped custom sparse (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void extract_values_for_ranking_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Index primary_idx,
    std::vector<std::pair<Real, Index>>& out_values
) {
    SCL_CHECK_ARG(primary_idx >= 0 && primary_idx < scl::primary_size(matrix),
                  "MWU: Primary index out of bounds");

    Index start = matrix.indptr[primary_idx];
    Index end = matrix.indptr[primary_idx + 1];
    Index len = end - start;

    out_values.clear();
    out_values.reserve(static_cast<size_t>(len));

    const T* SCL_RESTRICT values = matrix.data + start;
    const Index* SCL_RESTRICT indices = matrix.indices + start;

    for (Index k = 0; k < len; ++k) {
        out_values.emplace_back(static_cast<Real>(values[k]), indices[k]);
    }
}

/// @brief Extract values for ranking from mapped virtual sparse (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void extract_values_for_ranking_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Index primary_idx,
    std::vector<std::pair<Real, Index>>& out_values
) {
    SCL_CHECK_ARG(primary_idx >= 0 && primary_idx < scl::primary_size(matrix),
                  "MWU: Primary index out of bounds");

    Index len = matrix.lengths[primary_idx];

    out_values.clear();
    out_values.reserve(static_cast<size_t>(len));

    const T* SCL_RESTRICT values = static_cast<const T*>(matrix.data_ptrs[primary_idx]);
    const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[primary_idx]);

    for (Index k = 0; k < len; ++k) {
        out_values.emplace_back(static_cast<Real>(values[k]), indices[k]);
    }
}

// =============================================================================
// Batch Extraction for Multiple Features
// =============================================================================

/// @brief Batch extract values for multiple features (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void batch_extract_for_ranking_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Index> feature_indices,
    std::vector<std::vector<std::pair<Real, Index>>>& out_values_batch
) {
    const Size n_features = feature_indices.len;
    out_values_batch.resize(n_features);

    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        Index primary_idx = feature_indices[i];
        extract_values_for_ranking_mapped(matrix, primary_idx, out_values_batch[i]);
    });
}

/// @brief Batch extract values for multiple features (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void batch_extract_for_ranking_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Index> feature_indices,
    std::vector<std::vector<std::pair<Real, Index>>>& out_values_batch
) {
    const Size n_features = feature_indices.len;
    out_values_batch.resize(n_features);

    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        Index primary_idx = feature_indices[i];
        extract_values_for_ranking_mapped(matrix, primary_idx, out_values_batch[i]);
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void extract_values_for_ranking_mapped_dispatch(
    const MatrixT& matrix,
    Index primary_idx,
    std::vector<std::pair<Real, Index>>& out_values
) {
    extract_values_for_ranking_mapped(matrix, primary_idx, out_values);
}

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void batch_extract_for_ranking_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Index> feature_indices,
    std::vector<std::vector<std::pair<Real, Index>>>& out_values_batch
) {
    batch_extract_for_ranking_mapped(matrix, feature_indices, out_values_batch);
}

} // namespace scl::kernel::mwu::mapped

