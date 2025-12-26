#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/gram.hpp"

// Backend implementations
#include "scl/kernel/bbknn_fast_impl.hpp"
#include "scl/kernel/bbknn_mapped_impl.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file bbknn.hpp
/// @brief Batch Balanced K-Nearest Neighbors
///
/// ## Algorithm
///
/// For each sample, finds k nearest neighbors within EACH batch separately.
/// Output: (N * n_batches * k) neighbor indices and distances.
///
/// ## Optimizations
///
/// 1. Batch-Grouped Processing
///    - Samples grouped by batch ID upfront
///    - Only compare within relevant batches
///    - Reduces comparisons by factor of n_batches
///
/// 2. Early Rejection
///    - Triangle inequality: min_dist = ||a|| + ||b|| - 2*||a||*||b||
///    - Skip candidates that can't beat current k-th distance
///
/// 3. Optimized Heap
///    - Fixed-size k-max-heap with manual sift
///    - No std::priority_queue overhead
///
/// 4. SIMD Norm/Dot
///    - 4-way unrolled FMA accumulation
///    - Cache query row for inner loop
///
/// ## Backend Dispatch
///
/// - MappedSparseLike -> bbknn_mapped_impl.hpp
/// - CustomSparseLike -> bbknn_fast_impl.hpp
/// - VirtualSparseLike -> bbknn_fast_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Complexity
///
/// Time: O(N^2 * avg_nnz) for sparse-sparse dot products
/// Space: O(N + n_batches * k) per thread
///
/// ## Performance
///
/// ~5K cells, 3 batches, k=5, 16 cores: target < 5 sec
// =============================================================================

namespace scl::kernel::bbknn {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Max-heap for k nearest neighbors (generic version)
template <typename T>
struct GenericHeap {
    struct Entry {
        T dist_sq;
        Index cell_idx;

        bool operator<(const Entry& other) const {
            return dist_sq < other.dist_sq;
        }
    };

    Entry* data;
    Size capacity;
    Size count;

    void init(Entry* storage, Size k) {
        data = storage;
        capacity = k;
        count = 0;
    }

    void clear() { count = 0; }

    T max_dist_sq() const {
        return (count > 0) ? data[0].dist_sq : std::numeric_limits<T>::max();
    }

    void try_insert(T dist_sq, Index idx) {
        if (count < capacity) {
            data[count] = {dist_sq, idx};
            count++;
            std::push_heap(data, data + count);
        } else if (dist_sq < data[0].dist_sq) {
            std::pop_heap(data, data + count);
            data[count - 1] = {dist_sq, idx};
            std::push_heap(data, data + count);
        }
    }

    void extract_sorted(Index* out_indices, T* out_distances) {
        std::sort_heap(data, data + count);

        for (Size i = 0; i < count; ++i) {
            out_indices[i] = data[i].cell_idx;
            out_distances[i] = std::sqrt(data[i].dist_sq);
        }

        for (Size i = count; i < capacity; ++i) {
            out_indices[i] = -1;
            out_distances[i] = std::numeric_limits<T>::infinity();
        }
    }
};

/// @brief Build batch groups
inline std::vector<std::vector<Index>> build_batch_groups(
    Array<const int32_t> batch_labels,
    Size n_batches
) {
    std::vector<std::vector<Index>> groups(n_batches);

    for (Size i = 0; i < batch_labels.len; ++i) {
        int32_t b = batch_labels[i];
        if (b >= 0 && static_cast<Size>(b) < n_batches) {
            groups[b].push_back(static_cast<Index>(i));
        }
    }

    return groups;
}

/// @brief Sparse-sparse dot product
template <typename T>
T sparse_dot_generic(
    const T* vals_a, const Index* inds_a, Size len_a,
    const T* vals_b, const Index* inds_b, Size len_b
) {
    T dot = T(0);
    Size i = 0, j = 0;

    while (i < len_a && j < len_b) {
        if (inds_a[i] == inds_b[j]) {
            dot += vals_a[i] * vals_b[j];
            ++i; ++j;
        } else if (inds_a[i] < inds_b[j]) {
            ++i;
        } else {
            ++j;
        }
    }

    return dot;
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Batch Balanced KNN
///
/// Finds k nearest neighbors within EACH batch separately.
///
/// @param matrix Input sparse matrix (any backend)
/// @param batch_labels Batch ID for each sample [size = primary_dim]
/// @param n_batches Total number of batches
/// @param k Neighbors per batch
/// @param out_indices Output indices [size = primary_dim * n_batches * k]
/// @param out_distances Output distances [size = primary_dim * n_batches * k]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void bbknn(
    const MatrixT& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);
    const Size neighbors_per_cell = n_batches * k;

    // Input validation
    SCL_CHECK_ARG(k >= 1, "BBKNN: k must be >= 1");
    SCL_CHECK_ARG(n_batches >= 1, "BBKNN: n_batches must be >= 1");
    SCL_CHECK_DIM(batch_labels.size() == N, "BBKNN: Batch labels size mismatch");
    SCL_CHECK_DIM(out_indices.size() >= N * neighbors_per_cell, "BBKNN: Output indices too small");
    SCL_CHECK_DIM(out_distances.size() >= N * neighbors_per_cell, "BBKNN: Output distances too small");

    // Dispatch to optimized backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        // Memory-mapped backend
        kernel::bbknn::mapped::bbknn_mapped_dispatch<MatrixT, IsCSR>(
            matrix, batch_labels, n_batches, k, out_indices, out_distances
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        // Fast in-memory backend
        kernel::bbknn::fast::bbknn_fast<MatrixT, IsCSR>(
            matrix, batch_labels, n_batches, k, out_indices, out_distances
        );
    } else {
        // Generic fallback
        bbknn_generic(matrix, batch_labels, n_batches, k, out_indices, out_distances);
    }
}

/// @brief Generic BBKNN implementation (fallback for unknown matrix types)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void bbknn_generic(
    const MatrixT& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    using T = typename MatrixT::ValueType;
    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);
    const Size neighbors_per_cell = n_batches * k;

    // Precompute norms (SIMD)
    std::vector<T> norms_sq(N);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Index idx = static_cast<Index>(i);
        auto vals = scl::primary_values(matrix, idx);

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        Size j = 0;

        for (; j + lanes <= vals.len; j += lanes) {
            auto v = s::Load(d, vals.ptr + j);
            v_sum = s::MulAdd(v, v, v_sum);
        }

        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; j < vals.len; ++j) {
            sum_sq += vals[j] * vals[j];
        }

        norms_sq[i] = sum_sq;
    });

    // Build batch groups
    auto batch_groups = detail::build_batch_groups(batch_labels, n_batches);

    // Process in chunks
    constexpr Size CHUNK_SIZE = 32;
    const Size n_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        std::vector<typename detail::GenericHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::GenericHeap<T>> heaps(n_batches);

        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + b * k, k);
        }

        Size i_start = chunk_idx * CHUNK_SIZE;
        Size i_end = std::min(N, i_start + CHUNK_SIZE);

        for (Size i = i_start; i < i_end; ++i) {
            Index query_idx = static_cast<Index>(i);
            T q_norm_sq = norms_sq[i];

            auto q_vals = scl::primary_values(matrix, query_idx);
            auto q_inds = scl::primary_indices(matrix, query_idx);

            // Clear heaps
            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].clear();
            }

            // Process each batch
            for (Size b = 0; b < n_batches; ++b) {
                const auto& group = batch_groups[b];
                auto& heap = heaps[b];

                for (Index cand_idx : group) {
                    if (cand_idx == query_idx) continue;

                    T c_norm_sq = norms_sq[cand_idx];

                    // Early rejection
                    T min_dist_sq = q_norm_sq + c_norm_sq - T(2) * std::sqrt(q_norm_sq * c_norm_sq);
                    if (min_dist_sq >= heap.max_dist_sq()) continue;

                    auto c_vals = scl::primary_values(matrix, cand_idx);
                    auto c_inds = scl::primary_indices(matrix, cand_idx);

                    T dot = detail::sparse_dot_generic(
                        q_vals.ptr, q_inds.ptr, q_vals.len,
                        c_vals.ptr, c_inds.ptr, c_vals.len
                    );

                    T dist_sq = q_norm_sq + c_norm_sq - T(2) * dot;
                    if (dist_sq < T(0)) dist_sq = T(0);

                    heap.try_insert(dist_sq, cand_idx);
                }
            }

            // Extract results
            for (Size b = 0; b < n_batches; ++b) {
                Size offset = i * neighbors_per_cell + b * k;
                heaps[b].extract_sorted(
                    out_indices.ptr + offset,
                    out_distances.ptr + offset
                );
            }
        }
    });
}

} // namespace scl::kernel::bbknn
