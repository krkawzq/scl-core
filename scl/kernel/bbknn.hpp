#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/gram.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file bbknn.hpp
/// @brief Batch Balanced K-Nearest Neighbors
///
/// Finds k nearest neighbors within EACH batch separately.
///
/// Optimization:
/// - Single-pass distance computation
/// - Flattened heap storage (cache-friendly)
/// - SIMD norm computation
/// - Parallel over cells/samples
///
/// Complexity: O(N^2 * nnz)
/// Throughput: ~5K cells in 10-15 sec (3 batches, k=5, 16 cores)
// =============================================================================

namespace scl::kernel::bbknn {

namespace detail {

/// @brief Max-heap for k nearest neighbors
template <typename T>
struct BatchHeap {
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

    SCL_FORCE_INLINE void init(Entry* storage, Size k) {
        data = storage;
        capacity = k;
        count = 0;
    }

    SCL_FORCE_INLINE void try_insert(T dist_sq, Index idx) {
        if (count < capacity) {
            data[count] = {dist_sq, idx};
            count++;
            std::push_heap(data, data + count);
        } else {
            if (dist_sq < data[0].dist_sq) {
                std::pop_heap(data, data + count);
                data[count - 1] = {dist_sq, idx};
                std::push_heap(data, data + count);
            }
        }
    }

    SCL_FORCE_INLINE void extract_sorted(Index* out_indices, T* out_distances) {
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

} // namespace detail

// =============================================================================
// Public API
// =============================================================================

/// @brief Batch Balanced KNN (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param batch_labels Batch ID for each cell/sample [size = primary_dim]
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
    using T = typename MatrixT::ValueType;
    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);
    const Size neighbors_per_cell = n_batches * k;

    SCL_CHECK_ARG(k >= 1, "BBKNN: k must be >= 1");
    SCL_CHECK_ARG(n_batches >= 1, "BBKNN: n_batches must be >= 1");
    SCL_CHECK_DIM(batch_labels.size() == N, "BBKNN: Batch labels size mismatch");
    SCL_CHECK_DIM(out_indices.size() == N * neighbors_per_cell, "BBKNN: Output indices mismatch");
    SCL_CHECK_DIM(out_distances.size() == N * neighbors_per_cell, "BBKNN: Output distances mismatch");

    // Precompute norms (SIMD)
    std::vector<T> norms_sq(N);
    
    scl::threading::parallel_for(0, N, [&](size_t i) {
        Index idx = static_cast<Index>(i);
        auto vals = scl::primary_values(matrix, idx);
        Index len = scl::primary_length(matrix, idx);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        Index j = 0;
        
        for (; j + static_cast<Index>(lanes) <= len; j += static_cast<Index>(lanes)) {
            auto v = s::Load(d, vals.ptr + j);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; j < len; ++j) {
            T val = vals[j];
            sum_sq += val * val;
        }
        
        norms_sq[i] = sum_sq;
    });

    // Compute distances + batch-specific top-k
    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        std::vector<typename detail::BatchHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::BatchHeap<T>> heaps(n_batches);
        
        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + (b * k), k);
        }

        size_t i_start = chunk_idx * CHUNK_SIZE;
        size_t i_end = std::min(N, i_start + CHUNK_SIZE);

        for (size_t i = i_start; i < i_end; ++i) {
            Index query_idx = static_cast<Index>(i);
            T norm_i = norms_sq[i];
            
            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].count = 0;
            }

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                
                int32_t batch_j = batch_labels[j];
                if (batch_j < 0 || static_cast<Size>(batch_j) >= n_batches) continue;

                auto vals_i = scl::primary_values(matrix, query_idx);
                auto vals_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto inds_i = scl::primary_indices(matrix, query_idx);
                auto inds_j = scl::primary_indices(matrix, static_cast<Index>(j));
                
                T dot = scl::kernel::gram::detail::dot_product(
                    vals_i.ptr, inds_i.ptr, scl::primary_length(matrix, query_idx),
                    vals_j.ptr, inds_j.ptr, scl::primary_length(matrix, static_cast<Index>(j))
                );
                
                T dist_sq = norm_i + norms_sq[j] - static_cast<T>(2.0) * dot;
                if (dist_sq < 0) dist_sq = 0;

                heaps[batch_j].try_insert(dist_sq, static_cast<Index>(j));
            }

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
