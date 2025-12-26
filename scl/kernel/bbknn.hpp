#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/gram.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file bbknn.hpp
/// @brief Batch Balanced K-Nearest Neighbors (BBKNN) Kernel
///
/// Implements BBKNN algorithm for batch effect correction in single-cell data.
///
/// Core Idea: Find k nearest neighbors within EACH batch separately,
/// ensuring balanced representation across batches.
///
/// Algorithm:
///
/// For each cell i:
///   For each batch b:
///     Find k nearest neighbors of i within batch b
///   Merge all neighbors (total: n_batches x k)
///
/// Optimization Strategy:
///
/// 1. Single-Pass Distance Computation:
///    - Compute distances to all cells once
///    - Route to batch-specific heaps during traversal
///
/// 2. Flattened Heap Storage:
///    - Pre-allocate: n_batches x k entries per thread
///    - Cache-friendly: continuous memory layout
///    - Zero allocation in hot loop
///
/// 3. Sparse Distance:
///    - Reuse gram::detail::dot_product (adaptive merge/galloping)
///    - SIMD norm computation
///
/// Performance:
///
/// - Complexity: O(N^2 * nnz) same as KNN, but with batch routing overhead
/// - Throughput: ~5K cells in 10-15 seconds (3 batches, k=5, 16 cores)
/// - Memory: O(N + n_batches*k) thread-local
///
/// Use Cases:
///
/// - Multi-batch integration (10x Genomics datasets)
/// - Cross-platform harmonization (Smart-seq2 + 10x)
/// - Temporal integration (time-series scRNA-seq)
// =============================================================================

namespace scl::kernel::bbknn {

namespace detail {

/// @brief Lightweight max-heap for tracking k nearest neighbors.
///
/// Stores largest distance at top for efficient replacement.
template <typename T>
struct BatchHeap {
    struct Entry {
        T dist_sq;      // Squared distance (avoid sqrt until final output)
        Index cell_idx;  // Global cell index
        
        // Max-heap ordering (largest dist_sq on top)
        bool operator<(const Entry& other) const {
            return dist_sq < other.dist_sq;
        }
    };

    Entry* data;
    Size capacity;
    Size count;

    /// @brief Initialize heap with external storage.
    SCL_FORCE_INLINE void init(Entry* storage, Size k) {
        data = storage;
        capacity = k;
        count = 0;
    }

    /// @brief Try to insert new element.
    ///
    /// If heap not full, always insert. If full, replace top if distance smaller.
    SCL_FORCE_INLINE void try_insert(T dist_sq, Index idx) {
        if (count < capacity) {
            // Heap not full: insert and maintain heap property
            data[count] = {dist_sq, idx};
            count++;
            std::push_heap(data, data + count);
        } else {
            // Heap full: check if new distance is smaller than max
            if (SCL_LIKELY(dist_sq < data[0].dist_sq)) {
                std::pop_heap(data, data + capacity);  // Move max to end
                data[capacity - 1] = {dist_sq, idx};    // Replace
                std::push_heap(data, data + capacity);  // Restore heap
            }
        }
    }

    /// @brief Finalize heap into sorted array (ascending distance).
    SCL_FORCE_INLINE void finalize() {
        // std::sort_heap is optimal for heap data structure
        // VQSort doesn't provide heap-specific operations
        std::sort_heap(data, data + count);
    }
};

} // namespace detail

// =============================================================================
// Public API
// =============================================================================

// =============================================================================
// Layer 1: Virtual Interface (ISparse-based, Generic but Slower)
// =============================================================================

/// @brief Batch Balanced K-Nearest Neighbors (Virtual Interface).
///
/// Generic implementation using ISparse base class.
/// Works with any sparse matrix type but may have virtual call overhead.
///
/// Finds k nearest neighbors within each batch for every cell.
///
/// Output Format: Flattened array with batch-major ordering:
/// [Cell0_Batch0_Neighbors | Cell0_Batch1_Neighbors | ... | Cell1_Batch0_Neighbors | ...]
///
/// Access pattern: output[cell_id * (n_batches * k) + batch_id * k + neighbor_rank]
///
/// @param matrix CSR sparse matrix (via ISparse interface)
/// @param batch_labels Batch ID for each cell [size = n_cells]
/// @param n_batches Total number of batches
/// @param k Number of neighbors to find per batch
/// @param out_indices Output neighbor indices [size = n_cells x n_batches x k]
/// @param out_distances Output distances [size = n_cells x n_batches x k]
template <typename T>
void bbknn(
    const ICSR<T>& matrix,
    Span<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    MutableSpan<Index> out_indices,
    MutableSpan<T> out_distances
) {
    const Index R = matrix.rows();
    const Size N = static_cast<Size>(R);
    const Size neighbors_per_cell = n_batches * k;

    SCL_CHECK_ARG(k >= 1, "BBKNN: k must be >= 1");
    SCL_CHECK_ARG(n_batches >= 1, "BBKNN: n_batches must be >= 1");
    SCL_CHECK_DIM(batch_labels.size == N, "BBKNN: Batch labels size mismatch");
    SCL_CHECK_DIM(out_indices.size == N * neighbors_per_cell, 
                  "BBKNN: Output indices size mismatch");
    SCL_CHECK_DIM(out_distances.size == N * neighbors_per_cell, 
                  "BBKNN: Output distances size mismatch");

    // -------------------------------------------------------------------------
    // Step 1: Precompute Squared Norms (SIMD optimized)
    // -------------------------------------------------------------------------
    
    std::vector<T> norms_sq(N);
    
    scl::threading::parallel_for(0, N, [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        auto vals = matrix.primary_values(row_idx);
        Index len = matrix.primary_length(row_idx);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        Index idx = 0;
        
        for (; idx + static_cast<Index>(lanes) <= len; idx += static_cast<Index>(lanes)) {
            auto v = s::Load(d, vals.ptr + idx);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; idx < len; ++idx) {
            T val = vals[idx];
            sum_sq += val * val;
        }
        
        norms_sq[i] = sum_sq;
    });

    // -------------------------------------------------------------------------
    // Step 2: Compute Distances + Batch-Specific Top-K (Fused, Chunked)
    // -------------------------------------------------------------------------
    
    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        // Thread-local flattened heap storage
        std::vector<typename detail::BatchHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::BatchHeap<T>> heaps(n_batches);
        
        // Initialize batch heaps
        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + (b * k), k);
        }

        size_t i_start = chunk_idx * CHUNK_SIZE;
        size_t i_end = std::min(N, i_start + CHUNK_SIZE);

        for (size_t i = i_start; i < i_end; ++i) {
            // Reset all heaps for new query cell
            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].count = 0;
            }

            const Index query_idx = static_cast<Index>(i);
            auto idx_i = scl::primary_indices(matrix, query_idx);
            auto val_i = scl::primary_values(matrix, query_idx);
            Index len_i = scl::primary_length(matrix, query_idx);
            const T norm_i = norms_sq[i];

            // Single-pass distance computation with batch routing
            for (Size j = 0; j < N; ++j) {
                // Skip self
                if (SCL_UNLIKELY(i == j)) continue;

                // Get batch ID
                int32_t batch_j = batch_labels[j];
                
                // Validate batch ID
                if (SCL_UNLIKELY(batch_j < 0 || static_cast<Size>(batch_j) >= n_batches)) {
                    continue;
                }

                // Compute distance
                const Index target_idx = static_cast<Index>(j);
                auto idx_j = scl::primary_indices(matrix, target_idx);
                auto val_j = scl::primary_values(matrix, target_idx);
                Index len_j = scl::primary_length(matrix, target_idx);

                T dot = scl::kernel::gram::detail::dot_product(
                    idx_i.ptr, val_i.ptr, static_cast<Size>(len_i),
                    idx_j.ptr, val_j.ptr, static_cast<Size>(len_j)
                );

                T d2 = norm_i + norms_sq[j] - static_cast<T>(2.0) * dot;
                if (d2 < static_cast<T>(0.0)) {
                    d2 = static_cast<T>(0.0);
                }

                // Route to batch-specific heap
                heaps[batch_j].try_insert(d2, target_idx);
            }

            // Write output (batch-major layout)
            Size out_base = i * neighbors_per_cell;

            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].finalize();  // Sort by distance
                
                Size batch_offset = out_base + (b * k);
                
                // Write valid neighbors
                for (Size r = 0; r < heaps[b].count; ++r) {
                    out_indices[batch_offset + r] = heaps[b].data[r].cell_idx;
                    out_distances[batch_offset + r] = std::sqrt(heaps[b].data[r].dist_sq);
                }
                
                // Fill remaining slots if batch doesn't have k neighbors
                for (Size r = heaps[b].count; r < k; ++r) {
                    out_indices[batch_offset + r] = static_cast<Index>(-1);
                    out_distances[batch_offset + r] = std::numeric_limits<T>::infinity();
                }
            }
        }
    });
}

// =============================================================================
// Layer 2: Concept-Based (CSRLike, Optimized for Custom/Virtual)
// =============================================================================

/// @brief Batch Balanced K-Nearest Neighbors (Concept-based, Optimized).
///
/// High-performance implementation for CSRLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// Finds k nearest neighbors within each batch for every cell.
///
/// Output Format: Flattened array with batch-major ordering:
/// [Cell0_Batch0_Neighbors | Cell0_Batch1_Neighbors | ... | Cell1_Batch0_Neighbors | ...]
///
/// Access pattern: output[cell_id * (n_batches * k) + batch_id * k + neighbor_rank]
///
/// @tparam MatrixT Any CSR-like matrix type (CustomSparse or VirtualSparse)
/// @param matrix Input CSR-like sparse matrix (cells x features)
/// @param batch_labels Batch ID for each cell [size = n_cells]
/// @param n_batches Total number of batches
/// @param k Number of neighbors to find per batch
/// @param out_indices Output neighbor indices [size = n_cells x n_batches x k]
/// @param out_distances Output distances [size = n_cells x n_batches x k]
template <CSRLike MatrixT>
void bbknn(
    const MatrixT& matrix,
    Span<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    MutableSpan<Index> out_indices,
    MutableSpan<typename MatrixT::ValueType> out_distances
) {
    using T = typename MatrixT::ValueType;
    const Index R = scl::rows(matrix);
    const Size N = static_cast<Size>(R);
    const Size neighbors_per_cell = n_batches * k;

    SCL_CHECK_ARG(k >= 1, "BBKNN: k must be >= 1");
    SCL_CHECK_ARG(n_batches >= 1, "BBKNN: n_batches must be >= 1");
    SCL_CHECK_DIM(batch_labels.size == N, "BBKNN: Batch labels size mismatch");
    SCL_CHECK_DIM(out_indices.size == N * neighbors_per_cell, 
                  "BBKNN: Output indices size mismatch");
    SCL_CHECK_DIM(out_distances.size == N * neighbors_per_cell, 
                  "BBKNN: Output distances size mismatch");

    // -------------------------------------------------------------------------
    // Step 1: Precompute Squared Norms (SIMD optimized)
    // -------------------------------------------------------------------------
    
    std::vector<T> norms_sq(N);
    
    scl::threading::parallel_for(0, N, [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        auto vals = scl::primary_values(matrix, row_idx);
        Index len = scl::primary_length(matrix, row_idx);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        Index idx = 0;
        
        for (; idx + static_cast<Index>(lanes) <= len; idx += static_cast<Index>(lanes)) {
            auto v = s::Load(d, vals.ptr + idx);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; idx < len; ++idx) {
            T val = vals[idx];
            sum_sq += val * val;
        }
        
        norms_sq[i] = sum_sq;
    });

    // -------------------------------------------------------------------------
    // Step 2: Compute Distances + Batch-Specific Top-K (Fused, Chunked)
    // -------------------------------------------------------------------------
    
    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        // Thread-local flattened heap storage
        std::vector<typename detail::BatchHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::BatchHeap<T>> heaps(n_batches);
        
        // Initialize batch heaps
        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + (b * k), k);
        }

        size_t i_start = chunk_idx * CHUNK_SIZE;
        size_t i_end = std::min(N, i_start + CHUNK_SIZE);

        for (size_t i = i_start; i < i_end; ++i) {
            // Reset all heaps for new query cell
            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].count = 0;
            }

            const Index query_idx = static_cast<Index>(i);
            auto idx_i = scl::primary_indices(matrix, query_idx);
            auto val_i = scl::primary_values(matrix, query_idx);
            Index len_i = scl::primary_length(matrix, query_idx);
            const T norm_i = norms_sq[i];

            // Single-pass distance computation with batch routing
            for (Size j = 0; j < N; ++j) {
                // Skip self
                if (SCL_UNLIKELY(i == j)) continue;

                // Get batch ID
                int32_t batch_j = batch_labels[j];
                
                // Validate batch ID
                if (SCL_UNLIKELY(batch_j < 0 || static_cast<Size>(batch_j) >= n_batches)) {
                    continue;
                }

                // Compute distance
                const Index target_idx = static_cast<Index>(j);
                auto idx_j = scl::primary_indices(matrix, target_idx);
                auto val_j = scl::primary_values(matrix, target_idx);
                Index len_j = scl::primary_length(matrix, target_idx);

                T dot = scl::kernel::gram::detail::dot_product(
                    idx_i.ptr, val_i.ptr, static_cast<Size>(len_i),
                    idx_j.ptr, val_j.ptr, static_cast<Size>(len_j)
                );

                T d2 = norm_i + norms_sq[j] - static_cast<T>(2.0) * dot;
                if (d2 < static_cast<T>(0.0)) {
                    d2 = static_cast<T>(0.0);
                }

                // Route to batch-specific heap
                heaps[batch_j].try_insert(d2, target_idx);
            }

            // Write output (batch-major layout)
            Size out_base = i * neighbors_per_cell;

            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].finalize();  // Sort by distance
                
                Size batch_offset = out_base + (b * k);
                
                // Write valid neighbors
                for (Size r = 0; r < heaps[b].count; ++r) {
                    out_indices[batch_offset + r] = heaps[b].data[r].cell_idx;
                    out_distances[batch_offset + r] = std::sqrt(heaps[b].data[r].dist_sq);
                }
                
                // Fill remaining slots if batch doesn't have k neighbors
                for (Size r = heaps[b].count; r < k; ++r) {
                    out_indices[batch_offset + r] = static_cast<Index>(-1);
                    out_distances[batch_offset + r] = std::numeric_limits<T>::infinity();
                }
            }
        }
    });
}

/// @brief BBKNN with automatic batch count detection (Concept-based, Optimized).
///
/// Convenience wrapper that infers n_batches from batch_labels.
///
/// @tparam MatrixT Any CSR-like matrix type (CustomSparse or VirtualSparse)
/// @param matrix Input CSR-like sparse matrix
/// @param batch_labels Batch ID for each cell
/// @param k Number of neighbors per batch
/// @param out_indices Output indices [size will be determined]
/// @param out_distances Output distances [size will be determined]
/// @return Number of batches detected
template <CSRLike MatrixT>
Size bbknn_auto(
    const MatrixT& matrix,
    Span<const int32_t> batch_labels,
    Size k,
    MutableSpan<Index> out_indices,
    MutableSpan<typename MatrixT::ValueType> out_distances
) {
    const Size N = static_cast<Size>(scl::rows(matrix));
    
    // Find max batch ID
    int32_t max_batch = -1;
    for (Size i = 0; i < N; ++i) {
        if (batch_labels[i] > max_batch) {
            max_batch = batch_labels[i];
        }
    }
    
    Size n_batches = static_cast<Size>(max_batch + 1);
    
    SCL_CHECK_ARG(n_batches > 0, "BBKNN: No valid batches found");
    
    bbknn(matrix, batch_labels, n_batches, k, out_indices, out_distances);
    
    return n_batches;
}

} // namespace scl::kernel::bbknn

