#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/neighbors_fast_impl.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file neighbors.hpp
/// @brief K-Nearest Neighbors for Sparse Data
///
/// ## Algorithm: Distance Decomposition
///
/// d^2(x, y) = ||x||^2 + ||y||^2 - 2<x,y>
///
/// Key insight: For sparse vectors, computing <x,y> via sparse dot product
/// is O(nnz) instead of O(D) for dense.
///
/// ## Strategy
///
/// 1. Precompute norms: O(nnz) total
/// 2. Compute sparse dot products: O(N^2 * avg_nnz)
/// 3. Heap-based Top-K selection: O(N^2 log k)
///
/// ## Backend Dispatch
///
/// - CustomSparseLike -> neighbors_fast_impl.hpp
/// - VirtualSparseLike -> neighbors_fast_impl.hpp
/// - MappedSparseLike -> neighbors_mapped_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Key Optimizations
///
/// 1. SIMD Norm Computation (4-way unrolled)
/// 2. Adaptive Sparse Dot Product (linear/binary/gallop)
/// 3. Heap-Based Top-K (O(N log k) not O(N log N))
/// 4. Thread-Local Buffers (avoid allocation in hot path)
///
/// ## Complexity
///
/// Time: O(N^2 * avg_nnz + N^2 log k)
/// Space: O(N) per thread for buffers
///
/// ## Performance
///
/// Speedup: 20-100x vs dense for 1-5% density
/// Target: ~10K cells in 5-10 sec (k=15, 1% density, 16 cores)
// =============================================================================

namespace scl::kernel::neighbors {

// =============================================================================
// SECTION 1: Utilities
// =============================================================================

namespace detail {

/// @brief Binary search for sigma (UMAP smooth k-NN)
template <typename T>
SCL_FORCE_INLINE void find_sigma(
    Array<const T> dists,
    T target_perplexity,
    T& rho,
    T& sigma
) {
    const Size k = dists.len;

    if (SCL_UNLIKELY(k == 0)) {
        rho = T(0);
        sigma = T(1);
        return;
    }

    rho = dists[0];

    T lo = T(1e-4);
    T hi = T(1e4);
    sigma = T(1);

    constexpr int MAX_ITER = 64;
    constexpr T TOL = T(1e-5);

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        T p_sum = T(0);

        for (Size j = 0; j < k; ++j) {
            T adjusted_dist = dists[j] - rho;
            if (adjusted_dist < T(0)) adjusted_dist = T(0);
            p_sum += std::exp(-adjusted_dist / sigma);
        }

        T error = p_sum - target_perplexity;
        if (std::abs(error) < TOL) break;

        if (p_sum > target_perplexity) {
            hi = sigma;
        } else {
            lo = sigma;
        }

        sigma = (lo + hi) * T(0.5);
    }
}

/// @brief Generic sparse dot product
template <typename T>
SCL_FORCE_INLINE T sparse_dot_generic(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    T sum = T(0);
    Size i = 0, j = 0;

    while (i < n1 && j < n2) {
        Index r1 = idx1[i], r2 = idx2[j];
        if (r1 == r2) {
            sum += val1[i] * val2[j];
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }

    return sum;
}

/// @brief Generic SIMD norm computation
template <typename T>
SCL_FORCE_INLINE T compute_norm_sq_generic(const T* vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum = s::Zero(d);
    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::MulAdd(v, v, v_sum);
    }

    T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum_sq += vals[k] * vals[k];
    }

    return sum_sq;
}

/// @brief Heap element for generic top-k
template <typename T>
struct HeapElement {
    T dist;
    Index idx;
    bool operator<(const HeapElement& o) const { return dist < o.dist; }
};

/// @brief Generic heap-based top-k
template <typename T>
SCL_FORCE_INLINE void select_topk_generic(
    const T* dists, const Index* indices, Size n, Size k,
    T* out_dists, Index* out_indices
) {
    if (k >= n) {
        std::copy(dists, dists + n, out_dists);
        std::copy(indices, indices + n, out_indices);
        for (Size i = n; i < k; ++i) {
            out_dists[i] = std::numeric_limits<T>::infinity();
            out_indices[i] = -1;
        }
        return;
    }

    std::vector<HeapElement<T>> heap(k);
    for (Size i = 0; i < k; ++i) heap[i] = {dists[i], indices[i]};
    std::make_heap(heap.begin(), heap.end());

    for (Size i = k; i < n; ++i) {
        if (dists[i] < heap[0].dist) {
            std::pop_heap(heap.begin(), heap.end());
            heap[k - 1] = {dists[i], indices[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }

    std::sort_heap(heap.begin(), heap.end());
    for (Size i = 0; i < k; ++i) {
        out_dists[i] = heap[i].dist;
        out_indices[i] = heap[i].idx;
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: Generic Implementation
// =============================================================================

namespace generic {

/// @brief Generic KNN implementation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void knn_generic(
    const MatrixT& matrix,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    using T = typename MatrixT::ValueType;
    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);

    SCL_CHECK_ARG(k >= 1 && k < N, "KNN: k must be in [1, N)");
    SCL_CHECK_DIM(out_indices.len >= N * k, "KNN: Indices size mismatch");
    SCL_CHECK_DIM(out_distances.len >= N * k, "KNN: Distances size mismatch");

    // Precompute norms
    std::vector<T> norms_sq(N);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(i));
        norms_sq[i] = detail::compute_norm_sq_generic(vals.ptr, vals.len);
    });

    // Compute distances + Top-K
    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        thread_local std::vector<T> dist_buffer;
        thread_local std::vector<Index> idx_buffer;

        if (dist_buffer.size() < N) {
            dist_buffer.resize(N);
            idx_buffer.resize(N);
        }

        Index query_idx = static_cast<Index>(i);
        T norm_i = norms_sq[i];

        auto vals_i = scl::primary_values(matrix, query_idx);
        auto inds_i = scl::primary_indices(matrix, query_idx);

        Size valid_count = 0;

        for (Size j = 0; j < N; ++j) {
            if (i == j) continue;

            auto vals_j = scl::primary_values(matrix, static_cast<Index>(j));
            auto inds_j = scl::primary_indices(matrix, static_cast<Index>(j));

            T dot = detail::sparse_dot_generic(
                inds_i.ptr, vals_i.ptr, inds_i.len,
                inds_j.ptr, vals_j.ptr, inds_j.len
            );

            T dist_sq = norm_i + norms_sq[j] - T(2) * dot;
            if (dist_sq < T(0)) dist_sq = T(0);

            dist_buffer[valid_count] = std::sqrt(dist_sq);
            idx_buffer[valid_count] = static_cast<Index>(j);
            valid_count++;
        }

        detail::select_topk_generic(
            dist_buffer.data(), idx_buffer.data(), valid_count, k,
            out_distances.ptr + i * k, out_indices.ptr + i * k
        );
    });
}

} // namespace generic

// =============================================================================
// SECTION 3: Public API
// =============================================================================

/// @brief Exact K-Nearest Neighbors
///
/// @param matrix Input sparse matrix
/// @param k Number of neighbors
/// @param out_indices Output indices [size = primary_dim * k], PRE-ALLOCATED
/// @param out_distances Output distances [size = primary_dim * k], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void knn_sparse(
    const MatrixT& matrix,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);

    SCL_CHECK_ARG(k >= 1 && k < N, "KNN: k must be in [1, N)");
    SCL_CHECK_DIM(out_indices.len >= N * k, "KNN: Indices size mismatch");
    SCL_CHECK_DIM(out_distances.len >= N * k, "KNN: Distances size mismatch");

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        // Use optimized backend
        std::vector<T> norms_sq(N);
        Array<T> norms_arr(norms_sq.data(), N);

        fast::compute_norms_fast<MatrixT, IsCSR>(matrix, norms_arr);

        Array<const T> norms_const(norms_sq.data(), N);
        fast::knn_fast<MatrixT, IsCSR>(matrix, norms_const, k, out_indices, out_distances);
    } else {
        generic::knn_generic(matrix, k, out_indices, out_distances);
    }
}

/// @brief Smooth K-NN with UMAP-style bandwidth
///
/// Computes k-NN plus smooth bandwidth parameters (rho, sigma) for each point.
/// Used in UMAP-style embeddings.
///
/// @param matrix Input sparse matrix
/// @param k Number of neighbors
/// @param out_indices Output indices [size = primary_dim * k], PRE-ALLOCATED
/// @param out_distances Output distances [size = primary_dim * k], PRE-ALLOCATED
/// @param out_rho Output rho values [size = primary_dim], PRE-ALLOCATED
/// @param out_sigma Output sigma values [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void knn_smooth(
    const MatrixT& matrix,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances,
    Array<typename MatrixT::ValueType> out_rho,
    Array<typename MatrixT::ValueType> out_sigma
) {
    using T = typename MatrixT::ValueType;
    const Index R = scl::primary_size(matrix);
    const Size N = static_cast<Size>(R);

    SCL_CHECK_DIM(out_rho.len >= N, "KNN: Rho size mismatch");
    SCL_CHECK_DIM(out_sigma.len >= N, "KNN: Sigma size mismatch");

    // First compute exact KNN
    knn_sparse(matrix, k, out_indices, out_distances);

    // Then compute smooth bandwidth for each point
    const T target_perplexity = std::log2(static_cast<T>(k));

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Array<const T> dists(out_distances.ptr + i * k, k);

        T rho, sigma;
        detail::find_sigma(dists, target_perplexity, rho, sigma);

        out_rho[i] = rho;
        out_sigma[i] = sigma;
    });
}

/// @brief Compute squared norms only (utility function)
///
/// @param matrix Input sparse matrix
/// @param norms_sq Output squared norms [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_norms_sq(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> norms_sq
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::compute_norms_fast<MatrixT, IsCSR>(matrix, norms_sq);
    } else {
        const Index R = scl::primary_size(matrix);
        const Size N = static_cast<Size>(R);

        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto vals = scl::primary_values(matrix, static_cast<Index>(i));
            norms_sq[i] = detail::compute_norm_sq_generic(vals.ptr, vals.len);
        });
    }
}

} // namespace scl::kernel::neighbors
