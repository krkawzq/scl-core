#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/neighbors_mapped_impl.hpp"

// =============================================================================
/// @file neighbors_fast_impl.hpp
/// @brief Extreme Performance K-Nearest Neighbors
///
/// ## Key Optimizations
///
/// 1. SIMD Norm Computation
///    - 4-way unrolled FMA accumulation
///    - Parallel over all rows
///
/// 2. Heap-Based Top-K Selection
///    - O(N log k) instead of O(N log N) full sort
///    - Max-heap for k-smallest elements
///
/// 3. Thread-Local Buffers
///    - Reuse distance/index buffers across queries
///    - Avoid allocation in hot path
///
/// 4. Sparse Dot Product Optimization
///    - Adaptive algorithm (linear/binary/gallop)
///    - Prefetch for random index access
///
/// 5. Distance Decomposition
///    - d^2(x,y) = ||x||^2 + ||y||^2 - 2<x,y>
///    - Precompute norms once
///
/// Performance Target: 1.5-2x faster than generic
// =============================================================================

namespace scl::kernel::neighbors::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size CHUNK_SIZE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD squared norm computation (4-way unrolled)
template <typename T>
SCL_FORCE_INLINE T compute_norm_sq_simd(const T* SCL_RESTRICT vals, Index len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Index k = 0;

    // 4-way unrolled
    for (; k + 4 * static_cast<Index>(lanes) <= len; k += 4 * static_cast<Index>(lanes)) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
        v_sum2 = s::MulAdd(v2, v2, v_sum2);
        v_sum3 = s::MulAdd(v3, v3, v_sum3);
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
        auto v = s::Load(d, vals + k);
        v_sum = s::MulAdd(v, v, v_sum);
    }

    T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum_sq += vals[k] * vals[k];
    }

    return sum_sq;
}

// =============================================================================
// SECTION 3: Sparse Dot Product (Adaptive)
// =============================================================================

/// @brief Linear merge dot product
template <typename T>
SCL_FORCE_INLINE T dot_linear(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum = T(0);
    Size i = 0, j = 0;

    while (i < n1 && j < n2) {
        Index r1 = idx1[i];
        Index r2 = idx2[j];

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

/// @brief Binary search dot product
template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    const Index* base = idx_large;
    Size len = n_large;

    for (Size i = 0; i < n_small; ++i) {
        Index target = idx_small[i];
        auto it = std::lower_bound(base, base + len, target);

        if (it != base + len && *it == target) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];

            Size step = static_cast<Size>(it - base) + 1;
            if (step >= len) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (step >= len) break;
            base += step;
            len -= step;
        }
    }

    return sum;
}

/// @brief Galloping search dot product
template <typename T>
SCL_FORCE_INLINE T dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    Size j = 0;

    for (Size i = 0; i < n_small && j < n_large; ++i) {
        Index target = idx_small[i];

        Size step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        Size lo = j;
        Size hi = std::min(j + step, n_large);

        while (lo < hi) {
            Size mid = lo + (hi - lo) / 2;
            if (idx_large[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        j = lo;
        if (j < n_large && idx_large[j] == target) {
            sum += val_small[i] * val_large[j];
            ++j;
        }
    }

    return sum;
}

/// @brief Adaptive dot product dispatcher
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    if (n1 > n2) {
        std::swap(idx1, idx2);
        std::swap(val1, val2);
        std::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        return dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else if (ratio >= config::RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear(idx1, val1, n1, idx2, val2, n2);
    }
}

// =============================================================================
// SECTION 4: Heap-Based Top-K Selection
// =============================================================================

/// @brief Max-heap element
template <typename T>
struct HeapElement {
    T dist;
    Index idx;

    bool operator<(const HeapElement& other) const {
        return dist < other.dist;  // Max-heap: larger on top
    }
};

/// @brief Heap-based top-k selection (O(N log k))
template <typename T>
SCL_FORCE_INLINE void select_topk_heap(
    const T* SCL_RESTRICT distances,
    const Index* SCL_RESTRICT indices,
    Size n,
    Size k,
    T* SCL_RESTRICT out_distances,
    Index* SCL_RESTRICT out_indices
) {
    if (k >= n) {
        // Copy all
        std::copy(distances, distances + n, out_distances);
        std::copy(indices, indices + n, out_indices);
        for (Size i = n; i < k; ++i) {
            out_distances[i] = std::numeric_limits<T>::infinity();
            out_indices[i] = -1;
        }
        return;
    }

    // Use max-heap to track k smallest
    std::vector<HeapElement<T>> heap(k);

    // Initialize with first k elements
    for (Size i = 0; i < k; ++i) {
        heap[i] = {distances[i], indices[i]};
    }
    std::make_heap(heap.begin(), heap.end());

    // Process remaining elements
    for (Size i = k; i < n; ++i) {
        if (distances[i] < heap[0].dist) {
            std::pop_heap(heap.begin(), heap.end());
            heap[k - 1] = {distances[i], indices[i]};
            std::push_heap(heap.begin(), heap.end());
        }
    }

    // Sort heap to get ordered results
    std::sort_heap(heap.begin(), heap.end());

    // Copy to output (now in ascending order)
    for (Size i = 0; i < k; ++i) {
        out_distances[i] = heap[i].dist;
        out_indices[i] = heap[i].idx;
    }
}

} // namespace detail

// =============================================================================
// SECTION 5: CustomSparse Fast Path
// =============================================================================

/// @brief Compute squared norms for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void compute_norms_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(primary_dim), "Norms size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;

        norms_sq[p] = detail::compute_norm_sq_simd(matrix.data + start, len);
    });
}

/// @brief Full KNN for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void knn_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const T> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);

    // Process in chunks with thread-local buffers
    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        // Thread-local buffers
        thread_local std::vector<T> dist_buffer;
        thread_local std::vector<Index> idx_buffer;

        if (dist_buffer.size() < N) {
            dist_buffer.resize(N);
            idx_buffer.resize(N);
        }

        T norm_i = norms_sq[i];

        Index start_i = matrix.indptr[i];
        Index end_i = matrix.indptr[i + 1];
        Index len_i = end_i - start_i;

        const T* vals_i = matrix.data + start_i;
        const Index* inds_i = matrix.indices + start_i;

        Size valid_count = 0;

        for (Size j = 0; j < N; ++j) {
            if (i == j) continue;

            Index start_j = matrix.indptr[j];
            Index end_j = matrix.indptr[j + 1];
            Index len_j = end_j - start_j;

            const T* vals_j = matrix.data + start_j;
            const Index* inds_j = matrix.indices + start_j;

            T dot = detail::sparse_dot_adaptive(
                inds_i, vals_i, static_cast<Size>(len_i),
                inds_j, vals_j, static_cast<Size>(len_j)
            );

            T dist_sq = norm_i + norms_sq[j] - T(2) * dot;
            if (dist_sq < T(0)) dist_sq = T(0);

            dist_buffer[valid_count] = std::sqrt(dist_sq);
            idx_buffer[valid_count] = static_cast<Index>(j);
            valid_count++;
        }

        // Top-K selection
        detail::select_topk_heap(
            dist_buffer.data(), idx_buffer.data(), valid_count, k,
            out_distances.ptr + i * k, out_indices.ptr + i * k
        );
    });
}

// =============================================================================
// SECTION 6: VirtualSparse Fast Path
// =============================================================================

/// @brief Compute squared norms for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void compute_norms_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(primary_dim), "Norms size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);

        norms_sq[p] = detail::compute_norm_sq_simd(vals, len);
    });
}

/// @brief Full KNN for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void knn_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const T> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        thread_local std::vector<T> dist_buffer;
        thread_local std::vector<Index> idx_buffer;

        if (dist_buffer.size() < N) {
            dist_buffer.resize(N);
            idx_buffer.resize(N);
        }

        T norm_i = norms_sq[i];
        Index len_i = matrix.lengths[i];
        const T* vals_i = static_cast<const T*>(matrix.data_ptrs[i]);
        const Index* inds_i = static_cast<const Index*>(matrix.indices_ptrs[i]);

        Size valid_count = 0;

        for (Size j = 0; j < N; ++j) {
            if (i == j) continue;

            Index len_j = matrix.lengths[j];
            const T* vals_j = static_cast<const T*>(matrix.data_ptrs[j]);
            const Index* inds_j = static_cast<const Index*>(matrix.indices_ptrs[j]);

            T dot = detail::sparse_dot_adaptive(
                inds_i, vals_i, static_cast<Size>(len_i),
                inds_j, vals_j, static_cast<Size>(len_j)
            );

            T dist_sq = norm_i + norms_sq[j] - T(2) * dot;
            if (dist_sq < T(0)) dist_sq = T(0);

            dist_buffer[valid_count] = std::sqrt(dist_sq);
            idx_buffer[valid_count] = static_cast<Index>(j);
            valid_count++;
        }

        detail::select_topk_heap(
            dist_buffer.data(), idx_buffer.data(), valid_count, k,
            out_distances.ptr + i * k, out_indices.ptr + i * k
        );
    });
}

// =============================================================================
// SECTION 7: Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch norm computation
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void compute_norms_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> norms_sq
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::neighbors::mapped::compute_norms_mapped_dispatch<MatrixT, IsCSR>(matrix, norms_sq);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        compute_norms_custom(matrix, norms_sq);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        compute_norms_virtual(matrix, norms_sq);
    }
}

/// @brief Auto-dispatch full KNN
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void knn_fast(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::neighbors::mapped::knn_mapped_dispatch<MatrixT, IsCSR>(
            matrix, norms_sq, k, out_indices, out_distances);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        knn_custom(matrix, norms_sq, k, out_indices, out_distances);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        knn_virtual(matrix, norms_sq, k, out_indices, out_distances);
    }
}

} // namespace scl::kernel::neighbors::fast
