#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file neighbors_mapped_impl.hpp
/// @brief KNN Operations for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Norm Computation
///    - Prefetch hints for OS page cache
///    - 4-way SIMD unrolled accumulation
///
/// 2. Heap-Based Top-K Selection
///    - O(N log k) instead of O(N log N)
///    - Thread-local heap buffers
///
/// 3. Adaptive Sparse Dot Product
///    - Linear/binary/galloping based on length ratio
///
/// 4. Full Parallelization
///    - No serial chunk loop, direct parallel_for
// =============================================================================

namespace scl::kernel::neighbors::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}

// =============================================================================
// SECTION 2: Utilities
// =============================================================================

namespace detail {

/// @brief SIMD squared norm (4-way unrolled)
template <typename T>
SCL_FORCE_INLINE T compute_norm_sq_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
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

/// @brief Adaptive sparse dot product
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    if (n1 == 0 || n2 == 0) return T(0);

    if (n1 > n2) {
        std::swap(idx1, idx2);
        std::swap(val1, val2);
        std::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        // Galloping
        T sum = T(0);
        Size j = 0;

        for (Size i = 0; i < n1 && j < n2; ++i) {
            Index target = idx1[i];

            Size step = 1;
            while (j + step < n2 && idx2[j + step] < target) step *= 2;

            Size lo = j, hi = std::min(j + step, n2);
            while (lo < hi) {
                Size mid = lo + (hi - lo) / 2;
                if (idx2[mid] < target) lo = mid + 1;
                else hi = mid;
            }

            j = lo;
            if (j < n2 && idx2[j] == target) {
                sum += val1[i] * val2[j];
                ++j;
            }
        }
        return sum;
    } else if (ratio >= config::RATIO_THRESHOLD) {
        // Binary search
        T sum = T(0);
        const Index* base = idx2;
        Size len = n2;

        for (Size i = 0; i < n1; ++i) {
            Index target = idx1[i];
            auto it = std::lower_bound(base, base + len, target);

            if (it != base + len && *it == target) {
                Size offset = static_cast<Size>(it - idx2);
                sum += val1[i] * val2[offset];

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
    } else {
        // Linear merge
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
}

/// @brief Heap element for top-k
template <typename T>
struct HeapElement {
    T dist;
    Index idx;
    bool operator<(const HeapElement& o) const { return dist < o.dist; }
};

/// @brief Heap-based top-k selection
template <typename T>
SCL_FORCE_INLINE void select_topk_heap(
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
// SECTION 3: MappedCustomSparse
// =============================================================================

/// @brief Compute squared norms for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(n_primary), "Norms size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        if (values.len == 0) {
            norms_sq[p] = T(0);
        } else {
            norms_sq[p] = detail::compute_norm_sq_simd(values.ptr, values.len);
        }
    });
}

/// @brief Full KNN for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void knn_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size N = static_cast<Size>(n_primary);

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        thread_local std::vector<T> dist_buffer;
        thread_local std::vector<Index> idx_buffer;

        if (dist_buffer.size() < N) {
            dist_buffer.resize(N);
            idx_buffer.resize(N);
        }

        auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
        auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));
        T norm_i = norms_sq[i];

        Size valid_count = 0;

        for (Size j = 0; j < N; ++j) {
            if (i == j) continue;

            auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
            auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));

            T dot = detail::sparse_dot_adaptive(
                indices_i.ptr, values_i.ptr, values_i.len,
                indices_j.ptr, values_j.ptr, values_j.len
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

/// @brief Euclidean norms (sqrt of squared norms)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_euclidean_norms_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms
) {
    compute_norms_mapped(matrix, norms);

    scl::threading::parallel_for(Size(0), norms.len, [&](size_t i) {
        norms[i] = std::sqrt(norms[i]);
    });
}

// =============================================================================
// SECTION 4: MappedVirtualSparse
// =============================================================================

/// @brief Compute squared norms for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(n_primary), "Norms size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        if (values.len == 0) {
            norms_sq[p] = T(0);
        } else {
            norms_sq[p] = detail::compute_norm_sq_simd(values.ptr, values.len);
        }
    });
}

/// @brief Full KNN for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void knn_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size N = static_cast<Size>(n_primary);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        thread_local std::vector<T> dist_buffer;
        thread_local std::vector<Index> idx_buffer;

        if (dist_buffer.size() < N) {
            dist_buffer.resize(N);
            idx_buffer.resize(N);
        }

        auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
        auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));
        T norm_i = norms_sq[i];

        Size valid_count = 0;

        for (Size j = 0; j < N; ++j) {
            if (i == j) continue;

            auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
            auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));

            T dot = detail::sparse_dot_adaptive(
                indices_i.ptr, values_i.ptr, values_i.len,
                indices_j.ptr, values_j.ptr, values_j.len
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
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch norm computation
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_norms_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> norms_sq
) {
    compute_norms_mapped(matrix, norms_sq);
}

/// @brief Auto-dispatch full KNN
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void knn_mapped_dispatch(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    knn_mapped(matrix, norms_sq, k, out_indices, out_distances);
}

} // namespace scl::kernel::neighbors::mapped
