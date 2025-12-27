#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// FILE: scl/kernel/neighbors.hpp
// BRIEF: K-nearest neighbors computation with SIMD optimization
// =============================================================================

namespace scl::kernel::neighbors {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size CHUNK_SIZE = 32;
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE T dot_linear(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2
) {
    T sum = T(0);
    Size i = 0, j = 0;

    // 8-way skip optimization for large non-overlapping ranges
    while (i + 8 <= n1 && j + 8 <= n2) {
        Index i7 = idx1[i+7], j0 = idx2[j];
        Index i0 = idx1[i], j7 = idx2[j+7];

        if (i7 < j0) { i += 8; continue; }
        if (j7 < i0) { j += 8; continue; }
        break;
    }

    // 4-way skip optimization for non-overlapping ranges
    while (i + 4 <= n1 && j + 4 <= n2) {
        Index i3 = idx1[i+3], j0 = idx2[j];
        Index i0 = idx1[i], j3 = idx2[j+3];

        if (i3 < j0) { i += 4; continue; }
        if (j3 < i0) { j += 4; continue; }
        break;
    }

    // Main merge with prefetch
    while (i < n1 && j < n2) {
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < n1)) {
            SCL_PREFETCH_READ(&idx1[i + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val1[i + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < n2)) {
            SCL_PREFETCH_READ(&idx2[j + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&val2[j + config::PREFETCH_DISTANCE], 0);
        }

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

template <typename T>
SCL_FORCE_INLINE T dot_binary(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    const Index* base = idx_large;
    Size len = n_large;

    for (Size i = 0; i < n_small; ++i) {
        // Prefetch next small element
        if (SCL_LIKELY(i + 4 < n_small)) {
            SCL_PREFETCH_READ(&idx_small[i + 4], 0);
            SCL_PREFETCH_READ(&val_small[i + 4], 0);
        }

        Index target = idx_small[i];
        const Index* it = scl::algo::lower_bound(base, base + len, target);

        if (it != base + len && *it == target) {
            Size offset = static_cast<Size>(it - idx_large);
            sum += val_small[i] * val_large[offset];

            Size step = static_cast<Size>(it - base) + 1;
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        } else {
            Size step = static_cast<Size>(it - base);
            if (SCL_UNLIKELY(step >= len)) break;
            base += step;
            len -= step;
        }
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T dot_gallop(
    const Index* SCL_RESTRICT idx_small, const T* SCL_RESTRICT val_small, Size n_small,
    const Index* SCL_RESTRICT idx_large, const T* SCL_RESTRICT val_large, Size n_large
) {
    T sum = T(0);
    Size j = 0;

    for (Size i = 0; i < n_small && j < n_large; ++i) {
        // Prefetch next small element
        if (SCL_LIKELY(i + 4 < n_small)) {
            SCL_PREFETCH_READ(&idx_small[i + 4], 0);
            SCL_PREFETCH_READ(&val_small[i + 4], 0);
        }

        Index target = idx_small[i];

        // Exponential search (galloping)
        Size step = 1;
        while (j + step < n_large && idx_large[j + step] < target) {
            step *= 2;
        }

        Size lo = j;
        Size hi = (j + step < n_large) ? (j + step) : n_large;

        // Binary search within bounds
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

template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    // Early exit for empty vectors
    if (SCL_UNLIKELY(n1 == 0 || n2 == 0)) {
        return T(0);
    }

    // O(1) range disjointness check - critical optimization
    if (SCL_UNLIKELY(idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0])) {
        return T(0);
    }

    // Ensure n1 <= n2 for algorithm selection
    if (n1 > n2) {
        const Index* tmp_idx = idx1; idx1 = idx2; idx2 = tmp_idx;
        const T* tmp_val = val1; val1 = val2; val2 = tmp_val;
        Size tmp_n = n1; n1 = n2; n2 = tmp_n;
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

template <typename T>
struct HeapElement {
    T dist;
    Index idx;

    bool operator<(const HeapElement& other) const {
        return dist < other.dist;
    }
};

template <typename T>
SCL_FORCE_INLINE void select_topk_heap(
    const T* SCL_RESTRICT distances,
    const Index* SCL_RESTRICT indices,
    Size n,
    Size k,
    T* SCL_RESTRICT out_distances,
    Index* SCL_RESTRICT out_indices,
    HeapElement<T>* SCL_RESTRICT heap
) {
    if (k >= n) {
        scl::algo::copy(distances, out_distances, n);
        scl::algo::copy(indices, out_indices, n);
        for (Size i = n; i < k; ++i) {
            out_distances[i] = std::numeric_limits<T>::infinity();
            out_indices[i] = -1;
        }
        return;
    }

    for (Size i = 0; i < k; ++i) {
        heap[i] = {distances[i], indices[i]};
    }

    // Build max-heap manually for HeapElement
    auto heap_sift_down = [&](Size pos, Size heap_size) {
        while (true) {
            Size largest = pos;
            Size left = 2 * pos + 1;
            Size right = 2 * pos + 2;
            if (left < heap_size && heap[left].dist > heap[largest].dist) largest = left;
            if (right < heap_size && heap[right].dist > heap[largest].dist) largest = right;
            if (largest == pos) break;
            HeapElement<T> tmp = heap[pos]; heap[pos] = heap[largest]; heap[largest] = tmp;
            pos = largest;
        }
    };

    auto heap_sift_up = [&](Size pos) {
        while (pos > 0) {
            Size parent = (pos - 1) / 2;
            if (heap[pos].dist <= heap[parent].dist) break;
            HeapElement<T> tmp = heap[pos]; heap[pos] = heap[parent]; heap[parent] = tmp;
            pos = parent;
        }
    };

    // Build heap
    if (k >= 2) {
        for (Size i = k / 2; i > 0; --i) {
            heap_sift_down(i - 1, k);
        }
    }

    // Select top-k
    for (Size i = k; i < n; ++i) {
        if (distances[i] < heap[0].dist) {
            heap[0] = {distances[i], indices[i]};
            heap_sift_down(0, k);
        }
    }

    // Sort heap (heap sort)
    for (Size i = k; i > 1; --i) {
        HeapElement<T> tmp = heap[0]; heap[0] = heap[i - 1]; heap[i - 1] = tmp;
        heap_sift_down(0, i - 1);
    }

    for (Size i = 0; i < k; ++i) {
        out_distances[i] = heap[i].dist;
        out_indices[i] = heap[i].idx;
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void compute_norms(
    const Sparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(primary_dim), "Norms size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        auto values = matrix.primary_values(idx);
        norms_sq[p] = scl::vectorize::sum_squared(Array<const T>(values.data(), values.size()));
    });
}

template <typename T, bool IsCSR>
void knn(
    const Sparse<T, IsCSR>& matrix,
    Array<const T> norms_sq,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    struct KnnEntry {
        T dist_sq;
        Index idx;
    };

    // Pre-allocate heap storage for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<KnnEntry> heap_pool;
    heap_pool.init(n_threads, k);

    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        KnnEntry* SCL_RESTRICT heap_storage = heap_pool.get(thread_rank);

        Size heap_count = 0;
        T max_dist_sq = std::numeric_limits<T>::max();

        auto sift_up = [&](Size pos) {
            while (pos > 0) {
                Size parent = (pos - 1) / 2;
                if (heap_storage[pos].dist_sq <= heap_storage[parent].dist_sq) break;
                KnnEntry tmp = heap_storage[pos];
                heap_storage[pos] = heap_storage[parent];
                heap_storage[parent] = tmp;
                pos = parent;
            }
        };

        auto sift_down = [&](Size pos) {
            while (true) {
                Size largest = pos;
                Size left = 2 * pos + 1;
                Size right = 2 * pos + 2;
                if (left < heap_count && heap_storage[left].dist_sq > heap_storage[largest].dist_sq)
                    largest = left;
                if (right < heap_count && heap_storage[right].dist_sq > heap_storage[largest].dist_sq)
                    largest = right;
                if (largest == pos) break;
                KnnEntry tmp = heap_storage[pos];
                heap_storage[pos] = heap_storage[largest];
                heap_storage[largest] = tmp;
                pos = largest;
            }
        };

        auto try_insert = [&](T dist_sq, Index idx) {
            if (heap_count < k) {
                heap_storage[heap_count] = {dist_sq, idx};
                sift_up(heap_count);
                heap_count++;
                if (heap_count == k) {
                    max_dist_sq = heap_storage[0].dist_sq;
                }
            } else if (dist_sq < heap_storage[0].dist_sq) {
                heap_storage[0] = {dist_sq, idx};
                sift_down(0);
                max_dist_sq = heap_storage[0].dist_sq;
            }
        };

        T norm_i = norms_sq[i];
        T sqrt_norm_i = std::sqrt(norm_i);
        const Index idx_i = static_cast<Index>(i);
        auto vals_i_arr = matrix.primary_values(idx_i);
        auto inds_i_arr = matrix.primary_indices(idx_i);
        const Size len_i_sz = vals_i_arr.size();

        for (Size j = 0; j < N; ++j) {
            if (SCL_UNLIKELY(i == j)) continue;

            if (SCL_LIKELY(j + config::CHUNK_SIZE < N)) {
                SCL_PREFETCH_READ(&norms_sq[j + config::CHUNK_SIZE], 0);
            }

            T norm_j = norms_sq[j];

            // Cauchy-Schwarz lower bound: |a-b|^2 >= (|a| - |b|)^2
            T sqrt_norm_j = std::sqrt(norm_j);
            T norm_diff = sqrt_norm_i - sqrt_norm_j;
            T min_dist_sq = norm_diff * norm_diff;

            // Early pruning using lower bound
            if (SCL_UNLIKELY(min_dist_sq >= max_dist_sq)) continue;

            const Index idx_j = static_cast<Index>(j);
            auto vals_j_arr = matrix.primary_values(idx_j);
            auto inds_j_arr = matrix.primary_indices(idx_j);
            const Size len_j_sz = vals_j_arr.size();

            T dot = detail::sparse_dot_adaptive(
                inds_i_arr.ptr, vals_i_arr.ptr, len_i_sz,
                inds_j_arr.ptr, vals_j_arr.ptr, len_j_sz
            );

            T dist_sq = norm_i + norm_j - T(2) * dot;
            if (SCL_UNLIKELY(dist_sq < T(0))) dist_sq = T(0);

            try_insert(dist_sq, static_cast<Index>(j));
        }

        // Extract sorted results using heap sort
        auto sort_sift_down = [&](Size pos, Size size) {
            while (true) {
                Size smallest = pos;
                Size left = 2 * pos + 1;
                Size right = 2 * pos + 2;
                if (left < size && heap_storage[left].dist_sq < heap_storage[smallest].dist_sq)
                    smallest = left;
                if (right < size && heap_storage[right].dist_sq < heap_storage[smallest].dist_sq)
                    smallest = right;
                if (smallest == pos) break;
                KnnEntry tmp = heap_storage[pos];
                heap_storage[pos] = heap_storage[smallest];
                heap_storage[smallest] = tmp;
                pos = smallest;
            }
        };

        // Build min-heap for sorting
        if (heap_count >= 2) {
            for (Size m = heap_count / 2; m > 0; --m) {
                sort_sift_down(m - 1, heap_count);
            }
        }

        // Heapsort extract
        for (Size m = heap_count; m > 1; --m) {
            KnnEntry tmp = heap_storage[0];
            heap_storage[0] = heap_storage[m - 1];
            heap_storage[m - 1] = tmp;
            sort_sift_down(0, m - 1);
        }

        // Reverse to get ascending order
        for (Size m = 0; m < heap_count / 2; ++m) {
            KnnEntry tmp = heap_storage[m];
            heap_storage[m] = heap_storage[heap_count - 1 - m];
            heap_storage[heap_count - 1 - m] = tmp;
        }

        for (Size m = 0; m < heap_count; ++m) {
            out_distances[i * k + m] = std::sqrt(heap_storage[m].dist_sq);
            out_indices[i * k + m] = heap_storage[m].idx;
        }
        for (Size m = heap_count; m < k; ++m) {
            out_distances[i * k + m] = std::numeric_limits<T>::infinity();
            out_indices[i * k + m] = -1;
        }
    });
}

} // namespace scl::kernel::neighbors

