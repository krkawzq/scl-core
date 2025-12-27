#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// Batch Balanced KNN (BBKNN) for Sparse Matrices
//
// Key Optimizations:
// 1. Batch-Grouped Processing - Group samples by batch, only compare within batches
// 2. Cache-Blocked Distance Computation - Cache query row data in registers/L1
// 3. Optimized Heap Operations - Fixed-size k-heap with manual sift
// 4. SIMD Norm/Dot Computation - 4-way unrolled accumulation with FMA
// =============================================================================

namespace scl::kernel::bbknn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size MIN_SAMPLES_PARALLEL = 128;
}

// =============================================================================
// Optimized K-Heap
// =============================================================================

namespace detail {

template <typename T>
struct KHeap {
    struct Entry {
        T dist_sq;
        Index idx;
    };

    Entry* data;
    Size k;
    Size count;

    SCL_FORCE_INLINE void init(Entry* storage, Size capacity) {
        data = storage;
        k = capacity;
        count = 0;
    }

    SCL_FORCE_INLINE void clear() {
        count = 0;
    }

    SCL_FORCE_INLINE T max_dist_sq() const {
        return (count > 0) ? data[0].dist_sq : std::numeric_limits<T>::max();
    }

    SCL_FORCE_INLINE void sift_down(Size i) {
        while (true) {
            Size largest = i;
            Size left = 2 * i + 1;
            Size right = 2 * i + 2;

            if (left < count && data[left].dist_sq > data[largest].dist_sq) {
                largest = left;
            }
            if (right < count && data[right].dist_sq > data[largest].dist_sq) {
                largest = right;
            }

            if (largest == i) break;

            Entry tmp = data[i];
            data[i] = data[largest];
            data[largest] = tmp;
            i = largest;
        }
    }

    SCL_FORCE_INLINE void sift_up(Size i) {
        while (i > 0) {
            Size parent = (i - 1) / 2;
            if (data[i].dist_sq <= data[parent].dist_sq) break;
            Entry tmp = data[i];
            data[i] = data[parent];
            data[parent] = tmp;
            i = parent;
        }
    }

    SCL_FORCE_INLINE void try_insert(T dist_sq, Index idx) {
        if (count < k) {
            data[count] = {dist_sq, idx};
            sift_up(count);
            count++;
        } else if (dist_sq < data[0].dist_sq) {
            data[0] = {dist_sq, idx};
            sift_down(0);
        }
    }

    void extract_sorted(Index* out_indices, T* out_distances) {
        // Heap sort by dist_sq ascending
        auto sort_sift_down = [this](Size pos, Size size) {
            while (true) {
                Size smallest = pos;
                Size left = 2 * pos + 1;
                Size right = 2 * pos + 2;
                if (left < size && data[left].dist_sq < data[smallest].dist_sq) smallest = left;
                if (right < size && data[right].dist_sq < data[smallest].dist_sq) smallest = right;
                if (smallest == pos) break;
                Entry tmp = data[pos]; data[pos] = data[smallest]; data[smallest] = tmp;
                pos = smallest;
            }
        };

        // Build min-heap
        if (count >= 2) {
            for (Size i = count / 2; i > 0; --i) {
                sort_sift_down(i - 1, count);
            }
        }

        // Heapsort extract
        for (Size i = count; i > 1; --i) {
            Entry tmp = data[0]; data[0] = data[i - 1]; data[i - 1] = tmp;
            sort_sift_down(0, i - 1);
        }

        // Reverse for ascending order
        for (Size i = 0; i < count / 2; ++i) {
            Entry tmp = data[i]; data[i] = data[count - 1 - i]; data[count - 1 - i] = tmp;
        }

        for (Size i = 0; i < count; ++i) {
            out_indices[i] = data[i].idx;
            out_distances[i] = std::sqrt(data[i].dist_sq);
        }

        for (Size i = count; i < k; ++i) {
            out_indices[i] = -1;
            out_distances[i] = std::numeric_limits<T>::infinity();
        }
    }
};

// =============================================================================
// SIMD Utilities
// =============================================================================

template <typename T>
SCL_FORCE_INLINE T sparse_dot(
    const T* SCL_RESTRICT vals_a, const Index* SCL_RESTRICT inds_a, Size len_a,
    const T* SCL_RESTRICT vals_b, const Index* SCL_RESTRICT inds_b, Size len_b
) {
    // Early exit for empty vectors
    if (SCL_UNLIKELY(len_a == 0 || len_b == 0)) {
        return T(0);
    }

    // O(1) range disjointness check
    if (SCL_UNLIKELY(inds_a[len_a-1] < inds_b[0] || inds_b[len_b-1] < inds_a[0])) {
        return T(0);
    }

    T dot = T(0);
    Size i = 0, j = 0;

    // 8-way skip optimization for large non-overlapping ranges
    while (i + 8 <= len_a && j + 8 <= len_b) {
        Index ia7 = inds_a[i+7], ib0 = inds_b[j];
        Index ia0 = inds_a[i], ib7 = inds_b[j+7];

        if (ia7 < ib0) { i += 8; continue; }
        if (ib7 < ia0) { j += 8; continue; }
        break;
    }

    // 4-way skip optimization
    while (i + 4 <= len_a && j + 4 <= len_b) {
        Index ia3 = inds_a[i+3], ib0 = inds_b[j];
        Index ia0 = inds_a[i], ib3 = inds_b[j+3];

        if (ia3 < ib0) { i += 4; continue; }
        if (ib3 < ia0) { j += 4; continue; }
        break;
    }

    // Main merge with prefetch
    while (i < len_a && j < len_b) {
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < len_a)) {
            SCL_PREFETCH_READ(&inds_a[i + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&vals_a[i + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < len_b)) {
            SCL_PREFETCH_READ(&inds_b[j + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&vals_b[j + config::PREFETCH_DISTANCE], 0);
        }

        Index ia = inds_a[i];
        Index ib = inds_b[j];

        if (ia == ib) {
            dot += vals_a[i] * vals_b[j];
            ++i; ++j;
        } else if (ia < ib) {
            ++i;
        } else {
            ++j;
        }
    }

    return dot;
}

} // namespace detail

// =============================================================================
// Norm Precomputation
// =============================================================================

template <typename T, bool IsCSR>
void compute_norms(
    const Sparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = matrix.primary_dim();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = matrix.primary_values(static_cast<Index>(p));
        norms_sq[p] = scl::vectorize::sum_squared(Array<const T>(vals.data(), vals.size()));
    });
}

// =============================================================================
// Batch-Grouped BBKNN
// =============================================================================

// Batch group structure for memory-efficient storage
struct BatchGroups {
    Index* indices;      // All indices concatenated
    Size* offsets;       // Offset for each batch [n_batches + 1]
    Size n_batches;
    Size total_size;

    SCL_FORCE_INLINE Size batch_size(Size b) const {
        return offsets[b + 1] - offsets[b];
    }

    SCL_FORCE_INLINE const Index* batch_data(Size b) const {
        return indices + offsets[b];
    }
};

inline void build_batch_groups(
    Array<const int32_t> batch_labels,
    Size n_batches,
    BatchGroups& out
) {
    const Size N = batch_labels.len;

    // Count elements per batch
    out.offsets = scl::memory::aligned_alloc<Size>(n_batches + 1, SCL_ALIGNMENT);
    scl::algo::zero(out.offsets, n_batches + 1);

    for (Size i = 0; i < N; ++i) {
        int32_t b = batch_labels[i];
        if (b >= 0 && static_cast<Size>(b) < n_batches) {
            out.offsets[b + 1]++;
        }
    }

    // Prefix sum to get offsets
    for (Size b = 1; b <= n_batches; ++b) {
        out.offsets[b] += out.offsets[b - 1];
    }

    out.total_size = out.offsets[n_batches];
    out.n_batches = n_batches;

    // Fill indices
    out.indices = scl::memory::aligned_alloc<Index>(out.total_size, SCL_ALIGNMENT);
    Size* write_pos = scl::memory::aligned_alloc<Size>(n_batches, SCL_ALIGNMENT);
    scl::algo::copy(out.offsets, write_pos, n_batches);

    for (Size i = 0; i < N; ++i) {
        int32_t b = batch_labels[i];
        if (b >= 0 && static_cast<Size>(b) < n_batches) {
            out.indices[write_pos[b]++] = static_cast<Index>(i);
        }
    }

    scl::memory::aligned_free(write_pos, SCL_ALIGNMENT);
}

inline void free_batch_groups(BatchGroups& groups) {
    if (groups.indices) scl::memory::aligned_free(groups.indices, SCL_ALIGNMENT);
    if (groups.offsets) scl::memory::aligned_free(groups.offsets, SCL_ALIGNMENT);
    groups.indices = nullptr;
    groups.offsets = nullptr;
}

template <typename T, bool IsCSR>
void bbknn(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances,
    Array<const T> norms_sq
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);
    const Size neighbors_per_cell = n_batches * k;

    BatchGroups batch_groups = {};
    build_batch_groups(batch_labels, n_batches, batch_groups);

    // Pre-allocate workspace for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Size heap_entry_size = n_batches * k;

    // Allocate heap storage for all threads
    using HeapEntry = typename detail::KHeap<T>::Entry;
    HeapEntry* all_heap_storage = scl::memory::aligned_alloc<HeapEntry>(
        n_threads * heap_entry_size, SCL_ALIGNMENT);

    // Allocate heaps array for all threads
    detail::KHeap<T>* all_heaps = scl::memory::aligned_alloc<detail::KHeap<T>>(
        n_threads * n_batches, SCL_ALIGNMENT);

    // Initialize heaps for each thread
    for (size_t t = 0; t < n_threads; ++t) {
        HeapEntry* thread_storage = all_heap_storage + t * heap_entry_size;
        detail::KHeap<T>* thread_heaps = all_heaps + t * n_batches;
        for (Size b = 0; b < n_batches; ++b) {
            thread_heaps[b].init(thread_storage + b * k, k);
        }
    }

    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        detail::KHeap<T>* heaps = all_heaps + thread_rank * n_batches;

        Index query_idx = static_cast<Index>(i);

        auto q_vals_arr = matrix.primary_values(query_idx);
        auto q_inds_arr = matrix.primary_indices(query_idx);

        Size q_len = q_vals_arr.size();
        const T* q_vals = q_vals_arr.data();
        const Index* q_inds = q_inds_arr.data();
        T q_norm_sq = norms_sq[i];

        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].clear();
        }

        for (Size b = 0; b < n_batches; ++b) {
            const Index* group_data = batch_groups.batch_data(b);
            const Size group_size = batch_groups.batch_size(b);
            detail::KHeap<T>& heap = heaps[b];

            for (Size g = 0; g < group_size; ++g) {
                Index cand_idx = group_data[g];
                if (cand_idx == query_idx) continue;

                auto c_vals_arr = matrix.primary_values(cand_idx);
                auto c_inds_arr = matrix.primary_indices(cand_idx);

                Size c_len = c_vals_arr.size();
                const T* c_vals = c_vals_arr.data();
                const Index* c_inds = c_inds_arr.data();
                T c_norm_sq = norms_sq[cand_idx];

                T min_dist_sq = q_norm_sq + c_norm_sq - T(2) * std::sqrt(q_norm_sq * c_norm_sq);
                if (min_dist_sq >= heap.max_dist_sq()) continue;

                T dot = detail::sparse_dot(q_vals, q_inds, q_len, c_vals, c_inds, c_len);
                T dist_sq = q_norm_sq + c_norm_sq - T(2) * dot;
                if (dist_sq < T(0)) dist_sq = T(0);

                heap.try_insert(dist_sq, cand_idx);
            }
        }

        for (Size b = 0; b < n_batches; ++b) {
            Size offset = i * neighbors_per_cell + b * k;
            heaps[b].extract_sorted(
                out_indices.ptr + offset,
                out_distances.ptr + offset
            );
        }
    });

    // Cleanup
    scl::memory::aligned_free(all_heap_storage, SCL_ALIGNMENT);
    scl::memory::aligned_free(all_heaps, SCL_ALIGNMENT);
    free_batch_groups(batch_groups);
}

template <typename T, bool IsCSR>
void bbknn(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    T* norms_sq = scl::memory::aligned_alloc<T>(N, SCL_ALIGNMENT);
    compute_norms(matrix, Array<T>(norms_sq, N));

    bbknn(matrix, batch_labels, n_batches, k, out_indices, out_distances,
          Array<const T>(norms_sq, N));

    scl::memory::aligned_free(norms_sq, SCL_ALIGNMENT);
}

} // namespace scl::kernel::bbknn
