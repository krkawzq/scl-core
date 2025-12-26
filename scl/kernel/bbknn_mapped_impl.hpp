#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// =============================================================================
/// @file bbknn_mapped_impl.hpp
/// @brief BBKNN for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access Pattern
///    - Sequential row access for cache efficiency
///    - Prefetch hints for OS page cache
///
/// 2. Batch-Grouped Processing
///    - Group samples by batch upfront
///    - Minimize comparisons
///
/// 3. Shared Implementation with Fast Path
///    - Reuse KHeap and sparse_dot from bbknn_fast_impl.hpp
///
/// Performance: Near-RAM performance for cached data
// =============================================================================

namespace scl::kernel::bbknn::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size PREFETCH_ROWS = 4;
}

// =============================================================================
// SECTION 2: Heap and Utilities (shared with fast impl)
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

    SCL_FORCE_INLINE void clear() { count = 0; }

    SCL_FORCE_INLINE T max_dist_sq() const {
        return (count > 0) ? data[0].dist_sq : std::numeric_limits<T>::max();
    }

    SCL_FORCE_INLINE void sift_down(Size i) {
        while (true) {
            Size largest = i;
            Size left = 2 * i + 1;
            Size right = 2 * i + 2;

            if (left < count && data[left].dist_sq > data[largest].dist_sq)
                largest = left;
            if (right < count && data[right].dist_sq > data[largest].dist_sq)
                largest = right;

            if (largest == i) break;
            std::swap(data[i], data[largest]);
            i = largest;
        }
    }

    SCL_FORCE_INLINE void sift_up(Size i) {
        while (i > 0) {
            Size parent = (i - 1) / 2;
            if (data[i].dist_sq <= data[parent].dist_sq) break;
            std::swap(data[i], data[parent]);
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
        std::sort(data, data + count, [](const Entry& a, const Entry& b) {
            return a.dist_sq < b.dist_sq;
        });

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

/// @brief SIMD squared norm
template <typename T>
SCL_FORCE_INLINE T compute_norm_sq_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= len; k += 2 * lanes) {
        auto v0 = s::Load(d, vals + k);
        auto v1 = s::Load(d, vals + k + lanes);
        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);

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

/// @brief Sparse-sparse dot product
template <typename T>
SCL_FORCE_INLINE T sparse_dot(
    const T* vals_a, const Index* inds_a, Size len_a,
    const T* vals_b, const Index* inds_b, Size len_b
) {
    T dot = T(0);
    Size i = 0, j = 0;

    while (i < len_a && j < len_b) {
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

} // namespace detail

// =============================================================================
// SECTION 3: Norm Computation
// =============================================================================

/// @brief Compute squared norms (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(primary_dim), "Norms size mismatch");

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        norms_sq[p] = detail::compute_norm_sq_simd(vals.ptr, vals.len);
    });
}

/// @brief Compute squared norms (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_norms_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len >= static_cast<Size>(primary_dim), "Norms size mismatch");

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        norms_sq[p] = detail::compute_norm_sq_simd(vals.ptr, vals.len);
    });
}

// =============================================================================
// SECTION 4: BBKNN Implementation
// =============================================================================

/// @brief BBKNN for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void bbknn_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances,
    Array<const T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);
    const Size neighbors_per_cell = n_batches * k;

    auto batch_groups = detail::build_batch_groups(batch_labels, n_batches);

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    const Size n_chunks = (N + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        std::vector<typename detail::KHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::KHeap<T>> heaps(n_batches);

        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + b * k, k);
        }

        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = std::min(N, i_start + config::CHUNK_SIZE);

        for (Size i = i_start; i < i_end; ++i) {
            Index query_idx = static_cast<Index>(i);

            auto q_vals = scl::primary_values(matrix, query_idx);
            auto q_inds = scl::primary_indices(matrix, query_idx);
            T q_norm_sq = norms_sq[i];

            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].clear();
            }

            for (Size b = 0; b < n_batches; ++b) {
                const auto& group = batch_groups[b];
                auto& heap = heaps[b];

                for (Index cand_idx : group) {
                    if (cand_idx == query_idx) continue;

                    auto c_vals = scl::primary_values(matrix, cand_idx);
                    auto c_inds = scl::primary_indices(matrix, cand_idx);
                    T c_norm_sq = norms_sq[cand_idx];

                    // Early rejection
                    T min_dist_sq = q_norm_sq + c_norm_sq - T(2) * std::sqrt(q_norm_sq * c_norm_sq);
                    if (min_dist_sq >= heap.max_dist_sq()) continue;

                    T dot = detail::sparse_dot(
                        q_vals.ptr, q_inds.ptr, q_vals.len,
                        c_vals.ptr, c_inds.ptr, c_vals.len
                    );

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
        }
    });
}

/// @brief BBKNN for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void bbknn_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<T> out_distances,
    Array<const T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);
    const Size neighbors_per_cell = n_batches * k;

    auto batch_groups = detail::build_batch_groups(batch_labels, n_batches);

    const Size n_chunks = (N + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        std::vector<typename detail::KHeap<T>::Entry> heap_storage(n_batches * k);
        std::vector<detail::KHeap<T>> heaps(n_batches);

        for (Size b = 0; b < n_batches; ++b) {
            heaps[b].init(heap_storage.data() + b * k, k);
        }

        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = std::min(N, i_start + config::CHUNK_SIZE);

        for (Size i = i_start; i < i_end; ++i) {
            Index query_idx = static_cast<Index>(i);

            auto q_vals = scl::primary_values(matrix, query_idx);
            auto q_inds = scl::primary_indices(matrix, query_idx);
            T q_norm_sq = norms_sq[i];

            for (Size b = 0; b < n_batches; ++b) {
                heaps[b].clear();
            }

            for (Size b = 0; b < n_batches; ++b) {
                const auto& group = batch_groups[b];
                auto& heap = heaps[b];

                for (Index cand_idx : group) {
                    if (cand_idx == query_idx) continue;

                    auto c_vals = scl::primary_values(matrix, cand_idx);
                    auto c_inds = scl::primary_indices(matrix, cand_idx);
                    T c_norm_sq = norms_sq[cand_idx];

                    T min_dist_sq = q_norm_sq + c_norm_sq - T(2) * std::sqrt(q_norm_sq * c_norm_sq);
                    if (min_dist_sq >= heap.max_dist_sq()) continue;

                    T dot = detail::sparse_dot(
                        q_vals.ptr, q_inds.ptr, q_vals.len,
                        c_vals.ptr, c_inds.ptr, c_vals.len
                    );

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
        }
    });
}

// =============================================================================
// SECTION 5: Unified Dispatcher
// =============================================================================

/// @brief Mapped BBKNN dispatcher (computes norms internally)
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void bbknn_mapped_dispatch(
    const MatrixT& matrix,
    Array<const int32_t> batch_labels,
    Size n_batches,
    Size k,
    Array<Index> out_indices,
    Array<typename MatrixT::ValueType> out_distances
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size N = static_cast<Size>(primary_dim);

    // Pre-compute norms
    std::vector<T> norms_sq(N);
    compute_norms_mapped(matrix, Array<T>(norms_sq.data(), N));

    // Dispatch to appropriate implementation
    bbknn_mapped(matrix, batch_labels, n_batches, k, out_indices, out_distances,
                 Array<const T>(norms_sq.data(), N));
}

} // namespace scl::kernel::bbknn::mapped
