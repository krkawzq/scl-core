#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <atomic>
#include <algorithm>

// =============================================================================
// FILE: scl/kernel/alignment.hpp
// BRIEF: Multi-modal data alignment and batch integration
//
// APPLICATIONS:
// - MNN-based batch correction
// - Anchor-based integration (Seurat-style)
// - Label transfer across datasets
// - Integration quality assessment
// =============================================================================

namespace scl::kernel::alignment {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_K = 30;
    constexpr Real ANCHOR_SCORE_THRESHOLD = Real(0.5);
    constexpr Size MAX_ANCHORS_PER_CELL = 10;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

namespace detail {

// Compute squared Euclidean distance between sparse vectors
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real sparse_distance_squared(
    const Sparse<T, IsCSR>& data1,
    Index row1,
    const Sparse<T, IsCSR>& data2,
    Index row2
) {
    Real dist = Real(0.0);

    auto row1_vals = data1.row_values_unsafe(row1);
    auto row1_idxs = data1.row_indices_unsafe(row1);
    Index row1_len = data1.row_length_unsafe(row1);
    
    auto row2_vals = data2.row_values_unsafe(row2);
    auto row2_idxs = data2.row_indices_unsafe(row2);
    Index row2_len = data2.row_length_unsafe(row2);

    Index i1 = 0, i2 = 0;
    while (i1 < row1_len && i2 < row2_len) {
        Index col1 = row1_idxs.ptr[i1];
        Index col2 = row2_idxs.ptr[i2];

        if (col1 == col2) {
            Real diff = static_cast<Real>(row1_vals.ptr[i1]) -
                       static_cast<Real>(row2_vals.ptr[i2]);
            dist += diff * diff;
            ++i1;
            ++i2;
        } else if (col1 < col2) {
            Real val = static_cast<Real>(row1_vals.ptr[i1]);
            dist += val * val;
            ++i1;
        } else {
            Real val = static_cast<Real>(row2_vals.ptr[i2]);
            dist += val * val;
            ++i2;
        }
    }
    while (i1 < row1_len) {
        Real val = static_cast<Real>(row1_vals.ptr[i1++]);
        dist += val * val;
    }
    while (i2 < row2_len) {
        Real val = static_cast<Real>(row2_vals.ptr[i2++]);
        dist += val * val;
    }

    return dist;
}

// Find k nearest neighbors from data2 for each row in data1
template <typename T, bool IsCSR>
void find_cross_knn(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    Index* knn_indices,  // [n1 * k]
    Real* knn_distances   // [n1 * k]
) {
    const Size n1 = static_cast<Size>(data1.rows());
    const Size n2 = static_cast<Size>(data2.rows());

    k = scl::algo::min2(k, static_cast<Index>(n2));

    // Create workspace pools for thread-local buffers
    scl::threading::WorkspacePool<Real> dists_pool;
    scl::threading::WorkspacePool<Index> indices_pool;
    dists_pool.init(scl::threading::Scheduler::get_num_threads(), n2);
    indices_pool.init(scl::threading::Scheduler::get_num_threads(), n2);

    // Parallel over points in data1
    scl::threading::parallel_for(Size(0), n1, [&](size_t i, size_t thread_rank) {
        Real* all_dists = dists_pool.get(thread_rank);
        Index* all_indices = indices_pool.get(thread_rank);

        // Compute distances to all points in data2
        for (Size j = 0; j < n2; ++j) {
            all_dists[j] = sparse_distance_squared(data1, static_cast<Index>(i),
                                                   data2, static_cast<Index>(j));
            all_indices[j] = static_cast<Index>(j);
        }

        // Partial sort indices by distance
        // Initialize indices
        for (Size j = 0; j < n2; ++j) {
            all_indices[j] = static_cast<Index>(j);
        }
        // Sort indices by distance (k smallest)
        std::partial_sort(all_indices, all_indices + k, all_indices + n2,
            [&](Index a, Index b) { return all_dists[a] < all_dists[b]; });

        // Store k nearest neighbors
        for (Index ki = 0; ki < k; ++ki) {
            knn_indices[i * k + ki] = all_indices[ki];
            knn_distances[i * k + ki] = all_dists[ki];
        }
    });
}

// Check if two cells are mutual nearest neighbors
SCL_FORCE_INLINE bool is_mnn(
    Index cell1,
    Index cell2,
    const Index* knn1_to_2,  // k nearest of data1 in data2
    const Index* knn2_to_1,  // k nearest of data2 in data1
    Index k
) {
    // Check if cell2 is in cell1's k nearest neighbors
    bool in_knn1 = false;
    for (Index ki = 0; ki < k; ++ki) {
        if (knn1_to_2[cell1 * k + ki] == cell2) {
            in_knn1 = true;
            break;
        }
    }
    if (!in_knn1) return false;

    // Check if cell1 is in cell2's k nearest neighbors
    for (Index ki = 0; ki < k; ++ki) {
        if (knn2_to_1[cell2 * k + ki] == cell1) {
            return true;
        }
    }
    return false;
}

} // namespace detail

// =============================================================================
// Mutual Nearest Neighbors (MNN) Pairs
// =============================================================================

template <typename T, bool IsCSR>
void mnn_pairs(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    Index* mnn_cell1,
    Index* mnn_cell2,
    Size& n_pairs
) {
    const Size n1 = static_cast<Size>(data1.rows());
    const Size n2 = static_cast<Size>(data2.rows());

    n_pairs = 0;
    if (n1 == 0 || n2 == 0 || k == 0) return;

    k = scl::algo::min2(k, static_cast<Index>(scl::algo::min2(n1, n2)));

    // Find k nearest neighbors in both directions
    Index* knn1_to_2 = scl::memory::aligned_alloc<Index>(n1 * k, SCL_ALIGNMENT);
    Real* dist1_to_2 = scl::memory::aligned_alloc<Real>(n1 * k, SCL_ALIGNMENT);
    Index* knn2_to_1 = scl::memory::aligned_alloc<Index>(n2 * k, SCL_ALIGNMENT);
    Real* dist2_to_1 = scl::memory::aligned_alloc<Real>(n2 * k, SCL_ALIGNMENT);

    detail::find_cross_knn(data1, data2, k, knn1_to_2, dist1_to_2);
    detail::find_cross_knn(data2, data1, k, knn2_to_1, dist2_to_1);

    // Find mutual nearest neighbors
    for (Size i = 0; i < n1; ++i) {
        for (Index ki = 0; ki < k; ++ki) {
            Index j = knn1_to_2[i * k + ki];

            // Check if this is a mutual pair
            if (detail::is_mnn(static_cast<Index>(i), j, knn1_to_2, knn2_to_1, k)) {
                mnn_cell1[n_pairs] = static_cast<Index>(i);
                mnn_cell2[n_pairs] = j;
                ++n_pairs;
            }
        }
    }

    scl::memory::aligned_free(knn1_to_2);
    scl::memory::aligned_free(dist1_to_2);
    scl::memory::aligned_free(knn2_to_1);
    scl::memory::aligned_free(dist2_to_1);
}

// =============================================================================
// Anchor Finding (Seurat-style)
// =============================================================================

template <typename T, bool IsCSR>
void find_anchors(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    Index* anchor_cell1,
    Index* anchor_cell2,
    Real* anchor_scores,
    Size& n_anchors
) {
    const Size n1 = static_cast<Size>(data1.rows());
    const Size n2 = static_cast<Size>(data2.rows());

    n_anchors = 0;
    if (n1 == 0 || n2 == 0 || k == 0) return;

    k = scl::algo::min2(k, static_cast<Index>(scl::algo::min2(n1, n2)));

    // Find k nearest neighbors in both directions
    Index* knn1_to_2 = scl::memory::aligned_alloc<Index>(n1 * k, SCL_ALIGNMENT);
    Real* dist1_to_2 = scl::memory::aligned_alloc<Real>(n1 * k, SCL_ALIGNMENT);
    Index* knn2_to_1 = scl::memory::aligned_alloc<Index>(n2 * k, SCL_ALIGNMENT);
    Real* dist2_to_1 = scl::memory::aligned_alloc<Real>(n2 * k, SCL_ALIGNMENT);

    detail::find_cross_knn(data1, data2, k, knn1_to_2, dist1_to_2);
    detail::find_cross_knn(data2, data1, k, knn2_to_1, dist2_to_1);

    // Compute anchor scores based on shared neighbor overlap
    for (Size i = 0; i < n1; ++i) {
        for (Index ki = 0; ki < k; ++ki) {
            Index j = knn1_to_2[i * k + ki];

            // Compute anchor score: Jaccard-like overlap of neighborhoods
            Size shared = 0;
            for (Index ki1 = 0; ki1 < k; ++ki1) {
                Index neighbor_of_i = knn1_to_2[i * k + ki1];
                for (Index ki2 = 0; ki2 < k; ++ki2) {
                    // Check if neighbor of i (in data2) is also neighbor of j (in data2)
                    // This requires finding j's neighbors in data2 - use reverse mapping
                    if (knn2_to_1[neighbor_of_i * k + ki2] == static_cast<Index>(i)) {
                        ++shared;
                        break;
                    }
                }
            }

            Real score = static_cast<Real>(shared) / static_cast<Real>(2 * k - shared);

            if (score >= config::ANCHOR_SCORE_THRESHOLD) {
                anchor_cell1[n_anchors] = static_cast<Index>(i);
                anchor_cell2[n_anchors] = j;
                anchor_scores[n_anchors] = score;
                ++n_anchors;
            }
        }
    }

    scl::memory::aligned_free(knn1_to_2);
    scl::memory::aligned_free(dist1_to_2);
    scl::memory::aligned_free(knn2_to_1);
    scl::memory::aligned_free(dist2_to_1);
}

// =============================================================================
// Label Transfer via Anchors
// =============================================================================

void transfer_labels(
    const Index* anchor_cell1,
    const Index* anchor_cell2,
    const Real* anchor_weights,
    Size n_anchors,
    Array<const Index> source_labels,
    Size n_target,
    Array<Index> target_labels,
    Array<Real> transfer_confidence
) {
    SCL_CHECK_DIM(target_labels.len == n_target, "Target labels length mismatch");
    SCL_CHECK_DIM(transfer_confidence.len == n_target, "Confidence length mismatch");

    if (n_anchors == 0 || n_target == 0) return;

    // Find number of unique labels
    Index n_labels = 0;
    for (Size i = 0; i < source_labels.len; ++i) {
        n_labels = scl::algo::max2(n_labels, source_labels.ptr[i] + 1);
    }

    // Create workspace pool for thread-local label scores
    scl::threading::WorkspacePool<Real> pool;
    pool.init(scl::threading::Scheduler::get_num_threads(), static_cast<Size>(n_labels));

    // Parallel over target cells
    scl::threading::parallel_for(Size(0), n_target, [&](size_t t, size_t thread_rank) {
        Real* label_scores = pool.get(thread_rank);

        // Reset scores
        scl::algo::zero(label_scores, static_cast<Size>(n_labels));

        Real total_weight = Real(0.0);

        // Find anchors involving this target cell
        for (Size a = 0; a < n_anchors; ++a) {
            if (anchor_cell2[a] == static_cast<Index>(t)) {
                Index source_cell = anchor_cell1[a];
                Index label = source_labels.ptr[source_cell];
                Real weight = anchor_weights[a];

                label_scores[label] += weight;
                total_weight += weight;
            }
        }

        if (total_weight < config::EPSILON) {
            target_labels.ptr[t] = 0;
            transfer_confidence.ptr[t] = Real(0.0);
        } else {
            // Find label with highest score
            Index best_label = 0;
            Real best_score = label_scores[0];
            for (Index l = 1; l < n_labels; ++l) {
                if (label_scores[l] > best_score) {
                    best_score = label_scores[l];
                    best_label = l;
                }
            }

            target_labels.ptr[t] = best_label;
            transfer_confidence.ptr[t] = best_score / total_weight;
        }
    });
}

// =============================================================================
// Integration Quality Score
// =============================================================================

template <typename T, bool IsCSR>
Real integration_score(
    const Sparse<T, IsCSR>& integrated_data,
    Array<const Index> batch_labels,
    const Sparse<Index, IsCSR>& neighbors
) {
    const Size n_cells = static_cast<Size>(integrated_data.rows());
    SCL_CHECK_DIM(n_cells == batch_labels.len, "Batch labels length mismatch");
    SCL_CHECK_DIM(n_cells == static_cast<Size>(neighbors.rows()), "Neighbors rows mismatch");

    if (n_cells == 0) return Real(1.0);

    // Find number of batches
    Index n_batches = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_batches = scl::algo::max2(n_batches, batch_labels.ptr[i] + 1);
    }

    if (n_batches <= 1) return Real(1.0);

    // Create workspace pool for thread-local batch counts
    scl::threading::WorkspacePool<Size> pool;
    pool.init(scl::threading::Scheduler::get_num_threads(), static_cast<Size>(n_batches));

    // Compute entropy per cell in parallel
    Real* cell_entropies = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real max_entropy = std::log(static_cast<Real>(n_batches));

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
        Size* batch_counts = pool.get(thread_rank);
        scl::algo::zero(batch_counts, static_cast<Size>(n_batches));

        // Count batches in neighborhood
        auto row_idxs = neighbors.row_indices_unsafe(i);
        Index row_len = neighbors.row_length_unsafe(i);
        Size n_neighbors = static_cast<Size>(row_len);

        for (Index j = 0; j < row_len; ++j) {
            Index neighbor = row_idxs.ptr[j];
            ++batch_counts[batch_labels.ptr[neighbor]];
        }

        // Include self
        ++batch_counts[batch_labels.ptr[i]];
        ++n_neighbors;

        // Compute entropy
        Real entropy = Real(0.0);
        for (Index b = 0; b < n_batches; ++b) {
            if (batch_counts[b] > 0) {
                Real p = static_cast<Real>(batch_counts[b]) / static_cast<Real>(n_neighbors);
                entropy -= p * std::log(p + config::EPSILON);
            }
        }

        cell_entropies[i] = entropy / max_entropy;
    });

    // Sum all entropies
    Real total_entropy = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        total_entropy += cell_entropies[i];
    }

    scl::memory::aligned_free(cell_entropies);

    return total_entropy / static_cast<Real>(n_cells);
}

// =============================================================================
// Batch Mixing Metric
// =============================================================================

template <bool IsCSR>
void batch_mixing(
    Array<const Index> batch_labels,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> mixing_scores
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    SCL_CHECK_DIM(n_cells == batch_labels.len, "Batch labels length mismatch");
    SCL_CHECK_DIM(n_cells == mixing_scores.len, "Mixing scores length mismatch");

    if (n_cells == 0) return;

    // Find number of batches
    Index n_batches = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_batches = scl::algo::max2(n_batches, batch_labels.ptr[i] + 1);
    }

    // Create workspace pool for thread-local batch counts
    scl::threading::WorkspacePool<Size> pool;
    pool.init(scl::threading::Scheduler::get_num_threads(), static_cast<Size>(n_batches));

    // Parallel over cells
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
        Size* batch_counts = pool.get(thread_rank);
        scl::algo::zero(batch_counts, static_cast<Size>(n_batches));

        // Count batches in neighborhood
        auto row_idxs = neighbors.row_indices_unsafe(i);
        Index row_len = neighbors.row_length_unsafe(i);
        Size n_neighbors = static_cast<Size>(row_len);

        Index my_batch = batch_labels.ptr[i];

        for (Index j = 0; j < row_len; ++j) {
            Index neighbor = row_idxs.ptr[j];
            ++batch_counts[batch_labels.ptr[neighbor]];
        }

        // Mixing score: fraction of neighbors from different batches
        Size same_batch = batch_counts[my_batch];
        Size diff_batch = n_neighbors - same_batch;

        if (n_neighbors > 0) {
            mixing_scores.ptr[i] = static_cast<Real>(diff_batch) / static_cast<Real>(n_neighbors);
        } else {
            mixing_scores.ptr[i] = Real(0.0);
        }
    });
}

// =============================================================================
// Compute Correction Vectors (MNN-based)
// =============================================================================

template <typename T, bool IsCSR>
void compute_correction_vectors(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    const Index* mnn_cell1,
    const Index* mnn_cell2,
    Size n_pairs,
    Real* correction_vectors,  // [n2 * n_features]
    Size n_features
) {
    const Size n2 = static_cast<Size>(data2.rows());

    if (n_pairs == 0 || n2 == 0) return;

    // Initialize correction vectors to zero
    for (Size i = 0; i < n2 * n_features; ++i) {
        correction_vectors[i] = Real(0.0);
    }

    // Count pairs per cell in data2
    Size* pair_count = scl::memory::aligned_alloc<Size>(n2, SCL_ALIGNMENT);
    for (Size i = 0; i < n2; ++i) {
        pair_count[i] = 0;
    }

    for (Size p = 0; p < n_pairs; ++p) {
        Index cell2 = mnn_cell2[p];
        ++pair_count[cell2];
    }

    // Accumulate correction vectors
    Real* cell1_dense = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
    Real* cell2_dense = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);

    for (Size p = 0; p < n_pairs; ++p) {
        Index cell1 = mnn_cell1[p];
        Index cell2 = mnn_cell2[p];

        // Convert to dense representation
        for (Size f = 0; f < n_features; ++f) {
            cell1_dense[f] = Real(0.0);
            cell2_dense[f] = Real(0.0);
        }

        auto row1_vals = data1.row_values_unsafe(cell1);
        auto row1_idxs = data1.row_indices_unsafe(cell1);
        Index row1_len = data1.row_length_unsafe(cell1);
        for (Index j = 0; j < row1_len; ++j) {
            Index col = row1_idxs.ptr[j];
            cell1_dense[col] = static_cast<Real>(row1_vals.ptr[j]);
        }

        auto row2_vals = data2.row_values_unsafe(cell2);
        auto row2_idxs = data2.row_indices_unsafe(cell2);
        Index row2_len = data2.row_length_unsafe(cell2);
        for (Index j = 0; j < row2_len; ++j) {
            Index col = row2_idxs.ptr[j];
            cell2_dense[col] = static_cast<Real>(row2_vals.ptr[j]);
        }

        // Add correction vector (cell1 - cell2)
        for (Size f = 0; f < n_features; ++f) {
            correction_vectors[cell2 * n_features + f] += cell1_dense[f] - cell2_dense[f];
        }
    }

    // Average correction vectors
    for (Size i = 0; i < n2; ++i) {
        if (pair_count[i] > 0) {
            Real inv_count = Real(1.0) / static_cast<Real>(pair_count[i]);
            for (Size f = 0; f < n_features; ++f) {
                correction_vectors[i * n_features + f] *= inv_count;
            }
        }
    }

    scl::memory::aligned_free(pair_count);
    scl::memory::aligned_free(cell1_dense);
    scl::memory::aligned_free(cell2_dense);
}

// =============================================================================
// Smooth Correction Vectors using Gaussian Kernel
// =============================================================================

template <typename T, bool IsCSR>
void smooth_correction_vectors(
    const Sparse<T, IsCSR>& data2,
    Real* correction_vectors,  // [n2 * n_features]
    Size n_features,
    Real sigma
) {
    const Size n2 = static_cast<Size>(data2.rows());

    if (n2 == 0) return;

    // Create a copy for smoothing
    Real* smoothed = scl::memory::aligned_alloc<Real>(n2 * n_features, SCL_ALIGNMENT);
    Real* weights = scl::memory::aligned_alloc<Real>(n2, SCL_ALIGNMENT);

    Real sigma_sq = sigma * sigma;

    for (Size i = 0; i < n2; ++i) {
        // Initialize
        for (Size f = 0; f < n_features; ++f) {
            smoothed[i * n_features + f] = Real(0.0);
        }
        Real total_weight = Real(0.0);

        // Compute weighted average of correction vectors
        for (Size j = 0; j < n2; ++j) {
            Real dist_sq = detail::sparse_distance_squared(
                data2, static_cast<Index>(i),
                data2, static_cast<Index>(j));

            Real w = std::exp(-dist_sq / (Real(2.0) * sigma_sq));
            weights[j] = w;
            total_weight += w;
        }

        // Accumulate weighted correction
        if (total_weight > config::EPSILON) {
            for (Size j = 0; j < n2; ++j) {
                Real w = weights[j] / total_weight;
                for (Size f = 0; f < n_features; ++f) {
                    smoothed[i * n_features + f] += w * correction_vectors[j * n_features + f];
                }
            }
        }
    }

    // Copy back
    for (Size i = 0; i < n2 * n_features; ++i) {
        correction_vectors[i] = smoothed[i];
    }

    scl::memory::aligned_free(smoothed);
    scl::memory::aligned_free(weights);
}

// =============================================================================
// Canonical Correlation Analysis (CCA) for multimodal alignment
// =============================================================================

template <typename T, bool IsCSR>
void cca_projection(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Size n_components,
    Real* projection1,  // [n1 * n_components]
    Real* projection2   // [n2 * n_components]
) {
    const Size n1 = static_cast<Size>(data1.rows());
    const Size n2 = static_cast<Size>(data2.rows());
    const Size d1 = static_cast<Size>(data1.cols());
    const Size d2 = static_cast<Size>(data2.cols());

    if (n1 == 0 || n2 == 0 || n_components == 0) return;

    n_components = scl::algo::min2(n_components, scl::algo::min2(d1, d2));

    // Simplified CCA: project onto shared PCA space
    // For simplicity, use random projection as approximation

    Real* proj_matrix1 = scl::memory::aligned_alloc<Real>(d1 * n_components, SCL_ALIGNMENT);
    Real* proj_matrix2 = scl::memory::aligned_alloc<Real>(d2 * n_components, SCL_ALIGNMENT);

    // Random orthogonal projection (simplified)
    uint64_t seed = 42;
    for (Size i = 0; i < d1 * n_components; ++i) {
        seed = seed * 6364136223846793005ULL + 1;
        proj_matrix1[i] = (static_cast<Real>(seed >> 33) / Real(2147483648.0)) - Real(0.5);
    }
    for (Size i = 0; i < d2 * n_components; ++i) {
        seed = seed * 6364136223846793005ULL + 1;
        proj_matrix2[i] = (static_cast<Real>(seed >> 33) / Real(2147483648.0)) - Real(0.5);
    }

    // Project data1 in parallel
    scl::threading::parallel_for(Size(0), n1, [&](size_t i) {
        for (Size c = 0; c < n_components; ++c) {
            projection1[i * n_components + c] = Real(0.0);
        }

        auto row_vals = data1.row_values_unsafe(i);
        auto row_idxs = data1.row_indices_unsafe(i);
        Index row_len = data1.row_length_unsafe(i);

        for (Index j = 0; j < row_len; ++j) {
            Index col = row_idxs.ptr[j];
            Real val = static_cast<Real>(row_vals.ptr[j]);
            for (Size c = 0; c < n_components; ++c) {
                projection1[i * n_components + c] += val * proj_matrix1[col * n_components + c];
            }
        }
    });

    // Project data2 in parallel
    scl::threading::parallel_for(Size(0), n2, [&](size_t i) {
        for (Size c = 0; c < n_components; ++c) {
            projection2[i * n_components + c] = Real(0.0);
        }

        auto row_vals = data2.row_values_unsafe(i);
        auto row_idxs = data2.row_indices_unsafe(i);
        Index row_len = data2.row_length_unsafe(i);

        for (Index j = 0; j < row_len; ++j) {
            Index col = row_idxs.ptr[j];
            Real val = static_cast<Real>(row_vals.ptr[j]);
            for (Size c = 0; c < n_components; ++c) {
                projection2[i * n_components + c] += val * proj_matrix2[col * n_components + c];
            }
        }
    });

    scl::memory::aligned_free(proj_matrix1);
    scl::memory::aligned_free(proj_matrix2);
}

// =============================================================================
// kBET (k-nearest neighbor Batch Effect Test)
// =============================================================================

template <bool IsCSR>
Real kbet_score(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> batch_labels
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    SCL_CHECK_DIM(n_cells == batch_labels.len, "Batch labels length mismatch");

    if (n_cells == 0) return Real(1.0);

    // Find number of batches and their proportions
    Index n_batches = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_batches = scl::algo::max2(n_batches, batch_labels.ptr[i] + 1);
    }

    Size* batch_counts_global = scl::memory::aligned_alloc<Size>(n_batches, SCL_ALIGNMENT);
    scl::algo::zero(batch_counts_global, static_cast<Size>(n_batches));
    for (Size i = 0; i < n_cells; ++i) {
        ++batch_counts_global[batch_labels.ptr[i]];
    }

    // Expected proportions
    Real* expected_props = scl::memory::aligned_alloc<Real>(n_batches, SCL_ALIGNMENT);
    for (Index b = 0; b < n_batches; ++b) {
        expected_props[b] = static_cast<Real>(batch_counts_global[b]) / static_cast<Real>(n_cells);
    }

    Real threshold = static_cast<Real>(n_batches - 1) + Real(2.0) * std::sqrt(static_cast<Real>(2 * (n_batches - 1)));

    // Create workspace pool for thread-local batch counts
    scl::threading::WorkspacePool<Size> pool;
    pool.init(scl::threading::Scheduler::get_num_threads(), static_cast<Size>(n_batches));

    // Count rejections per cell in parallel
    Index* rejection_flags = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
        Size* batch_counts_local = pool.get(thread_rank);
        scl::algo::zero(batch_counts_local, static_cast<Size>(n_batches));

        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        Size k = static_cast<Size>(row_end - row_start);

        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor = neighbors.col_indices_unsafe()[j];
            ++batch_counts_local[batch_labels.ptr[neighbor]];
        }

        // Chi-squared test
        Real chi2 = Real(0.0);
        for (Index b = 0; b < n_batches; ++b) {
            Real observed = static_cast<Real>(batch_counts_local[b]);
            Real expected = expected_props[b] * static_cast<Real>(k);
            if (expected > config::EPSILON) {
                Real diff = observed - expected;
                chi2 += diff * diff / expected;
            }
        }

        rejection_flags[i] = (chi2 > threshold) ? 1 : 0;
    });

    // Sum rejection flags
    Size n_rejected = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_rejected += rejection_flags[i];
    }

    scl::memory::aligned_free(batch_counts_global);
    scl::memory::aligned_free(expected_props);
    scl::memory::aligned_free(rejection_flags);

    // Acceptance rate (1 - rejection rate)
    return Real(1.0) - static_cast<Real>(n_rejected) / static_cast<Real>(n_cells);
}

} // namespace scl::kernel::alignment
