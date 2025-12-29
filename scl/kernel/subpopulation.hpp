#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/subpopulation.hpp
// BRIEF: Subpopulation analysis and cluster refinement
//
// APPLICATIONS:
// - Recursive sub-clustering
// - Cluster stability assessment
// - Rare cell detection
// - Population balance analysis
// =============================================================================

namespace scl::kernel::subpopulation {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CLUSTER_SIZE = 10;
    constexpr Size DEFAULT_K = 5;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Size DEFAULT_BOOTSTRAP = 100;
}

namespace detail {

// Simple LCG random number generator
struct LCG {
    uint64_t state;

    explicit LCG(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }

    Real next_real() {
        return static_cast<Real>(next() >> 11) * Real(1.1102230246251565e-16);
    }

    Size next_index(Size max_val) {
        return static_cast<Size>(next() % max_val);
    }
};

// K-means clustering for subclustering
template <typename T, bool IsCSR>
void kmeans_cluster(
    const Sparse<T, IsCSR>& data,
    const Index* cell_indices,
    Size n_cells,
    Size k,
    Index* assignments,
    LCG& rng
) {
    if (n_cells == 0 || k == 0) return;

    k = scl::algo::min2(k, n_cells);
    const Size n_features = static_cast<Size>(data.cols());

    // Allocate centroids
    auto centroids_ptr = scl::memory::aligned_alloc<Real>(k * n_features, SCL_ALIGNMENT);

    Real* centroids = centroids_ptr.release();
    auto new_centroids_ptr = scl::memory::aligned_alloc<Real>(k * n_features, SCL_ALIGNMENT);

    Real* new_centroids = new_centroids_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(k, SCL_ALIGNMENT);

    Size* cluster_sizes = cluster_sizes_ptr.release();

    // Initialize centroids with random cells
    for (Size c = 0; c < k; ++c) {
        Size idx = rng.next_index(n_cells);
        Index cell = cell_indices[idx];

        for (Size f = 0; f < n_features; ++f) {
            centroids[c * n_features + f] = Real(0.0);
        }

        const Index row_start = data.row_indices_unsafe()[cell];
        const Index row_end = data.row_indices_unsafe()[cell + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = data.col_indices_unsafe()[j];
            centroids[c * n_features + feature] = static_cast<Real>(data.values()[j]);
        }
    }

    // K-means iterations
    for (Size iter = 0; iter < config::MAX_ITERATIONS; ++iter) {
        // Assign cells to nearest centroid
        bool changed = false;

        for (Size i = 0; i < n_cells; ++i) {
            Index cell = cell_indices[i];
            Real min_dist = std::numeric_limits<Real>::max();
            Index best_cluster = 0;

            for (Size c = 0; c < k; ++c) {
                Real dist = Real(0.0);

                const Index row_start = data.row_indices_unsafe()[cell];
                const Index row_end = data.row_indices_unsafe()[cell + 1];

                // Compute squared distance to centroid
                for (Size f = 0; f < n_features; ++f) {
                    Real val = Real(0.0);
                    for (Index j = row_start; j < row_end; ++j) {
                        if (data.col_indices_unsafe()[j] == static_cast<Index>(f)) {
                            val = static_cast<Real>(data.values()[j]);
                            break;
                        }
                    }
                    Real diff = val - centroids[c * n_features + f];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<Index>(c);
                }
            }

            if (assignments[i] != best_cluster) {
                changed = true;
                assignments[i] = best_cluster;
            }
        }

        if (!changed) break;

        // Update centroids
        for (Size c = 0; c < k; ++c) {
            cluster_sizes[c] = 0;
            for (Size f = 0; f < n_features; ++f) {
                new_centroids[c * n_features + f] = Real(0.0);
            }
        }

        for (Size i = 0; i < n_cells; ++i) {
            Index cell = cell_indices[i];
            Index cluster = assignments[i];
            ++cluster_sizes[cluster];

            const Index row_start = data.row_indices_unsafe()[cell];
            const Index row_end = data.row_indices_unsafe()[cell + 1];

            for (Index j = row_start; j < row_end; ++j) {
                Index feature = data.col_indices_unsafe()[j];
                new_centroids[cluster * n_features + feature] +=
                    static_cast<Real>(data.values()[j]);
            }
        }

        for (Size c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (Size f = 0; f < n_features; ++f) {
                    centroids[c * n_features + f] =
                        new_centroids[c * n_features + f] / static_cast<Real>(cluster_sizes[c]);
                }
            }
        }
    }

    scl::memory::aligned_free(centroids);
    scl::memory::aligned_free(new_centroids);
    scl::memory::aligned_free(cluster_sizes);
}

// Compute sparse distance squared
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real sparse_distance_squared(
    const Sparse<T, IsCSR>& data,
    Index row1,
    Index row2
) {
    Real dist = Real(0.0);

    const Index start1 = data.row_indices_unsafe()[row1];
    const Index end1 = data.row_indices_unsafe()[row1 + 1];
    const Index start2 = data.row_indices_unsafe()[row2];
    const Index end2 = data.row_indices_unsafe()[row2 + 1];

    Index i1 = start1, i2 = start2;
    while (i1 < end1 && i2 < end2) {
        Index col1 = data.col_indices_unsafe()[i1];
        Index col2 = data.col_indices_unsafe()[i2];

        if (col1 == col2) {
            Real diff = static_cast<Real>(data.values()[i1]) -
                       static_cast<Real>(data.values()[i2]);
            dist += diff * diff;
            ++i1; ++i2;
        } else if (col1 < col2) {
            Real val = static_cast<Real>(data.values()[i1]);
            dist += val * val;
            ++i1;
        } else {
            Real val = static_cast<Real>(data.values()[i2]);
            dist += val * val;
            ++i2;
        }
    }
    while (i1 < end1) {
        Real val = static_cast<Real>(data.values()[i1++]);
        dist += val * val;
    }
    while (i2 < end2) {
        Real val = static_cast<Real>(data.values()[i2++]);
        dist += val * val;
    }

    return dist;
}

} // namespace detail

// =============================================================================
// Recursive Sub-clustering
// =============================================================================

template <typename T, bool IsCSR>
void subclustering(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> parent_labels,
    Index parent_cluster,
    Size n_subclusters,
    Array<Index> subcluster_labels,
    uint64_t seed
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == parent_labels.len, "Parent labels must match cell count");
    SCL_CHECK_DIM(n_cells == subcluster_labels.len, "Subcluster labels must match cell count");

    if (n_cells == 0) return;

    // Count cells in parent cluster
    Size n_in_cluster = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (parent_labels.ptr[i] == parent_cluster) {
            ++n_in_cluster;
        }
    }

    if (n_in_cluster < config::MIN_CLUSTER_SIZE || n_subclusters == 0) {
        // Not enough cells to subcluster
        for (Size i = 0; i < n_cells; ++i) {
            if (parent_labels.ptr[i] == parent_cluster) {
                subcluster_labels.ptr[i] = 0;
            } else {
                subcluster_labels.ptr[i] = -1;
            }
        }
        return;
    }

    // Collect cell indices in parent cluster
    auto cell_indices_ptr = scl::memory::aligned_alloc<Index>(n_in_cluster, SCL_ALIGNMENT);

    Index* cell_indices = cell_indices_ptr.release();
    Size idx = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (parent_labels.ptr[i] == parent_cluster) {
            cell_indices[idx++] = static_cast<Index>(i);
        }
    }

    // Run k-means on subset
    auto assignments_ptr = scl::memory::aligned_alloc<Index>(n_in_cluster, SCL_ALIGNMENT);

    Index* assignments = assignments_ptr.release();
    for (Size i = 0; i < n_in_cluster; ++i) {
        assignments[i] = 0;
    }

    detail::LCG rng(seed);
    detail::kmeans_cluster(expression, cell_indices, n_in_cluster, n_subclusters, assignments, rng);

    // Map back to original indices
    for (Size i = 0; i < n_cells; ++i) {
        subcluster_labels.ptr[i] = -1;
    }

    for (Size i = 0; i < n_in_cluster; ++i) {
        Index cell = cell_indices[i];
        subcluster_labels.ptr[cell] = assignments[i];
    }

    scl::memory::aligned_free(cell_indices);
    scl::memory::aligned_free(assignments);
}

// =============================================================================
// Cluster Stability via Bootstrap
// =============================================================================

template <typename T, bool IsCSR>
void cluster_stability(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> original_labels,
    Size n_bootstraps,
    Array<Real> stability_scores,
    uint64_t seed
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == original_labels.len, "Labels must match cell count");

    if (n_cells == 0 || n_bootstraps == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (original_labels.ptr[i] >= n_clusters) {
            n_clusters = original_labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(stability_scores.len >= static_cast<Size>(n_clusters),
        "Stability scores array too small");

    // Initialize stability scores
    for (Index c = 0; c < n_clusters; ++c) {
        stability_scores.ptr[c] = Real(0.0);
    }

    detail::LCG rng(seed);

    // Co-occurrence matrix for each cluster
    auto cooccur_same_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cooccur_same = cooccur_same_ptr.release();
    auto cooccur_total_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cooccur_total = cooccur_total_ptr.release();

    for (Index c = 0; c < n_clusters; ++c) {
        cooccur_same[c] = 0;
        cooccur_total[c] = 0;
    }

    // Bootstrap iterations
    auto sample_indices_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    Index* sample_indices = sample_indices_ptr.release();
    auto bootstrap_labels_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    Index* bootstrap_labels = bootstrap_labels_ptr.release();

    for (Size b = 0; b < n_bootstraps; ++b) {
        // Sample with replacement
        for (Size i = 0; i < n_cells; ++i) {
            sample_indices[i] = static_cast<Index>(rng.next_index(n_cells));
        }

        // Run k-means clustering on bootstrap sample
        for (Size i = 0; i < n_cells; ++i) {
            bootstrap_labels[i] = 0;
        }
        detail::kmeans_cluster(expression, sample_indices, n_cells,
            static_cast<Size>(n_clusters), bootstrap_labels, rng);

        // Count co-occurrences
        for (Size i = 0; i < n_cells; ++i) {
            for (Size j = i + 1; j < n_cells; ++j) {
                Index orig_i = original_labels.ptr[sample_indices[i]];
                Index orig_j = original_labels.ptr[sample_indices[j]];

                if (orig_i == orig_j) {
                    ++cooccur_total[orig_i];
                    if (bootstrap_labels[i] == bootstrap_labels[j]) {
                        ++cooccur_same[orig_i];
                    }
                }
            }
        }
    }

    // Compute stability as co-occurrence rate
    for (Index c = 0; c < n_clusters; ++c) {
        if (cooccur_total[c] > 0) {
            stability_scores.ptr[c] = static_cast<Real>(cooccur_same[c]) /
                                      static_cast<Real>(cooccur_total[c]);
        } else {
            stability_scores.ptr[c] = Real(1.0);
        }
    }

    scl::memory::aligned_free(cooccur_same);
    scl::memory::aligned_free(cooccur_total);
    scl::memory::aligned_free(sample_indices);
    scl::memory::aligned_free(bootstrap_labels);
}

// =============================================================================
// Cluster Purity
// =============================================================================

void cluster_purity(
    Array<const Index> cluster_labels,
    Array<const Index> true_labels,
    Array<Real> purity_per_cluster
) {
    const Size n = cluster_labels.len;
    SCL_CHECK_DIM(n == true_labels.len, "Labels must have same length");

    if (n == 0) return;

    // Find number of clusters and true classes
    Index n_clusters = 0;
    Index n_classes = 0;
    for (Size i = 0; i < n; ++i) {
        if (cluster_labels.ptr[i] >= n_clusters) {
            n_clusters = cluster_labels.ptr[i] + 1;
        }
        if (true_labels.ptr[i] >= n_classes) {
            n_classes = true_labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(purity_per_cluster.len >= static_cast<Size>(n_clusters),
        "Purity array too small");

    // Count cells per cluster per class
    auto counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters * n_classes, SCL_ALIGNMENT);

    Size* counts = counts_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cluster_sizes = cluster_sizes_ptr.release();

    for (Size i = 0; i < static_cast<Size>(n_clusters * n_classes); ++i) counts[i] = 0;
    for (Index c = 0; c < n_clusters; ++c) cluster_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index cluster = cluster_labels.ptr[i];
        Index cls = true_labels.ptr[i];
        ++counts[cluster * n_classes + cls];
        ++cluster_sizes[cluster];
    }

    // Compute purity for each cluster
    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] == 0) {
            purity_per_cluster.ptr[c] = Real(1.0);
            continue;
        }

        // Find majority class
        Size max_count = 0;
        for (Index cls = 0; cls < n_classes; ++cls) {
            if (counts[c * n_classes + cls] > max_count) {
                max_count = counts[c * n_classes + cls];
            }
        }

        purity_per_cluster.ptr[c] = static_cast<Real>(max_count) /
                                    static_cast<Real>(cluster_sizes[c]);
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(cluster_sizes);
}

// =============================================================================
// Rare Cell Detection
// =============================================================================

template <typename T, bool IsCSR>
void rare_cell_detection(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> rarity_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == rarity_scores.len, "Rarity scores must match cell count");
    SCL_CHECK_DIM(n_cells == static_cast<Size>(neighbors.rows()), "Neighbors must match cell count");

    if (n_cells == 0) return;

    // Compute average distance to k nearest neighbors
    auto avg_distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* avg_distances = avg_distances_ptr.release();

    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        Size n_neighbors = static_cast<Size>(row_end - row_start);

        if (n_neighbors == 0) {
            avg_distances[i] = std::numeric_limits<Real>::max();
            continue;
        }

        Real dist_sum = Real(0.0);
        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor = neighbors.col_indices_unsafe()[j];
            Real dist = std::sqrt(detail::sparse_distance_squared(
                expression, static_cast<Index>(i), neighbor));
            dist_sum += dist;
        }

        avg_distances[i] = dist_sum / static_cast<Real>(n_neighbors);
    }

    // Compute local outlier factor (simplified LOF)
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        Size n_neighbors = static_cast<Size>(row_end - row_start);

        if (n_neighbors == 0) {
            rarity_scores.ptr[i] = Real(1.0);
            continue;
        }

        Real neighbor_avg_dist = Real(0.0);
        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor = neighbors.col_indices_unsafe()[j];
            neighbor_avg_dist += avg_distances[neighbor];
        }
        neighbor_avg_dist /= static_cast<Real>(n_neighbors);

        if (neighbor_avg_dist > config::EPSILON) {
            rarity_scores.ptr[i] = avg_distances[i] / neighbor_avg_dist;
        } else {
            rarity_scores.ptr[i] = Real(1.0);
        }
    }

    scl::memory::aligned_free(avg_distances);
}

// =============================================================================
// Population Balance
// =============================================================================

void population_balance(
    Array<const Index> labels,
    Array<const Index> condition,
    Real* balance_matrix,  // [n_clusters * n_conditions]
    Size n_clusters,
    Size n_conditions
) {
    const Size n = labels.len;
    SCL_CHECK_DIM(n == condition.len, "Labels and condition must have same length");

    if (n == 0 || n_clusters == 0 || n_conditions == 0) return;

    // Count cells per cluster per condition
    auto counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters * n_conditions, SCL_ALIGNMENT);

    Size* counts = counts_ptr.release();
    auto total_per_cond_ptr = scl::memory::aligned_alloc<Size>(n_conditions, SCL_ALIGNMENT);

    Size* total_per_cond = total_per_cond_ptr.release();

    for (Size i = 0; i < n_clusters * n_conditions; ++i) counts[i] = 0;
    for (Size c = 0; c < n_conditions; ++c) total_per_cond[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index cluster = labels.ptr[i];
        Index cond = condition.ptr[i];
        if (cluster < static_cast<Index>(n_clusters) &&
            cond < static_cast<Index>(n_conditions)) {
            ++counts[cluster * n_conditions + cond];
            ++total_per_cond[cond];
        }
    }

    // Compute proportions
    for (Size c = 0; c < n_clusters; ++c) {
        for (Size d = 0; d < n_conditions; ++d) {
            if (total_per_cond[d] > 0) {
                balance_matrix[c * n_conditions + d] =
                    static_cast<Real>(counts[c * n_conditions + d]) /
                    static_cast<Real>(total_per_cond[d]);
            } else {
                balance_matrix[c * n_conditions + d] = Real(0.0);
            }
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(total_per_cond);
}

// =============================================================================
// Cluster Cohesion (intra-cluster distance)
// =============================================================================

template <typename T, bool IsCSR>
void cluster_cohesion(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Array<Real> cohesion_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");

    if (n_cells == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (labels.ptr[i] >= n_clusters) {
            n_clusters = labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(cohesion_scores.len >= static_cast<Size>(n_clusters),
        "Cohesion scores array too small");

    // Compute average intra-cluster distance
    auto total_dist_ptr = scl::memory::aligned_alloc<Real>(n_clusters, SCL_ALIGNMENT);

    Real* total_dist = total_dist_ptr.release();
    auto pair_counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* pair_counts = pair_counts_ptr.release();

    for (Index c = 0; c < n_clusters; ++c) {
        total_dist[c] = Real(0.0);
        pair_counts[c] = 0;
    }

    // Sample pairs for efficiency
    const Size max_pairs_per_cluster = 1000;

    for (Size i = 0; i < n_cells; ++i) {
        Index cluster_i = labels.ptr[i];

        for (Size j = i + 1; j < n_cells; ++j) {
            if (labels.ptr[j] != cluster_i) continue;
            if (pair_counts[cluster_i] >= max_pairs_per_cluster) break;

            Real dist = std::sqrt(detail::sparse_distance_squared(
                expression, static_cast<Index>(i), static_cast<Index>(j)));
            total_dist[cluster_i] += dist;
            ++pair_counts[cluster_i];
        }
    }

    // Compute cohesion (inverse of average distance)
    for (Index c = 0; c < n_clusters; ++c) {
        if (pair_counts[c] > 0) {
            Real avg_dist = total_dist[c] / static_cast<Real>(pair_counts[c]);
            cohesion_scores.ptr[c] = Real(1.0) / (avg_dist + config::EPSILON);
        } else {
            cohesion_scores.ptr[c] = Real(0.0);
        }
    }

    scl::memory::aligned_free(total_dist);
    scl::memory::aligned_free(pair_counts);
}

// =============================================================================
// Cluster Separation (inter-cluster distance)
// =============================================================================

template <typename T, bool IsCSR>
void cluster_separation(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Real* separation_matrix,  // [n_clusters * n_clusters]
    Size n_clusters
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");

    if (n_cells == 0 || n_clusters == 0) return;

    // Compute centroids
    const Size n_features = static_cast<Size>(expression.cols());
    auto centroids_ptr = scl::memory::aligned_alloc<Real>(n_clusters * n_features, SCL_ALIGNMENT);

    Real* centroids = centroids_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cluster_sizes = cluster_sizes_ptr.release();

    for (Size i = 0; i < n_clusters * n_features; ++i) centroids[i] = Real(0.0);
    for (Size c = 0; c < n_clusters; ++c) cluster_sizes[c] = 0;

    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = labels.ptr[i];
        ++cluster_sizes[cluster];

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = expression.col_indices_unsafe()[j];
            centroids[cluster * n_features + feature] +=
                static_cast<Real>(expression.values()[j]);
        }
    }

    for (Size c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 0) {
            for (Size f = 0; f < n_features; ++f) {
                centroids[c * n_features + f] /= static_cast<Real>(cluster_sizes[c]);
            }
        }
    }

    // Compute pairwise centroid distances
    for (Size c1 = 0; c1 < n_clusters; ++c1) {
        separation_matrix[c1 * n_clusters + c1] = Real(0.0);

        for (Size c2 = c1 + 1; c2 < n_clusters; ++c2) {
            Real dist = Real(0.0);
            for (Size f = 0; f < n_features; ++f) {
                Real diff = centroids[c1 * n_features + f] -
                           centroids[c2 * n_features + f];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            separation_matrix[c1 * n_clusters + c2] = dist;
            separation_matrix[c2 * n_clusters + c1] = dist;
        }
    }

    scl::memory::aligned_free(centroids);
    scl::memory::aligned_free(cluster_sizes);
}

// =============================================================================
// Identify Heterogeneous Clusters
// =============================================================================

template <typename T, bool IsCSR>
void identify_heterogeneous_clusters(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Real heterogeneity_threshold,
    Array<bool> is_heterogeneous
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");

    if (n_cells == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (labels.ptr[i] >= n_clusters) {
            n_clusters = labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(is_heterogeneous.len >= static_cast<Size>(n_clusters),
        "Heterogeneity array too small");

    // Compute variance within each cluster
    const Size n_features = static_cast<Size>(expression.cols());
    auto cluster_var_ptr = scl::memory::aligned_alloc<Real>(n_clusters, SCL_ALIGNMENT);

    Real* cluster_var = cluster_var_ptr.release();
    auto cluster_mean_ptr = scl::memory::aligned_alloc<Real>(n_clusters * n_features, SCL_ALIGNMENT);

    Real* cluster_mean = cluster_mean_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cluster_sizes = cluster_sizes_ptr.release();

    for (Index c = 0; c < n_clusters; ++c) {
        cluster_var[c] = Real(0.0);
        cluster_sizes[c] = 0;
    }
    for (Size i = 0; i < n_clusters * n_features; ++i) cluster_mean[i] = Real(0.0);

    // Compute means
    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = labels.ptr[i];
        ++cluster_sizes[cluster];

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = expression.col_indices_unsafe()[j];
            cluster_mean[cluster * n_features + feature] +=
                static_cast<Real>(expression.values()[j]);
        }
    }

    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 0) {
            for (Size f = 0; f < n_features; ++f) {
                cluster_mean[c * n_features + f] /= static_cast<Real>(cluster_sizes[c]);
            }
        }
    }

    // Compute variances
    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = labels.ptr[i];

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        Real cell_variance = Real(0.0);
        for (Index j = row_start; j < row_end; ++j) {
            Index feature = expression.col_indices_unsafe()[j];
            Real diff = static_cast<Real>(expression.values()[j]) -
                       cluster_mean[cluster * n_features + feature];
            cell_variance += diff * diff;
        }

        cluster_var[cluster] += cell_variance;
    }

    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 1) {
            cluster_var[c] /= static_cast<Real>(cluster_sizes[c] - 1);
        }
    }

    // Compute overall variance for threshold
    Real total_var = Real(0.0);
    Size total_count = 0;
    for (Index c = 0; c < n_clusters; ++c) {
        total_var += cluster_var[c] * static_cast<Real>(cluster_sizes[c]);
        total_count += cluster_sizes[c];
    }
    Real mean_var = (total_count > 0) ? total_var / static_cast<Real>(total_count) : Real(1.0);

    // Mark heterogeneous clusters
    for (Index c = 0; c < n_clusters; ++c) {
        is_heterogeneous.ptr[c] = (cluster_var[c] > heterogeneity_threshold * mean_var);
    }

    scl::memory::aligned_free(cluster_var);
    scl::memory::aligned_free(cluster_mean);
    scl::memory::aligned_free(cluster_sizes);
}

// =============================================================================
// Marker-based Subpopulation Detection
// =============================================================================

template <typename T, bool IsCSR>
void marker_based_subpopulation(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> marker_genes,
    Real threshold,
    Array<Index> subpop_labels
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_markers = marker_genes.len;
    SCL_CHECK_DIM(n_cells == subpop_labels.len, "Subpop labels must match cell count");

    if (n_cells == 0 || n_markers == 0) return;

    // For each cell, determine subpopulation based on marker expression pattern
    for (Size i = 0; i < n_cells; ++i) {
        Index pattern = 0;

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Size m = 0; m < n_markers; ++m) {
            Index marker = marker_genes.ptr[m];
            Real val = Real(0.0);

            for (Index j = row_start; j < row_end; ++j) {
                if (expression.col_indices_unsafe()[j] == marker) {
                    val = static_cast<Real>(expression.values()[j]);
                    break;
                }
            }

            if (val > threshold) {
                pattern |= (Index(1) << m);
            }
        }

        subpop_labels.ptr[i] = pattern;
    }
}

// =============================================================================
// Cluster Quality Score
// =============================================================================

template <typename T, bool IsCSR>
void cluster_quality_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Array<Real> quality_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");

    if (n_cells == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (labels.ptr[i] >= n_clusters) {
            n_clusters = labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(quality_scores.len >= static_cast<Size>(n_clusters),
        "Quality scores array too small");

    // Compute cohesion
    auto cohesion_ptr = scl::memory::aligned_alloc<Real>(n_clusters, SCL_ALIGNMENT);

    Real* cohesion = cohesion_ptr.release();
    Array<Real> cohesion_arr = {cohesion, static_cast<Size>(n_clusters)};
    cluster_cohesion(expression, labels, cohesion_arr);

    // Compute separation
    auto separation_ptr = scl::memory::aligned_alloc<Real>(n_clusters * n_clusters, SCL_ALIGNMENT);

    Real* separation = separation_ptr.release();
    cluster_separation(expression, labels, separation, static_cast<Size>(n_clusters));

    // Quality = cohesion * min_separation
    for (Index c = 0; c < n_clusters; ++c) {
        Real min_sep = std::numeric_limits<Real>::max();
        for (Index c2 = 0; c2 < n_clusters; ++c2) {
            if (c != c2 && separation[c * n_clusters + c2] < min_sep) {
                min_sep = separation[c * n_clusters + c2];
            }
        }

        if (min_sep < std::numeric_limits<Real>::max()) {
            quality_scores.ptr[c] = cohesion[c] * min_sep;
        } else {
            quality_scores.ptr[c] = cohesion[c];
        }
    }

    scl::memory::aligned_free(cohesion);
    scl::memory::aligned_free(separation);
}

// =============================================================================
// Find Optimal Number of Subclusters
// =============================================================================

template <typename T, bool IsCSR>
Index find_optimal_subclusters(
    const Sparse<T, IsCSR>& expression,
    const Index* cell_indices,
    Size n_cells,
    Size max_k,
    uint64_t seed
) {
    if (n_cells < config::MIN_CLUSTER_SIZE || max_k == 0) {
        return 1;
    }

    max_k = scl::algo::min2(max_k, n_cells / config::MIN_CLUSTER_SIZE);
    max_k = scl::algo::max2(max_k, Size(1));

    detail::LCG rng(seed);

    // Compute silhouette-like scores for different k values
    Real best_score = Real(-1.0);
    Index best_k = 1;

    auto assignments_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);


    Index* assignments = assignments_ptr.release();
    auto avg_dists_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* avg_dists = avg_dists_ptr.release();

    for (Size k = 2; k <= max_k; ++k) {
        // Cluster
        for (Size i = 0; i < n_cells; ++i) {
            assignments[i] = 0;
        }
        detail::kmeans_cluster(expression, cell_indices, n_cells, k, assignments, rng);

        // Compute silhouette-like score
        Real total_score = Real(0.0);
        Size valid_count = 0;

        for (Size i = 0; i < n_cells; ++i) {
            Index cluster_i = assignments[i];

            // Average distance to own cluster
            Real a = Real(0.0);
            Size a_count = 0;
            for (Size j = 0; j < n_cells; ++j) {
                if (i != j && assignments[j] == cluster_i) {
                    Real dist = std::sqrt(detail::sparse_distance_squared(
                        expression, cell_indices[i], cell_indices[j]));
                    a += dist;
                    ++a_count;
                }
            }
            if (a_count > 0) a /= static_cast<Real>(a_count);

            // Minimum average distance to other clusters
            Real b = std::numeric_limits<Real>::max();
            for (Size c = 0; c < k; ++c) {
                if (static_cast<Index>(c) == cluster_i) continue;

                Real dist_sum = Real(0.0);
                Size count = 0;
                for (Size j = 0; j < n_cells; ++j) {
                    if (assignments[j] == static_cast<Index>(c)) {
                        Real dist = std::sqrt(detail::sparse_distance_squared(
                            expression, cell_indices[i], cell_indices[j]));
                        dist_sum += dist;
                        ++count;
                    }
                }
                if (count > 0) {
                    Real avg = dist_sum / static_cast<Real>(count);
                    if (avg < b) b = avg;
                }
            }

            if (b < std::numeric_limits<Real>::max()) {
                Real max_ab = scl::algo::max2(a, b);
                if (max_ab > config::EPSILON) {
                    total_score += (b - a) / max_ab;
                    ++valid_count;
                }
            }
        }

        if (valid_count > 0) {
            Real avg_score = total_score / static_cast<Real>(valid_count);
            if (avg_score > best_score) {
                best_score = avg_score;
                best_k = static_cast<Index>(k);
            }
        }
    }

    scl::memory::aligned_free(assignments);
    scl::memory::aligned_free(avg_dists);

    return best_k;
}

// =============================================================================
// Cell Type Proportion Per Cluster
// =============================================================================

void cell_type_proportions(
    Array<const Index> cluster_labels,
    Array<const Index> cell_types,
    Real* proportion_matrix,  // [n_clusters * n_types]
    Size n_clusters,
    Size n_types
) {
    const Size n = cluster_labels.len;
    SCL_CHECK_DIM(n == cell_types.len, "Labels must have same length");

    if (n == 0 || n_clusters == 0 || n_types == 0) return;

    // Count cells per cluster per type
    auto counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters * n_types, SCL_ALIGNMENT);

    Size* counts = counts_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);

    Size* cluster_sizes = cluster_sizes_ptr.release();

    for (Size i = 0; i < n_clusters * n_types; ++i) counts[i] = 0;
    for (Size c = 0; c < n_clusters; ++c) cluster_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index cluster = cluster_labels.ptr[i];
        Index type = cell_types.ptr[i];
        if (cluster < static_cast<Index>(n_clusters) &&
            type < static_cast<Index>(n_types)) {
            ++counts[cluster * n_types + type];
            ++cluster_sizes[cluster];
        }
    }

    // Compute proportions
    for (Size c = 0; c < n_clusters; ++c) {
        for (Size t = 0; t < n_types; ++t) {
            if (cluster_sizes[c] > 0) {
                proportion_matrix[c * n_types + t] =
                    static_cast<Real>(counts[c * n_types + t]) /
                    static_cast<Real>(cluster_sizes[c]);
            } else {
                proportion_matrix[c * n_types + t] = Real(0.0);
            }
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(cluster_sizes);
}

} // namespace scl::kernel::subpopulation
