#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>
#include <atomic>

// =============================================================================
// FILE: scl/kernel/metrics.hpp
// BRIEF: Quality metrics for clustering and integration evaluation
//
// APPLICATIONS:
// - Clustering evaluation (Silhouette, ARI, NMI)
// - Integration quality assessment
// - Batch mixing metrics (LISI, batch entropy)
// =============================================================================

namespace scl::kernel::metrics {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real LOG2_E = Real(1.4426950408889634);
    constexpr Size PARALLEL_THRESHOLD = 256;
}

namespace detail {

// Find maximum label value
SCL_FORCE_INLINE Index find_max_label(Array<const Index> labels) {
    Index max_label = 0;
    for (Size i = 0; i < labels.len; ++i) {
        if (labels.ptr[i] > max_label) {
            max_label = labels.ptr[i];
        }
    }
    return max_label;
}

// Count elements per cluster
SCL_FORCE_INLINE void count_per_cluster(
    Array<const Index> labels,
    Index n_clusters,
    Size* counts
) {
    for (Index c = 0; c < n_clusters; ++c) {
        counts[c] = 0;
    }
    for (Size i = 0; i < labels.len; ++i) {
        ++counts[labels.ptr[i]];
    }
}

// Compute contingency table for two labelings
SCL_FORCE_INLINE void build_contingency_table(
    Array<const Index> labels1,
    Array<const Index> labels2,
    Index n_clusters1,
    Index n_clusters2,
    Size* contingency
) {
    const Size table_size = static_cast<Size>(n_clusters1) * static_cast<Size>(n_clusters2);
    for (Size i = 0; i < table_size; ++i) {
        contingency[i] = 0;
    }

    for (Size i = 0; i < labels1.len; ++i) {
        Index c1 = labels1.ptr[i];
        Index c2 = labels2.ptr[i];
        ++contingency[c1 * n_clusters2 + c2];
    }
}

// Compute entropy from counts
SCL_FORCE_INLINE Real compute_entropy(Size* counts, Index n, Size total) {
    if (total == 0) return Real(0.0);

    Real entropy = Real(0.0);
    const Real total_real = static_cast<Real>(total);

    for (Index i = 0; i < n; ++i) {
        if (counts[i] > 0) {
            Real p = static_cast<Real>(counts[i]) / total_real;
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

// Compute sum of n choose 2 for a set of counts
SCL_FORCE_INLINE Real sum_comb2(Size* counts, Index n) {
    Real sum = Real(0.0);
    for (Index i = 0; i < n; ++i) {
        if (counts[i] > 1) {
            Real c = static_cast<Real>(counts[i]);
            sum += c * (c - Real(1.0)) / Real(2.0);
        }
    }
    return sum;
}

// n choose 2
SCL_FORCE_INLINE Real comb2(Size n) {
    if (n < 2) return Real(0.0);
    Real n_real = static_cast<Real>(n);
    return n_real * (n_real - Real(1.0)) / Real(2.0);
}

} // namespace detail

// =============================================================================
// Silhouette Score
// =============================================================================

template <typename T, bool IsCSR>
Real silhouette_score(
    const Sparse<T, IsCSR>& distances,
    Array<const Index> labels
) {
    const Size n_cells = static_cast<Size>(distances.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Distance matrix rows must match labels length");

    if (n_cells < 2) return Real(0.0);

    const Index n_clusters = detail::find_max_label(labels) + 1;
    if (n_clusters < 2) return Real(0.0);

    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* cluster_sizes = cluster_sizes_ptr.get();
    detail::count_per_cluster(labels, n_clusters, cluster_sizes);

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        std::atomic<Real> total_silhouette{Real(0.0)};
        std::atomic<Size> valid_count{0};

        scl::threading::WorkspacePool<Real> workspace;
        workspace.init(scl::threading::get_num_threads_runtime(), static_cast<Size>(n_clusters));

        scl::threading::parallel_for(Size(0), n_cells, [&](Size i, Size thread_id) {
            const Index my_cluster = labels.ptr[i];

            if (cluster_sizes[my_cluster] < 2) return;

            Real* cluster_dist_sum = workspace.get(thread_id);
            for (Index c = 0; c < n_clusters; ++c) {
                cluster_dist_sum[c] = Real(0.0);
            }

            const Index row_start = distances.row_indices_unsafe(i);
            const Index row_end = distances.row_indices_unsafe(i + 1);

            for (Index j = row_start; j < row_end; ++j) {
                const Index neighbor = distances.col_indices_unsafe(j);
                if (neighbor == static_cast<Index>(i)) continue;

                const T dist_val = distances.values()[j];
                const Index neighbor_cluster = labels.ptr[neighbor];
                cluster_dist_sum[neighbor_cluster] += static_cast<Real>(dist_val);
            }

            const Real a_i = cluster_dist_sum[my_cluster] / static_cast<Real>(cluster_sizes[my_cluster] - 1);

            Real b_i = std::numeric_limits<Real>::max();
            for (Index c = 0; c < n_clusters; ++c) {
                if (c != my_cluster && cluster_sizes[c] > 0) {
                    const Real mean_dist = cluster_dist_sum[c] / static_cast<Real>(cluster_sizes[c]);
                    if (mean_dist < b_i) {
                        b_i = mean_dist;
                    }
                }
            }

            if (b_i < std::numeric_limits<Real>::max()) {
                const Real max_ab = scl::algo::max2(a_i, b_i);
                if (max_ab > config::EPSILON) {
                    const Real s_i = (b_i - a_i) / max_ab;
                    Real expected = total_silhouette.load(std::memory_order_relaxed);
                    while (!total_silhouette.compare_exchange_weak(expected, expected + s_i,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {}
                    valid_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

        const Size count = valid_count.load();
        if (count == 0) return Real(0.0);
        return total_silhouette.load() / static_cast<Real>(count);
    }

    // Sequential fallback for small inputs
    Real total_silhouette = Real(0.0);
    Size valid_count = 0;
    auto cluster_dist_sum_ptr = scl::memory::aligned_alloc<Real>(n_clusters, SCL_ALIGNMENT);
    Real* cluster_dist_sum = cluster_dist_sum_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        const Index my_cluster = labels.ptr[i];

        if (cluster_sizes[my_cluster] < 2) continue;

        for (Index c = 0; c < n_clusters; ++c) {
            cluster_dist_sum[c] = Real(0.0);
        }

        const Index row_start = distances.row_indices_unsafe(i);
        const Index row_end = distances.row_indices_unsafe(i + 1);

        for (Index j = row_start; j < row_end; ++j) {
            const Index neighbor = distances.col_indices_unsafe(j);
            if (neighbor == static_cast<Index>(i)) continue;

            const T dist_val = distances.values()[j];
            const Index neighbor_cluster = labels.ptr[neighbor];
            cluster_dist_sum[neighbor_cluster] += static_cast<Real>(dist_val);
        }

        const Real a_i = cluster_dist_sum[my_cluster] / static_cast<Real>(cluster_sizes[my_cluster] - 1);

        Real b_i = std::numeric_limits<Real>::max();
        for (Index c = 0; c < n_clusters; ++c) {
            if (c != my_cluster && cluster_sizes[c] > 0) {
                const Real mean_dist = cluster_dist_sum[c] / static_cast<Real>(cluster_sizes[c]);
                if (mean_dist < b_i) {
                    b_i = mean_dist;
                }
            }
        }

        if (b_i < std::numeric_limits<Real>::max()) {
            const Real max_ab = scl::algo::max2(a_i, b_i);
            if (max_ab > config::EPSILON) {
                const Real s_i = (b_i - a_i) / max_ab;
                total_silhouette += s_i;
                ++valid_count;
            }
        }
    }

    if (valid_count == 0) return Real(0.0);
    return total_silhouette / static_cast<Real>(valid_count);
}

// Silhouette score per sample
template <typename T, bool IsCSR>
void silhouette_samples(
    const Sparse<T, IsCSR>& distances,
    Array<const Index> labels,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(distances.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Distance matrix rows must match labels length");
    SCL_CHECK_DIM(n_cells == scores.len, "Scores length must match number of cells");

    if (n_cells < 2) {
        for (Size i = 0; i < n_cells; ++i) {
            scores.ptr[i] = Real(0.0);
        }
        return;
    }

    const Index n_clusters = detail::find_max_label(labels) + 1;

    // PERFORMANCE: RAII memory management with unique_ptr
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* cluster_sizes = cluster_sizes_ptr.get();
    detail::count_per_cluster(labels, n_clusters, cluster_sizes);

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::WorkspacePool<Real> workspace;
        workspace.init(scl::threading::get_num_threads_runtime(), static_cast<Size>(n_clusters));

        scl::threading::parallel_for(Size(0), n_cells, [&](Size i, Size thread_id) {
            const Index my_cluster = labels.ptr[i];

            if (cluster_sizes[my_cluster] < 2 || n_clusters < 2) {
                scores.ptr[i] = Real(0.0);
                return;
            }

            Real* cluster_dist_sum = workspace.get(thread_id);
            for (Index c = 0; c < n_clusters; ++c) {
                cluster_dist_sum[c] = Real(0.0);
            }

            const Index row_start = distances.row_indices_unsafe(i);
            const Index row_end = distances.row_indices_unsafe(i + 1);

            for (Index j = row_start; j < row_end; ++j) {
                const Index neighbor = distances.col_indices_unsafe(j);
                if (neighbor == static_cast<Index>(i)) continue;

                const T dist_val = distances.values()[j];
                const Index neighbor_cluster = labels.ptr[neighbor];
                cluster_dist_sum[neighbor_cluster] += static_cast<Real>(dist_val);
            }

            const Real a_i = cluster_dist_sum[my_cluster] / static_cast<Real>(cluster_sizes[my_cluster] - 1);

            Real b_i = std::numeric_limits<Real>::max();
            for (Index c = 0; c < n_clusters; ++c) {
                if (c != my_cluster && cluster_sizes[c] > 0) {
                    const Real mean_dist = cluster_dist_sum[c] / static_cast<Real>(cluster_sizes[c]);
                    if (mean_dist < b_i) {
                        b_i = mean_dist;
                    }
                }
            }

            if (b_i < std::numeric_limits<Real>::max()) {
                const Real max_ab = scl::algo::max2(a_i, b_i);
                if (max_ab > config::EPSILON) {
                    scores.ptr[i] = (b_i - a_i) / max_ab;
                } else {
                    scores.ptr[i] = Real(0.0);
                }
            } else {
                scores.ptr[i] = Real(0.0);
            }
        });
    } else {
        auto cluster_dist_sum_ptr = scl::memory::aligned_alloc<Real>(n_clusters, SCL_ALIGNMENT);
        Real* cluster_dist_sum = cluster_dist_sum_ptr.get();

        for (Size i = 0; i < n_cells; ++i) {
            const Index my_cluster = labels.ptr[i];

            if (cluster_sizes[my_cluster] < 2 || n_clusters < 2) {
                scores.ptr[i] = Real(0.0);
                continue;
            }

            for (Index c = 0; c < n_clusters; ++c) {
                cluster_dist_sum[c] = Real(0.0);
            }

            const Index row_start = distances.row_indices_unsafe(i);
            const Index row_end = distances.row_indices_unsafe(i + 1);

            for (Index j = row_start; j < row_end; ++j) {
                const Index neighbor = distances.col_indices_unsafe(j);
                if (neighbor == static_cast<Index>(i)) continue;

                const T dist_val = distances.values()[j];
                const Index neighbor_cluster = labels.ptr[neighbor];
                cluster_dist_sum[neighbor_cluster] += static_cast<Real>(dist_val);
            }

            const Real a_i = cluster_dist_sum[my_cluster] / static_cast<Real>(cluster_sizes[my_cluster] - 1);

            Real b_i = std::numeric_limits<Real>::max();
            for (Index c = 0; c < n_clusters; ++c) {
                if (c != my_cluster && cluster_sizes[c] > 0) {
                    const Real mean_dist = cluster_dist_sum[c] / static_cast<Real>(cluster_sizes[c]);
                    if (mean_dist < b_i) {
                        b_i = mean_dist;
                    }
                }
            }

            if (b_i < std::numeric_limits<Real>::max()) {
                const Real max_ab = scl::algo::max2(a_i, b_i);
                if (max_ab > config::EPSILON) {
                    scores.ptr[i] = (b_i - a_i) / max_ab;
                } else {
                    scores.ptr[i] = Real(0.0);
                }
            } else {
                scores.ptr[i] = Real(0.0);
            }
        }
    }
}

// =============================================================================
// Adjusted Rand Index (ARI)
// =============================================================================

Real adjusted_rand_index(
    Array<const Index> labels1,
    Array<const Index> labels2
) {
    SCL_CHECK_DIM(labels1.len == labels2.len, "Both label arrays must have same length");

    const Size n = labels1.len;
    if (n < 2) return Real(1.0);

    Index n_clusters1 = detail::find_max_label(labels1) + 1;
    Index n_clusters2 = detail::find_max_label(labels2) + 1;

    // Build contingency table
    auto contingency_ptr = scl::memory::aligned_alloc<Size>(
        static_cast<Size>(n_clusters1) * static_cast<Size>(n_clusters2), SCL_ALIGNMENT);
    Size* contingency = contingency_ptr.get();
    detail::build_contingency_table(labels1, labels2, n_clusters1, n_clusters2, contingency);

    // Compute row and column sums
    // PERFORMANCE: RAII memory management with unique_ptr
    auto row_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters1, SCL_ALIGNMENT);
    auto col_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters2, SCL_ALIGNMENT);
    Size* row_sums = row_sums_ptr.get();
    Size* col_sums = col_sums_ptr.get();

    for (Index i = 0; i < n_clusters1; ++i) row_sums[i] = 0;
    for (Index j = 0; j < n_clusters2; ++j) col_sums[j] = 0;

    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size count = contingency[i * n_clusters2 + j];
            row_sums[i] += count;
            col_sums[j] += count;
        }
    }

    // Compute sum of n_ij choose 2
    Real sum_nij_comb = Real(0.0);
    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size nij = contingency[i * n_clusters2 + j];
            if (nij > 1) {
                sum_nij_comb += detail::comb2(nij);
            }
        }
    }

    // Compute sum of a_i choose 2 and b_j choose 2
    Real sum_ai_comb = detail::sum_comb2(row_sums, n_clusters1);
    Real sum_bj_comb = detail::sum_comb2(col_sums, n_clusters2);

    // n choose 2
    Real n_comb = detail::comb2(n);

    // ARI = (sum_nij_comb - expected) / (mean_comb - expected)
    Real expected = sum_ai_comb * sum_bj_comb / n_comb;
    Real mean_comb = (sum_ai_comb + sum_bj_comb) / Real(2.0);

    Real denom = mean_comb - expected;
    if (std::abs(denom) < config::EPSILON) {
        return Real(1.0);
    }

    return (sum_nij_comb - expected) / denom;
}

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

Real normalized_mutual_information(
    Array<const Index> labels1,
    Array<const Index> labels2
) {
    SCL_CHECK_DIM(labels1.len == labels2.len, "Both label arrays must have same length");

    const Size n = labels1.len;
    if (n == 0) return Real(0.0);

    Index n_clusters1 = detail::find_max_label(labels1) + 1;
    Index n_clusters2 = detail::find_max_label(labels2) + 1;

    // Build contingency table
    auto contingency_ptr = scl::memory::aligned_alloc<Size>(
        static_cast<Size>(n_clusters1) * static_cast<Size>(n_clusters2), SCL_ALIGNMENT);
    Size* contingency = contingency_ptr.get();
    detail::build_contingency_table(labels1, labels2, n_clusters1, n_clusters2, contingency);

    // Compute row and column sums
    // PERFORMANCE: RAII memory management with unique_ptr
    auto row_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters1, SCL_ALIGNMENT);
    auto col_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters2, SCL_ALIGNMENT);
    Size* row_sums = row_sums_ptr.get();
    Size* col_sums = col_sums_ptr.get();

    for (Index i = 0; i < n_clusters1; ++i) row_sums[i] = 0;
    for (Index j = 0; j < n_clusters2; ++j) col_sums[j] = 0;

    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size count = contingency[i * n_clusters2 + j];
            row_sums[i] += count;
            col_sums[j] += count;
        }
    }

    // Compute entropies
    Real H1 = detail::compute_entropy(row_sums, n_clusters1, n);
    Real H2 = detail::compute_entropy(col_sums, n_clusters2, n);

    // Compute mutual information
    Real MI = Real(0.0);
    const Real n_real = static_cast<Real>(n);

    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size nij = contingency[i * n_clusters2 + j];
            if (nij > 0) {
                Real p_ij = static_cast<Real>(nij) / n_real;
                Real p_i = static_cast<Real>(row_sums[i]) / n_real;
                Real p_j = static_cast<Real>(col_sums[j]) / n_real;
                MI += p_ij * std::log(p_ij / (p_i * p_j));
            }
        }
    }

    // Normalized by arithmetic mean of entropies
    Real denom = (H1 + H2) / Real(2.0);
    if (denom < config::EPSILON) {
        return Real(1.0);
    }

    return MI / denom;
}

// =============================================================================
// Graph Connectivity
// =============================================================================

template <typename T, bool IsCSR>
Real graph_connectivity(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Index> labels
) {
    const Size n_cells = static_cast<Size>(adjacency.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Adjacency matrix rows must match labels length");

    if (n_cells == 0) return Real(1.0);

    Index n_clusters = detail::find_max_label(labels) + 1;
    if (n_clusters == 0) return Real(1.0);

    // For each cluster, count how many connected components exist
    auto component_id_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Index* component_id = component_id_ptr.get();
    for (Size i = 0; i < n_cells; ++i) {
        component_id[i] = -1;
    }

    Size total_components = 0;
    auto queue_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Index* queue = queue_ptr.get();

    // BFS to find connected components within each cluster
    for (Size start = 0; start < n_cells; ++start) {
        if (component_id[start] >= 0) continue;

        Index my_cluster = labels.ptr[start];
        component_id[start] = static_cast<Index>(total_components);

        Size queue_start = 0, queue_end = 1;
        queue[0] = static_cast<Index>(start);

        while (queue_start < queue_end) {
            Index curr = queue[queue_start++];

            const Index row_start = adjacency.row_indices_unsafe()[curr];
            const Index row_end = adjacency.row_indices_unsafe()[curr + 1];

            for (Index j = row_start; j < row_end; ++j) {
                Index neighbor = adjacency.col_indices_unsafe()[j];
                if (component_id[neighbor] < 0 && labels.ptr[neighbor] == my_cluster) {
                    component_id[neighbor] = static_cast<Index>(total_components);
                    queue[queue_end++] = neighbor;
                }
            }
        }

        ++total_components;
    }

    // Count components per cluster
    auto components_per_cluster_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* components_per_cluster = components_per_cluster_ptr.get();
    for (Index c = 0; c < n_clusters; ++c) {
        components_per_cluster[c] = 0;
    }

    // Use a set-like approach: track unique component IDs per cluster
    auto seen_components_ptr = scl::memory::aligned_alloc<Index>(total_components, SCL_ALIGNMENT);
    Index* seen_components = seen_components_ptr.get();
    for (Size i = 0; i < total_components; ++i) {
        seen_components[i] = -1;
    }

    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = labels.ptr[i];
        Index comp = component_id[i];
        if (seen_components[comp] != cluster) {
            seen_components[comp] = cluster;
            ++components_per_cluster[cluster];
        }
    }

    // Connectivity = fraction of clusters that are fully connected (1 component)
    Size connected_clusters = 0;
    Size non_empty_clusters = 0;
    for (Index c = 0; c < n_clusters; ++c) {
        if (components_per_cluster[c] > 0) {
            ++non_empty_clusters;
            if (components_per_cluster[c] == 1) {
                ++connected_clusters;
            }
        }
    }

    if (non_empty_clusters == 0) return Real(1.0);
    return static_cast<Real>(connected_clusters) / static_cast<Real>(non_empty_clusters);
}

// =============================================================================
// Batch Entropy
// =============================================================================

template <bool IsCSR>
void batch_entropy(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> batch_labels,
    Array<Real> entropy_scores
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    SCL_CHECK_DIM(n_cells == batch_labels.len, "Neighbors rows must match batch labels length");
    SCL_CHECK_DIM(n_cells == entropy_scores.len, "Entropy scores length must match number of cells");

    if (n_cells == 0) return;

    Index n_batches = detail::find_max_label(batch_labels) + 1;
    const Real max_entropy = std::log(static_cast<Real>(n_batches));

    auto compute_cell_entropy = [&](Size i, Size* batch_counts) {
        for (Index b = 0; b < n_batches; ++b) {
            batch_counts[b] = 0;
        }

        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        Size total_neighbors = static_cast<Size>(row_end - row_start);

        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor_idx = neighbors.col_indices_unsafe()[j];
            Index batch = batch_labels.ptr[neighbor_idx];
            ++batch_counts[batch];
        }

        ++batch_counts[batch_labels.ptr[i]];
        ++total_neighbors;

        Real entropy = Real(0.0);
        const Real total_real = static_cast<Real>(total_neighbors);

        for (Index b = 0; b < n_batches; ++b) {
            if (batch_counts[b] > 0) {
                Real p = static_cast<Real>(batch_counts[b]) / total_real;
                entropy -= p * std::log(p);
            }
        }

        if (max_entropy > config::EPSILON) {
            entropy_scores.ptr[i] = entropy / max_entropy;
        } else {
            entropy_scores.ptr[i] = Real(1.0);
        }
    };

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::WorkspacePool<Size> workspace;
        workspace.init(scl::threading::get_num_threads_runtime(), static_cast<Size>(n_batches));

        scl::threading::parallel_for(Size(0), n_cells, [&](Size i, Size thread_id) {
            compute_cell_entropy(i, workspace.get(thread_id));
        });
    } else {
        auto batch_counts_ptr = scl::memory::aligned_alloc<Size>(n_batches, SCL_ALIGNMENT);
        Size* batch_counts = batch_counts_ptr.get();
        for (Size i = 0; i < n_cells; ++i) {
            compute_cell_entropy(i, batch_counts);
        }
    }
}

// =============================================================================
// Local Inverse Simpson's Index (LISI)
// =============================================================================

template <bool IsCSR>
void lisi(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> labels,
    Array<Real> lisi_scores
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    SCL_CHECK_DIM(n_cells == labels.len, "Neighbors rows must match labels length");
    SCL_CHECK_DIM(n_cells == lisi_scores.len, "LISI scores length must match number of cells");

    if (n_cells == 0) return;

    Index n_labels = detail::find_max_label(labels) + 1;

    auto compute_lisi = [&](Size i, Size* label_counts) {
        for (Index l = 0; l < n_labels; ++l) {
            label_counts[l] = 0;
        }

        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        Size total_neighbors = 0;

        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor_idx = neighbors.col_indices_unsafe()[j];
            Index label = labels.ptr[neighbor_idx];
            ++label_counts[label];
            ++total_neighbors;
        }

        if (total_neighbors == 0) {
            lisi_scores.ptr[i] = Real(1.0);
            return;
        }

        Real simpson = Real(0.0);
        const Real total_real = static_cast<Real>(total_neighbors);

        for (Index l = 0; l < n_labels; ++l) {
            if (label_counts[l] > 0) {
                Real p = static_cast<Real>(label_counts[l]) / total_real;
                simpson += p * p;
            }
        }

        if (simpson > config::EPSILON) {
            lisi_scores.ptr[i] = Real(1.0) / simpson;
        } else {
            lisi_scores.ptr[i] = static_cast<Real>(n_labels);
        }
    };

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::WorkspacePool<Size> workspace;
        workspace.init(scl::threading::get_num_threads_runtime(), static_cast<Size>(n_labels));

        scl::threading::parallel_for(Size(0), n_cells, [&](Size i, Size thread_id) {
            compute_lisi(i, workspace.get(thread_id));
        });
    } else {
        auto label_counts_ptr = scl::memory::aligned_alloc<Size>(n_labels, SCL_ALIGNMENT);
        Size* label_counts = label_counts_ptr.get();
        for (Size i = 0; i < n_cells; ++i) {
            compute_lisi(i, label_counts);
        }
    }
}

// =============================================================================
// Additional metrics
// =============================================================================

// Fowlkes-Mallows Index
Real fowlkes_mallows_index(
    Array<const Index> labels1,
    Array<const Index> labels2
) {
    SCL_CHECK_DIM(labels1.len == labels2.len, "Both label arrays must have same length");

    const Size n = labels1.len;
    if (n < 2) return Real(1.0);

    Index n_clusters1 = detail::find_max_label(labels1) + 1;
    Index n_clusters2 = detail::find_max_label(labels2) + 1;

    // PERFORMANCE: RAII memory management with unique_ptr
    auto contingency_ptr = scl::memory::aligned_alloc<Size>(
        static_cast<Size>(n_clusters1) * static_cast<Size>(n_clusters2), SCL_ALIGNMENT);
    Size* contingency = contingency_ptr.get();
    detail::build_contingency_table(labels1, labels2, n_clusters1, n_clusters2, contingency);

    auto row_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters1, SCL_ALIGNMENT);
    auto col_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters2, SCL_ALIGNMENT);
    Size* row_sums = row_sums_ptr.get();
    Size* col_sums = col_sums_ptr.get();

    for (Index i = 0; i < n_clusters1; ++i) row_sums[i] = 0;
    for (Index j = 0; j < n_clusters2; ++j) col_sums[j] = 0;

    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size count = contingency[i * n_clusters2 + j];
            row_sums[i] += count;
            col_sums[j] += count;
        }
    }

    // TP = sum(n_ij choose 2)
    Real TP = Real(0.0);
    for (Index i = 0; i < n_clusters1; ++i) {
        for (Index j = 0; j < n_clusters2; ++j) {
            Size nij = contingency[i * n_clusters2 + j];
            if (nij > 1) {
                TP += detail::comb2(nij);
            }
        }
    }

    // P = sum(a_i choose 2), Q = sum(b_j choose 2)
    Real P = detail::sum_comb2(row_sums, n_clusters1);
    Real Q = detail::sum_comb2(col_sums, n_clusters2);

    if (P < config::EPSILON || Q < config::EPSILON) {
        return Real(0.0);
    }

    return TP / std::sqrt(P * Q);
}

// V-measure (harmonic mean of homogeneity and completeness)
Real v_measure(
    Array<const Index> labels_true,
    Array<const Index> labels_pred,
    Real beta
) {
    SCL_CHECK_DIM(labels_true.len == labels_pred.len, "Both label arrays must have same length");

    const Size n = labels_true.len;
    if (n == 0) return Real(0.0);

    Index n_classes = detail::find_max_label(labels_true) + 1;
    Index n_clusters = detail::find_max_label(labels_pred) + 1;
    auto contingency_ptr = scl::memory::aligned_alloc<Size>(
        static_cast<Size>(n_classes) * static_cast<Size>(n_clusters), SCL_ALIGNMENT);
    Size* contingency = contingency_ptr.get();

    // Build contingency with classes as rows, clusters as columns
    const Size table_size = static_cast<Size>(n_classes) * static_cast<Size>(n_clusters);
    for (Size i = 0; i < table_size; ++i) {
        contingency[i] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index c = labels_true.ptr[i];
        Index k = labels_pred.ptr[i];
        ++contingency[c * n_clusters + k];
    }

    auto class_sums_ptr = scl::memory::aligned_alloc<Size>(n_classes, SCL_ALIGNMENT);
    Size* class_sums = class_sums_ptr.get();
    auto cluster_sums_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* cluster_sums = cluster_sums_ptr.get();

    for (Index c = 0; c < n_classes; ++c) class_sums[c] = 0;
    for (Index k = 0; k < n_clusters; ++k) cluster_sums[k] = 0;

    for (Index c = 0; c < n_classes; ++c) {
        for (Index k = 0; k < n_clusters; ++k) {
            Size count = contingency[c * n_clusters + k];
            class_sums[c] += count;
            cluster_sums[k] += count;
        }
    }

    // H(C) and H(K)
    Real H_C = detail::compute_entropy(class_sums, n_classes, n);
    Real H_K = detail::compute_entropy(cluster_sums, n_clusters, n);

    // H(C|K) = -sum(n_ck/n * log(n_ck/n_k))
    Real H_C_given_K = Real(0.0);
    const Real n_real = static_cast<Real>(n);

    for (Index c = 0; c < n_classes; ++c) {
        for (Index k = 0; k < n_clusters; ++k) {
            Size n_ck = contingency[c * n_clusters + k];
            if (n_ck > 0 && cluster_sums[k] > 0) {
                Real p_ck = static_cast<Real>(n_ck) / n_real;
                H_C_given_K -= p_ck * std::log(static_cast<Real>(n_ck) / static_cast<Real>(cluster_sums[k]));
            }
        }
    }

    // H(K|C)
    Real H_K_given_C = Real(0.0);
    for (Index c = 0; c < n_classes; ++c) {
        for (Index k = 0; k < n_clusters; ++k) {
            Size n_ck = contingency[c * n_clusters + k];
            if (n_ck > 0 && class_sums[c] > 0) {
                H_K_given_C -= static_cast<Real>(n_ck) / n_real *
                    std::log(static_cast<Real>(n_ck) / static_cast<Real>(class_sums[c]));
            }
        }
    }

    // Homogeneity h = 1 - H(C|K)/H(C)
    Real h = (H_C > config::EPSILON) ? (Real(1.0) - H_C_given_K / H_C) : Real(1.0);

    // Completeness c = 1 - H(K|C)/H(K)
    Real c = (H_K > config::EPSILON) ? (Real(1.0) - H_K_given_C / H_K) : Real(1.0);

    // V-measure
    if (h + c < config::EPSILON) {
        return Real(0.0);
    }

    return (Real(1.0) + beta) * h * c / (beta * h + c);
}

// Homogeneity score
Real homogeneity_score(
    Array<const Index> labels_true,
    Array<const Index> labels_pred
) {
    return v_measure(labels_true, labels_pred, Real(0.0));
}

// Completeness score
Real completeness_score(
    Array<const Index> labels_true,
    Array<const Index> labels_pred
) {
    // Swap true and pred to get completeness from homogeneity
    return v_measure(labels_pred, labels_true, Real(0.0));
}

// Purity score
Real purity_score(
    Array<const Index> labels_true,
    Array<const Index> labels_pred
) {
    SCL_CHECK_DIM(labels_true.len == labels_pred.len, "Both label arrays must have same length");

    const Size n = labels_true.len;
    if (n == 0) return Real(0.0);

    Index n_classes = detail::find_max_label(labels_true) + 1;
    Index n_clusters = detail::find_max_label(labels_pred) + 1;

    auto contingency_ptr = scl::memory::aligned_alloc<Size>(
        static_cast<Size>(n_classes) * static_cast<Size>(n_clusters), SCL_ALIGNMENT);
    Size* contingency = contingency_ptr.get();

    const Size table_size = static_cast<Size>(n_classes) * static_cast<Size>(n_clusters);
    for (Size i = 0; i < table_size; ++i) {
        contingency[i] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index c = labels_true.ptr[i];
        Index k = labels_pred.ptr[i];
        ++contingency[c * n_clusters + k];
    }

    // For each cluster, find max class count
    Size total_correct = 0;
    for (Index k = 0; k < n_clusters; ++k) {
        Size max_count = 0;
        for (Index c = 0; c < n_classes; ++c) {
            Size count = contingency[c * n_clusters + k];
            if (count > max_count) {
                max_count = count;
            }
        }
        total_correct += max_count;
    }

    return static_cast<Real>(total_correct) / static_cast<Real>(n);
}

// Mean LISI score
template <bool IsCSR>
Real mean_lisi(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> labels
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    if (n_cells == 0) return Real(1.0);

    auto scores_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* scores = scores_ptr.get();
    Array<Real> scores_array = {scores, n_cells};

    lisi(neighbors, labels, scores_array);

    Real sum = scl::vectorize::sum(Array<const Real>(scores, n_cells));

    return sum / static_cast<Real>(n_cells);
}

// Mean batch entropy
template <bool IsCSR>
Real mean_batch_entropy(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> batch_labels
) {
    const Size n_cells = static_cast<Size>(neighbors.rows());
    if (n_cells == 0) return Real(1.0);

    auto scores_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* scores = scores_ptr.get();
    Array<Real> scores_array = {scores, n_cells};

    batch_entropy(neighbors, batch_labels, scores_array);

    Real sum = scl::vectorize::sum(Array<const Real>(scores, n_cells));

    return sum / static_cast<Real>(n_cells);
}

} // namespace scl::kernel::metrics
