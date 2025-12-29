#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <memory>

// =============================================================================
// FILE: scl/kernel/clonotype.hpp
// BRIEF: TCR/BCR clonal analysis
//
// APPLICATIONS:
// - Clonal expansion analysis
// - Clonal diversity
// - Clone-phenotype association
// - Repertoire overlap analysis
// =============================================================================

namespace scl::kernel::clonotype {

// =============================================================================
// C++20 Concepts
// =============================================================================

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    inline constexpr Real EPSILON = Real(1e-10);
    inline constexpr Index NO_CLONE = -1;
    inline constexpr Size MIN_CLONE_SIZE = 2;
}

namespace detail {

// Count unique clones and their sizes
SCL_FORCE_INLINE Size count_clones(
    const Index* clone_ids,
    Size n,
    Size* clone_sizes,
    Size max_clones
) {
    // Initialize
    for (Size i = 0; i < max_clones; ++i) {
        clone_sizes[i] = 0;
    }

    // Count
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids[i];
        if (clone >= 0 && clone < static_cast<Index>(max_clones)) {
            ++clone_sizes[clone];
            if (clone > max_clone) max_clone = clone;
        }
    }

    return (max_clone >= 0) ? static_cast<Size>(max_clone + 1) : 0;
}

// Compute Shannon entropy from frequencies
SCL_FORCE_INLINE Real shannon_entropy(const Real* freqs, Size n) {
    Real entropy = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        if (freqs[i] > config::EPSILON) {
            entropy -= freqs[i] * std::log(freqs[i]);
        }
    }
    return entropy;
}

// Compute Simpson's index from frequencies
SCL_FORCE_INLINE Real simpson_index(const Real* freqs, Size n) {
    Real sum_sq = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        sum_sq += freqs[i] * freqs[i];
    }
    return Real(1.0) - sum_sq;
}

// Compute Gini coefficient from sizes
SCL_FORCE_INLINE Real gini_coefficient(const Size* sizes, Size n) {
    if (n == 0) return Real(0.0);

    // Sort sizes (copy first)
    auto sorted_ptr = scl::memory::aligned_alloc<Size>(n, SCL_ALIGNMENT);
    Size* sorted = sorted_ptr.release();
    for (Size i = 0; i < n; ++i) {
        sorted[i] = sizes[i];
    }
    std::sort(sorted, sorted + n);

    // Compute Gini
    Real sum = Real(0.0);
    Real cumsum = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        sum += static_cast<Real>(sorted[i]);
        cumsum += static_cast<Real>(sorted[i]) * static_cast<Real>(n - i);
    }

    scl::memory::aligned_free(sorted);

    if (sum < config::EPSILON) return Real(0.0);

    Real n_r = static_cast<Real>(n);
    return (n_r + Real(1.0) - Real(2.0) * cumsum / sum) / n_r;
}

} // namespace detail

// =============================================================================
// Clone Size Distribution
// =============================================================================

void clone_size_distribution(
    Array<const Index> clone_ids,
    Size* clone_sizes,
    Size& n_clones,
    Size max_clones
) {
    const Size n = clone_ids.len;

    if (n == 0 || max_clones == 0) {
        n_clones = 0;
        return;
    }

    n_clones = detail::count_clones(clone_ids.ptr, n, clone_sizes, max_clones);
}

// =============================================================================
// Clonal Diversity Indices
// =============================================================================

void clonal_diversity(
    Array<const Index> clone_ids,
    Real& shannon_diversity,
    Real& simpson_diversity,
    Real& gini_index
) {
    const Size n = clone_ids.len;

    shannon_diversity = Real(0.0);
    simpson_diversity = Real(0.0);
    gini_index = Real(0.0);

    if (n == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) {
            max_clone = clone_ids.ptr[i];
        }
    }

    if (max_clone < 0) return;

    Size n_possible_clones = static_cast<Size>(max_clone + 1);

    // Count clone sizes
    auto clone_sizes_ptr = scl::memory::aligned_alloc<Size>(n_possible_clones, SCL_ALIGNMENT);
    Size* clone_sizes = clone_sizes_ptr.release();
    [[maybe_unused]] Size n_clones = detail::count_clones(clone_ids.ptr, n, clone_sizes, n_possible_clones);

    // Count non-empty clones and total cells with clone
    Size n_with_clone = 0;
    Size n_nonempty = 0;
    for (Size c = 0; c < n_possible_clones; ++c) {
        if (clone_sizes[c] > 0) {
            n_with_clone += clone_sizes[c];
            ++n_nonempty;
        }
    }

    if (n_with_clone == 0 || n_nonempty == 0) {
        scl::memory::aligned_free(clone_sizes);
        return;
    }

    // Compute frequencies
    auto freqs_ptr = scl::memory::aligned_alloc<Real>(n_nonempty, SCL_ALIGNMENT);
    auto sizes_ptr = scl::memory::aligned_alloc<Size>(n_nonempty, SCL_ALIGNMENT);
    Real* freqs = freqs_ptr.release();
    Size* sizes = sizes_ptr.release();
    Size idx = 0;

    for (Size c = 0; c < n_possible_clones; ++c) {
        if (clone_sizes[c] > 0) {
            freqs[idx] = static_cast<Real>(clone_sizes[c]) / static_cast<Real>(n_with_clone);
            sizes[idx] = clone_sizes[c];
            ++idx;
        }
    }

    // Compute diversity indices
    shannon_diversity = detail::shannon_entropy(freqs, n_nonempty);
    simpson_diversity = detail::simpson_index(freqs, n_nonempty);
    gini_index = detail::gini_coefficient(sizes, n_nonempty);

    scl::memory::aligned_free(clone_sizes);
    scl::memory::aligned_free(freqs);
    scl::memory::aligned_free(sizes);
}

// =============================================================================
// Clone Dynamics (expansion rates)
// =============================================================================

void clone_dynamics(
    Array<const Index> clone_ids_t1,
    Array<const Index> clone_ids_t2,
    Real* expansion_rates,
    Size& n_clones,
    Size max_clones
) {
    n_clones = 0;
    if (max_clones == 0) return;

    // Find max clone ID across both timepoints
    Index max_clone = -1;
    for (Size i = 0; i < clone_ids_t1.len; ++i) {
        if (clone_ids_t1.ptr[i] > max_clone) max_clone = clone_ids_t1.ptr[i];
    }
    for (Size i = 0; i < clone_ids_t2.len; ++i) {
        if (clone_ids_t2.ptr[i] > max_clone) max_clone = clone_ids_t2.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible = std::min(max_clones, static_cast<Size>(max_clone + 1));

    // Count clone sizes at each timepoint
    auto sizes_t1_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    auto sizes_t2_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    Size* sizes_t1 = sizes_t1_ptr.release();
    Size* sizes_t2 = sizes_t2_ptr.release();

    for (Size c = 0; c < n_possible; ++c) {
        sizes_t1[c] = 0;
        sizes_t2[c] = 0;
    }

    for (Size i = 0; i < clone_ids_t1.len; ++i) {
        Index clone = clone_ids_t1.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_possible)) {
            ++sizes_t1[clone];
        }
    }

    for (Size i = 0; i < clone_ids_t2.len; ++i) {
        Index clone = clone_ids_t2.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_possible)) {
            ++sizes_t2[clone];
        }
    }

    // Compute expansion rates
    for (Size c = 0; c < n_possible; ++c) {
        if (sizes_t1[c] > 0 || sizes_t2[c] > 0) {
            // Log2 fold change
            Real s1 = static_cast<Real>(sizes_t1[c]) + Real(1.0);  // pseudocount
            Real s2 = static_cast<Real>(sizes_t2[c]) + Real(1.0);
            expansion_rates[n_clones] = std::log2(s2 / s1);
            ++n_clones;
        }
    }

    scl::memory::aligned_free(sizes_t1);
    scl::memory::aligned_free(sizes_t2);
}

// =============================================================================
// Shared Clonotypes
// =============================================================================

void shared_clonotypes(
    Array<const Index> clone_ids_sample1,
    Array<const Index> clone_ids_sample2,
    Index* shared_clones,
    Size& n_shared,
    Real& jaccard_index,
    Size max_shared
) {
    n_shared = 0;
    jaccard_index = Real(0.0);

    if (clone_ids_sample1.len == 0 && clone_ids_sample2.len == 0) {
        jaccard_index = Real(1.0);
        return;
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < clone_ids_sample1.len; ++i) {
        if (clone_ids_sample1.ptr[i] > max_clone) max_clone = clone_ids_sample1.ptr[i];
    }
    for (Size i = 0; i < clone_ids_sample2.len; ++i) {
        if (clone_ids_sample2.ptr[i] > max_clone) max_clone = clone_ids_sample2.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible = static_cast<Size>(max_clone + 1);

    // Track presence in each sample
    auto in_sample1_ptr = scl::memory::aligned_alloc<bool>(n_possible, SCL_ALIGNMENT);
    auto in_sample2_ptr = scl::memory::aligned_alloc<bool>(n_possible, SCL_ALIGNMENT);
    bool* in_sample1 = in_sample1_ptr.release();
    bool* in_sample2 = in_sample2_ptr.release();

    for (Size c = 0; c < n_possible; ++c) {
        in_sample1[c] = false;
        in_sample2[c] = false;
    }

    for (Size i = 0; i < clone_ids_sample1.len; ++i) {
        Index clone = clone_ids_sample1.ptr[i];
        if (clone >= 0) in_sample1[clone] = true;
    }

    for (Size i = 0; i < clone_ids_sample2.len; ++i) {
        Index clone = clone_ids_sample2.ptr[i];
        if (clone >= 0) in_sample2[clone] = true;
    }

    // Find shared and compute Jaccard
    Size n_union = 0;
    Size n_intersection = 0;

    for (Size c = 0; c < n_possible; ++c) {
        if (in_sample1[c] || in_sample2[c]) {
            ++n_union;
            if (in_sample1[c] && in_sample2[c]) {
                ++n_intersection;
                if (n_shared < max_shared) {
                    shared_clones[n_shared++] = static_cast<Index>(c);
                }
            }
        }
    }

    if (n_union > 0) {
        jaccard_index = static_cast<Real>(n_intersection) / static_cast<Real>(n_union);
    }

    scl::memory::aligned_free(in_sample1);
    scl::memory::aligned_free(in_sample2);
}

// =============================================================================
// Clone Phenotype (mean expression per clone)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void clone_phenotype(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> clone_ids,
    Real* clone_profiles,  // [n_clones * n_genes]
    Size& n_clones,
    Size max_clones
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == clone_ids.len, "Clone IDs must match cell count");

    n_clones = 0;
    if (n_cells == 0 || n_genes == 0 || max_clones == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n_cells; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    n_clones = std::min(max_clones, static_cast<Size>(max_clone + 1));

    // Initialize profiles and counts
    auto clone_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);
    Size* clone_sizes = clone_sizes_ptr.release();

    for (Size c = 0; c < n_clones; ++c) {
        clone_sizes[c] = 0;
        for (Size g = 0; g < n_genes; ++g) {
            clone_profiles[c * n_genes + g] = Real(0.0);
        }
    }

    // Accumulate expression per clone
    for (Size i = 0; i < n_cells; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone < 0 || clone >= static_cast<Index>(n_clones)) continue;

        ++clone_sizes[clone];

        const auto cell_idx = static_cast<Index>(i);
        auto primary_vals = expression.primary_values_unsafe(cell_idx);
        auto primary_idxs = expression.primary_indices_unsafe(cell_idx);
        Index primary_len = expression.primary_length_unsafe(cell_idx);

        for (Index j = 0; j < primary_len; ++j) {
            Index gene = primary_idxs.ptr[j];
            clone_profiles[clone * n_genes + gene] +=
                static_cast<Real>(primary_vals.ptr[j]);
        }
    }

    // Normalize by clone size
    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] > 0) {
            Real inv_size = Real(1.0) / static_cast<Real>(clone_sizes[c]);
            for (Size g = 0; g < n_genes; ++g) {
                clone_profiles[c * n_genes + g] *= inv_size;
            }
        }
    }

    scl::memory::aligned_free(clone_sizes);
}

// =============================================================================
// Clonality Score (per cluster)
// =============================================================================

void clonality_score(
    Array<const Index> clone_ids,
    Array<const Index> cluster_labels,
    Array<Real> clonality_per_cluster
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cluster_labels.len, "Clone IDs and cluster labels must match");

    if (n == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n; ++i) {
        if (cluster_labels.ptr[i] >= n_clusters) {
            n_clusters = cluster_labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(clonality_per_cluster.len >= static_cast<Size>(n_clusters),
        "Clonality array too small");

    // For each cluster, compute clonality
    for (Index c = 0; c < n_clusters; ++c) {
        clonality_per_cluster.ptr[c] = Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible_clones = static_cast<Size>(max_clone + 1);

    // Count per cluster
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    auto clone_counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters * n_possible_clones, SCL_ALIGNMENT);
    Size* cluster_sizes = cluster_sizes_ptr.release();
    Size* clone_counts = clone_counts_ptr.release();

    for (Index c = 0; c < n_clusters; ++c) {
        cluster_sizes[c] = 0;
    }
    for (Size i = 0; i < static_cast<Size>(n_clusters) * n_possible_clones; ++i) {
        clone_counts[i] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index cluster = cluster_labels.ptr[i];
        Index clone = clone_ids.ptr[i];
        if (cluster >= 0 && cluster < n_clusters) {
            ++cluster_sizes[cluster];
            if (clone >= 0 && clone < static_cast<Index>(n_possible_clones)) {
                ++clone_counts[cluster * n_possible_clones + clone];
            }
        }
    }

    // Compute clonality (1 - normalized entropy)
    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] == 0) continue;

        Real entropy = Real(0.0);
        Size n_clones_in_cluster = 0;

        for (Size cl = 0; cl < n_possible_clones; ++cl) {
            Size count = clone_counts[c * n_possible_clones + cl];
            if (count > 0) {
                Real p = static_cast<Real>(count) / static_cast<Real>(cluster_sizes[c]);
                entropy -= p * std::log(p);
                ++n_clones_in_cluster;
            }
        }

        // Normalize by max entropy
        if (n_clones_in_cluster > 1) {
            Real max_entropy = std::log(static_cast<Real>(n_clones_in_cluster));
            clonality_per_cluster.ptr[c] = Real(1.0) - (entropy / max_entropy);
        } else {
            clonality_per_cluster.ptr[c] = Real(1.0);
        }
    }

    scl::memory::aligned_free(cluster_sizes);
    scl::memory::aligned_free(clone_counts);
}

// =============================================================================
// Repertoire Overlap (Morisita-Horn index)
// =============================================================================

Real repertoire_overlap_morisita(
    Array<const Index> clone_ids_1,
    Array<const Index> clone_ids_2
) {
    if (clone_ids_1.len == 0 || clone_ids_2.len == 0) {
        return Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < clone_ids_1.len; ++i) {
        if (clone_ids_1.ptr[i] > max_clone) max_clone = clone_ids_1.ptr[i];
    }
    for (Size i = 0; i < clone_ids_2.len; ++i) {
        if (clone_ids_2.ptr[i] > max_clone) max_clone = clone_ids_2.ptr[i];
    }

    if (max_clone < 0) return Real(0.0);

    Size n_possible = static_cast<Size>(max_clone + 1);

    // Count clone sizes
    auto sizes_1_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    auto sizes_2_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    Size* sizes_1 = sizes_1_ptr.release();
    Size* sizes_2 = sizes_2_ptr.release();

    for (Size c = 0; c < n_possible; ++c) {
        sizes_1[c] = 0;
        sizes_2[c] = 0;
    }

    Size total_1 = 0, total_2 = 0;

    for (Size i = 0; i < clone_ids_1.len; ++i) {
        Index clone = clone_ids_1.ptr[i];
        if (clone >= 0) {
            ++sizes_1[clone];
            ++total_1;
        }
    }

    for (Size i = 0; i < clone_ids_2.len; ++i) {
        Index clone = clone_ids_2.ptr[i];
        if (clone >= 0) {
            ++sizes_2[clone];
            ++total_2;
        }
    }

    if (total_1 == 0 || total_2 == 0) {
        scl::memory::aligned_free(sizes_1);
        scl::memory::aligned_free(sizes_2);
        return Real(0.0);
    }

    // Compute Morisita-Horn index
    Real sum_p1p2 = Real(0.0);
    Real sum_p1_sq = Real(0.0);
    Real sum_p2_sq = Real(0.0);

    for (Size c = 0; c < n_possible; ++c) {
        Real p1 = static_cast<Real>(sizes_1[c]) / static_cast<Real>(total_1);
        Real p2 = static_cast<Real>(sizes_2[c]) / static_cast<Real>(total_2);
        sum_p1p2 += p1 * p2;
        sum_p1_sq += p1 * p1;
        sum_p2_sq += p2 * p2;
    }

    scl::memory::aligned_free(sizes_1);
    scl::memory::aligned_free(sizes_2);

    Real denom = sum_p1_sq + sum_p2_sq;
    if (denom < config::EPSILON) return Real(0.0);

    return Real(2.0) * sum_p1p2 / denom;
}

// =============================================================================
// Diversity Per Cluster
// =============================================================================

void diversity_per_cluster(
    Array<const Index> clone_ids,
    Array<const Index> cluster_labels,
    Array<Real> shannon_per_cluster,
    Array<Real> simpson_per_cluster
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cluster_labels.len, "Clone IDs and cluster labels must match");

    if (n == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n; ++i) {
        if (cluster_labels.ptr[i] >= n_clusters) {
            n_clusters = cluster_labels.ptr[i] + 1;
        }
    }

    SCL_CHECK_DIM(shannon_per_cluster.len >= static_cast<Size>(n_clusters),
        "Shannon array too small");
    SCL_CHECK_DIM(simpson_per_cluster.len >= static_cast<Size>(n_clusters),
        "Simpson array too small");

    // For each cluster, compute diversity
    for (Index c = 0; c < n_clusters; ++c) {
        // Collect clone IDs for this cluster
        Size cluster_size = 0;
        for (Size i = 0; i < n; ++i) {
            if (cluster_labels.ptr[i] == c) ++cluster_size;
        }

        if (cluster_size == 0) {
            shannon_per_cluster.ptr[c] = Real(0.0);
            simpson_per_cluster.ptr[c] = Real(0.0);
            continue;
        }

        auto cluster_clones_ptr = scl::memory::aligned_alloc<Index>(cluster_size, SCL_ALIGNMENT);
        Index* cluster_clones = cluster_clones_ptr.release();
        Size idx = 0;
        for (Size i = 0; i < n; ++i) {
            if (cluster_labels.ptr[i] == c) {
                cluster_clones[idx++] = clone_ids.ptr[i];
            }
        }

        Array<const Index> cluster_arr = {cluster_clones, cluster_size};
        Real shannon = Real(0.0);
        Real simpson = Real(0.0);
        Real gini = Real(0.0);
        clonal_diversity(cluster_arr, shannon, simpson, gini);

        shannon_per_cluster.ptr[c] = shannon;
        simpson_per_cluster.ptr[c] = simpson;

        scl::memory::aligned_free(cluster_clones);
    }
}

// =============================================================================
// Clone Transition Matrix (between clusters)
// =============================================================================

void clone_transition_matrix(
    Array<const Index> clone_ids,
    Array<const Index> cluster_labels,
    Real* transition_matrix,  // [n_clusters * n_clusters]
    Size n_clusters
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cluster_labels.len, "Clone IDs and cluster labels must match");

    if (n == 0 || n_clusters == 0) return;

    // Initialize matrix
    for (Size i = 0; i < n_clusters * n_clusters; ++i) {
        transition_matrix[i] = Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible_clones = static_cast<Size>(max_clone + 1);

    // For each clone, find which clusters its cells belong to
    auto cluster_count_per_clone_ptr = scl::memory::aligned_alloc<Size>(
        n_possible_clones * n_clusters, SCL_ALIGNMENT);
    Size* cluster_count_per_clone = cluster_count_per_clone_ptr.release();

    for (Size i = 0; i < n_possible_clones * n_clusters; ++i) {
        cluster_count_per_clone[i] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index cluster = cluster_labels.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_possible_clones) &&
            cluster >= 0 && cluster < static_cast<Index>(n_clusters)) {
            ++cluster_count_per_clone[clone * n_clusters + cluster];
        }
    }

    // Compute transitions: for each clone present in multiple clusters
    for (Size cl = 0; cl < n_possible_clones; ++cl) {
        Size total = 0;
        for (Size c = 0; c < n_clusters; ++c) {
            total += cluster_count_per_clone[cl * n_clusters + c];
        }

        if (total < config::MIN_CLONE_SIZE) continue;

        // Add contribution to transition matrix
        for (Size c1 = 0; c1 < n_clusters; ++c1) {
            Real p1 = static_cast<Real>(cluster_count_per_clone[cl * n_clusters + c1]) /
                     static_cast<Real>(total);
            if (p1 < config::EPSILON) continue;

            for (Size c2 = 0; c2 < n_clusters; ++c2) {
                Real p2 = static_cast<Real>(cluster_count_per_clone[cl * n_clusters + c2]) /
                         static_cast<Real>(total);
                transition_matrix[c1 * n_clusters + c2] += p1 * p2;
            }
        }
    }

    // Normalize rows
    for (Size c1 = 0; c1 < n_clusters; ++c1) {
        Real row_sum = Real(0.0);
        for (Size c2 = 0; c2 < n_clusters; ++c2) {
            row_sum += transition_matrix[c1 * n_clusters + c2];
        }
        if (row_sum > config::EPSILON) {
            for (Size c2 = 0; c2 < n_clusters; ++c2) {
                transition_matrix[c1 * n_clusters + c2] /= row_sum;
            }
        }
    }

    scl::memory::aligned_free(cluster_count_per_clone);
}

// =============================================================================
// Rarefaction Diversity
// =============================================================================

void rarefaction_diversity(
    Array<const Index> clone_ids,
    Size subsample_size,
    Size n_iterations,
    Real& mean_diversity,
    Real& std_diversity,
    uint64_t seed
) {
    mean_diversity = Real(0.0);
    std_diversity = Real(0.0);

    const Size n = clone_ids.len;
    if (n == 0 || subsample_size == 0 || n_iterations == 0) return;

    subsample_size = std::min(subsample_size, n);

    // Simple LCG for random sampling
    uint64_t state = seed;
    auto lcg_next = [&]() -> uint64_t {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    };

    auto diversities_ptr = scl::memory::aligned_alloc<Real>(n_iterations, SCL_ALIGNMENT);
    auto subsample_ptr = scl::memory::aligned_alloc<Index>(subsample_size, SCL_ALIGNMENT);
    auto indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* diversities = diversities_ptr.release();
    Index* subsample = subsample_ptr.release();
    Index* indices = indices_ptr.release();

    for (Size iter = 0; iter < n_iterations; ++iter) {
        // Shuffle and take subsample
        for (Size i = 0; i < n; ++i) {
            indices[i] = static_cast<Index>(i);
        }
        for (Size i = 0; i < subsample_size; ++i) {
            Size j = i + (lcg_next() % (n - i));
            std::swap(indices[i], indices[j]);
            subsample[i] = clone_ids.ptr[indices[i]];
        }

        // Compute diversity
        Array<const Index> subsample_arr = {subsample, subsample_size};
        Real shannon = Real(0.0);
        Real simpson = Real(0.0);
        Real gini = Real(0.0);
        clonal_diversity(subsample_arr, shannon, simpson, gini);

        diversities[iter] = shannon;
    }

    // Compute mean and std
    Real sum = Real(0.0);
    for (Size i = 0; i < n_iterations; ++i) {
        sum += diversities[i];
    }
    mean_diversity = sum / static_cast<Real>(n_iterations);

    Real var_sum = Real(0.0);
    for (Size i = 0; i < n_iterations; ++i) {
        Real diff = diversities[i] - mean_diversity;
        var_sum += diff * diff;
    }
    if (n_iterations > 1) {
        std_diversity = std::sqrt(var_sum / static_cast<Real>(n_iterations - 1));
    }

    scl::memory::aligned_free(diversities);
    scl::memory::aligned_free(subsample);
    scl::memory::aligned_free(indices);
}

// =============================================================================
// Clone Expansion Detection
// =============================================================================

void detect_expanded_clones(
    Array<const Index> clone_ids,
    Size expansion_threshold,
    Index* expanded_clones,
    Size& n_expanded,
    Size max_expanded
) {
    n_expanded = 0;
    const Size n = clone_ids.len;

    if (n == 0 || max_expanded == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible = static_cast<Size>(max_clone + 1);

    // Count clone sizes
    auto sizes_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    Size* sizes = sizes_ptr.release();
    scl::algo::zero(sizes, n_possible);

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0) ++sizes[clone];
    }

    // Find expanded clones
    for (Size c = 0; c < n_possible && n_expanded < max_expanded; ++c) {
        if (sizes[c] >= expansion_threshold) {
            expanded_clones[n_expanded++] = static_cast<Index>(c);
        }
    }

    scl::memory::aligned_free<Size>(sizes, SCL_ALIGNMENT);
}

// =============================================================================
// Clone Size Statistics
// =============================================================================

void clone_size_statistics(
    Array<const Index> clone_ids,
    Real& mean_size,
    Real& median_size,
    Real& max_size,
    Size& n_singletons,
    Size& n_clones
) {
    mean_size = Real(0.0);
    median_size = Real(0.0);
    max_size = Real(0.0);
    n_singletons = 0;
    n_clones = 0;

    const Size n = clone_ids.len;
    if (n == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_possible = static_cast<Size>(max_clone + 1);

    // Count clone sizes
    auto sizes_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    Size* sizes = sizes_ptr.release();
    scl::algo::zero(sizes, n_possible);

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0) ++sizes[clone];
    }

    // Collect non-empty sizes
    auto nonempty_sizes_ptr = scl::memory::aligned_alloc<Size>(n_possible, SCL_ALIGNMENT);
    Size* nonempty_sizes = nonempty_sizes_ptr.release();
    Size n_nonempty = 0;

    for (Size c = 0; c < n_possible; ++c) {
        if (sizes[c] > 0) {
            nonempty_sizes[n_nonempty++] = sizes[c];
            if (sizes[c] == 1) ++n_singletons;
        }
    }

    n_clones = n_nonempty;

    if (n_nonempty == 0) {
        scl::memory::aligned_free(sizes);
        scl::memory::aligned_free(nonempty_sizes);
        return;
    }

    // Compute statistics
    Real sum = Real(0.0);
    Size max_s = 0;
    for (Size i = 0; i < n_nonempty; ++i) {
        sum += static_cast<Real>(nonempty_sizes[i]);
        if (nonempty_sizes[i] > max_s) max_s = nonempty_sizes[i];
    }

    mean_size = sum / static_cast<Real>(n_nonempty);
    max_size = static_cast<Real>(max_s);

    // Median
    std::sort(nonempty_sizes, nonempty_sizes + n_nonempty);
    if (n_nonempty % 2 == 0) {
        const auto idx1 = static_cast<Size>(n_nonempty / 2 - 1);
        const auto idx2 = static_cast<Size>(n_nonempty / 2);
        median_size = (static_cast<Real>(nonempty_sizes[idx1]) +
                      static_cast<Real>(nonempty_sizes[idx2])) / Real(2.0);
    } else {
        const auto idx = static_cast<Size>(n_nonempty / 2);
        median_size = static_cast<Real>(nonempty_sizes[idx]);
    }

    scl::memory::aligned_free(sizes);
    scl::memory::aligned_free(nonempty_sizes);
}

} // namespace scl::kernel::clonotype
