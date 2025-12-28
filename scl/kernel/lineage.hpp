#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"

#include <algorithm>
#include <cmath>

// =============================================================================
// FILE: scl/kernel/lineage.hpp
// BRIEF: Lineage tracing and fate mapping
//
// APPLICATIONS:
// - Developmental lineage reconstruction
// - Fate bias quantification
// - Barcode analysis
// - Lineage coupling
// =============================================================================

namespace scl::kernel::lineage {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_PARENT = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}

namespace detail {

// Union-Find data structure for lineage clustering
struct UnionFind {
    Index* parent;
    Index* rank;
    Size n;

    UnionFind(Size size) : n(size) {
        parent = scl::memory::aligned_alloc<Index>(size, SCL_ALIGNMENT);
        rank = scl::memory::aligned_alloc<Index>(size, SCL_ALIGNMENT);
        for (Size i = 0; i < size; ++i) {
            parent[i] = static_cast<Index>(i);
            rank[i] = 0;
        }
    }

    ~UnionFind() {
        scl::memory::aligned_free(parent);
        scl::memory::aligned_free(rank);
    }

    Index find(Index x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unite(Index x, Index y) {
        Index px = find(x);
        Index py = find(y);
        if (px == py) return;

        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            ++rank[px];
        }
    }
};

// Simple hash for barcode comparison
SCL_FORCE_INLINE uint64_t hash_string(const char* str, Size len) {
    uint64_t h = 14695981039346656037ULL;
    for (Size i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(str[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace detail

// =============================================================================
// Lineage Coupling Matrix
// =============================================================================

void lineage_coupling(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Real* coupling_matrix,  // [n_clones * n_types]
    Size n_clones,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");

    if (n == 0 || n_clones == 0 || n_types == 0) return;

    // Initialize coupling matrix
    for (Size i = 0; i < n_clones * n_types; ++i) {
        coupling_matrix[i] = Real(0.0);
    }

    // Count cells per clone per type
    Size* counts = scl::memory::aligned_alloc<Size>(n_clones * n_types, SCL_ALIGNMENT);
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);

    for (Size i = 0; i < n_clones * n_types; ++i) counts[i] = 0;
    for (Size c = 0; c < n_clones; ++c) clone_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index type = cell_types.ptr[i];

        if (clone >= 0 && clone < static_cast<Index>(n_clones) &&
            type >= 0 && type < static_cast<Index>(n_types)) {
            ++counts[clone * n_types + type];
            ++clone_sizes[clone];
        }
    }

    // Compute coupling as normalized frequency
    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] > 0) {
            for (Size t = 0; t < n_types; ++t) {
                coupling_matrix[c * n_types + t] =
                    static_cast<Real>(counts[c * n_types + t]) /
                    static_cast<Real>(clone_sizes[c]);
            }
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(clone_sizes);
}

// =============================================================================
// Fate Bias Score
// =============================================================================

void fate_bias(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<Real> bias_scores,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");

    if (n == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_clones = static_cast<Size>(max_clone + 1);
    SCL_CHECK_DIM(bias_scores.len >= n_clones, "Bias scores array too small");

    // Count cells per clone per type
    Size* counts = scl::memory::aligned_alloc<Size>(n_clones * n_types, SCL_ALIGNMENT);
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);

    for (Size i = 0; i < n_clones * n_types; ++i) counts[i] = 0;
    for (Size c = 0; c < n_clones; ++c) clone_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index type = cell_types.ptr[i];

        if (clone >= 0 && clone < static_cast<Index>(n_clones) &&
            type >= 0 && type < static_cast<Index>(n_types)) {
            ++counts[clone * n_types + type];
            ++clone_sizes[clone];
        }
    }

    // Compute fate bias as 1 - entropy (normalized)
    Real max_entropy = std::log(static_cast<Real>(n_types));

    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] == 0) {
            bias_scores.ptr[c] = Real(0.0);
            continue;
        }

        Real entropy = Real(0.0);
        for (Size t = 0; t < n_types; ++t) {
            if (counts[c * n_types + t] > 0) {
                Real p = static_cast<Real>(counts[c * n_types + t]) /
                        static_cast<Real>(clone_sizes[c]);
                entropy -= p * std::log(p + config::EPSILON);
            }
        }

        // Bias = 1 - normalized entropy
        if (max_entropy > config::EPSILON) {
            bias_scores.ptr[c] = Real(1.0) - (entropy / max_entropy);
        } else {
            bias_scores.ptr[c] = Real(1.0);
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(clone_sizes);
}

// =============================================================================
// Lineage Tree Construction (parent array representation)
// =============================================================================

void build_lineage_tree(
    Array<const Index> clone_ids,
    Array<const Real> pseudotime,
    Index* parent,  // [n_clones], parent[i] = parent of clone i, -1 for root
    Size n_clones
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == pseudotime.len, "Clone IDs and pseudotime must have same length");

    if (n == 0 || n_clones == 0) return;

    // Initialize all as roots
    for (Size c = 0; c < n_clones; ++c) {
        parent[c] = config::NO_PARENT;
    }

    // Compute mean pseudotime per clone
    Real* mean_time = scl::memory::aligned_alloc<Real>(n_clones, SCL_ALIGNMENT);
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);

    for (Size c = 0; c < n_clones; ++c) {
        mean_time[c] = Real(0.0);
        clone_sizes[c] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_clones)) {
            mean_time[clone] += pseudotime.ptr[i];
            ++clone_sizes[clone];
        }
    }

    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] > 0) {
            mean_time[c] /= static_cast<Real>(clone_sizes[c]);
        }
    }

    // Sort clones by mean pseudotime
    Index* sorted = scl::memory::aligned_alloc<Index>(n_clones, SCL_ALIGNMENT);
    for (Size c = 0; c < n_clones; ++c) {
        sorted[c] = static_cast<Index>(c);
    }
    std::sort(sorted, sorted + n_clones, [&](Index a, Index b) {
        return mean_time[a] < mean_time[b];
    });

    // Build tree: each clone's parent is the closest earlier clone
    for (Size i = 1; i < n_clones; ++i) {
        Index clone = sorted[i];
        if (clone_sizes[clone] == 0) continue;

        // Find closest earlier clone
        Real min_time_diff = std::numeric_limits<Real>::max();
        Index best_parent = config::NO_PARENT;

        for (Size j = 0; j < i; ++j) {
            Index candidate = sorted[j];
            if (clone_sizes[candidate] == 0) continue;

            Real time_diff = mean_time[clone] - mean_time[candidate];
            if (time_diff > Real(0.0) && time_diff < min_time_diff) {
                min_time_diff = time_diff;
                best_parent = candidate;
            }
        }

        parent[clone] = best_parent;
    }

    scl::memory::aligned_free(mean_time);
    scl::memory::aligned_free(clone_sizes);
    scl::memory::aligned_free(sorted);
}

// =============================================================================
// Lineage Distance Matrix
// =============================================================================

void lineage_distance(
    const Index* parent,  // Parent array representation of tree
    Size n_clones,
    Real* distance_matrix  // [n_clones * n_clones]
) {
    if (n_clones == 0) return;

    // Initialize distances
    for (Size i = 0; i < n_clones * n_clones; ++i) {
        distance_matrix[i] = Real(0.0);
    }

    // Compute depth of each node
    Index* depth = scl::memory::aligned_alloc<Index>(n_clones, SCL_ALIGNMENT);
    for (Size c = 0; c < n_clones; ++c) {
        depth[c] = 0;
        Index current = static_cast<Index>(c);
        while (parent[current] != config::NO_PARENT) {
            ++depth[c];
            current = parent[current];
        }
    }

    // Compute pairwise distances (number of edges)
    for (Size i = 0; i < n_clones; ++i) {
        for (Size j = i + 1; j < n_clones; ++j) {
            // Find LCA (Lowest Common Ancestor)
            Index* path_i = scl::memory::aligned_alloc<Index>(depth[i] + 1, SCL_ALIGNMENT);
            Index* path_j = scl::memory::aligned_alloc<Index>(depth[j] + 1, SCL_ALIGNMENT);

            // Build paths to root
            Size len_i = 0, len_j = 0;
            Index curr = static_cast<Index>(i);
            while (curr != config::NO_PARENT) {
                path_i[len_i++] = curr;
                curr = parent[curr];
            }

            curr = static_cast<Index>(j);
            while (curr != config::NO_PARENT) {
                path_j[len_j++] = curr;
                curr = parent[curr];
            }

            // Find LCA by comparing paths from root
            Size dist = len_i + len_j;

            for (Size pi = 0; pi < len_i; ++pi) {
                for (Size pj = 0; pj < len_j; ++pj) {
                    if (path_i[pi] == path_j[pj]) {
                        Size d = pi + pj;
                        if (d < dist) dist = d;
                    }
                }
            }

            distance_matrix[i * n_clones + j] = static_cast<Real>(dist);
            distance_matrix[j * n_clones + i] = static_cast<Real>(dist);

            scl::memory::aligned_free(path_i);
            scl::memory::aligned_free(path_j);
        }
    }

    scl::memory::aligned_free(depth);
}

// =============================================================================
// Barcode Clone Assignment
// =============================================================================

void barcode_clone_assignment(
    const uint64_t* barcode_hashes,  // Hash of each cell's barcode
    Size n_cells,
    Index* clone_ids,
    Size& n_unique_clones
) {
    n_unique_clones = 0;

    if (n_cells == 0) return;

    // Sort by hash to group identical barcodes
    Index* sorted = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    for (Size i = 0; i < n_cells; ++i) {
        sorted[i] = static_cast<Index>(i);
    }

    std::sort(sorted, sorted + n_cells, [&](Index a, Index b) {
        return barcode_hashes[a] < barcode_hashes[b];
    });

    // Assign clone IDs
    Index current_clone = 0;
    clone_ids[sorted[0]] = current_clone;

    for (Size i = 1; i < n_cells; ++i) {
        if (barcode_hashes[sorted[i]] != barcode_hashes[sorted[i - 1]]) {
            ++current_clone;
        }
        clone_ids[sorted[i]] = current_clone;
    }

    n_unique_clones = static_cast<Size>(current_clone + 1);

    scl::memory::aligned_free(sorted);
}

// =============================================================================
// Clonal Fate Probability
// =============================================================================

void clonal_fate_probability(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Real* fate_probs,  // [n_clones * n_types]
    Size n_clones,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");

    if (n == 0 || n_clones == 0 || n_types == 0) return;

    // Initialize
    for (Size i = 0; i < n_clones * n_types; ++i) {
        fate_probs[i] = Real(0.0);
    }

    // Count
    Size* counts = scl::memory::aligned_alloc<Size>(n_clones * n_types, SCL_ALIGNMENT);
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);

    for (Size i = 0; i < n_clones * n_types; ++i) counts[i] = 0;
    for (Size c = 0; c < n_clones; ++c) clone_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index type = cell_types.ptr[i];

        if (clone >= 0 && clone < static_cast<Index>(n_clones) &&
            type >= 0 && type < static_cast<Index>(n_types)) {
            ++counts[clone * n_types + type];
            ++clone_sizes[clone];
        }
    }

    // Compute probabilities
    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] > 0) {
            for (Size t = 0; t < n_types; ++t) {
                fate_probs[c * n_types + t] =
                    static_cast<Real>(counts[c * n_types + t]) /
                    static_cast<Real>(clone_sizes[c]);
            }
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(clone_sizes);
}

// =============================================================================
// Fate Bias Per Cell Type
// =============================================================================

void fate_bias_per_type(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<Real> bias_per_type,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");
    SCL_CHECK_DIM(bias_per_type.len >= n_types, "Bias array too small");

    if (n == 0 || n_types == 0) return;

    // Initialize
    for (Size t = 0; t < n_types; ++t) {
        bias_per_type.ptr[t] = Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_clones = static_cast<Size>(max_clone + 1);

    // Count cells per type per clone
    Size* counts = scl::memory::aligned_alloc<Size>(n_types * n_clones, SCL_ALIGNMENT);
    Size* type_sizes = scl::memory::aligned_alloc<Size>(n_types, SCL_ALIGNMENT);

    for (Size i = 0; i < n_types * n_clones; ++i) counts[i] = 0;
    for (Size t = 0; t < n_types; ++t) type_sizes[t] = 0;

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index type = cell_types.ptr[i];

        if (clone >= 0 && clone < static_cast<Index>(n_clones) &&
            type >= 0 && type < static_cast<Index>(n_types)) {
            ++counts[type * n_clones + clone];
            ++type_sizes[type];
        }
    }

    // Compute bias for each type (entropy of clone distribution)
    Real max_entropy = std::log(static_cast<Real>(n_clones));

    for (Size t = 0; t < n_types; ++t) {
        if (type_sizes[t] == 0) continue;

        Real entropy = Real(0.0);
        for (Size c = 0; c < n_clones; ++c) {
            if (counts[t * n_clones + c] > 0) {
                Real p = static_cast<Real>(counts[t * n_clones + c]) /
                        static_cast<Real>(type_sizes[t]);
                entropy -= p * std::log(p + config::EPSILON);
            }
        }

        // High bias = few clones dominate
        if (max_entropy > config::EPSILON) {
            bias_per_type.ptr[t] = Real(1.0) - (entropy / max_entropy);
        } else {
            bias_per_type.ptr[t] = Real(0.0);
        }
    }

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(type_sizes);
}

// =============================================================================
// Lineage Sharing Between Types
// =============================================================================

void lineage_sharing(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Real* sharing_matrix,  // [n_types * n_types]
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");

    if (n == 0 || n_types == 0) return;

    // Initialize
    for (Size i = 0; i < n_types * n_types; ++i) {
        sharing_matrix[i] = Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_clones = static_cast<Size>(max_clone + 1);

    // Track which types each clone contributes to
    bool* clone_has_type = scl::memory::aligned_alloc<bool>(n_clones * n_types, SCL_ALIGNMENT);

    for (Size i = 0; i < n_clones * n_types; ++i) {
        clone_has_type[i] = false;
    }

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        Index type = cell_types.ptr[i];

        if (clone >= 0 && clone < static_cast<Index>(n_clones) &&
            type >= 0 && type < static_cast<Index>(n_types)) {
            clone_has_type[clone * n_types + type] = true;
        }
    }

    // Count shared clones between type pairs
    Size* n_shared = scl::memory::aligned_alloc<Size>(n_types * n_types, SCL_ALIGNMENT);
    Size* n_type_clones = scl::memory::aligned_alloc<Size>(n_types, SCL_ALIGNMENT);

    for (Size i = 0; i < n_types * n_types; ++i) n_shared[i] = 0;
    for (Size t = 0; t < n_types; ++t) n_type_clones[t] = 0;

    for (Size c = 0; c < n_clones; ++c) {
        for (Size t1 = 0; t1 < n_types; ++t1) {
            if (!clone_has_type[c * n_types + t1]) continue;
            ++n_type_clones[t1];

            for (Size t2 = t1; t2 < n_types; ++t2) {
                if (clone_has_type[c * n_types + t2]) {
                    ++n_shared[t1 * n_types + t2];
                    if (t1 != t2) {
                        ++n_shared[t2 * n_types + t1];
                    }
                }
            }
        }
    }

    // Compute Jaccard-like sharing index
    for (Size t1 = 0; t1 < n_types; ++t1) {
        for (Size t2 = 0; t2 < n_types; ++t2) {
            Size n_union = n_type_clones[t1] + n_type_clones[t2] - n_shared[t1 * n_types + t2];
            if (n_union > 0) {
                sharing_matrix[t1 * n_types + t2] =
                    static_cast<Real>(n_shared[t1 * n_types + t2]) /
                    static_cast<Real>(n_union);
            }
        }
    }

    scl::memory::aligned_free(clone_has_type);
    scl::memory::aligned_free(n_shared);
    scl::memory::aligned_free(n_type_clones);
}

// =============================================================================
// Lineage Commitment Score
// =============================================================================

void lineage_commitment(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<Real> commitment_scores,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");
    SCL_CHECK_DIM(n == commitment_scores.len, "Commitment scores must match cell count");

    if (n == 0 || n_types == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) {
        for (Size i = 0; i < n; ++i) {
            commitment_scores.ptr[i] = Real(0.0);
        }
        return;
    }

    Size n_clones = static_cast<Size>(max_clone + 1);

    // Compute fate bias per clone
    Real* clone_bias = scl::memory::aligned_alloc<Real>(n_clones, SCL_ALIGNMENT);
    Array<Real> bias_arr = {clone_bias, n_clones};
    fate_bias({clone_ids.ptr, n}, {cell_types.ptr, n}, bias_arr, n_types);

    // Assign commitment score to each cell based on its clone's bias
    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_clones)) {
            commitment_scores.ptr[i] = clone_bias[clone];
        } else {
            commitment_scores.ptr[i] = Real(0.0);
        }
    }

    scl::memory::aligned_free(clone_bias);
}

// =============================================================================
// Progenitor Score
// =============================================================================

void progenitor_score(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<const Real> pseudotime,
    Array<Real> progenitor_scores,
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");
    SCL_CHECK_DIM(n == pseudotime.len, "Pseudotime must match cell count");
    SCL_CHECK_DIM(n == progenitor_scores.len, "Scores must match cell count");

    if (n == 0) return;

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) {
        for (Size i = 0; i < n; ++i) {
            progenitor_scores.ptr[i] = Real(0.0);
        }
        return;
    }

    Size n_clones = static_cast<Size>(max_clone + 1);

    // Compute min pseudotime per clone
    Real* min_time = scl::memory::aligned_alloc<Real>(n_clones, SCL_ALIGNMENT);
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);

    for (Size c = 0; c < n_clones; ++c) {
        min_time[c] = std::numeric_limits<Real>::max();
        clone_sizes[c] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_clones)) {
            if (pseudotime.ptr[i] < min_time[clone]) {
                min_time[clone] = pseudotime.ptr[i];
            }
            ++clone_sizes[clone];
        }
    }

    // Progenitor score = how close cell is to clone's minimum time
    // normalized by clone size
    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_clones) && clone_sizes[clone] > 1) {
            Real time_diff = pseudotime.ptr[i] - min_time[clone];
            // Higher score for cells closer to min time
            progenitor_scores.ptr[i] = std::exp(-time_diff);
        } else {
            progenitor_scores.ptr[i] = Real(0.0);
        }
    }

    scl::memory::aligned_free(min_time);
    scl::memory::aligned_free(clone_sizes);
}

// =============================================================================
// Transition Probability (type to type via lineage)
// =============================================================================

void lineage_transition_probability(
    Array<const Index> clone_ids,
    Array<const Index> cell_types,
    Array<const Real> pseudotime,
    Real* transition_prob,  // [n_types * n_types]
    Size n_types
) {
    const Size n = clone_ids.len;
    SCL_CHECK_DIM(n == cell_types.len, "Clone IDs and cell types must have same length");
    SCL_CHECK_DIM(n == pseudotime.len, "Pseudotime must match cell count");

    if (n == 0 || n_types == 0) return;

    // Initialize
    for (Size i = 0; i < n_types * n_types; ++i) {
        transition_prob[i] = Real(0.0);
    }

    // Find max clone ID
    Index max_clone = -1;
    for (Size i = 0; i < n; ++i) {
        if (clone_ids.ptr[i] > max_clone) max_clone = clone_ids.ptr[i];
    }

    if (max_clone < 0) return;

    Size n_clones = static_cast<Size>(max_clone + 1);

    // For each clone, find transitions from early to late cells
    Size* transition_counts = scl::memory::aligned_alloc<Size>(n_types * n_types, SCL_ALIGNMENT);
    Size* type_counts = scl::memory::aligned_alloc<Size>(n_types, SCL_ALIGNMENT);

    for (Size i = 0; i < n_types * n_types; ++i) transition_counts[i] = 0;
    for (Size t = 0; t < n_types; ++t) type_counts[t] = 0;

    // Group cells by clone
    Size* clone_sizes = scl::memory::aligned_alloc<Size>(n_clones, SCL_ALIGNMENT);
    for (Size c = 0; c < n_clones; ++c) clone_sizes[c] = 0;

    for (Size i = 0; i < n; ++i) {
        Index clone = clone_ids.ptr[i];
        if (clone >= 0 && clone < static_cast<Index>(n_clones)) {
            ++clone_sizes[clone];
        }
    }

    // For each clone with at least 2 cells, count transitions
    Index* clone_cells = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    for (Size c = 0; c < n_clones; ++c) {
        if (clone_sizes[c] < 2) continue;

        // Collect cells in this clone
        Size idx = 0;
        for (Size i = 0; i < n; ++i) {
            if (clone_ids.ptr[i] == static_cast<Index>(c)) {
                clone_cells[idx++] = static_cast<Index>(i);
            }
        }

        // Sort by pseudotime
        std::sort(clone_cells, clone_cells + clone_sizes[c], [&](Index a, Index b) {
            return pseudotime.ptr[a] < pseudotime.ptr[b];
        });

        // Count transitions
        for (Size j = 0; j < clone_sizes[c] - 1; ++j) {
            Index cell1 = clone_cells[j];
            Index cell2 = clone_cells[j + 1];

            Index type1 = cell_types.ptr[cell1];
            Index type2 = cell_types.ptr[cell2];

            if (type1 >= 0 && type1 < static_cast<Index>(n_types) &&
                type2 >= 0 && type2 < static_cast<Index>(n_types)) {
                ++transition_counts[type1 * n_types + type2];
                ++type_counts[type1];
            }
        }
    }

    // Normalize to probabilities
    for (Size t1 = 0; t1 < n_types; ++t1) {
        if (type_counts[t1] > 0) {
            for (Size t2 = 0; t2 < n_types; ++t2) {
                transition_prob[t1 * n_types + t2] =
                    static_cast<Real>(transition_counts[t1 * n_types + t2]) /
                    static_cast<Real>(type_counts[t1]);
            }
        }
    }

    scl::memory::aligned_free(transition_counts);
    scl::memory::aligned_free(type_counts);
    scl::memory::aligned_free(clone_sizes);
    scl::memory::aligned_free(clone_cells);
}

// =============================================================================
// Clone Generation Assignment
// =============================================================================

void clone_generation(
    const Index* parent,  // Tree parent array
    Size n_clones,
    Index* generation  // Output: generation number for each clone
) {
    if (n_clones == 0) return;

    for (Size c = 0; c < n_clones; ++c) {
        generation[c] = 0;
        Index current = static_cast<Index>(c);
        while (parent[current] != config::NO_PARENT) {
            ++generation[c];
            current = parent[current];
        }
    }
}

} // namespace scl::kernel::lineage
