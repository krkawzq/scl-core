#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/tissue.hpp
// BRIEF: Tissue architecture and organization analysis
//
// APPLICATIONS:
// - Tissue structure quantification
// - Layer assignment
// - Zonation scoring
// - Morphological features
// - Tissue module detection
// =============================================================================

namespace scl::kernel::tissue {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_LAYER = 5;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real PI = Real(3.14159265358979323846);
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

// Euclidean distance
SCL_FORCE_INLINE Real euclidean_distance(
    const Real* coords1,
    const Real* coords2,
    Size n_dims
) {
    Real dist_sq = Real(0.0);
    for (Size d = 0; d < n_dims; ++d) {
        Real diff = coords1[d] - coords2[d];
        dist_sq += diff * diff;
    }
    return std::sqrt(dist_sq);
}

// Compute centroid
SCL_FORCE_INLINE void compute_centroid(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Real* centroid
) {
    for (Size d = 0; d < n_dims; ++d) {
        centroid[d] = Real(0.0);
    }

    for (Size i = 0; i < n_cells; ++i) {
        for (Size d = 0; d < n_dims; ++d) {
            centroid[d] += coordinates[i * n_dims + d];
        }
    }

    for (Size d = 0; d < n_dims; ++d) {
        centroid[d] /= static_cast<Real>(n_cells);
    }
}

// Compute convex hull area (2D only, simplified)
SCL_FORCE_INLINE Real compute_convex_hull_area_2d(
    const Real* coords_x,
    const Real* coords_y,
    Size n_points
) {
    if (n_points < 3) return Real(0.0);

    // Simple shoelace formula for approximate area
    Real area = Real(0.0);

    // Find convex hull using gift wrapping (simplified)
    auto hull_unique = scl::memory::aligned_alloc<Index>(n_points, SCL_ALIGNMENT);
    Index* hull = hull_unique.get();
    auto in_hull_unique = scl::memory::aligned_alloc<bool>(n_points, SCL_ALIGNMENT);
    bool* in_hull = in_hull_unique.get();


    for (Size i = 0; i < n_points; ++i) {
        in_hull[i] = false;
    }

    // Start with leftmost point
    Index start = 0;
    for (Size i = 1; i < n_points; ++i) {
        if (coords_x[i] < coords_x[start]) {
            start = static_cast<Index>(i);
        }
    }

    Size hull_size = 0;
    Index current = start;

    do {
        hull[hull_size++] = current;
        in_hull[current] = true;

        Index next = 0;
        for (Size i = 0; i < n_points; ++i) {
            if (static_cast<Index>(i) == current) continue;

            // Cross product to determine orientation
            Real cross = (coords_x[i] - coords_x[current]) *
                        (coords_y[next] - coords_y[current]) -
                        (coords_y[i] - coords_y[current]) *
                        (coords_x[next] - coords_x[current]);

            if (next == current || cross > Real(0.0)) {
                next = static_cast<Index>(i);
            }
        }

        current = next;
    } while (current != start && hull_size < n_points);

    // Compute area using shoelace formula
    for (Size i = 0; i < hull_size; ++i) {
        Size j = (i + 1) % hull_size;
        area += coords_x[hull[i]] * coords_y[hull[j]];
        area -= coords_x[hull[j]] * coords_y[hull[i]];
    }
    area = std::abs(area) / Real(2.0);

    scl::memory::aligned_free(hull);
    scl::memory::aligned_free(in_hull);

    return area;
}

// K-nearest neighbors
void find_knn(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Size k,
    Index* neighbors  // [n_cells * k]
) {
    k = scl::algo::min2(k, n_cells - 1);

    // Use unique_ptr for automatic RAII; extract raw pointer for usage
    auto distances_uptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* distances = distances_uptr.get();

    auto sorted_idx_uptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Index* sorted_idx = sorted_idx_uptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        for (Size j = 0; j < n_cells; ++j) {
            sorted_idx[j] = static_cast<Index>(j);
            distances[j] = euclidean_distance(
                coordinates + i * n_dims,
                coordinates + j * n_dims,
                n_dims);
        }

        scl::algo::partial_sort(sorted_idx, static_cast<Size>(n_cells), k + 1,
            [&](Index a, Index b) { return distances[a] < distances[b]; });

        for (Size ki = 0; ki < k; ++ki) {
            neighbors[i * k + ki] = sorted_idx[ki + 1];  // Skip self
        }
    }
    // aligned_free unnecessary (unique_ptr handles deallocation)
}

} // namespace detail

// =============================================================================
// Tissue Architecture Quantification
// =============================================================================

void tissue_architecture(
    const Real* coordinates,  // [n_cells * n_dims]
    Size n_cells,
    Size n_dims,
    Array<const Index> cell_types,
    Real* density,           // Output: local cell density per cell
    Real* heterogeneity,     // Output: local type heterogeneity per cell
    Real* clustering_coef,   // Output: clustering coefficient per cell
    Size n_neighbors
) {
    SCL_CHECK_DIM(n_cells == cell_types.len, "Cell types must match cell count");

    if (n_cells == 0) return;

    n_neighbors = scl::algo::min2(n_neighbors, n_cells - 1);

    // Find KNN
    auto neighbors_uptr = scl::memory::aligned_alloc<Index>(n_cells * n_neighbors, SCL_ALIGNMENT);
    Index* neighbors = neighbors_uptr.get();
    detail::find_knn(coordinates, n_cells, n_dims, n_neighbors, neighbors);

    // Find number of cell types
    Index n_types = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (cell_types.ptr[i] >= n_types) {
            n_types = cell_types.ptr[i] + 1;
        }
    }

    auto type_counts_uptr = scl::memory::aligned_alloc<Size>(n_types, SCL_ALIGNMENT);
    Size* type_counts = type_counts_uptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        // Compute local density (inverse of average distance to neighbors)
        Real avg_dist = Real(0.0);
        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];
            avg_dist += detail::euclidean_distance(
                coordinates + i * n_dims,
                coordinates + neighbor * n_dims,
                n_dims);
        }
        avg_dist /= static_cast<Real>(n_neighbors);
        density[i] = (avg_dist > config::EPSILON) ? Real(1.0) / avg_dist : Real(0.0);

        // Compute local heterogeneity (entropy of neighbor types)
        for (Index t = 0; t < n_types; ++t) {
            type_counts[t] = 0;
        }

        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];
            Index type = cell_types.ptr[neighbor];
            if (type >= 0 && type < n_types) {
                ++type_counts[type];
            }
        }

        Real entropy = Real(0.0);
        for (Index t = 0; t < n_types; ++t) {
            if (type_counts[t] > 0) {
                Real p = static_cast<Real>(type_counts[t]) /
                        static_cast<Real>(n_neighbors);
                entropy -= p * std::log(p);
            }
        }

        Real max_entropy = std::log(static_cast<Real>(n_types));
        heterogeneity[i] = (max_entropy > config::EPSILON) ?
            entropy / max_entropy : Real(0.0);

        // Compute clustering coefficient
        // (fraction of neighbor pairs that are also neighbors of each other)
        Size n_connected = 0;
        Size n_possible = (n_neighbors * (n_neighbors - 1)) / 2;

        for (Size k1 = 0; k1 < n_neighbors; ++k1) {
            Index n1 = neighbors[i * n_neighbors + k1];
            for (Size k2 = k1 + 1; k2 < n_neighbors; ++k2) {
                Index n2 = neighbors[i * n_neighbors + k2];

                // Check if n1 and n2 are neighbors
                for (Size k3 = 0; k3 < n_neighbors; ++k3) {
                    if (neighbors[n1 * n_neighbors + k3] == n2) {
                        ++n_connected;
                        break;
                    }
                }
            }
        }

        clustering_coef[i] = (n_possible > 0) ?
            static_cast<Real>(n_connected) / static_cast<Real>(n_possible) : Real(0.0);
    }
    // aligned_free unnecessary (unique_ptr handles deallocation)
}

// =============================================================================
// Layer Assignment
// =============================================================================

void layer_assignment(
    const Real* coordinates,  // [n_cells * n_dims]
    Size n_cells,
    Size n_dims,
    Index n_layers,
    Array<Index> layer_labels,
    Index reference_dim  // Dimension along which to define layers
) {
    SCL_CHECK_DIM(n_cells == layer_labels.len, "Layer labels must match cell count");

    if (n_cells == 0 || n_layers == 0) return;

    if (reference_dim >= static_cast<Index>(n_dims)) {
        reference_dim = 0;
    }

    // Extract reference coordinate
    auto ref_coords_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* ref_coords = ref_coords_ptr.get();
    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = std::numeric_limits<Real>::lowest();

    for (Size i = 0; i < n_cells; ++i) {
        ref_coords[i] = coordinates[i * n_dims + reference_dim];
        if (ref_coords[i] < min_val) min_val = ref_coords[i];
        if (ref_coords[i] > max_val) max_val = ref_coords[i];
    }

    Real range = max_val - min_val;

    if (range < config::EPSILON) {
        for (Size i = 0; i < n_cells; ++i) {
            layer_labels.ptr[i] = 0;
        }
        return;
    }

    // Assign layers based on quantiles
    for (Size i = 0; i < n_cells; ++i) {
        Real normalized = (ref_coords[i] - min_val) / range;
        auto layer = static_cast<Index>(normalized * static_cast<Real>(n_layers));
        layer = scl::algo::min2(layer, n_layers - 1);
        layer_labels.ptr[i] = layer;
    }
}

void radial_layer_assignment(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    const Real* center,  // [n_dims]
    Index n_layers,
    Array<Index> layer_labels
) {
    SCL_CHECK_DIM(n_cells == layer_labels.len, "Layer labels must match cell count");

    if (n_cells == 0 || n_layers == 0) return;

    // Compute distance from center for each cell
    auto distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* distances = distances_ptr.get();
    Real max_dist = Real(0.0);

    for (Size i = 0; i < n_cells; ++i) {
        distances[i] = detail::euclidean_distance(
            coordinates + i * n_dims, center, n_dims);
        if (distances[i] > max_dist) {
            max_dist = distances[i];
        }
    }

    if (max_dist < config::EPSILON) {
        for (Size i = 0; i < n_cells; ++i) {
            layer_labels.ptr[i] = 0;
        }
        return;
    }

    // Assign layers based on distance
    for (Size i = 0; i < n_cells; ++i) {
        Real normalized = distances[i] / max_dist;
        auto layer = static_cast<Index>(normalized * static_cast<Real>(n_layers));
        layer = scl::algo::min2(layer, n_layers - 1);
        layer_labels.ptr[i] = layer;
    }
}

template <typename T, bool IsCSR>
void zonation_score(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    const Real* reference_point,  // Central/reference point
    Array<Real> zonation_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == zonation_scores.len, "Zonation scores must match cell count");

    if (n_cells == 0 || n_genes == 0) return;

    // Compute distance from reference for each cell
    auto distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* distances = distances_ptr.get();
    Real max_dist = Real(0.0);

    for (Size i = 0; i < n_cells; ++i) {
        distances[i] = detail::euclidean_distance(
            coordinates + i * n_dims, reference_point, n_dims);
        if (distances[i] > max_dist) {
            max_dist = distances[i];
        }
    }

    // Compute zonation score as correlation with distance
    // Higher score = gene expression varies with distance from reference

    // For each cell, compute total expression
    auto total_expr_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* total_expr = total_expr_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        total_expr[i] = Real(0.0);
        auto row_values = expression.primary_values_unsafe(static_cast<Index>(i));
        const Size row_len = static_cast<Size>(expression.primary_length_unsafe(static_cast<Index>(i)));

        for (Size j = 0; j < row_len; ++j) {
            total_expr[i] += static_cast<Real>(row_values.ptr[j]);
        }
    }

    // Normalize distances
    if (max_dist > config::EPSILON) {
        for (Size i = 0; i < n_cells; ++i) {
            distances[i] /= max_dist;
        }
    }

    // Compute correlation between expression and distance
    Real mean_dist = Real(0.0), mean_expr = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        mean_dist += distances[i];
        mean_expr += total_expr[i];
    }
    mean_dist /= static_cast<Real>(n_cells);
    mean_expr /= static_cast<Real>(n_cells);

    Real cov = Real(0.0), var_dist = Real(0.0), var_expr = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        Real dd = distances[i] - mean_dist;
        Real de = total_expr[i] - mean_expr;
        cov += dd * de;
        var_dist += dd * dd;
        var_expr += de * de;
    }

    Real global_zonation = Real(0.0);
    if (var_dist > config::EPSILON && var_expr > config::EPSILON) {
        global_zonation = cov / std::sqrt(var_dist * var_expr);
    }

    // Assign per-cell zonation score based on local expression gradient
    for (Size i = 0; i < n_cells; ++i) {
        // Simple approach: use normalized distance weighted by global correlation
        zonation_scores.ptr[i] = distances[i] * std::abs(global_zonation);
    }
}


// =============================================================================
// Morphological Features
// =============================================================================

void morphological_features(
    const Real* coordinates,  // [n_cells * n_dims]
    Size n_cells,
    Size n_dims,
    Array<const Index> labels,
    Real* area,              // [n_groups]
    Real* perimeter,         // [n_groups]
    Real* circularity,       // [n_groups]
    Real* eccentricity,      // [n_groups]
    Size n_groups
) {
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");

    if (n_cells == 0 || n_groups == 0 || n_dims < 2) return;

    // Initialize outputs
    for (Size g = 0; g < n_groups; ++g) {
        area[g] = Real(0.0);
        perimeter[g] = Real(0.0);
        circularity[g] = Real(0.0);
        eccentricity[g] = Real(0.0);
    }

    // Count cells per group
    auto group_sizes_ptr = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
    Size* group_sizes = group_sizes_ptr.get();
    for (Size g = 0; g < n_groups; ++g) {
        group_sizes[g] = 0;
    }

    for (Size i = 0; i < n_cells; ++i) {
        Index label = labels.ptr[i];
        if (label >= 0 && label < static_cast<Index>(n_groups)) {
            ++group_sizes[label];
        }
    }

    // Process each group
    for (Size g = 0; g < n_groups; ++g) {
        if (group_sizes[g] < 3) continue;

        // Collect coordinates for this group
        auto group_x_ptr = scl::memory::aligned_alloc<Real>(group_sizes[g], SCL_ALIGNMENT);
        Real* group_x = group_x_ptr.release();
        auto group_y_ptr = scl::memory::aligned_alloc<Real>(group_sizes[g], SCL_ALIGNMENT);
        Real* group_y = group_y_ptr.release();

        Size idx = 0;
        for (Size i = 0; i < n_cells; ++i) {
            if (labels.ptr[i] == static_cast<Index>(g)) {
                group_x[idx] = coordinates[i * n_dims + 0];
                group_y[idx] = coordinates[i * n_dims + 1];
                ++idx;
            }
        }

        // Compute area (convex hull area)
        area[g] = detail::compute_convex_hull_area_2d(group_x, group_y, group_sizes[g]);

        // Compute centroid
        Real cx = Real(0.0), cy = Real(0.0);
        for (Size i = 0; i < group_sizes[g]; ++i) {
            cx += group_x[i];
            cy += group_y[i];
        }
        cx /= static_cast<Real>(group_sizes[g]);
        cy /= static_cast<Real>(group_sizes[g]);

        // Compute radius statistics
        Real max_radius = Real(0.0);

        for (Size i = 0; i < group_sizes[g]; ++i) {
            Real dx = group_x[i] - cx;
            Real dy = group_y[i] - cy;
            Real r = std::sqrt(dx * dx + dy * dy);
            if (r > max_radius) max_radius = r;
        }

        // Perimeter approximation (2 * pi * max_radius)
        perimeter[g] = Real(2.0) * config::PI * max_radius;

        // Circularity = 4 * pi * area / perimeter^2
        if (perimeter[g] > config::EPSILON) {
            circularity[g] = Real(4.0) * config::PI * area[g] /
                (perimeter[g] * perimeter[g]);
        }

        // Eccentricity from second moments
        Real Ixx = Real(0.0), Iyy = Real(0.0), Ixy = Real(0.0);
        for (Size i = 0; i < group_sizes[g]; ++i) {
            Real dx = group_x[i] - cx;
            Real dy = group_y[i] - cy;
            Ixx += dx * dx;
            Iyy += dy * dy;
            Ixy += dx * dy;
        }

        // Principal moments
        Real trace = Ixx + Iyy;
        Real det = Ixx * Iyy - Ixy * Ixy;
        Real discriminant = trace * trace - Real(4.0) * det;

        if (discriminant > Real(0.0)) {
            Real lambda1 = (trace + std::sqrt(discriminant)) / Real(2.0);
            Real lambda2 = (trace - std::sqrt(discriminant)) / Real(2.0);

            if (lambda1 > config::EPSILON && lambda2 > Real(0.0)) {
                eccentricity[g] = std::sqrt(Real(1.0) - lambda2 / lambda1);
            }
        }

        scl::memory::aligned_free(group_x);
        scl::memory::aligned_free(group_y);
    }

    scl::memory::aligned_free(group_sizes);
}

// =============================================================================
// Tissue Module Detection
// =============================================================================

template <typename T, bool IsCSR>
void tissue_module(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Index n_modules,
    Array<Index> module_labels,
    uint64_t seed
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == module_labels.len, "Module labels must match cell count");

    if (n_cells == 0 || n_modules == 0) return;

    n_modules = scl::algo::min2(static_cast<Size>(n_modules), n_cells);

    // Combined spatial-expression features
    Size n_expr_features = scl::algo::min2(n_genes, Size(20));
    Size n_features = n_dims + n_expr_features;

    auto features_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_features, SCL_ALIGNMENT);
    Real* features = features_ptr.release();

    // Normalize and add spatial coordinates
    auto coord_min_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    Real* coord_min = coord_min_ptr.release();
    auto coord_max_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    Real* coord_max = coord_max_ptr.release();

    for (Size d = 0; d < n_dims; ++d) {
        coord_min[d] = std::numeric_limits<Real>::max();
        coord_max[d] = std::numeric_limits<Real>::lowest();
    }

    for (Size i = 0; i < n_cells; ++i) {
        for (Size d = 0; d < n_dims; ++d) {
            Real val = coordinates[i * n_dims + d];
            if (val < coord_min[d]) coord_min[d] = val;
            if (val > coord_max[d]) coord_max[d] = val;
        }
    }

    for (Size i = 0; i < n_cells; ++i) {
        for (Size d = 0; d < n_dims; ++d) {
            Real range = coord_max[d] - coord_min[d];
            if (range > config::EPSILON) {
                features[i * n_features + d] =
                    (coordinates[i * n_dims + d] - coord_min[d]) / range;
            } else {
                features[i * n_features + d] = Real(0.5);
            }
        }
    }

    // Compute gene variances for feature selection
    auto gene_mean_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* gene_mean = gene_mean_ptr.release();
    auto gene_var_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* gene_var = gene_var_ptr.release();

    for (Size g = 0; g < n_genes; ++g) {
        gene_mean[g] = Real(0.0);
        gene_var[g] = Real(0.0);
    }

    for (Size i = 0; i < n_cells; ++i) {
        auto row_indices = expression.primary_indices_unsafe(static_cast<Index>(i));
        auto row_values = expression.primary_values_unsafe(static_cast<Index>(i));
        const Size row_len = static_cast<Size>(expression.primary_length_unsafe(static_cast<Index>(i)));

        for (Size j = 0; j < row_len; ++j) {
            gene_mean[row_indices.ptr[j]] +=
                static_cast<Real>(row_values.ptr[j]);
        }
    }

    for (Size g = 0; g < n_genes; ++g) {
        gene_mean[g] /= static_cast<Real>(n_cells);
    }

    for (Size i = 0; i < n_cells; ++i) {
        auto row_indices = expression.primary_indices_unsafe(static_cast<Index>(i));
        auto row_values = expression.primary_values_unsafe(static_cast<Index>(i));
        const Size row_len = static_cast<Size>(expression.primary_length_unsafe(static_cast<Index>(i)));

        for (Size j = 0; j < row_len; ++j) {
            Real diff = static_cast<Real>(row_values.ptr[j]) -
                       gene_mean[row_indices.ptr[j]];
            gene_var[row_indices.ptr[j]] += diff * diff;
        }
    }

    // Select top variable genes
    auto top_genes_ptr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
    Index* top_genes = top_genes_ptr.release();
    for (Size g = 0; g < n_genes; ++g) {
        top_genes[g] = static_cast<Index>(g);
    }

    scl::algo::partial_sort(top_genes, static_cast<Size>(n_genes), n_expr_features,
        [&](Index a, Index b) { return gene_var[a] > gene_var[b]; });

    // Add normalized expression features
    for (Size i = 0; i < n_cells; ++i) {
        for (Size f = 0; f < n_expr_features; ++f) {
            features[i * n_features + n_dims + f] = Real(0.0);
        }

        auto row_indices = expression.primary_indices_unsafe(static_cast<Index>(i));
        auto row_values = expression.primary_values_unsafe(static_cast<Index>(i));
        const Size row_len = static_cast<Size>(expression.primary_length_unsafe(static_cast<Index>(i)));

        for (Size j = 0; j < row_len; ++j) {
            Index gene = row_indices.ptr[j];

            for (Size f = 0; f < n_expr_features; ++f) {
                if (top_genes[f] == gene) {
                    Real val = static_cast<Real>(row_values.ptr[j]);
                    Real std_dev = std::sqrt(gene_var[gene] / static_cast<Real>(n_cells));
                    if (std_dev > config::EPSILON) {
                        features[i * n_features + n_dims + f] =
                            (val - gene_mean[gene]) / std_dev;
                    }
                    break;
                }
            }
        }
    }

    // K-means clustering
    auto centroids_ptr = scl::memory::aligned_alloc<Real>(n_modules * n_features, SCL_ALIGNMENT);
    Real* centroids = centroids_ptr.release();
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_modules, SCL_ALIGNMENT);
    Size* cluster_sizes = cluster_sizes_ptr.release();

    detail::LCG rng(seed);

    // Initialize centroids
    for (Size c = 0; c < static_cast<Size>(n_modules); ++c) {
        Size idx = rng.next_index(n_cells);
        for (Size f = 0; f < n_features; ++f) {
            centroids[c * n_features + f] = features[idx * n_features + f];
        }
    }

    // K-means iterations
    for (Size iter = 0; iter < config::MAX_ITERATIONS; ++iter) {
        bool changed = false;

        for (Size i = 0; i < n_cells; ++i) {
            Real min_dist = std::numeric_limits<Real>::max();
            Index best = 0;

            for (Size c = 0; c < static_cast<Size>(n_modules); ++c) {
                Real dist = Real(0.0);
                for (Size f = 0; f < n_features; ++f) {
                    Real diff = features[i * n_features + f] -
                               centroids[c * n_features + f];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best = static_cast<Index>(c);
                }
            }

            if (module_labels.ptr[i] != best) {
                changed = true;
                module_labels.ptr[i] = best;
            }
        }

        if (!changed) break;

        // Update centroids
        for (Size c = 0; c < static_cast<Size>(n_modules); ++c) {
            cluster_sizes[c] = 0;
            for (Size f = 0; f < n_features; ++f) {
                centroids[c * n_features + f] = Real(0.0);
            }
        }

        for (Size i = 0; i < n_cells; ++i) {
            Index cluster = module_labels.ptr[i];
            ++cluster_sizes[cluster];
            for (Size f = 0; f < n_features; ++f) {
                centroids[cluster * n_features + f] += features[i * n_features + f];
            }
        }

        for (Size c = 0; c < static_cast<Size>(n_modules); ++c) {
            if (cluster_sizes[c] > 0) {
                for (Size f = 0; f < n_features; ++f) {
                    centroids[c * n_features + f] /= static_cast<Real>(cluster_sizes[c]);
                }
            }
        }
    }

    scl::memory::aligned_free(features);
    scl::memory::aligned_free(coord_min);
    scl::memory::aligned_free(coord_max);
    scl::memory::aligned_free(gene_mean);
    scl::memory::aligned_free(gene_var);
    scl::memory::aligned_free(top_genes);
    scl::memory::aligned_free(centroids);
    scl::memory::aligned_free(cluster_sizes);
}

// =============================================================================
// Cell Type Neighborhood Composition
// =============================================================================

void neighborhood_composition(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Array<const Index> cell_types,
    Size n_types,
    Real* composition_matrix,  // [n_cells * n_types]
    Size n_neighbors
) {
    SCL_CHECK_DIM(n_cells == cell_types.len, "Cell types must match cell count");

    if (n_cells == 0 || n_types == 0) return;

    n_neighbors = scl::algo::min2(n_neighbors, n_cells - 1);

    // Find KNN
    auto neighbors_ptr = scl::memory::aligned_alloc<Index>(n_cells * n_neighbors, SCL_ALIGNMENT);
    Index* neighbors = neighbors_ptr.release();
    detail::find_knn(coordinates, n_cells, n_dims, n_neighbors, neighbors);

    // Compute composition for each cell
    for (Size i = 0; i < n_cells; ++i) {
        // Initialize
        for (Size t = 0; t < n_types; ++t) {
            composition_matrix[i * n_types + t] = Real(0.0);
        }

        // Count neighbor types
        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];
            Index type = cell_types.ptr[neighbor];
            if (type >= 0 && type < static_cast<Index>(n_types)) {
                composition_matrix[i * n_types + type] += Real(1.0);
            }
        }

        // Normalize to proportions
        Real total = Real(0.0);
        for (Size t = 0; t < n_types; ++t) {
            total += composition_matrix[i * n_types + t];
        }
        if (total > config::EPSILON) {
            for (Size t = 0; t < n_types; ++t) {
                composition_matrix[i * n_types + t] /= total;
            }
        }
    }

    scl::memory::aligned_free(neighbors);
}

// =============================================================================
// Cell Type Interaction Score
// =============================================================================

void cell_type_interaction(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Array<const Index> cell_types,
    Size n_types,
    Real* interaction_matrix,  // [n_types * n_types]
    Real interaction_radius
) {
    SCL_CHECK_DIM(n_cells == cell_types.len, "Cell types must match cell count");

    if (n_cells == 0 || n_types == 0) return;

    // Initialize
    for (Size i = 0; i < n_types * n_types; ++i) {
        interaction_matrix[i] = Real(0.0);
    }

    // Count type pairs
    auto type_counts_ptr = scl::memory::aligned_alloc<Size>(n_types, SCL_ALIGNMENT);
    Size* type_counts = type_counts_ptr.release();
    for (Size t = 0; t < n_types; ++t) {
        type_counts[t] = 0;
    }

    for (Size i = 0; i < n_cells; ++i) {
        Index type = cell_types.ptr[i];
        if (type >= 0 && type < static_cast<Index>(n_types)) {
            ++type_counts[type];
        }
    }

    // Count interactions
    auto interaction_counts_ptr = scl::memory::aligned_alloc<Size>(n_types * n_types, SCL_ALIGNMENT);
    Size* interaction_counts = interaction_counts_ptr.release();
    for (Size i = 0; i < n_types * n_types; ++i) {
        interaction_counts[i] = 0;
    }

    for (Size i = 0; i < n_cells; ++i) {
        Index type_i = cell_types.ptr[i];
        if (type_i < 0 || type_i >= static_cast<Index>(n_types)) continue;

        for (Size j = i + 1; j < n_cells; ++j) {
            Index type_j = cell_types.ptr[j];
            if (type_j < 0 || type_j >= static_cast<Index>(n_types)) continue;

            Real dist = detail::euclidean_distance(
                coordinates + i * n_dims,
                coordinates + j * n_dims,
                n_dims);

            if (dist <= interaction_radius) {
                ++interaction_counts[type_i * n_types + type_j];
                ++interaction_counts[type_j * n_types + type_i];
            }
        }
    }

    // Compute interaction score (observed / expected)
    Size total_cells = 0;
    for (Size t = 0; t < n_types; ++t) {
        total_cells += type_counts[t];
    }

    if (total_cells > 1) {
        for (Size t1 = 0; t1 < n_types; ++t1) {
            for (Size t2 = 0; t2 < n_types; ++t2) {
                Real expected = static_cast<Real>(type_counts[t1] * type_counts[t2]) /
                               static_cast<Real>(total_cells * total_cells);

                if (expected > config::EPSILON) {
                    Real observed = static_cast<Real>(interaction_counts[t1 * n_types + t2]) /
                                   static_cast<Real>(n_cells * n_cells);
                    interaction_matrix[t1 * n_types + t2] = std::log2(
                        (observed + config::EPSILON) / (expected + config::EPSILON));
                }
            }
        }
    }

    scl::memory::aligned_free(type_counts);
    scl::memory::aligned_free(interaction_counts);
}

// =============================================================================
// Boundary Cells Detection
// =============================================================================

void boundary_cells(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Array<const Index> labels,
    Array<bool> is_boundary,
    Size n_neighbors
) {
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");
    SCL_CHECK_DIM(n_cells == is_boundary.len, "Boundary flags must match cell count");

    if (n_cells == 0) return;

    n_neighbors = scl::algo::min2(n_neighbors, n_cells - 1);

    // Find KNN
    auto neighbors_ptr = scl::memory::aligned_alloc<Index>(n_cells * n_neighbors, SCL_ALIGNMENT);
    Index* neighbors = neighbors_ptr.release();
    detail::find_knn(coordinates, n_cells, n_dims, n_neighbors, neighbors);

    // Check for boundary cells
    for (Size i = 0; i < n_cells; ++i) {
        Index my_label = labels.ptr[i];
        bool has_different_neighbor = false;

        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];
            if (labels.ptr[neighbor] != my_label) {
                has_different_neighbor = true;
                break;
            }
        }

        is_boundary.ptr[i] = has_different_neighbor;
    }

    scl::memory::aligned_free(neighbors);
}

// =============================================================================
// Tissue Region Statistics
// =============================================================================

void region_statistics(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Array<const Index> region_labels,
    Size n_regions,
    Real* region_centroids,  // [n_regions * n_dims]
    Real* region_sizes,      // [n_regions] (cell count)
    Real* region_densities   // [n_regions]
) {
    SCL_CHECK_DIM(n_cells == region_labels.len, "Region labels must match cell count");

    if (n_cells == 0 || n_regions == 0) return;

    // Initialize
    for (Size r = 0; r < n_regions; ++r) {
        region_sizes[r] = Real(0.0);
        region_densities[r] = Real(0.0);
        for (Size d = 0; d < n_dims; ++d) {
            region_centroids[r * n_dims + d] = Real(0.0);
        }
    }

    // Count cells and accumulate coordinates
    for (Size i = 0; i < n_cells; ++i) {
        Index region = region_labels.ptr[i];
        if (region < 0 || region >= static_cast<Index>(n_regions)) continue;

        region_sizes[region] += Real(1.0);
        for (Size d = 0; d < n_dims; ++d) {
            region_centroids[region * n_dims + d] += coordinates[i * n_dims + d];
        }
    }

    // Compute centroids
    for (Size r = 0; r < n_regions; ++r) {
        if (region_sizes[r] > Real(0.0)) {
            for (Size d = 0; d < n_dims; ++d) {
                region_centroids[r * n_dims + d] /= region_sizes[r];
            }
        }
    }

    // Compute densities (cells per unit area)
    for (Size r = 0; r < n_regions; ++r) {
        if (region_sizes[r] < Real(2.0)) {
            region_densities[r] = Real(0.0);
            continue;
        }

        // Compute average pairwise distance within region
        Real total_dist = Real(0.0);
        Size pair_count = 0;

        for (Size i = 0; i < n_cells; ++i) {
            if (region_labels.ptr[i] != static_cast<Index>(r)) continue;

            for (Size j = i + 1; j < n_cells; ++j) {
                if (region_labels.ptr[j] != static_cast<Index>(r)) continue;

                total_dist += detail::euclidean_distance(
                    coordinates + i * n_dims,
                    coordinates + j * n_dims,
                    n_dims);
                ++pair_count;
            }
        }

        if (pair_count > 0) {
            Real avg_dist = total_dist / static_cast<Real>(pair_count);
            if (avg_dist > config::EPSILON) {
                // Density ~ n / (avg_dist^n_dims)
                Real area_proxy = std::pow(avg_dist, static_cast<Real>(n_dims));
                region_densities[r] = region_sizes[r] / area_proxy;
            }
        }
    }
}

// =============================================================================
// Spatial Coherence Score
// =============================================================================

void spatial_coherence(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Array<const Index> labels,
    Array<Real> coherence_scores,
    Size n_neighbors
) {
    SCL_CHECK_DIM(n_cells == labels.len, "Labels must match cell count");
    SCL_CHECK_DIM(n_cells == coherence_scores.len, "Coherence scores must match cell count");

    if (n_cells == 0) return;

    n_neighbors = scl::algo::min2(n_neighbors, n_cells - 1);

    // Find KNN
    auto neighbors_ptr = scl::memory::aligned_alloc<Index>(n_cells * n_neighbors, SCL_ALIGNMENT);
    Index* neighbors = neighbors_ptr.release();
    detail::find_knn(coordinates, n_cells, n_dims, n_neighbors, neighbors);

    // Compute coherence (fraction of same-label neighbors)
    for (Size i = 0; i < n_cells; ++i) {
        Index my_label = labels.ptr[i];
        Size same_label_count = 0;

        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];
            if (labels.ptr[neighbor] == my_label) {
                ++same_label_count;
            }
        }

        coherence_scores.ptr[i] = static_cast<Real>(same_label_count) /
                                  static_cast<Real>(n_neighbors);
    }

    scl::memory::aligned_free(neighbors);
}

} // namespace scl::kernel::tissue
