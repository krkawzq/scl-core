#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/spatial_pattern.hpp
// BRIEF: Spatial pattern detection (SpatialDE-style)
//
// APPLICATIONS:
// - Spatially variable genes
// - Spatial gradients
// - Periodic patterns
// - Domain identification
// =============================================================================

namespace scl::kernel::spatial_pattern {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_NEIGHBORS = 3;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real BANDWIDTH_SCALE = Real(0.3);
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

// Compute Euclidean distance between two points
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

// Gaussian kernel
SCL_FORCE_INLINE Real gaussian_kernel(Real dist, Real bandwidth) {
    Real x = dist / bandwidth;
    return std::exp(-Real(0.5) * x * x);
}

// Compute spatial weights matrix (Gaussian kernel)
void compute_spatial_weights(
    const Real* coords,  // [n_cells * n_dims]
    Size n_cells,
    Size n_dims,
    Real bandwidth,
    Real* weights  // [n_cells * n_cells]
) {
    for (Size i = 0; i < n_cells; ++i) {
        weights[i * n_cells + i] = Real(0.0);

        for (Size j = i + 1; j < n_cells; ++j) {
            Real dist = euclidean_distance(
                coords + i * n_dims,
                coords + j * n_dims,
                n_dims);

            Real w = gaussian_kernel(dist, bandwidth);
            weights[i * n_cells + j] = w;
            weights[j * n_cells + i] = w;
        }
    }

    // Row normalize
    for (Size i = 0; i < n_cells; ++i) {
        Real row_sum = Real(0.0);
        for (Size j = 0; j < n_cells; ++j) {
            row_sum += weights[i * n_cells + j];
        }
        if (row_sum > config::EPSILON) {
            for (Size j = 0; j < n_cells; ++j) {
                weights[i * n_cells + j] /= row_sum;
            }
        }
    }
}

// Moran's I statistic for spatial autocorrelation
SCL_FORCE_INLINE Real morans_i(
    const Real* values,
    const Real* weights,
    Size n
) {
    // Compute mean
    Real mean = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        mean += values[i];
    }
    mean /= static_cast<Real>(n);

    // Compute numerator and denominator
    Real num = Real(0.0);
    Real denom = Real(0.0);
    Real w_sum = Real(0.0);

    for (Size i = 0; i < n; ++i) {
        Real di = values[i] - mean;
        denom += di * di;

        for (Size j = 0; j < n; ++j) {
            Real dj = values[j] - mean;
            num += weights[i * n + j] * di * dj;
            w_sum += weights[i * n + j];
        }
    }

    if (denom < config::EPSILON || w_sum < config::EPSILON) {
        return Real(0.0);
    }

    return (static_cast<Real>(n) / w_sum) * (num / denom);
}

// Geary's C statistic
SCL_FORCE_INLINE Real gearys_c(
    const Real* values,
    const Real* weights,
    Size n
) {
    // Compute mean
    Real mean = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        mean += values[i];
    }
    mean /= static_cast<Real>(n);

    // Compute numerator and denominator
    Real num = Real(0.0);
    Real denom = Real(0.0);
    Real w_sum = Real(0.0);

    for (Size i = 0; i < n; ++i) {
        Real di = values[i] - mean;
        denom += di * di;

        for (Size j = 0; j < n; ++j) {
            Real diff = values[i] - values[j];
            num += weights[i * n + j] * diff * diff;
            w_sum += weights[i * n + j];
        }
    }

    if (denom < config::EPSILON || w_sum < config::EPSILON) {
        return Real(1.0);
    }

    return ((static_cast<Real>(n - 1)) / (Real(2.0) * w_sum)) * (num / denom);
}

// Compute bandwidth from data
SCL_FORCE_INLINE Real estimate_bandwidth(
    const Real* coords,
    Size n_cells,
    Size n_dims
) {
    if (n_cells < 2) return Real(1.0);

    // Compute average nearest neighbor distance
    Real total_dist = Real(0.0);

    for (Size i = 0; i < scl::algo::min2(n_cells, Size(100)); ++i) {
        Real min_dist = std::numeric_limits<Real>::max();

        for (Size j = 0; j < n_cells; ++j) {
            if (i == j) continue;

            Real dist = euclidean_distance(
                coords + i * n_dims,
                coords + j * n_dims,
                n_dims);

            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        if (min_dist < std::numeric_limits<Real>::max()) {
            total_dist += min_dist;
        }
    }

    Size sample_size = scl::algo::min2(n_cells, Size(100));
    Real avg_nn_dist = total_dist / static_cast<Real>(sample_size);

    return avg_nn_dist * config::DEFAULT_N_NEIGHBORS * config::BANDWIDTH_SCALE;
}

} // namespace detail

// =============================================================================
// Spatial Variability (SpatialDE-style)
// =============================================================================

template <typename T, bool IsCSR>
void spatial_variability(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,  // [n_cells * n_dims]
    Size n_dims,
    Real* variability_scores,  // [n_genes]
    Real* p_values,           // [n_genes]
    Size n_permutations,
    uint64_t seed
) {
    static_assert(IsCSR, "spatial_variability requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    if (n_cells == 0 || n_genes == 0) return;

    // Estimate bandwidth
    Real bandwidth = detail::estimate_bandwidth(coordinates, n_cells, n_dims);

    // Compute spatial weights matrix
    // PERFORMANCE: RAII memory management with unique_ptr
    auto weights_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_cells, SCL_ALIGNMENT);
    Real* weights = weights_ptr.get();
    detail::compute_spatial_weights(coordinates, n_cells, n_dims, bandwidth, weights);

    // Extract gene expression values
    auto gene_values_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* gene_values = gene_values_ptr.get();
    auto permuted_values_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* permuted_values = permuted_values_ptr.get();
    auto perm_indices_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Index* perm_indices = perm_indices_ptr.get();

    detail::LCG rng(seed);

    for (Size g = 0; g < n_genes; ++g) {
        // Extract expression for this gene
        for (Size i = 0; i < n_cells; ++i) {
            gene_values[i] = Real(0.0);
        }

        for (Size i = 0; i < n_cells; ++i) {
            auto row_col_indices = expression.row_indices_unsafe(static_cast<Index>(i));
            auto row_values = expression.row_values_unsafe(static_cast<Index>(i));

            for (Size j = 0; j < row_col_indices.len; ++j) {
                if (row_col_indices[j] == static_cast<Index>(g)) {
                    gene_values[i] = static_cast<Real>(row_values[j]);
                    break;
                }
            }
        }

        // Compute Moran's I
        Real observed_I = detail::morans_i(gene_values, weights, n_cells);
        variability_scores[g] = observed_I;

        // Permutation test for p-value
        Size n_greater = 0;

        for (Size p = 0; p < n_permutations; ++p) {
            // Shuffle
            for (Size i = 0; i < n_cells; ++i) {
                perm_indices[i] = static_cast<Index>(i);
            }
            for (Size i = n_cells - 1; i > 0; --i) {
                Size j = rng.next_index(i + 1);
                std::swap(perm_indices[i], perm_indices[j]);
            }

            for (Size i = 0; i < n_cells; ++i) {
                permuted_values[i] = gene_values[perm_indices[i]];
            }

            Real perm_I = detail::morans_i(permuted_values, weights, n_cells);
            if (perm_I >= observed_I) {
                ++n_greater;
            }
        }

        p_values[g] = (static_cast<Real>(n_greater) + Real(1.0)) /
                     (static_cast<Real>(n_permutations) + Real(1.0));
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Spatial Gradient Detection
// =============================================================================

void spatial_gradient(
    const Real* expression,    // [n_cells]
    const Real* coordinates,   // [n_cells * n_dims]
    Size n_cells,
    Size n_dims,
    Real* gradient_direction,  // [n_dims]
    Real& gradient_strength
) {
    if (n_cells == 0 || n_dims == 0) {
        gradient_strength = Real(0.0);
        return;
    }

    // Compute gradient using least squares regression
    // expression ~ a + b1*x1 + b2*x2 + ...

    // Compute means
    // PERFORMANCE: RAII memory management with unique_ptr
    Real expr_mean = Real(0.0);
    auto coord_means_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    Real* coord_means = coord_means_ptr.get();

    for (Size d = 0; d < n_dims; ++d) {
        coord_means[d] = Real(0.0);
    }

    for (Size i = 0; i < n_cells; ++i) {
        expr_mean += expression[i];
        for (Size d = 0; d < n_dims; ++d) {
            coord_means[d] += coordinates[i * n_dims + d];
        }
    }

    expr_mean /= static_cast<Real>(n_cells);
    for (Size d = 0; d < n_dims; ++d) {
        coord_means[d] /= static_cast<Real>(n_cells);
    }

    // Compute covariance and variance
    // PERFORMANCE: RAII memory management with unique_ptr
    auto cov_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    auto var_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    Real* cov = cov_ptr.get();
    Real* var = var_ptr.get();

    for (Size d = 0; d < n_dims; ++d) {
        cov[d] = Real(0.0);
        var[d] = Real(0.0);
    }

    for (Size i = 0; i < n_cells; ++i) {
        Real expr_diff = expression[i] - expr_mean;

        for (Size d = 0; d < n_dims; ++d) {
            Real coord_diff = coordinates[i * n_dims + d] - coord_means[d];
            cov[d] += expr_diff * coord_diff;
            var[d] += coord_diff * coord_diff;
        }
    }

    // Compute gradient direction (regression coefficients)
    Real grad_mag_sq = Real(0.0);

    for (Size d = 0; d < n_dims; ++d) {
        if (var[d] > config::EPSILON) {
            gradient_direction[d] = cov[d] / var[d];
        } else {
            gradient_direction[d] = Real(0.0);
        }
        grad_mag_sq += gradient_direction[d] * gradient_direction[d];
    }

    gradient_strength = std::sqrt(grad_mag_sq);

    // Normalize direction
    if (gradient_strength > config::EPSILON) {
        for (Size d = 0; d < n_dims; ++d) {
            gradient_direction[d] /= gradient_strength;
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Periodic Pattern Detection
// =============================================================================

template <typename T, bool IsCSR>
void periodic_pattern(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Real* periodicity_scores,  // [n_genes]
    Real* dominant_wavelengths,  // [n_genes]
    Size n_wavelengths,
    const Real* test_wavelengths  // wavelengths to test
) {
    static_assert(IsCSR, "periodic_pattern requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    if (n_cells == 0 || n_genes == 0 || n_wavelengths == 0) return;

    // PERFORMANCE: RAII memory management with unique_ptr
    auto gene_values_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    auto cos_basis_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    auto sin_basis_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* gene_values = gene_values_ptr.get();
    Real* cos_basis = cos_basis_ptr.get();
    Real* sin_basis = sin_basis_ptr.get();

    for (Size g = 0; g < n_genes; ++g) {
        // Extract gene expression
        for (Size i = 0; i < n_cells; ++i) {
            gene_values[i] = Real(0.0);
        }

        for (Size i = 0; i < n_cells; ++i) {
            auto row_col_indices = expression.row_indices_unsafe(i);
            auto row_vals = expression.row_values_unsafe(i);

            for (Size j = 0; j < row_col_indices.len; ++j) {
                if (row_col_indices[j] == static_cast<Index>(g)) {
                    gene_values[i] = static_cast<Real>(row_vals[j]);
                    break;
                }
            }
        }

        // Test each wavelength
        Real best_score = Real(0.0);
        Real best_wavelength = test_wavelengths[0];

        for (Size w = 0; w < n_wavelengths; ++w) {
            Real wavelength = test_wavelengths[w];
            Real freq = Real(2.0) * Real(3.14159265358979323846) / wavelength;

            // For each dimension, test periodicity
            Real total_score = Real(0.0);

            for (Size d = 0; d < n_dims; ++d) {
                // Create basis functions
                for (Size i = 0; i < n_cells; ++i) {
                    Real x = coordinates[i * n_dims + d];
                    cos_basis[i] = std::cos(freq * x);
                    sin_basis[i] = std::sin(freq * x);
                }

                // Compute correlation with expression
                Real mean_expr = Real(0.0);
                Real mean_cos = Real(0.0);
                Real mean_sin = Real(0.0);

                for (Size i = 0; i < n_cells; ++i) {
                    mean_expr += gene_values[i];
                    mean_cos += cos_basis[i];
                    mean_sin += sin_basis[i];
                }
                mean_expr /= static_cast<Real>(n_cells);
                mean_cos /= static_cast<Real>(n_cells);
                mean_sin /= static_cast<Real>(n_cells);

                Real cov_cos = Real(0.0), var_cos = Real(0.0);
                Real cov_sin = Real(0.0), var_sin = Real(0.0);
                Real var_expr = Real(0.0);

                for (Size i = 0; i < n_cells; ++i) {
                    Real de = gene_values[i] - mean_expr;
                    Real dc = cos_basis[i] - mean_cos;
                    Real ds = sin_basis[i] - mean_sin;

                    var_expr += de * de;
                    cov_cos += de * dc;
                    var_cos += dc * dc;
                    cov_sin += de * ds;
                    var_sin += ds * ds;
                }

                Real r_cos = Real(0.0), r_sin = Real(0.0);
                if (var_expr > config::EPSILON && var_cos > config::EPSILON) {
                    r_cos = cov_cos / std::sqrt(var_expr * var_cos);
                }
                if (var_expr > config::EPSILON && var_sin > config::EPSILON) {
                    r_sin = cov_sin / std::sqrt(var_expr * var_sin);
                }

                // Combined score (R^2)
                total_score += r_cos * r_cos + r_sin * r_sin;
            }

            if (total_score > best_score) {
                best_score = total_score;
                best_wavelength = wavelength;
            }
        }

        periodicity_scores[g] = best_score / static_cast<Real>(n_dims);
        dominant_wavelengths[g] = best_wavelength;
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Boundary Detection
// =============================================================================

template <typename T, bool IsCSR>
void boundary_detection(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Array<Real> boundary_scores,
    Size n_neighbors
) {
    static_assert(IsCSR, "boundary_detection requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == boundary_scores.len, "Boundary scores must match cell count");

    if (n_cells == 0 || n_genes == 0) return;

    n_neighbors = scl::algo::min2(n_neighbors, n_cells - 1);

    // Find k-nearest neighbors for each cell
    // PERFORMANCE: RAII memory management with unique_ptr
    auto neighbors_ptr = scl::memory::aligned_alloc<Index>(n_cells * n_neighbors, SCL_ALIGNMENT);
    auto distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    auto sorted_idx_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Index* neighbors = neighbors_ptr.get();
    Real* distances = distances_ptr.get();
    Index* sorted_idx = sorted_idx_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        // Compute distances to all other cells
        for (Size j = 0; j < n_cells; ++j) {
            sorted_idx[j] = static_cast<Index>(j);
            distances[j] = detail::euclidean_distance(
                coordinates + i * n_dims,
                coordinates + j * n_dims,
                n_dims);
        }

        // Partial sort to get k nearest
        scl::algo::partial_sort(sorted_idx, static_cast<Size>(n_cells), n_neighbors + 1,
            [&](Index a, Index b) { return distances[a] < distances[b]; });

        // Skip self (index 0)
        for (Size k = 0; k < n_neighbors; ++k) {
            neighbors[i * n_neighbors + k] = sorted_idx[k + 1];
        }
    }

    // Compute boundary score based on expression heterogeneity
    // PERFORMANCE: RAII memory management with unique_ptr
    auto expr_i_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    auto expr_j_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* expr_i = expr_i_ptr.get();
    Real* expr_j = expr_j_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        // Extract expression for cell i
        for (Size g = 0; g < n_genes; ++g) {
            expr_i[g] = Real(0.0);
        }

        auto row_col_indices_i = expression.row_indices_unsafe(i);
        auto row_vals_i = expression.row_values_unsafe(i);

        for (Size j = 0; j < row_col_indices_i.len; ++j) {
            expr_i[row_col_indices_i[j]] = static_cast<Real>(row_vals_i[j]);
        }

        // Compute average distance to neighbors in expression space
        Real total_expr_dist = Real(0.0);

        for (Size k = 0; k < n_neighbors; ++k) {
            Index neighbor = neighbors[i * n_neighbors + k];

            // Extract expression for neighbor
            for (Size g = 0; g < n_genes; ++g) {
                expr_j[g] = Real(0.0);
            }

            auto row_col_indices_j = expression.row_indices_unsafe(neighbor);
            auto row_vals_j = expression.row_values_unsafe(neighbor);

            for (Size j = 0; j < row_col_indices_j.len; ++j) {
                expr_j[row_col_indices_j[j]] = static_cast<Real>(row_vals_j[j]);
            }

            // Compute expression distance
            Real expr_dist = Real(0.0);
            for (Size g = 0; g < n_genes; ++g) {
                Real diff = expr_i[g] - expr_j[g];
                expr_dist += diff * diff;
            }
            total_expr_dist += std::sqrt(expr_dist);
        }

        boundary_scores.ptr[i] = total_expr_dist / static_cast<Real>(n_neighbors);
    }

    // Normalize scores
    Real max_score = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        if (boundary_scores.ptr[i] > max_score) {
            max_score = boundary_scores.ptr[i];
        }
    }

    if (max_score > config::EPSILON) {
        for (Size i = 0; i < n_cells; ++i) {
            boundary_scores.ptr[i] /= max_score;
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Spatial Domain Identification
// =============================================================================

template <typename T, bool IsCSR>
void spatial_domain(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Index n_domains,
    Array<Index> domain_labels,
    uint64_t seed
) {
    static_assert(IsCSR, "spatial_domain requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == domain_labels.len, "Domain labels must match cell count");

    if (n_cells == 0 || n_domains == 0) return;

    n_domains = scl::algo::min2(static_cast<Size>(n_domains), n_cells);

    // Combined feature space: spatial + expression (PCA-reduced)
    // PERFORMANCE: RAII memory management with unique_ptr
    Size n_features = n_dims + scl::algo::min2(n_genes, Size(10));
    auto features_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_features, SCL_ALIGNMENT);
    Real* features = features_ptr.get();

    // Copy spatial coordinates (normalized)
    auto coord_min_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    auto coord_max_ptr = scl::memory::aligned_alloc<Real>(n_dims, SCL_ALIGNMENT);
    Real* coord_min = coord_min_ptr.get();
    Real* coord_max = coord_max_ptr.get();

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

    // Add expression features (top variable genes)
    // PERFORMANCE: RAII memory management with unique_ptr
    auto gene_var_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    auto gene_mean_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* gene_var = gene_var_ptr.get();
    Real* gene_mean = gene_mean_ptr.get();

    for (Size g = 0; g < n_genes; ++g) {
        gene_mean[g] = Real(0.0);
        gene_var[g] = Real(0.0);
    }

    // Compute gene means
    for (Size i = 0; i < n_cells; ++i) {
        auto row_col_indices = expression.row_indices_unsafe(i);
        auto row_vals = expression.row_values_unsafe(i);

        for (Size j = 0; j < row_col_indices.len; ++j) {
            gene_mean[row_col_indices[j]] +=
                static_cast<Real>(row_vals[j]);
        }
    }

    for (Size g = 0; g < n_genes; ++g) {
        gene_mean[g] /= static_cast<Real>(n_cells);
    }

    // Compute gene variances
    for (Size i = 0; i < n_cells; ++i) {
        auto row_col_indices = expression.row_indices_unsafe(i);
        auto row_vals = expression.row_values_unsafe(i);

        for (Size j = 0; j < row_col_indices.len; ++j) {
            Real diff = static_cast<Real>(row_vals[j]) -
                       gene_mean[row_col_indices[j]];
            gene_var[row_col_indices[j]] += diff * diff;
        }
    }

    // Select top variable genes
    // PERFORMANCE: RAII memory management with unique_ptr
    Size n_expr_features = n_features - n_dims;
    auto top_genes_ptr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
    Index* top_genes = top_genes_ptr.get();
    for (Size g = 0; g < n_genes; ++g) {
        top_genes[g] = static_cast<Index>(g);
    }

    scl::algo::partial_sort(top_genes, static_cast<Size>(n_genes), n_expr_features,
        [&](Index a, Index b) { return gene_var[a] > gene_var[b]; });

    // Add expression features
    for (Size i = 0; i < n_cells; ++i) {
        for (Size f = 0; f < n_expr_features; ++f) {
            features[i * n_features + n_dims + f] = Real(0.0);
        }

        auto row_col_indices = expression.row_indices_unsafe(i);
        auto row_vals = expression.row_values_unsafe(i);

        for (Size j = 0; j < row_col_indices.len; ++j) {
            Index gene = row_col_indices[j];

            for (Size f = 0; f < n_expr_features; ++f) {
                if (top_genes[f] == gene) {
                    Real val = static_cast<Real>(row_vals[j]);
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

    // K-means clustering on combined features
    // PERFORMANCE: RAII memory management with unique_ptr
    auto centroids_ptr = scl::memory::aligned_alloc<Real>(n_domains * n_features, SCL_ALIGNMENT);
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_domains, SCL_ALIGNMENT);
    Real* centroids = centroids_ptr.get();
    Size* cluster_sizes = cluster_sizes_ptr.get();

    detail::LCG rng(seed);

    // Initialize centroids randomly
    for (Size c = 0; c < static_cast<Size>(n_domains); ++c) {
        Size idx = rng.next_index(n_cells);
        for (Size f = 0; f < n_features; ++f) {
            centroids[c * n_features + f] = features[idx * n_features + f];
        }
    }

    // K-means iterations
    for (Size iter = 0; iter < config::MAX_ITERATIONS; ++iter) {
        bool changed = false;

        // Assign cells to nearest centroid
        for (Size i = 0; i < n_cells; ++i) {
            Real min_dist = std::numeric_limits<Real>::max();
            Index best_cluster = 0;

            for (Size c = 0; c < static_cast<Size>(n_domains); ++c) {
                Real dist = Real(0.0);
                for (Size f = 0; f < n_features; ++f) {
                    Real diff = features[i * n_features + f] -
                               centroids[c * n_features + f];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<Index>(c);
                }
            }

            if (domain_labels.ptr[i] != best_cluster) {
                changed = true;
                domain_labels.ptr[i] = best_cluster;
            }
        }

        if (!changed) break;

        // Update centroids
        for (Size c = 0; c < static_cast<Size>(n_domains); ++c) {
            cluster_sizes[c] = 0;
            for (Size f = 0; f < n_features; ++f) {
                centroids[c * n_features + f] = Real(0.0);
            }
        }

        for (Size i = 0; i < n_cells; ++i) {
            Index cluster = domain_labels.ptr[i];
            ++cluster_sizes[cluster];
            for (Size f = 0; f < n_features; ++f) {
                centroids[cluster * n_features + f] += features[i * n_features + f];
            }
        }

        for (Size c = 0; c < static_cast<Size>(n_domains); ++c) {
            if (cluster_sizes[c] > 0) {
                for (Size f = 0; f < n_features; ++f) {
                    centroids[c * n_features + f] /= static_cast<Real>(cluster_sizes[c]);
                }
            }
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Local Spatial Statistics (Getis-Ord Gi*)
// =============================================================================

void hotspot_analysis(
    const Real* values,
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    Real bandwidth,
    Real* gi_scores,
    Real* z_scores
) {
    if (n_cells == 0) return;

    // Compute global mean and std
    Real global_mean = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        global_mean += values[i];
    }
    global_mean /= static_cast<Real>(n_cells);

    Real global_var = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        Real diff = values[i] - global_mean;
        global_var += diff * diff;
    }
    Real global_std = std::sqrt(global_var / static_cast<Real>(n_cells));

    // Compute Gi* for each cell
    for (Size i = 0; i < n_cells; ++i) {
        Real w_sum = Real(0.0);
        Real w_sq_sum = Real(0.0);
        Real weighted_sum = Real(0.0);

        for (Size j = 0; j < n_cells; ++j) {
            Real dist = detail::euclidean_distance(
                coordinates + i * n_dims,
                coordinates + j * n_dims,
                n_dims);

            Real w = detail::gaussian_kernel(dist, bandwidth);
            w_sum += w;
            w_sq_sum += w * w;
            weighted_sum += w * values[j];
        }

        if (w_sum < config::EPSILON) {
            gi_scores[i] = Real(0.0);
            z_scores[i] = Real(0.0);
            continue;
        }

        gi_scores[i] = weighted_sum / (global_mean * w_sum);

        // Z-score
        Real n_r = static_cast<Real>(n_cells);
        Real expected = w_sum * global_mean;
        Real var = global_std * global_std *
            (n_r * w_sq_sum - w_sum * w_sum) / (n_r * n_r);

        if (var > config::EPSILON) {
            z_scores[i] = (weighted_sum - expected) / std::sqrt(var);
        } else {
            z_scores[i] = Real(0.0);
        }
    }
}

// =============================================================================
// Spatial Autocorrelation Per Gene
// =============================================================================

template <typename T, bool IsCSR>
void spatial_autocorrelation(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Real* morans_i,
    Real* gearys_c
) {
    static_assert(IsCSR, "spatial_autocorrelation requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    if (n_cells == 0 || n_genes == 0) return;

    // Estimate bandwidth and compute weights
    // PERFORMANCE: RAII memory management with unique_ptr
    Real bandwidth = detail::estimate_bandwidth(coordinates, n_cells, n_dims);
    auto weights_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_cells, SCL_ALIGNMENT);
    Real* weights = weights_ptr.get();
    detail::compute_spatial_weights(coordinates, n_cells, n_dims, bandwidth, weights);

    auto gene_values_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* gene_values = gene_values_ptr.get();

    for (Size g = 0; g < n_genes; ++g) {
        // Extract gene expression
        for (Size i = 0; i < n_cells; ++i) {
            gene_values[i] = Real(0.0);
        }

        for (Size i = 0; i < n_cells; ++i) {
            auto row_col_indices = expression.row_indices_unsafe(i);
            auto row_vals = expression.row_values_unsafe(i);

            for (Size j = 0; j < row_col_indices.len; ++j) {
                if (row_col_indices[j] == static_cast<Index>(g)) {
                    gene_values[i] = static_cast<Real>(row_vals[j]);
                    break;
                }
            }
        }

        morans_i[g] = detail::morans_i(gene_values, weights, n_cells);
        gearys_c[g] = detail::gearys_c(gene_values, weights, n_cells);
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Spatial Smoothing
// =============================================================================

template <typename T, bool IsCSR>
void spatial_smoothing(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    Real bandwidth,
    Real* smoothed  // [n_cells * n_genes]
) {
    static_assert(IsCSR, "spatial_smoothing requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    if (n_cells == 0 || n_genes == 0) return;

    // Compute spatial weights
    // PERFORMANCE: RAII memory management with unique_ptr
    auto weights_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_cells, SCL_ALIGNMENT);
    Real* weights = weights_ptr.get();
    detail::compute_spatial_weights(coordinates, n_cells, n_dims, bandwidth, weights);

    // Initialize smoothed expression
    for (Size i = 0; i < n_cells * n_genes; ++i) {
        smoothed[i] = Real(0.0);
    }

    // Apply smoothing
    for (Size i = 0; i < n_cells; ++i) {
        // Accumulate weighted expression from neighbors
        for (Size j = 0; j < n_cells; ++j) {
            Real w = weights[i * n_cells + j];
            if (w < config::EPSILON) continue;

            auto row_col_indices = expression.row_indices_unsafe(j);
            auto row_vals = expression.row_values_unsafe(j);

            for (Size k = 0; k < row_col_indices.len; ++k) {
                Index gene = row_col_indices[k];
                smoothed[i * n_genes + gene] += w * static_cast<Real>(row_vals[k]);
            }
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Spatial Co-expression
// =============================================================================

template <typename T, bool IsCSR>
void spatial_coexpression(
    const Sparse<T, IsCSR>& expression,
    const Real* coordinates,
    Size n_dims,
    const Index* gene_pairs,  // [n_pairs * 2]
    Size n_pairs,
    Real* coexpression_scores
) {
    static_assert(IsCSR, "spatial_coexpression requires CSR format (cells × genes)");
    
    const Size n_cells = static_cast<Size>(expression.rows());

    if (n_cells == 0 || n_pairs == 0) return;

    // Compute spatial weights
    // PERFORMANCE: RAII memory management with unique_ptr
    Real bandwidth = detail::estimate_bandwidth(coordinates, n_cells, n_dims);
    auto weights_ptr = scl::memory::aligned_alloc<Real>(n_cells * n_cells, SCL_ALIGNMENT);
    Real* weights = weights_ptr.get();
    detail::compute_spatial_weights(coordinates, n_cells, n_dims, bandwidth, weights);

    auto expr1_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    auto expr2_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* expr1 = expr1_ptr.get();
    Real* expr2 = expr2_ptr.get();

    for (Size p = 0; p < n_pairs; ++p) {
        Index gene1 = gene_pairs[p * 2];
        Index gene2 = gene_pairs[p * 2 + 1];

        // Extract expression for both genes
        for (Size i = 0; i < n_cells; ++i) {
            expr1[i] = Real(0.0);
            expr2[i] = Real(0.0);
        }

        for (Size i = 0; i < n_cells; ++i) {
            auto row_col_indices = expression.row_indices_unsafe(i);
            auto row_vals = expression.row_values_unsafe(i);

            for (Size j = 0; j < row_col_indices.len; ++j) {
                if (row_col_indices[j] == gene1) {
                    expr1[i] = static_cast<Real>(row_vals[j]);
                }
                if (row_col_indices[j] == gene2) {
                    expr2[i] = static_cast<Real>(row_vals[j]);
                }
            }
        }

        // Compute Lee's L (bivariate spatial correlation)
        Real mean1 = Real(0.0), mean2 = Real(0.0);
        for (Size i = 0; i < n_cells; ++i) {
            mean1 += expr1[i];
            mean2 += expr2[i];
        }
        mean1 /= static_cast<Real>(n_cells);
        mean2 /= static_cast<Real>(n_cells);

        Real num = Real(0.0);
        Real denom1 = Real(0.0), denom2 = Real(0.0);
        Real w_sum = Real(0.0);

        for (Size i = 0; i < n_cells; ++i) {
            Real d1 = expr1[i] - mean1;
            denom1 += d1 * d1;

            Real weighted_d2 = Real(0.0);
            for (Size j = 0; j < n_cells; ++j) {
                weighted_d2 += weights[i * n_cells + j] * (expr2[j] - mean2);
                w_sum += weights[i * n_cells + j];
            }

            num += d1 * weighted_d2;
        }

        for (Size i = 0; i < n_cells; ++i) {
            Real d2 = expr2[i] - mean2;
            denom2 += d2 * d2;
        }

        if (denom1 > config::EPSILON && denom2 > config::EPSILON && w_sum > config::EPSILON) {
            coexpression_scores[p] = (static_cast<Real>(n_cells) / w_sum) *
                num / std::sqrt(denom1 * denom2);
        } else {
            coexpression_scores[p] = Real(0.0);
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Ripley's K Function
// =============================================================================

void ripleys_k(
    const Real* coordinates,
    Size n_cells,
    Size n_dims,
    const Real* radii,
    Size n_radii,
    Real* k_values,
    Real study_area
) {
    if (n_cells < 2 || n_radii == 0) return;

    Real n_r = static_cast<Real>(n_cells);
    Real intensity = n_r / study_area;

    for (Size r = 0; r < n_radii; ++r) {
        Real radius = radii[r];
        Size count = 0;

        for (Size i = 0; i < n_cells; ++i) {
            for (Size j = i + 1; j < n_cells; ++j) {
                Real dist = detail::euclidean_distance(
                    coordinates + i * n_dims,
                    coordinates + j * n_dims,
                    n_dims);

                if (dist <= radius) {
                    count += 2;  // Count both (i,j) and (j,i)
                }
            }
        }

        // K(r) = count / (n * intensity)
        k_values[r] = static_cast<Real>(count) / (n_r * intensity);
    }
}

// =============================================================================
// Spatial Entropy
// =============================================================================

void spatial_entropy(
    Array<const Index> labels,
    const Real* coordinates,
    Size n_dims,
    Real bandwidth,
    Array<Real> entropy_scores
) {
    const Size n_cells = labels.len;
    SCL_CHECK_DIM(n_cells == entropy_scores.len, "Entropy scores must match cell count");

    if (n_cells == 0) return;

    // Find number of unique labels
    Index n_labels = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (labels.ptr[i] >= n_labels) {
            n_labels = labels.ptr[i] + 1;
        }
    }

    // PERFORMANCE: RAII memory management with unique_ptr
    auto local_counts_ptr = scl::memory::aligned_alloc<Real>(n_labels, SCL_ALIGNMENT);
    Real* local_counts = local_counts_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        // Compute local label distribution
        for (Index l = 0; l < n_labels; ++l) {
            local_counts[l] = Real(0.0);
        }

        Real total_weight = Real(0.0);

        for (Size j = 0; j < n_cells; ++j) {
            Real dist = detail::euclidean_distance(
                coordinates + i * n_dims,
                coordinates + j * n_dims,
                n_dims);

            Real w = detail::gaussian_kernel(dist, bandwidth);
            local_counts[labels.ptr[j]] += w;
            total_weight += w;
        }

        // Compute entropy
        Real entropy = Real(0.0);
        if (total_weight > config::EPSILON) {
            for (Index l = 0; l < n_labels; ++l) {
                Real p = local_counts[l] / total_weight;
                if (p > config::EPSILON) {
                    entropy -= p * std::log(p);
                }
            }
        }

        // Normalize by max entropy
        Real max_entropy = std::log(static_cast<Real>(n_labels));
        if (max_entropy > config::EPSILON) {
            entropy_scores.ptr[i] = entropy / max_entropy;
        } else {
            entropy_scores.ptr[i] = Real(0.0);
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

} // namespace scl::kernel::spatial_pattern
