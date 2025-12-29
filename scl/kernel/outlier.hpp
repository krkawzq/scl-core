#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/outlier.hpp
// BRIEF: Outlier and anomaly detection for single-cell data
//
// APPLICATIONS:
// - Quality control (LOF, isolation)
// - Ambient RNA detection
// - Empty droplet detection (EmptyDrops-style)
// - Outlier gene identification
// =============================================================================

namespace scl::kernel::outlier {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_K_NEIGHBORS = 5;
    constexpr Size DEFAULT_K = 20;
    constexpr Real LOF_THRESHOLD = Real(1.5);
    constexpr Real AMBIENT_THRESHOLD = Real(0.1);
    constexpr Size EMPTY_DROPS_MIN_UMI = 100;
    constexpr Size EMPTY_DROPS_MAX_AMBIENT = 10;
    constexpr Size MONTE_CARLO_ITERATIONS = 10000;
    constexpr Size PARALLEL_THRESHOLD = 256;
}

namespace detail {

// Compute k-distance (distance to k-th nearest neighbor)
SCL_FORCE_INLINE Real compute_k_distance(
    const Real* neighbor_distances,
    Size n_neighbors,
    Size k
) {
    if (n_neighbors == 0) return Real(0.0);
    Size actual_k = scl::algo::min2(k, n_neighbors);
    return neighbor_distances[actual_k - 1];
}

// Compute reachability distance
SCL_FORCE_INLINE Real reachability_distance(
    Real dist_to_point,
    Real k_distance_of_point
) {
    return scl::algo::max2(dist_to_point, k_distance_of_point);
}

// Compute local reachability density
SCL_FORCE_INLINE Real local_reachability_density(
    const Real* reach_distances,
    Size n_neighbors
) {
    if (n_neighbors == 0) return Real(0.0);

    Real sum = scl::vectorize::sum(Array<const Real>(reach_distances, n_neighbors));
    Real mean_reach = sum / static_cast<Real>(n_neighbors);
    if (mean_reach < config::EPSILON) {
        return Real(1.0) / config::EPSILON;
    }
    return Real(1.0) / mean_reach;
}

// Chi-squared CDF approximation for EmptyDrops p-value
SCL_FORCE_INLINE Real chi2_cdf_approx(Real x, Real df) {
    if (x <= Real(0.0)) return Real(0.0);
    if (df <= Real(0.0)) return Real(1.0);

    // Wilson-Hilferty approximation
    Real z = std::pow(x / df, Real(1.0) / Real(3.0));
    Real mean = Real(1.0) - Real(2.0) / (Real(9.0) * df);
    Real std_dev = std::sqrt(Real(2.0) / (Real(9.0) * df));

    Real normal_z = (z - mean) / std_dev;

    // Standard normal CDF approximation
    Real t = Real(1.0) / (Real(1.0) + Real(0.2316419) * std::abs(normal_z));
    Real d = Real(0.3989423) * std::exp(-Real(0.5) * normal_z * normal_z);
    Real cdf = d * t * (Real(0.3193815) + t * (Real(-0.3565638) + t *
        (Real(1.781478) + t * (Real(-1.821256) + t * Real(1.330274)))));

    if (normal_z > Real(0.0)) {
        return Real(1.0) - cdf;
    }
    return cdf;
}

// Dirichlet-multinomial log-likelihood for EmptyDrops
SCL_FORCE_INLINE Real dirichlet_multinomial_loglik(
    const Real* observed,
    const Real* ambient_profile,
    Size n_genes,
    Real total_count
) {
    Real loglik = Real(0.0);
    Real alpha_sum = Real(0.0);

    for (Size g = 0; g < n_genes; ++g) {
        Real alpha = ambient_profile[g] * total_count + Real(1.0);
        alpha_sum += alpha;

        // log Gamma(observed + alpha) - log Gamma(alpha)
        Real obs = observed[g];
        if (obs > Real(0.0)) {
            loglik += std::lgamma(obs + alpha) - std::lgamma(alpha);
        }
    }

    // Normalization terms
    loglik += std::lgamma(alpha_sum) - std::lgamma(total_count + alpha_sum);

    return loglik;
}

// Compute mean and variance
SCL_FORCE_INLINE void compute_mean_var(
    const Real* data,
    Size n,
    Real& mean,
    Real& var
) {
    if (n == 0) {
        mean = Real(0.0);
        var = Real(0.0);
        return;
    }

    Real sum = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        sum += data[i];
    }
    mean = sum / static_cast<Real>(n);

    Real var_sum = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        Real diff = data[i] - mean;
        var_sum += diff * diff;
    }
    var = (n > 1) ? var_sum / static_cast<Real>(n - 1) : Real(0.0);
}

// Median absolute deviation
SCL_FORCE_INLINE Real compute_mad(Real* data, Size n, Real* temp) {
    if (n == 0) return Real(0.0);

    // Copy data
    scl::algo::copy(data, temp, n);

    // Find median using correct nth_element signature
    scl::algo::nth_element<Real>(temp, temp + (n / 2), temp + n);
    Real median = temp[n / 2];

    // Compute absolute deviations
    for (Size i = 0; i < n; ++i) {
        temp[i] = std::abs(data[i] - median);
    }

    // Find MAD (median absolute deviation)
    scl::algo::nth_element<Real>(temp, temp + (n / 2), temp + n);
    return temp[n / 2] * Real(1.4826);  // Scale factor for normal distribution
}

} // namespace detail

// =============================================================================
// Isolation Score
// =============================================================================

template <typename T, bool IsCSR>
void isolation_score(
    const Sparse<T, IsCSR>& data,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(data.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores length must match number of cells");

    if (n_cells == 0) return;

    const Size n_features = static_cast<Size>(data.cols());

    // Compute cell-wise statistics
    auto cell_means_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* cell_means = cell_means_ptr.release();
    auto cell_vars_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* cell_vars = cell_vars_ptr.release();

    // Build offset array for CSR access
    auto row_offsets = data.build_offset_array();
    const T* values_ptr = data.contiguous_data();

    // Global statistics accumulators
    std::atomic<Real> global_sum{Real(0.0)};
    std::atomic<Real> global_sum_sq{Real(0.0)};

    auto compute_cell_stats = [&](Size i) {
        const Index row_start = row_offsets[i];
        const Index row_end = row_offsets[i + 1];

        Real sum = Real(0.0);
        Real sum_sq = Real(0.0);

        for (Index j = row_start; j < row_end; ++j) {
            Real val = static_cast<Real>(values_ptr[j]);
            sum += val;
            sum_sq += val * val;
        }

        cell_means[i] = sum / static_cast<Real>(n_features);
        Real mean_sq = sum_sq / static_cast<Real>(n_features);
        cell_vars[i] = mean_sq - cell_means[i] * cell_means[i];

        return std::make_pair(sum, sum_sq);
    };

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), n_cells, [&](Size i) {
            auto [sum, sum_sq] = compute_cell_stats(i);
            // Atomic accumulation
            Real expected = global_sum.load(std::memory_order_relaxed);
            while (!global_sum.compare_exchange_weak(expected, expected + sum,
                std::memory_order_relaxed, std::memory_order_relaxed)) {}
            expected = global_sum_sq.load(std::memory_order_relaxed);
            while (!global_sum_sq.compare_exchange_weak(expected, expected + sum_sq,
                std::memory_order_relaxed, std::memory_order_relaxed)) {}
        });
    } else {
        Real local_sum = Real(0.0);
        Real local_sum_sq = Real(0.0);
        for (Size i = 0; i < n_cells; ++i) {
            auto [sum, sum_sq] = compute_cell_stats(i);
            local_sum += sum;
            local_sum_sq += sum_sq;
        }
        global_sum.store(local_sum, std::memory_order_relaxed);
        global_sum_sq.store(local_sum_sq, std::memory_order_relaxed);
    }

    // Global normalization
    Real total_elements = static_cast<Real>(n_cells * n_features);
    Real global_mean = global_sum.load() / total_elements;
    Real global_var = global_sum_sq.load() / total_elements - global_mean * global_mean;
    Real global_std = std::sqrt(scl::algo::max2(global_var, config::EPSILON));

    // Compute isolation scores
    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), n_cells, [&](Size i) {
            Real mean_dev = std::abs(cell_means[i] - global_mean) / global_std;
            Real var_dev = std::abs(cell_vars[i] - global_var) / (global_var + config::EPSILON);
            scores.ptr[i] = (mean_dev + var_dev) / Real(2.0);
        });
    } else {
        for (Size i = 0; i < n_cells; ++i) {
            Real mean_dev = std::abs(cell_means[i] - global_mean) / global_std;
            Real var_dev = std::abs(cell_vars[i] - global_var) / (global_var + config::EPSILON);
            scores.ptr[i] = (mean_dev + var_dev) / Real(2.0);
        }
    }

    scl::memory::aligned_free(cell_means);
    scl::memory::aligned_free(cell_vars);
}

// =============================================================================
// Local Outlier Factor (LOF)
// =============================================================================

template <typename T, bool IsCSR>
void local_outlier_factor(
    const Sparse<T, IsCSR>& data,
    const Sparse<Index, IsCSR>& neighbors,
    const Sparse<Real, IsCSR>& distances,
    Array<Real> lof_scores
) {
    const Size n_cells = static_cast<Size>(data.rows());
    SCL_CHECK_DIM(n_cells == lof_scores.len, "LOF scores length must match number of cells");
    SCL_CHECK_DIM(n_cells == static_cast<Size>(neighbors.rows()),
        "Neighbors matrix rows must match data");
    SCL_CHECK_DIM(n_cells == static_cast<Size>(distances.rows()),
        "Distances matrix rows must match data");

    if (n_cells == 0) return;

    // Build offset arrays for CSR access
    auto neighbors_offsets = neighbors.build_offset_array();
    auto distances_offsets = distances.build_offset_array();
    const Real* distances_values = distances.contiguous_data();
    const Index* neighbors_indices = neighbors.contiguous_indices();

    // Determine k from neighbor graph
    Size k = 0;
    for (Size i = 0; i < n_cells; ++i) {
        Size nnz = static_cast<Size>(neighbors_offsets[i + 1] - neighbors_offsets[i]);
        if (nnz > k) k = nnz;
    }
    k = scl::algo::max2(k, config::MIN_K_NEIGHBORS);

    // Step 1: Compute k-distance for each point (parallel)
    auto k_distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* k_distances = k_distances_ptr.get();

    auto compute_k_distance = [&](Size i) {
        const Index row_start = distances_offsets[i];
        const Index row_end = distances_offsets[i + 1];
        const Size n_neighbors = static_cast<Size>(row_end - row_start);

        if (n_neighbors == 0) {
            k_distances[i] = Real(0.0);
            return;
        }

        Size actual_k = scl::algo::min2(k, n_neighbors);
        k_distances[i] = distances_values[row_start + actual_k - 1];
    };

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), n_cells, [&](Size i) {
            compute_k_distance(i);
        });
    } else {
        for (Size i = 0; i < n_cells; ++i) {
            compute_k_distance(i);
        }
    }

    // Step 2: Compute local reachability density (LRD) for each point (parallel)
    auto lrd_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* lrd = lrd_ptr.get();

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::WorkspacePool<Real> reach_pool;
        reach_pool.init(scl::threading::get_num_threads_runtime(), k);

        scl::threading::parallel_for(Size(0), n_cells, [&](Size i, Size thread_id) {
            const Index row_start = neighbors_offsets[i];
            const Index row_end = neighbors_offsets[i + 1];
            const Size n_neighbors = static_cast<Size>(row_end - row_start);

            if (n_neighbors == 0) {
                lrd[i] = Real(1.0);
                return;
            }

            Real* reach_dists = reach_pool.get(thread_id);
            for (Size j = 0; j < n_neighbors; ++j) {
                Index neighbor_idx = neighbors_indices[row_start + j];
                Real dist = distances_values[distances_offsets[i] + j];
                reach_dists[j] = detail::reachability_distance(dist, k_distances[neighbor_idx]);
            }

            lrd[i] = detail::local_reachability_density(reach_dists, n_neighbors);
        });
    } else {
        auto reach_dists_ptr = scl::memory::aligned_alloc<Real>(k, SCL_ALIGNMENT);

        Real* reach_dists = reach_dists_ptr.get();

        for (Size i = 0; i < n_cells; ++i) {
            const Index row_start = neighbors_offsets[i];
            const Index row_end = neighbors_offsets[i + 1];
            const Size n_neighbors = static_cast<Size>(row_end - row_start);

            if (n_neighbors == 0) {
                lrd[i] = Real(1.0);
                continue;
            }

            for (Size j = 0; j < n_neighbors; ++j) {
                Index neighbor_idx = neighbors_indices[row_start + j];
                Real dist = distances_values[distances_offsets[i] + j];
                reach_dists[j] = detail::reachability_distance(dist, k_distances[neighbor_idx]);
            }

            lrd[i] = detail::local_reachability_density(reach_dists, n_neighbors);
        }
    }

    // Step 3: Compute LOF for each point (parallel)
    auto compute_lof = [&](Size i) {
        const Index row_start = neighbors_offsets[i];
        const Index row_end = neighbors_offsets[i + 1];
        const Size n_neighbors = static_cast<Size>(row_end - row_start);

        if (n_neighbors == 0 || lrd[i] < config::EPSILON) {
            lof_scores.ptr[i] = Real(1.0);
            return;
        }

        Real sum_neighbor_lrd = Real(0.0);
        for (Index j = row_start; j < row_end; ++j) {
            Index neighbor_idx = neighbors_indices[j];
            sum_neighbor_lrd += lrd[neighbor_idx];
        }

        Real mean_neighbor_lrd = sum_neighbor_lrd / static_cast<Real>(n_neighbors);
        lof_scores.ptr[i] = mean_neighbor_lrd / lrd[i];
    };

    if (n_cells >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), n_cells, [&](Size i) {
            compute_lof(i);
        });
    } else {
        for (Size i = 0; i < n_cells; ++i) {
            compute_lof(i);
        }
    }

    scl::memory::aligned_free(k_distances);
    scl::memory::aligned_free(lrd);
}

// =============================================================================
// Ambient RNA Detection
// =============================================================================

template <typename T, bool IsCSR>
void ambient_detection(
    const Sparse<T, IsCSR>& expression,
    Array<Real> ambient_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    SCL_CHECK_DIM(n_cells == ambient_scores.len,
        "Ambient scores length must match number of cells");

    if (n_cells == 0 || n_genes == 0) return;

    // Build offset arrays for CSR access
    auto expression_offsets = expression.build_offset_array();
    const T* expression_values = expression.contiguous_data();
    const Index* expression_indices = expression.contiguous_indices();

    // Compute total UMI per cell
    auto total_umi_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* total_umi = total_umi_ptr.get();

    for (Size i = 0; i < n_cells; ++i) {
        Real sum = Real(0.0);
        const Index row_start = expression_offsets[i];
        const Index row_end = expression_offsets[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            sum += static_cast<Real>(expression_values[j]);
        }
        total_umi[i] = sum;
    }

    // Find low-UMI cells to estimate ambient profile
    auto sorted_umi_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* sorted_umi = sorted_umi_ptr.get();
    scl::algo::copy(total_umi, sorted_umi, n_cells);
    scl::sort::sort(Array<Real>(sorted_umi, n_cells));

    // Use bottom 10% as ambient reference
    Size n_ambient = scl::algo::max2(Size(1), n_cells / 10);
    Real umi_threshold = sorted_umi[n_ambient];

    // Compute ambient gene profile
    auto ambient_profile_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* ambient_profile = ambient_profile_ptr.get();
    for (Size g = 0; g < n_genes; ++g) {
        ambient_profile[g] = Real(0.0);
    }

    Real ambient_total = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        if (total_umi[i] > umi_threshold) continue;

        const Index row_start = expression_offsets[i];
        const Index row_end = expression_offsets[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression_indices[j];
            Real val = static_cast<Real>(expression_values[j]);
            ambient_profile[gene] += val;
            ambient_total += val;
        }
    }

    // Normalize ambient profile
    if (ambient_total > config::EPSILON) {
        for (Size g = 0; g < n_genes; ++g) {
            ambient_profile[g] /= ambient_total;
        }
    }

    // Score each cell by correlation with ambient profile
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = expression_offsets[i];
        const Index row_end = expression_offsets[i + 1];

        if (total_umi[i] < config::EPSILON) {
            ambient_scores.ptr[i] = Real(1.0);
            continue;
        }

        // Compute correlation with ambient profile
        Real dot_product = Real(0.0);
        Real cell_norm = Real(0.0);

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression_indices[j];
            Real val = static_cast<Real>(expression_values[j]) / total_umi[i];
            dot_product += val * ambient_profile[gene];
            cell_norm += val * val;
        }

        Real ambient_norm = Real(0.0);
        for (Size g = 0; g < n_genes; ++g) {
            ambient_norm += ambient_profile[g] * ambient_profile[g];
        }

        // Cosine similarity
        Real denom = std::sqrt(cell_norm * ambient_norm);
        if (denom > config::EPSILON) {
            ambient_scores.ptr[i] = dot_product / denom;
        } else {
            ambient_scores.ptr[i] = Real(0.0);
        }
    }

    scl::memory::aligned_free(total_umi);
    scl::memory::aligned_free(sorted_umi);
    scl::memory::aligned_free(ambient_profile);
}

// =============================================================================
// Empty Droplet Detection (EmptyDrops-style)
// =============================================================================

template <typename T, bool IsCSR>
void empty_drops(
    const Sparse<T, IsCSR>& raw_counts,
    Array<bool> is_empty,
    Real fdr_threshold
) {
    const Size n_cells = static_cast<Size>(raw_counts.rows());
    const Size n_genes = static_cast<Size>(raw_counts.cols());

    SCL_CHECK_DIM(n_cells == is_empty.len,
        "is_empty length must match number of cells");

    if (n_cells == 0 || n_genes == 0) return;

    // Compute total UMI per cell
    auto total_umi_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* total_umi = total_umi_ptr.release();
    auto sorted_indices_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    Index* sorted_indices = sorted_indices_ptr.release();

    for (Size i = 0; i < n_cells; ++i) {
        Real sum = Real(0.0);
        const Index row_start = raw_counts.row_indices_unsafe()[i];
        const Index row_end = raw_counts.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            sum += static_cast<Real>(raw_counts.values()[j]);
        }
        total_umi[i] = sum;
        sorted_indices[i] = static_cast<Index>(i);
    }

    // Sort by total UMI using sort_pairs
    scl::sort::sort_pairs(
        Array<Real>(total_umi, n_cells),
        Array<Index>(sorted_indices, n_cells)
    );

    // Identify ambient barcodes (lowest UMI, below threshold)
    Size n_ambient = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (total_umi[i] <= static_cast<Real>(config::EMPTY_DROPS_MAX_AMBIENT)) {
            ++n_ambient;
        } else {
            break;
        }
    }

    if (n_ambient == 0) {
        // No clear ambient population, mark all as non-empty
        for (Size i = 0; i < n_cells; ++i) {
            is_empty.ptr[i] = false;
        }
        scl::memory::aligned_free(total_umi);
        scl::memory::aligned_free(sorted_indices);
        return;
    }

    // Estimate ambient profile from lowest-UMI barcodes
    auto ambient_profile_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* ambient_profile = ambient_profile_ptr.release();
    for (Size g = 0; g < n_genes; ++g) {
        ambient_profile[g] = Real(0.0);
    }

    Real ambient_total = Real(0.0);
    for (Size i = 0; i < n_ambient; ++i) {
        Index cell_idx = sorted_indices[i];
        const Index row_start = raw_counts.row_indices_unsafe()[cell_idx];
        const Index row_end = raw_counts.row_indices_unsafe()[cell_idx + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = raw_counts.col_indices_unsafe()[j];
            Real val = static_cast<Real>(raw_counts.values()[j]);
            ambient_profile[gene] += val;
            ambient_total += val;
        }
    }

    // Normalize and add pseudocount
    if (ambient_total > config::EPSILON) {
        for (Size g = 0; g < n_genes; ++g) {
            ambient_profile[g] = (ambient_profile[g] + Real(1.0)) / (ambient_total + static_cast<Real>(n_genes));
        }
    }

    // Test each cell against ambient profile
    auto p_values_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* p_values = p_values_ptr.release();
    auto cell_profile_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* cell_profile = cell_profile_ptr.release();

    for (Size i = 0; i < n_cells; ++i) {
        Real umi = total_umi[i];

        // Cells with very low UMI are likely empty
        if (umi < static_cast<Real>(config::EMPTY_DROPS_MIN_UMI)) {
            p_values[i] = Real(1.0);
            continue;
        }

        // Build cell profile
        for (Size g = 0; g < n_genes; ++g) {
            cell_profile[g] = Real(0.0);
        }

        const Index row_start = raw_counts.row_indices_unsafe()[i];
        const Index row_end = raw_counts.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = raw_counts.col_indices_unsafe()[j];
            cell_profile[gene] = static_cast<Real>(raw_counts.values()[j]);
        }

        // Compute deviance statistic (goodness of fit test)
        Real deviance = Real(0.0);
        for (Size g = 0; g < n_genes; ++g) {
            Real observed = cell_profile[g];
            Real expected = ambient_profile[g] * umi;

            if (observed > Real(0.0) && expected > Real(0.0)) {
                deviance += Real(2.0) * observed * std::log(observed / expected);
            }
        }

        // P-value from chi-squared distribution
        Real df = static_cast<Real>(n_genes - 1);
        p_values[i] = Real(1.0) - detail::chi2_cdf_approx(deviance, df);
    }

    // Apply BH correction and determine empty status
    auto p_order_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    Index* p_order = p_order_ptr.release();
    auto p_values_copy_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* p_values_copy = p_values_copy_ptr.release();
    for (Size i = 0; i < n_cells; ++i) {
        p_order[i] = static_cast<Index>(i);
        p_values_copy[i] = p_values[i];
    }
    scl::sort::sort_pairs(
        Array<Real>(p_values_copy, n_cells),
        Array<Index>(p_order, n_cells)
    );

    // BH correction
    auto adj_p_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* adj_p = adj_p_ptr.release();
    const Real n_real = static_cast<Real>(n_cells);

    for (Size i = 0; i < n_cells; ++i) {
        Real rank = static_cast<Real>(i + 1);
        adj_p[p_order[i]] = p_values[p_order[i]] * n_real / rank;
    }

    // Enforce monotonicity
    for (Size i = n_cells - 1; i > 0; --i) {
        adj_p[p_order[i - 1]] = scl::algo::min2(adj_p[p_order[i - 1]], adj_p[p_order[i]]);
    }
    for (Size i = 0; i < n_cells; ++i) {
        adj_p[i] = scl::algo::min2(adj_p[i], Real(1.0));
    }

    // Mark empty droplets
    for (Size i = 0; i < n_cells; ++i) {
        // Empty if adjusted p-value > threshold (fails to reject ambient hypothesis)
        // OR if total UMI is very low
        is_empty.ptr[i] = (adj_p[i] > (Real(1.0) - fdr_threshold)) ||
                          (total_umi[sorted_indices[i]] < static_cast<Real>(config::EMPTY_DROPS_MIN_UMI));
    }

    scl::memory::aligned_free(total_umi);
    scl::memory::aligned_free(sorted_indices);
    scl::memory::aligned_free(p_values_copy);
    scl::memory::aligned_free(ambient_profile);
    scl::memory::aligned_free(p_values);
    scl::memory::aligned_free(cell_profile);
    scl::memory::aligned_free(p_order);
    scl::memory::aligned_free(adj_p);
}

// =============================================================================
// Outlier Gene Detection
// =============================================================================

template <typename T, bool IsCSR>
void outlier_genes(
    const Sparse<T, IsCSR>& expression,
    Index* outlier_gene_indices,
    Size& n_outliers,
    Real threshold
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());

    n_outliers = 0;
    if (n_cells == 0 || n_genes == 0) return;

    // Compute gene-wise statistics
    auto gene_means_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* gene_means = gene_means_ptr.release();
    auto gene_vars_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* gene_vars = gene_vars_ptr.release();
    auto gene_nnz_ptr = scl::memory::aligned_alloc<Size>(n_genes, SCL_ALIGNMENT);

    Size* gene_nnz = gene_nnz_ptr.release();

    for (Size g = 0; g < n_genes; ++g) {
        gene_means[g] = Real(0.0);
        gene_vars[g] = Real(0.0);
        gene_nnz[g] = 0;
    }

    // First pass: compute sums
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression.col_indices_unsafe()[j];
            Real val = static_cast<Real>(expression.values()[j]);
            gene_means[gene] += val;
            ++gene_nnz[gene];
        }
    }

    // Compute means
    for (Size g = 0; g < n_genes; ++g) {
        gene_means[g] /= static_cast<Real>(n_cells);
    }

    // Second pass: compute variances
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        // Track which genes have non-zero values in this cell
        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression.col_indices_unsafe()[j];
            Real val = static_cast<Real>(expression.values()[j]);
            Real diff = val - gene_means[gene];
            gene_vars[gene] += diff * diff;
        }
    }

    // Add variance contribution from zero entries
    for (Size g = 0; g < n_genes; ++g) {
        Size n_zeros = n_cells - gene_nnz[g];
        gene_vars[g] += static_cast<Real>(n_zeros) * gene_means[g] * gene_means[g];
        gene_vars[g] /= static_cast<Real>(n_cells - 1);
    }

    // Compute log mean and log CV^2 for outlier detection
    auto log_means_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* log_means = log_means_ptr.release();
    auto log_cv2_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* log_cv2 = log_cv2_ptr.release();
    Size valid_genes = 0;

    for (Size g = 0; g < n_genes; ++g) {
        if (gene_means[g] > config::EPSILON) {
            log_means[valid_genes] = std::log(gene_means[g]);
            Real cv2 = gene_vars[g] / (gene_means[g] * gene_means[g]);
            log_cv2[valid_genes] = std::log(cv2 + config::EPSILON);
            ++valid_genes;
        }
    }

    if (valid_genes < 3) {
        scl::memory::aligned_free(gene_means);
        scl::memory::aligned_free(gene_vars);
        scl::memory::aligned_free(gene_nnz);
        scl::memory::aligned_free(log_means);
        scl::memory::aligned_free(log_cv2);
        return;
    }

    // Compute median and MAD of log CV^2
    auto temp_ptr = scl::memory::aligned_alloc<Real>(valid_genes, SCL_ALIGNMENT);

    Real* temp = temp_ptr.release();
    Real mad = detail::compute_mad(log_cv2, valid_genes, temp);

    // Find median of log_cv2
    scl::algo::copy(log_cv2, temp, valid_genes);
    scl::algo::nth_element(temp, temp + (valid_genes / 2), temp + valid_genes);
    Real median_log_cv2 = temp[valid_genes / 2];

    scl::memory::aligned_free(temp);

    // Identify outlier genes
    Size gene_idx = 0;
    for (Size g = 0; g < n_genes; ++g) {
        if (gene_means[g] > config::EPSILON) {
            Real z_score = std::abs(log_cv2[gene_idx] - median_log_cv2) / (mad + config::EPSILON);
            if (z_score > threshold) {
                outlier_gene_indices[n_outliers++] = static_cast<Index>(g);
            }
            ++gene_idx;
        }
    }

    scl::memory::aligned_free(gene_means);
    scl::memory::aligned_free(gene_vars);
    scl::memory::aligned_free(gene_nnz);
    scl::memory::aligned_free(log_means);
    scl::memory::aligned_free(log_cv2);
}

// =============================================================================
// Additional outlier detection methods
// =============================================================================

// Detect doublets based on expression profiles
template <typename T, bool IsCSR>
void doublet_score(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores length must match number of cells");

    if (n_cells == 0) return;

    // Doublet detection based on neighborhood mixing
    // Cells that appear as "intermediate" between clusters are potential doublets

    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start_n = neighbors.row_indices_unsafe()[i];
        const Index row_end_n = neighbors.row_indices_unsafe()[i + 1];
        const Size n_neighbors = static_cast<Size>(row_end_n - row_start_n);

        if (n_neighbors < 2) {
            scores.ptr[i] = Real(0.0);
            continue;
        }

        Real total_variance = Real(0.0);
        Size features_counted = 0;

        // For each feature, compute variance across neighbors
        const Index cell_row_start = expression.row_indices_unsafe()[i];
        const Index cell_row_end = expression.row_indices_unsafe()[i + 1];

        for (Index f = cell_row_start; f < cell_row_end; ++f) {
            Index feature = expression.col_indices_unsafe()[f];
            Real cell_val = static_cast<Real>(expression.values()[f]);

            Real sum = Real(0.0);
            Real sum_sq = Real(0.0);
            Size count = 0;

            for (Index j = row_start_n; j < row_end_n; ++j) {
                Index neighbor = neighbors.col_indices_unsafe()[j];

                // Find this feature in neighbor's expression
                const Index n_row_start = expression.row_indices_unsafe()[neighbor];
                const Index n_row_end = expression.row_indices_unsafe()[neighbor + 1];

                for (Index k = n_row_start; k < n_row_end; ++k) {
                    if (expression.col_indices_unsafe()[k] == feature) {
                        Real val = static_cast<Real>(expression.values()[k]);
                        sum += val;
                        sum_sq += val * val;
                        ++count;
                        break;
                    }
                }
            }

            if (count > 1) {
                Real mean = sum / static_cast<Real>(count);
                Real var = sum_sq / static_cast<Real>(count) - mean * mean;

                // Compare cell value to neighbor distribution
                Real z = std::abs(cell_val - mean) / (std::sqrt(var) + config::EPSILON);
                total_variance += z;
                ++features_counted;
            }
        }

        // Score: average z-score across features
        // Low z-scores (cell similar to neighbors) suggest non-doublet
        // High z-scores (cell dissimilar to neighbors) could indicate doublet
        if (features_counted > 0) {
            scores.ptr[i] = total_variance / static_cast<Real>(features_counted);
        } else {
            scores.ptr[i] = Real(0.0);
        }
    }
}

// Detect cells with high mitochondrial content
template <typename T, bool IsCSR>
void mitochondrial_outliers(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> mito_genes,
    Array<Real> mito_fraction,
    Array<bool> is_outlier,
    Real threshold
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == mito_fraction.len, "Mito fraction length must match cells");
    SCL_CHECK_DIM(n_cells == is_outlier.len, "is_outlier length must match cells");

    if (n_cells == 0) return;

    // Create lookup for mitochondrial genes
    Index max_gene = 0;
    for (Size i = 0; i < mito_genes.len; ++i) {
        if (mito_genes.ptr[i] > max_gene) {
            max_gene = mito_genes.ptr[i];
        }
    }

    auto is_mito_ptr = scl::memory::aligned_alloc<bool>(max_gene + 1, SCL_ALIGNMENT);


    bool* is_mito = is_mito_ptr.release();
    for (Index g = 0; g <= max_gene; ++g) {
        is_mito[g] = false;
    }
    for (Size i = 0; i < mito_genes.len; ++i) {
        is_mito[mito_genes.ptr[i]] = true;
    }

    // Compute mitochondrial fraction for each cell
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        Real total = Real(0.0);
        Real mito_total = Real(0.0);

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression.col_indices_unsafe()[j];
            Real val = static_cast<Real>(expression.values()[j]);
            total += val;

            if (gene <= max_gene && is_mito[gene]) {
                mito_total += val;
            }
        }

        if (total > config::EPSILON) {
            mito_fraction.ptr[i] = mito_total / total;
        } else {
            mito_fraction.ptr[i] = Real(0.0);
        }

        is_outlier.ptr[i] = mito_fraction.ptr[i] > threshold;
    }

    scl::memory::aligned_free(is_mito);
}

// Combined QC filtering
template <typename T, bool IsCSR>
void qc_filter(
    const Sparse<T, IsCSR>& expression,
    Real min_genes,
    Real max_genes,
    Real min_counts,
    Real max_counts,
    Real max_mito_fraction,
    Array<const Index> mito_genes,
    Array<bool> pass_qc
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == pass_qc.len, "pass_qc length must match cells");

    if (n_cells == 0) return;

    // Create mito gene lookup
    Index max_gene_idx = 0;
    for (Size i = 0; i < mito_genes.len; ++i) {
        if (mito_genes.ptr[i] > max_gene_idx) {
            max_gene_idx = mito_genes.ptr[i];
        }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<bool[], scl::memory::AlignedDeleter<bool>> is_mito;
    if (mito_genes.len > 0) {
        is_mito = scl::memory::aligned_alloc<bool>(max_gene_idx + 1, SCL_ALIGNMENT);
        for (Index g = 0; g <= max_gene_idx; ++g) {
            is_mito[g] = false;
        }
        for (Size i = 0; i < mito_genes.len; ++i) {
            is_mito[mito_genes.ptr[i]] = true;
        }
    }

    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];
        const Size n_genes_detected = static_cast<Size>(row_end - row_start);

        Real total_counts = Real(0.0);
        Real mito_counts = Real(0.0);

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression.col_indices_unsafe()[j];
            Real val = static_cast<Real>(expression.values()[j]);
            total_counts += val;

            if (is_mito != nullptr && gene <= max_gene_idx && is_mito[gene]) {
                mito_counts += val;
            }
        }

        Real mito_frac = (total_counts > config::EPSILON) ?
            (mito_counts / total_counts) : Real(0.0);

        // Apply filters
        bool pass = true;
        pass = pass && (static_cast<Real>(n_genes_detected) >= min_genes);
        pass = pass && (static_cast<Real>(n_genes_detected) <= max_genes);
        pass = pass && (total_counts >= min_counts);
        pass = pass && (total_counts <= max_counts);
        pass = pass && (mito_frac <= max_mito_fraction);

        pass_qc.ptr[i] = pass;
    }
    // is_mito will be automatically freed by unique_ptr and its custom deleter
}

} // namespace scl::kernel::outlier
