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
// FILE: scl/kernel/association.hpp
// BRIEF: Feature association analysis across modalities (RNA + ATAC)
//
// APPLICATIONS:
// - Gene-peak correlation for multi-omic data
// - Cis-regulatory element identification
// - Enhancer-gene linking
// - Multi-modal neighborhood construction
// =============================================================================

namespace scl::kernel::association {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real MIN_CORRELATION = Real(0.1);
    constexpr Size MIN_CELLS_FOR_CORRELATION = 10;
    constexpr Size MAX_LINKS_PER_GENE = 1000;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

namespace detail {

// Binary search for feature in sorted CSR indices
SCL_FORCE_INLINE Index binary_search_feature(
    const Index* indices,
    Index start,
    Index end,
    Index feature
) noexcept {
    while (start < end) {
        Index mid = start + (end - start) / 2;
        if (indices[mid] < feature) {
            start = mid + 1;
        } else if (indices[mid] > feature) {
            end = mid;
        } else {
            return mid;
        }
    }
    return -1;
}

// Pearson correlation between two sparse vectors
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real pearson_correlation(
    const Sparse<T, IsCSR>& data1,
    Index feature1,
    const Sparse<T, IsCSR>& data2,
    Index feature2,
    Size n_cells
) {
    if (n_cells < config::MIN_CELLS_FOR_CORRELATION) {
        return Real(0.0);
    }

    // For CSR: need to iterate by cells
    Real sum_x = Real(0.0), sum_y = Real(0.0);
    Real sum_xx = Real(0.0), sum_yy = Real(0.0);
    Real sum_xy = Real(0.0);
    Size n_valid = 0;

    // This is expensive - we need to gather values per cell
    Real* vals1 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* vals2 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    for (Size c = 0; c < n_cells; ++c) {
        vals1[c] = Real(0.0);
        vals2[c] = Real(0.0);
    }

    // Gather values for feature1 from data1 using binary search
    for (Size c = 0; c < n_cells; ++c) {
        const Index row_start = data1.row_indices_unsafe()[c];
        const Index row_end = data1.row_indices_unsafe()[c + 1];
        Index pos = binary_search_feature(data1.col_indices_unsafe(), row_start, row_end, feature1);
        if (pos >= 0) {
            vals1[c] = static_cast<Real>(data1.values()[pos]);
        }
    }

    // Gather values for feature2 from data2 using binary search
    for (Size c = 0; c < n_cells; ++c) {
        const Index row_start = data2.row_indices_unsafe()[c];
        const Index row_end = data2.row_indices_unsafe()[c + 1];
        Index pos = binary_search_feature(data2.col_indices_unsafe(), row_start, row_end, feature2);
        if (pos >= 0) {
            vals2[c] = static_cast<Real>(data2.values()[pos]);
        }
    }

    // Compute correlation
    for (Size c = 0; c < n_cells; ++c) {
        sum_x += vals1[c];
        sum_y += vals2[c];
        sum_xx += vals1[c] * vals1[c];
        sum_yy += vals2[c] * vals2[c];
        sum_xy += vals1[c] * vals2[c];
    }
    n_valid = n_cells;

    scl::memory::aligned_free(vals1);
    scl::memory::aligned_free(vals2);

    if (n_valid < config::MIN_CELLS_FOR_CORRELATION) {
        return Real(0.0);
    }

    Real n = static_cast<Real>(n_valid);
    Real mean_x = sum_x / n;
    Real mean_y = sum_y / n;

    Real cov_xy = sum_xy / n - mean_x * mean_y;
    Real var_x = sum_xx / n - mean_x * mean_x;
    Real var_y = sum_yy / n - mean_y * mean_y;

    if (var_x < config::EPSILON || var_y < config::EPSILON) {
        return Real(0.0);
    }

    return cov_xy / (std::sqrt(var_x) * std::sqrt(var_y));
}

// Spearman correlation via rank transformation
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real spearman_correlation(
    const Sparse<T, IsCSR>& data1,
    Index feature1,
    const Sparse<T, IsCSR>& data2,
    Index feature2,
    Size n_cells
) {
    if (n_cells < config::MIN_CELLS_FOR_CORRELATION) {
        return Real(0.0);
    }

    // Gather values
    Real* vals1 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* vals2 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* ranks1 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* ranks2 = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Index* indices = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);
    Real* sorted_vals = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    for (Size c = 0; c < n_cells; ++c) {
        vals1[c] = Real(0.0);
        vals2[c] = Real(0.0);
        indices[c] = static_cast<Index>(c);
    }

    // Gather values for feature1 using binary search
    for (Size c = 0; c < n_cells; ++c) {
        const Index row_start = data1.row_indices_unsafe()[c];
        const Index row_end = data1.row_indices_unsafe()[c + 1];
        Index pos = binary_search_feature(data1.col_indices_unsafe(), row_start, row_end, feature1);
        if (pos >= 0) {
            vals1[c] = static_cast<Real>(data1.values()[pos]);
        }
    }

    // Gather values for feature2 using binary search
    for (Size c = 0; c < n_cells; ++c) {
        const Index row_start = data2.row_indices_unsafe()[c];
        const Index row_end = data2.row_indices_unsafe()[c + 1];
        Index pos = binary_search_feature(data2.col_indices_unsafe(), row_start, row_end, feature2);
        if (pos >= 0) {
            vals2[c] = static_cast<Real>(data2.values()[pos]);
        }
    }

    // Compute ranks for vals1 using VQSort
    scl::algo::copy(vals1, sorted_vals, n_cells);
    for (Size i = 0; i < n_cells; ++i) indices[i] = static_cast<Index>(i);
    scl::sort::sort_pairs(
        Array<Real>(sorted_vals, n_cells),
        Array<Index>(indices, n_cells)
    );
    for (Size i = 0; i < n_cells; ++i) {
        ranks1[indices[i]] = static_cast<Real>(i + 1);
    }

    // Compute ranks for vals2 using VQSort
    scl::algo::copy(vals2, sorted_vals, n_cells);
    for (Size i = 0; i < n_cells; ++i) indices[i] = static_cast<Index>(i);
    scl::sort::sort_pairs(
        Array<Real>(sorted_vals, n_cells),
        Array<Index>(indices, n_cells)
    );
    for (Size i = 0; i < n_cells; ++i) {
        ranks2[indices[i]] = static_cast<Real>(i + 1);
    }

    // Compute Pearson on ranks
    Real sum_d2 = Real(0.0);
    for (Size c = 0; c < n_cells; ++c) {
        Real d = ranks1[c] - ranks2[c];
        sum_d2 += d * d;
    }

    Real n = static_cast<Real>(n_cells);
    Real rho = Real(1.0) - (Real(6.0) * sum_d2) / (n * (n * n - Real(1.0)));

    scl::memory::aligned_free(vals1);
    scl::memory::aligned_free(vals2);
    scl::memory::aligned_free(ranks1);
    scl::memory::aligned_free(ranks2);
    scl::memory::aligned_free(indices);
    scl::memory::aligned_free(sorted_vals);

    return rho;
}

} // namespace detail

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

template <typename T, bool IsCSR_RNA, bool IsCSR_ATAC>
void gene_peak_correlation(
    const Sparse<T, IsCSR_RNA>& rna_expression,
    const Sparse<T, IsCSR_ATAC>& atac_accessibility,
    Index* gene_indices,
    Index* peak_indices,
    Real* correlations,
    Size& n_correlations,
    Real min_correlation
) {
    const Size n_cells = static_cast<Size>(rna_expression.rows());
    const Size n_genes = static_cast<Size>(rna_expression.cols());
    const Size n_peaks = static_cast<Size>(atac_accessibility.cols());

    SCL_CHECK_DIM(n_cells == static_cast<Size>(atac_accessibility.rows()),
        "RNA and ATAC must have same number of cells");

    n_correlations = 0;
    if (n_cells < config::MIN_CELLS_FOR_CORRELATION || n_genes == 0 || n_peaks == 0) {
        return;
    }

    // Compute correlations for all gene-peak pairs
    // This is expensive O(n_genes * n_peaks * n_cells)

    for (Size g = 0; g < n_genes; ++g) {
        for (Size p = 0; p < n_peaks; ++p) {
            Real corr = detail::pearson_correlation(
                rna_expression, static_cast<Index>(g),
                atac_accessibility, static_cast<Index>(p),
                n_cells);

            Real abs_corr = (corr >= Real(0.0)) ? corr : -corr;
            if (abs_corr >= min_correlation) {
                gene_indices[n_correlations] = static_cast<Index>(g);
                peak_indices[n_correlations] = static_cast<Index>(p);
                correlations[n_correlations] = corr;
                ++n_correlations;
            }
        }
    }
}

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

template <typename T, bool IsCSR>
void cis_regulatory(
    const Sparse<T, IsCSR>& rna_expression,
    const Sparse<T, IsCSR>& atac_accessibility,
    const Index* gene_indices,
    const Index* peak_indices,
    Size n_pairs,
    Real* correlations,
    Real* p_values
) {
    const Size n_cells = static_cast<Size>(rna_expression.rows());

    SCL_CHECK_DIM(n_cells == static_cast<Size>(atac_accessibility.rows()),
        "RNA and ATAC must have same number of cells");

    if (n_pairs == 0 || n_cells < config::MIN_CELLS_FOR_CORRELATION) {
        return;
    }

    // Parallel over pairs
    scl::threading::parallel_for(Size(0), n_pairs, [&](size_t i) {
        Index gene = gene_indices[i];
        Index peak = peak_indices[i];

        correlations[i] = detail::pearson_correlation(
            rna_expression, gene,
            atac_accessibility, peak,
            n_cells);

        // P-value using t-distribution approximation
        Real r = correlations[i];
        Real n = static_cast<Real>(n_cells);

        Real abs_r = (r >= Real(0.0)) ? r : -r;
        if (abs_r > Real(1.0) - config::EPSILON) {
            p_values[i] = Real(0.0);
        } else {
            Real t = r * std::sqrt((n - Real(2.0)) / (Real(1.0) - r * r));
            // Approximate two-tailed p-value
            Real df = n - Real(2.0);
            Real abs_t = (t >= Real(0.0)) ? t : -t;
            // Use normal approximation for large df
            Real z = abs_t * std::sqrt(Real(1.0) - Real(0.25) / df);
            Real p = Real(2.0) * (Real(1.0) - Real(0.5) * (Real(1.0) +
                std::erf(z / std::sqrt(Real(2.0)))));
            p_values[i] = p;
        }
    });
}

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

template <typename T, bool IsCSR>
void enhancer_gene_link(
    const Sparse<T, IsCSR>& rna,
    const Sparse<T, IsCSR>& atac,
    Real correlation_threshold,
    Index* link_genes,
    Index* link_peaks,
    Real* link_correlations,
    Size& n_links
) {
    const Size n_cells = static_cast<Size>(rna.rows());
    const Size n_genes = static_cast<Size>(rna.cols());
    const Size n_peaks = static_cast<Size>(atac.cols());

    SCL_CHECK_DIM(n_cells == static_cast<Size>(atac.rows()),
        "RNA and ATAC must have same number of cells");

    n_links = 0;
    if (n_cells < config::MIN_CELLS_FOR_CORRELATION) return;

    correlation_threshold = scl::algo::max2(correlation_threshold, config::MIN_CORRELATION);

    // Identify significant gene-peak links
    for (Size g = 0; g < n_genes; ++g) {
        Size links_for_gene = 0;

        for (Size p = 0; p < n_peaks && links_for_gene < config::MAX_LINKS_PER_GENE; ++p) {
            Real corr = detail::pearson_correlation(
                rna, static_cast<Index>(g),
                atac, static_cast<Index>(p),
                n_cells);

            // Only positive correlations for enhancer-gene links
            if (corr >= correlation_threshold) {
                link_genes[n_links] = static_cast<Index>(g);
                link_peaks[n_links] = static_cast<Index>(p);
                link_correlations[n_links] = corr;
                ++n_links;
                ++links_for_gene;
            }
        }
    }
}

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

template <typename T, bool IsCSR>
void multimodal_neighbors(
    const Sparse<T, IsCSR>& modality1,
    const Sparse<T, IsCSR>& modality2,
    Real weight1,
    Real weight2,
    Index k,
    Index* neighbor_indices,
    Real* neighbor_distances
) {
    const Size n_cells = static_cast<Size>(modality1.rows());

    SCL_CHECK_DIM(n_cells == static_cast<Size>(modality2.rows()),
        "Both modalities must have same number of cells");

    if (n_cells == 0 || k == 0) return;

    k = scl::algo::min2(k, static_cast<Index>(n_cells - 1));

    // Normalize weights
    Real total_weight = weight1 + weight2;
    weight1 /= total_weight;
    weight2 /= total_weight;

    // Create workspace pools for thread-local buffers
    scl::threading::WorkspacePool<Real> dists_pool;
    scl::threading::WorkspacePool<Index> indices_pool;
    dists_pool.init(scl::threading::Scheduler::get_num_threads(), n_cells);
    indices_pool.init(scl::threading::Scheduler::get_num_threads(), n_cells);

    // Parallel over cells
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
        Real* all_dists = dists_pool.get(thread_rank);
        Index* all_indices = indices_pool.get(thread_rank);

        // Compute distances from cell i to all others
        for (Size j = 0; j < n_cells; ++j) {
            if (i == j) {
                all_dists[j] = std::numeric_limits<Real>::max();
                all_indices[j] = static_cast<Index>(j);
                continue;
            }

            // Distance in modality 1
            Real dist1 = Real(0.0);
            const Index start1_i = modality1.row_indices_unsafe()[i];
            const Index end1_i = modality1.row_indices_unsafe()[i + 1];
            const Index start1_j = modality1.row_indices_unsafe()[j];
            const Index end1_j = modality1.row_indices_unsafe()[j + 1];

            Index i1 = start1_i, j1 = start1_j;
            while (i1 < end1_i && j1 < end1_j) {
                Index col_i = modality1.col_indices_unsafe()[i1];
                Index col_j = modality1.col_indices_unsafe()[j1];
                if (col_i == col_j) {
                    Real diff = static_cast<Real>(modality1.values()[i1]) -
                               static_cast<Real>(modality1.values()[j1]);
                    dist1 += diff * diff;
                    ++i1; ++j1;
                } else if (col_i < col_j) {
                    Real val = static_cast<Real>(modality1.values()[i1]);
                    dist1 += val * val;
                    ++i1;
                } else {
                    Real val = static_cast<Real>(modality1.values()[j1]);
                    dist1 += val * val;
                    ++j1;
                }
            }
            while (i1 < end1_i) {
                Real val = static_cast<Real>(modality1.values()[i1++]);
                dist1 += val * val;
            }
            while (j1 < end1_j) {
                Real val = static_cast<Real>(modality1.values()[j1++]);
                dist1 += val * val;
            }

            // Distance in modality 2
            Real dist2 = Real(0.0);
            auto row2_i_vals = modality2.row_values_unsafe(i);
            auto row2_i_idxs = modality2.row_indices_unsafe(i);
            Index row2_i_len = modality2.row_length_unsafe(i);
            
            auto row2_j_vals = modality2.row_values_unsafe(j);
            auto row2_j_idxs = modality2.row_indices_unsafe(j);
            Index row2_j_len = modality2.row_length_unsafe(j);

            Index i2 = 0, j2 = 0;
            while (i2 < row2_i_len && j2 < row2_j_len) {
                Index col_i = row2_i_idxs.ptr[i2];
                Index col_j = row2_j_idxs.ptr[j2];
                if (col_i == col_j) {
                    Real diff = static_cast<Real>(row2_i_vals.ptr[i2]) -
                               static_cast<Real>(row2_j_vals.ptr[j2]);
                    dist2 += diff * diff;
                    ++i2; ++j2;
                } else if (col_i < col_j) {
                    Real val = static_cast<Real>(row2_i_vals.ptr[i2]);
                    dist2 += val * val;
                    ++i2;
                } else {
                    Real val = static_cast<Real>(row2_j_vals.ptr[j2]);
                    dist2 += val * val;
                    ++j2;
                }
            }
            while (i2 < row2_i_len) {
                Real val = static_cast<Real>(row2_i_vals.ptr[i2++]);
                dist2 += val * val;
            }
            while (j2 < row2_j_len) {
                Real val = static_cast<Real>(row2_j_vals.ptr[j2++]);
                dist2 += val * val;
            }

            // Combined distance
            all_dists[j] = weight1 * std::sqrt(dist1) + weight2 * std::sqrt(dist2);
            all_indices[j] = static_cast<Index>(j);
        }

        // Find k nearest by sorting indices by distance
        // Initialize indices
        for (Size j = 0; j < n_cells; ++j) {
            all_indices[j] = static_cast<Index>(j);
        }
        // Sort indices by distance (k smallest)
        std::partial_sort(all_indices, all_indices + k, all_indices + n_cells,
            [&](Index a, Index b) { return all_dists[a] < all_dists[b]; });

        for (Index ki = 0; ki < k; ++ki) {
            neighbor_indices[i * k + ki] = all_indices[ki];
            neighbor_distances[i * k + ki] = all_dists[ki];
        }
    });
}

// =============================================================================
// Feature Coupling
// =============================================================================

template <typename T, bool IsCSR>
void feature_coupling(
    const Sparse<T, IsCSR>& modality1,
    const Sparse<T, IsCSR>& modality2,
    Index* feature1_indices,
    Index* feature2_indices,
    Real* coupling_scores,
    Size& n_couplings,
    Real min_score
) {
    const Size n_cells = static_cast<Size>(modality1.rows());
    const Size n_features1 = static_cast<Size>(modality1.cols());
    const Size n_features2 = static_cast<Size>(modality2.cols());

    SCL_CHECK_DIM(n_cells == static_cast<Size>(modality2.rows()),
        "Both modalities must have same number of cells");

    n_couplings = 0;
    if (n_cells < config::MIN_CELLS_FOR_CORRELATION) return;

    // Compute mutual information-based coupling
    for (Size f1 = 0; f1 < n_features1; ++f1) {
        for (Size f2 = 0; f2 < n_features2; ++f2) {
            Real corr = detail::spearman_correlation(
                modality1, static_cast<Index>(f1),
                modality2, static_cast<Index>(f2),
                n_cells);
            Real abs_corr = (corr >= Real(0.0)) ? corr : -corr;

            if (abs_corr >= min_score) {
                feature1_indices[n_couplings] = static_cast<Index>(f1);
                feature2_indices[n_couplings] = static_cast<Index>(f2);
                coupling_scores[n_couplings] = abs_corr;
                ++n_couplings;
            }
        }
    }
}

// =============================================================================
// Additional association methods
// =============================================================================

// Compute correlation for specific cell subset
template <typename T, bool IsCSR>
void correlation_in_subset(
    const Sparse<T, IsCSR>& data1,
    Index feature1,
    const Sparse<T, IsCSR>& data2,
    Index feature2,
    Array<const Index> cell_indices,
    Real& correlation
) {
    const Size n_subset = cell_indices.len;

    if (n_subset < config::MIN_CELLS_FOR_CORRELATION) {
        correlation = Real(0.0);
        return;
    }

    Real* vals1 = scl::memory::aligned_alloc<Real>(n_subset, SCL_ALIGNMENT);
    Real* vals2 = scl::memory::aligned_alloc<Real>(n_subset, SCL_ALIGNMENT);

    for (Size s = 0; s < n_subset; ++s) {
        Index c = cell_indices.ptr[s];
        vals1[s] = Real(0.0);
        vals2[s] = Real(0.0);

        const Index row_start1 = data1.row_indices_unsafe()[c];
        const Index row_end1 = data1.row_indices_unsafe()[c + 1];
        Index pos1 = detail::binary_search_feature(data1.col_indices_unsafe(), row_start1, row_end1, feature1);
        if (pos1 >= 0) {
            vals1[s] = static_cast<Real>(data1.values()[pos1]);
        }

        const Index row_start2 = data2.row_indices_unsafe()[c];
        const Index row_end2 = data2.row_indices_unsafe()[c + 1];
        Index pos2 = detail::binary_search_feature(data2.col_indices_unsafe(), row_start2, row_end2, feature2);
        if (pos2 >= 0) {
            vals2[s] = static_cast<Real>(data2.values()[pos2]);
        }
    }

    // Compute Pearson correlation
    Real sum_x = Real(0.0), sum_y = Real(0.0);
    Real sum_xx = Real(0.0), sum_yy = Real(0.0), sum_xy = Real(0.0);

    for (Size s = 0; s < n_subset; ++s) {
        sum_x += vals1[s];
        sum_y += vals2[s];
        sum_xx += vals1[s] * vals1[s];
        sum_yy += vals2[s] * vals2[s];
        sum_xy += vals1[s] * vals2[s];
    }

    Real n = static_cast<Real>(n_subset);
    Real mean_x = sum_x / n;
    Real mean_y = sum_y / n;

    Real cov_xy = sum_xy / n - mean_x * mean_y;
    Real var_x = sum_xx / n - mean_x * mean_x;
    Real var_y = sum_yy / n - mean_y * mean_y;

    if (var_x < config::EPSILON || var_y < config::EPSILON) {
        correlation = Real(0.0);
    } else {
        correlation = cov_xy / (std::sqrt(var_x) * std::sqrt(var_y));
    }

    scl::memory::aligned_free(vals1);
    scl::memory::aligned_free(vals2);
}

// Peak-to-gene score (activity score)
template <typename T, bool IsCSR>
void peak_to_gene_activity(
    const Sparse<T, IsCSR>& atac,
    const Index* peak_to_gene_map,  // For each peak, which gene it maps to (-1 for none)
    Size n_peaks,
    Size n_genes,
    Real* gene_activity  // [n_cells * n_genes]
) {
    const Size n_cells = static_cast<Size>(atac.rows());

    // Initialize and aggregate in parallel over cells
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t c) {
        // Initialize this cell's activity to zero
        Real* cell_activity = gene_activity + c * n_genes;
        scl::algo::zero(cell_activity, n_genes);

        // Aggregate peak accessibility to gene activity
        const Index row_start = atac.row_indices_unsafe()[c];
        const Index row_end = atac.row_indices_unsafe()[c + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index peak = atac.col_indices_unsafe()[j];
            if (peak < static_cast<Index>(n_peaks)) {
                Index gene = peak_to_gene_map[peak];
                if (gene >= 0 && gene < static_cast<Index>(n_genes)) {
                    cell_activity[gene] += static_cast<Real>(atac.values()[j]);
                }
            }
        }
    });
}

} // namespace scl::kernel::association
