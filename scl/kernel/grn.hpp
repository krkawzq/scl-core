#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/grn.hpp
// BRIEF: Gene regulatory network inference (OPTIMIZED)
// =============================================================================

namespace scl::kernel::grn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_CORRELATION_THRESHOLD = Real(0.3);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Index DEFAULT_N_TREES = 100;
    constexpr Index DEFAULT_SUBSAMPLE = 500;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// GRN Method Types
// =============================================================================

enum class GRNMethod {
    Correlation,
    PartialCorrelation,
    MutualInformation,
    GENIE3,
    Combined
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Fast PRNG for random forests
struct FastRNG {
    uint64_t state;

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint64_t>(n));
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }
};

// SIMD-optimized mean computation
SCL_HOT SCL_FORCE_INLINE Real compute_mean(const Real* SCL_RESTRICT values, Index n) {
    if (SCL_UNLIKELY(n == 0)) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Index i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, values + i));
        v_sum1 = s::Add(v_sum1, s::Load(d, values + i + lanes));
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < n; ++i) {
        sum += values[i];
    }

    return sum / static_cast<Real>(n);
}

// SIMD-optimized variance computation
SCL_HOT SCL_FORCE_INLINE Real compute_variance(const Real* SCL_RESTRICT values, Index n, Real mean) {
    if (SCL_UNLIKELY(n <= 1)) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_mean = s::Set(d, mean);
    auto v_var0 = s::Zero(d);
    auto v_var1 = s::Zero(d);

    Index i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        auto v0 = s::Sub(s::Load(d, values + i), v_mean);
        auto v1 = s::Sub(s::Load(d, values + i + lanes), v_mean);
        v_var0 = s::MulAdd(v0, v0, v_var0);
        v_var1 = s::MulAdd(v1, v1, v_var1);
    }

    Real sum_sq = s::GetLane(s::SumOfLanes(d, s::Add(v_var0, v_var1)));

    for (; i < n; ++i) {
        Real d_val = values[i] - mean;
        sum_sq += d_val * d_val;
    }

    return sum_sq / static_cast<Real>(n - 1);
}

// SIMD-optimized Pearson correlation
SCL_HOT Real pearson_correlation(
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT y,
    Index n
) {
    if (SCL_UNLIKELY(n < 2)) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Compute means with multi-accumulator
    auto v_sum_x0 = s::Zero(d), v_sum_x1 = s::Zero(d);
    auto v_sum_y0 = s::Zero(d), v_sum_y1 = s::Zero(d);

    Index i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        v_sum_x0 = s::Add(v_sum_x0, s::Load(d, x + i));
        v_sum_x1 = s::Add(v_sum_x1, s::Load(d, x + i + lanes));
        v_sum_y0 = s::Add(v_sum_y0, s::Load(d, y + i));
        v_sum_y1 = s::Add(v_sum_y1, s::Load(d, y + i + lanes));
    }

    Real sum_x = s::GetLane(s::SumOfLanes(d, s::Add(v_sum_x0, v_sum_x1)));
    Real sum_y = s::GetLane(s::SumOfLanes(d, s::Add(v_sum_y0, v_sum_y1)));

    for (; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
    }

    Real inv_n = Real(1) / static_cast<Real>(n);
    Real mean_x = sum_x * inv_n;
    Real mean_y = sum_y * inv_n;

    // Compute covariance and variances with SIMD
    auto v_mean_x = s::Set(d, mean_x);
    auto v_mean_y = s::Set(d, mean_y);
    auto v_cov0 = s::Zero(d), v_cov1 = s::Zero(d);
    auto v_var_x0 = s::Zero(d), v_var_x1 = s::Zero(d);
    auto v_var_y0 = s::Zero(d), v_var_y1 = s::Zero(d);

    i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        auto dx0 = s::Sub(s::Load(d, x + i), v_mean_x);
        auto dx1 = s::Sub(s::Load(d, x + i + lanes), v_mean_x);
        auto dy0 = s::Sub(s::Load(d, y + i), v_mean_y);
        auto dy1 = s::Sub(s::Load(d, y + i + lanes), v_mean_y);

        v_cov0 = s::MulAdd(dx0, dy0, v_cov0);
        v_cov1 = s::MulAdd(dx1, dy1, v_cov1);
        v_var_x0 = s::MulAdd(dx0, dx0, v_var_x0);
        v_var_x1 = s::MulAdd(dx1, dx1, v_var_x1);
        v_var_y0 = s::MulAdd(dy0, dy0, v_var_y0);
        v_var_y1 = s::MulAdd(dy1, dy1, v_var_y1);
    }

    Real sum_xy = s::GetLane(s::SumOfLanes(d, s::Add(v_cov0, v_cov1)));
    Real sum_xx = s::GetLane(s::SumOfLanes(d, s::Add(v_var_x0, v_var_x1)));
    Real sum_yy = s::GetLane(s::SumOfLanes(d, s::Add(v_var_y0, v_var_y1)));

    for (; i < n; ++i) {
        Real dx = x[i] - mean_x;
        Real dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    Real denom = std::sqrt(sum_xx * sum_yy);
    return (SCL_LIKELY(denom > config::EPSILON)) ? sum_xy / denom : Real(0);
}

// Spearman correlation using efficient ranking
SCL_HOT Real spearman_correlation(
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT y,
    Index n
) {
    if (SCL_UNLIKELY(n < 2)) return Real(0);

    Real* rank_x = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* rank_y = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* idx = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* x_copy = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* y_copy = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    std::memcpy(x_copy, x, static_cast<Size>(n) * sizeof(Real));
    std::memcpy(y_copy, y, static_cast<Size>(n) * sizeof(Real));

    // Compute ranks for x using efficient sort
    for (Index i = 0; i < n; ++i) idx[i] = i;

    scl::sort::sort_pairs(Array<Real>(x_copy, static_cast<Size>(n)),
                         Array<Index>(idx, static_cast<Size>(n)));

    for (Index i = 0; i < n; ++i) {
        rank_x[idx[i]] = static_cast<Real>(i + 1);
    }

    // Compute ranks for y
    for (Index i = 0; i < n; ++i) idx[i] = i;

    scl::sort::sort_pairs(Array<Real>(y_copy, static_cast<Size>(n)),
                         Array<Index>(idx, static_cast<Size>(n)));

    for (Index i = 0; i < n; ++i) {
        rank_y[idx[i]] = static_cast<Real>(i + 1);
    }

    Real corr = pearson_correlation(rank_x, rank_y, n);

    scl::memory::aligned_free(y_copy, SCL_ALIGNMENT);
    scl::memory::aligned_free(x_copy, SCL_ALIGNMENT);
    scl::memory::aligned_free(idx, SCL_ALIGNMENT);
    scl::memory::aligned_free(rank_y, SCL_ALIGNMENT);
    scl::memory::aligned_free(rank_x, SCL_ALIGNMENT);

    return corr;
}

// SIMD-optimized mutual information
SCL_HOT Real mutual_information(
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT y,
    Index n,
    Index n_bins
) {
    if (SCL_UNLIKELY(n < 2 || n_bins < 2)) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Find min/max with SIMD
    Real min_x = x[0], max_x = x[0];
    Real min_y = y[0], max_y = y[0];
    auto v_min_x = s::Set(d, min_x), v_max_x = s::Set(d, max_x);
    auto v_min_y = s::Set(d, min_y), v_max_y = s::Set(d, max_y);

    Index i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto vx = s::Load(d, x + i);
        auto vy = s::Load(d, y + i);
        v_min_x = s::Min(v_min_x, vx);
        v_max_x = s::Max(v_max_x, vx);
        v_min_y = s::Min(v_min_y, vy);
        v_max_y = s::Max(v_max_y, vy);
    }

    min_x = s::GetLane(s::MinOfLanes(d, v_min_x));
    max_x = s::GetLane(s::MaxOfLanes(d, v_max_x));
    min_y = s::GetLane(s::MinOfLanes(d, v_min_y));
    max_y = s::GetLane(s::MaxOfLanes(d, v_max_y));

    for (; i < n; ++i) {
        min_x = scl::algo::min2(min_x, x[i]);
        max_x = scl::algo::max2(max_x, x[i]);
        min_y = scl::algo::min2(min_y, y[i]);
        max_y = scl::algo::max2(max_y, y[i]);
    }

    Real inv_range_x = static_cast<Real>(n_bins) / (max_x - min_x + config::EPSILON);
    Real inv_range_y = static_cast<Real>(n_bins) / (max_y - min_y + config::EPSILON);

    // Build joint and marginal histograms
    const Size hist_size = static_cast<Size>(n_bins) * static_cast<Size>(n_bins);
    Real* joint = scl::memory::aligned_alloc<Real>(hist_size, SCL_ALIGNMENT);
    Real* marg_x = scl::memory::aligned_alloc<Real>(n_bins, SCL_ALIGNMENT);
    Real* marg_y = scl::memory::aligned_alloc<Real>(n_bins, SCL_ALIGNMENT);

    std::memset(joint, 0, hist_size * sizeof(Real));
    std::memset(marg_x, 0, static_cast<Size>(n_bins) * sizeof(Real));
    std::memset(marg_y, 0, static_cast<Size>(n_bins) * sizeof(Real));

    Real inv_n = Real(1) / static_cast<Real>(n);

    for (Index i = 0; i < n; ++i) {
        Index bin_x = static_cast<Index>((x[i] - min_x) * inv_range_x);
        Index bin_y = static_cast<Index>((y[i] - min_y) * inv_range_y);
        bin_x = scl::algo::min2(bin_x, n_bins - 1);
        bin_y = scl::algo::min2(bin_y, n_bins - 1);

        joint[static_cast<Size>(bin_x) * n_bins + bin_y] += inv_n;
        marg_x[bin_x] += inv_n;
        marg_y[bin_y] += inv_n;
    }

    // Compute MI
    Real mi = Real(0);
    for (Index bx = 0; bx < n_bins; ++bx) {
        Real p_x = marg_x[bx];
        if (SCL_UNLIKELY(p_x <= config::EPSILON)) continue;

        for (Index by = 0; by < n_bins; ++by) {
            Real p_xy = joint[static_cast<Size>(bx) * n_bins + by];
            Real p_y = marg_y[by];

            if (SCL_LIKELY(p_xy > config::EPSILON && p_y > config::EPSILON)) {
                mi += p_xy * std::log(p_xy / (p_x * p_y));
            }
        }
    }

    scl::memory::aligned_free(marg_y, SCL_ALIGNMENT);
    scl::memory::aligned_free(marg_x, SCL_ALIGNMENT);
    scl::memory::aligned_free(joint, SCL_ALIGNMENT);

    return mi;
}

// Extract gene expression with binary search for CSR
template <typename T, bool IsCSR>
SCL_HOT void extract_gene_expression(
    const Sparse<T, IsCSR>& X,
    Index gene,
    Index n_cells,
    Real* SCL_RESTRICT output
) {
    std::memset(output, 0, static_cast<Size>(n_cells) * sizeof(Real));

    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            auto indices = X.row_indices(c);
            auto values = X.row_values(c);
            Index len = X.row_length(c);

            // Binary search for gene
            const Index* found = scl::algo::lower_bound(
                indices.ptr, indices.ptr + len, gene);

            if (found != indices.ptr + len && *found == gene) {
                Index k = static_cast<Index>(found - indices.ptr);
                output[c] = static_cast<Real>(values[k]);
            }
        }
    } else {
        auto indices = X.col_indices(gene);
        auto values = X.col_values(gene);
        Index len = X.col_length(gene);

        for (Index k = 0; k < len; ++k) {
            Index c = indices[k];
            if (SCL_LIKELY(c < n_cells)) {
                output[c] = static_cast<Real>(values[k]);
            }
        }
    }
}

// Random forest importance (correlation-based proxy)
SCL_HOT void random_forest_importance(
    const Real* SCL_RESTRICT X_features,
    const Real* SCL_RESTRICT y_target,
    Index n_cells,
    Index n_tfs,
    Index n_trees,
    Real* SCL_RESTRICT importance,
    FastRNG& rng
) {
    std::memset(importance, 0, static_cast<Size>(n_tfs) * sizeof(Real));

    if (SCL_UNLIKELY(n_cells < 4 || n_tfs == 0 || n_trees == 0)) return;

    Real base_mean = compute_mean(y_target, n_cells);
    Real base_var = compute_variance(y_target, n_cells, base_mean);
    if (SCL_UNLIKELY(base_var < config::EPSILON)) return;

    // Use squared correlation as importance proxy
    Real max_imp = Real(0);
    for (Index tf = 0; tf < n_tfs; ++tf) {
        const Real* tf_expr = X_features + static_cast<Size>(tf) * n_cells;
        Real corr = pearson_correlation(tf_expr, y_target, n_cells);
        importance[tf] = corr * corr;
        max_imp = scl::algo::max2(max_imp, importance[tf]);
    }

    // Normalize
    if (SCL_LIKELY(max_imp > config::EPSILON)) {
        Real inv_max = Real(1) / max_imp;
        for (Index tf = 0; tf < n_tfs; ++tf) {
            importance[tf] *= inv_max;
        }
    }
}

} // namespace detail

// =============================================================================
// Correlation Network - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void correlation_network(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real threshold,
    Real* SCL_RESTRICT correlation_matrix,
    bool use_spearman = false
) {
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_genes);
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    std::memset(correlation_matrix, 0, total * sizeof(Real));

    // Set diagonal to 1
    for (Index i = 0; i < n_genes; ++i) {
        correlation_matrix[static_cast<Size>(i) * n_genes + i] = Real(1);
    }

    const bool use_parallel = (n_genes_sz >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Pre-extract all gene expressions for better cache locality
    Real* all_gene_expr = scl::memory::aligned_alloc<Real>(n_genes_sz * n_cells_sz, SCL_ALIGNMENT);

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_genes_sz, [&](size_t g) {
            Real* gene_expr = all_gene_expr + g * n_cells_sz;
            detail::extract_gene_expression(expression, static_cast<Index>(g), n_cells, gene_expr);
        });
    } else {
        for (Index g = 0; g < n_genes; ++g) {
            Real* gene_expr = all_gene_expr + static_cast<Size>(g) * n_cells_sz;
            detail::extract_gene_expression(expression, g, n_cells, gene_expr);
        }
    }

    // Compute correlations - parallelize over gene pairs (sequential for simplicity)
    for (Index i = 0; i < n_genes; ++i) {
        const Real* expr_i = all_gene_expr + static_cast<Size>(i) * n_cells_sz;

        for (Index j = i + 1; j < n_genes; ++j) {
            const Real* expr_j = all_gene_expr + static_cast<Size>(j) * n_cells_sz;

            Real corr;
            if (use_spearman) {
                corr = detail::spearman_correlation(expr_i, expr_j, n_cells);
            } else {
                corr = detail::pearson_correlation(expr_i, expr_j, n_cells);
            }

            if (std::abs(corr) >= threshold) {
                correlation_matrix[static_cast<Size>(i) * n_genes + j] = corr;
                correlation_matrix[static_cast<Size>(j) * n_genes + i] = corr;
            }
        }
    }

    scl::memory::aligned_free(all_gene_expr, SCL_ALIGNMENT);
}

// =============================================================================
// Sparse Correlation Network - Optimized
// =============================================================================

template <typename T, bool IsCSR>
Index correlation_network_sparse(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real threshold,
    Index* SCL_RESTRICT edge_row,
    Index* SCL_RESTRICT edge_col,
    Real* SCL_RESTRICT edge_weight,
    Index max_edges,
    bool use_spearman = false
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    // Pre-extract all gene expressions
    Real* all_gene_expr = scl::memory::aligned_alloc<Real>(n_genes_sz * n_cells_sz, SCL_ALIGNMENT);

    for (Index g = 0; g < n_genes; ++g) {
        Real* gene_expr = all_gene_expr + static_cast<Size>(g) * n_cells_sz;
        detail::extract_gene_expression(expression, g, n_cells, gene_expr);
    }

    Index n_edges = 0;
    for (Index i = 0; i < n_genes && n_edges < max_edges; ++i) {
        const Real* expr_i = all_gene_expr + static_cast<Size>(i) * n_cells_sz;

        for (Index j = i + 1; j < n_genes && n_edges < max_edges; ++j) {
            const Real* expr_j = all_gene_expr + static_cast<Size>(j) * n_cells_sz;

            Real corr;
            if (use_spearman) {
                corr = detail::spearman_correlation(expr_i, expr_j, n_cells);
            } else {
                corr = detail::pearson_correlation(expr_i, expr_j, n_cells);
            }

            if (std::abs(corr) >= threshold) {
                edge_row[n_edges] = i;
                edge_col[n_edges] = j;
                edge_weight[n_edges] = corr;
                ++n_edges;
            }
        }
    }

    scl::memory::aligned_free(all_gene_expr, SCL_ALIGNMENT);
    return n_edges;
}

// =============================================================================
// Partial Correlation Network - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void partial_correlation_network(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real threshold,
    Real* SCL_RESTRICT partial_corr_matrix,
    Real regularization = Real(0.1)
) {
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_genes);

    Real* corr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    correlation_network(expression, n_cells, n_genes, Real(0), corr, false);

    // Add regularization to diagonal
    for (Index i = 0; i < n_genes; ++i) {
        corr[static_cast<Size>(i) * n_genes + i] += regularization;
    }

    std::memset(partial_corr_matrix, 0, total * sizeof(Real));

    // Compute partial correlations
    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto compute_row = [&](Index i) {
        partial_corr_matrix[static_cast<Size>(i) * n_genes + i] = Real(1);
        Real var_i = corr[static_cast<Size>(i) * n_genes + i];

        for (Index j = i + 1; j < n_genes; ++j) {
            Real var_j = corr[static_cast<Size>(j) * n_genes + j];
            Real cov_ij = corr[static_cast<Size>(i) * n_genes + j];
            Real denom = std::sqrt(var_i * var_j);
            Real pcorr = (SCL_LIKELY(denom > config::EPSILON)) ? -cov_ij / denom : Real(0);

            if (std::abs(pcorr) >= threshold) {
                partial_corr_matrix[static_cast<Size>(i) * n_genes + j] = pcorr;
                partial_corr_matrix[static_cast<Size>(j) * n_genes + i] = pcorr;
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t i) {
            compute_row(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n_genes; ++i) {
            compute_row(i);
        }
    }

    scl::memory::aligned_free(corr, SCL_ALIGNMENT);
}

// =============================================================================
// Mutual Information Network - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void mutual_information_network(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real threshold,
    Real* SCL_RESTRICT mi_matrix,
    Index n_bins = config::DEFAULT_N_BINS
) {
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_genes);
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    std::memset(mi_matrix, 0, total * sizeof(Real));

    // Set diagonal
    for (Index i = 0; i < n_genes; ++i) {
        mi_matrix[static_cast<Size>(i) * n_genes + i] = Real(1);
    }

    // Pre-extract all gene expressions
    Real* all_gene_expr = scl::memory::aligned_alloc<Real>(n_genes_sz * n_cells_sz, SCL_ALIGNMENT);

    for (Index g = 0; g < n_genes; ++g) {
        Real* gene_expr = all_gene_expr + static_cast<Size>(g) * n_cells_sz;
        detail::extract_gene_expression(expression, g, n_cells, gene_expr);
    }

    Real max_mi = std::log(static_cast<Real>(n_bins));
    Real inv_max_mi = (max_mi > config::EPSILON) ? Real(1) / max_mi : Real(0);

    const bool use_parallel = (n_genes_sz >= config::PARALLEL_THRESHOLD);

    if (use_parallel) {
        // Parallelize over rows
        scl::threading::parallel_for(Size(0), n_genes_sz, [&](size_t i) {
            const Real* expr_i = all_gene_expr + i * n_cells_sz;

            for (Index j = static_cast<Index>(i) + 1; j < n_genes; ++j) {
                const Real* expr_j = all_gene_expr + static_cast<Size>(j) * n_cells_sz;
                Real mi = detail::mutual_information(expr_i, expr_j, n_cells, n_bins);
                Real nmi = mi * inv_max_mi;

                if (nmi >= threshold) {
                    mi_matrix[i * n_genes + j] = nmi;
                    mi_matrix[static_cast<Size>(j) * n_genes + i] = nmi;
                }
            }
        });
    } else {
        for (Index i = 0; i < n_genes; ++i) {
            const Real* expr_i = all_gene_expr + static_cast<Size>(i) * n_cells_sz;

            for (Index j = i + 1; j < n_genes; ++j) {
                const Real* expr_j = all_gene_expr + static_cast<Size>(j) * n_cells_sz;
                Real mi = detail::mutual_information(expr_i, expr_j, n_cells, n_bins);
                Real nmi = mi * inv_max_mi;

                if (nmi >= threshold) {
                    mi_matrix[static_cast<Size>(i) * n_genes + j] = nmi;
                    mi_matrix[static_cast<Size>(j) * n_genes + i] = nmi;
                }
            }
        }
    }

    scl::memory::aligned_free(all_gene_expr, SCL_ALIGNMENT);
}

// =============================================================================
// TF-Target Score - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void tf_target_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> tf_genes,
    Array<const Index> target_genes,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT scores
) {
    const Index n_tfs = static_cast<Index>(tf_genes.len);
    const Index n_targets = static_cast<Index>(target_genes.len);
    const Size total = static_cast<Size>(n_tfs) * static_cast<Size>(n_targets);
    const Size n_cells_sz = static_cast<Size>(n_cells);

    std::memset(scores, 0, total * sizeof(Real));

    if (SCL_UNLIKELY(n_tfs == 0 || n_targets == 0)) return;

    // Pre-extract TF expressions
    Real* tf_expressions = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n_tfs) * n_cells_sz, SCL_ALIGNMENT);

    for (Index t = 0; t < n_tfs; ++t) {
        Index tf = tf_genes[t];
        Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;

        if (SCL_LIKELY(tf >= 0 && tf < n_genes)) {
            detail::extract_gene_expression(expression, tf, n_cells, tf_expr);
        } else {
            std::memset(tf_expr, 0, n_cells_sz * sizeof(Real));
        }
    }

    // Pre-extract target expressions
    Real* target_expressions = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n_targets) * n_cells_sz, SCL_ALIGNMENT);

    for (Index g = 0; g < n_targets; ++g) {
        Index target = target_genes[g];
        Real* target_expr = target_expressions + static_cast<Size>(g) * n_cells_sz;

        if (SCL_LIKELY(target >= 0 && target < n_genes)) {
            detail::extract_gene_expression(expression, target, n_cells, target_expr);
        } else {
            std::memset(target_expr, 0, n_cells_sz * sizeof(Real));
        }
    }

    // Compute correlations
    const bool use_parallel = (static_cast<Size>(n_tfs) * n_targets >= config::PARALLEL_THRESHOLD);

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_tfs), [&](size_t t) {
            Index tf = tf_genes[t];
            const Real* tf_expr = tf_expressions + t * n_cells_sz;

            for (Index g = 0; g < n_targets; ++g) {
                Index target = target_genes[g];
                if (tf == target) continue;

                const Real* target_expr = target_expressions + static_cast<Size>(g) * n_cells_sz;
                Real corr = detail::pearson_correlation(tf_expr, target_expr, n_cells);
                scores[t * n_targets + g] = corr;
            }
        });
    } else {
        for (Index t = 0; t < n_tfs; ++t) {
            Index tf = tf_genes[t];
            const Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;

            for (Index g = 0; g < n_targets; ++g) {
                Index target = target_genes[g];
                if (tf == target) continue;

                const Real* target_expr = target_expressions + static_cast<Size>(g) * n_cells_sz;
                Real corr = detail::pearson_correlation(tf_expr, target_expr, n_cells);
                scores[static_cast<Size>(t) * n_targets + g] = corr;
            }
        }
    }

    scl::memory::aligned_free(target_expressions, SCL_ALIGNMENT);
    scl::memory::aligned_free(tf_expressions, SCL_ALIGNMENT);
}

// =============================================================================
// GENIE3-Style Importance Scoring - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void genie3_importance(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> tf_genes,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT importance_matrix,
    Index n_trees = config::DEFAULT_N_TREES
) {
    const Index n_tfs = static_cast<Index>(tf_genes.len);
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_tfs);
    const Size n_cells_sz = static_cast<Size>(n_cells);

    std::memset(importance_matrix, 0, total * sizeof(Real));

    // Pre-extract TF expressions
    Real* tf_expressions = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n_tfs) * n_cells_sz, SCL_ALIGNMENT);

    for (Index t = 0; t < n_tfs; ++t) {
        Index tf = tf_genes[t];
        Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;

        if (SCL_LIKELY(tf >= 0 && tf < n_genes)) {
            detail::extract_gene_expression(expression, tf, n_cells, tf_expr);
        } else {
            std::memset(tf_expr, 0, n_cells_sz * sizeof(Real));
        }
    }

    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::DualWorkspacePool<Real> workspace;
    if (use_parallel) {
        workspace.init(n_threads, n_cells_sz, static_cast<Size>(n_tfs));
    }

    auto process_gene = [&](Index g, Real* target_expr, Real* importance, uint64_t seed) {
        detail::extract_gene_expression(expression, g, n_cells, target_expr);

        // Check if this gene is a TF
        Index tf_idx = -1;
        for (Index t = 0; t < n_tfs; ++t) {
            if (tf_genes[t] == g) {
                tf_idx = t;
                break;
            }
        }

        detail::FastRNG rng(seed);
        detail::random_forest_importance(
            tf_expressions, target_expr, n_cells, n_tfs, n_trees, importance, rng
        );

        // Store importance scores
        for (Index t = 0; t < n_tfs; ++t) {
            if (t == tf_idx) {
                importance_matrix[static_cast<Size>(g) * n_tfs + t] = Real(0);
            } else {
                importance_matrix[static_cast<Size>(g) * n_tfs + t] = importance[t];
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t g, size_t thread_rank) {
            Real* target_expr = workspace.get1(thread_rank);
            Real* importance = workspace.get2(thread_rank);
            process_gene(static_cast<Index>(g), target_expr, importance, 42 + g);
        });
    } else {
        Real* target_expr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);
        Real* importance = scl::memory::aligned_alloc<Real>(n_tfs, SCL_ALIGNMENT);

        for (Index g = 0; g < n_genes; ++g) {
            process_gene(g, target_expr, importance, 42 + g);
        }

        scl::memory::aligned_free(importance, SCL_ALIGNMENT);
        scl::memory::aligned_free(target_expr, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(tf_expressions, SCL_ALIGNMENT);
}

// =============================================================================
// Regulon Activity Score - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void regulon_activity(
    const Sparse<T, IsCSR>& expression,
    const Index* const* regulon_genes,
    const Index* regulon_sizes,
    Index n_regulons,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT activity_scores
) {
    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_regulons);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    std::memset(activity_scores, 0, total * sizeof(Real));

    const bool use_parallel = (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> cell_expr_pool;
    if (use_parallel && IsCSR) {
        cell_expr_pool.init(n_threads, n_genes_sz);
    }

    auto process_cell = [&](Index c, Real* cell_expr) {
        if (IsCSR) {
            std::memset(cell_expr, 0, n_genes_sz * sizeof(Real));

            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    cell_expr[g] = static_cast<Real>(values[k]);
                }
            }
        }

        for (Index r = 0; r < n_regulons; ++r) {
            const Index* genes = regulon_genes[r];
            Index n_reg_genes = regulon_sizes[r];
            if (SCL_UNLIKELY(n_reg_genes == 0)) continue;

            Real sum = Real(0);
            Index count = 0;

            for (Index k = 0; k < n_reg_genes; ++k) {
                Index g = genes[k];
                if (SCL_UNLIKELY(g < 0 || g >= n_genes)) continue;

                Real expr;
                if (IsCSR) {
                    expr = cell_expr[g];
                } else {
                    expr = Real(0);
                    auto indices = expression.col_indices(g);
                    auto values = expression.col_values(g);
                    Index len = expression.col_length(g);

                    // Binary search
                    const Index* found = scl::algo::lower_bound(
                        indices.ptr, indices.ptr + len, c);

                    if (found != indices.ptr + len && *found == c) {
                        Index idx = static_cast<Index>(found - indices.ptr);
                        expr = static_cast<Real>(values[idx]);
                    }
                }

                sum += expr;
                ++count;
            }

            if (count > 0) {
                activity_scores[static_cast<Size>(c) * n_regulons + r] =
                    sum / static_cast<Real>(count);
            }
        }
    };

    if (use_parallel) {
        if (IsCSR) {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c, size_t thread_rank) {
                Real* cell_expr = cell_expr_pool.get(thread_rank);
                process_cell(static_cast<Index>(c), cell_expr);
            });
        } else {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c) {
                process_cell(static_cast<Index>(c), nullptr);
            });
        }
    } else {
        Real* cell_expr = nullptr;
        if (IsCSR) {
            cell_expr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        }

        for (Index c = 0; c < n_cells; ++c) {
            process_cell(c, cell_expr);
        }

        if (cell_expr) {
            scl::memory::aligned_free(cell_expr, SCL_ALIGNMENT);
        }
    }
}

// =============================================================================
// Regulon Enrichment Score (AUCell-Style) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void regulon_auc_score(
    const Sparse<T, IsCSR>& expression,
    const Index* const* regulon_genes,
    const Index* regulon_sizes,
    Index n_regulons,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT auc_scores,
    Index top_percent = 5
) {
    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_regulons);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    std::memset(auc_scores, 0, total * sizeof(Real));

    Index top_n = scl::algo::max2(Index(1), n_genes * top_percent / 100);

    const bool use_parallel = (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace: cell_expr, gene_indices, in_regulon
    struct CellWorkspace {
        Real* cell_expr;
        Index* gene_indices;
        bool* in_regulon;
    };

    CellWorkspace* workspaces = nullptr;
    if (use_parallel) {
        workspaces = scl::memory::aligned_alloc<CellWorkspace>(n_threads, SCL_ALIGNMENT);

        for (size_t t = 0; t < n_threads; ++t) {
            workspaces[t].cell_expr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
            workspaces[t].gene_indices = scl::memory::aligned_alloc<Index>(n_genes_sz, SCL_ALIGNMENT);
            workspaces[t].in_regulon = scl::memory::aligned_alloc<bool>(n_genes_sz, SCL_ALIGNMENT);
        }
    }

    auto process_cell = [&](Index c, Real* cell_expr, Index* gene_indices, bool* in_regulon) {
        // Extract and rank gene expression
        std::memset(cell_expr, 0, n_genes_sz * sizeof(Real));

        if (IsCSR) {
            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    cell_expr[g] = static_cast<Real>(values[k]);
                }
            }
        } else {
            for (Index g = 0; g < n_genes; ++g) {
                auto indices = expression.col_indices(g);
                auto values = expression.col_values(g);
                Index len = expression.col_length(g);

                const Index* found = scl::algo::lower_bound(indices.ptr, indices.ptr + len, c);
                if (found != indices.ptr + len && *found == c) {
                    Index idx = static_cast<Index>(found - indices.ptr);
                    cell_expr[g] = static_cast<Real>(values[idx]);
                }
            }
        }

        // Sort genes by expression (descending) - use efficient sort
        for (Index g = 0; g < n_genes; ++g) {
            gene_indices[g] = g;
        }

        // Use descending sort and only use top_n
        scl::sort::sort_pairs_descending(
            Array<Real>(cell_expr, n_genes_sz),
            Array<Index>(gene_indices, n_genes_sz)
        );

        // Compute AUC for each regulon
        for (Index r = 0; r < n_regulons; ++r) {
            const Index* genes = regulon_genes[r];
            Index n_reg_genes = regulon_sizes[r];
            if (SCL_UNLIKELY(n_reg_genes == 0)) continue;

            // Build regulon membership
            std::memset(in_regulon, 0, n_genes_sz * sizeof(bool));
            for (Index k = 0; k < n_reg_genes; ++k) {
                Index g = genes[k];
                if (SCL_LIKELY(g >= 0 && g < n_genes)) {
                    in_regulon[g] = true;
                }
            }

            // Count regulon genes in top_n
            Index count_in_top = 0;
            for (Index i = 0; i < top_n; ++i) {
                if (in_regulon[gene_indices[i]]) {
                    ++count_in_top;
                }
            }

            auc_scores[static_cast<Size>(c) * n_regulons + r] =
                static_cast<Real>(count_in_top) / static_cast<Real>(n_reg_genes);
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c, size_t thread_rank) {
            process_cell(static_cast<Index>(c),
                        workspaces[thread_rank].cell_expr,
                        workspaces[thread_rank].gene_indices,
                        workspaces[thread_rank].in_regulon);
        });

        for (size_t t = 0; t < n_threads; ++t) {
            scl::memory::aligned_free(workspaces[t].in_regulon, SCL_ALIGNMENT);
            scl::memory::aligned_free(workspaces[t].gene_indices, SCL_ALIGNMENT);
            scl::memory::aligned_free(workspaces[t].cell_expr, SCL_ALIGNMENT);
        }

        scl::memory::aligned_free(workspaces, SCL_ALIGNMENT);
    } else {
        Real* cell_expr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        Index* gene_indices = scl::memory::aligned_alloc<Index>(n_genes_sz, SCL_ALIGNMENT);
        bool* in_regulon = scl::memory::aligned_alloc<bool>(n_genes_sz, SCL_ALIGNMENT);

        for (Index c = 0; c < n_cells; ++c) {
            process_cell(c, cell_expr, gene_indices, in_regulon);
        }

        scl::memory::aligned_free(in_regulon, SCL_ALIGNMENT);
        scl::memory::aligned_free(gene_indices, SCL_ALIGNMENT);
        scl::memory::aligned_free(cell_expr, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Network Statistics - Optimized
// =============================================================================

inline void network_statistics(
    const Real* SCL_RESTRICT adjacency,
    Index n,
    Real* SCL_RESTRICT out_degree,
    Real* SCL_RESTRICT clustering_coef,
    Real threshold = Real(0)
) {
    const Size n_sz = static_cast<Size>(n);
    const bool use_parallel = (n_sz >= config::PARALLEL_THRESHOLD);

    // Compute degree
    auto compute_degree = [&](Index i) {
        Real deg = Real(0);

        // Unrolled loop
        Index j = 0;
        for (; j + 4 <= n; j += 4) {
            if (j != i && adjacency[static_cast<Size>(i) * n + j] > threshold) deg += Real(1);
            if (j + 1 != i && adjacency[static_cast<Size>(i) * n + j + 1] > threshold) deg += Real(1);
            if (j + 2 != i && adjacency[static_cast<Size>(i) * n + j + 2] > threshold) deg += Real(1);
            if (j + 3 != i && adjacency[static_cast<Size>(i) * n + j + 3] > threshold) deg += Real(1);
        }

        for (; j < n; ++j) {
            if (i != j && adjacency[static_cast<Size>(i) * n + j] > threshold) {
                deg += Real(1);
            }
        }

        out_degree[i] = deg;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_sz, [&](size_t i) {
            compute_degree(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            compute_degree(i);
        }
    }

    // Compute clustering coefficient
    auto compute_clustering = [&](Index i) {
        Real ki = out_degree[i];
        if (SCL_UNLIKELY(ki < Real(2))) {
            clustering_coef[i] = Real(0);
            return;
        }

        Real triangles = Real(0);
        for (Index j = 0; j < n; ++j) {
            if (i == j || adjacency[static_cast<Size>(i) * n + j] <= threshold) continue;

            for (Index k = j + 1; k < n; ++k) {
                if (k == i || adjacency[static_cast<Size>(i) * n + k] <= threshold) continue;

                if (adjacency[static_cast<Size>(j) * n + k] > threshold) {
                    triangles += Real(1);
                }
            }
        }

        Real max_triangles = ki * (ki - Real(1)) * Real(0.5);
        clustering_coef[i] = (SCL_LIKELY(max_triangles > Real(0))) ? triangles / max_triangles : Real(0);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_sz, [&](size_t i) {
            compute_clustering(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            compute_clustering(i);
        }
    }
}

// =============================================================================
// Hub Gene Identification - Optimized
// =============================================================================

inline Index identify_hub_genes(
    const Real* SCL_RESTRICT adjacency,
    Index n,
    Real degree_threshold,
    Index* SCL_RESTRICT hub_genes,
    Index max_hubs
) {
    const Size n_sz = static_cast<Size>(n);

    Real* degrees = scl::memory::aligned_alloc<Real>(n_sz, SCL_ALIGNMENT);
    Real* clustering = scl::memory::aligned_alloc<Real>(n_sz, SCL_ALIGNMENT);

    network_statistics(adjacency, n, degrees, clustering);

    // Find maximum degree with SIMD
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    Real max_deg = degrees[0];
    auto v_max = s::Set(d, max_deg);

    Size i = 0;
    for (; i + lanes <= n_sz; i += lanes) {
        v_max = s::Max(v_max, s::Load(d, degrees + i));
    }

    max_deg = s::GetLane(s::MaxOfLanes(d, v_max));

    for (; i < n_sz; ++i) {
        max_deg = scl::algo::max2(max_deg, degrees[i]);
    }

    // Identify hubs
    Index n_hubs = 0;
    Real threshold = max_deg * degree_threshold;

    for (Index i = 0; i < n && n_hubs < max_hubs; ++i) {
        if (degrees[i] >= threshold) {
            hub_genes[n_hubs++] = i;
        }
    }

    scl::memory::aligned_free(clustering, SCL_ALIGNMENT);
    scl::memory::aligned_free(degrees, SCL_ALIGNMENT);
    return n_hubs;
}

// =============================================================================
// GRN Inference (Combined Method) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void infer_grn(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> tf_genes,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT grn_matrix,
    GRNMethod method = GRNMethod::Correlation,
    Real threshold = config::DEFAULT_CORRELATION_THRESHOLD
) {
    const Index n_tfs = static_cast<Index>(tf_genes.len);
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_tfs);

    std::memset(grn_matrix, 0, total * sizeof(Real));

    switch (method) {
        case GRNMethod::Correlation:
        case GRNMethod::Combined: {
            Index* all_genes = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
            for (Index g = 0; g < n_genes; ++g) {
                all_genes[g] = g;
            }

            tf_target_score(expression, tf_genes,
                           Array<const Index>(all_genes, n_genes),
                           n_cells, n_genes, grn_matrix);

            // Apply threshold
            for (Size i = 0; i < total; ++i) {
                if (std::abs(grn_matrix[i]) < threshold) {
                    grn_matrix[i] = Real(0);
                }
            }

            scl::memory::aligned_free(all_genes, SCL_ALIGNMENT);
            break;
        }

        case GRNMethod::GENIE3: {
            genie3_importance(expression, tf_genes, n_cells, n_genes,
                             grn_matrix, config::DEFAULT_N_TREES);

            for (Size i = 0; i < total; ++i) {
                if (grn_matrix[i] < threshold) {
                    grn_matrix[i] = Real(0);
                }
            }
            break;
        }

        case GRNMethod::MutualInformation: {
            const Size n_cells_sz = static_cast<Size>(n_cells);

            // Pre-extract expressions
            Real* tf_expressions = scl::memory::aligned_alloc<Real>(
                static_cast<Size>(n_tfs) * n_cells_sz, SCL_ALIGNMENT);
            Real* target_expr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);

            for (Index t = 0; t < n_tfs; ++t) {
                Index tf = tf_genes[t];
                Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;

                if (SCL_LIKELY(tf >= 0 && tf < n_genes)) {
                    detail::extract_gene_expression(expression, tf, n_cells, tf_expr);
                } else {
                    std::memset(tf_expr, 0, n_cells_sz * sizeof(Real));
                }
            }

            Real inv_log_bins = Real(1) / std::log(static_cast<Real>(config::DEFAULT_N_BINS));
            const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

            if (use_parallel) {
                scl::threading::WorkspacePool<Real> target_pool;
                target_pool.init(scl::threading::Scheduler::get_num_threads(), n_cells_sz);

                scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t g, size_t thread_rank) {
                    Real* local_target = target_pool.get(thread_rank);
                    detail::extract_gene_expression(expression, static_cast<Index>(g), n_cells, local_target);

                    for (Index t = 0; t < n_tfs; ++t) {
                        Index tf = tf_genes[t];
                        if (tf == static_cast<Index>(g)) continue;

                        const Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;
                        Real mi = detail::mutual_information(tf_expr, local_target, n_cells,
                                                            config::DEFAULT_N_BINS);
                        Real nmi = mi * inv_log_bins;

                        if (nmi >= threshold) {
                            grn_matrix[g * n_tfs + t] = nmi;
                        }
                    }
                });
            } else {
                for (Index g = 0; g < n_genes; ++g) {
                    detail::extract_gene_expression(expression, g, n_cells, target_expr);

                    for (Index t = 0; t < n_tfs; ++t) {
                        Index tf = tf_genes[t];
                        if (tf == g) continue;

                        const Real* tf_expr = tf_expressions + static_cast<Size>(t) * n_cells_sz;
                        Real mi = detail::mutual_information(tf_expr, target_expr, n_cells,
                                                            config::DEFAULT_N_BINS);
                        Real nmi = mi * inv_log_bins;

                        if (nmi >= threshold) {
                            grn_matrix[static_cast<Size>(g) * n_tfs + t] = nmi;
                        }
                    }
                }
            }

            scl::memory::aligned_free(target_expr, SCL_ALIGNMENT);
            scl::memory::aligned_free(tf_expressions, SCL_ALIGNMENT);
            break;
        }

        default:
            break;
    }
}

// =============================================================================
// Build Regulons from GRN - Optimized
// =============================================================================

inline Index build_regulons(
    const Real* SCL_RESTRICT grn_matrix,
    Index n_genes,
    Index n_tfs,
    Real threshold,
    Index min_regulon_size,
    Index* SCL_RESTRICT regulon_tf,
    Index* SCL_RESTRICT regulon_offsets,
    Index* SCL_RESTRICT regulon_targets,
    Index max_total_targets
) {
    Index n_regulons = 0;
    Index total_targets = 0;
    regulon_offsets[0] = 0;

    for (Index t = 0; t < n_tfs; ++t) {
        // Count targets for this TF
        Index n_targets = 0;
        for (Index g = 0; g < n_genes; ++g) {
            Real score = grn_matrix[static_cast<Size>(g) * n_tfs + t];
            if (std::abs(score) >= threshold) {
                ++n_targets;
            }
        }

        if (n_targets < min_regulon_size) continue;
        if (total_targets + n_targets > max_total_targets) break;

        regulon_tf[n_regulons] = t;

        for (Index g = 0; g < n_genes; ++g) {
            Real score = grn_matrix[static_cast<Size>(g) * n_tfs + t];
            if (std::abs(score) >= threshold) {
                regulon_targets[total_targets++] = g;
            }
        }

        ++n_regulons;
        regulon_offsets[n_regulons] = total_targets;
    }

    return n_regulons;
}

// =============================================================================
// TF Activity from Regulon Expression - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void tf_activity_from_regulons(
    const Sparse<T, IsCSR>& expression,
    const Index* regulon_tf,
    const Index* regulon_offsets,
    const Index* regulon_targets,
    const Real* grn_matrix,
    Index n_regulons,
    Index n_tfs,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT tf_activity
) {
    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_tfs);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    std::memset(tf_activity, 0, total * sizeof(Real));

    const bool use_parallel = (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> cell_expr_pool;
    if (use_parallel && IsCSR) {
        cell_expr_pool.init(n_threads, n_genes_sz);
    }

    auto process_cell = [&](Index c, Real* cell_expr) {
        if (IsCSR) {
            std::memset(cell_expr, 0, n_genes_sz * sizeof(Real));

            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    cell_expr[g] = static_cast<Real>(values[k]);
                }
            }
        }

        for (Index r = 0; r < n_regulons; ++r) {
            Index tf = regulon_tf[r];
            Index start = regulon_offsets[r];
            Index end = regulon_offsets[r + 1];

            Real weighted_sum = Real(0);
            Real weight_total = Real(0);

            for (Index k = start; k < end; ++k) {
                Index target = regulon_targets[k];
                if (SCL_UNLIKELY(target >= n_genes)) continue;

                Real weight = std::abs(grn_matrix[static_cast<Size>(target) * n_tfs + tf]);
                Real expr;

                if (IsCSR) {
                    expr = cell_expr[target];
                } else {
                    expr = Real(0);
                    auto indices = expression.col_indices(target);
                    auto values = expression.col_values(target);
                    Index len = expression.col_length(target);

                    const Index* found = scl::algo::lower_bound(indices.ptr, indices.ptr + len, c);
                    if (found != indices.ptr + len && *found == c) {
                        Index idx = static_cast<Index>(found - indices.ptr);
                        expr = static_cast<Real>(values[idx]);
                    }
                }

                weighted_sum += weight * expr;
                weight_total += weight;
            }

            if (SCL_LIKELY(weight_total > config::EPSILON)) {
                tf_activity[static_cast<Size>(c) * n_tfs + tf] = weighted_sum / weight_total;
            }
        }
    };

    if (use_parallel) {
        if (IsCSR) {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c, size_t thread_rank) {
                Real* cell_expr = cell_expr_pool.get(thread_rank);
                process_cell(static_cast<Index>(c), cell_expr);
            });
        } else {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c) {
                process_cell(static_cast<Index>(c), nullptr);
            });
        }
    } else {
        Real* cell_expr = nullptr;
        if (IsCSR) {
            cell_expr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        }

        for (Index c = 0; c < n_cells; ++c) {
            process_cell(c, cell_expr);
        }

        if (cell_expr) {
            scl::memory::aligned_free(cell_expr, SCL_ALIGNMENT);
        }
    }
}

} // namespace scl::kernel::grn
