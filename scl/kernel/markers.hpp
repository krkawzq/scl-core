#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/markers.hpp
// BRIEF: Marker gene selection and specificity scoring (OPTIMIZED)
// =============================================================================

namespace scl::kernel::markers {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_MIN_FC = Real(1.5);
    constexpr Real DEFAULT_MIN_PCT = Real(0.1);
    constexpr Real DEFAULT_MAX_PVAL = Real(0.05);
    constexpr Real MIN_EXPR = Real(1e-9);
    constexpr Real PSEUDO_COUNT = Real(1.0);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
}

// =============================================================================
// Ranking Methods
// =============================================================================

enum class RankingMethod {
    FoldChange,
    EffectSize,
    PValue,
    Combined
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Cohen's d effect size
SCL_FORCE_INLINE Real cohens_d(
    Real mean1,
    Real mean2,
    Real var1,
    Real var2,
    Size n1,
    Size n2
) {
    Real pooled_var = ((static_cast<Real>(n1) - Real(1)) * var1 +
                       (static_cast<Real>(n2) - Real(1)) * var2) /
                      (static_cast<Real>(n1 + n2) - Real(2));
    Real pooled_std = std::sqrt(pooled_var + config::MIN_EXPR);
    return (mean1 - mean2) / pooled_std;
}

// Combined score: fold_change * -log10(p_value)
SCL_FORCE_INLINE Real combined_score(Real fold_change, Real p_value) {
    Real safe_p = scl::algo::max2(p_value, Real(1e-300));
    return fold_change * (-std::log10(safe_p));
}

// SIMD-optimized Gini coefficient
SCL_FORCE_INLINE Real gini_coefficient(const Real* SCL_RESTRICT values, Size n) {
    if (SCL_UNLIKELY(n <= 1)) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    // Compute sum and weighted sum
    auto v_sum = s::Zero(d);
    auto v_weighted = s::Zero(d);

    Size i = 0;
    if (n >= static_cast<Size>(lanes)) {
        // Create index vector [1, 2, 3, ...]
        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, values + i);
            v_sum = s::Add(v_sum, v);

            // Weighted sum: (i+1) * values[i]
            // Use static array of Real, initialized below
            std::array<Real, 16> weights = {};
            for (Size j = 0; j < lanes; ++j) {
                weights[static_cast<Size>(j)] = static_cast<Real>(i + j + 1);
            }
            auto v_weights = s::Load(d, weights.data());
            v_weighted = s::MulAdd(v_weights, v, v_weighted);
        }
    }

    Real sum_val = s::GetLane(s::SumOfLanes(d, v_sum));
    Real weighted_sum = s::GetLane(s::SumOfLanes(d, v_weighted));

    // Scalar cleanup
    for (; i < n; ++i) {
        sum_val += values[i];
        weighted_sum += static_cast<Real>(i + 1) * values[i];
    }

    if (SCL_UNLIKELY(sum_val < config::MIN_EXPR)) return Real(0);

    Real n_real = static_cast<Real>(n);
    return (Real(2) * weighted_sum) / (n_real * sum_val) - (n_real + Real(1)) / n_real;
}

// Simple PRNG for tiebreaking
struct FastRNG {
    uint64_t state;

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }
};

// SIMD-optimized log2 fold change computation
SCL_FORCE_INLINE void compute_log2_fc_batch(
    const Real* SCL_RESTRICT target_means,
    const Real* SCL_RESTRICT other_means,
    Real* SCL_RESTRICT log_fc,
    Index n_genes,
    Real pseudo_count
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_pseudo = s::Set(d, pseudo_count);
    auto v_log2_e = s::Set(d, Real(1.4426950408889634));  // 1/ln(2)

    Index i = 0;
    for (; i + static_cast<Index>(lanes) <= n_genes; i += static_cast<Index>(lanes)) {
        auto v_target = s::Add(s::Load(d, target_means + i), v_pseudo);
        auto v_other = s::Add(s::Load(d, other_means + i), v_pseudo);

        // log2(a/b) = log2(a) - log2(b) = (ln(a) - ln(b)) / ln(2)
        auto v_log_target = s::Log(d, v_target);
        auto v_log_other = s::Log(d, v_other);
        auto v_fc = s::Mul(s::Sub(v_log_target, v_log_other), v_log2_e);

        s::Store(v_fc, d, log_fc + i);
    }

    // Scalar cleanup
    for (; i < n_genes; ++i) {
        log_fc[i] = std::log2(target_means[i] + pseudo_count) -
                    std::log2(other_means[i] + pseudo_count);
    }
}

} // namespace detail

// =============================================================================
// Group Mean Expression - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void group_mean_expression(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Array<Real> mean_expr,
    Index n_genes
) {
    const Index n_cells = IsCSR ? X.rows() : X.cols();
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_groups);

    SCL_CHECK_DIM(group_labels.len >= static_cast<Size>(n_cells),
                  "Markers: group_labels buffer too small");
    SCL_CHECK_DIM(mean_expr.len >= total, "Markers: mean_expr buffer too small");

    std::memset(mean_expr.ptr, 0, total * sizeof(Real));

    // Count cells per group
    auto group_counts_ptr = scl::memory::aligned_alloc<Index>(n_groups, SCL_ALIGNMENT);

    Index* group_counts = group_counts_ptr.release();
    std::memset(group_counts, 0, static_cast<Size>(n_groups) * sizeof(Index));

    for (Index c = 0; c < n_cells; ++c) {
        Index g = group_labels[c];
        if (SCL_LIKELY(g >= 0 && g < n_groups)) {
            ++group_counts[g];
        }
    }

    // Precompute inverse counts for division
    auto inv_counts_ptr = scl::memory::aligned_alloc<Real>(n_groups, SCL_ALIGNMENT);

    Real* inv_counts = inv_counts_ptr.release();
    for (Index g = 0; g < n_groups; ++g) {
        inv_counts[g] = (group_counts[g] > 0) ?
            Real(1) / static_cast<Real>(group_counts[g]) : Real(0);
    }

    // Accumulate expression sums
    if constexpr (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            Index g = group_labels[c];
            if (SCL_UNLIKELY(g < 0 || g >= n_groups)) continue;

            auto indices = X.row_indices_unsafe(c);
            auto values = X.row_values_unsafe(c);
            const Index len = X.row_length_unsafe(c);

            // Unrolled accumulation
            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                Index gene0 = indices[k], gene1 = indices[k + 1];
                Index gene2 = indices[k + 2], gene3 = indices[k + 3];
                if (SCL_LIKELY(gene0 < n_genes))
                    mean_expr[static_cast<Index>(static_cast<Size>(gene0) * n_groups + g)] += static_cast<Real>(values[k]);
                if (SCL_LIKELY(gene1 < n_genes))
                    mean_expr[static_cast<Index>(static_cast<Size>(gene1) * n_groups + g)] += static_cast<Real>(values[k + 1]);
                if (SCL_LIKELY(gene2 < n_genes))
                    mean_expr[static_cast<Index>(static_cast<Size>(gene2) * n_groups + g)] += static_cast<Real>(values[k + 2]);
                if (SCL_LIKELY(gene3 < n_genes))
                    mean_expr[static_cast<Index>(static_cast<Size>(gene3) * n_groups + g)] += static_cast<Real>(values[k + 3]);
            }

            for (; k < len; ++k) {
                Index gene = indices[k];
                if (SCL_LIKELY(gene < n_genes)) {
                    mean_expr[static_cast<Index>(static_cast<Size>(gene) * n_groups + g)] += static_cast<Real>(values[k]);
                }
            }
        }
    } else {
        // CSC: can parallelize over genes
        const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

        auto process_gene = [&](Index gene) {
            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            Real* gene_means = mean_expr.ptr + static_cast<Size>(gene) * n_groups;

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                Index g = group_labels[c];
                if (SCL_LIKELY(g >= 0 && g < n_groups)) {
                    gene_means[g] += static_cast<Real>(values[k]);
                }
            }
        };

        if (use_parallel) {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
                process_gene(static_cast<Index>(gene));
            });
        } else {
            for (Index gene = 0; gene < n_genes; ++gene) {
                process_gene(gene);
            }
        }
    }

    // Divide by group counts - parallelize over genes
    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto normalize_gene = [&](Index gene) {
        Real* gene_means = mean_expr.ptr + static_cast<Size>(gene) * n_groups;
        for (Index g = 0; g < n_groups; ++g) {
            gene_means[g] *= inv_counts[g];
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
            normalize_gene(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            normalize_gene(gene);
        }
    }

    scl::memory::aligned_free(inv_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(group_counts, SCL_ALIGNMENT);
}

// =============================================================================
// Percent Expressed - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void percent_expressed(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Array<Real> pct_expr,
    Index n_genes,
    Real threshold = Real(0)
) {
    const Index n_cells = IsCSR ? X.rows() : X.cols();
    const Size total = static_cast<Size>(n_genes) * static_cast<Size>(n_groups);

    SCL_CHECK_DIM(group_labels.len >= static_cast<Size>(n_cells),
                  "Markers: group_labels buffer too small");
    SCL_CHECK_DIM(pct_expr.len >= total, "Markers: pct_expr buffer too small");

    std::memset(pct_expr.ptr, 0, total * sizeof(Real));

    // Count cells per group
    auto group_counts_ptr = scl::memory::aligned_alloc<Index>(n_groups, SCL_ALIGNMENT);

    Index* group_counts = group_counts_ptr.release();
    std::memset(group_counts, 0, static_cast<Size>(n_groups) * sizeof(Index));

    for (Index c = 0; c < n_cells; ++c) {
        Index g = group_labels[c];
        if (SCL_LIKELY(g >= 0 && g < n_groups)) {
            ++group_counts[g];
        }
    }

    // Precompute inverse counts
    auto inv_counts_ptr = scl::memory::aligned_alloc<Real>(n_groups, SCL_ALIGNMENT);

    Real* inv_counts = inv_counts_ptr.release();
    for (Index g = 0; g < n_groups; ++g) {
        inv_counts[g] = (group_counts[g] > 0) ?
            Real(1) / static_cast<Real>(group_counts[g]) : Real(0);
    }

    // Count expressing cells
    if constexpr (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            Index g = group_labels[c];
            if (SCL_UNLIKELY(g < 0 || g >= n_groups)) continue;

            auto indices = X.row_indices_unsafe(c);
            auto values = X.row_values_unsafe(c);
            const Index len = X.row_length_unsafe(c);

            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (SCL_LIKELY(gene < n_genes && static_cast<Real>(values[k]) > threshold)) {
                    pct_expr[static_cast<Index>(static_cast<Size>(gene) * n_groups + g)] += Real(1);
                }
            }
        }
    } else {
        const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

        auto process_gene = [&](Index gene) {
            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            Real* gene_pct = pct_expr.ptr + static_cast<Size>(gene) * n_groups;

            for (Index k = 0; k < len; ++k) {
                if (static_cast<Real>(values[k]) > threshold) {
                    Index c = indices[k];
                    Index g = group_labels[c];
                    if (SCL_LIKELY(g >= 0 && g < n_groups)) {
                        gene_pct[g] += Real(1);
                    }
                }
            }
        };

        if (use_parallel) {
            scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
                process_gene(static_cast<Index>(gene));
            });
        } else {
            for (Index gene = 0; gene < n_genes; ++gene) {
                process_gene(gene);
            }
        }
    }

    // Convert to percentage - parallelize
    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto normalize_gene = [&](Index gene) {
        Real* gene_pct = pct_expr.ptr + static_cast<Size>(gene) * n_groups;
        for (Index g = 0; g < n_groups; ++g) {
            gene_pct[g] *= inv_counts[g];
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
            normalize_gene(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            normalize_gene(gene);
        }
    }

    scl::memory::aligned_free(inv_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(group_counts, SCL_ALIGNMENT);
}

// =============================================================================
// Log Fold Change - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void log_fold_change(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Index target_group,
    Array<Real> log_fc,
    Index n_genes,
    Real pseudo_count = config::PSEUDO_COUNT
) {
    SCL_CHECK_ARG(target_group >= 0 && target_group < n_groups,
                  "Markers: target_group out of range");
    SCL_CHECK_DIM(log_fc.len >= static_cast<Size>(n_genes),
                  "Markers: log_fc buffer too small");

    const Size mean_size = static_cast<Size>(n_genes) * static_cast<Size>(n_groups);
    auto mean_expr_ptr = scl::memory::aligned_alloc<Real>(mean_size, SCL_ALIGNMENT);

    Real* mean_expr = mean_expr_ptr.release();
    group_mean_expression(X, group_labels, n_groups, Array<Real>(mean_expr, mean_size), n_genes);

    // Extract target means and compute other means
    auto target_means_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* target_means = target_means_ptr.release();
    auto other_means_ptr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    Real* other_means = other_means_ptr.release();

    Real inv_other_count = (n_groups > 1) ? Real(1) / static_cast<Real>(n_groups - 1) : Real(0);

    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto compute_means = [&](Index gene) {
        const Real* gene_means = mean_expr + static_cast<Size>(gene) * n_groups;
        target_means[gene] = gene_means[target_group];

        Real other_sum = Real(0);
        for (Index g = 0; g < n_groups; ++g) {
            if (g != target_group) {
                other_sum += gene_means[g];
            }
        }
        other_means[gene] = other_sum * inv_other_count;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
            compute_means(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            compute_means(gene);
        }
    }

    // Compute log2 fold change with SIMD
    detail::compute_log2_fc_batch(target_means, other_means, log_fc.ptr, n_genes, pseudo_count);

    scl::memory::aligned_free(other_means, SCL_ALIGNMENT);
    scl::memory::aligned_free(target_means, SCL_ALIGNMENT);
    scl::memory::aligned_free(mean_expr, SCL_ALIGNMENT);
}

// =============================================================================
// One-vs-Rest Statistics - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void one_vs_rest_stats(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index target_group,
    Array<Real> log_fc,
    Array<Real> effect_size,
    Array<Real> pct_in,
    Array<Real> pct_out,
    Index n_genes
) {
    const Index n_cells = IsCSR ? X.rows() : X.cols();
    const Size n_genes_sz = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(log_fc.len >= n_genes_sz, "Markers: log_fc buffer too small");
    SCL_CHECK_DIM(effect_size.len >= n_genes_sz, "Markers: effect_size buffer too small");
    SCL_CHECK_DIM(pct_in.len >= n_genes_sz, "Markers: pct_in buffer too small");
    SCL_CHECK_DIM(pct_out.len >= n_genes_sz, "Markers: pct_out buffer too small");

    // Allocate workspace
    auto sum_in_ptr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);

    Real* sum_in = sum_in_ptr.release();
    auto sum_out_ptr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);

    Real* sum_out = sum_out_ptr.release();
    auto sum_sq_in_ptr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);

    Real* sum_sq_in = sum_sq_in_ptr.release();
    auto sum_sq_out_ptr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);

    Real* sum_sq_out = sum_sq_out_ptr.release();
    auto count_in_ptr = scl::memory::aligned_alloc<Index>(n_genes_sz, SCL_ALIGNMENT);

    Index* count_in = count_in_ptr.release();
    auto count_out_ptr = scl::memory::aligned_alloc<Index>(n_genes_sz, SCL_ALIGNMENT);

    Index* count_out = count_out_ptr.release();

    std::memset(sum_in, 0, n_genes_sz * sizeof(Real));
    std::memset(sum_out, 0, n_genes_sz * sizeof(Real));
    std::memset(sum_sq_in, 0, n_genes_sz * sizeof(Real));
    std::memset(sum_sq_out, 0, n_genes_sz * sizeof(Real));
    std::memset(count_in, 0, n_genes_sz * sizeof(Index));
    std::memset(count_out, 0, n_genes_sz * sizeof(Index));

    // Count total cells in/out
    Index n_in = 0, n_out = 0;
    for (Index c = 0; c < n_cells; ++c) {
        Index g = group_labels[c];
        if (g == target_group) ++n_in;
        else if (g >= 0) ++n_out;
    }

    // Precompute inverse counts
    Real inv_n_in = (n_in > 0) ? Real(1) / static_cast<Real>(n_in) : Real(0);
    Real inv_n_out = (n_out > 0) ? Real(1) / static_cast<Real>(n_out) : Real(0);

    // Accumulate statistics
    if constexpr (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            Index g = group_labels[c];
            if (SCL_UNLIKELY(g < 0)) continue;

            bool is_target = (g == target_group);

            auto indices = X.row_indices_unsafe(c);
            auto values = X.row_values_unsafe(c);
            const Index len = X.row_length_unsafe(c);

            if (is_target) {
                for (Index k = 0; k < len; ++k) {
                    Index gene = indices[k];
                    if (SCL_LIKELY(gene < n_genes)) {
                        Real val = static_cast<Real>(values[k]);
                        sum_in[gene] += val;
                        sum_sq_in[gene] += val * val;
                        ++count_in[gene];
                    }
                }
            } else {
                for (Index k = 0; k < len; ++k) {
                    Index gene = indices[k];
                    if (SCL_LIKELY(gene < n_genes)) {
                        Real val = static_cast<Real>(values[k]);
                        sum_out[gene] += val;
                        sum_sq_out[gene] += val * val;
                        ++count_out[gene];
                    }
                }
            }
        }
    } else {
        // CSC: parallelize over genes
        const bool use_parallel = (n_genes_sz >= config::PARALLEL_THRESHOLD);

        auto process_gene = [&](Index gene) {
            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            Real local_sum_in = Real(0), local_sum_out = Real(0);
            Real local_sq_in = Real(0), local_sq_out = Real(0);
            Index local_count_in = 0, local_count_out = 0;

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                Index g = group_labels[c];
                if (SCL_UNLIKELY(g < 0)) continue;

                Real val = static_cast<Real>(values[k]);
                Real val_sq = val * val;

                if (g == target_group) {
                    local_sum_in += val;
                    local_sq_in += val_sq;
                    ++local_count_in;
                } else {
                    local_sum_out += val;
                    local_sq_out += val_sq;
                    ++local_count_out;
                }
            }

            sum_in[gene] = local_sum_in;
            sum_out[gene] = local_sum_out;
            sum_sq_in[gene] = local_sq_in;
            sum_sq_out[gene] = local_sq_out;
            count_in[gene] = local_count_in;
            count_out[gene] = local_count_out;
        };

        if (use_parallel) {
            scl::threading::parallel_for(Size(0), n_genes_sz, [&](size_t gene) {
                process_gene(static_cast<Index>(gene));
            });
        } else {
            for (Index gene = 0; gene < n_genes; ++gene) {
                process_gene(gene);
            }
        }
    }

    // Compute statistics per gene - parallelize
    const bool use_parallel = (n_genes_sz >= config::PARALLEL_THRESHOLD);

    auto compute_stats = [&](Index gene) {
        Real mean_in_val = sum_in[gene] * inv_n_in;
        Real mean_out_val = sum_out[gene] * inv_n_out;

        // Variance: E[X^2] - E[X]^2
        Real var_in = Real(0), var_out = Real(0);
        if (n_in > 1) {
            var_in = sum_sq_in[gene] * inv_n_in - mean_in_val * mean_in_val;
            var_in = scl::algo::max2(var_in, Real(0));
        }
        if (n_out > 1) {
            var_out = sum_sq_out[gene] * inv_n_out - mean_out_val * mean_out_val;
            var_out = scl::algo::max2(var_out, Real(0));
        }

        // Log fold change
        log_fc[gene] = std::log2(mean_in_val + config::PSEUDO_COUNT) -
                       std::log2(mean_out_val + config::PSEUDO_COUNT);

        // Cohen's d
        effect_size[gene] = detail::cohens_d(mean_in_val, mean_out_val, var_in, var_out, n_in, n_out);

        // Percent expressed
        pct_in[gene] = static_cast<Real>(count_in[gene]) * inv_n_in;
        pct_out[gene] = static_cast<Real>(count_out[gene]) * inv_n_out;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_genes_sz, [&](size_t gene) {
            compute_stats(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            compute_stats(gene);
        }
    }

    scl::memory::aligned_free(count_out, SCL_ALIGNMENT);
    scl::memory::aligned_free(count_in, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_sq_out, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_sq_in, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_out, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_in, SCL_ALIGNMENT);
}

// =============================================================================
// Tau Specificity Score - Optimized
// =============================================================================

inline void tau_specificity(
    Array<const Real> group_means,
    Index n_genes,
    Index n_groups,
    Array<Real> tau_scores
) {
    SCL_CHECK_DIM(tau_scores.len >= static_cast<Size>(n_genes),
                  "Markers: tau_scores buffer too small");

    if (SCL_UNLIKELY(n_groups <= 1)) {
        std::memset(tau_scores.ptr, 0, static_cast<Size>(n_genes) * sizeof(Real));
        return;
    }

    const Real inv_n_minus_1 = Real(1) / static_cast<Real>(n_groups - 1);
    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto compute_tau = [&](Index gene) {
        const Real* means = group_means.ptr + static_cast<Size>(gene) * n_groups;

        // Find max with SIMD
        namespace s = scl::simd;
        using SimdTag = s::SimdTagFor<Real>;
        const SimdTag d;
        const Size lanes = s::Lanes(d);

        Real max_expr = means[0];
        Index g = 1;

        if (n_groups >= static_cast<Index>(lanes)) {
            auto v_max = s::Set(d, means[0]);
            for (; g + static_cast<Index>(lanes) <= n_groups; g += static_cast<Index>(lanes)) {
                v_max = s::Max(v_max, s::Load(d, means + g));
            }
            max_expr = s::GetLane(s::MaxOfLanes(d, v_max));
        }

        for (; g < n_groups; ++g) {
            max_expr = scl::algo::max2(max_expr, means[g]);
        }

        if (SCL_UNLIKELY(max_expr < config::MIN_EXPR)) {
            tau_scores[gene] = Real(0);
            return;
        }

        Real inv_max = Real(1) / max_expr;

        // Compute tau sum with SIMD
        Real tau_sum = Real(0);
        g = 0;

        if (n_groups >= static_cast<Index>(lanes)) {
            auto v_one = s::Set(d, Real(1));
            auto v_inv_max = s::Set(d, inv_max);
            auto v_sum = s::Zero(d);

            for (; g + static_cast<Index>(lanes) <= n_groups; g += static_cast<Index>(lanes)) {
                auto v_means = s::Load(d, means + g);
                auto v_scaled = s::Mul(v_means, v_inv_max);
                v_sum = s::Add(v_sum, s::Sub(v_one, v_scaled));
            }

            tau_sum = s::GetLane(s::SumOfLanes(d, v_sum));
        }

        for (; g < n_groups; ++g) {
            tau_sum += Real(1) - means[g] * inv_max;
        }

        tau_scores[gene] = tau_sum * inv_n_minus_1;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
            compute_tau(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            compute_tau(gene);
        }
    }
}

// =============================================================================
// Gini Specificity Score - Optimized
// =============================================================================

inline void gini_specificity(
    Array<const Real> group_means,
    Index n_genes,
    Index n_groups,
    Array<Real> gini_scores
) {
    SCL_CHECK_DIM(gini_scores.len >= static_cast<Size>(n_genes),
                  "Markers: gini_scores buffer too small");

    if (SCL_UNLIKELY(n_groups <= 1)) {
        std::memset(gini_scores.ptr, 0, static_cast<Size>(n_genes) * sizeof(Real));
        return;
    }

    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace for sorting
    scl::threading::WorkspacePool<Real> sort_pool;
    if (use_parallel) {
        sort_pool.init(n_threads, static_cast<Size>(n_groups));
    }

    auto compute_gini = [&](Index gene, Real* sorted) {
        const Real* means = group_means.ptr + static_cast<Size>(gene) * n_groups;

        // Copy and sort ascending
        std::memcpy(sorted, means, static_cast<Size>(n_groups) * sizeof(Real));
        scl::sort::sort(Array<Real>(sorted, static_cast<Size>(n_groups)));

        gini_scores[gene] = detail::gini_coefficient(sorted, static_cast<Size>(n_groups));
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene, size_t thread_rank) {
            Real* sorted = sort_pool.get(thread_rank);
            compute_gini(static_cast<Index>(gene), sorted);
        });
    } else {
        auto sorted_ptr = scl::memory::aligned_alloc<Real>(n_groups, SCL_ALIGNMENT);

        Real* sorted = sorted_ptr.release();
        for (Index gene = 0; gene < n_genes; ++gene) {
            compute_gini(gene, sorted);
        }
        scl::memory::aligned_free(sorted, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Rank Genes by Score - Optimized
// =============================================================================

inline void rank_genes_by_score(
    Array<const Real> scores,
    Index n_genes,
    Array<Index> ranked_indices,
    Array<Real> ranked_scores
) {
    SCL_CHECK_DIM(ranked_indices.len >= static_cast<Size>(n_genes),
                  "Markers: ranked_indices buffer too small");
    SCL_CHECK_DIM(ranked_scores.len >= static_cast<Size>(n_genes),
                  "Markers: ranked_scores buffer too small");

    // Initialize indices
    for (Index i = 0; i < n_genes; ++i) {
        ranked_indices[i] = i;
        ranked_scores[i] = scores[i];
    }

    // Use efficient sort
    if (n_genes > 1) {
        scl::sort::sort_pairs_descending(
            Array<Real>(ranked_scores.ptr, static_cast<Size>(n_genes)),
            Array<Index>(ranked_indices.ptr, static_cast<Size>(n_genes))
        );
    }
}

// =============================================================================
// Rank Genes Groups (All Groups) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void rank_genes_groups(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Index n_genes,
    RankingMethod method,
    Array<Index> all_ranked_indices,
    Array<Real> all_ranked_scores
) {
    const Size total = static_cast<Size>(n_groups) * static_cast<Size>(n_genes);
    const Size n_genes_sz = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(all_ranked_indices.len >= total, "Markers: ranked_indices buffer too small");
    SCL_CHECK_DIM(all_ranked_scores.len >= total, "Markers: ranked_scores buffer too small");

    const bool use_parallel = (static_cast<Size>(n_groups) >= 4);  // Parallelize over groups
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    struct GroupWorkspace {
        Real* log_fc;
        Real* effect_size;
        Real* pct_in;
        Real* pct_out;
        Real* scores;
    };

    GroupWorkspace* workspaces = nullptr;
    size_t ws_count = use_parallel ? n_threads : 1;
    workspaces = new GroupWorkspace[ws_count];

    for (Size t = 0; t < ws_count; ++t) {
        workspaces[t].log_fc = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        workspaces[t].effect_size = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        workspaces[t].pct_in = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        workspaces[t].pct_out = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
        workspaces[t].scores = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);
    }

    auto process_group = [&](Index g, size_t ws_idx) {
        GroupWorkspace& ws = workspaces[ws_idx];

        one_vs_rest_stats(
            X, group_labels, n_groups, g,
            Array<Real>(ws.log_fc, n_genes_sz),
            Array<Real>(ws.effect_size, n_genes_sz),
            Array<Real>(ws.pct_in, n_genes_sz),
            Array<Real>(ws.pct_out, n_genes_sz),
            n_genes
        );

        // Compute final scores
        switch (method) {
            case RankingMethod::FoldChange:
                std::memcpy(ws.scores, ws.log_fc, n_genes_sz * sizeof(Real));
                break;

            case RankingMethod::EffectSize:
                std::memcpy(ws.scores, ws.effect_size, n_genes_sz * sizeof(Real));
                break;

            case RankingMethod::Combined:
                for (Index i = 0; i < n_genes; ++i) {
                    Real pct_diff = ws.pct_in[i] - ws.pct_out[i];
                    ws.scores[i] = ws.effect_size[i] * scl::algo::max2(pct_diff, Real(0.01));
                }
                break;

            default:
                std::memcpy(ws.scores, ws.log_fc, n_genes_sz * sizeof(Real));
                break;
        }

        // Rank genes
        Index* group_indices = all_ranked_indices.ptr + static_cast<Size>(g) * n_genes;
        Real* group_scores = all_ranked_scores.ptr + static_cast<Size>(g) * n_genes;

        rank_genes_by_score(
            Array<const Real>(ws.scores, n_genes_sz),
            n_genes,
            Array<Index>(group_indices, n_genes_sz),
            Array<Real>(group_scores, n_genes_sz)
        );
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_groups), [&](size_t g, size_t thread_rank) {
            process_group(static_cast<Index>(g), thread_rank);
        });
    } else {
        for (Index g = 0; g < n_groups; ++g) {
            process_group(g, 0);
        }
    }

    // Cleanup
    for (Size t = 0; t < ws_count; ++t) {
        scl::memory::aligned_free(workspaces[t].scores, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].pct_out, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].pct_in, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].effect_size, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].log_fc, SCL_ALIGNMENT);
    }

    delete[] workspaces;
}

// =============================================================================
// Filter Markers by Criteria - Optimized
// =============================================================================

inline Index filter_markers(
    Array<const Index> candidate_genes,
    Array<const Real> log_fc,
    Array<const Real> pct_expr,
    Real min_fc,
    Real min_pct,
    Array<Index> filtered_genes
) {
    const Size n = candidate_genes.len;
    SCL_CHECK_DIM(filtered_genes.len >= n, "Markers: filtered_genes buffer too small");

    Real log2_min_fc = std::log2(min_fc);
    Index count = 0;

    for (Size i = 0; i < n; ++i) {
        Index gene = candidate_genes[static_cast<Index>(i)];
        if (SCL_LIKELY(log_fc[gene] >= log2_min_fc && pct_expr[gene] >= min_pct)) {
            filtered_genes[count++] = gene;
        }
    }

    return count;
}

// =============================================================================
// Top N Markers
// =============================================================================

inline Index top_n_markers(
    Array<const Index> ranked_indices,
    Array<const Real> ranked_scores,
    Index n_top,
    Array<Index> top_indices,
    Array<Real> top_scores
) {
    auto n = static_cast<Index>(ranked_indices.len);
    Index actual_n = scl::algo::min2(n_top, n);

    SCL_CHECK_DIM(top_indices.len >= static_cast<Size>(actual_n),
                  "Markers: top_indices buffer too small");

    std::memcpy(top_indices.ptr, ranked_indices.ptr, static_cast<Size>(actual_n) * sizeof(Index));

    if (top_scores.ptr != nullptr && top_scores.len >= static_cast<Size>(actual_n)) {
        std::memcpy(top_scores.ptr, ranked_scores.ptr, static_cast<Size>(actual_n) * sizeof(Real));
    }

    return actual_n;
}

// =============================================================================
// Marker Overlap (Jaccard Similarity) - Optimized
// =============================================================================

inline void marker_overlap_jaccard(
    const Index* const* marker_sets,
    const Index* set_sizes,
    Index n_sets,
    Array<Real> overlap_matrix
) {
    const Size total = static_cast<Size>(n_sets) * static_cast<Size>(n_sets);
    SCL_CHECK_DIM(overlap_matrix.len >= total, "Markers: overlap_matrix buffer too small");

    // Initialize with identity
    std::memset(overlap_matrix.ptr, 0, total * sizeof(Real));
    for (Index i = 0; i < n_sets; ++i) {
        overlap_matrix[static_cast<Index>(static_cast<Size>(i) * n_sets + i)] = Real(1);
    }

    // Parallelize over pairs
    const Size n_pairs = static_cast<Size>(n_sets) * (n_sets - 1) / 2;
    const bool use_parallel = (n_pairs >= config::PARALLEL_THRESHOLD);

    auto compute_pair = [&](Index i, Index j) {
        const Index* set_i = marker_sets[i];
        const Index* set_j = marker_sets[j];
        Index size_i = set_sizes[i];
        Index size_j = set_sizes[j];

        // For better performance, use sorted sets and merge
        Index intersection = 0;
        // Simple O(n*m) for now - could use hash set for large sets
        for (Index ki = 0; ki < size_i; ++ki) {
            Index val = set_i[ki];
            for (Index kj = 0; kj < size_j; ++kj) {
                if (set_j[kj] == val) {
                    ++intersection;
                    break;
                }
            }
        }

        Index union_size = size_i + size_j - intersection;
        Real jaccard = (union_size > 0) ?
            static_cast<Real>(intersection) / static_cast<Real>(union_size) : Real(0);

        overlap_matrix[static_cast<Index>(static_cast<Size>(i) * n_sets + j)] = jaccard;
        overlap_matrix[static_cast<Index>(static_cast<Size>(j) * n_sets + i)] = jaccard;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_sets), [&](size_t i) {
            for (Index j = static_cast<Index>(i) + 1; j < n_sets; ++j) {
                compute_pair(static_cast<Index>(i), j);
            }
        });
    } else {
        for (Index i = 0; i < n_sets; ++i) {
            for (Index j = i + 1; j < n_sets; ++j) {
                compute_pair(i, j);
            }
        }
    }
}

// =============================================================================
// Find Unique Markers - Optimized
// =============================================================================

inline Index find_unique_markers(
    const Index* const* marker_sets,
    const Index* set_sizes,
    Index n_sets,
    Index target_set,
    Array<Index> unique_markers
) {
    Index target_size = set_sizes[target_set];
    const Index* target = marker_sets[target_set];

    // Build hash-like lookup for other sets
    // For simplicity, use a sorted merge approach
    Index count = 0;

    for (Index i = 0; i < target_size; ++i) {
        Index gene = target[i];
        bool is_unique = true;

        for (Index s = 0; s < n_sets && is_unique; ++s) {
            if (s == target_set) continue;

            const Index* other_set = marker_sets[s];
            Index other_size = set_sizes[s];

            // Binary search if sets are sorted, otherwise linear
            for (Index j = 0; j < other_size; ++j) {
                if (other_set[j] == gene) {
                    is_unique = false;
                    break;
                }
            }
        }

        if (is_unique && count < static_cast<Index>(unique_markers.len)) {
            unique_markers[count++] = gene;
        }
    }

    return count;
}

// =============================================================================
// Differential Expression Score (Volcano-style) - Optimized
// =============================================================================

inline void volcano_score(
    Array<const Real> log_fc,
    Array<const Real> effect_size,
    Array<Real> volcano_scores,
    Real fc_weight = Real(0.5),
    Real es_weight = Real(0.5)
) {
    const Size n = log_fc.len;
    SCL_CHECK_DIM(volcano_scores.len >= n, "Markers: volcano_scores buffer too small");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_fc_weight = s::Set(d, fc_weight);
    auto v_es_weight = s::Set(d, es_weight);

    Size i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v_fc = s::Load(d, log_fc.ptr + i);
        auto v_es = s::Load(d, effect_size.ptr + i);
        auto v_score = s::MulAdd(v_fc_weight, v_fc, s::Mul(v_es_weight, v_es));
        s::Store(v_score, d, volcano_scores.ptr + i);
    }

    for (; i < n; ++i) {
        volcano_scores[static_cast<Index>(i)] = fc_weight * log_fc[static_cast<Index>(i)] + es_weight * effect_size[static_cast<Index>(i)];
    }
}

// =============================================================================
// Expression Entropy - Optimized
// =============================================================================

inline void expression_entropy(
    Array<const Real> group_means,
    Index n_genes,
    Index n_groups,
    Array<Real> entropy
) {
    SCL_CHECK_DIM(entropy.len >= static_cast<Size>(n_genes),
                  "Markers: entropy buffer too small");

    const Real max_entropy = std::log(static_cast<Real>(n_groups));
    const Real inv_max_entropy = (max_entropy > Real(0)) ? Real(1) / max_entropy : Real(0);

    const bool use_parallel = (static_cast<Size>(n_genes) >= config::PARALLEL_THRESHOLD);

    auto compute_entropy = [&](Index gene) {
        const Real* means = group_means.ptr + static_cast<Size>(gene) * n_groups;

        // Compute sum with SIMD
        namespace s = scl::simd;
        using SimdTag = s::SimdTagFor<Real>;
        const SimdTag d;
        const Size lanes = s::Lanes(d);

        auto v_sum = s::Zero(d);
        Index g = 0;

        for (; g + static_cast<Index>(lanes) <= n_groups; g += static_cast<Index>(lanes)) {
            v_sum = s::Add(v_sum, s::Load(d, means + g));
        }

        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; g < n_groups; ++g) {
            sum += means[g];
        }

        if (SCL_UNLIKELY(sum < config::MIN_EXPR)) {
            entropy[gene] = Real(0);
            return;
        }

        Real inv_sum = Real(1) / sum;

        // Compute Shannon entropy
        Real H = Real(0);
        for (g = 0; g < n_groups; ++g) {
            Real p = means[g] * inv_sum;
            if (SCL_LIKELY(p > config::MIN_EXPR)) {
                H -= p * std::log(p);
            }
        }

        entropy[gene] = H * inv_max_entropy;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_genes), [&](size_t gene) {
            compute_entropy(static_cast<Index>(gene));
        });
    } else {
        for (Index gene = 0; gene < n_genes; ++gene) {
            compute_entropy(gene);
        }
    }
}

} // namespace scl::kernel::markers
