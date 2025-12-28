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

#include <cmath>

// =============================================================================
// FILE: scl/kernel/comparison.hpp
// BRIEF: Group comparison and differential abundance analysis
//
// APPLICATIONS:
// - Case-control studies
// - Composition analysis across conditions
// - Differential abundance testing (DAseq, Milo-style)
// =============================================================================

namespace scl::kernel::comparison {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_GROUP = 3;
    constexpr Size PERMUTATION_COUNT = 1000;
    constexpr Size PARALLEL_THRESHOLD = 32;
}

namespace detail {

// Binary search for gene in sorted CSR indices
SCL_FORCE_INLINE Index binary_search_gene(
    const Index* indices,
    Index start,
    Index end,
    Index gene
) noexcept {
    while (start < end) {
        Index mid = start + (end - start) / 2;
        if (indices[mid] < gene) {
            start = mid + 1;
        } else if (indices[mid] > gene) {
            end = mid;
        } else {
            return mid;
        }
    }
    return -1;
}

// Chi-squared test statistic for contingency table
SCL_FORCE_INLINE Real chi_squared_stat(
    const Size* observed,
    const Real* expected,
    Size n
) {
    Real chi2 = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        if (expected[i] > config::EPSILON) {
            Real diff = static_cast<Real>(observed[i]) - expected[i];
            chi2 += diff * diff / expected[i];
        }
    }
    return chi2;
}

// Fisher's exact test approximation using chi-squared
SCL_FORCE_INLINE Real fisher_pvalue_approx(
    Size a, Size b, Size c, Size d
) {
    Size total = a + b + c + d;
    if (total == 0) return Real(1.0);

    Real total_r = static_cast<Real>(total);
    Real row1 = static_cast<Real>(a + b);
    Real row2 = static_cast<Real>(c + d);
    Real col1 = static_cast<Real>(a + c);
    Real col2 = static_cast<Real>(b + d);

    Real exp_a = row1 * col1 / total_r;
    Real exp_b = row1 * col2 / total_r;
    Real exp_c = row2 * col1 / total_r;
    Real exp_d = row2 * col2 / total_r;

    Size obs[4] = {a, b, c, d};
    Real exp[4] = {exp_a, exp_b, exp_c, exp_d};

    Real chi2 = chi_squared_stat(obs, exp, 4);

    // P-value from chi-squared distribution with 1 df
    // Using Wilson-Hilferty approximation
    if (chi2 <= Real(0.0)) return Real(1.0);

    Real z = std::pow(chi2, Real(1.0) / Real(3.0));
    Real mean = Real(1.0) - Real(2.0) / Real(9.0);
    Real std_dev = std::sqrt(Real(2.0) / Real(9.0));
    Real normal_z = (z - mean) / std_dev;

    // Standard normal CDF
    Real p = Real(0.5) * (Real(1.0) + std::erf(normal_z / std::sqrt(Real(2.0))));
    return Real(1.0) - p;
}

// Wilcoxon rank sum test (optimized with VQSort)
SCL_FORCE_INLINE Real wilcoxon_pvalue(
    Real* group1, Size n1,
    Real* group2, Size n2
) {
    if (n1 < config::MIN_CELLS_PER_GROUP || n2 < config::MIN_CELLS_PER_GROUP) {
        return Real(1.0);
    }

    Size n_total = n1 + n2;
    Real* combined = scl::memory::aligned_alloc<Real>(n_total, SCL_ALIGNMENT);
    Index* indices = scl::memory::aligned_alloc<Index>(n_total, SCL_ALIGNMENT);
    Real* ranks = scl::memory::aligned_alloc<Real>(n_total, SCL_ALIGNMENT);

    // Combine groups
    scl::algo::copy(group1, combined, n1);
    scl::algo::copy(group2, combined + n1, n2);
    for (Size i = 0; i < n_total; ++i) {
        indices[i] = static_cast<Index>(i);
    }

    // Use VQSort for SIMD-optimized sorting
    scl::sort::sort_pairs(
        Array<Real>(combined, n_total),
        Array<Index>(indices, n_total)
    );

    // Assign ranks based on sorted order
    for (Size i = 0; i < n_total; ++i) {
        ranks[indices[i]] = static_cast<Real>(i + 1);
    }

    // Handle ties (average ranks)
    Size i = 0;
    while (i < n_total) {
        Size j = i;
        while (j < n_total && combined[j] == combined[i]) {
            ++j;
        }
        if (j > i + 1) {
            Real avg_rank = Real(0.0);
            for (Size k = i; k < j; ++k) {
                avg_rank += ranks[indices[k]];
            }
            avg_rank /= static_cast<Real>(j - i);
            for (Size k = i; k < j; ++k) {
                ranks[indices[k]] = avg_rank;
            }
        }
        i = j;
    }

    // Compute rank sum for group 1
    Real W = Real(0.0);
    for (Size i = 0; i < n1; ++i) {
        W += ranks[i];
    }

    // Expected value and variance under null
    Real n1_r = static_cast<Real>(n1);
    Real n2_r = static_cast<Real>(n2);
    Real n_r = static_cast<Real>(n_total);

    Real E_W = n1_r * (n_r + Real(1.0)) / Real(2.0);
    Real Var_W = n1_r * n2_r * (n_r + Real(1.0)) / Real(12.0);

    // Normal approximation
    Real z = (W - E_W) / std::sqrt(Var_W);
    Real abs_z = (z >= Real(0.0)) ? z : -z;
    Real p = Real(2.0) * (Real(1.0) - Real(0.5) * (Real(1.0) +
        std::erf(abs_z / std::sqrt(Real(2.0)))));

    scl::memory::aligned_free(combined);
    scl::memory::aligned_free(indices);
    scl::memory::aligned_free(ranks);

    return p;
}

} // namespace detail

// =============================================================================
// Composition Analysis
// =============================================================================

void composition_analysis(
    Array<const Index> cell_types,
    Array<const Index> conditions,
    Real* proportions,  // [n_types * n_conditions]
    Real* p_values,     // [n_types]
    Size n_types,
    Size n_conditions
) {
    const Size n_cells = cell_types.len;
    SCL_CHECK_DIM(n_cells == conditions.len, "Cell types and conditions must have same length");

    if (n_cells == 0 || n_types == 0 || n_conditions == 0) return;

    // Count cells per type per condition
    Size* counts = scl::memory::aligned_alloc<Size>(n_types * n_conditions, SCL_ALIGNMENT);
    Size* total_per_condition = scl::memory::aligned_alloc<Size>(n_conditions, SCL_ALIGNMENT);

    scl::algo::zero(counts, n_types * n_conditions);
    scl::algo::zero(total_per_condition, n_conditions);

    for (Size i = 0; i < n_cells; ++i) {
        Index type = cell_types.ptr[i];
        Index cond = conditions.ptr[i];
        if (type < static_cast<Index>(n_types) && cond < static_cast<Index>(n_conditions)) {
            ++counts[type * n_conditions + cond];
            ++total_per_condition[cond];
        }
    }

    // Compute proportions (parallel over types)
    scl::threading::parallel_for(Size(0), n_types, [&](size_t t) {
        for (Size c = 0; c < n_conditions; ++c) {
            if (total_per_condition[c] > 0) {
                proportions[t * n_conditions + c] =
                    static_cast<Real>(counts[t * n_conditions + c]) /
                    static_cast<Real>(total_per_condition[c]);
            } else {
                proportions[t * n_conditions + c] = Real(0.0);
            }
        }
    });

    // Chi-squared test for each cell type (parallel)
    Real total_r = static_cast<Real>(n_cells);

    scl::threading::parallel_for(Size(0), n_types, [&](size_t t) {
        Size total_type = 0;
        for (Size c = 0; c < n_conditions; ++c) {
            total_type += counts[t * n_conditions + c];
        }

        if (total_type == 0) {
            p_values[t] = Real(1.0);
            return;
        }

        // Expected counts under uniform distribution
        Real* expected = scl::memory::aligned_alloc<Real>(n_conditions, SCL_ALIGNMENT);

        for (Size c = 0; c < n_conditions; ++c) {
            expected[c] = static_cast<Real>(total_type) *
                         static_cast<Real>(total_per_condition[c]) / total_r;
        }

        Real chi2 = detail::chi_squared_stat(
            counts + t * n_conditions, expected, n_conditions);

        // P-value from chi-squared with (n_conditions - 1) df
        Real df = static_cast<Real>(n_conditions - 1);
        if (df > Real(0.0) && chi2 > Real(0.0)) {
            Real z = std::pow(chi2 / df, Real(1.0) / Real(3.0));
            Real mean = Real(1.0) - Real(2.0) / (Real(9.0) * df);
            Real std_dev = std::sqrt(Real(2.0) / (Real(9.0) * df));
            Real normal_z = (z - mean) / std_dev;
            Real p = Real(1.0) - Real(0.5) * (Real(1.0) +
                std::erf(normal_z / std::sqrt(Real(2.0))));
            p_values[t] = p;
        } else {
            p_values[t] = Real(1.0);
        }

        scl::memory::aligned_free(expected);
    });

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(total_per_condition);
}

// =============================================================================
// Abundance Test
// =============================================================================

void abundance_test(
    Array<const Index> cluster_labels,
    Array<const Index> condition,
    Array<Real> fold_changes,
    Array<Real> p_values
) {
    const Size n_cells = cluster_labels.len;
    SCL_CHECK_DIM(n_cells == condition.len, "Labels and conditions must have same length");

    if (n_cells == 0) return;

    // Find number of clusters and conditions
    Index n_clusters = 0;
    Index n_conds = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_clusters = scl::algo::max2(n_clusters, cluster_labels.ptr[i] + 1);
        n_conds = scl::algo::max2(n_conds, condition.ptr[i] + 1);
    }

    SCL_CHECK_DIM(fold_changes.len >= static_cast<Size>(n_clusters),
        "Fold changes array too small");
    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n_clusters),
        "P-values array too small");

    if (n_conds < 2) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_clusters), [&](size_t c) {
            fold_changes.ptr[c] = Real(1.0);
            p_values.ptr[c] = Real(1.0);
        });
        return;
    }

    // Count cells per cluster per condition
    Size* counts = scl::memory::aligned_alloc<Size>(n_clusters * n_conds, SCL_ALIGNMENT);
    Size* total_per_cond = scl::memory::aligned_alloc<Size>(n_conds, SCL_ALIGNMENT);

    scl::algo::zero(counts, static_cast<Size>(n_clusters * n_conds));
    scl::algo::zero(total_per_cond, static_cast<Size>(n_conds));

    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = cluster_labels.ptr[i];
        Index cond = condition.ptr[i];
        ++counts[cluster * n_conds + cond];
        ++total_per_cond[cond];
    }

    // Compare condition 0 vs condition 1 (parallel over clusters)
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_clusters), [&](size_t c) {
        Real prop0 = Real(0.0), prop1 = Real(0.0);

        if (total_per_cond[0] > 0) {
            prop0 = static_cast<Real>(counts[c * n_conds + 0]) /
                   static_cast<Real>(total_per_cond[0]);
        }
        if (total_per_cond[1] > 0) {
            prop1 = static_cast<Real>(counts[c * n_conds + 1]) /
                   static_cast<Real>(total_per_cond[1]);
        }

        // Fold change
        if (prop0 > config::EPSILON) {
            fold_changes.ptr[c] = prop1 / prop0;
        } else if (prop1 > config::EPSILON) {
            fold_changes.ptr[c] = std::numeric_limits<Real>::infinity();
        } else {
            fold_changes.ptr[c] = Real(1.0);
        }

        // Fisher's exact test
        Size a = counts[c * n_conds + 0];
        Size b = counts[c * n_conds + 1];
        Size c_val = total_per_cond[0] - a;
        Size d = total_per_cond[1] - b;

        p_values.ptr[c] = detail::fisher_pvalue_approx(a, b, c_val, d);
    });

    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(total_per_cond);
}

// =============================================================================
// Differential Abundance (DA)
// =============================================================================

void differential_abundance(
    Array<const Index> cluster_labels,
    Array<const Index> sample_ids,
    Array<const Index> conditions,
    Array<Real> da_scores,
    Array<Real> p_values
) {
    const Size n_cells = cluster_labels.len;
    SCL_CHECK_DIM(n_cells == sample_ids.len, "Labels and sample IDs must have same length");
    SCL_CHECK_DIM(n_cells == conditions.len, "Labels and conditions must have same length");

    if (n_cells == 0) return;

    // Find number of clusters, samples, and conditions
    Index n_clusters = 0;
    Index n_samples = 0;
    Index n_conds = 0;

    for (Size i = 0; i < n_cells; ++i) {
        n_clusters = scl::algo::max2(n_clusters, cluster_labels.ptr[i] + 1);
        n_samples = scl::algo::max2(n_samples, sample_ids.ptr[i] + 1);
        n_conds = scl::algo::max2(n_conds, conditions.ptr[i] + 1);
    }

    SCL_CHECK_DIM(da_scores.len >= static_cast<Size>(n_clusters), "DA scores array too small");
    SCL_CHECK_DIM(p_values.len >= static_cast<Size>(n_clusters), "P-values array too small");

    if (n_conds < 2 || n_samples < 2) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_clusters), [&](size_t c) {
            da_scores.ptr[c] = Real(0.0);
            p_values.ptr[c] = Real(1.0);
        });
        return;
    }

    // Map samples to conditions
    Index* sample_to_cond = scl::memory::aligned_alloc<Index>(n_samples, SCL_ALIGNMENT);
    scl::algo::fill(sample_to_cond, static_cast<Size>(n_samples), static_cast<Index>(-1));

    for (Size i = 0; i < n_cells; ++i) {
        Index sample = sample_ids.ptr[i];
        sample_to_cond[sample] = conditions.ptr[i];
    }

    // Count cells per cluster per sample
    Size* counts = scl::memory::aligned_alloc<Size>(n_clusters * n_samples, SCL_ALIGNMENT);
    Size* total_per_sample = scl::memory::aligned_alloc<Size>(n_samples, SCL_ALIGNMENT);

    scl::algo::zero(counts, static_cast<Size>(n_clusters * n_samples));
    scl::algo::zero(total_per_sample, static_cast<Size>(n_samples));

    for (Size i = 0; i < n_cells; ++i) {
        Index cluster = cluster_labels.ptr[i];
        Index sample = sample_ids.ptr[i];
        ++counts[cluster * n_samples + sample];
        ++total_per_sample[sample];
    }

    // Compute proportions per sample (parallel)
    Real* props = scl::memory::aligned_alloc<Real>(n_clusters * n_samples, SCL_ALIGNMENT);
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_clusters), [&](size_t c) {
        for (Index s = 0; s < n_samples; ++s) {
            if (total_per_sample[s] > 0) {
                props[c * n_samples + s] = static_cast<Real>(counts[c * n_samples + s]) /
                                          static_cast<Real>(total_per_sample[s]);
            } else {
                props[c * n_samples + s] = Real(0.0);
            }
        }
    });

    // Count samples per condition
    Size n_cond0 = 0, n_cond1 = 0;
    for (Index s = 0; s < n_samples; ++s) {
        if (sample_to_cond[s] == 0) ++n_cond0;
        else if (sample_to_cond[s] == 1) ++n_cond1;
    }

    // Wilcoxon test for each cluster (parallel with WorkspacePool)
    scl::threading::WorkspacePool<Real> pool0;
    scl::threading::WorkspacePool<Real> pool1;
    pool0.init(scl::threading::Scheduler::get_num_threads(), n_cond0);
    pool1.init(scl::threading::Scheduler::get_num_threads(), n_cond1);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_clusters), [&](size_t c, size_t thread_rank) {
        Real* group0 = pool0.get(thread_rank);
        Real* group1 = pool1.get(thread_rank);

        Size idx0 = 0, idx1 = 0;
        Real sum0 = Real(0.0), sum1 = Real(0.0);

        for (Index s = 0; s < n_samples; ++s) {
            Real prop = props[c * n_samples + s];
            if (sample_to_cond[s] == 0 && idx0 < n_cond0) {
                group0[idx0++] = prop;
                sum0 += prop;
            } else if (sample_to_cond[s] == 1 && idx1 < n_cond1) {
                group1[idx1++] = prop;
                sum1 += prop;
            }
        }

        // DA score: log2 fold change of mean proportions
        Real mean0 = (idx0 > 0) ? sum0 / static_cast<Real>(idx0) : Real(0.0);
        Real mean1 = (idx1 > 0) ? sum1 / static_cast<Real>(idx1) : Real(0.0);

        if (mean0 > config::EPSILON && mean1 > config::EPSILON) {
            da_scores.ptr[c] = std::log2(mean1 / mean0);
        } else if (mean1 > config::EPSILON) {
            da_scores.ptr[c] = Real(10.0);
        } else if (mean0 > config::EPSILON) {
            da_scores.ptr[c] = Real(-10.0);
        } else {
            da_scores.ptr[c] = Real(0.0);
        }

        // Wilcoxon test
        p_values.ptr[c] = detail::wilcoxon_pvalue(group0, idx0, group1, idx1);
    });

    scl::memory::aligned_free(sample_to_cond);
    scl::memory::aligned_free(counts);
    scl::memory::aligned_free(total_per_sample);
    scl::memory::aligned_free(props);
}

// =============================================================================
// Condition Response
// =============================================================================

template <typename T, bool IsCSR>
void condition_response(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> conditions,
    Real* response_scores,  // [n_genes]
    Real* p_values,         // [n_genes]
    Size n_genes
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == conditions.len, "Expression rows must match conditions length");

    if (n_cells == 0 || n_genes == 0) return;

    // Find number of conditions
    Index n_conds = 0;
    for (Size i = 0; i < n_cells; ++i) {
        n_conds = scl::algo::max2(n_conds, conditions.ptr[i] + 1);
    }

    if (n_conds < 2) {
        scl::threading::parallel_for(Size(0), n_genes, [&](size_t g) {
            response_scores[g] = Real(0.0);
            p_values[g] = Real(1.0);
        });
        return;
    }

    // Count cells per condition
    Size n_cond0 = 0, n_cond1 = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (conditions.ptr[i] == 0) ++n_cond0;
        else if (conditions.ptr[i] == 1) ++n_cond1;
    }

    // Parallel over genes with WorkspacePool
    scl::threading::DualWorkspacePool<Real> pool;
    pool.init(scl::threading::Scheduler::get_num_threads(), n_cond0, n_cond1);

    scl::threading::parallel_for(Size(0), n_genes, [&](size_t g, size_t thread_rank) {
        Real* group0 = pool.get_first(thread_rank);
        Real* group1 = pool.get_second(thread_rank);

        Size idx0 = 0, idx1 = 0;
        Real sum0 = Real(0.0), sum1 = Real(0.0);

        // Gather expression values for this gene using binary search
        for (Size c = 0; c < n_cells; ++c) {
            Real val = Real(0.0);

            const Index row_start = expression.row_indices_unsafe()[c];
            const Index row_end = expression.row_indices_unsafe()[c + 1];

            // Binary search for gene in sorted CSR indices
            Index pos = detail::binary_search_gene(
                expression.col_indices_unsafe(), row_start, row_end, static_cast<Index>(g));
            if (pos >= 0) {
                val = static_cast<Real>(expression.values()[pos]);
            }

            if (conditions.ptr[c] == 0 && idx0 < n_cond0) {
                group0[idx0++] = val;
                sum0 += val;
            } else if (conditions.ptr[c] == 1 && idx1 < n_cond1) {
                group1[idx1++] = val;
                sum1 += val;
            }
        }

        // Response score: log2 fold change
        Real mean0 = (idx0 > 0) ? sum0 / static_cast<Real>(idx0) : Real(0.0);
        Real mean1 = (idx1 > 0) ? sum1 / static_cast<Real>(idx1) : Real(0.0);

        if (mean0 > config::EPSILON && mean1 > config::EPSILON) {
            response_scores[g] = std::log2((mean1 + config::EPSILON) / (mean0 + config::EPSILON));
        } else if (mean1 > config::EPSILON) {
            response_scores[g] = Real(10.0);
        } else if (mean0 > config::EPSILON) {
            response_scores[g] = Real(-10.0);
        } else {
            response_scores[g] = Real(0.0);
        }

        // Wilcoxon test
        p_values[g] = detail::wilcoxon_pvalue(group0, idx0, group1, idx1);
    });
}

// =============================================================================
// Effect Size (Cohen's d)
// =============================================================================

namespace detail {

// SIMD-optimized sum and sum of squared differences
SCL_HOT void compute_mean_and_variance(
    const Real* values,
    Size n,
    Real& mean,
    Real& variance
) {
    if (n == 0) {
        mean = Real(0.0);
        variance = Real(0.0);
        return;
    }

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Pass 1: Compute sum for mean
    auto v_sum = s::Zero(d);
    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, values + k));
    }
    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    for (; k < n; ++k) {
        sum += values[k];
    }
    mean = sum / static_cast<Real>(n);

    if (n < 2) {
        variance = Real(0.0);
        return;
    }

    // Pass 2: Compute variance
    auto v_mean = s::Set(d, mean);
    auto v_var = s::Zero(d);
    k = 0;
    for (; k + lanes <= n; k += lanes) {
        auto v = s::Load(d, values + k);
        auto diff = s::Sub(v, v_mean);
        v_var = s::MulAdd(diff, diff, v_var);
    }
    Real var_sum = s::GetLane(s::SumOfLanes(d, v_var));
    for (; k < n; ++k) {
        Real diff = values[k] - mean;
        var_sum += diff * diff;
    }
    variance = var_sum / static_cast<Real>(n - 1);
}

} // namespace detail

Real effect_size(
    Array<const Real> group1,
    Array<const Real> group2
) {
    const Size n1 = group1.len;
    const Size n2 = group2.len;

    if (n1 < 2 || n2 < 2) return Real(0.0);

    // Compute means and variances using SIMD
    Real mean1, var1, mean2, var2;
    detail::compute_mean_and_variance(group1.ptr, n1, mean1, var1);
    detail::compute_mean_and_variance(group2.ptr, n2, mean2, var2);

    // Pooled standard deviation
    Real pooled_var = ((static_cast<Real>(n1 - 1) * var1 +
                        static_cast<Real>(n2 - 1) * var2)) /
                      static_cast<Real>(n1 + n2 - 2);

    Real pooled_sd = std::sqrt(pooled_var);

    if (pooled_sd < config::EPSILON) {
        return Real(0.0);
    }

    // Cohen's d
    return (mean2 - mean1) / pooled_sd;
}

// Glass's delta (uses control group SD)
Real glass_delta(
    Array<const Real> control,
    Array<const Real> treatment
) {
    const Size n_control = control.len;
    const Size n_treatment = treatment.len;

    if (n_control < 2 || n_treatment < 1) return Real(0.0);

    // Compute mean and variance using SIMD
    Real mean_c, var_c, mean_t, var_t;
    detail::compute_mean_and_variance(control.ptr, n_control, mean_c, var_c);
    detail::compute_mean_and_variance(treatment.ptr, n_treatment, mean_t, var_t);

    Real sd_c = std::sqrt(var_c);
    if (sd_c < config::EPSILON) return Real(0.0);

    return (mean_t - mean_c) / sd_c;
}

// Hedges' g (bias-corrected effect size)
Real hedges_g(
    Array<const Real> group1,
    Array<const Real> group2
) {
    Real d = effect_size(group1, group2);

    Size n1 = group1.len;
    Size n2 = group2.len;
    Size n = n1 + n2;

    if (n < 4) return d;

    // Correction factor
    Real correction = Real(1.0) - Real(3.0) / (Real(4.0) * static_cast<Real>(n) - Real(9.0));

    return d * correction;
}

} // namespace scl::kernel::comparison
