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

// =============================================================================
// FILE: scl/kernel/state.hpp
// BRIEF: Cell state scoring (stemness, differentiation, proliferation)
//
// APPLICATIONS:
// - Stemness quantification
// - Differentiation potential (CytoTRACE-style)
// - Proliferation/stress scoring
// - Cell cycle phase scoring
// - Metabolic and apoptosis states
// =============================================================================

namespace scl::kernel::state {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_GENES_FOR_SCORE = 3;
    constexpr Real PSEUDOCOUNT = Real(1.0);
    constexpr Size PARALLEL_THRESHOLD = 64;
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

// Compute gene set score for a single cell using mean expression
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_geneset_score(
    const Sparse<T, IsCSR>& expression,
    Index cell,
    const Index* gene_indices,
    Size n_genes
) {
    if (n_genes == 0) return Real(0.0);

    Real sum = Real(0.0);
    Size count = 0;

    const Index row_start = expression.row_indices_unsafe()[cell];
    const Index row_end = expression.row_indices_unsafe()[cell + 1];
    const Index row_len = row_end - row_start;

    // For each gene in the gene set, use binary search
    for (Size g = 0; g < n_genes; ++g) {
        Index gene = gene_indices[g];

        // Binary search for gene in sorted row indices
        if (row_len > 0) {
            Index pos = binary_search_gene(
                expression.col_indices_unsafe(), row_start, row_end, gene);
            if (pos >= 0) {
                sum += static_cast<Real>(expression.values()[pos]);
                ++count;
            }
        }
    }

    // Return mean expression of found genes
    if (count > 0) {
        return sum / static_cast<Real>(count);
    }
    return Real(0.0);
}

// Count number of expressed genes per cell
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Size count_expressed_genes(
    const Sparse<T, IsCSR>& expression,
    Index cell,
    Real threshold
) {
    const Index row_start = expression.row_indices_unsafe()[cell];
    const Index row_end = expression.row_indices_unsafe()[cell + 1];

    Size count = 0;
    for (Index j = row_start; j < row_end; ++j) {
        if (static_cast<Real>(expression.values()[j]) > threshold) {
            ++count;
        }
    }
    return count;
}

// Compute Shannon entropy of expression distribution
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_expression_entropy(
    const Sparse<T, IsCSR>& expression,
    Index cell
) {
    const Index row_start = expression.row_indices_unsafe()[cell];
    const Index row_end = expression.row_indices_unsafe()[cell + 1];

    // Sum total expression
    Real total = Real(0.0);
    for (Index j = row_start; j < row_end; ++j) {
        Real val = static_cast<Real>(expression.values()[j]);
        if (val > Real(0.0)) {
            total += val;
        }
    }

    if (total < config::EPSILON) {
        return Real(0.0);
    }

    // Compute entropy
    Real entropy = Real(0.0);
    for (Index j = row_start; j < row_end; ++j) {
        Real val = static_cast<Real>(expression.values()[j]);
        if (val > Real(0.0)) {
            Real p = val / total;
            entropy -= p * std::log(p + config::EPSILON);
        }
    }

    return entropy;
}

// Z-score normalization helper with SIMD
SCL_HOT void zscore_normalize(Real* values, Size n) {
    if (n < 2) return;

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Compute mean using SIMD
    auto v_sum = s::Zero(d);
    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, values + k));
    }
    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    for (; k < n; ++k) {
        sum += values[k];
    }
    Real mean = sum / static_cast<Real>(n);

    // Compute variance using SIMD
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
    Real std_dev = std::sqrt(var_sum / static_cast<Real>(n - 1));

    if (std_dev < config::EPSILON) {
        scl::algo::zero(values, n);
        return;
    }

    // Normalize using SIMD
    Real inv_std = Real(1.0) / std_dev;
    auto v_inv_std = s::Set(d, inv_std);
    k = 0;
    for (; k + lanes <= n; k += lanes) {
        auto v = s::Load(d, values + k);
        auto diff = s::Sub(v, v_mean);
        s::Store(s::Mul(diff, v_inv_std), d, values + k);
    }
    for (; k < n; ++k) {
        values[k] = (values[k] - mean) * inv_std;
    }
}

// Rank transform helper using VQSort
SCL_HOT void rank_transform(Real* values, Size n) {
    if (n == 0) return;

    Real* sorted_values = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* indices = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* ranks = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Size i = 0; i < n; ++i) {
        sorted_values[i] = values[i];
        indices[i] = static_cast<Index>(i);
    }

    // Use VQSort for SIMD-optimized sorting
    scl::sort::sort_pairs(
        Array<Real>(sorted_values, n),
        Array<Index>(indices, n)
    );

    // Handle ties with average ranks
    Size i = 0;
    while (i < n) {
        Size j = i;
        while (j < n && sorted_values[j] == sorted_values[i]) {
            ++j;
        }
        Real avg_rank = (static_cast<Real>(i + j - 1) / Real(2.0)) + Real(1.0);
        for (Size k = i; k < j; ++k) {
            ranks[indices[k]] = avg_rank;
        }
        i = j;
    }

    scl::algo::copy(ranks, values, n);

    scl::memory::aligned_free(indices);
    scl::memory::aligned_free(ranks);
    scl::memory::aligned_free(sorted_values);
}

} // namespace detail

// =============================================================================
// Stemness Score
// =============================================================================

template <typename T, bool IsCSR>
void stemness_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> stemness_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Compute raw stemness scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            stemness_genes.ptr, stemness_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(scores.ptr, n_cells);
}

// =============================================================================
// Differentiation Potential (CytoTRACE-style)
// =============================================================================

template <typename T, bool IsCSR>
void differentiation_potential(
    const Sparse<T, IsCSR>& expression,
    Array<Real> potency_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == potency_scores.len, "Scores array must match cell count");

    if (n_cells == 0 || n_genes == 0) return;

    // Step 1: Count genes expressed per cell (parallel)
    Real* gene_counts = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        gene_counts[i] = static_cast<Real>(
            detail::count_expressed_genes(expression, static_cast<Index>(i), Real(0.0)));
    });

    // Step 2: Compute gene-gene correlation with gene counts
    Real* gene_correlations = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* gene_means = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    scl::algo::zero(gene_means, n_genes);

    // Compute gene sums using atomic accumulation
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    constexpr int64_t SCALE = 1000000LL;

    std::atomic<int64_t>* atomic_sums = static_cast<std::atomic<int64_t>*>(
        scl::memory::aligned_alloc<std::atomic<int64_t>>(n_genes, SCL_ALIGNMENT));
    for (Size g = 0; g < n_genes; ++g) {
        atomic_sums[g].store(0, std::memory_order_relaxed);
    }

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index gene = expression.col_indices_unsafe()[j];
            int64_t scaled = static_cast<int64_t>(static_cast<Real>(expression.values()[j]) * SCALE);
            atomic_sums[gene].fetch_add(scaled, std::memory_order_relaxed);
        }
    });

    for (Size g = 0; g < n_genes; ++g) {
        gene_means[g] = static_cast<Real>(atomic_sums[g].load()) / SCALE / static_cast<Real>(n_cells);
    }
    scl::memory::aligned_free(atomic_sums);

    // Compute gene_counts statistics
    Real gc_mean = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        gc_mean += gene_counts[i];
    }
    gc_mean /= static_cast<Real>(n_cells);

    Real gc_var = Real(0.0);
    for (Size i = 0; i < n_cells; ++i) {
        Real diff = gene_counts[i] - gc_mean;
        gc_var += diff * diff;
    }
    gc_var /= static_cast<Real>(n_cells);

    // Compute correlation of each gene with gene_counts (parallel)
    scl::threading::parallel_for(Size(0), n_genes, [&](size_t g) {
        Real cov = Real(0.0);
        Real gene_var = Real(0.0);

        for (Size i = 0; i < n_cells; ++i) {
            Real gene_val = Real(0.0);
            const Index row_start = expression.row_indices_unsafe()[i];
            const Index row_end = expression.row_indices_unsafe()[i + 1];

            // Use binary search for gene lookup
            Index pos = detail::binary_search_gene(
                expression.col_indices_unsafe(), row_start, row_end, static_cast<Index>(g));
            if (pos >= 0) {
                gene_val = static_cast<Real>(expression.values()[pos]);
            }

            Real gc_diff = gene_counts[i] - gc_mean;
            Real gene_diff = gene_val - gene_means[g];
            cov += gc_diff * gene_diff;
            gene_var += gene_diff * gene_diff;
        }

        if (gene_var > config::EPSILON && gc_var > config::EPSILON) {
            gene_correlations[g] = cov / std::sqrt(gene_var * gc_var);
        } else {
            gene_correlations[g] = Real(0.0);
        }
    });

    // Step 3: Select top correlated genes (top 200 or 10%)
    Size n_top = scl::algo::min2(Size(200), n_genes / 10);
    n_top = scl::algo::max2(n_top, Size(10));
    n_top = scl::algo::min2(n_top, n_genes);

    Index* top_genes = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
    Real* sorted_corr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    for (Size g = 0; g < n_genes; ++g) {
        top_genes[g] = static_cast<Index>(g);
        sorted_corr[g] = -gene_correlations[g];  // Negate for descending sort
    }

    // Use VQSort for efficient partial sorting via full sort
    scl::sort::sort_pairs(
        Array<Real>(sorted_corr, n_genes),
        Array<Index>(top_genes, n_genes)
    );

    // Step 4: Compute final score as weighted sum of top gene expressions (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real score = Real(0.0);
        Real total_weight = Real(0.0);

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Size t = 0; t < n_top; ++t) {
            Index gene = top_genes[t];
            Real weight = (gene_correlations[gene] >= Real(0)) ?
                          gene_correlations[gene] : -gene_correlations[gene];

            // Use binary search for gene lookup
            Real val = Real(0.0);
            Index pos = detail::binary_search_gene(
                expression.col_indices_unsafe(), row_start, row_end, gene);
            if (pos >= 0) {
                val = static_cast<Real>(expression.values()[pos]);
            }

            score += weight * val;
            total_weight += weight;
        }

        if (total_weight > config::EPSILON) {
            potency_scores.ptr[i] = score / total_weight;
        } else {
            potency_scores.ptr[i] = gene_counts[i];
        }
    });

    // Normalize to [0, 1] range
    Real min_val = potency_scores.ptr[0], max_val = potency_scores.ptr[0];
    for (Size i = 1; i < n_cells; ++i) {
        min_val = scl::algo::min2(min_val, potency_scores.ptr[i]);
        max_val = scl::algo::max2(max_val, potency_scores.ptr[i]);
    }

    Real range = max_val - min_val;
    if (range > config::EPSILON) {
        scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
            potency_scores.ptr[i] = (potency_scores.ptr[i] - min_val) / range;
        });
    }

    scl::memory::aligned_free(gene_counts);
    scl::memory::aligned_free(gene_correlations);
    scl::memory::aligned_free(gene_means);
    scl::memory::aligned_free(sorted_corr);
    scl::memory::aligned_free(top_genes);
}

// =============================================================================
// Proliferation Score
// =============================================================================

template <typename T, bool IsCSR>
void proliferation_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> proliferation_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Compute raw proliferation scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            proliferation_genes.ptr, proliferation_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(scores.ptr, n_cells);
}

// =============================================================================
// Stress Score
// =============================================================================

template <typename T, bool IsCSR>
void stress_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> stress_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Compute raw stress scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            stress_genes.ptr, stress_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(scores.ptr, n_cells);
}

// =============================================================================
// State Entropy (Plasticity)
// =============================================================================

template <typename T, bool IsCSR>
void state_entropy(
    const Sparse<T, IsCSR>& expression,
    Array<Real> entropy_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == entropy_scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Compute expression entropy for each cell (parallel)
    Real max_entropy = std::log(static_cast<Real>(n_genes));

    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real entropy = detail::compute_expression_entropy(
            expression, static_cast<Index>(i));

        // Normalize by maximum possible entropy
        if (max_entropy > config::EPSILON) {
            entropy_scores.ptr[i] = entropy / max_entropy;
        } else {
            entropy_scores.ptr[i] = Real(0.0);
        }
    });
}

// =============================================================================
// Cell Cycle Score (G1/S/G2M)
// =============================================================================

template <typename T, bool IsCSR>
void cell_cycle_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> s_genes,
    Array<const Index> g2m_genes,
    Array<Real> s_scores,
    Array<Real> g2m_scores,
    Array<Index> phase_labels
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == s_scores.len, "S scores array must match cell count");
    SCL_CHECK_DIM(n_cells == g2m_scores.len, "G2M scores array must match cell count");
    SCL_CHECK_DIM(n_cells == phase_labels.len, "Phase labels array must match cell count");

    if (n_cells == 0) return;

    // Compute S phase and G2/M phase scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        s_scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            s_genes.ptr, s_genes.len);
        g2m_scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            g2m_genes.ptr, g2m_genes.len);
    });

    // Z-score normalize both
    detail::zscore_normalize(s_scores.ptr, n_cells);
    detail::zscore_normalize(g2m_scores.ptr, n_cells);

    // Assign phase labels: 0 = G1, 1 = S, 2 = G2M (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real s = s_scores.ptr[i];
        Real g2m = g2m_scores.ptr[i];

        if (s > Real(0.0) && s > g2m) {
            phase_labels.ptr[i] = 1;  // S phase
        } else if (g2m > Real(0.0) && g2m > s) {
            phase_labels.ptr[i] = 2;  // G2/M phase
        } else {
            phase_labels.ptr[i] = 0;  // G1 phase
        }
    });
}

// =============================================================================
// Quiescence Score
// =============================================================================

template <typename T, bool IsCSR>
void quiescence_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> quiescence_genes,
    Array<const Index> proliferation_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    Real* q_raw = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* p_raw = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    // Compute quiescence and proliferation gene expression (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        q_raw[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            quiescence_genes.ptr, quiescence_genes.len);
        p_raw[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            proliferation_genes.ptr, proliferation_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(q_raw, n_cells);
    detail::zscore_normalize(p_raw, n_cells);

    // Quiescence = quiescence_score - proliferation_score (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        scores.ptr[i] = q_raw[i] - p_raw[i];
    });

    scl::memory::aligned_free(q_raw);
    scl::memory::aligned_free(p_raw);
}

// =============================================================================
// Metabolic Activity Score
// =============================================================================

template <typename T, bool IsCSR>
void metabolic_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> glycolysis_genes,
    Array<const Index> oxphos_genes,
    Array<Real> glycolysis_scores,
    Array<Real> oxphos_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == glycolysis_scores.len, "Glycolysis scores length mismatch");
    SCL_CHECK_DIM(n_cells == oxphos_scores.len, "OXPHOS scores length mismatch");

    if (n_cells == 0) return;

    // Compute glycolysis and OXPHOS scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        glycolysis_scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            glycolysis_genes.ptr, glycolysis_genes.len);
        oxphos_scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            oxphos_genes.ptr, oxphos_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(glycolysis_scores.ptr, n_cells);
    detail::zscore_normalize(oxphos_scores.ptr, n_cells);
}

// =============================================================================
// Apoptosis Score
// =============================================================================

template <typename T, bool IsCSR>
void apoptosis_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> apoptosis_genes,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Compute apoptosis scores (parallel)
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        scores.ptr[i] = detail::compute_geneset_score(
            expression, static_cast<Index>(i),
            apoptosis_genes.ptr, apoptosis_genes.len);
    });

    // Z-score normalize
    detail::zscore_normalize(scores.ptr, n_cells);
}

// =============================================================================
// Gene Signature Score (generalized)
// =============================================================================

template <typename T, bool IsCSR>
void signature_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> gene_indices,
    Array<const Real> gene_weights,
    Array<Real> scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_signature = gene_indices.len;
    SCL_CHECK_DIM(n_cells == scores.len, "Scores array must match cell count");
    SCL_CHECK_DIM(n_signature == gene_weights.len, "Weights must match gene indices");

    if (n_cells == 0 || n_signature == 0) return;

    // Parallel signature scoring
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real weighted_sum = Real(0.0);
        Real weight_sum = Real(0.0);

        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        for (Size g = 0; g < n_signature; ++g) {
            Index gene = gene_indices.ptr[g];
            Real weight = gene_weights.ptr[g];

            // Use binary search for gene lookup
            Real val = Real(0.0);
            Index pos = detail::binary_search_gene(
                expression.col_indices_unsafe(), row_start, row_end, gene);
            if (pos >= 0) {
                val = static_cast<Real>(expression.values()[pos]);
            }

            weighted_sum += weight * val;
            weight_sum += (weight >= Real(0)) ? weight : -weight;
        }

        if (weight_sum > config::EPSILON) {
            scores.ptr[i] = weighted_sum / weight_sum;
        } else {
            scores.ptr[i] = Real(0.0);
        }
    });

    // Z-score normalize
    detail::zscore_normalize(scores.ptr, n_cells);
}

// =============================================================================
// Multi-signature Score Matrix
// =============================================================================

template <typename T, bool IsCSR>
void multi_signature_score(
    const Sparse<T, IsCSR>& expression,
    const Index* signature_gene_indices,  // Flat array of gene indices
    const Size* signature_offsets,        // Start offset for each signature
    Size n_signatures,
    Real* score_matrix                    // [n_cells * n_signatures]
) {
    const Size n_cells = static_cast<Size>(expression.rows());

    if (n_cells == 0 || n_signatures == 0) return;

    // Process each signature in parallel
    scl::threading::parallel_for(Size(0), n_signatures, [&](size_t s) {
        const Size start = signature_offsets[s];
        const Size end = signature_offsets[s + 1];
        const Size sig_len = end - start;

        // Compute scores for this signature across all cells
        for (Size i = 0; i < n_cells; ++i) {
            score_matrix[i * n_signatures + s] = detail::compute_geneset_score(
                expression, static_cast<Index>(i),
                signature_gene_indices + start, sig_len);
        }
    });

    // Z-score normalize each signature column (parallel)
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<Real> col_pool;
    col_pool.init(n_threads, n_cells);

    scl::threading::parallel_for(Size(0), n_signatures, [&](size_t s, size_t thread_rank) {
        Real* col_buffer = col_pool.get(thread_rank);

        // Extract column
        for (Size i = 0; i < n_cells; ++i) {
            col_buffer[i] = score_matrix[i * n_signatures + s];
        }

        // Normalize
        detail::zscore_normalize(col_buffer, n_cells);

        // Write back
        for (Size i = 0; i < n_cells; ++i) {
            score_matrix[i * n_signatures + s] = col_buffer[i];
        }
    });
}

// =============================================================================
// Transcriptional Diversity
// =============================================================================

template <typename T, bool IsCSR>
void transcriptional_diversity(
    const Sparse<T, IsCSR>& expression,
    Array<Real> diversity_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == diversity_scores.len, "Scores array must match cell count");

    if (n_cells == 0) return;

    // Parallel diversity computation
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        const Index row_start = expression.row_indices_unsafe()[i];
        const Index row_end = expression.row_indices_unsafe()[i + 1];

        // Simpson's diversity index
        Real total = Real(0.0);
        Real sum_sq = Real(0.0);

        for (Index j = row_start; j < row_end; ++j) {
            Real val = static_cast<Real>(expression.values()[j]);
            if (val > Real(0.0)) {
                total += val;
                sum_sq += val * val;
            }
        }

        if (total > config::EPSILON) {
            // Simpson's index = 1 - sum(p_i^2)
            Real simpson = Real(1.0) - (sum_sq / (total * total));
            diversity_scores.ptr[i] = simpson;
        } else {
            diversity_scores.ptr[i] = Real(0.0);
        }
    });
}

// =============================================================================
// Expression Complexity (number of expressed genes, normalized)
// =============================================================================

template <typename T, bool IsCSR>
void expression_complexity(
    const Sparse<T, IsCSR>& expression,
    Real expression_threshold,
    Array<Real> complexity_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    const Size n_genes = static_cast<Size>(expression.cols());
    SCL_CHECK_DIM(n_cells == complexity_scores.len, "Scores array must match cell count");

    if (n_cells == 0 || n_genes == 0) return;

    // Parallel complexity computation
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Size n_expressed = detail::count_expressed_genes(
            expression, static_cast<Index>(i), expression_threshold);

        // Normalize by total genes
        complexity_scores.ptr[i] = static_cast<Real>(n_expressed) /
                                   static_cast<Real>(n_genes);
    });
}

// =============================================================================
// Combined State Score
// =============================================================================

template <typename T, bool IsCSR>
void combined_state_score(
    const Sparse<T, IsCSR>& expression,
    const Index* const* gene_sets,      // Array of gene set pointers
    const Size* gene_set_sizes,          // Size of each gene set
    const Real* weights,                 // Weight for each gene set
    Size n_gene_sets,
    Array<Real> combined_scores
) {
    const Size n_cells = static_cast<Size>(expression.rows());
    SCL_CHECK_DIM(n_cells == combined_scores.len, "Scores array must match cell count");

    if (n_cells == 0 || n_gene_sets == 0) return;

    // Compute individual scores for each gene set (parallel per gene set)
    Real** individual_scores = scl::memory::aligned_alloc<Real*>(n_gene_sets, SCL_ALIGNMENT);
    for (Size s = 0; s < n_gene_sets; ++s) {
        individual_scores[s] = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    }

    // Compute scores in parallel over gene sets
    scl::threading::parallel_for(Size(0), n_gene_sets, [&](size_t s) {
        // Compute scores for all cells for this gene set
        for (Size i = 0; i < n_cells; ++i) {
            individual_scores[s][i] = detail::compute_geneset_score(
                expression, static_cast<Index>(i),
                gene_sets[s], gene_set_sizes[s]);
        }
        detail::zscore_normalize(individual_scores[s], n_cells);
    });

    // Compute weighted combination
    Real total_weight = Real(0.0);
    for (Size s = 0; s < n_gene_sets; ++s) {
        Real w = weights[s];
        total_weight += (w >= Real(0)) ? w : -w;
    }

    // Parallel weighted combination
    scl::threading::parallel_for(Size(0), n_cells, [&](size_t i) {
        Real sum = Real(0.0);
        for (Size s = 0; s < n_gene_sets; ++s) {
            sum += weights[s] * individual_scores[s][i];
        }
        if (total_weight > config::EPSILON) {
            combined_scores.ptr[i] = sum / total_weight;
        } else {
            combined_scores.ptr[i] = Real(0.0);
        }
    });

    // Cleanup
    for (Size s = 0; s < n_gene_sets; ++s) {
        scl::memory::aligned_free(individual_scores[s]);
    }
    scl::memory::aligned_free(individual_scores);
}

} // namespace scl::kernel::state
