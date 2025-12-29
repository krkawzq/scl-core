#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/scoring.hpp
// BRIEF: High-performance gene set scoring and cell signature analysis
//
// Optimizations applied:
// - Parallel cell scoring with WorkspacePool
// - SIMD-accelerated accumulation
// - Optimized ranking with counting sort for small ranges
// - Atomic accumulation for gene statistics
// - Bitset gene set lookup
// - Fused z-score computation
// - Cache-aligned data structures
// =============================================================================

namespace scl::kernel::scoring {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_CONTROL = 100;
    constexpr Index DEFAULT_N_BINS = 25;
    constexpr Real DEFAULT_QUANTILE = Real(0.05);
    constexpr Real MIN_VAR = Real(1e-9);
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// Scoring Methods
// =============================================================================

enum class ScoringMethod {
    Mean,
    RankBased,
    Weighted,
    SeuratModule,
    ZScore
};

// =============================================================================
// Cell Cycle Phases
// =============================================================================

enum class CellCyclePhase : Index {
    G1 = 0,
    S = 1,
    G2M = 2
};

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// Fast PRNG (Xoshiro128+)
// =============================================================================

struct alignas(16) FastRNG {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::array<uint32_t, 4> s{};

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (uint32_t& si : s) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            si = static_cast<uint32_t>(z >> 32);
        }
    }

    SCL_FORCE_INLINE uint32_t next() noexcept {
        uint32_t t = s[3];
        uint32_t const x = s[0];
        s[3] = s[2];
        s[2] = s[1];
        s[1] = x;
        t ^= t >> 11;
        t ^= t << 8;
        s[0] = t ^ x ^ (x << 19);
        return s[0];
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint32_t>(n));
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next()) * Real(2.3283064365386963e-10);
    }
};

// =============================================================================
// SIMD Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE Real sum_simd(const Real* SCL_RESTRICT x, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    Size k = 0;

    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, x + k));
        v_sum1 = s::Add(v_sum1, s::Load(d, x + k + lanes));
    }

    Real result = s::GetLane(s::SumOfLanes(d, s::Add(v_sum0, v_sum1)));

    for (; k < n; ++k) {
        result += x[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE void scale_simd(Real* SCL_RESTRICT x, Real alpha, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);
    Size k = 0;

    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x + k)), d, x + k);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + lanes)), d, x + k + lanes);
    }

    for (; k < n; ++k) {
        x[k] *= alpha;
    }
}

SCL_HOT SCL_FORCE_INLINE void axpy_simd(
    Real alpha,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);
    Size k = 0;

    for (; k + lanes <= n; k += lanes) {
        auto vy = s::Load(d, y + k);
        vy = s::MulAdd(v_alpha, s::Load(d, x + k), vy);
        s::Store(vy, d, y + k);
    }

    for (; k < n; ++k) {
        y[k] += alpha * x[k];
    }
}

// =============================================================================
// Bitset Gene Set Lookup
// =============================================================================

struct GeneSetLookup {
    uint64_t* bits;
    Size n_words;
    Index n_genes;

    void init(Index n_genes_total) {
        n_genes = n_genes_total;
        n_words = (static_cast<Size>(n_genes) + 63) / 64;
        auto bits_ptr = scl::memory::aligned_alloc<uint64_t>(n_words, SCL_ALIGNMENT);
        bits = bits_ptr.get();
        std::memset(bits, 0, n_words * sizeof(uint64_t));
    }

    void destroy() {
        scl::memory::aligned_free(bits, SCL_ALIGNMENT);
    }

    SCL_FORCE_INLINE void set(Index gene) noexcept {
        if (gene >= 0 && gene < n_genes) {
            bits[gene >> 6] |= (1ULL << (gene & 63));
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE bool contains(Index gene) const noexcept {
        if (gene < 0 || gene >= n_genes) return false;
        return (bits[gene >> 6] & (1ULL << (gene & 63))) != 0;
    }

    void clear() noexcept {
        std::memset(bits, 0, n_words * sizeof(uint64_t));
    }
};

// =============================================================================
// Optimized Ranking (Argsort + Rank Assignment)
// =============================================================================

SCL_HOT void compute_ranks_optimized(
    const Real* values,
    Index n,
    Real* ranks,
    Index* sorted_indices  // workspace
) noexcept {
    if (n == 0) return;

    // Initialize indices
    for (Index i = 0; i < n; ++i) {
        sorted_indices[i] = i;
    }

    // Sort indices by values (ascending) - use insertion sort for small n
    if (n <= 64) {
        for (Index i = 1; i < n; ++i) {
            Index idx = sorted_indices[i];
            Real val = values[idx];
            Index j = i;

            while (j > 0 && values[sorted_indices[j - 1]] > val) {
                sorted_indices[j] = sorted_indices[j - 1];
                --j;
            }
            sorted_indices[j] = idx;
        }
    } else {
        // Shell sort for larger n
        constexpr std::array<Index, 8> gaps = {701, 301, 132, 57, 23, 10, 4, 1};

        for (Index gap : gaps) {
            if (gap >= n) continue;

            for (Index i = gap; i < n; ++i) {
                Index idx = sorted_indices[i];
                Real val = values[idx];
                Index j = i;

                while (j >= gap && values[sorted_indices[j - gap]] > val) {
                    sorted_indices[j] = sorted_indices[j - gap];
                    j -= gap;
                }
                sorted_indices[j] = idx;
            }
        }
    }

    // Assign ranks with tie handling
    Index i = 0;
    while (i < n) {
        Index j = i;
        while (j < n - 1 && values[sorted_indices[j + 1]] == values[sorted_indices[i]]) {
            ++j;
        }

        Real avg_rank = static_cast<Real>(i + j + 2) / Real(2);

        for (Index k = i; k <= j; ++k) {
            ranks[sorted_indices[k]] = avg_rank;
        }

        i = j + 1;
    }
}

// =============================================================================
// AUC Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_auc(
    const Real* ranks,
    const Index* gene_set,
    Index n_genes_in_set,
    Index n_total_genes,
    Real quantile
) noexcept {
    if (n_genes_in_set == 0 || n_total_genes == 0) return Real(0);

    auto max_rank = static_cast<Index>(std::ceil(quantile * static_cast<Real>(n_total_genes)));
    max_rank = scl::algo::max2(max_rank, Index(1));

    Index count = 0;
    for (Index i = 0; i < n_genes_in_set; ++i) {
        Index gene = gene_set[i];
        if (ranks[gene] <= static_cast<Real>(max_rank)) {
            ++count;
        }
    }

    return static_cast<Real>(count) / static_cast<Real>(n_genes_in_set);
}

// =============================================================================
// Find Bin (Branchless)
// =============================================================================

SCL_FORCE_INLINE Index find_bin(Real value, const Real* bin_edges, Index n_bins) noexcept {
    Index bin = 0;
    for (Index b = 1; b < n_bins; ++b) {
        bin += (value > bin_edges[b]) ? 1 : 0;
    }
    return bin;
}

} // namespace detail

// =============================================================================
// Compute Gene Means (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void compute_gene_means(
    const Sparse<T, IsCSR>& X,
    Array<Real> gene_means,
    Index n_cells,
    Index n_genes
) {
    const Size G = static_cast<Size>(n_genes);
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(gene_means.len >= G, "Scoring: gene_means buffer too small");

    scl::algo::zero(gene_means.ptr, G);

    if (IsCSR) {
        // Use atomic accumulation for CSR
        constexpr int64_t SCALE = 1000000LL;

        auto atomic_sums_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(G, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_sums = atomic_sums_ptr.get();

        for (Size g = 0; g < G; ++g) {
            atomic_sums[g].store(0, std::memory_order_relaxed);
        }

        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes) {
                    auto scaled = static_cast<int64_t>(static_cast<Real>(values[k]) * SCALE);
                    atomic_sums[gene].fetch_add(scaled, std::memory_order_relaxed);
                }
            }
        });

        Real inv_n = Real(1) / static_cast<Real>(n_cells);

        for (Size g = 0; g < G; ++g) {
            gene_means[static_cast<Index>(g)] = static_cast<Real>(atomic_sums[g].load()) / SCALE * inv_n;
        }

        scl::memory::aligned_free(atomic_sums, SCL_ALIGNMENT);
    } else {
        // Parallel over genes for CSC
        scl::threading::parallel_for(Size(0), G, [&](size_t g) {
            auto values = X.col_values_unsafe(static_cast<Index>(static_cast<Index>(g)));
            const Index len = X.col_length_unsafe(static_cast<Index>(static_cast<Index>(g)));

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            gene_means[static_cast<Index>(g)] = sum / static_cast<Real>(n_cells);
        });
    }
}

// =============================================================================
// Mean Gene Set Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void mean_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    scl::algo::zero(scores.ptr, N);

    if (gene_set.len == 0) return;

    // Build gene set lookup
    detail::GeneSetLookup lookup{};
    lookup.init(n_genes);

    for (Size i = 0; i < gene_set.len; ++i) {
        lookup.set(static_cast<Index>(gene_set[static_cast<Index>(i)]));
    }

    Real inv_n_genes = Real(1) / static_cast<Real>(gene_set.len);

    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (lookup.contains(gene)) {
                    sum += static_cast<Real>(values[k]);
                }
            }

            scores[static_cast<Index>(c)] = sum * inv_n_genes;
        });
    } else {
        // Parallel over genes, atomic accumulation
        constexpr int64_t SCALE = 1000000LL;

        auto atomic_scores_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(N, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_scores = atomic_scores_ptr.get();

        for (Size c = 0; c < N; ++c) {
            atomic_scores[c].store(0, std::memory_order_relaxed);
        }

        scl::threading::parallel_for(Size(0), gene_set.len, [&](size_t i) {
            Index gene = gene_set[static_cast<Index>(i)];
            if (gene < 0 || gene >= n_genes) return;

            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    auto scaled = static_cast<int64_t>(static_cast<Real>(values[k]) * SCALE);
                    atomic_scores[static_cast<Index>(c)].fetch_add(scaled, std::memory_order_relaxed);
                }
            }
        });

        for (Size c = 0; c < N; ++c) {
            scores[static_cast<Index>(c)] = static_cast<Real>(atomic_scores[static_cast<Index>(c)].load()) / SCALE * inv_n_genes;
        }

        scl::memory::aligned_free(atomic_scores, SCL_ALIGNMENT);
    }

    lookup.destroy();
}

// =============================================================================
// Weighted Gene Set Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void weighted_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<const Real> gene_weights,
    Array<Real> scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");
    SCL_CHECK_DIM(gene_weights.len >= gene_set.len, "Scoring: gene_weights too small");

    scl::algo::zero(scores.ptr, N);

    if (gene_set.len == 0) return;

    // Compute total weight
    Real total_weight = detail::sum_simd(gene_weights.ptr, gene_set.len);
    if (total_weight < config::MIN_VAR) return;

    // Create gene -> weight mapping
    auto weight_map_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);

    Real* weight_map = weight_map_ptr.release();
    scl::algo::zero(weight_map, G);

    for (Size i = 0; i < gene_set.len; ++i) {
        auto g = static_cast<Index>(gene_set[static_cast<Index>(i)]);
        if (g >= 0 && g < n_genes) {
            weight_map[static_cast<Index>(g)] = gene_weights[static_cast<Index>(i)];
        }
    }

    Real inv_total = Real(1) / total_weight;

    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real weighted_sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes && weight_map[gene] > Real(0)) {
                    weighted_sum += weight_map[gene] * static_cast<Real>(values[k]);
                }
            }

            scores[static_cast<Index>(c)] = weighted_sum * inv_total;
        });
    } else {
        constexpr int64_t SCALE = 1000000LL;

        auto atomic_scores_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(N, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_scores = atomic_scores_ptr.get();

        for (Size c = 0; c < N; ++c) {
            atomic_scores[c].store(0, std::memory_order_relaxed);
        }

        scl::threading::parallel_for(Size(0), gene_set.len, [&](size_t i) {
            auto gene = static_cast<Index>(gene_set[static_cast<Index>(i)]);
            if (gene < 0 || gene >= n_genes) return;

            Real w = gene_weights[static_cast<Index>(i)];
            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    auto scaled = static_cast<int64_t>(w * static_cast<Real>(values[k]) * SCALE);
                    atomic_scores[static_cast<Index>(c)].fetch_add(scaled, std::memory_order_relaxed);
                }
            }
        });

        for (Size c = 0; c < N; ++c) {
            scores[static_cast<Index>(c)] = static_cast<Real>(atomic_scores[static_cast<Index>(c)].load()) / SCALE * inv_total;
        }

        scl::memory::aligned_free(atomic_scores, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(weight_map, SCL_ALIGNMENT);
}

// =============================================================================
// AUC Score (Parallel with WorkspacePool)
// =============================================================================

template <typename T, bool IsCSR>
void auc_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Index n_cells,
    Index n_genes,
    Real quantile = config::DEFAULT_QUANTILE
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    if (gene_set.len == 0 || n_genes == 0) {
        scl::algo::zero(scores.ptr, N);
        return;
    }

    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::WorkspacePool<Real> expr_pool;
    scl::threading::WorkspacePool<Real> ranks_pool;
    scl::threading::WorkspacePool<Index> indices_pool;

    expr_pool.init(n_threads, G);
    ranks_pool.init(n_threads, G);
    indices_pool.init(n_threads, G);

    scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
        Real* expr_values = expr_pool.get(thread_rank);
        Real* ranks = ranks_pool.get(thread_rank);
        Index* sorted_indices = indices_pool.get(thread_rank);

        scl::algo::zero(expr_values, G);

        if (IsCSR) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(static_cast<Index>(c)));
            auto values = X.row_values_unsafe(static_cast<Index>(static_cast<Index>(c)));
            const Index len = X.row_length_unsafe(static_cast<Index>(static_cast<Index>(c)));

            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes) {
                    expr_values[gene] = -static_cast<Real>(values[k]);  // Negate for descending
                }
            }
        } else {
            for (Index g = 0; g < n_genes; ++g) {
                auto indices = X.col_indices_unsafe(g);
                auto values = X.col_values_unsafe(g);
                const Index len = X.col_length_unsafe(g);

                for (Index k = 0; k < len; ++k) {
                    if (indices[k] == static_cast<Index>(c)) {
                        expr_values[g] = -static_cast<Real>(values[k]);
                        break;
                    }
                }
            }
        }

        detail::compute_ranks_optimized(expr_values, n_genes, ranks, sorted_indices);

        scores[static_cast<Index>(c)] = detail::compute_auc(
            ranks, gene_set.ptr, static_cast<Index>(gene_set.len), n_genes, quantile
        );
    });
}

// =============================================================================
// Bin Genes by Expression (Parallel)
// =============================================================================

inline void bin_genes_by_expression(
    Array<const Real> gene_means,
    Index n_genes,
    Index n_bins,
    Array<Index> gene_bins
) {
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(gene_bins.len >= G, "Scoring: gene_bins buffer too small");

    if (n_genes == 0 || n_bins == 0) return;

    // Find min and max
    Real min_expr = gene_means[0];
    Real max_expr = gene_means[0];

    for (Index g = 1; g < n_genes; ++g) {
        min_expr = scl::algo::min2(min_expr, gene_means[g]);
        max_expr = scl::algo::max2(max_expr, gene_means[g]);
    }

    Real range = max_expr - min_expr;
    if (range < config::MIN_VAR) {
        scl::algo::zero(gene_bins.ptr, G);
        return;
    }

    // Compute bin edges
    auto bin_edges_ptr = scl::memory::aligned_alloc<Real>(n_bins + 1, SCL_ALIGNMENT);

    Real* bin_edges = bin_edges_ptr.release();
    for (Index b = 0; b <= n_bins; ++b) {
        bin_edges[b] = min_expr + range * static_cast<Real>(b) / static_cast<Real>(n_bins);
    }

    // Parallel bin assignment
    scl::threading::parallel_for(Size(0), G, [&](size_t g) {
        gene_bins[static_cast<Index>(g)] = detail::find_bin(gene_means[static_cast<Index>(g)], bin_edges, n_bins);
    });

    scl::memory::aligned_free(bin_edges, SCL_ALIGNMENT);
}

// =============================================================================
// Select Control Genes (Optimized)
// =============================================================================

inline Index select_control_genes(
    Array<const Real> /* gene_means */,
    Index n_genes,
    Array<const Index> gene_set,
    Array<const Index> gene_bins,
    Index n_bins,
    Index n_control_per_gene,
    Array<Index> control_genes,
    uint64_t seed = 42
) {
    // Build gene set lookup
    detail::GeneSetLookup in_set{};
    in_set.init(n_genes);

    for (Size i = 0; i < gene_set.len; ++i) {
        in_set.set(static_cast<Index>(gene_set[static_cast<Index>(i)]));
    }

    // Count genes per bin
    auto bin_counts_ptr = scl::memory::aligned_alloc<Index>(n_bins, SCL_ALIGNMENT);

    Index* bin_counts = bin_counts_ptr.release();
    scl::algo::zero(bin_counts, static_cast<Size>(n_bins));

    for (Index g = 0; g < n_genes; ++g) {
        if (!in_set.contains(g)) {
            Index bin = gene_bins[g];
            if (bin >= 0 && bin < n_bins) {
                ++bin_counts[bin];
            }
        }
    }

    // Build per-bin gene lists
    auto bin_genes_ptr = scl::memory::aligned_alloc<Index*>(n_bins, SCL_ALIGNMENT);
    Index** bin_genes = bin_genes_ptr.get();
    auto bin_fill_ptr = scl::memory::aligned_alloc<Index>(n_bins, SCL_ALIGNMENT);

    Index* bin_fill = bin_fill_ptr.release();
    scl::algo::zero(bin_fill, static_cast<Size>(n_bins));

    for (Index b = 0; b < n_bins; ++b) {
        if (bin_counts[b] > 0) {
            auto bin_array_ptr = scl::memory::aligned_alloc<Index>(bin_counts[b], SCL_ALIGNMENT);
            bin_genes[b] = bin_array_ptr.get();
        } else {
            bin_genes[b] = nullptr;
        }
    }

    for (Index g = 0; g < n_genes; ++g) {
        if (!in_set.contains(g)) {
            Index bin = gene_bins[g];
            if (bin >= 0 && bin < n_bins && bin_genes[bin]) {
                bin_genes[bin][bin_fill[bin]++] = g;
            }
        }
    }

    // Sample control genes
    detail::FastRNG rng(seed);
    Index total_control = 0;
    auto max_control = static_cast<Index>(control_genes.len);

    for (Size i = 0; i < gene_set.len && total_control < max_control; ++i) {
        auto gene = static_cast<Index>(gene_set[static_cast<Index>(i)]);
        if (gene < 0 || gene >= n_genes) continue;

        Index bin = gene_bins[static_cast<Index>(gene)];
        if (bin < 0 || bin >= n_bins || !bin_genes[bin]) continue;

        Index bin_size = bin_counts[bin];
        if (bin_size == 0) continue;

        for (Index j = 0; j < n_control_per_gene && total_control < max_control; ++j) {
            auto idx = static_cast<Index>(rng.bounded(static_cast<Size>(bin_size)));
            control_genes[total_control++] = bin_genes[bin][idx];
        }
    }

    // Cleanup
    for (Index b = 0; b < n_bins; ++b) {
        if (bin_genes[b]) {
            scl::memory::aligned_free(bin_genes[b], SCL_ALIGNMENT);
        }
    }

    scl::memory::aligned_free(bin_fill, SCL_ALIGNMENT);
    scl::memory::aligned_free(bin_genes, SCL_ALIGNMENT);
    scl::memory::aligned_free(bin_counts, SCL_ALIGNMENT);
    in_set.destroy();

    return total_control;
}

// =============================================================================
// Seurat Module Score (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void module_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Index n_cells,
    Index n_genes,
    Index n_control_per_gene = 1,
    Index n_bins = config::DEFAULT_N_BINS,
    uint64_t seed = 42
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    if (gene_set.len == 0) {
        scl::algo::zero(scores.ptr, N);
        return;
    }

    // Compute gene means
    auto gene_means_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);

    Real* gene_means = gene_means_ptr.release();
    compute_gene_means(X, Array<Real>(gene_means, G), n_cells, n_genes);

    // Bin genes
    auto gene_bins_ptr = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);

    Index* gene_bins = gene_bins_ptr.release();
    bin_genes_by_expression(
        Array<const Real>(gene_means, G), n_genes, n_bins,
        Array<Index>(gene_bins, G)
    );

    // Select control genes
    Index max_control = static_cast<Index>(gene_set.len) * n_control_per_gene;
    auto control_genes_ptr = scl::memory::aligned_alloc<Index>(max_control, SCL_ALIGNMENT);

    Index* control_genes = control_genes_ptr.release();

    Index n_control = select_control_genes(
        Array<const Real>(gene_means, G), n_genes,
        gene_set, Array<const Index>(gene_bins, G), n_bins,
        n_control_per_gene, Array<Index>(control_genes, max_control), seed
    );

    // Compute scores
    auto set_scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    Real* set_scores = set_scores_ptr.release();
    auto ctrl_scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    Real* ctrl_scores = ctrl_scores_ptr.release();

    mean_score(X, gene_set, Array<Real>(set_scores, N), n_cells, n_genes);
    mean_score(X, Array<const Index>(control_genes, n_control),
               Array<Real>(ctrl_scores, N), n_cells, n_genes);

    // Final score (parallel)
    scl::threading::parallel_for(Size(0), N, [&](size_t c) {
        scores[static_cast<Index>(c)] = set_scores[static_cast<Index>(c)] - ctrl_scores[static_cast<Index>(c)];
    });

    scl::memory::aligned_free(ctrl_scores, SCL_ALIGNMENT);
    scl::memory::aligned_free(set_scores, SCL_ALIGNMENT);
    scl::memory::aligned_free(control_genes, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_bins, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_means, SCL_ALIGNMENT);
}

// =============================================================================
// Z-Score Gene Set Score (Fused, Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void zscore_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    if (gene_set.len == 0) {
        scl::algo::zero(scores.ptr, N);
        return;
    }

    // Compute gene statistics
    auto gene_means_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);

    Real* gene_means = gene_means_ptr.release();
    auto gene_inv_std_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);

    Real* gene_inv_std = gene_inv_std_ptr.release();

    compute_gene_means(X, Array<Real>(gene_means, G), n_cells, n_genes);

    // Compute variances (parallel)
    auto gene_vars_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);

    Real* gene_vars = gene_vars_ptr.release();
    scl::algo::zero(gene_vars, G);

    if (IsCSR) {
        constexpr int64_t SCALE = 1000000LL;

        auto atomic_vars_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(G, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_vars = atomic_vars_ptr.get();

        for (Size g = 0; g < G; ++g) {
            atomic_vars[g].store(0, std::memory_order_relaxed);
        }

        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(static_cast<Index>(c)));
            auto values = X.row_values_unsafe(static_cast<Index>(static_cast<Index>(c)));
            const Index len = X.row_length_unsafe(static_cast<Index>(static_cast<Index>(c)));

            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes) {
                    Real d = static_cast<Real>(values[k]) - gene_means[static_cast<Index>(gene)];
                    auto scaled = static_cast<int64_t>(d * d * SCALE);
                    atomic_vars[static_cast<Index>(gene)].fetch_add(scaled, std::memory_order_relaxed);
                }
            }
        });

        for (Size g = 0; g < G; ++g) {
            gene_vars[g] = static_cast<Real>(atomic_vars[g].load()) / SCALE;
        }

        scl::memory::aligned_free(atomic_vars, SCL_ALIGNMENT);
    } else {
        scl::threading::parallel_for(Size(0), G, [&](size_t g) {
            auto values = X.col_values_unsafe(static_cast<Index>(g));
            const Index len = X.col_length_unsafe(static_cast<Index>(g));

            Real mean_g = gene_means[g];
            Real var = Real(0);

            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]) - mean_g;
                var += d * d;
            }

            // Add zero contribution
            var += static_cast<Real>(n_cells - len) * mean_g * mean_g;
            gene_vars[g] = var;
        });
    }

    // Compute inverse std
    scl::threading::parallel_for(Size(0), G, [&](size_t g) {
        Real var = gene_vars[g] / static_cast<Real>(n_cells);
        gene_inv_std[g] = (var > config::MIN_VAR) ? Real(1) / std::sqrt(var) : Real(0);
    });

    scl::memory::aligned_free(gene_vars, SCL_ALIGNMENT);

    // Build gene set data for fast access
    auto n_set = static_cast<Index>(gene_set.len);
    auto set_means_ptr = scl::memory::aligned_alloc<Real>(n_set, SCL_ALIGNMENT);

    Real* set_means = set_means_ptr.release();
    auto set_inv_std_ptr = scl::memory::aligned_alloc<Real>(n_set, SCL_ALIGNMENT);

    Real* set_inv_std = set_inv_std_ptr.release();

    for (Index i = 0; i < n_set; ++i) {
        Index g = gene_set[i];
        if (g >= 0 && g < n_genes) {
            set_means[i] = gene_means[g];
            set_inv_std[i] = gene_inv_std[g];
        } else {
            set_means[i] = Real(0);
            set_inv_std[i] = Real(0);
        }
    }

    Real inv_n_set = Real(1) / static_cast<Real>(n_set);

    // Compute z-scores per cell (parallel)
    if (IsCSR) {
        // Build gene -> set index map
        auto gene_to_set_ptr = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);

        Index* gene_to_set = gene_to_set_ptr.release();

        for (Size g = 0; g < G; ++g) {
            gene_to_set[g] = -1;
        }

        for (Index i = 0; i < n_set; ++i) {
            Index g = gene_set[i];
            if (g >= 0 && g < n_genes) {
                gene_to_set[g] = i;
            }
        }

        // Precompute z-score for zeros
        auto z_zero_ptr = scl::memory::aligned_alloc<Real>(n_set, SCL_ALIGNMENT);

        Real* z_zero = z_zero_ptr.release();
        for (Index i = 0; i < n_set; ++i) {
            z_zero[i] = -set_means[i] * set_inv_std[i];
        }

        const Size n_threads = scl::threading::Scheduler::get_num_threads();
        scl::threading::WorkspacePool<Real> zscore_pool;
        zscore_pool.init(n_threads, static_cast<Size>(n_set));

        scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
            Real* cell_zscores = zscore_pool.get(thread_rank);

            // Initialize with z-scores for zero values
            std::memcpy(cell_zscores, z_zero, n_set * sizeof(Real));

            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            // Update for non-zero values
            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes) {
                    Index set_idx = gene_to_set[gene];
                    if (set_idx >= 0) {
                        Real val = static_cast<Real>(values[k]);
                        cell_zscores[set_idx] = (val - set_means[set_idx]) * set_inv_std[set_idx];
                    }
                }
            }

            // Average
            Real sum = detail::sum_simd(cell_zscores, static_cast<Size>(n_set));
            scores[static_cast<Index>(c)] = sum * inv_n_set;
        });

        scl::memory::aligned_free(z_zero, SCL_ALIGNMENT);
        scl::memory::aligned_free(gene_to_set, SCL_ALIGNMENT);
    } else {
        // CSC: accumulate per gene
        constexpr int64_t SCALE = 1000000LL;

        auto atomic_scores_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(N, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_scores = atomic_scores_ptr.get();

        // Initialize with z-score sum for all zeros
        Real z_zero_sum = Real(0);
        for (Index i = 0; i < n_set; ++i) {
            z_zero_sum += -set_means[i] * set_inv_std[i];
        }

        for (Size c = 0; c < N; ++c) {
            atomic_scores[c].store(static_cast<int64_t>(z_zero_sum * SCALE), std::memory_order_relaxed);
        }

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_set), [&](size_t i) {
            Index gene = gene_set[static_cast<Index>(i)];
            if (gene < 0 || gene >= n_genes) return;

            Real mean_g = set_means[static_cast<Index>(i)];
            Real inv_std = set_inv_std[static_cast<Index>(i)];
            Real z_zero = -mean_g * inv_std;

            auto indices = X.col_indices_unsafe(gene);
            auto values = X.col_values_unsafe(gene);
            const Index len = X.col_length_unsafe(gene);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    Real val = static_cast<Real>(values[k]);
                    Real z = (val - mean_g) * inv_std;
                    Real delta = z - z_zero;  // Correction from zero assumption
                    auto scaled_delta = static_cast<int64_t>(delta * SCALE);
                    atomic_scores[static_cast<Index>(c)].fetch_add(scaled_delta, std::memory_order_relaxed);
                }
            }
        });

        for (Size c = 0; c < N; ++c) {
            scores[static_cast<Index>(c)] = static_cast<Real>(atomic_scores[static_cast<Index>(c)].load()) / SCALE * inv_n_set;
        }

        scl::memory::aligned_free(atomic_scores, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(set_inv_std, SCL_ALIGNMENT);
    scl::memory::aligned_free(set_means, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_inv_std, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_means, SCL_ALIGNMENT);
}

// =============================================================================
// Generic Gene Set Score
// =============================================================================

template <typename T, bool IsCSR>
void gene_set_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    ScoringMethod method,
    Array<Real> scores,
    Index n_cells,
    Index n_genes,
    Real quantile = config::DEFAULT_QUANTILE
) {
    switch (method) {
        case ScoringMethod::Mean:
            mean_score(X, gene_set, scores, n_cells, n_genes);
            break;
        case ScoringMethod::RankBased:
            auc_score(X, gene_set, scores, n_cells, n_genes, quantile);
            break;
        case ScoringMethod::SeuratModule:
            module_score(X, gene_set, scores, n_cells, n_genes);
            break;
        case ScoringMethod::ZScore:
            zscore_score(X, gene_set, scores, n_cells, n_genes);
            break;
        default:
            mean_score(X, gene_set, scores, n_cells, n_genes);
            break;
    }
}

// =============================================================================
// Differential Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void differential_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> positive_genes,
    Array<const Index> negative_genes,
    Array<Real> scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    auto pos_scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);


    Real* pos_scores = pos_scores_ptr.release();
    auto neg_scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    Real* neg_scores = neg_scores_ptr.release();

    mean_score(X, positive_genes, Array<Real>(pos_scores, N), n_cells, n_genes);
    mean_score(X, negative_genes, Array<Real>(neg_scores, N), n_cells, n_genes);

    scl::threading::parallel_for(Size(0), N, [&](size_t c) {
        scores[static_cast<Index>(c)] = pos_scores[static_cast<Index>(c)] - neg_scores[static_cast<Index>(c)];
    });

    scl::memory::aligned_free(neg_scores, SCL_ALIGNMENT);
    scl::memory::aligned_free(pos_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Cell Cycle Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void cell_cycle_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> s_genes,
    Array<const Index> g2m_genes,
    Array<Real> s_scores,
    Array<Real> g2m_scores,
    Array<Index> phase_labels,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(s_scores.len >= N, "Scoring: s_scores buffer too small");
    SCL_CHECK_DIM(g2m_scores.len >= N, "Scoring: g2m_scores buffer too small");
    SCL_CHECK_DIM(phase_labels.len >= N, "Scoring: phase_labels buffer too small");

    module_score(X, s_genes, s_scores, n_cells, n_genes);
    module_score(X, g2m_genes, g2m_scores, n_cells, n_genes);

    scl::threading::parallel_for(Size(0), N, [&](size_t c) {
        const auto idx = static_cast<Index>(c);
        if (s_scores[idx] > Real(0) && s_scores[idx] > g2m_scores[idx]) {
            phase_labels[idx] = static_cast<Index>(CellCyclePhase::S);
        } else if (g2m_scores[idx] > Real(0) && g2m_scores[idx] > s_scores[idx]) {
            phase_labels[idx] = static_cast<Index>(CellCyclePhase::G2M);
        } else {
            phase_labels[idx] = static_cast<Index>(CellCyclePhase::G1);
        }
    });
}

// =============================================================================
// Quantile Score (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void quantile_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Real quantile,
    Array<Real> scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(scores.len >= N, "Scoring: scores buffer too small");

    if (gene_set.len == 0) {
        scl::algo::zero(scores.ptr, N);
        return;
    }

    auto n_set = static_cast<Index>(gene_set.len);

    // Build lookup
    detail::GeneSetLookup lookup{};
    lookup.init(n_genes);

    for (Size i = 0; i < gene_set.len; ++i) {
        lookup.set(static_cast<Index>(gene_set[static_cast<Index>(i)]));
    }

    // Gene to set index
    auto gene_to_set_ptr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);

    Index* gene_to_set = gene_to_set_ptr.release();

    for (Index g = 0; g < n_genes; ++g) {
        gene_to_set[g] = -1;
    }

    for (Index i = 0; i < n_set; ++i) {
        Index g = gene_set[i];
        if (g >= 0 && g < n_genes) {
            gene_to_set[g] = i;
        }
    }

    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<Real> values_pool;
    values_pool.init(n_threads, static_cast<Size>(n_set));

    scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
        Real* values = values_pool.get(thread_rank);
        scl::algo::zero(values, static_cast<Size>(n_set));

        if (IsCSR) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto vals = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            for (Index k = 0; k < len; ++k) {
                Index gene = indices[k];
                if (gene < n_genes) {
                    Index set_idx = gene_to_set[gene];
                    if (set_idx >= 0) {
                        values[set_idx] = static_cast<Real>(vals[k]);
                    }
                }
            }
        }

        // Sort (insertion sort for small n_set)
        for (Index i = 1; i < n_set; ++i) {
            Real v = values[i];
            Index j = i;
            while (j > 0 && values[j - 1] > v) {
                values[j] = values[j - 1];
                --j;
            }
            values[j] = v;
        }

        auto q_idx = static_cast<Index>(quantile * static_cast<Real>(n_set - 1));
        q_idx = scl::algo::min2(q_idx, n_set - 1);
        scores[static_cast<Index>(c)] = values[q_idx];
    });

    scl::memory::aligned_free(gene_to_set, SCL_ALIGNMENT);
    lookup.destroy();
}

// =============================================================================
// Multi-Signature Score (Parallel over signatures)
// =============================================================================

template <typename T, bool IsCSR>
void multi_signature_score(
    const Sparse<T, IsCSR>& X,
    const Index* const* gene_sets,
    const Index* set_sizes,
    Index n_sets,
    ScoringMethod method,
    Array<Real> all_scores,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size S = static_cast<Size>(n_sets);

    SCL_CHECK_DIM(all_scores.len >= N * S, "Scoring: all_scores buffer too small");

    // Parallel over signatures
    scl::threading::parallel_for(Size(0), S, [&](size_t s) {
        auto scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

        Real* scores = scores_ptr.release();

        gene_set_score(
            X,
            Array<const Index>(gene_sets[s], set_sizes[s]),
            method,
            Array<Real>(scores, N),
            n_cells,
            n_genes
        );

        // Copy to output (column s)
        for (Size c = 0; c < N; ++c) {
            all_scores[static_cast<Index>(c * S + s)] = scores[static_cast<Index>(c)];
        }

        scl::memory::aligned_free(scores, SCL_ALIGNMENT);
    });
}

} // namespace scl::kernel::scoring
