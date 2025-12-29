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

#include <cmath>
#include <cstring>
#include <atomic>
#include <concepts>
#include <memory>
#include <array>

// =============================================================================
// FILE: scl/kernel/coexpression.hpp
// BRIEF: High-performance co-expression module detection (WGCNA-style)
//
// Optimizations applied:
// - Parallel correlation matrix (upper triangular)
// - SIMD-accelerated Pearson correlation
// - Pre-extracted gene expression for cache locality
// - Parallel TOM computation
// - Parallel hierarchical clustering merge search
// - WorkspacePool for thread-local buffers
// - Optimized ranking with Shell sort
// =============================================================================

namespace scl::kernel::coexpression {

// =============================================================================
// C++20 Concepts
// =============================================================================

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    inline constexpr Real DEFAULT_SOFT_POWER = Real(6);
    inline constexpr Real EPSILON = Real(1e-15);
    inline constexpr Index DEFAULT_MIN_MODULE_SIZE = 30;
    inline constexpr Index DEFAULT_DEEP_SPLIT = 2;
    inline constexpr Real DEFAULT_MERGE_CUT_HEIGHT = Real(0.25);
    inline constexpr Index MAX_ITERATIONS = 100;
    inline constexpr Size PARALLEL_THRESHOLD = 64;
    inline constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// Co-expression Types
// =============================================================================

enum class CorrelationType {
    Pearson,
    Spearman,
    Bicor
};

enum class AdjacencyType {
    Unsigned,
    Signed,
    SignedHybrid
};

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD Vector Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE Real dot_simd(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
        v_sum1 = s::MulAdd(s::Load(d, a + k + lanes), s::Load(d, b + k + lanes), v_sum1);
        v_sum2 = s::MulAdd(s::Load(d, a + k + 2*lanes), s::Load(d, b + k + 2*lanes), v_sum2);
        v_sum3 = s::MulAdd(s::Load(d, a + k + 3*lanes), s::Load(d, b + k + 3*lanes), v_sum3);
    }

    v_sum0 = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= n; k += lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
    }

    Real result = s::GetLane(s::SumOfLanes(d, v_sum0));

    for (; k < n; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real sum_simd(const Real* SCL_RESTRICT x, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));

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

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(const Real* x, Size n) noexcept {
    return dot_simd(x, x, n);
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
    const auto lanes = static_cast<Size>(s::Lanes(d));

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

SCL_HOT SCL_FORCE_INLINE void scale_simd(Real* SCL_RESTRICT x, Real alpha, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x + k)), d, x + k);
    }

    for (; k < n; ++k) {
        x[k] *= alpha;
    }
}

// =============================================================================
// Optimized Mean Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_mean(const Real* values, Index n) noexcept {
    if (n == 0) return Real(0);
    return sum_simd(values, static_cast<Size>(n)) / static_cast<Real>(n);
}

// =============================================================================
// Optimized Median (Quickselect)
// =============================================================================

SCL_HOT Real quickselect_median(Real* arr, Index n) noexcept {
    if (n == 0) return Real(0);
    if (n == 1) return arr[0];
    if (n == 2) return (arr[0] + arr[1]) / Real(2);

    // For small arrays, use insertion sort
    if (n <= 32) {
        for (Index i = 1; i < n; ++i) {
            Real val = arr[i];
            Index j = i;
            while (j > 0 && arr[j-1] > val) {
                arr[j] = arr[j-1];
                --j;
            }
            arr[j] = val;
        }

        if (n % 2 == 1) {
            return arr[n / 2];
        } else {
            return (arr[n / 2 - 1] + arr[n / 2]) / Real(2);
        }
    }

    // Quickselect for larger arrays
    Index target = n / 2;
    Index left = 0, right = n - 1;

    while (left < right) {
        // Median of three pivot
        Index mid = (left + right) / 2;
        if (arr[mid] < arr[left]) scl::algo::swap(arr[left], arr[mid]);
        if (arr[right] < arr[left]) scl::algo::swap(arr[left], arr[right]);
        if (arr[right] < arr[mid]) scl::algo::swap(arr[mid], arr[right]);

        Real pivot = arr[mid];
        scl::algo::swap(arr[mid], arr[right - 1]);

        Index i = left, j = right - 1;
        while (true) {
            while (arr[++i] < pivot);
            while (arr[--j] > pivot);
            if (i >= j) break;
            scl::algo::swap(arr[i], arr[j]);
        }
        scl::algo::swap(arr[i], arr[right - 1]);

        if (i >= target) right = i - 1;
        if (i <= target) left = i + 1;
    }

    if (n % 2 == 1) {
        return arr[target];
    } else {
        Real second = arr[target - 1];
        for (Index i = 0; i < target - 1; ++i) {
            second = scl::algo::max2(second, arr[i]);
        }
        return (arr[target] + second) / Real(2);
    }
}

SCL_HOT Real compute_median(const Real* values, Index n) {
    if (n == 0) return Real(0);
    if (n == 1) return values[0];

    auto temp_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* temp = temp_ptr.release();
    std::memcpy(temp, values, static_cast<Size>(n) * sizeof(Real));

    Real median = quickselect_median(temp, n);

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    return median;
}

SCL_HOT Real compute_mad(const Real* values, Index n, Real median) {
    if (n == 0) return Real(0);

    auto abs_dev_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* abs_dev = abs_dev_ptr.release();
    for (Index i = 0; i < n; ++i) {
        abs_dev[i] = std::abs(values[i] - median);
    }

    Real mad = quickselect_median(abs_dev, n);
    scl::memory::aligned_free(abs_dev, SCL_ALIGNMENT);

    return mad * Real(1.4826);
}

// =============================================================================
// SIMD Pearson Correlation
// =============================================================================

SCL_HOT Real pearson_correlation_simd(
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT y,
    Index n,
    Real mean_x,
    Real mean_y
) noexcept {
    if (n < 2) return Real(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));

    auto v_mean_x = s::Set(d, mean_x);
    auto v_mean_y = s::Set(d, mean_y);

    auto v_sum_xy = s::Zero(d);
    auto v_sum_xx = s::Zero(d);
    auto v_sum_yy = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= static_cast<Size>(n); k += lanes) {
        auto vx = s::Sub(s::Load(d, x + k), v_mean_x);
        auto vy = s::Sub(s::Load(d, y + k), v_mean_y);

        v_sum_xy = s::MulAdd(vx, vy, v_sum_xy);
        v_sum_xx = s::MulAdd(vx, vx, v_sum_xx);
        v_sum_yy = s::MulAdd(vy, vy, v_sum_yy);
    }

    Real sum_xy = s::GetLane(s::SumOfLanes(d, v_sum_xy));
    Real sum_xx = s::GetLane(s::SumOfLanes(d, v_sum_xx));
    Real sum_yy = s::GetLane(s::SumOfLanes(d, v_sum_yy));

    for (; k < static_cast<Size>(n); ++k) {
        Real dx = x[k] - mean_x;
        Real dy = y[k] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    Real denom = std::sqrt(sum_xx * sum_yy);
    return (denom > config::EPSILON) ? sum_xy / denom : Real(0);
}

SCL_HOT Real pearson_correlation(const Real* x, const Real* y, Index n) {
    if (n < 2) return Real(0);
    Real mean_x = compute_mean(x, n);
    Real mean_y = compute_mean(y, n);
    return pearson_correlation_simd(x, y, n, mean_x, mean_y);
}

// =============================================================================
// Optimized Spearman (Shell Sort Ranking)
// =============================================================================

SCL_HOT void compute_ranks_shell(
    const Real* values,
    Index n,
    Real* ranks,
    Index* sorted_idx
) noexcept {
    for (Index i = 0; i < n; ++i) {
        sorted_idx[i] = i;
    }

    // Shell sort
    constexpr std::array<Index, 8> gaps = {701, 301, 132, 57, 23, 10, 4, 1};
    for (Index gap : gaps) {
        if (gap >= n) continue;

        for (Index i = gap; i < n; ++i) {
            Index idx = sorted_idx[i];
            Real val = values[idx];
            Index j = i;

            while (j >= gap && values[sorted_idx[j - gap]] > val) {
                sorted_idx[j] = sorted_idx[j - gap];
                j -= gap;
            }
            sorted_idx[j] = idx;
        }
    }

    // Assign ranks with tie handling
    Index i = 0;
    while (i < n) {
        Index j = i;
        while (j < n - 1 && values[sorted_idx[j + 1]] == values[sorted_idx[i]]) {
            ++j;
        }

        Real avg_rank = static_cast<Real>(i + j + 2) / Real(2);
        for (Index k = i; k <= j; ++k) {
            ranks[sorted_idx[k]] = avg_rank;
        }
        i = j + 1;
    }
}

SCL_HOT Real spearman_correlation(
    const Real* x,
    const Real* y,
    Index n,
    Real* rank_x,
    Real* rank_y,
    Index* temp_idx
) {
    if (n < 2) return Real(0);

    compute_ranks_shell(x, n, rank_x, temp_idx);
    compute_ranks_shell(y, n, rank_y, temp_idx);

    return pearson_correlation(rank_x, rank_y, n);
}

// =============================================================================
// Biweight Midcorrelation
// =============================================================================

SCL_HOT Real bicor(const Real* x, const Real* y, Index n) {
    if (n < 3) return pearson_correlation(x, y, n);

    Real med_x = compute_median(x, n);
    Real med_y = compute_median(y, n);
    Real mad_x = compute_mad(x, n, med_x);
    Real mad_y = compute_mad(y, n, med_y);

    if (mad_x < config::EPSILON || mad_y < config::EPSILON) {
        return pearson_correlation(x, y, n);
    }

    Real c = Real(9);
    Real inv_cmad_x = Real(1) / (c * mad_x);
    Real inv_cmad_y = Real(1) / (c * mad_y);

    Real sum_ab = Real(0);
    Real sum_aa = Real(0);
    Real sum_bb = Real(0);

    for (Index i = 0; i < n; ++i) {
        Real u = (x[i] - med_x) * inv_cmad_x;
        Real v = (y[i] - med_y) * inv_cmad_y;

        Real wu = (std::abs(u) < Real(1)) ? (Real(1) - u * u) * (Real(1) - u * u) : Real(0);
        Real wv = (std::abs(v) < Real(1)) ? (Real(1) - v * v) * (Real(1) - v * v) : Real(0);

        Real a = (x[i] - med_x) * wu;
        Real b = (y[i] - med_y) * wv;

        sum_ab += a * b;
        sum_aa += a * a;
        sum_bb += b * b;
    }

    Real denom = std::sqrt(sum_aa * sum_bb);
    return (denom > config::EPSILON) ? sum_ab / denom : Real(0);
}

// =============================================================================
// Extract All Gene Expressions (Optimized)
// =============================================================================

template <Arithmetic T, bool IsCSR>
SCL_HOT void extract_all_gene_expressions(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Real* gene_expr  // n_genes x n_cells, row-major
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    scl::algo::zero(gene_expr, G * N);

    if constexpr (IsCSR) {
        // Parallel over cells
        scl::threading::parallel_for(Size(0), N, [&](Size c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            const Index len = X.row_length_unsafe(static_cast<Index>(c));

            for (Index k = 0; k < len; ++k) {
                Index gene = indices.ptr[k];
                if (gene < n_genes) {
                    gene_expr[static_cast<Size>(gene) * N + c] = static_cast<Real>(values.ptr[k]);
                }
            }
        });
    } else {
        // Parallel over genes
        scl::threading::parallel_for(Size(0), G, [&](Size g) {
            auto indices = X.col_indices_unsafe(static_cast<Index>(g));
            auto values = X.col_values_unsafe(static_cast<Index>(g));
            const Index len = X.col_length_unsafe(static_cast<Index>(g));

            Real* row = gene_expr + g * N;
            for (Index k = 0; k < len; ++k) {
                Index c = indices.ptr[k];
                if (c < n_cells) {
                    row[c] = static_cast<Real>(values.ptr[k]);
                }
            }
        });
    }
}

// =============================================================================
// Compute Correlation
// =============================================================================

SCL_FORCE_INLINE Real compute_correlation(
    const Real* x,
    const Real* y,
    Index n,
    CorrelationType type,
    Real* rank_x = nullptr,
    Real* rank_y = nullptr,
    Index* temp_idx = nullptr
) {
    switch (type) {
        case CorrelationType::Pearson:
            return pearson_correlation(x, y, n);
        case CorrelationType::Spearman:
            return spearman_correlation(x, y, n, rank_x, rank_y, temp_idx);
        case CorrelationType::Bicor:
            return bicor(x, y, n);
        default:
            return pearson_correlation(x, y, n);
    }
}

// =============================================================================
// Correlation to Adjacency
// =============================================================================

SCL_FORCE_INLINE Real correlation_to_adjacency(
    Real corr,
    Real power,
    AdjacencyType type
) noexcept {
    switch (type) {
        case AdjacencyType::Unsigned:
            return std::pow(std::abs(corr), power);
        case AdjacencyType::Signed:
            return std::pow(Real(0.5) + Real(0.5) * corr, power);
        case AdjacencyType::SignedHybrid:
            return std::pow(std::abs(corr), power) * ((corr >= Real(0)) ? Real(1) : Real(-1));
        default:
            return std::pow(std::abs(corr), power);
    }
}

// =============================================================================
// Parallel First PC (Power Iteration)
// =============================================================================

void compute_first_pc_parallel(
    const Real* data,
    Index n_samples,
    Index n_genes,
    Real* eigenvector,
    Index max_iter = 100
) {
    if (n_samples == 0 || n_genes == 0) return;

    const Size N = static_cast<Size>(n_samples);
    const Size G = static_cast<Size>(n_genes);

    Real init_val = Real(1) / std::sqrt(static_cast<Real>(n_genes));
    for (Index g = 0; g < n_genes; ++g) {
        eigenvector[g] = init_val;
    }

    auto scores_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    auto temp_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Real* scores = scores_ptr.release();
    Real* temp = temp_ptr.release();

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    for (Index iter = 0; iter < max_iter; ++iter) {
        // scores = data * eigenvector (parallel)
        scl::threading::parallel_for(Size(0), N, [&](Size s) {
            const Real* row = data + s * G;
            scores[s] = dot_simd(row, eigenvector, G);
        });

        // temp = data^T * scores (parallel reduction)
        scl::algo::zero(temp, G);

        auto partials_ptr = scl::memory::aligned_alloc<Real*>(n_threads, SCL_ALIGNMENT);
        Real** partials = partials_ptr.release();
        for (Size t = 0; t < n_threads; ++t) {
            auto partial_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
            partials[t] = partial_ptr.release();
            scl::algo::zero(partials[t], G);
        }

        scl::threading::parallel_for(Size(0), N, [&](Size s, Size thread_rank) {
            const Real* row = data + s * G;
            axpy_simd(scores[s], row, partials[thread_rank], G);
        });

        for (Size t = 0; t < n_threads; ++t) {
            for (Size g = 0; g < G; ++g) {
                temp[g] += partials[t][g];
            }
            scl::memory::aligned_free(partials[t], SCL_ALIGNMENT);
        }
        scl::memory::aligned_free(partials, SCL_ALIGNMENT);

        // Normalize and check convergence
        Real norm = std::sqrt(norm_squared_simd(temp, G));
        if (norm < config::EPSILON) break;

        Real diff = Real(0);
        Real inv_norm = Real(1) / norm;
        for (Index g = 0; g < n_genes; ++g) {
            Real new_val = temp[g] * inv_norm;
            diff += std::abs(new_val - eigenvector[g]);
            eigenvector[g] = new_val;
        }

        if (diff < Real(1e-6)) break;
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(scores, SCL_ALIGNMENT);
}

} // namespace detail

// =============================================================================
// Correlation Matrix (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void correlation_matrix(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real* corr_matrix,
    CorrelationType corr_type = CorrelationType::Pearson
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size total = G * G;

    scl::algo::zero(corr_matrix, total);

    // Pre-extract all gene expressions
    auto gene_expr_ptr = scl::memory::aligned_alloc<Real>(G * N, SCL_ALIGNMENT);
    Real* gene_expr = gene_expr_ptr.release();
    detail::extract_all_gene_expressions(expression, n_cells, n_genes, gene_expr);

    // Precompute means for Pearson
    Real* means = nullptr;
    // NOLINTNEXTLINE
    std::unique_ptr<Real[], scl::memory::AlignedDeleter<Real>> means_ptr;
    if (corr_type == CorrelationType::Pearson) {
        means_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
        means = means_ptr.release();
        scl::threading::parallel_for(Size(0), G, [&](Size g) {
            means[g] = detail::compute_mean(gene_expr + g * N, n_cells);
        });
    }

    // Set diagonal
    for (Index i = 0; i < n_genes; ++i) {
        corr_matrix[static_cast<Size>(i) * n_genes + i] = Real(1);
    }

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    // For Spearman, need per-thread workspace
    scl::threading::WorkspacePool<Real> rank_x_pool;
    scl::threading::WorkspacePool<Real> rank_y_pool;
    scl::threading::WorkspacePool<Index> idx_pool;

    if (corr_type == CorrelationType::Spearman) {
        rank_x_pool.init(n_threads, N);
        rank_y_pool.init(n_threads, N);
        idx_pool.init(n_threads, N);
    }

    // Parallel over upper triangular pairs - convert linear index to (i, j)
    Size n_pairs = static_cast<Size>(n_genes) * (n_genes - 1) / 2;

    scl::threading::parallel_for(Size(0), n_pairs, [&](Size pair_idx, Size thread_rank) {
        // Convert pair index to (i, j) for upper triangular matrix
        // For pair_idx from 0 to n_pairs-1, we want i < j
        // Formula: pair_idx = i * (2*n - i - 1) / 2 + (j - i - 1)
        // Simplified: solve for i, j from linear index
        Index i = 0, j = 0;
        Size remaining = pair_idx;
        for (i = 0; i < n_genes - 1; ++i) {
            Size pairs_in_row = static_cast<Size>(n_genes - 1 - i);
            if (remaining < pairs_in_row) {
                j = static_cast<Index>(i + 1 + remaining);
                break;
            }
            remaining -= pairs_in_row;
        }

        if (i >= n_genes || j >= n_genes || i >= j) return;

        const Real* x = gene_expr + static_cast<Size>(i) * N;
        const Real* y = gene_expr + static_cast<Size>(j) * N;

        Real corr = Real(0);
        if (corr_type == CorrelationType::Pearson) {
            corr = detail::pearson_correlation_simd(x, y, n_cells, means[i], means[j]);
        } else if (corr_type == CorrelationType::Spearman) {
            Real* rank_x = rank_x_pool.get(thread_rank);
            Real* rank_y = rank_y_pool.get(thread_rank);
            Index* temp_idx = idx_pool.get(thread_rank);
            corr = detail::spearman_correlation(x, y, n_cells, rank_x, rank_y, temp_idx);
        } else {
            corr = detail::bicor(x, y, n_cells);
        }

        corr_matrix[static_cast<Size>(i) * n_genes + j] = corr;
        corr_matrix[static_cast<Size>(j) * n_genes + i] = corr;
    });

    if (means) {
        scl::memory::aligned_free(means, SCL_ALIGNMENT);
    }
    scl::memory::aligned_free(gene_expr, SCL_ALIGNMENT);
}

// =============================================================================
// WGCNA Adjacency Matrix (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void wgcna_adjacency(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real power,
    Real* adjacency,
    CorrelationType corr_type = CorrelationType::Pearson,
    AdjacencyType adj_type = AdjacencyType::Unsigned
) {
    const Size G = static_cast<Size>(n_genes);
    const Size total = G * G;

    auto corr_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* corr = corr_ptr.release();
    correlation_matrix(expression, n_cells, n_genes, corr, corr_type);

    // Parallel conversion to adjacency
    scl::threading::parallel_for(Size(0), G, [&](Size i) {
        adjacency[i * G + i] = Real(1);

        for (Size j = i + 1; j < G; ++j) {
            Real c = corr[i * G + j];
            Real adj = detail::correlation_to_adjacency(c, power, adj_type);

            adjacency[i * G + j] = adj;
            adjacency[j * G + i] = adj;
        }
    });

    scl::memory::aligned_free(corr, SCL_ALIGNMENT);
}

// =============================================================================
// TOM (Parallel)
// =============================================================================

inline void topological_overlap_matrix(
    const Real* adjacency,
    Index n_genes,
    Real* tom
) {
    const Size G = static_cast<Size>(n_genes);
    const Size total = G * G;

    scl::algo::zero(tom, total);

    // Precompute connectivity
    auto connectivity_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Real* connectivity = connectivity_ptr.release();

    scl::threading::parallel_for(Size(0), G, [&](Size i) {
        Real ki = Real(0);
        for (Index j = 0; j < n_genes; ++j) {
            if (static_cast<Size>(j) != i) {
                ki += adjacency[i * G + j];
            }
        }
        connectivity[i] = ki;
    });

    // Set diagonal
    for (Index i = 0; i < n_genes; ++i) {
        tom[static_cast<Size>(i) * n_genes + i] = Real(1);
    }

    // Parallel TOM computation (upper triangular)
    Size n_pairs = G * (G - 1) / 2;

    scl::threading::parallel_for(Size(0), n_pairs, [&](Size pair_idx) {
        // Convert pair index to (i, j) for upper triangular
        Index i = 0, j = 0;
        Size remaining = pair_idx;
        for (i = 0; i < n_genes - 1; ++i) {
            Size pairs_in_row = static_cast<Size>(n_genes - 1 - i);
            if (remaining < pairs_in_row) {
                j = static_cast<Index>(i + 1 + remaining);
                break;
            }
            remaining -= pairs_in_row;
        }

        if (i >= n_genes || j >= n_genes || i >= j) return;

        // Sum of common neighbors
        Real sum_common = Real(0);
        for (Index k = 0; k < n_genes; ++k) {
            if (k != i && k != j) {
                sum_common += adjacency[static_cast<Size>(i) * n_genes + k] *
                              adjacency[static_cast<Size>(k) * n_genes + j];
            }
        }

        Real a_ij = adjacency[static_cast<Size>(i) * n_genes + j];
        Real numerator = sum_common + a_ij;
        Real denom = scl::algo::min2(connectivity[i], connectivity[j]) + Real(1) - a_ij;

        Real tom_ij = (denom > config::EPSILON) ? numerator / denom : Real(0);

        tom[static_cast<Size>(i) * n_genes + j] = tom_ij;
        tom[static_cast<Size>(j) * n_genes + i] = tom_ij;
    });

    scl::memory::aligned_free(connectivity, SCL_ALIGNMENT);
}

// =============================================================================
// TOM Dissimilarity
// =============================================================================

inline void tom_dissimilarity(
    const Real* tom,
    Index n_genes,
    Real* dissim
) {
    const Size total = static_cast<Size>(n_genes) * n_genes;

    scl::threading::parallel_for(Size(0), total, [&](Size idx) {
        dissim[idx] = Real(1) - tom[idx];
    });
}

// =============================================================================
// Hierarchical Clustering (Optimized)
// =============================================================================

inline void hierarchical_clustering(
    const Real* dissim,
    Index n_genes,
    Index* merge_order,
    Real* merge_heights,
    [[maybe_unused]] Index* cluster_labels
) {
    const Size G = static_cast<Size>(n_genes);

    auto cluster_id_ptr = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);
    auto cluster_size_ptr = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);
    auto cluster_dist_ptr = scl::memory::aligned_alloc<Real>(G * G, SCL_ALIGNMENT);
    Index* cluster_id = cluster_id_ptr.release();
    Index* cluster_size = cluster_size_ptr.release();
    Real* cluster_dist = cluster_dist_ptr.release();

    for (Index i = 0; i < n_genes; ++i) {
        cluster_id[i] = i;
        cluster_size[i] = 1;
    }

    std::memcpy(cluster_dist, dissim, G * G * sizeof(Real));

    Index n_clusters = n_genes;
    Index merge_idx = 0;

    // Active cluster mask
    auto active_char_ptr = scl::memory::aligned_alloc<char>(G, SCL_ALIGNMENT);
    bool* active = reinterpret_cast<bool*>(active_char_ptr.release());
    for (Size i = 0; i < G; ++i) {
        active[i] = true;
    }

    while (n_clusters > 1) {
        // Find minimum distance (can parallelize for large n)
        Real min_dist = Real(1e30);
        Index min_i = 0, min_j = 1;

        if (n_genes >= static_cast<Index>(config::PARALLEL_THRESHOLD)) {
            const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());
            auto thread_min_dist_ptr = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
            auto thread_min_i_ptr = scl::memory::aligned_alloc<Index>(n_threads, SCL_ALIGNMENT);
            auto thread_min_j_ptr = scl::memory::aligned_alloc<Index>(n_threads, SCL_ALIGNMENT);
            Real* thread_min_dist = thread_min_dist_ptr.release();
            Index* thread_min_i = thread_min_i_ptr.release();
            Index* thread_min_j = thread_min_j_ptr.release();

            for (Size t = 0; t < n_threads; ++t) {
                thread_min_dist[t] = Real(1e30);
                thread_min_i[t] = 0;
                thread_min_j[t] = 1;
            }

            scl::threading::parallel_for(Size(0), G, [&](Size i, Size thread_rank) {
                if (!active[i]) return;

                for (Size j = i + 1; j < G; ++j) {
                    if (!active[j]) continue;

                    Real d = cluster_dist[i * G + j];
                    if (d < thread_min_dist[thread_rank]) {
                        thread_min_dist[thread_rank] = d;
                        thread_min_i[thread_rank] = static_cast<Index>(i);
                        thread_min_j[thread_rank] = static_cast<Index>(j);
                    }
                }
            });

            for (Size t = 0; t < n_threads; ++t) {
                if (thread_min_dist[t] < min_dist) {
                    min_dist = thread_min_dist[t];
                    min_i = thread_min_i[t];
                    min_j = thread_min_j[t];
                }
            }

            scl::memory::aligned_free(thread_min_j, SCL_ALIGNMENT);
            scl::memory::aligned_free(thread_min_i, SCL_ALIGNMENT);
            scl::memory::aligned_free(thread_min_dist, SCL_ALIGNMENT);
        } else {
            for (Index i = 0; i < n_genes; ++i) {
                if (!active[i]) continue;
                for (Index j = i + 1; j < n_genes; ++j) {
                    if (!active[j]) continue;
                    Real d = cluster_dist[static_cast<Size>(i) * n_genes + j];
                    if (d < min_dist) {
                        min_dist = d;
                        min_i = i;
                        min_j = j;
                    }
                }
            }
        }

        merge_order[merge_idx * 2] = min_i;
        merge_order[merge_idx * 2 + 1] = min_j;
        merge_heights[merge_idx] = min_dist;
        ++merge_idx;

        // Update distances (average linkage)
        Index new_size = cluster_size[min_i] + cluster_size[min_j];

        for (Index k = 0; k < n_genes; ++k) {
            if (k == min_i || k == min_j || !active[k]) continue;

            Real d_ik = cluster_dist[static_cast<Size>(min_i) * n_genes + k];
            Real d_jk = cluster_dist[static_cast<Size>(min_j) * n_genes + k];

            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            Real new_dist = (d_ik * static_cast<Real>(cluster_size[min_i]) + 
                           d_jk * static_cast<Real>(cluster_size[min_j])) /
                           static_cast<Real>(new_size);

            cluster_dist[static_cast<Size>(min_i) * n_genes + k] = new_dist;
            cluster_dist[static_cast<Size>(k) * n_genes + min_i] = new_dist;
        }

        cluster_size[min_i] = new_size;
        active[min_j] = false;
        --n_clusters;
    }

    scl::memory::aligned_free(reinterpret_cast<char*>(active), SCL_ALIGNMENT);
    scl::memory::aligned_free(cluster_dist, SCL_ALIGNMENT);
    scl::memory::aligned_free(cluster_size, SCL_ALIGNMENT);
    scl::memory::aligned_free(cluster_id, SCL_ALIGNMENT);
}

// =============================================================================
// Cut Tree at Height
// =============================================================================

inline Index cut_tree(
    const Index* merge_order,
    const Real* merge_heights,
    Index n_genes,
    Real cut_height,
    Index* module_labels
) {
    for (Index i = 0; i < n_genes; ++i) {
        module_labels[i] = i;
    }

    Index n_merges = n_genes - 1;
    auto parent_ptr = scl::memory::aligned_alloc<Index>(n_genes * 2, SCL_ALIGNMENT);
    Index* parent = parent_ptr.release();

    for (Index i = 0; i < n_genes * 2; ++i) {
        parent[i] = i;
    }

    auto find_root = [&parent](Index x) -> Index {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    for (Index m = 0; m < n_merges; ++m) {
        if (merge_heights[m] > cut_height) continue;

        Index a = merge_order[m * 2];
        Index b = merge_order[m * 2 + 1];

        Index root_a = find_root(a);
        Index root_b = find_root(b);

        if (root_a != root_b) {
            parent[root_b] = root_a;
        }
    }

    auto label_map_ptr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
    Index* label_map = label_map_ptr.release();
    for (Index i = 0; i < n_genes; ++i) {
        label_map[i] = -1;
    }

    Index next_label = 0;
    for (Index i = 0; i < n_genes; ++i) {
        Index root = find_root(i);
        if (label_map[root] < 0) {
            label_map[root] = next_label++;
        }
        module_labels[i] = label_map[root];
    }

    scl::memory::aligned_free(label_map, SCL_ALIGNMENT);
    scl::memory::aligned_free(parent, SCL_ALIGNMENT);

    return next_label;
}

// =============================================================================
// Detect Modules
// =============================================================================

inline Index detect_modules(
    const Real* dissim,
    Index n_genes,
    Index* module_labels,
    Index min_module_size = config::DEFAULT_MIN_MODULE_SIZE,
    Real merge_cut_height = config::DEFAULT_MERGE_CUT_HEIGHT
) {
    auto merge_order_ptr = scl::memory::aligned_alloc<Index>((n_genes - 1) * 2, SCL_ALIGNMENT);
    auto merge_heights_ptr = scl::memory::aligned_alloc<Real>(n_genes - 1, SCL_ALIGNMENT);
    Index* merge_order = merge_order_ptr.release();
    Real* merge_heights = merge_heights_ptr.release();

    hierarchical_clustering(dissim, n_genes, merge_order, merge_heights, nullptr);

    Index n_modules = cut_tree(merge_order, merge_heights, n_genes, merge_cut_height, module_labels);

    // Relabel small modules
    auto module_sizes_ptr = scl::memory::aligned_alloc<Index>(n_modules, SCL_ALIGNMENT);
    Index* module_sizes = module_sizes_ptr.release();
    scl::algo::zero(module_sizes, static_cast<Size>(n_modules));

    for (Index i = 0; i < n_genes; ++i) {
        ++module_sizes[module_labels[i]];
    }

    auto label_remap_ptr = scl::memory::aligned_alloc<Index>(n_modules, SCL_ALIGNMENT);
    Index* label_remap = label_remap_ptr.release();
    Index new_label = 1;

    for (Index m = 0; m < n_modules; ++m) {
        if (module_sizes[m] >= min_module_size) {
            label_remap[m] = new_label++;
        } else {
            label_remap[m] = 0;
        }
    }

    for (Index i = 0; i < n_genes; ++i) {
        module_labels[i] = label_remap[module_labels[i]];
    }

    scl::memory::aligned_free(label_remap, SCL_ALIGNMENT);
    scl::memory::aligned_free(module_sizes, SCL_ALIGNMENT);
    scl::memory::aligned_free(merge_heights, SCL_ALIGNMENT);
    scl::memory::aligned_free(merge_order, SCL_ALIGNMENT);

    return new_label;
}

// =============================================================================
// Module Eigengene (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void module_eigengene(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> module_labels,
    Index module_id,
    Index n_cells,
    Index n_genes,
    Array<Real> eigengene
) {
    const Size N = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(eigengene.len >= N, "Coexpression: eigengene buffer too small");

    // Find module genes
    Index n_module_genes = 0;
    for (Index g = 0; g < n_genes; ++g) {
        if (module_labels[g] == module_id) {
            ++n_module_genes;
        }
    }

    if (n_module_genes == 0) {
        scl::algo::zero(eigengene.ptr, N);
        return;
    }

    auto module_gene_idx_ptr = scl::memory::aligned_alloc<Index>(n_module_genes, SCL_ALIGNMENT);
    Index* module_gene_idx = module_gene_idx_ptr.release();
    Index idx = 0;
    for (Index g = 0; g < n_genes; ++g) {
        if (module_labels[g] == module_id) {
            module_gene_idx[idx++] = g;
        }
    }

    const Size M = static_cast<Size>(n_module_genes);

    // Extract and center module expression
    auto module_expr_ptr = scl::memory::aligned_alloc<Real>(N * M, SCL_ALIGNMENT);
    Real* module_expr = module_expr_ptr.release();
    scl::algo::zero(module_expr, N * M);

    // Extract (parallel over genes)
    scl::threading::parallel_for(Size(0), M, [&](Size m) {
        Index gene = module_gene_idx[m];

        if constexpr (IsCSR) {
            for (Index c = 0; c < n_cells; ++c) {
                auto indices = expression.row_indices_unsafe(c);
                auto values = expression.row_values_unsafe(c);
                const Index len = expression.row_length_unsafe(c);

                for (Index k = 0; k < len; ++k) {
                    if (indices.ptr[k] == gene) {
                        module_expr[static_cast<Size>(c) * M + m] = static_cast<Real>(values.ptr[k]);
                        break;
                    }
                }
            }
        } else {
            auto indices = expression.col_indices_unsafe(gene);
            auto values = expression.col_values_unsafe(gene);
            const Index len = expression.col_length_unsafe(gene);

            for (Index k = 0; k < len; ++k) {
                Index c = indices.ptr[k];
                if (c < n_cells) {
                    module_expr[static_cast<Size>(c) * M + m] = static_cast<Real>(values.ptr[k]);
                }
            }
        }
    });

    // Center each gene (parallel)
    scl::threading::parallel_for(Size(0), M, [&](Size m) {
        Real sum = Real(0);
        for (Index c = 0; c < n_cells; ++c) {
            sum += module_expr[static_cast<Size>(c) * M + m];
        }
        Real mean = sum / static_cast<Real>(n_cells);
        for (Index c = 0; c < n_cells; ++c) {
            module_expr[static_cast<Size>(c) * M + m] -= mean;
        }
    });

    // Compute first PC
    auto gene_loadings_ptr = scl::memory::aligned_alloc<Real>(M, SCL_ALIGNMENT);
    Real* gene_loadings = gene_loadings_ptr.release();
    detail::compute_first_pc_parallel(module_expr, n_cells, n_module_genes, gene_loadings);

    // Project (parallel)
    scl::threading::parallel_for(Size(0), N, [&](Size c) {
        const Real* row = module_expr + c * M;
        eigengene[static_cast<Index>(c)] = detail::dot_simd(row, gene_loadings, M);
    });

    scl::memory::aligned_free(gene_loadings, SCL_ALIGNMENT);
    scl::memory::aligned_free(module_expr, SCL_ALIGNMENT);
    scl::memory::aligned_free(module_gene_idx, SCL_ALIGNMENT);
}

// =============================================================================
// All Module Eigengenes (Parallel over modules)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void all_module_eigengenes(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> module_labels,
    Index n_modules,
    Index n_cells,
    Index n_genes,
    Real* eigengenes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size M = static_cast<Size>(n_modules);
    const Size total = N * M;

    scl::algo::zero(eigengenes, total);

    // Parallel over modules
    scl::threading::parallel_for(Size(0), M, [&](Size m) {
        auto temp_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        Real* temp = temp_ptr.release();

        module_eigengene(
            expression, module_labels, static_cast<Index>(m), n_cells, n_genes,
            Array<Real>(temp, N)
        );

        for (Size c = 0; c < N; ++c) {
            eigengenes[c * M + m] = temp[c];
        }

        scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    });
}

// =============================================================================
// Module-Trait Correlation (Parallel)
// =============================================================================

inline void module_trait_correlation(
    const Real* eigengenes,
    const Real* traits,
    Index n_samples,
    Index n_modules,
    Index n_traits,
    Real* correlations,
    Real* p_values = nullptr
) {
    const Size N = static_cast<Size>(n_samples);
    const Size M = static_cast<Size>(n_modules);
    const Size T = static_cast<Size>(n_traits);

    scl::threading::parallel_for(Size(0), M * T, [&](Size idx) {
        Size m = idx / T;
        Size t = idx % T;

        auto me_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        auto trait_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        Real* me = me_ptr.release();
        Real* trait = trait_ptr.release();

        for (Size s = 0; s < N; ++s) {
            me[s] = eigengenes[s * M + m];
            trait[s] = traits[s * T + t];
        }

        Real corr = detail::pearson_correlation(me, trait, n_samples);
        correlations[m * T + t] = corr;

        if (p_values) {
            Real z = Real(0.5) * std::log((Real(1) + corr + config::EPSILON) /
                                          (Real(1) - corr + config::EPSILON));
            Real se = Real(1) / std::sqrt(static_cast<Real>(n_samples - 3));
            Real t_stat = z / se;
            Real abs_t = std::abs(t_stat);
            Real p = Real(2) * std::exp(-Real(0.5) * abs_t * abs_t);
            p_values[m * T + t] = p;
        }

        scl::memory::aligned_free(trait, SCL_ALIGNMENT);
        scl::memory::aligned_free(me, SCL_ALIGNMENT);
    });
}

// =============================================================================
// Hub Gene Identification (Optimized)
// =============================================================================

inline void identify_hub_genes(
    const Real* adjacency,
    Array<const Index> module_labels,
    Index module_id,
    Index n_genes,
    Index* hub_genes,
    Real* hub_scores,
    Index max_hubs,
    Index& n_hubs
) {
    const Size G = static_cast<Size>(n_genes);

    // Parallel kME computation
    auto kme_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Real* kme = kme_ptr.release();
    scl::algo::zero(kme, G);

    std::atomic<Index> module_size_atomic{0};

    scl::threading::parallel_for(Size(0), G, [&](Size i) {
        if (module_labels[static_cast<Index>(i)] != module_id) return;

        module_size_atomic.fetch_add(1, std::memory_order_relaxed);

        Real k = Real(0);
        for (Index j = 0; j < n_genes; ++j) {
            if (static_cast<Size>(j) != i && module_labels[j] == module_id) {
                k += adjacency[i * G + j];
            }
        }
        kme[i] = k;
    });

    Index module_size = module_size_atomic.load();

    if (module_size > 1) {
        Real inv_size = Real(1) / static_cast<Real>(module_size - 1);
        for (Size i = 0; i < G; ++i) {
            if (module_labels[static_cast<Index>(i)] == module_id) {
                kme[i] *= inv_size;
            }
        }
    }

    // Collect and sort
    auto sorted_idx_ptr = scl::memory::aligned_alloc<Index>(module_size, SCL_ALIGNMENT);
    auto sorted_kme_ptr = scl::memory::aligned_alloc<Real>(module_size, SCL_ALIGNMENT);
    Index* sorted_idx = sorted_idx_ptr.release();
    Real* sorted_kme = sorted_kme_ptr.release();

    Index n_module_genes = 0;
    for (Index i = 0; i < n_genes; ++i) {
        if (module_labels[i] == module_id) {
            sorted_idx[n_module_genes] = i;
            sorted_kme[n_module_genes] = kme[i];
            ++n_module_genes;
        }
    }

    // Sort descending (insertion sort for small arrays)
    for (Index i = 1; i < n_module_genes; ++i) {
        Index idx = sorted_idx[i];
        Real val = sorted_kme[i];
        Index j = i;
        while (j > 0 && sorted_kme[j-1] < val) {
            sorted_idx[j] = sorted_idx[j-1];
            sorted_kme[j] = sorted_kme[j-1];
            --j;
        }
        sorted_idx[j] = idx;
        sorted_kme[j] = val;
    }

    n_hubs = scl::algo::min2(static_cast<Index>(max_hubs), n_module_genes);
    for (Index i = 0; i < n_hubs; ++i) {
        hub_genes[i] = sorted_idx[i];
        hub_scores[i] = sorted_kme[i];
    }

    scl::memory::aligned_free(sorted_kme, SCL_ALIGNMENT);
    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
    scl::memory::aligned_free(kme, SCL_ALIGNMENT);
}

// =============================================================================
// Gene Module Membership (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void gene_module_membership(
    const Sparse<T, IsCSR>& expression,
    const Real* eigengenes,
    Index n_cells,
    Index n_genes,
    Index n_modules,
    Real* kme_matrix
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size M = static_cast<Size>(n_modules);

    scl::algo::zero(kme_matrix, G * M);

    // Pre-extract gene expressions
    auto gene_expr_ptr = scl::memory::aligned_alloc<Real>(G * N, SCL_ALIGNMENT);
    Real* gene_expr = gene_expr_ptr.release();
    detail::extract_all_gene_expressions(expression, n_cells, n_genes, gene_expr);

    // Parallel over genes
    scl::threading::parallel_for(Size(0), G, [&](Size g) {
        const Real* gexpr = gene_expr + g * N;

        for (Index m = 0; m < n_modules; ++m) {
            auto me_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
            Real* me = me_ptr.release();
            for (Size c = 0; c < N; ++c) {
                me[c] = eigengenes[c * M + m];
            }

            Real corr = detail::pearson_correlation(gexpr, me, n_cells);
            kme_matrix[g * M + m] = corr;

            scl::memory::aligned_free(me, SCL_ALIGNMENT);
        }
    });

    scl::memory::aligned_free(gene_expr, SCL_ALIGNMENT);
}

// =============================================================================
// Module Preservation (Parallel)
// =============================================================================

inline void module_preservation(
    const Real* adjacency_ref,
    const Real* adjacency_test,
    Array<const Index> module_labels,
    Index n_genes,
    Index n_modules,
    Real* zsummary
) {
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_modules), [&](Size m) {
        Index module_size = 0;
        for (Index i = 0; i < n_genes; ++i) {
            if (module_labels[i] == static_cast<Index>(m)) ++module_size;
        }

        if (module_size < 3) {
            zsummary[m] = Real(0);
            return;
        }

        Real sum_kref = Real(0), sum_ktest = Real(0);
        Real sum_kref2 = Real(0), sum_ktest2 = Real(0);
        Real sum_kprod = Real(0);
        Index count = 0;

        for (Index i = 0; i < n_genes; ++i) {
            if (module_labels[i] != static_cast<Index>(m)) continue;

            Real k_ref = Real(0), k_test = Real(0);
            for (Index j = 0; j < n_genes; ++j) {
                if (i != j && module_labels[j] == static_cast<Index>(m)) {
                    k_ref += adjacency_ref[static_cast<Size>(i) * n_genes + j];
                    k_test += adjacency_test[static_cast<Size>(i) * n_genes + j];
                }
            }

            sum_kref += k_ref;
            sum_ktest += k_test;
            sum_kref2 += k_ref * k_ref;
            sum_ktest2 += k_test * k_test;
            sum_kprod += k_ref * k_test;
            ++count;
        }

        if (count < 2) {
            zsummary[m] = Real(0);
            return;
        }

        Real n = static_cast<Real>(count);
        Real mean_ref = sum_kref / n;
        Real mean_test = sum_ktest / n;

        Real var_ref = sum_kref2 / n - mean_ref * mean_ref;
        Real var_test = sum_ktest2 / n - mean_test * mean_test;
        Real cov = sum_kprod / n - mean_ref * mean_test;

        Real denom = std::sqrt(var_ref * var_test);
        Real cor_kIM = (denom > config::EPSILON) ? cov / denom : Real(0);

        zsummary[m] = cor_kIM * std::sqrt(n);
    });
}

// =============================================================================
// Pick Soft Threshold (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
Real pick_soft_threshold(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Real* powers_to_test,
    Index n_powers,
    Real* scale_free_fits,
    Real* mean_connectivity,
    CorrelationType corr_type = CorrelationType::Pearson
) {
    SCL_CHECK_ARG(n_powers > 0, "Coexpression: need at least one power to test");

    const Size G = static_cast<Size>(n_genes);
    const Size total = G * G;

    auto corr_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* corr = corr_ptr.release();
    correlation_matrix(expression, n_cells, n_genes, corr, corr_type);

    Real best_power = powers_to_test[0];
    Real best_fit = Real(-1);

    // Parallel over powers
    for (Index p = 0; p < n_powers; ++p) {
        Real power = powers_to_test[p];

        // Compute connectivity
        auto connectivity_ptr = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
        Real* connectivity = connectivity_ptr.release();

        scl::threading::parallel_for(Size(0), G, [&](Size i) {
            Real k = Real(0);
            for (Index j = 0; j < n_genes; ++j) {
                if (static_cast<Size>(j) != i) {
                    Real adj = std::pow(std::abs(corr[i * G + j]), power);
                    k += adj;
                }
            }
            connectivity[i] = k;
        });

        // Mean connectivity
        Real sum_k = detail::sum_simd(connectivity, G);
        mean_connectivity[p] = sum_k / static_cast<Real>(n_genes);

        // Scale-free fitting
        Real max_k = Real(0);
        for (Size i = 0; i < G; ++i) {
            max_k = scl::algo::max2(max_k, connectivity[i]);
        }

        if (max_k < config::EPSILON) {
            scale_free_fits[p] = Real(0);
            scl::memory::aligned_free(connectivity, SCL_ALIGNMENT);
            continue;
        }

        Index n_bins = 20;
        auto bin_counts_ptr = scl::memory::aligned_alloc<Real>(n_bins, SCL_ALIGNMENT);
        Real* bin_counts = bin_counts_ptr.release();
        scl::algo::zero(bin_counts, static_cast<Size>(n_bins));

        // Bin calculation: normalize connectivity to [0, 1] then map to bin index
        // max_k is guaranteed > EPSILON here, and n_bins >= 2, so division is safe
        const Real bin_width = max_k / static_cast<Real>(n_bins);
        for (Size i = 0; i < G; ++i) {
            auto bin = static_cast<Index>(connectivity[i] / bin_width);
            bin = scl::algo::min2(bin, n_bins - 1);
            bin_counts[bin] += Real(1);
        }

        auto log_k_ptr = scl::memory::aligned_alloc<Real>(n_bins, SCL_ALIGNMENT);
        auto log_pk_ptr = scl::memory::aligned_alloc<Real>(n_bins, SCL_ALIGNMENT);
        Real* log_k = log_k_ptr.release();
        Real* log_pk = log_pk_ptr.release();

        Index n_valid = 0;
        for (Index b = 0; b < n_bins; ++b) {
            if (bin_counts[b] > Real(0)) {
                // Convert bin index back to k value: bin center = (b + 0.5) * bin_width
                Real k_val = (static_cast<Real>(b) + Real(0.5)) * bin_width;
                if (k_val > config::EPSILON) {
                    log_k[n_valid] = std::log(k_val);
                    log_pk[n_valid] = std::log(bin_counts[b] / static_cast<Real>(n_genes));
                    ++n_valid;
                }
            }
        }

        if (n_valid >= 3) {
            Real mean_log_k = detail::compute_mean(log_k, n_valid);
            Real mean_log_pk = detail::compute_mean(log_pk, n_valid);

            Real sum_xy = Real(0), sum_xx = Real(0), sum_yy = Real(0);
            for (Index i = 0; i < n_valid; ++i) {
                Real dx = log_k[i] - mean_log_k;
                Real dy = log_pk[i] - mean_log_pk;
                sum_xy += dx * dy;
                sum_xx += dx * dx;
                sum_yy += dy * dy;
            }

            Real r2 = (sum_xx > config::EPSILON && sum_yy > config::EPSILON) ?
                      (sum_xy * sum_xy) / (sum_xx * sum_yy) : Real(0);

            Real slope = (sum_xx > config::EPSILON) ? sum_xy / sum_xx : Real(0);
            if (slope > Real(0)) {
                r2 = -r2;
            }

            scale_free_fits[p] = r2;

            if (r2 > best_fit) {
                best_fit = r2;
                best_power = power;
            }
        } else {
            scale_free_fits[p] = Real(0);
        }

        scl::memory::aligned_free(log_pk, SCL_ALIGNMENT);
        scl::memory::aligned_free(log_k, SCL_ALIGNMENT);
        scl::memory::aligned_free(bin_counts, SCL_ALIGNMENT);
        scl::memory::aligned_free(connectivity, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(corr, SCL_ALIGNMENT);
    return best_power;
}

// =============================================================================
// Blockwise Modules (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
void blockwise_modules(
    const Sparse<T, IsCSR>& expression,
    Index n_cells,
    Index n_genes,
    Index block_size,
    Real power,
    Index min_module_size,
    Index* module_labels,
    Index& n_modules,
    CorrelationType corr_type = CorrelationType::Pearson
) {
    Index n_blocks = (n_genes + block_size - 1) / block_size;

    if (n_blocks <= 1 || n_genes <= block_size) {
        const Size adj_size = static_cast<Size>(n_genes) * n_genes;
        auto adjacency_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
        auto tom_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
        auto dissim_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
        Real* adjacency = adjacency_ptr.release();
        Real* tom = tom_ptr.release();
        Real* dissim = dissim_ptr.release();

        wgcna_adjacency(expression, n_cells, n_genes, power, adjacency,
                       corr_type, AdjacencyType::Unsigned);
        topological_overlap_matrix(adjacency, n_genes, tom);
        tom_dissimilarity(tom, n_genes, dissim);

        n_modules = detect_modules(dissim, n_genes, module_labels, min_module_size);

        scl::memory::aligned_free(dissim, SCL_ALIGNMENT);
        scl::memory::aligned_free(tom, SCL_ALIGNMENT);
        scl::memory::aligned_free(adjacency, SCL_ALIGNMENT);
    } else {
        // Block-wise processing
        auto block_labels_ptr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
        Index* block_labels = block_labels_ptr.release();
        std::atomic<Index> total_modules{0};

        // Pre-extract all gene expressions for efficiency
        const Size N = static_cast<Size>(n_cells);
        const Size G = static_cast<Size>(n_genes);
        auto all_gene_expr_ptr = scl::memory::aligned_alloc<Real>(G * N, SCL_ALIGNMENT);
        Real* all_gene_expr = all_gene_expr_ptr.release();
        detail::extract_all_gene_expressions(expression, n_cells, n_genes, all_gene_expr);

        // Process blocks (could parallelize over blocks if memory permits)
        for (Index b = 0; b < n_blocks; ++b) {
            Index start = b * block_size;
            Index end = scl::algo::min2(start + block_size, n_genes);
            Index block_n = end - start;

            const Size adj_size = static_cast<Size>(block_n) * block_n;
            auto adjacency_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
            auto tom_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
            auto dissim_ptr = scl::memory::aligned_alloc<Real>(adj_size, SCL_ALIGNMENT);
            Real* adjacency = adjacency_ptr.release();
            Real* tom = tom_ptr.release();
            Real* dissim = dissim_ptr.release();

            // Compute adjacency for block
            for (Index i = 0; i < block_n; ++i) {
                adjacency[static_cast<Size>(i) * block_n + i] = Real(1);

                for (Index j = i + 1; j < block_n; ++j) {
                    const Real* expr_i = all_gene_expr + static_cast<Size>(start + i) * N;
                    const Real* expr_j = all_gene_expr + static_cast<Size>(start + j) * N;

                    Real corr = detail::pearson_correlation(expr_i, expr_j, n_cells);
                    Real adj = std::pow(std::abs(corr), power);

                    adjacency[static_cast<Size>(i) * block_n + j] = adj;
                    adjacency[static_cast<Size>(j) * block_n + i] = adj;
                }
            }

            topological_overlap_matrix(adjacency, block_n, tom);
            tom_dissimilarity(tom, block_n, dissim);

            auto block_module_labels_ptr = scl::memory::aligned_alloc<Index>(block_n, SCL_ALIGNMENT);
            Index* block_module_labels = block_module_labels_ptr.release();
            Index block_mods = detect_modules(dissim, block_n, block_module_labels, min_module_size);

            Index offset = total_modules.fetch_add(block_mods);
            for (Index i = 0; i < block_n; ++i) {
                block_labels[start + i] = block_module_labels[i] + offset;
            }

            scl::memory::aligned_free(block_module_labels, SCL_ALIGNMENT);
            scl::memory::aligned_free(dissim, SCL_ALIGNMENT);
            scl::memory::aligned_free(tom, SCL_ALIGNMENT);
            scl::memory::aligned_free(adjacency, SCL_ALIGNMENT);
        }

        std::memcpy(module_labels, block_labels, static_cast<Size>(n_genes) * sizeof(Index));
        n_modules = total_modules.load();

        scl::memory::aligned_free(all_gene_expr, SCL_ALIGNMENT);
        scl::memory::aligned_free(block_labels, SCL_ALIGNMENT);
    }
}

} // namespace scl::kernel::coexpression
