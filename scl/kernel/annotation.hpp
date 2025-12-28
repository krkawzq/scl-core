#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/annotation.hpp
// BRIEF: Cell type annotation from reference
//
// STRATEGIC POSITION: Sparse + Classification (Tier 2 - Annotation)
// - Reference-based cell type transfer
// - Marker gene scoring
// - Confidence estimation
//
// APPLICATIONS:
// - Automated cell type annotation
// - Reference mapping (Seurat-style)
// - SingleR-style correlation assignment
// - ScType-style marker scoring
// - Novel cell type detection
//
// KEY METHODS:
// - KNN voting from reference
// - Correlation with type profiles
// - Marker gene scoring
// - Consensus voting
// =============================================================================

namespace scl::kernel::annotation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_CONFIDENCE_THRESHOLD = Real(0.5);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_K = 15;
    constexpr Real DEFAULT_NOVELTY_THRESHOLD = Real(0.3);
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Annotation Method Types
// =============================================================================

enum class AnnotationMethod {
    KNNVoting,          // K-nearest neighbor voting
    Correlation,        // Correlation with reference profiles
    MarkerScore,        // Marker gene scoring
    Weighted            // Weighted combination
};

enum class DistanceMetric {
    Cosine,
    Euclidean,
    Correlation,
    Manhattan
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Compute mean of sparse row - SIMD optimized
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_row_mean(
    const Sparse<T, IsCSR>& X,
    Index row,
    Index n_cols
) {
    if (SCL_UNLIKELY(n_cols == 0)) return Real(0);

    if (IsCSR) {
        auto values = X.row_values(row);
        Index len = X.row_length(row);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) return Real(0);

        // Use vectorize::sum for SIMD acceleration
        Real sum = scl::vectorize::sum(Array<const Real>(
            reinterpret_cast<const Real*>(values.ptr), len_sz));
        return sum / static_cast<Real>(n_cols);
    } else {
        Real sum = Real(0);
        for (Index c = 0; c < n_cols; ++c) {
            auto indices = X.col_indices(c);
            auto values = X.col_values(c);
            Index len = X.col_length(c);

            // Binary search for row in sorted column indices
            Index lo = 0, hi = len;
            while (lo < hi) {
                Index mid = lo + (hi - lo) / 2;
                if (indices[mid] < row) lo = mid + 1;
                else hi = mid;
            }
            if (lo < len && indices[lo] == row) {
                sum += static_cast<Real>(values[lo]);
            }
        }
        return sum / static_cast<Real>(n_cols);
    }
}

// Compute L2 norm of sparse row - SIMD optimized
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_row_norm(
    const Sparse<T, IsCSR>& X,
    Index row
) {
    if (IsCSR) {
        auto values = X.row_values(row);
        Index len = X.row_length(row);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) return Real(0);

        // Use vectorize::sum_squared for SIMD acceleration
        Real sum_sq = scl::vectorize::sum_squared(Array<const Real>(
            reinterpret_cast<const Real*>(values.ptr), len_sz));
        return std::sqrt(sum_sq);
    } else {
        Real sum_sq = Real(0);
        Index n_cols = X.cols();
        for (Index c = 0; c < n_cols; ++c) {
            auto indices = X.col_indices(c);
            auto values = X.col_values(c);
            Index len = X.col_length(c);

            // Binary search for row
            Index lo = 0, hi = len;
            while (lo < hi) {
                Index mid = lo + (hi - lo) / 2;
                if (indices[mid] < row) lo = mid + 1;
                else hi = mid;
            }
            if (lo < len && indices[lo] == row) {
                Real v = static_cast<Real>(values[lo]);
                sum_sq += v * v;
            }
        }
        return std::sqrt(sum_sq);
    }
}

// Compute dot product between two sparse rows - using adaptive algorithm
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real sparse_dot_product(
    const Sparse<T, IsCSR>& X,
    Index row_x,
    const Sparse<T, IsCSR>& Y,
    Index row_y
) {
    if (IsCSR) {
        auto x_indices = X.row_indices(row_x);
        auto x_values = X.row_values(row_x);
        Index x_len = X.row_length(row_x);

        auto y_indices = Y.row_indices(row_y);
        auto y_values = Y.row_values(row_y);
        Index y_len = Y.row_length(row_y);

        if (SCL_UNLIKELY(x_len == 0 || y_len == 0)) return Real(0);

        // Use adaptive sparse dot product with skip optimization
        return scl::algo::sparse_dot_adaptive(
            x_indices.ptr, reinterpret_cast<const Real*>(x_values.ptr), static_cast<Size>(x_len),
            y_indices.ptr, reinterpret_cast<const Real*>(y_values.ptr), static_cast<Size>(y_len)
        );
    }

    return Real(0);
}

// Cosine similarity between sparse rows
template <typename T, bool IsCSR>
Real cosine_similarity(
    const Sparse<T, IsCSR>& X,
    Index row_x,
    const Sparse<T, IsCSR>& Y,
    Index row_y
) {
    Real dot = sparse_dot_product(X, row_x, Y, row_y);
    Real norm_x = compute_row_norm(X, row_x);
    Real norm_y = compute_row_norm(Y, row_y);

    Real denom = norm_x * norm_y;
    return (denom > config::EPSILON) ? dot / denom : Real(0);
}

// Pearson correlation using algebraic identity (avoids dense allocation)
// correlation(x, y) = (n*sum(xy) - sum(x)*sum(y)) / 
//                     sqrt((n*sum(x^2) - sum(x)^2) * (n*sum(y^2) - sum(y)^2))
template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real pearson_correlation(
    const Sparse<T, IsCSR>& X,
    Index row_x,
    const Sparse<T, IsCSR>& Y,
    Index row_y,
    Index n_features
) {
    if (SCL_UNLIKELY(n_features == 0)) return Real(0);

    if (IsCSR) {
        auto x_indices = X.row_indices(row_x);
        auto x_values = X.row_values(row_x);
        Index x_len = X.row_length(row_x);
        const Size x_len_sz = static_cast<Size>(x_len);

        auto y_indices = Y.row_indices(row_y);
        auto y_values = Y.row_values(row_y);
        Index y_len = Y.row_length(row_y);
        const Size y_len_sz = static_cast<Size>(y_len);

        // Compute sums and sum of squares using SIMD
        Real sum_x = scl::vectorize::sum(Array<const Real>(
            reinterpret_cast<const Real*>(x_values.ptr), x_len_sz));
        Real sum_y = scl::vectorize::sum(Array<const Real>(
            reinterpret_cast<const Real*>(y_values.ptr), y_len_sz));
        Real sum_xx = scl::vectorize::sum_squared(Array<const Real>(
            reinterpret_cast<const Real*>(x_values.ptr), x_len_sz));
        Real sum_yy = scl::vectorize::sum_squared(Array<const Real>(
            reinterpret_cast<const Real*>(y_values.ptr), y_len_sz));

        // Compute sum_xy using optimized sparse dot
        Real sum_xy = sparse_dot_product(X, row_x, Y, row_y);

        // Apply algebraic identity
        Real n = static_cast<Real>(n_features);
        Real numerator = n * sum_xy - sum_x * sum_y;
        Real denom_x = n * sum_xx - sum_x * sum_x;
        Real denom_y = n * sum_yy - sum_y * sum_y;

        Real denom = std::sqrt(denom_x * denom_y);
        return (denom > config::EPSILON) ? numerator / denom : Real(0);
    } else {
        // CSC fallback - convert to dense (less common path)
        Real mean_x = compute_row_mean(X, row_x, n_features);
        Real mean_y = compute_row_mean(Y, row_y, n_features);
        Real sum_xy = Real(0);
        Real sum_xx = Real(0);
        Real sum_yy = Real(0);

        Real* x_dense = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
        Real* y_dense = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
        scl::algo::zero(x_dense, static_cast<Size>(n_features));
        scl::algo::zero(y_dense, static_cast<Size>(n_features));

        for (Index c = 0; c < n_features; ++c) {
            auto x_indices = X.col_indices(c);
            auto x_values = X.col_values(c);
            Index x_len = X.col_length(c);

            // Binary search for row_x
            Index lo = 0, hi = x_len;
            while (lo < hi) {
                Index mid = lo + (hi - lo) / 2;
                if (x_indices[mid] < row_x) lo = mid + 1;
                else hi = mid;
            }
            if (lo < x_len && x_indices[lo] == row_x) {
                x_dense[c] = static_cast<Real>(x_values[lo]);
            }

            auto y_indices = Y.col_indices(c);
            auto y_values = Y.col_values(c);
            Index y_len = Y.col_length(c);

            lo = 0; hi = y_len;
            while (lo < hi) {
                Index mid = lo + (hi - lo) / 2;
                if (y_indices[mid] < row_y) lo = mid + 1;
                else hi = mid;
            }
            if (lo < y_len && y_indices[lo] == row_y) {
                y_dense[c] = static_cast<Real>(y_values[lo]);
            }
        }

        // SIMD-friendly correlation computation
        Index f = 0;
        for (; f + 4 <= n_features; f += 4) {
            Real dx0 = x_dense[f] - mean_x;
            Real dx1 = x_dense[f+1] - mean_x;
            Real dx2 = x_dense[f+2] - mean_x;
            Real dx3 = x_dense[f+3] - mean_x;

            Real dy0 = y_dense[f] - mean_y;
            Real dy1 = y_dense[f+1] - mean_y;
            Real dy2 = y_dense[f+2] - mean_y;
            Real dy3 = y_dense[f+3] - mean_y;

            sum_xy += dx0 * dy0 + dx1 * dy1 + dx2 * dy2 + dx3 * dy3;
            sum_xx += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
            sum_yy += dy0 * dy0 + dy1 * dy1 + dy2 * dy2 + dy3 * dy3;
        }

        for (; f < n_features; ++f) {
            Real dx = x_dense[f] - mean_x;
            Real dy = y_dense[f] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        scl::memory::aligned_free(y_dense, SCL_ALIGNMENT);
        scl::memory::aligned_free(x_dense, SCL_ALIGNMENT);

        Real denom = std::sqrt(sum_xx * sum_yy);
        return (denom > config::EPSILON) ? sum_xy / denom : Real(0);
    }
}

// Count label votes
void count_votes(
    const Index* labels,
    Index n,
    Index n_types,
    Index* counts
) {
    scl::algo::zero(counts, static_cast<Size>(n_types));

    for (Index i = 0; i < n; ++i) {
        Index lbl = labels[i];
        if (SCL_LIKELY(lbl >= 0 && lbl < n_types)) {
            ++counts[lbl];
        }
    }
}

// Find majority vote
Index majority_vote(
    const Index* counts,
    Index n_types,
    Real& confidence
) {
    Index total = 0;
    Index max_count = 0;
    Index max_label = 0;

    for (Index t = 0; t < n_types; ++t) {
        total += counts[t];
        if (SCL_LIKELY(counts[t] > max_count)) {
            max_count = counts[t];
            max_label = t;
        }
    }

    confidence = SCL_LIKELY(total > 0) ?
        static_cast<Real>(max_count) / static_cast<Real>(total) : Real(0);

    return max_label;
}

// Weighted majority vote
Index weighted_majority_vote(
    const Index* labels,
    const Real* weights,
    Index n,
    Index n_types,
    Real& confidence
) {
    Real* weighted_counts = scl::memory::aligned_alloc<Real>(n_types, SCL_ALIGNMENT);
    scl::algo::zero(weighted_counts, static_cast<Size>(n_types));

    Real total_weight = Real(0);
    for (Index i = 0; i < n; ++i) {
        Index lbl = labels[i];
        if (SCL_LIKELY(lbl >= 0 && lbl < n_types)) {
            weighted_counts[lbl] += weights[i];
            total_weight += weights[i];
        }
    }

    Real max_weight = Real(0);
    Index max_label = 0;

    for (Index t = 0; t < n_types; ++t) {
        if (weighted_counts[t] > max_weight) {
            max_weight = weighted_counts[t];
            max_label = t;
        }
    }

    confidence = (total_weight > config::EPSILON) ?
        max_weight / total_weight : Real(0);

    scl::memory::aligned_free(weighted_counts, SCL_ALIGNMENT);

    return max_label;
}

// -----------------------------------------------------------------------------
// SIMD softmax normalization using scl::simd::Exp
// -----------------------------------------------------------------------------
SCL_FORCE_INLINE SCL_HOT void softmax_normalize_simd(Real* scores, Index n) {
    if (SCL_UNLIKELY(n == 0)) return;
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    // Find max for numerical stability
    Real max_score = scores[0];
    auto v_max = s::Set(d, max_score);
    Index i = 0;
    
    // SIMD max reduction
    for (; i + static_cast<Index>(lanes) <= n; i += static_cast<Index>(lanes)) {
        auto v = s::Load(d, scores + i);
        v_max = s::Max(v_max, v);
    }
    max_score = s::GetLane(s::MaxOfLanes(d, v_max));
    
    // Scalar cleanup for max
    for (; i < n; ++i) {
        max_score = scl::algo::max2(max_score, scores[i]);
    }
    
    // Subtract max and compute exp + sum
    auto v_neg_max = s::Set(d, -max_score);
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    
    i = 0;
    
    // 4-way unrolled SIMD exp with multi-accumulator
    for (; i + 4 * static_cast<Index>(lanes) <= n; i += 4 * static_cast<Index>(lanes)) {
        auto v0 = s::Load(d, scores + i + 0 * lanes);
        auto v1 = s::Load(d, scores + i + 1 * lanes);
        auto v2 = s::Load(d, scores + i + 2 * lanes);
        auto v3 = s::Load(d, scores + i + 3 * lanes);
        
        auto e0 = s::Exp(d, s::Add(v0, v_neg_max));
        auto e1 = s::Exp(d, s::Add(v1, v_neg_max));
        auto e2 = s::Exp(d, s::Add(v2, v_neg_max));
        auto e3 = s::Exp(d, s::Add(v3, v_neg_max));
        
        s::Store(e0, d, scores + i + 0 * lanes);
        s::Store(e1, d, scores + i + 1 * lanes);
        s::Store(e2, d, scores + i + 2 * lanes);
        s::Store(e3, d, scores + i + 3 * lanes);
        
        v_sum0 = s::Add(v_sum0, s::Add(e0, e1));
        v_sum1 = s::Add(v_sum1, s::Add(e2, e3));
    }
    
    auto v_sum = s::Add(v_sum0, v_sum1);
    
    // Remaining SIMD lanes
    for (; i + static_cast<Index>(lanes) <= n; i += static_cast<Index>(lanes)) {
        auto v = s::Load(d, scores + i);
        auto e = s::Exp(d, s::Add(v, v_neg_max));
        s::Store(e, d, scores + i);
        v_sum = s::Add(v_sum, e);
    }
    
    Real sum_exp = s::GetLane(s::SumOfLanes(d, v_sum));
    
    // Scalar cleanup - use std::exp for efficiency
    for (; i < n; ++i) {
        Real e = std::exp(scores[i] - max_score);
        scores[i] = e;
        sum_exp += e;
    }
    
    // Normalize with SIMD
    if (SCL_LIKELY(sum_exp > config::EPSILON)) {
        Real inv_sum = Real(1) / sum_exp;
        auto v_inv_sum = s::Set(d, inv_sum);
        
        i = 0;
        for (; i + static_cast<Index>(lanes) <= n; i += static_cast<Index>(lanes)) {
            auto v = s::Load(d, scores + i);
            s::Store(s::Mul(v, v_inv_sum), d, scores + i);
        }
        
        // Scalar cleanup
        for (; i < n; ++i) {
            scores[i] *= inv_sum;
        }
    }
}

} // namespace detail

// =============================================================================
// Count Cell Types in Reference
// =============================================================================

inline Index count_cell_types(
    Array<const Index> labels,
    Index n
) {
    Index max_label = 0;
    for (Index i = 0; i < n; ++i) {
        max_label = scl::algo::max2(max_label, labels[i]);
    }
    return max_label + 1;
}

// =============================================================================
// Reference Mapping (KNN-based Transfer)
// =============================================================================

template <typename T, bool IsCSR>
void reference_mapping(
    const Sparse<T, IsCSR>& query_expression,
    const Sparse<T, IsCSR>& reference_expression,
    Array<const Index> reference_labels,
    const Sparse<Index, IsCSR>& query_to_ref_neighbors,
    Index n_query,
    Index n_ref,
    Index n_types,
    Array<Index> query_labels,
    Array<Real> confidence_scores
) {
    SCL_CHECK_DIM(query_labels.len >= static_cast<Size>(n_query),
                  "Annotation: query_labels buffer too small");
    SCL_CHECK_DIM(confidence_scores.len >= static_cast<Size>(n_query),
                  "Annotation: confidence_scores buffer too small");

    // Parallel processing for large datasets
    if (static_cast<Size>(n_query) >= config::PARALLEL_THRESHOLD) {
        // Estimate max neighbors per query for workspace sizing
        Index max_neighbors = 0;
        for (Index q = 0; q < scl::algo::min2(n_query, Index(100)); ++q) {
            max_neighbors = scl::algo::max2(max_neighbors, 
                query_to_ref_neighbors.primary_length(q));
        }

        Size workspace_size = static_cast<Size>(max_neighbors) * (sizeof(Index) + sizeof(Real)) +
                              static_cast<Size>(n_types) * sizeof(Real);

        scl::threading::WorkspacePool<Byte> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, workspace_size);

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_query), 
            [&](size_t q_idx, size_t thread_rank) {
            const Index q = static_cast<Index>(q_idx);
            Byte* workspace = pool.get(thread_rank);

            auto neighbor_indices = query_to_ref_neighbors.primary_indices(q);
            auto neighbor_weights = query_to_ref_neighbors.primary_values(q);
            Index n_neighbors = query_to_ref_neighbors.primary_length(q);

            if (SCL_UNLIKELY(n_neighbors == 0)) {
                query_labels[q] = 0;
                confidence_scores[q] = Real(0);
                return;
            }

            // Get labels of neighbors
            Index* neighbor_labels = reinterpret_cast<Index*>(workspace);
            Real* weights = reinterpret_cast<Real*>(workspace + 
                static_cast<Size>(n_neighbors) * sizeof(Index));

            for (Index k = 0; k < n_neighbors; ++k) {
                Index ref_idx = neighbor_indices[k];
                if (SCL_LIKELY(ref_idx >= 0 && ref_idx < n_ref)) {
                    neighbor_labels[k] = reference_labels[ref_idx];
                } else {
                    neighbor_labels[k] = -1;
                }
                weights[k] = static_cast<Real>(neighbor_weights[k]);
            }

            // Weighted voting
            Real confidence;
            query_labels[q] = detail::weighted_majority_vote(
                neighbor_labels, weights, n_neighbors, n_types, confidence
            );
            confidence_scores[q] = confidence;
        });
    } else {
        // Sequential path for small datasets
        for (Index q = 0; q < n_query; ++q) {
            auto neighbor_indices = query_to_ref_neighbors.primary_indices(q);
            auto neighbor_weights = query_to_ref_neighbors.primary_values(q);
            Index n_neighbors = query_to_ref_neighbors.primary_length(q);

            if (SCL_UNLIKELY(n_neighbors == 0)) {
                query_labels[q] = 0;
                confidence_scores[q] = Real(0);
                continue;
            }

            Index* neighbor_labels = scl::memory::aligned_alloc<Index>(n_neighbors, SCL_ALIGNMENT);
            Real* weights = scl::memory::aligned_alloc<Real>(n_neighbors, SCL_ALIGNMENT);

            for (Index k = 0; k < n_neighbors; ++k) {
                Index ref_idx = neighbor_indices[k];
                if (SCL_LIKELY(ref_idx >= 0 && ref_idx < n_ref)) {
                    neighbor_labels[k] = reference_labels[ref_idx];
                } else {
                    neighbor_labels[k] = -1;
                }
                weights[k] = static_cast<Real>(neighbor_weights[k]);
            }

            Real confidence;
            query_labels[q] = detail::weighted_majority_vote(
                neighbor_labels, weights, n_neighbors, n_types, confidence
            );
            confidence_scores[q] = confidence;

            scl::memory::aligned_free(weights, SCL_ALIGNMENT);
            scl::memory::aligned_free(neighbor_labels, SCL_ALIGNMENT);
        }
    }
}

// =============================================================================
// Correlation-Based Assignment (SingleR-style)
// =============================================================================

template <typename T, bool IsCSR>
void correlation_assignment(
    const Sparse<T, IsCSR>& query_expression,
    const Sparse<T, IsCSR>& reference_profiles,  // n_types x n_genes
    Index n_query,
    Index n_types,
    Index n_genes,
    Array<Index> assigned_labels,
    Array<Real> correlation_scores,
    Array<Real> all_correlations  // Optional: n_query x n_types, can be NULL
) {
    SCL_CHECK_DIM(assigned_labels.len >= static_cast<Size>(n_query),
                  "Annotation: assigned_labels buffer too small");
    SCL_CHECK_DIM(correlation_scores.len >= static_cast<Size>(n_query),
                  "Annotation: correlation_scores buffer too small");

    // Parallel processing for large datasets
    if (static_cast<Size>(n_query) >= config::PARALLEL_THRESHOLD) {
        Size workspace_size = static_cast<Size>(n_types) * sizeof(Real);

        scl::threading::WorkspacePool<Real> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, static_cast<Size>(n_types));

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_query),
            [&](size_t q_idx, size_t thread_rank) {
            const Index q = static_cast<Index>(q_idx);
            Real* type_corrs = pool.get(thread_rank);

            Real max_corr = -Real(2);
            Index max_type = 0;

            for (Index t = 0; t < n_types; ++t) {
                Real corr = detail::pearson_correlation(
                    query_expression, q, reference_profiles, t, n_genes
                );

                type_corrs[t] = corr;

                if (corr > max_corr) {
                    max_corr = corr;
                    max_type = t;
                }
            }

            assigned_labels[q] = max_type;
            correlation_scores[q] = max_corr;

            // Store all correlations if requested
            if (all_correlations.ptr != nullptr) {
                for (Index t = 0; t < n_types; ++t) {
                    all_correlations[static_cast<Size>(q) * n_types + t] = type_corrs[t];
                }
            }
        });
    } else {
        // Sequential path
        Real* type_corrs = scl::memory::aligned_alloc<Real>(n_types, SCL_ALIGNMENT);

        for (Index q = 0; q < n_query; ++q) {
            Real max_corr = -Real(2);
            Index max_type = 0;

            for (Index t = 0; t < n_types; ++t) {
                Real corr = detail::pearson_correlation(
                    query_expression, q, reference_profiles, t, n_genes
                );

                type_corrs[t] = corr;

                if (corr > max_corr) {
                    max_corr = corr;
                    max_type = t;
                }
            }

            assigned_labels[q] = max_type;
            correlation_scores[q] = max_corr;

            if (all_correlations.ptr != nullptr) {
                for (Index t = 0; t < n_types; ++t) {
                    all_correlations[static_cast<Size>(q) * n_types + t] = type_corrs[t];
                }
            }
        }

        scl::memory::aligned_free(type_corrs, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Build Reference Profiles (Mean per Cell Type)
// =============================================================================

template <typename T, bool IsCSR>
void build_reference_profiles(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Index n_cells,
    Index n_genes,
    Index n_types,
    Real* profiles  // n_types x n_genes output
) {
    Size total = static_cast<Size>(n_types) * static_cast<Size>(n_genes);
    scl::algo::zero(profiles, total);

    Index* type_counts = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);
    scl::algo::zero(type_counts, static_cast<Size>(n_types));

    // Count cells per type
    for (Index c = 0; c < n_cells; ++c) {
        Index t = labels[c];
        if (SCL_LIKELY(t >= 0 && t < n_types)) {
            ++type_counts[t];
        }
    }

    // Accumulate expression
    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            Index t = labels[c];
            if (SCL_UNLIKELY(t < 0 || t >= n_types)) continue;

            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            Real* profile = profiles + static_cast<Size>(t) * n_genes;

            // 4-way unrolled accumulation
            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                Index g0 = indices[k], g1 = indices[k+1], g2 = indices[k+2], g3 = indices[k+3];
                if (SCL_LIKELY(g0 < n_genes)) profile[g0] += static_cast<Real>(values[k]);
                if (SCL_LIKELY(g1 < n_genes)) profile[g1] += static_cast<Real>(values[k+1]);
                if (SCL_LIKELY(g2 < n_genes)) profile[g2] += static_cast<Real>(values[k+2]);
                if (SCL_LIKELY(g3 < n_genes)) profile[g3] += static_cast<Real>(values[k+3]);
            }
            for (; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    profile[g] += static_cast<Real>(values[k]);
                }
            }
        }
    } else {
        for (Index g = 0; g < n_genes; ++g) {
            auto indices = expression.col_indices(g);
            auto values = expression.col_values(g);
            Index len = expression.col_length(g);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c >= n_cells) continue;

                Index t = labels[c];
                if (t >= 0 && t < n_types) {
                    profiles[static_cast<Size>(t) * n_genes + g] += static_cast<Real>(values[k]);
                }
            }
        }
    }

    // Average - parallelize for large type counts
    if (static_cast<Size>(n_types) >= config::PARALLEL_THRESHOLD / 10) {
        scl::threading::parallel_for(Index(0), n_types, [&](Index t) {
            if (SCL_LIKELY(type_counts[t] > 0)) {
                Real inv = Real(1) / static_cast<Real>(type_counts[t]);
                Real* profile = profiles + static_cast<Size>(t) * n_genes;
                scl::vectorize::scale(Array<Real>(profile, static_cast<Size>(n_genes)), inv);
            }
        });
    } else {
        for (Index t = 0; t < n_types; ++t) {
            if (SCL_LIKELY(type_counts[t] > 0)) {
                Real inv = Real(1) / static_cast<Real>(type_counts[t]);
                Real* profile = profiles + static_cast<Size>(t) * n_genes;
                scl::vectorize::scale(Array<Real>(profile, static_cast<Size>(n_genes)), inv);
            }
        }
    }

    scl::memory::aligned_free(type_counts, SCL_ALIGNMENT);
}

// =============================================================================
// Marker Gene Score (scType-style)
// =============================================================================

template <typename T, bool IsCSR>
void marker_gene_score(
    const Sparse<T, IsCSR>& expression,
    const Index* const* marker_genes,      // Array of marker gene arrays per type
    const Index* marker_counts,            // Number of markers per type
    Index n_cells,
    Index n_genes,
    Index n_types,
    Real* scores,                          // n_cells x n_types output
    bool normalize = true
) {
    Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_types);
    scl::algo::zero(scores, total);

    // Parallel processing for large datasets
    if (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD && IsCSR) {
        // Parallel path with thread-local gene expression buffer
        Size workspace_size = static_cast<Size>(n_genes) * sizeof(Real);

        scl::threading::WorkspacePool<Real> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, static_cast<Size>(n_genes));

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells),
            [&](size_t c_idx, size_t thread_rank) {
            const Index c = static_cast<Index>(c_idx);
            Real* cell_expr = pool.get(thread_rank);

            // Extract cell expression
            scl::algo::zero(cell_expr, static_cast<Size>(n_genes));
            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    cell_expr[g] = static_cast<Real>(values[k]);
                }
            }

            // Score for each cell type
            Real* cell_scores = scores + static_cast<Size>(c) * n_types;

            for (Index t = 0; t < n_types; ++t) {
                const Index* markers = marker_genes[t];
                Index n_markers = marker_counts[t];

                if (SCL_UNLIKELY(n_markers == 0)) continue;

                Real sum = Real(0);
                Index valid = 0;

                // 4-way unrolled marker accumulation
                Index m = 0;
                for (; m + 4 <= n_markers; m += 4) {
                    Index g0 = markers[m], g1 = markers[m+1], g2 = markers[m+2], g3 = markers[m+3];
                    if (g0 >= 0 && g0 < n_genes) { sum += cell_expr[g0]; ++valid; }
                    if (g1 >= 0 && g1 < n_genes) { sum += cell_expr[g1]; ++valid; }
                    if (g2 >= 0 && g2 < n_genes) { sum += cell_expr[g2]; ++valid; }
                    if (g3 >= 0 && g3 < n_genes) { sum += cell_expr[g3]; ++valid; }
                }
                for (; m < n_markers; ++m) {
                    Index g = markers[m];
                    if (SCL_LIKELY(g >= 0 && g < n_genes)) {
                        sum += cell_expr[g];
                        ++valid;
                    }
                }

                if (SCL_LIKELY(valid > 0)) {
                    cell_scores[t] = sum / static_cast<Real>(valid);
                }
            }
        });
    } else {
        // Sequential path
        Real* cell_expr = nullptr;
        if (IsCSR) {
            cell_expr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
        }

        for (Index c = 0; c < n_cells; ++c) {
            // Extract cell expression
            if (IsCSR) {
                scl::algo::zero(cell_expr, static_cast<Size>(n_genes));
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

            // Score for each cell type
            for (Index t = 0; t < n_types; ++t) {
                const Index* markers = marker_genes[t];
                Index n_markers = marker_counts[t];

                if (SCL_UNLIKELY(n_markers == 0)) continue;

                Real sum = Real(0);
                Index valid = 0;

                // 4-way unrolled marker accumulation for CSR format
                Index m = 0;
                if (IsCSR) {
                    for (; m + 4 <= n_markers; m += 4) {
                        Index g0 = markers[m], g1 = markers[m+1], g2 = markers[m+2], g3 = markers[m+3];
                        if (SCL_LIKELY(g0 >= 0 && g0 < n_genes)) { sum += cell_expr[g0]; ++valid; }
                        if (SCL_LIKELY(g1 >= 0 && g1 < n_genes)) { sum += cell_expr[g1]; ++valid; }
                        if (SCL_LIKELY(g2 >= 0 && g2 < n_genes)) { sum += cell_expr[g2]; ++valid; }
                        if (SCL_LIKELY(g3 >= 0 && g3 < n_genes)) { sum += cell_expr[g3]; ++valid; }
                    }
                }

                // Scalar cleanup (or full loop for CSC)
                for (; m < n_markers; ++m) {
                    Index g = markers[m];
                    if (SCL_UNLIKELY(g < 0 || g >= n_genes)) continue;

                    Real expr;
                    if (IsCSR) {
                        expr = cell_expr[g];
                    } else {
                        expr = Real(0);
                        auto indices = expression.col_indices(g);
                        auto values = expression.col_values(g);
                        Index len = expression.col_length(g);
                        // Binary search for CSC format
                        const Index* found = scl::algo::lower_bound(indices.ptr, indices.ptr + len, c);
                        if (found != indices.ptr + len && *found == c) {
                            expr = static_cast<Real>(values[static_cast<Size>(found - indices.ptr)]);
                        }
                    }

                    sum += expr;
                    ++valid;
                }

                if (SCL_LIKELY(valid > 0)) {
                    scores[static_cast<Size>(c) * n_types + t] = sum / static_cast<Real>(valid);
                }
            }
        }

        if (IsCSR && cell_expr != nullptr) {
            scl::memory::aligned_free(cell_expr, SCL_ALIGNMENT);
        }
    }

    // Normalize scores per cell using optimized SIMD softmax
    if (SCL_LIKELY(normalize)) {
        for (Index c = 0; c < n_cells; ++c) {
            Real* cell_scores = scores + static_cast<Size>(c) * n_types;
            detail::softmax_normalize_simd(cell_scores, n_types);
        }
    }
}

// =============================================================================
// Assign from Marker Scores
// =============================================================================

inline void assign_from_marker_scores(
    const Real* scores,          // n_cells x n_types
    Index n_cells,
    Index n_types,
    Array<Index> labels,
    Array<Real> confidence
) {
    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n_cells),
                  "Annotation: labels buffer too small");
    SCL_CHECK_DIM(confidence.len >= static_cast<Size>(n_cells),
                  "Annotation: confidence buffer too small");

    for (Index c = 0; c < n_cells; ++c) {
        const Real* cell_scores = scores + static_cast<Size>(c) * n_types;

        Real max_score = cell_scores[0];
        Index max_type = 0;

        for (Index t = 1; t < n_types; ++t) {
            if (cell_scores[t] > max_score) {
                max_score = cell_scores[t];
                max_type = t;
            }
        }

        labels[c] = max_type;
        confidence[c] = max_score;
    }
}

// =============================================================================
// Consensus Annotation (Multiple Methods)
// =============================================================================

inline void consensus_annotation(
    const Index* const* predictions,   // Array of prediction arrays
    const Real* const* confidences,    // Array of confidence arrays (optional)
    Index n_methods,
    Index n_cells,
    Index n_types,
    Array<Index> consensus_labels,
    Array<Real> consensus_confidence
) {
    SCL_CHECK_DIM(consensus_labels.len >= static_cast<Size>(n_cells),
                  "Annotation: consensus_labels buffer too small");
    SCL_CHECK_DIM(consensus_confidence.len >= static_cast<Size>(n_cells),
                  "Annotation: consensus_confidence buffer too small");

    // Parallel processing for large datasets
    if (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD) {
        Size workspace_size = static_cast<Size>(n_types) * (sizeof(Index) + sizeof(Real));

        scl::threading::WorkspacePool<Byte> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, workspace_size);

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells),
            [&](size_t c_idx, size_t thread_rank) {
            const Index c = static_cast<Index>(c_idx);
            Byte* workspace = pool.get(thread_rank);

            Index* vote_counts = reinterpret_cast<Index*>(workspace);
            Real* weighted_votes = reinterpret_cast<Real*>(workspace + 
                static_cast<Size>(n_types) * sizeof(Index));

            scl::algo::zero(vote_counts, static_cast<Size>(n_types));
            scl::algo::zero(weighted_votes, static_cast<Size>(n_types));

            Real total_weight = Real(0);

            for (Index m = 0; m < n_methods; ++m) {
                Index pred = predictions[m][c];
                if (SCL_UNLIKELY(pred < 0 || pred >= n_types)) continue;

                ++vote_counts[pred];

                Real weight = (confidences != nullptr && confidences[m] != nullptr) ?
                    confidences[m][c] : Real(1);
                weighted_votes[pred] += weight;
                total_weight += weight;
            }

            // Find max weighted vote
            Real max_vote = Real(0);
            Index max_type = 0;

            for (Index t = 0; t < n_types; ++t) {
                if (weighted_votes[t] > max_vote) {
                    max_vote = weighted_votes[t];
                    max_type = t;
                }
            }

            consensus_labels[c] = max_type;
            consensus_confidence[c] = (total_weight > config::EPSILON) ?
                max_vote / total_weight : Real(0);
        });
    } else {
        // Sequential path
        Index* vote_counts = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);
        Real* weighted_votes = scl::memory::aligned_alloc<Real>(n_types, SCL_ALIGNMENT);

        for (Index c = 0; c < n_cells; ++c) {
            scl::algo::zero(vote_counts, static_cast<Size>(n_types));
            scl::algo::zero(weighted_votes, static_cast<Size>(n_types));

            Real total_weight = Real(0);

            for (Index m = 0; m < n_methods; ++m) {
                Index pred = predictions[m][c];
                if (SCL_UNLIKELY(pred < 0 || pred >= n_types)) continue;

                ++vote_counts[pred];

                Real weight = (confidences != nullptr && confidences[m] != nullptr) ?
                    confidences[m][c] : Real(1);
                weighted_votes[pred] += weight;
                total_weight += weight;
            }

            Real max_vote = Real(0);
            Index max_type = 0;

            for (Index t = 0; t < n_types; ++t) {
                if (weighted_votes[t] > max_vote) {
                    max_vote = weighted_votes[t];
                    max_type = t;
                }
            }

            consensus_labels[c] = max_type;
            consensus_confidence[c] = (total_weight > config::EPSILON) ?
                max_vote / total_weight : Real(0);
        }

        scl::memory::aligned_free(weighted_votes, SCL_ALIGNMENT);
        scl::memory::aligned_free(vote_counts, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Detect Novel Cell Types
// =============================================================================

template <typename T, bool IsCSR>
void detect_novel_types(
    const Sparse<T, IsCSR>& query_expression,
    Array<const Real> confidence_scores,
    Index n_query,
    Real threshold,
    Array<bool> is_novel
) {
    SCL_CHECK_DIM(is_novel.len >= static_cast<Size>(n_query),
                  "Annotation: is_novel buffer too small");

    // Parallelize for large datasets
    if (static_cast<Size>(n_query) >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_query), [&](size_t q_idx) {
            const Index q = static_cast<Index>(q_idx);
            is_novel[q] = confidence_scores[q] < threshold;
        });
    } else {
        for (Index q = 0; q < n_query; ++q) {
            is_novel[q] = confidence_scores[q] < threshold;
        }
    }
}

// =============================================================================
// Detect Novel Types with Distance Criterion
// =============================================================================

template <typename T, bool IsCSR>
void detect_novel_types_by_distance(
    const Sparse<T, IsCSR>& query_expression,
    const Real* reference_profiles,    // n_types x n_genes
    Array<const Index> assigned_labels,
    Index n_query,
    Index n_types,
    Index n_genes,
    Real distance_threshold,
    Array<bool> is_novel,
    Array<Real> distance_to_assigned   // Optional output
) {
    SCL_CHECK_DIM(is_novel.len >= static_cast<Size>(n_query),
                  "Annotation: is_novel buffer too small");

    // Parallel processing for large datasets
    if (static_cast<Size>(n_query) >= config::PARALLEL_THRESHOLD) {
        Size workspace_size = static_cast<Size>(n_genes) * sizeof(Real);

        scl::threading::WorkspacePool<Real> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, static_cast<Size>(n_genes));

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_query),
            [&](size_t q_idx, size_t thread_rank) {
            const Index q = static_cast<Index>(q_idx);
            Real* query_dense = pool.get(thread_rank);

            Index assigned = assigned_labels[q];

            if (SCL_UNLIKELY(assigned < 0 || assigned >= n_types)) {
                is_novel[q] = true;
                if (distance_to_assigned.ptr != nullptr) {
                    distance_to_assigned[q] = Real(1);
                }
                return;
            }

            // Extract query expression
            scl::algo::zero(query_dense, static_cast<Size>(n_genes));

            if (IsCSR) {
                auto indices = query_expression.row_indices(q);
                auto values = query_expression.row_values(q);
                Index len = query_expression.row_length(q);

                for (Index k = 0; k < len; ++k) {
                    Index g = indices[k];
                    if (SCL_LIKELY(g < n_genes)) {
                        query_dense[g] = static_cast<Real>(values[k]);
                    }
                }
            }

        // Compute distance to assigned profile (cosine distance) - SIMD optimized
        const Real* profile = reference_profiles + static_cast<Size>(assigned) * n_genes;

        // SIMD-friendly dot product and norm computation
        Real dot = scl::vectorize::dot(
            Array<const Real>(query_dense, static_cast<Size>(n_genes)),
            Array<const Real>(profile, static_cast<Size>(n_genes))
        );
        Real norm_q = std::sqrt(scl::vectorize::sum_squared(
            Array<const Real>(query_dense, static_cast<Size>(n_genes))
        ));
        Real norm_p = std::sqrt(scl::vectorize::sum_squared(
            Array<const Real>(profile, static_cast<Size>(n_genes))
        ));

        Real cosine_sim = Real(0);
        if (SCL_LIKELY(norm_q > config::EPSILON && norm_p > config::EPSILON)) {
            cosine_sim = dot / (norm_q * norm_p);
        }

        Real cosine_dist = Real(1) - cosine_sim;

        is_novel[q] = cosine_dist > distance_threshold;

        if (distance_to_assigned.ptr != nullptr) {
            distance_to_assigned[q] = cosine_dist;
        }
    }

    scl::memory::aligned_free(query_dense, SCL_ALIGNMENT);
}

// =============================================================================
// Cluster-Based Novel Type Detection
// =============================================================================

inline void cluster_novel_cells(
    Array<const bool> is_novel,
    Array<const Index> cluster_labels,  // Pre-computed clusters
    Index n_cells,
    Index n_clusters,
    Real min_novel_fraction,
    Array<bool> is_novel_cluster        // n_clusters output
) {
    SCL_CHECK_DIM(is_novel_cluster.len >= static_cast<Size>(n_clusters),
                  "Annotation: is_novel_cluster buffer too small");

    Index* cluster_sizes = scl::memory::aligned_alloc<Index>(n_clusters, SCL_ALIGNMENT);
    Index* novel_counts = scl::memory::aligned_alloc<Index>(n_clusters, SCL_ALIGNMENT);

    scl::algo::zero(cluster_sizes, static_cast<Size>(n_clusters));
    scl::algo::zero(novel_counts, static_cast<Size>(n_clusters));

    for (Index c = 0; c < n_cells; ++c) {
        Index cl = cluster_labels[c];
        if (cl >= 0 && cl < n_clusters) {
            ++cluster_sizes[cl];
            if (is_novel[c]) {
                ++novel_counts[cl];
            }
        }
    }

    for (Index cl = 0; cl < n_clusters; ++cl) {
        Real fraction = (cluster_sizes[cl] > 0) ?
            static_cast<Real>(novel_counts[cl]) / static_cast<Real>(cluster_sizes[cl]) : Real(0);
        is_novel_cluster[cl] = fraction >= min_novel_fraction;
    }

    scl::memory::aligned_free(novel_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(cluster_sizes, SCL_ALIGNMENT);
}

// =============================================================================
// Label Propagation from Neighbors
// =============================================================================

template <typename T, bool IsCSR>
void label_propagation(
    const Sparse<T, IsCSR>& neighbor_graph,
    Array<const Index> initial_labels,   // -1 for unlabeled
    Index n_cells,
    Index n_types,
    Index max_iter,
    Array<Index> final_labels,
    Array<Real> label_confidence
) {
    SCL_CHECK_DIM(final_labels.len >= static_cast<Size>(n_cells),
                  "Annotation: final_labels buffer too small");
    SCL_CHECK_DIM(label_confidence.len >= static_cast<Size>(n_cells),
                  "Annotation: label_confidence buffer too small");

    // Initialize with known labels
    for (Index c = 0; c < n_cells; ++c) {
        final_labels[c] = initial_labels[c];
        label_confidence[c] = (initial_labels[c] >= 0) ? Real(1) : Real(0);
    }

    Real* type_scores = scl::memory::aligned_alloc<Real>(n_types, SCL_ALIGNMENT);
    Index* prev_labels = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Copy current labels
        bool changed = false;
        for (Index c = 0; c < n_cells; ++c) {
            prev_labels[c] = final_labels[c];
        }

        // Update unlabeled cells
        for (Index c = 0; c < n_cells; ++c) {
            if (initial_labels[c] >= 0) continue;  // Skip labeled cells

            auto neighbors = neighbor_graph.primary_indices(c);
            auto weights = neighbor_graph.primary_values(c);
            Index n_neighbors = neighbor_graph.primary_length(c);

            if (n_neighbors == 0) continue;

            scl::algo::zero(type_scores, static_cast<Size>(n_types));

            Real total_weight = Real(0);
            for (Index k = 0; k < n_neighbors; ++k) {
                Index j = neighbors[k];
                Real w = static_cast<Real>(weights[k]);

                Index lbl = prev_labels[j];
                if (lbl >= 0 && lbl < n_types) {
                    type_scores[lbl] += w;
                    total_weight += w;
                }
            }

            if (total_weight > config::EPSILON) {
                Real max_score = Real(0);
                Index max_type = -1;

                for (Index t = 0; t < n_types; ++t) {
                    if (type_scores[t] > max_score) {
                        max_score = type_scores[t];
                        max_type = t;
                    }
                }

                if (max_type >= 0 && final_labels[c] != max_type) {
                    final_labels[c] = max_type;
                    label_confidence[c] = max_score / total_weight;
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }

    scl::memory::aligned_free(prev_labels, SCL_ALIGNMENT);
    scl::memory::aligned_free(type_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Annotation Quality Metrics
// =============================================================================

inline void annotation_quality_metrics(
    Array<const Index> predicted_labels,
    Array<const Index> true_labels,  // Ground truth
    Index n_cells,
    Index n_types,
    Real& accuracy,
    Real& macro_f1,
    Real* per_class_f1   // Optional: n_types output
) {
    // Count predictions
    Index correct = 0;

    Index* true_positives = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);
    Index* false_positives = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);
    Index* false_negatives = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);

    scl::algo::zero(true_positives, static_cast<Size>(n_types));
    scl::algo::zero(false_positives, static_cast<Size>(n_types));
    scl::algo::zero(false_negatives, static_cast<Size>(n_types));

    for (Index c = 0; c < n_cells; ++c) {
        Index pred = predicted_labels[c];
        Index true_lbl = true_labels[c];

        if (pred == true_lbl) {
            ++correct;
            if (pred >= 0 && pred < n_types) {
                ++true_positives[pred];
            }
        } else {
            if (pred >= 0 && pred < n_types) {
                ++false_positives[pred];
            }
            if (true_lbl >= 0 && true_lbl < n_types) {
                ++false_negatives[true_lbl];
            }
        }
    }

    accuracy = (n_cells > 0) ?
        static_cast<Real>(correct) / static_cast<Real>(n_cells) : Real(0);

    // Compute F1 scores
    Real sum_f1 = Real(0);
    Index valid_classes = 0;

    for (Index t = 0; t < n_types; ++t) {
        Index tp = true_positives[t];
        Index fp = false_positives[t];
        Index fn = false_negatives[t];

        Real precision = (tp + fp > 0) ?
            static_cast<Real>(tp) / static_cast<Real>(tp + fp) : Real(0);
        Real recall = (tp + fn > 0) ?
            static_cast<Real>(tp) / static_cast<Real>(tp + fn) : Real(0);

        Real f1 = (precision + recall > config::EPSILON) ?
            Real(2) * precision * recall / (precision + recall) : Real(0);

        if (per_class_f1 != nullptr) {
            per_class_f1[t] = f1;
        }

        if (tp + fp + fn > 0) {
            sum_f1 += f1;
            ++valid_classes;
        }
    }

    macro_f1 = (valid_classes > 0) ?
        sum_f1 / static_cast<Real>(valid_classes) : Real(0);

    scl::memory::aligned_free(false_negatives, SCL_ALIGNMENT);
    scl::memory::aligned_free(false_positives, SCL_ALIGNMENT);
    scl::memory::aligned_free(true_positives, SCL_ALIGNMENT);
}

// =============================================================================
// Confusion Matrix
// =============================================================================

inline void confusion_matrix(
    Array<const Index> predicted_labels,
    Array<const Index> true_labels,
    Index n_cells,
    Index n_types,
    Index* confusion       // n_types x n_types output (true x predicted)
) {
    Size total = static_cast<Size>(n_types) * static_cast<Size>(n_types);
    scl::algo::zero(confusion, total);

    for (Index c = 0; c < n_cells; ++c) {
        Index true_lbl = true_labels[c];
        Index pred = predicted_labels[c];

        if (true_lbl >= 0 && true_lbl < n_types &&
            pred >= 0 && pred < n_types) {
            ++confusion[static_cast<Size>(true_lbl) * n_types + pred];
        }
    }
}

// =============================================================================
// Cell Type Marker Expression
// =============================================================================

template <typename T, bool IsCSR>
void cell_type_marker_expression(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    const Index* marker_genes,
    Index n_markers,
    Index n_cells,
    Index n_genes,
    Index n_types,
    Real* mean_expression,    // n_types x n_markers
    Real* pct_expressed       // n_types x n_markers
) {
    Size total = static_cast<Size>(n_types) * static_cast<Size>(n_markers);
    scl::algo::zero(mean_expression, total);
    scl::algo::zero(pct_expressed, total);

    Index* type_counts = scl::memory::aligned_alloc<Index>(n_types, SCL_ALIGNMENT);
    scl::algo::zero(type_counts, static_cast<Size>(n_types));

    // Count cells per type
    for (Index c = 0; c < n_cells; ++c) {
        Index t = labels[c];
        if (t >= 0 && t < n_types) {
            ++type_counts[t];
        }
    }

    // Accumulate marker expression
    Index* expressing = scl::memory::aligned_alloc<Index>(total, SCL_ALIGNMENT);
    scl::algo::zero(expressing, total);

    for (Index m = 0; m < n_markers; ++m) {
        Index g = marker_genes[m];
        if (g < 0 || g >= n_genes) continue;

        if (IsCSR) {
            for (Index c = 0; c < n_cells; ++c) {
                Index t = labels[c];
                if (SCL_UNLIKELY(t < 0 || t >= n_types)) continue;

                auto indices = expression.row_indices(c);
                auto values = expression.row_values(c);
                Index len = expression.row_length(c);

                // Binary search for gene in sorted row indices
                Index lo = 0, hi = len;
                while (lo < hi) {
                    Index mid = lo + (hi - lo) / 2;
                    if (indices[mid] < g) lo = mid + 1;
                    else hi = mid;
                }

                if (lo < len && indices[lo] == g) {
                    Real v = static_cast<Real>(values[lo]);
                    mean_expression[static_cast<Size>(t) * n_markers + m] += v;
                    if (SCL_LIKELY(v > Real(0))) {
                        ++expressing[static_cast<Size>(t) * n_markers + m];
                    }
                }
            }
        } else {
            auto indices = expression.col_indices(g);
            auto values = expression.col_values(g);
            Index len = expression.col_length(g);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c >= n_cells) continue;

                Index t = labels[c];
                if (t >= 0 && t < n_types) {
                    Real v = static_cast<Real>(values[k]);
                    mean_expression[static_cast<Size>(t) * n_markers + m] += v;
                    if (v > Real(0)) {
                        ++expressing[static_cast<Size>(t) * n_markers + m];
                    }
                }
            }
        }
    }

    // Compute averages and percentages
    for (Index t = 0; t < n_types; ++t) {
        if (type_counts[t] == 0) continue;

        Real inv = Real(1) / static_cast<Real>(type_counts[t]);
        for (Index m = 0; m < n_markers; ++m) {
            Size idx = static_cast<Size>(t) * n_markers + m;
            mean_expression[idx] *= inv;
            pct_expressed[idx] = static_cast<Real>(expressing[idx]) * inv;
        }
    }

    scl::memory::aligned_free(expressing, SCL_ALIGNMENT);
    scl::memory::aligned_free(type_counts, SCL_ALIGNMENT);
}

// =============================================================================
// Fine-Grained Annotation (Hierarchical)
// =============================================================================

template <typename T, bool IsCSR>
void hierarchical_annotation(
    const Sparse<T, IsCSR>& expression,
    const Real* coarse_profiles,      // n_coarse x n_genes
    const Real* fine_profiles,        // n_fine x n_genes
    const Index* coarse_to_fine_map,  // For each fine type, its coarse parent
    Index n_cells,
    Index n_genes,
    Index n_coarse,
    Index n_fine,
    Array<Index> coarse_labels,
    Array<Index> fine_labels,
    Array<Real> coarse_confidence,
    Array<Real> fine_confidence
) {
    SCL_CHECK_DIM(coarse_labels.len >= static_cast<Size>(n_cells),
                  "Annotation: coarse_labels buffer too small");
    SCL_CHECK_DIM(fine_labels.len >= static_cast<Size>(n_cells),
                  "Annotation: fine_labels buffer too small");

    Real* cell_expr = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

    for (Index c = 0; c < n_cells; ++c) {
        // Extract cell expression
        scl::algo::zero(cell_expr, static_cast<Size>(n_genes));

        if (IsCSR) {
            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    cell_expr[g] = static_cast<Real>(values[k]);
                }
            }
        }

        // Compute correlation with coarse profiles
        Real max_coarse_corr = -Real(2);
        Index max_coarse = 0;

        for (Index t = 0; t < n_coarse; ++t) {
            const Real* profile = coarse_profiles + static_cast<Size>(t) * n_genes;

            Real mean_c = Real(0), mean_p = Real(0);
            for (Index g = 0; g < n_genes; ++g) {
                mean_c += cell_expr[g];
                mean_p += profile[g];
            }
            mean_c /= static_cast<Real>(n_genes);
            mean_p /= static_cast<Real>(n_genes);

            Real sum_cp = Real(0), sum_cc = Real(0), sum_pp = Real(0);
            for (Index g = 0; g < n_genes; ++g) {
                Real dc = cell_expr[g] - mean_c;
                Real dp = profile[g] - mean_p;
                sum_cp += dc * dp;
                sum_cc += dc * dc;
                sum_pp += dp * dp;
            }

            Real denom = std::sqrt(sum_cc * sum_pp);
            Real corr = (denom > config::EPSILON) ? sum_cp / denom : Real(0);

            if (corr > max_coarse_corr) {
                max_coarse_corr = corr;
                max_coarse = t;
            }
        }

        coarse_labels[c] = max_coarse;
        coarse_confidence[c] = (max_coarse_corr + Real(1)) / Real(2);  // Normalize to [0,1]

        // Refine within coarse type
        Real max_fine_corr = -Real(2);
        Index max_fine = 0;

        for (Index f = 0; f < n_fine; ++f) {
            if (coarse_to_fine_map[f] != max_coarse) continue;

            const Real* profile = fine_profiles + static_cast<Size>(f) * n_genes;

            Real mean_c = Real(0), mean_p = Real(0);
            for (Index g = 0; g < n_genes; ++g) {
                mean_c += cell_expr[g];
                mean_p += profile[g];
            }
            mean_c /= static_cast<Real>(n_genes);
            mean_p /= static_cast<Real>(n_genes);

            Real sum_cp = Real(0), sum_cc = Real(0), sum_pp = Real(0);
            for (Index g = 0; g < n_genes; ++g) {
                Real dc = cell_expr[g] - mean_c;
                Real dp = profile[g] - mean_p;
                sum_cp += dc * dp;
                sum_cc += dc * dc;
                sum_pp += dp * dp;
            }

            Real denom = std::sqrt(sum_cc * sum_pp);
            Real corr = (denom > config::EPSILON) ? sum_cp / denom : Real(0);

            if (corr > max_fine_corr) {
                max_fine_corr = corr;
                max_fine = f;
            }
        }

        fine_labels[c] = max_fine;
        fine_confidence[c] = (max_fine_corr + Real(1)) / Real(2);
    }

    scl::memory::aligned_free(cell_expr, SCL_ALIGNMENT);
}

// =============================================================================
// Entropy-Based Annotation Uncertainty
// =============================================================================

inline void annotation_entropy(
    const Real* type_probabilities,  // n_cells x n_types
    Index n_cells,
    Index n_types,
    Array<Real> entropy
) {
    SCL_CHECK_DIM(entropy.len >= static_cast<Size>(n_cells),
                  "Annotation: entropy buffer too small");

    Real max_entropy = std::log(static_cast<Real>(n_types));

    // Parallelize for large datasets
    if (static_cast<Size>(n_cells) >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t c_idx) {
            const Index c = static_cast<Index>(c_idx);
            const Real* probs = type_probabilities + static_cast<Size>(c) * n_types;

            Real h = Real(0);
            for (Index t = 0; t < n_types; ++t) {
                if (SCL_LIKELY(probs[t] > config::EPSILON)) {
                    h -= probs[t] * std::log(probs[t]);
                }
            }

            // Normalize to [0, 1]
            entropy[c] = (max_entropy > config::EPSILON) ? h / max_entropy : Real(0);
        });
    } else {
        for (Index c = 0; c < n_cells; ++c) {
            const Real* probs = type_probabilities + static_cast<Size>(c) * n_types;

            Real h = Real(0);
            for (Index t = 0; t < n_types; ++t) {
                if (SCL_LIKELY(probs[t] > config::EPSILON)) {
                    h -= probs[t] * std::log(probs[t]);
                }
            }

            entropy[c] = (max_entropy > config::EPSILON) ? h / max_entropy : Real(0);
        }
    }
}

// =============================================================================
// Differential Marker Expression
// =============================================================================

template <typename T, bool IsCSR>
void differential_markers(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> labels,
    Index target_type,
    Index n_cells,
    Index n_genes,
    Index n_types,
    Array<Real> log_fold_change,
    Array<Real> pct_in,
    Array<Real> pct_out
) {
    SCL_CHECK_DIM(log_fold_change.len >= static_cast<Size>(n_genes),
                  "Annotation: log_fold_change buffer too small");

    Real* sum_in = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Real* sum_out = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);
    Index* count_in_expr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
    Index* count_out_expr = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);

    scl::algo::zero(sum_in, static_cast<Size>(n_genes));
    scl::algo::zero(sum_out, static_cast<Size>(n_genes));
    scl::algo::zero(count_in_expr, static_cast<Size>(n_genes));
    scl::algo::zero(count_out_expr, static_cast<Size>(n_genes));

    Index n_in = 0, n_out = 0;

    // Accumulate expression
    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            bool is_target = (labels[c] == target_type);
            if (is_target) ++n_in;
            else ++n_out;

            auto indices = expression.row_indices(c);
            auto values = expression.row_values(c);
            Index len = expression.row_length(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g >= n_genes) continue;

                Real v = static_cast<Real>(values[k]);
                if (is_target) {
                    sum_in[g] += v;
                    if (v > Real(0)) ++count_in_expr[g];
                } else {
                    sum_out[g] += v;
                    if (v > Real(0)) ++count_out_expr[g];
                }
            }
        }
    } else {
        // Count cells
        for (Index c = 0; c < n_cells; ++c) {
            if (labels[c] == target_type) ++n_in;
            else ++n_out;
        }

        for (Index g = 0; g < n_genes; ++g) {
            auto indices = expression.col_indices(g);
            auto values = expression.col_values(g);
            Index len = expression.col_length(g);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c >= n_cells) continue;

                Real v = static_cast<Real>(values[k]);
                bool is_target = (labels[c] == target_type);

                if (is_target) {
                    sum_in[g] += v;
                    if (v > Real(0)) ++count_in_expr[g];
                } else {
                    sum_out[g] += v;
                    if (v > Real(0)) ++count_out_expr[g];
                }
            }
        }
    }

    // Compute statistics
    for (Index g = 0; g < n_genes; ++g) {
        Real mean_in = (n_in > 0) ? sum_in[g] / static_cast<Real>(n_in) : Real(0);
        Real mean_out = (n_out > 0) ? sum_out[g] / static_cast<Real>(n_out) : Real(0);

        // Log fold change (add pseudocount)
        log_fold_change[g] = std::log2((mean_in + Real(1)) / (mean_out + Real(1)));

        if (pct_in.ptr != nullptr) {
            pct_in[g] = (n_in > 0) ?
                static_cast<Real>(count_in_expr[g]) / static_cast<Real>(n_in) : Real(0);
        }

        if (pct_out.ptr != nullptr) {
            pct_out[g] = (n_out > 0) ?
                static_cast<Real>(count_out_expr[g]) / static_cast<Real>(n_out) : Real(0);
        }
    }

    scl::memory::aligned_free(count_out_expr, SCL_ALIGNMENT);
    scl::memory::aligned_free(count_in_expr, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_out, SCL_ALIGNMENT);
    scl::memory::aligned_free(sum_in, SCL_ALIGNMENT);
}

// =============================================================================
// Top Markers per Cell Type
// =============================================================================

inline void top_markers_per_type(
    const Real* log_fold_changes,    // n_types x n_genes (precomputed)
    Index n_types,
    Index n_genes,
    Index n_top,
    Index* top_markers,              // n_types x n_top output
    Real* top_lfc                    // n_types x n_top output
) {
    // Parallelize over types for large datasets
    if (static_cast<Size>(n_types) >= config::PARALLEL_THRESHOLD / 10) {
        Size workspace_size = static_cast<Size>(n_genes) * (sizeof(Index) + sizeof(Real));

        scl::threading::WorkspacePool<Byte> pool;
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        pool.init(n_threads, workspace_size);

        scl::threading::parallel_for(Size(0), static_cast<Size>(n_types),
            [&](size_t t_idx, size_t thread_rank) {
            const Index t = static_cast<Index>(t_idx);
            Byte* workspace = pool.get(thread_rank);

            const Real* lfc = log_fold_changes + static_cast<Size>(t) * n_genes;
            Index* out_markers = top_markers + static_cast<Size>(t) * n_top;
            Real* out_lfc = top_lfc + static_cast<Size>(t) * n_top;

            Index* sorted_idx = reinterpret_cast<Index*>(workspace);
            Real* sorted_lfc = reinterpret_cast<Real*>(workspace + 
                static_cast<Size>(n_genes) * sizeof(Index));

            // Initialize indices
            for (Index g = 0; g < n_genes; ++g) {
                sorted_idx[g] = g;
                sorted_lfc[g] = lfc[g];
            }

            // Use partial_sort for O(n log k) instead of O(n^2) insertion sort
            // Sort in descending order (largest first)
            scl::algo::partial_sort(sorted_idx, static_cast<Size>(n_genes), 
                static_cast<Size>(n_top), [&](Index a, Index b) {
                return sorted_lfc[a] > sorted_lfc[b];  // Descending
            });

            // Copy top markers
            Index n_copy = scl::algo::min2(n_top, n_genes);
            for (Index i = 0; i < n_copy; ++i) {
                out_markers[i] = sorted_idx[i];
                out_lfc[i] = sorted_lfc[sorted_idx[i]];
            }

            // Fill remaining with -1
            for (Index i = n_copy; i < n_top; ++i) {
                out_markers[i] = -1;
                out_lfc[i] = Real(0);
            }
        });
    } else {
        // Sequential path
        Index* sorted_idx = scl::memory::aligned_alloc<Index>(n_genes, SCL_ALIGNMENT);
        Real* sorted_lfc = scl::memory::aligned_alloc<Real>(n_genes, SCL_ALIGNMENT);

        for (Index t = 0; t < n_types; ++t) {
            const Real* lfc = log_fold_changes + static_cast<Size>(t) * n_genes;
            Index* out_markers = top_markers + static_cast<Size>(t) * n_top;
            Real* out_lfc = top_lfc + static_cast<Size>(t) * n_top;

            // Initialize
            for (Index g = 0; g < n_genes; ++g) {
                sorted_idx[g] = g;
                sorted_lfc[g] = lfc[g];
            }

            // Use partial_sort for O(n log k) instead of O(n^2) insertion sort
            scl::algo::partial_sort(sorted_idx, static_cast<Size>(n_genes), 
                static_cast<Size>(n_top), [&](Index a, Index b) {
                return sorted_lfc[a] > sorted_lfc[b];  // Descending
            });

            // Copy top markers
            Index n_copy = scl::algo::min2(n_top, n_genes);
            for (Index i = 0; i < n_copy; ++i) {
                out_markers[i] = sorted_idx[i];
                out_lfc[i] = sorted_lfc[sorted_idx[i]];
            }

            // Fill remaining with -1
            for (Index i = n_copy; i < n_top; ++i) {
                out_markers[i] = -1;
                out_lfc[i] = Real(0);
            }
        }

        scl::memory::aligned_free(sorted_lfc, SCL_ALIGNMENT);
        scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
    }
}

} // namespace scl::kernel::annotation
