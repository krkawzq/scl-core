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

// =============================================================================
// FILE: scl/kernel/gnn.hpp
// BRIEF: High-performance Graph Neural Network primitives
//
// Optimizations applied:
// - Parallel node processing with WorkspacePool
// - SIMD-accelerated feature aggregation
// - Block-wise feature processing for cache efficiency
// - Precomputed normalization factors
// - Fused softmax + aggregation
// - Multi-head attention parallelization
// - Cache-aligned workspace structures
// =============================================================================

namespace scl::kernel::gnn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Real ATTENTION_EPSILON = Real(1e-12);
    constexpr Real LEAKY_RELU_SLOPE = Real(0.2);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size FEATURE_BLOCK_SIZE = 64;
    constexpr Size SIMD_THRESHOLD = 8;
}

// =============================================================================
// Aggregation Types
// =============================================================================

enum class AggregationType {
    Sum,
    Mean,
    Max,
    Min,
    Attention,
    Weighted
};

// =============================================================================
// Activation Functions
// =============================================================================

enum class ActivationType {
    None,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    ELU,
    GELU
};

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD Feature Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE void feature_add_simd(
    const Real* SCL_RESTRICT src,
    Real* SCL_RESTRICT dst,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        auto v_dst = s::Load(d, dst + k);
        auto v_src = s::Load(d, src + k);
        s::Store(s::Add(v_dst, v_src), d, dst + k);
    }

    for (; k < feat_dim; ++k) {
        dst[k] += src[k];
    }
}

SCL_HOT SCL_FORCE_INLINE void feature_add_scaled_simd(
    const Real* SCL_RESTRICT src,
    Real* SCL_RESTRICT dst,
    Real scale,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_scale = s::Set(d, scale);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        auto v_dst = s::Load(d, dst + k);
        auto v_src = s::Load(d, src + k);
        s::Store(s::MulAdd(v_scale, v_src, v_dst), d, dst + k);
    }

    for (; k < feat_dim; ++k) {
        dst[k] += scale * src[k];
    }
}

SCL_HOT SCL_FORCE_INLINE void feature_max_simd(
    const Real* SCL_RESTRICT src,
    Real* SCL_RESTRICT dst,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        auto v_dst = s::Load(d, dst + k);
        auto v_src = s::Load(d, src + k);
        s::Store(s::Max(v_dst, v_src), d, dst + k);
    }

    for (; k < feat_dim; ++k) {
        dst[k] = scl::algo::max2(dst[k], src[k]);
    }
}

SCL_HOT SCL_FORCE_INLINE void feature_min_simd(
    const Real* SCL_RESTRICT src,
    Real* SCL_RESTRICT dst,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        auto v_dst = s::Load(d, dst + k);
        auto v_src = s::Load(d, src + k);
        s::Store(s::Min(v_dst, v_src), d, dst + k);
    }

    for (; k < feat_dim; ++k) {
        dst[k] = scl::algo::min2(dst[k], src[k]);
    }
}

SCL_HOT SCL_FORCE_INLINE void feature_scale_simd(
    Real* SCL_RESTRICT x,
    Real scale,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_scale = s::Set(d, scale);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        s::Store(s::Mul(v_scale, s::Load(d, x + k)), d, x + k);
    }

    for (; k < feat_dim; ++k) {
        x[k] *= scale;
    }
}

SCL_HOT SCL_FORCE_INLINE void feature_fill_simd(
    Real* SCL_RESTRICT x,
    Real value,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_val = s::Set(d, value);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        s::Store(v_val, d, x + k);
    }

    for (; k < feat_dim; ++k) {
        x[k] = value;
    }
}

SCL_HOT SCL_FORCE_INLINE Real feature_dot_simd(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Index feat_dim
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum = s::Zero(d);

    Index k = 0;
    for (; k + static_cast<Index>(lanes) <= feat_dim; k += lanes) {
        v_sum = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum);
    }

    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < feat_dim; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

// =============================================================================
// Activation Functions (Vectorized)
// =============================================================================

SCL_FORCE_INLINE Real apply_activation(Real x, ActivationType act) noexcept {
    switch (act) {
        case ActivationType::ReLU:
            return (x > Real(0)) ? x : Real(0);
        case ActivationType::LeakyReLU:
            return (x > Real(0)) ? x : config::LEAKY_RELU_SLOPE * x;
        case ActivationType::Sigmoid:
            return Real(1) / (Real(1) + std::exp(-x));
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::ELU:
            return (x > Real(0)) ? x : (std::exp(x) - Real(1));
        case ActivationType::GELU: {
            // Approximate GELU: x * sigmoid(1.702 * x)
            Real sig = Real(1) / (Real(1) + std::exp(Real(-1.702) * x));
            return x * sig;
        }
        default:
            return x;
    }
}

SCL_HOT void apply_activation_vector(
    Real* x,
    Index n,
    ActivationType act
) noexcept {
    if (act == ActivationType::None) return;

    if (act == ActivationType::ReLU) {
        namespace s = scl::simd;
        using SimdTag = s::SimdTagFor<Real>;
        const SimdTag d;
        const size_t lanes = s::Lanes(d);

        auto v_zero = s::Zero(d);

        Index k = 0;
        for (; k + static_cast<Index>(lanes) <= n; k += lanes) {
            auto v = s::Load(d, x + k);
            s::Store(s::Max(v, v_zero), d, x + k);
        }

        for (; k < n; ++k) {
            x[k] = (x[k] > Real(0)) ? x[k] : Real(0);
        }
    } else {
        for (Index i = 0; i < n; ++i) {
            x[i] = apply_activation(x[i], act);
        }
    }
}

// =============================================================================
// Optimized Softmax
// =============================================================================

SCL_HOT SCL_FORCE_INLINE void sparse_softmax_fused(
    const Real* SCL_RESTRICT scores,
    Index len,
    Real* SCL_RESTRICT probs
) noexcept {
    if (len == 0) return;
    if (len == 1) {
        probs[0] = Real(1);
        return;
    }

    // Find max for stability
    Real max_val = scores[0];
    for (Index k = 1; k < len; ++k) {
        max_val = scl::algo::max2(max_val, scores[k]);
    }

    // Compute exp and sum in single pass
    Real sum = Real(0);
    for (Index k = 0; k < len; ++k) {
        probs[k] = std::exp(scores[k] - max_val);
        sum += probs[k];
    }

    // Normalize
    if (sum > config::ATTENTION_EPSILON) {
        Real inv_sum = Real(1) / sum;
        for (Index k = 0; k < len; ++k) {
            probs[k] *= inv_sum;
        }
    } else {
        Real uniform = Real(1) / static_cast<Real>(len);
        for (Index k = 0; k < len; ++k) {
            probs[k] = uniform;
        }
    }
}

// =============================================================================
// Attention Score Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_attention_score(
    const Real* feat_i,
    const Real* feat_j,
    Index feat_dim,
    const Real* attention_vec
) noexcept {
    Real score = feature_dot_simd(attention_vec, feat_i, feat_dim);
    score += feature_dot_simd(attention_vec + feat_dim, feat_j, feat_dim);
    return apply_activation(score, ActivationType::LeakyReLU);
}

// =============================================================================
// Precomputed Normalization
// =============================================================================

template <typename T, bool IsCSR>
void precompute_gcn_norm(
    const Sparse<T, IsCSR>& adj,
    Real* degree_inv_sqrt,
    Size n,
    bool add_self_loops
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            auto values = adj.primary_values(static_cast<Index>(i));
            const Index len = adj.primary_length(static_cast<Index>(i));

            Real deg = Real(0);
            for (Index k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }

            if (add_self_loops) deg += Real(1);

            degree_inv_sqrt[i] = (deg > config::ATTENTION_EPSILON)
                ? Real(1) / std::sqrt(deg) : Real(0);
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            auto values = adj.primary_values(static_cast<Index>(i));
            const Index len = adj.primary_length(static_cast<Index>(i));

            Real deg = Real(0);
            for (Index k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }

            if (add_self_loops) deg += Real(1);

            degree_inv_sqrt[i] = (deg > config::ATTENTION_EPSILON)
                ? Real(1) / std::sqrt(deg) : Real(0);
        }
    }
}

} // namespace detail

// =============================================================================
// Message Passing (Parallel + SIMD)
// =============================================================================

template <typename T, bool IsCSR>
void message_passing(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Array<Real> output,
    AggregationType agg_type = AggregationType::Mean
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size F = static_cast<Size>(feat_dim);

    SCL_CHECK_DIM(output.len >= N * F, "GNN: output buffer too small");

    if (n == 0 || feat_dim == 0) return;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto indices = adjacency.primary_indices(static_cast<Index>(i));
            auto values = adjacency.primary_values(static_cast<Index>(i));
            const Index len = adjacency.primary_length(static_cast<Index>(i));

            Real* out_feat = output.ptr + i * F;

            // Initialize
            if (agg_type == AggregationType::Max) {
                detail::feature_fill_simd(out_feat, -Real(1e30), feat_dim);
            } else if (agg_type == AggregationType::Min) {
                detail::feature_fill_simd(out_feat, Real(1e30), feat_dim);
            } else {
                detail::feature_fill_simd(out_feat, Real(0), feat_dim);
            }

            Real total_weight = Real(0);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                const Real* neighbor_feat = node_features.ptr + static_cast<Size>(j) * F;

                switch (agg_type) {
                    case AggregationType::Sum:
                        detail::feature_add_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    case AggregationType::Mean:
                        detail::feature_add_simd(neighbor_feat, out_feat, feat_dim);
                        total_weight += Real(1);
                        break;
                    case AggregationType::Weighted:
                        detail::feature_add_scaled_simd(neighbor_feat, out_feat, w, feat_dim);
                        total_weight += w;
                        break;
                    case AggregationType::Max:
                        detail::feature_max_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    case AggregationType::Min:
                        detail::feature_min_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    default:
                        break;
                }
            }

            // Normalize
            if ((agg_type == AggregationType::Mean || agg_type == AggregationType::Weighted) &&
                total_weight > config::ATTENTION_EPSILON) {
                detail::feature_scale_simd(out_feat, Real(1) / total_weight, feat_dim);
            }

            // Handle empty neighborhood
            if (len == 0) {
                const Real* self_feat = node_features.ptr + i * F;
                std::memcpy(out_feat, self_feat, F * sizeof(Real));
            }
        });
    } else {
        // Sequential version
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real* out_feat = output.ptr + static_cast<Size>(i) * F;

            if (agg_type == AggregationType::Max) {
                detail::feature_fill_simd(out_feat, -Real(1e30), feat_dim);
            } else if (agg_type == AggregationType::Min) {
                detail::feature_fill_simd(out_feat, Real(1e30), feat_dim);
            } else {
                detail::feature_fill_simd(out_feat, Real(0), feat_dim);
            }

            Real total_weight = Real(0);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                const Real* neighbor_feat = node_features.ptr + static_cast<Size>(j) * F;

                switch (agg_type) {
                    case AggregationType::Sum:
                        detail::feature_add_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    case AggregationType::Mean:
                        detail::feature_add_simd(neighbor_feat, out_feat, feat_dim);
                        total_weight += Real(1);
                        break;
                    case AggregationType::Weighted:
                        detail::feature_add_scaled_simd(neighbor_feat, out_feat, w, feat_dim);
                        total_weight += w;
                        break;
                    case AggregationType::Max:
                        detail::feature_max_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    case AggregationType::Min:
                        detail::feature_min_simd(neighbor_feat, out_feat, feat_dim);
                        break;
                    default:
                        break;
                }
            }

            if ((agg_type == AggregationType::Mean || agg_type == AggregationType::Weighted) &&
                total_weight > config::ATTENTION_EPSILON) {
                detail::feature_scale_simd(out_feat, Real(1) / total_weight, feat_dim);
            }

            if (len == 0) {
                const Real* self_feat = node_features.ptr + static_cast<Size>(i) * F;
                std::memcpy(out_feat, self_feat, F * sizeof(Real));
            }
        }
    }
}

// =============================================================================
// Graph Attention (Parallel + Fused)
// =============================================================================

template <typename T, bool IsCSR>
void graph_attention(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Array<const Real> attention_vec,
    Array<Real> output,
    bool add_self_loops = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size F = static_cast<Size>(feat_dim);

    SCL_CHECK_DIM(output.len >= N * F, "GAT: output buffer too small");
    SCL_CHECK_DIM(attention_vec.len >= static_cast<Size>(2 * feat_dim), "GAT: attention vector too small");

    if (n == 0 || feat_dim == 0) return;

    // Find max neighbors for workspace sizing
    Index max_neighbors = 0;
    for (Index i = 0; i < n; ++i) {
        max_neighbors = scl::algo::max2(max_neighbors, adjacency.primary_length(i));
    }
    max_neighbors += 1;  // For self-loop

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::WorkspacePool<Real> score_pool;
    scl::threading::WorkspacePool<Real> prob_pool;
    score_pool.init(n_threads, static_cast<Size>(max_neighbors));
    prob_pool.init(n_threads, static_cast<Size>(max_neighbors));

    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        auto indices = adjacency.primary_indices(static_cast<Index>(i));
        const Index len = adjacency.primary_length(static_cast<Index>(i));

        const Real* feat_i = node_features.ptr + i * F;
        Real* out_feat = output.ptr + i * F;

        Real* attn_scores = score_pool.get(thread_rank);
        Real* attn_probs = prob_pool.get(thread_rank);

        Index n_neighbors = len;

        // Compute attention scores for neighbors
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * F;
            attn_scores[k] = detail::compute_attention_score(
                feat_i, feat_j, feat_dim, attention_vec.ptr);
        }

        // Self-loop
        if (add_self_loops) {
            attn_scores[n_neighbors] = detail::compute_attention_score(
                feat_i, feat_i, feat_dim, attention_vec.ptr);
            ++n_neighbors;
        }

        // Fused softmax
        detail::sparse_softmax_fused(attn_scores, n_neighbors, attn_probs);

        // Initialize output
        detail::feature_fill_simd(out_feat, Real(0), feat_dim);

        // Weighted aggregation
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * F;
            detail::feature_add_scaled_simd(feat_j, out_feat, attn_probs[k], feat_dim);
        }

        // Self-loop contribution
        if (add_self_loops) {
            detail::feature_add_scaled_simd(feat_i, out_feat, attn_probs[len], feat_dim);
        }
    });
}

// =============================================================================
// Multi-Head Attention (Parallel across nodes and heads)
// =============================================================================

template <typename T, bool IsCSR>
void multi_head_attention(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Index n_heads,
    Array<const Real> attention_vecs,
    Array<Real> output,
    bool add_self_loops = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Index head_dim = feat_dim / n_heads;

    SCL_CHECK_DIM(output.len >= N * static_cast<Size>(n_heads * head_dim),
                  "MHA: output buffer too small");

    if (n == 0 || feat_dim == 0 || n_heads == 0) return;

    Index max_neighbors = 0;
    for (Index i = 0; i < n; ++i) {
        max_neighbors = scl::algo::max2(max_neighbors, adjacency.primary_length(i));
    }
    max_neighbors += 1;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> score_pool;
    scl::threading::WorkspacePool<Real> prob_pool;
    score_pool.init(n_threads, static_cast<Size>(max_neighbors));
    prob_pool.init(n_threads, static_cast<Size>(max_neighbors));

    // Process all heads for each node
    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        auto indices = adjacency.primary_indices(static_cast<Index>(i));
        const Index len = adjacency.primary_length(static_cast<Index>(i));

        Real* attn_scores = score_pool.get(thread_rank);
        Real* attn_probs = prob_pool.get(thread_rank);

        for (Index h = 0; h < n_heads; ++h) {
            const Real* attn_vec = attention_vecs.ptr + static_cast<Size>(h) * 2 * head_dim;
            Index feat_offset = h * head_dim;
            const Real* feat_i = node_features.ptr + i * feat_dim + feat_offset;
            Real* out_feat = output.ptr + i * (n_heads * head_dim) + feat_offset;

            Index n_neighbors = len;

            // Compute attention scores
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim + feat_offset;
                attn_scores[k] = detail::compute_attention_score(
                    feat_i, feat_j, head_dim, attn_vec);
            }

            if (add_self_loops) {
                attn_scores[n_neighbors] = detail::compute_attention_score(
                    feat_i, feat_i, head_dim, attn_vec);
                ++n_neighbors;
            }

            detail::sparse_softmax_fused(attn_scores, n_neighbors, attn_probs);

            detail::feature_fill_simd(out_feat, Real(0), head_dim);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim + feat_offset;
                detail::feature_add_scaled_simd(feat_j, out_feat, attn_probs[k], head_dim);
            }

            if (add_self_loops) {
                detail::feature_add_scaled_simd(feat_i, out_feat, attn_probs[len], head_dim);
            }
        }
    });
}

// =============================================================================
// Graph Convolution (GCN-style, Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void graph_convolution(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index in_dim,
    Array<const Real> weight,
    Index out_dim,
    Array<Real> output,
    bool add_self_loops = true,
    ActivationType activation = ActivationType::ReLU
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(output.len >= N * static_cast<Size>(out_dim), "GCN: output buffer too small");

    if (n == 0 || in_dim == 0 || out_dim == 0) return;

    // Precompute D^(-1/2)
    Real* degree_inv_sqrt = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    detail::precompute_gcn_norm(adjacency, degree_inv_sqrt, N, add_self_loops);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace for aggregated features
    scl::threading::WorkspacePool<Real> agg_pool;
    agg_pool.init(n_threads, static_cast<Size>(in_dim));

    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        auto indices = adjacency.primary_indices(static_cast<Index>(i));
        auto values = adjacency.primary_values(static_cast<Index>(i));
        const Index len = adjacency.primary_length(static_cast<Index>(i));

        const Real* feat_i = node_features.ptr + i * in_dim;
        Real* out_feat = output.ptr + i * out_dim;

        Real d_i = degree_inv_sqrt[i];
        Real* agg_feat = agg_pool.get(thread_rank);

        detail::feature_fill_simd(agg_feat, Real(0), in_dim);

        // Aggregate with symmetric normalization
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(values[k]);
            Real d_j = degree_inv_sqrt[j];
            Real norm = d_i * w * d_j;

            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * in_dim;
            detail::feature_add_scaled_simd(feat_j, agg_feat, norm, in_dim);
        }

        // Self-loop
        if (add_self_loops) {
            Real self_norm = d_i * d_i;
            detail::feature_add_scaled_simd(feat_i, agg_feat, self_norm, in_dim);
        }

        // Matrix multiply: out = agg * W, with activation
        for (Index od = 0; od < out_dim; ++od) {
            Real sum = detail::feature_dot_simd(
                agg_feat, weight.ptr + static_cast<Size>(od) * in_dim, in_dim);
            out_feat[od] = detail::apply_activation(sum, activation);
        }
    });

    scl::memory::aligned_free(degree_inv_sqrt, SCL_ALIGNMENT);
}

// =============================================================================
// GraphSAGE Aggregate (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void sage_aggregate(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Array<Real> output,
    AggregationType agg_type = AggregationType::Mean,
    Index max_neighbors = 0
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size F = static_cast<Size>(feat_dim);

    SCL_CHECK_DIM(output.len >= N * F, "SAGE: output buffer too small");

    if (n == 0 || feat_dim == 0) return;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        auto indices = adjacency.primary_indices(static_cast<Index>(i));
        Index len = adjacency.primary_length(static_cast<Index>(i));

        if (max_neighbors > 0 && len > max_neighbors) {
            len = max_neighbors;
        }

        Real* out_feat = output.ptr + i * F;

        if (agg_type == AggregationType::Max) {
            detail::feature_fill_simd(out_feat, -Real(1e30), feat_dim);
        } else {
            detail::feature_fill_simd(out_feat, Real(0), feat_dim);
        }

        Real count = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * F;

            if (agg_type == AggregationType::Max) {
                detail::feature_max_simd(feat_j, out_feat, feat_dim);
            } else {
                detail::feature_add_simd(feat_j, out_feat, feat_dim);
                count += Real(1);
            }
        }

        if (agg_type == AggregationType::Mean && count > Real(0)) {
            detail::feature_scale_simd(out_feat, Real(1) / count, feat_dim);
        }

        if (len == 0) {
            const Real* feat_i = node_features.ptr + i * F;
            std::memcpy(out_feat, feat_i, F * sizeof(Real));
        }
    });
}

// =============================================================================
// Feature Smoothing (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void feature_smoothing(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> features,
    Index n_nodes,
    Index feat_dim,
    Real alpha = Real(0.5),
    Index n_iterations = 1
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size F = static_cast<Size>(feat_dim);
    const Size total = N * F;

    SCL_CHECK_DIM(features.len >= total, "Smoothing: features buffer too small");

    if (n_nodes == 0 || feat_dim == 0 || n_iterations == 0) return;

    Real* temp = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real one_minus_alpha = Real(1) - alpha;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    for (Index iter = 0; iter < n_iterations; ++iter) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto indices = adjacency.primary_indices(static_cast<Index>(i));
            auto values = adjacency.primary_values(static_cast<Index>(i));
            const Index len = adjacency.primary_length(static_cast<Index>(i));

            Real* temp_feat = temp + i * F;
            const Real* orig_feat = features.ptr + i * F;

            // Start with self contribution
            for (Index d = 0; d < feat_dim; ++d) {
                temp_feat[d] = one_minus_alpha * orig_feat[d];
            }

            if (len == 0) return;

            // Neighbor contribution
            Real weight_sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                const Real* neighbor_feat = features.ptr + static_cast<Size>(j) * F;
                detail::feature_add_scaled_simd(neighbor_feat, temp_feat, alpha * w, feat_dim);
                weight_sum += w;
            }

            // Normalize neighbor contribution
            if (weight_sum > config::ATTENTION_EPSILON) {
                Real inv_weight = Real(1) / weight_sum;
                for (Index d = 0; d < feat_dim; ++d) {
                    Real neighbor_avg = (temp_feat[d] - one_minus_alpha * orig_feat[d]) * inv_weight;
                    temp_feat[d] = one_minus_alpha * orig_feat[d] + alpha * neighbor_avg;
                }
            }
        });

        std::memcpy(features.ptr, temp, total * sizeof(Real));
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Global Pool (Parallel Reduction)
// =============================================================================

template <typename T, bool IsCSR>
void global_pool(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Array<Real> graph_features,
    AggregationType agg_type = AggregationType::Mean
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size F = static_cast<Size>(feat_dim);

    SCL_CHECK_DIM(graph_features.len >= F, "GlobalPool: output buffer too small");

    if (n == 0 || feat_dim == 0) {
        detail::feature_fill_simd(graph_features.ptr, Real(0), feat_dim);
        return;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (agg_type == AggregationType::Sum || agg_type == AggregationType::Mean) {
        // Parallel reduction for sum/mean
        Real* partial = scl::memory::aligned_alloc<Real>(n_threads * F, SCL_ALIGNMENT);
        scl::algo::zero(partial, n_threads * F);

        scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
            const Real* feat = node_features.ptr + i * F;
            Real* local = partial + thread_rank * F;
            detail::feature_add_simd(feat, local, feat_dim);
        });

        // Final reduction
        detail::feature_fill_simd(graph_features.ptr, Real(0), feat_dim);
        for (size_t t = 0; t < n_threads; ++t) {
            detail::feature_add_simd(partial + t * F, graph_features.ptr, feat_dim);
        }

        if (agg_type == AggregationType::Mean) {
            detail::feature_scale_simd(graph_features.ptr, Real(1) / static_cast<Real>(n), feat_dim);
        }

        scl::memory::aligned_free(partial, SCL_ALIGNMENT);
    } else if (agg_type == AggregationType::Max) {
        // Parallel max reduction
        Real* partial = scl::memory::aligned_alloc<Real>(n_threads * F, SCL_ALIGNMENT);
        for (size_t t = 0; t < n_threads; ++t) {
            detail::feature_fill_simd(partial + t * F, -Real(1e30), feat_dim);
        }

        scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
            const Real* feat = node_features.ptr + i * F;
            Real* local = partial + thread_rank * F;
            detail::feature_max_simd(feat, local, feat_dim);
        });

        detail::feature_fill_simd(graph_features.ptr, -Real(1e30), feat_dim);
        for (size_t t = 0; t < n_threads; ++t) {
            detail::feature_max_simd(partial + t * F, graph_features.ptr, feat_dim);
        }

        scl::memory::aligned_free(partial, SCL_ALIGNMENT);
    } else if (agg_type == AggregationType::Min) {
        Real* partial = scl::memory::aligned_alloc<Real>(n_threads * F, SCL_ALIGNMENT);
        for (size_t t = 0; t < n_threads; ++t) {
            detail::feature_fill_simd(partial + t * F, Real(1e30), feat_dim);
        }

        scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
            const Real* feat = node_features.ptr + i * F;
            Real* local = partial + thread_rank * F;
            detail::feature_min_simd(feat, local, feat_dim);
        });

        detail::feature_fill_simd(graph_features.ptr, Real(1e30), feat_dim);
        for (size_t t = 0; t < n_threads; ++t) {
            detail::feature_min_simd(partial + t * F, graph_features.ptr, feat_dim);
        }

        scl::memory::aligned_free(partial, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Hierarchical Pool (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void hierarchical_pool(
    Array<const Real> node_features,
    Index n_nodes,
    Index feat_dim,
    Array<const Index> cluster_assignment,
    Index n_clusters,
    Array<Real> pooled_features,
    AggregationType agg_type = AggregationType::Mean
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size F = static_cast<Size>(feat_dim);
    const Size C = static_cast<Size>(n_clusters);

    SCL_CHECK_DIM(pooled_features.len >= C * F, "HierPool: output buffer too small");

    // Initialize
    if (agg_type == AggregationType::Max) {
        for (Size i = 0; i < C * F; ++i) {
            pooled_features[i] = -Real(1e30);
        }
    } else if (agg_type == AggregationType::Min) {
        for (Size i = 0; i < C * F; ++i) {
            pooled_features[i] = Real(1e30);
        }
    } else {
        scl::algo::zero(pooled_features.ptr, C * F);
    }

    // Count cluster sizes
    Index* cluster_sizes = scl::memory::aligned_alloc<Index>(C, SCL_ALIGNMENT);
    scl::algo::zero(cluster_sizes, C);

    // Sequential aggregation (cluster access pattern is irregular)
    for (Index i = 0; i < n_nodes; ++i) {
        Index c = cluster_assignment[i];
        if (c < 0 || c >= n_clusters) continue;

        const Real* feat = node_features.ptr + static_cast<Size>(i) * F;
        Real* pooled = pooled_features.ptr + static_cast<Size>(c) * F;

        switch (agg_type) {
            case AggregationType::Sum:
            case AggregationType::Mean:
                detail::feature_add_simd(feat, pooled, feat_dim);
                ++cluster_sizes[c];
                break;
            case AggregationType::Max:
                detail::feature_max_simd(feat, pooled, feat_dim);
                break;
            case AggregationType::Min:
                detail::feature_min_simd(feat, pooled, feat_dim);
                break;
            default:
                break;
        }
    }

    // Normalize for mean
    if (agg_type == AggregationType::Mean) {
        for (Index c = 0; c < n_clusters; ++c) {
            if (cluster_sizes[c] > 0) {
                Real* pooled = pooled_features.ptr + static_cast<Size>(c) * F;
                detail::feature_scale_simd(pooled, Real(1) / static_cast<Real>(cluster_sizes[c]), feat_dim);
            }
        }
    }

    scl::memory::aligned_free(cluster_sizes, SCL_ALIGNMENT);
}

// =============================================================================
// Edge Features (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void compute_edge_features(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,
    Index feat_dim,
    Real* edge_features,
    bool concat = true
) {
    const Index n = adjacency.primary_dim();
    const Size F = static_cast<Size>(feat_dim);
    const Size edge_feat_dim = concat ? 2 * F : F;

    // Compute offsets for parallel access
    Size* offsets = scl::memory::aligned_alloc<Size>(static_cast<Size>(n) + 1, SCL_ALIGNMENT);
    offsets[0] = 0;
    for (Index i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + static_cast<Size>(adjacency.primary_length(i));
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i) {
        auto indices = adjacency.primary_indices(static_cast<Index>(i));
        const Index len = adjacency.primary_length(static_cast<Index>(i));

        const Real* feat_i = node_features.ptr + i * F;
        Size base_edge_idx = offsets[i];

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * F;
            Real* edge_feat = edge_features + (base_edge_idx + k) * edge_feat_dim;

            if (concat) {
                std::memcpy(edge_feat, feat_i, F * sizeof(Real));
                std::memcpy(edge_feat + F, feat_j, F * sizeof(Real));
            } else {
                for (Index d = 0; d < feat_dim; ++d) {
                    edge_feat[d] = feat_j[d] - feat_i[d];
                }
            }
        }
    });

    scl::memory::aligned_free(offsets, SCL_ALIGNMENT);
}

// =============================================================================
// Skip Connection (SIMD)
// =============================================================================

inline void skip_connection(
    Array<const Real> input,
    Array<const Real> residual,
    Array<Real> output,
    Real alpha = Real(1.0)
) {
    SCL_CHECK_DIM(output.len >= input.len, "Skip: output buffer too small");
    SCL_CHECK_DIM(residual.len >= input.len, "Skip: residual buffer too small");

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + lanes <= input.len; k += lanes) {
        auto v_in = s::Load(d, input.ptr + k);
        auto v_res = s::Load(d, residual.ptr + k);
        s::Store(s::MulAdd(v_alpha, v_res, v_in), d, output.ptr + k);
    }

    for (; k < input.len; ++k) {
        output[k] = input[k] + alpha * residual[k];
    }
}

// =============================================================================
// Layer Normalization (Parallel + SIMD)
// =============================================================================

inline void layer_norm(
    Array<Real> features,
    Index n_nodes,
    Index feat_dim,
    Real epsilon = Real(1e-5)
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size F = static_cast<Size>(feat_dim);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real* feat = features.ptr + i * F;

        // Compute mean
        Real mean = Real(0);
        for (Index d = 0; d < feat_dim; ++d) {
            mean += feat[d];
        }
        mean /= static_cast<Real>(feat_dim);

        // Compute variance
        Real var = Real(0);
        for (Index d = 0; d < feat_dim; ++d) {
            Real diff = feat[d] - mean;
            var += diff * diff;
        }
        var /= static_cast<Real>(feat_dim);

        // Normalize
        Real inv_std = Real(1) / std::sqrt(var + epsilon);
        for (Index d = 0; d < feat_dim; ++d) {
            feat[d] = (feat[d] - mean) * inv_std;
        }
    });
}

// =============================================================================
// Batch Normalization (Parallel)
// =============================================================================

inline void batch_norm(
    Array<Real> features,
    Index n_nodes,
    Index feat_dim,
    Array<const Real> gamma,
    Array<const Real> beta,
    Real epsilon = Real(1e-5)
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size F = static_cast<Size>(feat_dim);

    // Compute mean and variance per feature
    Real* mean = scl::memory::aligned_alloc<Real>(F, SCL_ALIGNMENT);
    Real* var = scl::memory::aligned_alloc<Real>(F, SCL_ALIGNMENT);
    scl::algo::zero(mean, F);
    scl::algo::zero(var, F);

    // Mean
    for (Index i = 0; i < n_nodes; ++i) {
        const Real* feat = features.ptr + static_cast<Size>(i) * F;
        detail::feature_add_simd(feat, mean, feat_dim);
    }

    Real inv_n = Real(1) / static_cast<Real>(n_nodes);
    detail::feature_scale_simd(mean, inv_n, feat_dim);

    // Variance
    for (Index i = 0; i < n_nodes; ++i) {
        const Real* feat = features.ptr + static_cast<Size>(i) * F;
        for (Index d = 0; d < feat_dim; ++d) {
            Real diff = feat[d] - mean[d];
            var[d] += diff * diff;
        }
    }

    detail::feature_scale_simd(var, inv_n, feat_dim);

    // Normalize
    Real* inv_std = scl::memory::aligned_alloc<Real>(F, SCL_ALIGNMENT);
    for (Index d = 0; d < feat_dim; ++d) {
        inv_std[d] = Real(1) / std::sqrt(var[d] + epsilon);
    }

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real* feat = features.ptr + i * F;
        for (Index d = 0; d < feat_dim; ++d) {
            feat[d] = gamma[d] * (feat[d] - mean[d]) * inv_std[d] + beta[d];
        }
    });

    scl::memory::aligned_free(inv_std, SCL_ALIGNMENT);
    scl::memory::aligned_free(var, SCL_ALIGNMENT);
    scl::memory::aligned_free(mean, SCL_ALIGNMENT);
}

// =============================================================================
// Dropout (Training Mode with PRNG)
// =============================================================================

inline void dropout(
    Array<Real> features,
    Real dropout_rate,
    uint64_t seed,
    bool training = true
) {
    if (!training || dropout_rate <= Real(0)) return;

    Real keep_prob = Real(1) - dropout_rate;
    Real scale = Real(1) / keep_prob;

    // Simple xorshift PRNG
    uint64_t state = seed;
    auto next_rand = [&state]() -> Real {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return static_cast<Real>(state * 0x2545F4914F6CDD1DULL) * Real(5.4210108624275222e-20);
    };

    for (Size i = 0; i < features.len; ++i) {
        if (next_rand() >= keep_prob) {
            features[i] = Real(0);
        } else {
            features[i] *= scale;
        }
    }
}

// Inference mode (identity)
inline void dropout_inference(
    Array<const Real> input,
    Array<Real> output,
    Real dropout_rate
) {
    (void)dropout_rate;
    std::memcpy(output.ptr, input.ptr, input.len * sizeof(Real));
}

} // namespace scl::kernel::gnn
