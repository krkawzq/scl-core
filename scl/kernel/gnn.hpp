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
// BRIEF: Graph Neural Network primitives for message passing and aggregation
// =============================================================================

namespace scl::kernel::gnn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real ATTENTION_EPSILON = Real(1e-12);
    constexpr Real LEAKY_RELU_SLOPE = Real(0.2);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Size PARALLEL_THRESHOLD = 500;
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
    ELU
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Activation functions
SCL_FORCE_INLINE Real apply_activation(Real x, ActivationType act) {
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
        default:
            return x;
    }
}

// Sparse softmax over row elements
template <typename T>
SCL_FORCE_INLINE void sparse_row_softmax(
    const T* values,
    Index len,
    Real* output
) {
    if (len == 0) return;

    // Find max for numerical stability
    Real max_val = static_cast<Real>(values[0]);
    for (Index k = 1; k < len; ++k) {
        max_val = scl::algo::max2(max_val, static_cast<Real>(values[k]));
    }

    // Compute exp and sum
    Real sum = Real(0);
    for (Index k = 0; k < len; ++k) {
        output[k] = std::exp(static_cast<Real>(values[k]) - max_val);
        sum += output[k];
    }

    // Normalize
    if (sum > config::ATTENTION_EPSILON) {
        for (Index k = 0; k < len; ++k) {
            output[k] /= sum;
        }
    } else {
        Real uniform = Real(1) / static_cast<Real>(len);
        for (Index k = 0; k < len; ++k) {
            output[k] = uniform;
        }
    }
}

// Compute attention score: LeakyReLU(a^T [W*h_i || W*h_j])
SCL_FORCE_INLINE Real compute_attention_score(
    const Real* feat_i,
    const Real* feat_j,
    Index feat_dim,
    const Real* attention_vec  // Size: 2 * feat_dim
) {
    Real score = Real(0);

    // First half for source node
    for (Index d = 0; d < feat_dim; ++d) {
        score += attention_vec[d] * feat_i[d];
    }

    // Second half for target node
    for (Index d = 0; d < feat_dim; ++d) {
        score += attention_vec[feat_dim + d] * feat_j[d];
    }

    // LeakyReLU
    return apply_activation(score, ActivationType::LeakyReLU);
}

} // namespace detail

// =============================================================================
// Message Passing (Aggregate Neighbor Features)
// =============================================================================

template <typename T, bool IsCSR>
void message_passing(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim, row-major
    Index feat_dim,
    Array<Real> output,               // n_nodes x feat_dim, row-major
    AggregationType agg_type = AggregationType::Mean
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n) * static_cast<Size>(feat_dim),
                  "GNN: output buffer too small");

    if (n == 0 || feat_dim == 0) return;

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);

        Real* out_feat = output.ptr + static_cast<Size>(i) * feat_dim;

        // Initialize output
        if (agg_type == AggregationType::Max) {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = -Real(1e30);
            }
        } else if (agg_type == AggregationType::Min) {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = Real(1e30);
            }
        } else {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = Real(0);
            }
        }

        Real total_weight = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(values[k]);
            const Real* neighbor_feat = node_features.ptr + static_cast<Size>(j) * feat_dim;

            switch (agg_type) {
                case AggregationType::Sum:
                    for (Index d = 0; d < feat_dim; ++d) {
                        out_feat[d] += neighbor_feat[d];
                    }
                    break;

                case AggregationType::Mean:
                    for (Index d = 0; d < feat_dim; ++d) {
                        out_feat[d] += neighbor_feat[d];
                    }
                    total_weight += Real(1);
                    break;

                case AggregationType::Weighted:
                    for (Index d = 0; d < feat_dim; ++d) {
                        out_feat[d] += w * neighbor_feat[d];
                    }
                    total_weight += w;
                    break;

                case AggregationType::Max:
                    for (Index d = 0; d < feat_dim; ++d) {
                        out_feat[d] = scl::algo::max2(out_feat[d], neighbor_feat[d]);
                    }
                    break;

                case AggregationType::Min:
                    for (Index d = 0; d < feat_dim; ++d) {
                        out_feat[d] = scl::algo::min2(out_feat[d], neighbor_feat[d]);
                    }
                    break;

                default:
                    break;
            }
        }

        // Normalize for mean/weighted aggregation
        if (agg_type == AggregationType::Mean || agg_type == AggregationType::Weighted) {
            if (total_weight > config::ATTENTION_EPSILON) {
                for (Index d = 0; d < feat_dim; ++d) {
                    out_feat[d] /= total_weight;
                }
            }
        }

        // Handle empty neighborhood for max/min
        if (len == 0) {
            const Real* self_feat = node_features.ptr + static_cast<Size>(i) * feat_dim;
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = self_feat[d];
            }
        }
    }
}

// =============================================================================
// Graph Attention (GAT-style)
// =============================================================================

template <typename T, bool IsCSR>
void graph_attention(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index feat_dim,
    Array<const Real> attention_vec,  // 2 * feat_dim attention parameters
    Array<Real> output,               // n_nodes x feat_dim
    bool add_self_loops = true
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n) * static_cast<Size>(feat_dim),
                  "GAT: output buffer too small");
    SCL_CHECK_DIM(attention_vec.len >= static_cast<Size>(2 * feat_dim),
                  "GAT: attention vector too small");

    if (n == 0 || feat_dim == 0) return;

    // Workspace for attention scores
    Index max_neighbors = 0;
    for (Index i = 0; i < n; ++i) {
        max_neighbors = scl::algo::max2(max_neighbors, adjacency.primary_length(i));
    }
    max_neighbors += 1;  // For self-loop

    Real* attn_scores = scl::memory::aligned_alloc<Real>(max_neighbors, SCL_ALIGNMENT);
    Real* attn_probs = scl::memory::aligned_alloc<Real>(max_neighbors, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        const Index len = adjacency.primary_length(i);

        const Real* feat_i = node_features.ptr + static_cast<Size>(i) * feat_dim;
        Real* out_feat = output.ptr + static_cast<Size>(i) * feat_dim;

        Index n_neighbors = len;

        // Compute attention scores
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim;
            attn_scores[k] = detail::compute_attention_score(
                feat_i, feat_j, feat_dim, attention_vec.ptr);
        }

        // Add self-loop attention
        if (add_self_loops) {
            attn_scores[n_neighbors] = detail::compute_attention_score(
                feat_i, feat_i, feat_dim, attention_vec.ptr);
            ++n_neighbors;
        }

        // Softmax over attention scores
        detail::sparse_row_softmax(attn_scores, n_neighbors, attn_probs);

        // Initialize output
        for (Index d = 0; d < feat_dim; ++d) {
            out_feat[d] = Real(0);
        }

        // Weighted aggregation
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim;
            Real alpha = attn_probs[k];

            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] += alpha * feat_j[d];
            }
        }

        // Add self-loop contribution
        if (add_self_loops) {
            Real alpha_self = attn_probs[len];
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] += alpha_self * feat_i[d];
            }
        }
    }

    scl::memory::aligned_free(attn_probs, SCL_ALIGNMENT);
    scl::memory::aligned_free(attn_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Multi-Head Attention
// =============================================================================

template <typename T, bool IsCSR>
void multi_head_attention(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index feat_dim,
    Index n_heads,
    Array<const Real> attention_vecs,  // n_heads * (2 * head_dim)
    Array<Real> output,                // n_nodes x (n_heads * head_dim)
    bool add_self_loops = true
) {
    const Index n = adjacency.primary_dim();
    Index head_dim = feat_dim / n_heads;

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n) * static_cast<Size>(n_heads * head_dim),
                  "MHA: output buffer too small");

    if (n == 0 || feat_dim == 0 || n_heads == 0) return;

    Index max_neighbors = 0;
    for (Index i = 0; i < n; ++i) {
        max_neighbors = scl::algo::max2(max_neighbors, adjacency.primary_length(i));
    }
    max_neighbors += 1;

    Real* attn_scores = scl::memory::aligned_alloc<Real>(max_neighbors, SCL_ALIGNMENT);
    Real* attn_probs = scl::memory::aligned_alloc<Real>(max_neighbors, SCL_ALIGNMENT);

    for (Index h = 0; h < n_heads; ++h) {
        const Real* attn_vec = attention_vecs.ptr + static_cast<Size>(h) * 2 * head_dim;
        Index feat_offset = h * head_dim;

        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            const Index len = adjacency.primary_length(i);

            const Real* feat_i = node_features.ptr + static_cast<Size>(i) * feat_dim + feat_offset;
            Real* out_feat = output.ptr + static_cast<Size>(i) * (n_heads * head_dim) + feat_offset;

            Index n_neighbors = len;

            // Compute attention scores for this head
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

            detail::sparse_row_softmax(attn_scores, n_neighbors, attn_probs);

            // Initialize
            for (Index d = 0; d < head_dim; ++d) {
                out_feat[d] = Real(0);
            }

            // Aggregate
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim + feat_offset;
                Real alpha = attn_probs[k];

                for (Index d = 0; d < head_dim; ++d) {
                    out_feat[d] += alpha * feat_j[d];
                }
            }

            if (add_self_loops) {
                Real alpha_self = attn_probs[len];
                for (Index d = 0; d < head_dim; ++d) {
                    out_feat[d] += alpha_self * feat_i[d];
                }
            }
        }
    }

    scl::memory::aligned_free(attn_probs, SCL_ALIGNMENT);
    scl::memory::aligned_free(attn_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Sparse Softmax Over Neighbors
// =============================================================================

template <typename T, bool IsCSR>
void sparse_softmax_neighbors(
    const Sparse<T, IsCSR>& logits,
    Real* output_probs  // Same sparsity pattern as logits
) {
    const Index n = logits.primary_dim();

    Size val_idx = 0;
    for (Index i = 0; i < n; ++i) {
        auto values = logits.primary_values(i);
        const Index len = logits.primary_length(i);

        if (len == 0) continue;

        // Find max
        Real max_val = static_cast<Real>(values[0]);
        for (Index k = 1; k < len; ++k) {
            max_val = scl::algo::max2(max_val, static_cast<Real>(values[k]));
        }

        // Compute exp and sum
        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            output_probs[val_idx + k] = std::exp(static_cast<Real>(values[k]) - max_val);
            sum += output_probs[val_idx + k];
        }

        // Normalize
        if (sum > config::ATTENTION_EPSILON) {
            for (Index k = 0; k < len; ++k) {
                output_probs[val_idx + k] /= sum;
            }
        } else {
            Real uniform = Real(1) / static_cast<Real>(len);
            for (Index k = 0; k < len; ++k) {
                output_probs[val_idx + k] = uniform;
            }
        }

        val_idx += len;
    }
}

// =============================================================================
// Graph Convolution (GCN-style)
// =============================================================================

template <typename T, bool IsCSR>
void graph_convolution(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x in_dim
    Index in_dim,
    Array<const Real> weight,         // in_dim x out_dim
    Index out_dim,
    Array<Real> output,               // n_nodes x out_dim
    bool add_self_loops = true,
    ActivationType activation = ActivationType::ReLU
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n) * static_cast<Size>(out_dim),
                  "GCN: output buffer too small");

    if (n == 0 || in_dim == 0 || out_dim == 0) return;

    // Compute degree for normalization (D^-1/2)
    Real* degree_inv_sqrt = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);

        Real deg = Real(0);
        for (Index k = 0; k < len; ++k) {
            deg += static_cast<Real>(values[k]);
        }
        if (add_self_loops) deg += Real(1);  // Self-loop

        degree_inv_sqrt[i] = (deg > config::ATTENTION_EPSILON)
            ? Real(1) / std::sqrt(deg) : Real(0);
    }

    // Temporary for aggregated features
    Real* agg_feat = scl::memory::aligned_alloc<Real>(in_dim, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);

        const Real* feat_i = node_features.ptr + static_cast<Size>(i) * in_dim;
        Real* out_feat = output.ptr + static_cast<Size>(i) * out_dim;
        Real d_i = degree_inv_sqrt[i];

        // Aggregate neighbor features with symmetric normalization
        for (Index d = 0; d < in_dim; ++d) {
            agg_feat[d] = Real(0);
        }

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(values[k]);
            Real d_j = degree_inv_sqrt[j];
            Real norm = d_i * w * d_j;

            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * in_dim;
            for (Index d = 0; d < in_dim; ++d) {
                agg_feat[d] += norm * feat_j[d];
            }
        }

        // Add self-loop
        if (add_self_loops) {
            Real self_norm = d_i * d_i;  // D^-1/2 * 1 * D^-1/2
            for (Index d = 0; d < in_dim; ++d) {
                agg_feat[d] += self_norm * feat_i[d];
            }
        }

        // Apply weight matrix and activation
        for (Index od = 0; od < out_dim; ++od) {
            Real sum = Real(0);
            for (Index id = 0; id < in_dim; ++id) {
                sum += agg_feat[id] * weight[static_cast<Size>(id) * out_dim + od];
            }
            out_feat[od] = detail::apply_activation(sum, activation);
        }
    }

    scl::memory::aligned_free(agg_feat, SCL_ALIGNMENT);
    scl::memory::aligned_free(degree_inv_sqrt, SCL_ALIGNMENT);
}

// =============================================================================
// GraphSAGE-style Sampling and Aggregation
// =============================================================================

template <typename T, bool IsCSR>
void sage_aggregate(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index feat_dim,
    Array<Real> output,               // n_nodes x feat_dim
    AggregationType agg_type = AggregationType::Mean,
    Index max_neighbors = 0  // 0 = use all
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n) * static_cast<Size>(feat_dim),
                  "SAGE: output buffer too small");

    if (n == 0 || feat_dim == 0) return;

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        auto values = adjacency.primary_values(i);
        Index len = adjacency.primary_length(i);

        // Limit neighbors if specified
        if (max_neighbors > 0 && len > max_neighbors) {
            len = max_neighbors;
        }

        Real* out_feat = output.ptr + static_cast<Size>(i) * feat_dim;

        // Initialize based on aggregation type
        if (agg_type == AggregationType::Max) {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = -Real(1e30);
            }
        } else {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = Real(0);
            }
        }

        Real count = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim;

            if (agg_type == AggregationType::Max) {
                for (Index d = 0; d < feat_dim; ++d) {
                    out_feat[d] = scl::algo::max2(out_feat[d], feat_j[d]);
                }
            } else {
                for (Index d = 0; d < feat_dim; ++d) {
                    out_feat[d] += feat_j[d];
                }
                count += Real(1);
            }
        }

        // Normalize for mean aggregation
        if (agg_type == AggregationType::Mean && count > Real(0)) {
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] /= count;
            }
        }

        // Handle empty neighborhood
        if (len == 0) {
            const Real* feat_i = node_features.ptr + static_cast<Size>(i) * feat_dim;
            for (Index d = 0; d < feat_dim; ++d) {
                out_feat[d] = feat_i[d];
            }
        }
    }
}

// =============================================================================
// Feature Smoothing via Graph (Laplacian Smoothing)
// =============================================================================

template <typename T, bool IsCSR>
void feature_smoothing(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> features,  // In/Out: n_nodes x feat_dim
    Index n_nodes,
    Index feat_dim,
    Real alpha = Real(0.5),
    Index n_iterations = 1
) {
    SCL_CHECK_DIM(features.len >= static_cast<Size>(n_nodes) * static_cast<Size>(feat_dim),
                  "Smoothing: features buffer too small");

    if (n_nodes == 0 || feat_dim == 0 || n_iterations == 0) return;

    Size total = static_cast<Size>(n_nodes) * static_cast<Size>(feat_dim);
    Real* temp = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    for (Index iter = 0; iter < n_iterations; ++iter) {
        for (Index i = 0; i < n_nodes; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real* temp_feat = temp + static_cast<Size>(i) * feat_dim;
            const Real* orig_feat = features.ptr + static_cast<Size>(i) * feat_dim;

            // Initialize with self feature scaled by (1 - alpha)
            for (Index d = 0; d < feat_dim; ++d) {
                temp_feat[d] = (Real(1) - alpha) * orig_feat[d];
            }

            if (len == 0) continue;

            // Add neighbor contribution
            Real weight_sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                const Real* neighbor_feat = features.ptr + static_cast<Size>(j) * feat_dim;

                for (Index d = 0; d < feat_dim; ++d) {
                    temp_feat[d] += alpha * w * neighbor_feat[d];
                }
                weight_sum += w;
            }

            // Normalize neighbor contribution
            if (weight_sum > config::ATTENTION_EPSILON) {
                for (Index d = 0; d < feat_dim; ++d) {
                    // Recompute: (1-alpha)*self + alpha*(weighted_avg)
                    Real neighbor_avg = (temp_feat[d] - (Real(1) - alpha) * orig_feat[d]) / weight_sum;
                    temp_feat[d] = (Real(1) - alpha) * orig_feat[d] + alpha * neighbor_avg;
                }
            }
        }

        // Copy back
        for (Size i = 0; i < total; ++i) {
            features[i] = temp[i];
        }
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Graph Pooling (Global)
// =============================================================================

template <typename T, bool IsCSR>
void global_pool(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index feat_dim,
    Array<Real> graph_features,       // feat_dim output
    AggregationType agg_type = AggregationType::Mean
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(graph_features.len >= static_cast<Size>(feat_dim),
                  "GlobalPool: output buffer too small");

    if (n == 0 || feat_dim == 0) {
        for (Index d = 0; d < feat_dim; ++d) {
            graph_features[d] = Real(0);
        }
        return;
    }

    // Initialize
    if (agg_type == AggregationType::Max) {
        for (Index d = 0; d < feat_dim; ++d) {
            graph_features[d] = -Real(1e30);
        }
    } else if (agg_type == AggregationType::Min) {
        for (Index d = 0; d < feat_dim; ++d) {
            graph_features[d] = Real(1e30);
        }
    } else {
        for (Index d = 0; d < feat_dim; ++d) {
            graph_features[d] = Real(0);
        }
    }

    // Aggregate all nodes
    for (Index i = 0; i < n; ++i) {
        const Real* feat = node_features.ptr + static_cast<Size>(i) * feat_dim;

        switch (agg_type) {
            case AggregationType::Sum:
            case AggregationType::Mean:
                for (Index d = 0; d < feat_dim; ++d) {
                    graph_features[d] += feat[d];
                }
                break;
            case AggregationType::Max:
                for (Index d = 0; d < feat_dim; ++d) {
                    graph_features[d] = scl::algo::max2(graph_features[d], feat[d]);
                }
                break;
            case AggregationType::Min:
                for (Index d = 0; d < feat_dim; ++d) {
                    graph_features[d] = scl::algo::min2(graph_features[d], feat[d]);
                }
                break;
            default:
                break;
        }
    }

    // Normalize for mean
    if (agg_type == AggregationType::Mean) {
        Real inv_n = Real(1) / static_cast<Real>(n);
        for (Index d = 0; d < feat_dim; ++d) {
            graph_features[d] *= inv_n;
        }
    }
}

// =============================================================================
// Hierarchical Graph Pooling (By Cluster Assignment)
// =============================================================================

template <typename T, bool IsCSR>
void hierarchical_pool(
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index n_nodes,
    Index feat_dim,
    Array<const Index> cluster_assignment,  // n_nodes
    Index n_clusters,
    Array<Real> pooled_features,  // n_clusters x feat_dim
    AggregationType agg_type = AggregationType::Mean
) {
    SCL_CHECK_DIM(pooled_features.len >= static_cast<Size>(n_clusters) * static_cast<Size>(feat_dim),
                  "HierPool: output buffer too small");

    // Initialize
    Size total = static_cast<Size>(n_clusters) * static_cast<Size>(feat_dim);
    if (agg_type == AggregationType::Max) {
        for (Size i = 0; i < total; ++i) {
            pooled_features[i] = -Real(1e30);
        }
    } else if (agg_type == AggregationType::Min) {
        for (Size i = 0; i < total; ++i) {
            pooled_features[i] = Real(1e30);
        }
    } else {
        scl::algo::zero(pooled_features.ptr, total);
    }

    // Count nodes per cluster (for mean)
    Index* cluster_sizes = scl::memory::aligned_alloc<Index>(n_clusters, SCL_ALIGNMENT);
    scl::algo::zero(cluster_sizes, static_cast<Size>(n_clusters));

    // Aggregate
    for (Index i = 0; i < n_nodes; ++i) {
        Index c = cluster_assignment[i];
        if (c < 0 || c >= n_clusters) continue;

        const Real* feat = node_features.ptr + static_cast<Size>(i) * feat_dim;
        Real* pooled = pooled_features.ptr + static_cast<Size>(c) * feat_dim;

        switch (agg_type) {
            case AggregationType::Sum:
            case AggregationType::Mean:
                for (Index d = 0; d < feat_dim; ++d) {
                    pooled[d] += feat[d];
                }
                ++cluster_sizes[c];
                break;
            case AggregationType::Max:
                for (Index d = 0; d < feat_dim; ++d) {
                    pooled[d] = scl::algo::max2(pooled[d], feat[d]);
                }
                break;
            case AggregationType::Min:
                for (Index d = 0; d < feat_dim; ++d) {
                    pooled[d] = scl::algo::min2(pooled[d], feat[d]);
                }
                break;
            default:
                break;
        }
    }

    // Normalize for mean
    if (agg_type == AggregationType::Mean) {
        for (Index c = 0; c < n_clusters; ++c) {
            if (cluster_sizes[c] > 0) {
                Real inv_size = Real(1) / static_cast<Real>(cluster_sizes[c]);
                Real* pooled = pooled_features.ptr + static_cast<Size>(c) * feat_dim;
                for (Index d = 0; d < feat_dim; ++d) {
                    pooled[d] *= inv_size;
                }
            }
        }
    }

    scl::memory::aligned_free(cluster_sizes, SCL_ALIGNMENT);
}

// =============================================================================
// Edge Feature Computation
// =============================================================================

template <typename T, bool IsCSR>
void compute_edge_features(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> node_features,  // n_nodes x feat_dim
    Index feat_dim,
    Real* edge_features,  // nnz x (2 * feat_dim) - concat of src and dst features
    bool concat = true    // false = difference
) {
    const Index n = adjacency.primary_dim();

    Size edge_feat_dim = concat ? 2 * feat_dim : feat_dim;
    Size edge_idx = 0;

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        const Index len = adjacency.primary_length(i);

        const Real* feat_i = node_features.ptr + static_cast<Size>(i) * feat_dim;

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            const Real* feat_j = node_features.ptr + static_cast<Size>(j) * feat_dim;
            Real* edge_feat = edge_features + edge_idx * edge_feat_dim;

            if (concat) {
                // Concatenate source and destination features
                for (Index d = 0; d < feat_dim; ++d) {
                    edge_feat[d] = feat_i[d];
                    edge_feat[feat_dim + d] = feat_j[d];
                }
            } else {
                // Difference: dst - src
                for (Index d = 0; d < feat_dim; ++d) {
                    edge_feat[d] = feat_j[d] - feat_i[d];
                }
            }

            ++edge_idx;
        }
    }
}

// =============================================================================
// Skip Connection (Residual)
// =============================================================================

inline void skip_connection(
    Array<const Real> input,
    Array<const Real> residual,
    Array<Real> output,
    Real alpha = Real(1.0)  // Weight for residual
) {
    SCL_CHECK_DIM(output.len >= input.len, "Skip: output buffer too small");
    SCL_CHECK_DIM(residual.len >= input.len, "Skip: residual buffer too small");

    for (Size i = 0; i < input.len; ++i) {
        output[i] = input[i] + alpha * residual[i];
    }
}

// =============================================================================
// Layer Normalization
// =============================================================================

inline void layer_norm(
    Array<Real> features,  // In/Out: n_nodes x feat_dim
    Index n_nodes,
    Index feat_dim,
    Real epsilon = Real(1e-5)
) {
    for (Index i = 0; i < n_nodes; ++i) {
        Real* feat = features.ptr + static_cast<Size>(i) * feat_dim;

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
    }
}

// =============================================================================
// Dropout (Inference Mode - Identity, Training Would Need RNG)
// =============================================================================

inline void dropout_inference(
    Array<const Real> input,
    Array<Real> output,
    Real dropout_rate  // Unused in inference, kept for API consistency
) {
    (void)dropout_rate;
    for (Size i = 0; i < input.len; ++i) {
        output[i] = input[i];
    }
}

} // namespace scl::kernel::gnn
