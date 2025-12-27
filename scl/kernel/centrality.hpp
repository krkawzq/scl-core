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
// FILE: scl/kernel/centrality.hpp
// BRIEF: Graph centrality measures for node importance ranking
// =============================================================================

namespace scl::kernel::centrality {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_DAMPING = Real(0.85);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Real MIN_SCORE = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Check convergence via L1 norm difference
SCL_FORCE_INLINE bool check_convergence(
    const Real* scores_old,
    const Real* scores_new,
    Size n,
    Real tol
) {
    Real diff = Real(0);
    for (Size i = 0; i < n; ++i) {
        Real d = scores_new[i] - scores_old[i];
        diff += (d >= Real(0)) ? d : -d;
    }
    return diff < tol;
}

// Normalize scores to sum to 1
SCL_FORCE_INLINE void normalize_l1(Real* scores, Size n) {
    Real sum = Real(0);
    for (Size i = 0; i < n; ++i) {
        sum += scores[i];
    }
    if (sum > config::MIN_SCORE) {
        Real inv_sum = Real(1) / sum;
        for (Size i = 0; i < n; ++i) {
            scores[i] *= inv_sum;
        }
    }
}

// Normalize scores to unit L2 norm
SCL_FORCE_INLINE void normalize_l2(Real* scores, Size n) {
    Real sum_sq = Real(0);
    for (Size i = 0; i < n; ++i) {
        sum_sq += scores[i] * scores[i];
    }
    if (sum_sq > config::MIN_SCORE) {
        Real inv_norm = Real(1) / std::sqrt(sum_sq);
        for (Size i = 0; i < n; ++i) {
            scores[i] *= inv_norm;
        }
    }
}

// Fast queue for BFS
struct FastQueue {
    Index* data;
    Size head;
    Size tail;

    SCL_FORCE_INLINE void clear() noexcept { head = 0; tail = 0; }
    SCL_FORCE_INLINE bool empty() const noexcept { return head == tail; }
    SCL_FORCE_INLINE void push(Index v) noexcept { data[tail++] = v; }
    SCL_FORCE_INLINE Index pop() noexcept { return data[head++]; }
};

// Simple PRNG
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
};

} // namespace detail

// =============================================================================
// Degree Centrality
// =============================================================================

template <typename T, bool IsCSR>
void degree_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    for (Index i = 0; i < n; ++i) {
        centrality[i] = static_cast<Real>(adjacency.primary_length(i));
    }

    if (normalize && n > 1) {
        Real max_deg = static_cast<Real>(n - 1);
        for (Index i = 0; i < n; ++i) {
            centrality[i] /= max_deg;
        }
    }
}

// =============================================================================
// Weighted Degree Centrality
// =============================================================================

template <typename T, bool IsCSR>
void weighted_degree_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    Real max_weight = Real(0);

    for (Index i = 0; i < n; ++i) {
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(values[k]);
        }

        centrality[i] = sum;
        max_weight = scl::algo::max2(max_weight, sum);
    }

    if (normalize && max_weight > config::MIN_SCORE) {
        for (Index i = 0; i < n; ++i) {
            centrality[i] /= max_weight;
        }
    }
}

// =============================================================================
// PageRank
// =============================================================================

template <typename T, bool IsCSR>
void pagerank(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> scores,
    Real damping = config::DEFAULT_DAMPING,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n),
                  "PageRank: output buffer too small");

    if (n == 0) return;

    // Compute out-degrees
    Real* out_degree = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* scores_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        Real deg = Real(0);
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);
        for (Index k = 0; k < len; ++k) {
            deg += static_cast<Real>(values[k]);
        }
        out_degree[i] = deg;
    }

    // Initialize scores uniformly
    Real init_score = Real(1) / static_cast<Real>(n);
    for (Index i = 0; i < n; ++i) {
        scores[i] = init_score;
    }

    Real teleport = (Real(1) - damping) / static_cast<Real>(n);

    // Power iteration
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Initialize with teleportation
        for (Index i = 0; i < n; ++i) {
            scores_new[i] = teleport;
        }

        // Distribute scores via edges
        // For CSR: row i has outgoing edges to adjacency.row_indices(i)
        // scores_new[j] += damping * scores[i] * weight[i,j] / out_degree[i]
        for (Index i = 0; i < n; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) continue;

            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real contrib = damping * scores[i] / out_degree[i];

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                scores_new[j] += contrib * static_cast<Real>(values[k]);
            }
        }

        // Handle dangling nodes (redistribute to all)
        Real dangling_sum = Real(0);
        for (Index i = 0; i < n; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) {
                dangling_sum += scores[i];
            }
        }
        Real dangling_contrib = damping * dangling_sum / static_cast<Real>(n);
        for (Index i = 0; i < n; ++i) {
            scores_new[i] += dangling_contrib;
        }

        // Check convergence
        if (detail::check_convergence(scores.ptr, scores_new, n, tol)) {
            for (Index i = 0; i < n; ++i) {
                scores[i] = scores_new[i];
            }
            break;
        }

        // Swap
        for (Index i = 0; i < n; ++i) {
            scores[i] = scores_new[i];
        }
    }

    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(out_degree, SCL_ALIGNMENT);
}

// =============================================================================
// Personalized PageRank
// =============================================================================

template <typename T, bool IsCSR>
void personalized_pagerank(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Index> seed_nodes,
    Array<Real> scores,
    Real damping = config::DEFAULT_DAMPING,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n),
                  "PageRank: output buffer too small");

    if (n == 0 || seed_nodes.len == 0) return;

    Real* out_degree = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* scores_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* personalization = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    // Compute out-degrees
    for (Index i = 0; i < n; ++i) {
        Real deg = Real(0);
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);
        for (Index k = 0; k < len; ++k) {
            deg += static_cast<Real>(values[k]);
        }
        out_degree[i] = deg;
    }

    // Build personalization vector
    scl::algo::zero(personalization, static_cast<Size>(n));
    Real seed_weight = Real(1) / static_cast<Real>(seed_nodes.len);
    for (Size s = 0; s < seed_nodes.len; ++s) {
        Index idx = seed_nodes[s];
        if (idx >= 0 && idx < n) {
            personalization[idx] += seed_weight;
        }
    }

    // Initialize scores with personalization
    for (Index i = 0; i < n; ++i) {
        scores[i] = personalization[i];
    }

    // Power iteration
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Teleport to personalization
        for (Index i = 0; i < n; ++i) {
            scores_new[i] = (Real(1) - damping) * personalization[i];
        }

        // Distribute via edges
        for (Index i = 0; i < n; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) continue;

            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real contrib = damping * scores[i] / out_degree[i];

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                scores_new[j] += contrib * static_cast<Real>(values[k]);
            }
        }

        // Dangling nodes teleport to personalization
        Real dangling_sum = Real(0);
        for (Index i = 0; i < n; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) {
                dangling_sum += scores[i];
            }
        }
        for (Index i = 0; i < n; ++i) {
            scores_new[i] += damping * dangling_sum * personalization[i];
        }

        if (detail::check_convergence(scores.ptr, scores_new, n, tol)) {
            for (Index i = 0; i < n; ++i) {
                scores[i] = scores_new[i];
            }
            break;
        }

        for (Index i = 0; i < n; ++i) {
            scores[i] = scores_new[i];
        }
    }

    scl::memory::aligned_free(personalization, SCL_ALIGNMENT);
    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(out_degree, SCL_ALIGNMENT);
}

// =============================================================================
// HITS Algorithm (Hub and Authority Scores)
// =============================================================================

template <typename T, bool IsCSR>
void hits(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> hub_scores,
    Array<Real> authority_scores,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(hub_scores.len >= static_cast<Size>(n),
                  "HITS: hub buffer too small");
    SCL_CHECK_DIM(authority_scores.len >= static_cast<Size>(n),
                  "HITS: authority buffer too small");

    if (n == 0) return;

    Real* hub_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* auth_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    // Initialize uniformly
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    for (Index i = 0; i < n; ++i) {
        hub_scores[i] = init_score;
        authority_scores[i] = init_score;
    }

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Authority update: auth[i] = sum_j(hub[j] * weight[j,i])
        // For CSR: column i receives from rows that link to it
        scl::algo::zero(auth_new, static_cast<Size>(n));
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                auth_new[j] += hub_scores[i] * static_cast<Real>(values[k]);
            }
        }

        // Hub update: hub[i] = sum_j(auth[j] * weight[i,j])
        scl::algo::zero(hub_new, static_cast<Size>(n));
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                sum += auth_new[j] * static_cast<Real>(values[k]);
            }
            hub_new[i] = sum;
        }

        // Normalize
        detail::normalize_l2(auth_new, n);
        detail::normalize_l2(hub_new, n);

        // Check convergence
        bool auth_conv = detail::check_convergence(authority_scores.ptr, auth_new, n, tol);
        bool hub_conv = detail::check_convergence(hub_scores.ptr, hub_new, n, tol);

        for (Index i = 0; i < n; ++i) {
            authority_scores[i] = auth_new[i];
            hub_scores[i] = hub_new[i];
        }

        if (auth_conv && hub_conv) break;
    }

    scl::memory::aligned_free(auth_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(hub_new, SCL_ALIGNMENT);
}

// =============================================================================
// Eigenvector Centrality
// =============================================================================

template <typename T, bool IsCSR>
void eigenvector_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    if (n == 0) return;

    Real* centrality_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    // Initialize uniformly
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    for (Index i = 0; i < n; ++i) {
        centrality[i] = init_score;
    }

    for (Index iter = 0; iter < max_iter; ++iter) {
        // c_new[i] = sum_j(A[i,j] * c[j])
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                sum += static_cast<Real>(values[k]) * centrality[j];
            }
            centrality_new[i] = sum;
        }

        // Normalize
        detail::normalize_l2(centrality_new, n);

        if (detail::check_convergence(centrality.ptr, centrality_new, n, tol)) {
            for (Index i = 0; i < n; ++i) {
                centrality[i] = centrality_new[i];
            }
            break;
        }

        for (Index i = 0; i < n; ++i) {
            centrality[i] = centrality_new[i];
        }
    }

    scl::memory::aligned_free(centrality_new, SCL_ALIGNMENT);
}

// =============================================================================
// Closeness Centrality
// =============================================================================

template <typename T, bool IsCSR>
void closeness_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    if (n <= 1) {
        for (Index i = 0; i < n; ++i) {
            centrality[i] = Real(0);
        }
        return;
    }

    // BFS from each node
    Index* distances = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* queue = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    for (Index s = 0; s < n; ++s) {
        // BFS from s
        scl::algo::fill(distances, static_cast<Size>(n), Index(-1));

        Size queue_head = 0, queue_tail = 0;
        distances[s] = 0;
        queue[queue_tail++] = s;

        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];

            auto indices = adjacency.primary_indices(u);
            const Index len = adjacency.primary_length(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];
                if (distances[v] == -1) {
                    distances[v] = distances[u] + 1;
                    queue[queue_tail++] = v;
                }
            }
        }

        // Compute closeness: n-1 / sum of distances
        int64_t total_dist = 0;
        Index reachable = 0;
        for (Index i = 0; i < n; ++i) {
            if (i != s && distances[i] > 0) {
                total_dist += distances[i];
                ++reachable;
            }
        }

        if (total_dist > 0) {
            centrality[s] = static_cast<Real>(reachable) / static_cast<Real>(total_dist);
            if (normalize && reachable < n - 1) {
                // Normalize by fraction of reachable nodes
                centrality[s] *= static_cast<Real>(reachable) / static_cast<Real>(n - 1);
            }
        } else {
            centrality[s] = Real(0);
        }
    }

    scl::memory::aligned_free(queue, SCL_ALIGNMENT);
    scl::memory::aligned_free(distances, SCL_ALIGNMENT);
}

// =============================================================================
// Betweenness Centrality (Brandes Algorithm)
// =============================================================================

template <typename T, bool IsCSR>
void betweenness_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    if (n <= 2) {
        for (Index i = 0; i < n; ++i) {
            centrality[i] = Real(0);
        }
        return;
    }

    scl::algo::zero(centrality.ptr, static_cast<Size>(n));

    // Allocate per-source workspace
    Index* distances = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* sigma = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);      // Number of shortest paths
    Real* delta = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);      // Dependency
    Index* queue = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* stack = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    // Predecessor lists stored as flat arrays
    Size max_pred_size = adjacency.nnz();
    Index* pred_data = scl::memory::aligned_alloc<Index>(max_pred_size, SCL_ALIGNMENT);
    Index* pred_ptr = scl::memory::aligned_alloc<Index>(n + 1, SCL_ALIGNMENT);

    for (Index s = 0; s < n; ++s) {
        // Initialize
        scl::algo::fill(distances, static_cast<Size>(n), Index(-1));
        scl::algo::zero(sigma, static_cast<Size>(n));
        scl::algo::zero(delta, static_cast<Size>(n));

        for (Index i = 0; i <= n; ++i) {
            pred_ptr[i] = 0;
        }

        distances[s] = 0;
        sigma[s] = Real(1);

        Size queue_head = 0, queue_tail = 0;
        Size stack_top = 0;

        queue[queue_tail++] = s;

        // BFS phase
        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];
            stack[stack_top++] = u;

            auto indices = adjacency.primary_indices(u);
            const Index len = adjacency.primary_length(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (distances[v] == -1) {
                    // First visit
                    distances[v] = distances[u] + 1;
                    queue[queue_tail++] = v;
                }

                if (distances[v] == distances[u] + 1) {
                    // Shortest path via u
                    sigma[v] += sigma[u];
                    // Store predecessor (simple version - count only)
                }
            }
        }

        // Back-propagation phase (simplified - use distances for ordering)
        // Process nodes in reverse BFS order
        while (stack_top > 0) {
            Index w = stack[--stack_top];

            auto indices = adjacency.primary_indices(w);
            const Index len = adjacency.primary_length(w);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];
                if (distances[v] == distances[w] - 1) {
                    // v is a predecessor of w
                    delta[v] += (sigma[v] / sigma[w]) * (Real(1) + delta[w]);
                }
            }

            if (w != s) {
                centrality[w] += delta[w];
            }
        }
    }

    // For undirected graphs, divide by 2
    for (Index i = 0; i < n; ++i) {
        centrality[i] /= Real(2);
    }

    if (normalize && n > 2) {
        Real norm_factor = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        for (Index i = 0; i < n; ++i) {
            centrality[i] *= norm_factor;
        }
    }

    scl::memory::aligned_free(pred_ptr, SCL_ALIGNMENT);
    scl::memory::aligned_free(pred_data, SCL_ALIGNMENT);
    scl::memory::aligned_free(stack, SCL_ALIGNMENT);
    scl::memory::aligned_free(queue, SCL_ALIGNMENT);
    scl::memory::aligned_free(delta, SCL_ALIGNMENT);
    scl::memory::aligned_free(sigma, SCL_ALIGNMENT);
    scl::memory::aligned_free(distances, SCL_ALIGNMENT);
}

// =============================================================================
// Sampled Betweenness Centrality (Approximate)
// =============================================================================

template <typename T, bool IsCSR>
void betweenness_centrality_sampled(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    Index n_samples,
    bool normalize = true,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    if (n <= 2) {
        for (Index i = 0; i < n; ++i) {
            centrality[i] = Real(0);
        }
        return;
    }

    n_samples = scl::algo::min2(n_samples, n);
    scl::algo::zero(centrality.ptr, static_cast<Size>(n));

    // Allocate workspace
    Index* distances = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* sigma = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* delta = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* queue = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* stack = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    bool* sampled = reinterpret_cast<bool*>(scl::memory::aligned_alloc<char>(n, SCL_ALIGNMENT));

    std::memset(sampled, 0, n);

    detail::FastRNG rng(seed);

    // Sample source nodes
    Index samples_done = 0;
    while (samples_done < n_samples) {
        Index s = static_cast<Index>(rng.bounded(n));
        if (sampled[s]) continue;
        sampled[s] = true;

        // BFS from s (same as full betweenness)
        scl::algo::fill(distances, static_cast<Size>(n), Index(-1));
        scl::algo::zero(sigma, static_cast<Size>(n));
        scl::algo::zero(delta, static_cast<Size>(n));

        distances[s] = 0;
        sigma[s] = Real(1);

        Size queue_head = 0, queue_tail = 0;
        Size stack_top = 0;

        queue[queue_tail++] = s;

        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];
            stack[stack_top++] = u;

            auto indices = adjacency.primary_indices(u);
            const Index len = adjacency.primary_length(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (distances[v] == -1) {
                    distances[v] = distances[u] + 1;
                    queue[queue_tail++] = v;
                }

                if (distances[v] == distances[u] + 1) {
                    sigma[v] += sigma[u];
                }
            }
        }

        while (stack_top > 0) {
            Index w = stack[--stack_top];

            auto indices = adjacency.primary_indices(w);
            const Index len = adjacency.primary_length(w);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];
                if (distances[v] == distances[w] - 1) {
                    delta[v] += (sigma[v] / sigma[w]) * (Real(1) + delta[w]);
                }
            }

            if (w != s) {
                centrality[w] += delta[w];
            }
        }

        ++samples_done;
    }

    // Scale by n/n_samples to get unbiased estimate
    Real scale = static_cast<Real>(n) / static_cast<Real>(n_samples) / Real(2);
    for (Index i = 0; i < n; ++i) {
        centrality[i] *= scale;
    }

    if (normalize && n > 2) {
        Real norm_factor = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        for (Index i = 0; i < n; ++i) {
            centrality[i] *= norm_factor;
        }
    }

    scl::memory::aligned_free(reinterpret_cast<char*>(sampled), SCL_ALIGNMENT);
    scl::memory::aligned_free(stack, SCL_ALIGNMENT);
    scl::memory::aligned_free(queue, SCL_ALIGNMENT);
    scl::memory::aligned_free(delta, SCL_ALIGNMENT);
    scl::memory::aligned_free(sigma, SCL_ALIGNMENT);
    scl::memory::aligned_free(distances, SCL_ALIGNMENT);
}

// =============================================================================
// Katz Centrality
// =============================================================================

template <typename T, bool IsCSR>
void katz_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    Real alpha = Real(0.1),
    Real beta = Real(1.0),
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(centrality.len >= static_cast<Size>(n),
                  "Centrality: output buffer too small");

    if (n == 0) return;

    Real* centrality_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    // Initialize
    for (Index i = 0; i < n; ++i) {
        centrality[i] = beta;
    }

    // Iterative: x = alpha * A * x + beta
    for (Index iter = 0; iter < max_iter; ++iter) {
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                sum += static_cast<Real>(values[k]) * centrality[j];
            }
            centrality_new[i] = alpha * sum + beta;
        }

        if (detail::check_convergence(centrality.ptr, centrality_new, n, tol)) {
            for (Index i = 0; i < n; ++i) {
                centrality[i] = centrality_new[i];
            }
            break;
        }

        for (Index i = 0; i < n; ++i) {
            centrality[i] = centrality_new[i];
        }
    }

    scl::memory::aligned_free(centrality_new, SCL_ALIGNMENT);
}

} // namespace scl::kernel::centrality
