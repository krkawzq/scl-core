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

// =============================================================================
// FILE: scl/kernel/centrality.hpp
// BRIEF: High-performance graph centrality measures
//
// Optimizations applied:
// - Parallel PageRank/HITS/Eigenvector with atomic accumulation
// - SIMD-accelerated normalization and convergence checks
// - Parallel Brandes algorithm for betweenness
// - Optimized BFS with prefetching
// - Cache-aligned workspace structures
// - Multi-source parallel closeness
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
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// SIMD-Accelerated Vector Operations
// =============================================================================

namespace detail {

SCL_HOT SCL_FORCE_INLINE Real sum_simd(const Real* SCL_RESTRICT x, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, x + k));
        v_sum1 = s::Add(v_sum1, s::Load(d, x + k + lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, x + k + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, x + k + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        result += x[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(const Real* SCL_RESTRICT x, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto v0 = s::Load(d, x + k);
        auto v1 = s::Load(d, x + k + lanes);
        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        result += x[k] * x[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real l1_diff_simd(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        auto diff = s::Sub(s::Load(d, a + k), s::Load(d, b + k));
        v_sum = s::Add(v_sum, s::Abs(diff));
    }

    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        Real diff = a[k] - b[k];
        result += (diff >= Real(0)) ? diff : -diff;
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE void scale_simd(
    Real* SCL_RESTRICT x,
    Real alpha,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x + k)), d, x + k);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + lanes)), d, x + k + lanes);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + 2*lanes)), d, x + k + 2*lanes);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + 3*lanes)), d, x + k + 3*lanes);
    }

    for (; k < n; ++k) {
        x[k] *= alpha;
    }
}

SCL_HOT SCL_FORCE_INLINE void axpby_simd(
    Real alpha,
    const Real* SCL_RESTRICT x,
    Real beta,
    Real* SCL_RESTRICT y,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);
    auto v_beta = s::Set(d, beta);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto y0 = s::Mul(v_beta, s::Load(d, y + k));
        auto y1 = s::Mul(v_beta, s::Load(d, y + k + lanes));
        y0 = s::MulAdd(v_alpha, s::Load(d, x + k), y0);
        y1 = s::MulAdd(v_alpha, s::Load(d, x + k + lanes), y1);
        s::Store(y0, d, y + k);
        s::Store(y1, d, y + k + lanes);
    }

    for (; k < n; ++k) {
        y[k] = alpha * x[k] + beta * y[k];
    }
}

// Normalize to sum = 1
SCL_FORCE_INLINE void normalize_l1(Real* scores, Size n) {
    Real sum = sum_simd(scores, n);
    if (sum > config::MIN_SCORE) {
        scale_simd(scores, Real(1) / sum, n);
    }
}

// Normalize to L2 norm = 1
SCL_FORCE_INLINE void normalize_l2(Real* scores, Size n) {
    Real norm_sq = norm_squared_simd(scores, n);
    if (norm_sq > config::MIN_SCORE) {
        scale_simd(scores, Real(1) / std::sqrt(norm_sq), n);
    }
}

// Check convergence
SCL_FORCE_INLINE bool check_convergence(
    const Real* old_scores,
    const Real* new_scores,
    Size n,
    Real tol
) {
    return l1_diff_simd(old_scores, new_scores, n) < tol;
}

// =============================================================================
// Fast PRNG
// =============================================================================

struct alignas(16) FastRNG {
    uint32_t s[4];

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s[i] = static_cast<uint32_t>(z >> 32);
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

    SCL_FORCE_INLINE Index bounded(Index n) noexcept {
        return static_cast<Index>(next() % static_cast<uint32_t>(n));
    }
};

// =============================================================================
// Optimized BFS Queue
// =============================================================================

struct alignas(64) FastQueue {
    Index* data;
    Size head;
    Size tail;
    Size capacity;

    void init(Size cap) {
        capacity = cap;
        data = scl::memory::aligned_alloc<Index>(cap, SCL_ALIGNMENT);
        clear();
    }

    void destroy() {
        if (data) scl::memory::aligned_free(data, SCL_ALIGNMENT);
    }

    SCL_FORCE_INLINE void clear() noexcept { head = 0; tail = 0; }
    SCL_FORCE_INLINE bool empty() const noexcept { return head == tail; }
    SCL_FORCE_INLINE Size size() const noexcept { return tail - head; }
    SCL_FORCE_INLINE void push(Index v) noexcept { data[tail++] = v; }
    SCL_FORCE_INLINE Index pop() noexcept { return data[head++]; }
    SCL_FORCE_INLINE Index pop_prefetch() noexcept {
        if (SCL_LIKELY(head + config::PREFETCH_DISTANCE < tail)) {
            SCL_PREFETCH_READ(&data[head + config::PREFETCH_DISTANCE], 0);
        }
        return data[head++];
    }
};

// =============================================================================
// Parallel Accumulator for PageRank-style algorithms
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void parallel_distribute(
    const Sparse<T, IsCSR>& adj,
    const Real* SCL_RESTRICT scores,
    const Real* SCL_RESTRICT out_degree_inv,
    Real* SCL_RESTRICT scores_new,
    Size n,
    Real damping
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        // Use atomic accumulation for parallel distribution
        std::atomic<int64_t>* atomic_scores = static_cast<std::atomic<int64_t>*>(
            scl::memory::aligned_alloc<int64_t>(n, SCL_ALIGNMENT));

        // Initialize
        for (Size i = 0; i < n; ++i) {
            atomic_scores[i].store(0, std::memory_order_relaxed);
        }

        constexpr int64_t SCALE = 1000000000LL;

        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            Real contrib = damping * scores[i] * out_degree_inv[i];

            if (contrib <= config::MIN_SCORE) return;

            auto indices = adj.primary_indices(idx);
            auto values = adj.primary_values(idx);
            const Index len = adj.primary_length(idx);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                int64_t delta = static_cast<int64_t>(contrib * static_cast<Real>(values[k]) * SCALE);
                atomic_scores[j].fetch_add(delta, std::memory_order_relaxed);
            }
        });

        // Convert back
        Real inv_scale = Real(1) / static_cast<Real>(SCALE);
        for (Size i = 0; i < n; ++i) {
            scores_new[i] += static_cast<Real>(atomic_scores[i].load()) * inv_scale;
        }

        scl::memory::aligned_free(reinterpret_cast<int64_t*>(atomic_scores), SCL_ALIGNMENT);
    } else {
        // Sequential distribution
        for (Size i = 0; i < n; ++i) {
            const Index idx = static_cast<Index>(i);
            Real contrib = damping * scores[i] * out_degree_inv[i];

            if (contrib <= config::MIN_SCORE) continue;

            auto indices = adj.primary_indices(idx);
            auto values = adj.primary_values(idx);
            const Index len = adj.primary_length(idx);

            // 4-way unrolled
            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                scores_new[indices[k + 0]] += contrib * static_cast<Real>(values[k + 0]);
                scores_new[indices[k + 1]] += contrib * static_cast<Real>(values[k + 1]);
                scores_new[indices[k + 2]] += contrib * static_cast<Real>(values[k + 2]);
                scores_new[indices[k + 3]] += contrib * static_cast<Real>(values[k + 3]);
            }

            for (; k < len; ++k) {
                scores_new[indices[k]] += contrib * static_cast<Real>(values[k]);
            }
        }
    }
}

// =============================================================================
// Parallel SpMV for centrality (y = A * x)
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmv_centrality(
    const Sparse<T, IsCSR>& adj,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            auto indices = adj.primary_indices(idx);
            auto values = adj.primary_values(idx);
            const Index len = adj.primary_length(idx);

            Real sum = Real(0);

            // Prefetch
            if (SCL_LIKELY(i + 1 < n)) {
                SCL_PREFETCH_READ(adj.primary_indices(idx + 1).ptr, 0);
            }

            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                sum += static_cast<Real>(values[k + 0]) * x[indices[k + 0]];
                sum += static_cast<Real>(values[k + 1]) * x[indices[k + 1]];
                sum += static_cast<Real>(values[k + 2]) * x[indices[k + 2]];
                sum += static_cast<Real>(values[k + 3]) * x[indices[k + 3]];
            }

            for (; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }

            y[i] = sum;
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const Index idx = static_cast<Index>(i);
            auto indices = adj.primary_indices(idx);
            auto values = adj.primary_values(idx);
            const Index len = adj.primary_length(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        }
    }
}

// Compute weighted out-degrees
template <typename T, bool IsCSR>
void compute_out_degrees(
    const Sparse<T, IsCSR>& adj,
    Real* out_degree,
    Real* out_degree_inv,
    Size n
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

            out_degree[i] = deg;
            out_degree_inv[i] = (deg > config::MIN_SCORE) ? Real(1) / deg : Real(0);
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            auto values = adj.primary_values(static_cast<Index>(i));
            const Index len = adj.primary_length(static_cast<Index>(i));

            Real deg = Real(0);
            for (Index k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }

            out_degree[i] = deg;
            out_degree_inv[i] = (deg > config::MIN_SCORE) ? Real(1) / deg : Real(0);
        }
    }
}

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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            centrality[i] = static_cast<Real>(adjacency.primary_length(static_cast<Index>(i)));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            centrality[i] = static_cast<Real>(adjacency.primary_length(i));
        }
    }

    if (normalize && n > 1) {
        Real inv_max = Real(1) / static_cast<Real>(n - 1);
        detail::scale_simd(centrality.ptr, inv_max, N);
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto values = adjacency.primary_values(static_cast<Index>(i));
            const Index len = adjacency.primary_length(static_cast<Index>(i));

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            centrality[i] = sum;
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            centrality[i] = sum;
        }
    }

    if (normalize) {
        // Find max and normalize
        Real max_weight = Real(0);
        for (Size i = 0; i < N; ++i) {
            max_weight = scl::algo::max2(max_weight, centrality[i]);
        }

        if (max_weight > config::MIN_SCORE) {
            detail::scale_simd(centrality.ptr, Real(1) / max_weight, N);
        }
    }
}

// =============================================================================
// PageRank (Optimized)
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(scores.len >= N, "PageRank: output buffer too small");

    if (n == 0) return;

    // Allocate working memory
    Real* out_degree = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* out_degree_inv = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* scores_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Compute out-degrees
    detail::compute_out_degrees(adjacency, out_degree, out_degree_inv, N);

    // Initialize scores uniformly
    Real init_score = Real(1) / static_cast<Real>(n);
    scl::algo::fill(scores.ptr, N, init_score);

    Real teleport = (Real(1) - damping) / static_cast<Real>(n);
    Real dangling_factor = damping / static_cast<Real>(n);

    // Power iteration
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Initialize with teleportation
        scl::algo::fill(scores_new, N, teleport);

        // Distribute via edges
        detail::parallel_distribute(adjacency, scores.ptr, out_degree_inv, 
                                    scores_new, N, damping);

        // Handle dangling nodes
        Real dangling_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) {
                dangling_sum += scores[i];
            }
        }

        if (dangling_sum > config::MIN_SCORE) {
            Real dangling_contrib = dangling_factor * dangling_sum;
            const size_t n_threads = scl::threading::Scheduler::get_num_threads();
            
            if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
                scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                    scores_new[i] += dangling_contrib;
                });
            } else {
                for (Size i = 0; i < N; ++i) {
                    scores_new[i] += dangling_contrib;
                }
            }
        }

        // Check convergence
        if (detail::check_convergence(scores.ptr, scores_new, N, tol)) {
            scl::algo::copy(scores_new, scores.ptr, N);
            break;
        }

        scl::algo::swap(scores.ptr, scores_new);
    }

    // Ensure result is in output array
    if (scores.ptr == scores_new) {
        // Already in place
    }

    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(out_degree_inv, SCL_ALIGNMENT);
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(scores.len >= N, "PageRank: output buffer too small");

    if (n == 0 || seed_nodes.len == 0) return;

    Real* out_degree = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* out_degree_inv = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* scores_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* personalization = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    detail::compute_out_degrees(adjacency, out_degree, out_degree_inv, N);

    // Build personalization vector
    scl::algo::zero(personalization, N);
    Size valid_seeds = 0;
    for (Size s = 0; s < seed_nodes.len; ++s) {
        Index idx = seed_nodes[s];
        if (idx >= 0 && idx < n) {
            personalization[idx] += Real(1);
            ++valid_seeds;
        }
    }

    if (valid_seeds > 0) {
        detail::scale_simd(personalization, Real(1) / static_cast<Real>(valid_seeds), N);
    }

    // Initialize
    scl::algo::copy(personalization, scores.ptr, N);

    Real one_minus_d = Real(1) - damping;

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Teleport to personalization
        for (Size i = 0; i < N; ++i) {
            scores_new[i] = one_minus_d * personalization[i];
        }

        // Distribute
        detail::parallel_distribute(adjacency, scores.ptr, out_degree_inv,
                                    scores_new, N, damping);

        // Dangling nodes
        Real dangling_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (out_degree[i] <= config::MIN_SCORE) {
                dangling_sum += scores[i];
            }
        }

        if (dangling_sum > config::MIN_SCORE) {
            detail::axpby_simd(damping * dangling_sum, personalization, 
                               Real(1), scores_new, N);
        }

        if (detail::check_convergence(scores.ptr, scores_new, N, tol)) {
            scl::algo::copy(scores_new, scores.ptr, N);
            break;
        }

        scl::algo::copy(scores_new, scores.ptr, N);
    }

    scl::memory::aligned_free(personalization, SCL_ALIGNMENT);
    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(out_degree_inv, SCL_ALIGNMENT);
    scl::memory::aligned_free(out_degree, SCL_ALIGNMENT);
}

// =============================================================================
// HITS Algorithm (Optimized)
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(hub_scores.len >= N, "HITS: hub buffer too small");
    SCL_CHECK_DIM(authority_scores.len >= N, "HITS: authority buffer too small");

    if (n == 0) return;

    Real* hub_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* auth_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Initialize
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    scl::algo::fill(hub_scores.ptr, N, init_score);
    scl::algo::fill(authority_scores.ptr, N, init_score);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Authority update: auth[j] = sum_i(hub[i] * A[i,j])
        scl::algo::zero(auth_new, N);

        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            // Parallel with atomics
            std::atomic<int64_t>* atomic_auth = static_cast<std::atomic<int64_t>*>(
                scl::memory::aligned_alloc<int64_t>(N, SCL_ALIGNMENT));

            for (Size i = 0; i < N; ++i) {
                atomic_auth[i].store(0, std::memory_order_relaxed);
            }

            constexpr int64_t SCALE = 1000000000LL;

            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                auto indices = adjacency.primary_indices(static_cast<Index>(i));
                auto values = adjacency.primary_values(static_cast<Index>(i));
                const Index len = adjacency.primary_length(static_cast<Index>(i));

                Real hub_i = hub_scores[i];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    int64_t delta = static_cast<int64_t>(hub_i * static_cast<Real>(values[k]) * SCALE);
                    atomic_auth[j].fetch_add(delta, std::memory_order_relaxed);
                }
            });

            Real inv_scale = Real(1) / static_cast<Real>(SCALE);
            for (Size i = 0; i < N; ++i) {
                auth_new[i] = static_cast<Real>(atomic_auth[i].load()) * inv_scale;
            }

            scl::memory::aligned_free(reinterpret_cast<int64_t*>(atomic_auth), SCL_ALIGNMENT);
        } else {
            for (Index i = 0; i < n; ++i) {
                auto indices = adjacency.primary_indices(i);
                auto values = adjacency.primary_values(i);
                const Index len = adjacency.primary_length(i);

                Real hub_i = hub_scores[i];

                for (Index k = 0; k < len; ++k) {
                    auth_new[indices[k]] += hub_i * static_cast<Real>(values[k]);
                }
            }
        }

        // Hub update: hub[i] = sum_j(auth[j] * A[i,j])
        detail::spmv_centrality(adjacency, auth_new, hub_new, N);

        // Normalize
        detail::normalize_l2(auth_new, N);
        detail::normalize_l2(hub_new, N);

        // Check convergence
        bool converged = detail::check_convergence(authority_scores.ptr, auth_new, N, tol) &&
                         detail::check_convergence(hub_scores.ptr, hub_new, N, tol);

        scl::algo::copy(auth_new, authority_scores.ptr, N);
        scl::algo::copy(hub_new, hub_scores.ptr, N);

        if (converged) break;
    }

    scl::memory::aligned_free(auth_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(hub_new, SCL_ALIGNMENT);
}

// =============================================================================
// Eigenvector Centrality (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void eigenvector_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n == 0) return;

    Real* centrality_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Initialize
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    scl::algo::fill(centrality.ptr, N, init_score);

    for (Index iter = 0; iter < max_iter; ++iter) {
        detail::spmv_centrality(adjacency, centrality.ptr, centrality_new, N);

        detail::normalize_l2(centrality_new, N);

        if (detail::check_convergence(centrality.ptr, centrality_new, N, tol)) {
            scl::algo::copy(centrality_new, centrality.ptr, N);
            break;
        }

        scl::algo::copy(centrality_new, centrality.ptr, N);
    }

    scl::memory::aligned_free(centrality_new, SCL_ALIGNMENT);
}

// =============================================================================
// Katz Centrality (Optimized)
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n == 0) return;

    Real* centrality_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Initialize
    scl::algo::fill(centrality.ptr, N, beta);

    for (Index iter = 0; iter < max_iter; ++iter) {
        // temp = A * centrality
        detail::spmv_centrality(adjacency, centrality.ptr, temp, N);

        // centrality_new = alpha * temp + beta
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                centrality_new[i] = alpha * temp[i] + beta;
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                centrality_new[i] = alpha * temp[i] + beta;
            }
        }

        if (detail::check_convergence(centrality.ptr, centrality_new, N, tol)) {
            scl::algo::copy(centrality_new, centrality.ptr, N);
            break;
        }

        scl::algo::copy(centrality_new, centrality.ptr, N);
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(centrality_new, SCL_ALIGNMENT);
}

// =============================================================================
// Closeness Centrality (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void closeness_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n <= 1) {
        scl::algo::zero(centrality.ptr, N);
        return;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::WorkspacePool<Index> dist_pool;
    scl::threading::WorkspacePool<Index> queue_pool;
    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), N, [&](size_t s, size_t thread_rank) {
        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);

        scl::algo::fill(distances, N, Index(-1));

        Size queue_head = 0, queue_tail = 0;
        distances[s] = 0;
        queue[queue_tail++] = static_cast<Index>(s);

        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];

            // Prefetch
            if (queue_head + config::PREFETCH_DISTANCE < queue_tail) {
                Index next_u = queue[queue_head + config::PREFETCH_DISTANCE];
                SCL_PREFETCH_READ(adjacency.primary_indices(next_u).ptr, 0);
            }

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

        // Compute closeness
        int64_t total_dist = 0;
        Index reachable = 0;

        for (Size i = 0; i < N; ++i) {
            if (i != s && distances[i] > 0) {
                total_dist += distances[i];
                ++reachable;
            }
        }

        if (total_dist > 0) {
            centrality[s] = static_cast<Real>(reachable) / static_cast<Real>(total_dist);
            if (normalize && reachable < n - 1) {
                centrality[s] *= static_cast<Real>(reachable) / static_cast<Real>(n - 1);
            }
        } else {
            centrality[s] = Real(0);
        }
    });
}

// =============================================================================
// Betweenness Centrality (Parallel Brandes)
// =============================================================================

template <typename T, bool IsCSR>
void betweenness_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n <= 2) {
        scl::algo::zero(centrality.ptr, N);
        return;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread partial centrality
    Real* thread_centrality = scl::memory::aligned_alloc<Real>(n_threads * N, SCL_ALIGNMENT);
    scl::algo::zero(thread_centrality, n_threads * N);

    // Per-thread workspace
    struct BrandesWorkspace {
        Index* distances;
        Real* sigma;
        Real* delta;
        Index* queue;
        Index* stack;
    };

    BrandesWorkspace* workspaces = static_cast<BrandesWorkspace*>(
        scl::memory::aligned_alloc<BrandesWorkspace>(n_threads, SCL_ALIGNMENT));

    for (size_t t = 0; t < n_threads; ++t) {
        workspaces[t].distances = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
        workspaces[t].sigma = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        workspaces[t].delta = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        workspaces[t].queue = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
        workspaces[t].stack = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    }

    scl::threading::parallel_for(Size(0), N, [&](size_t s, size_t thread_rank) {
        auto& ws = workspaces[thread_rank];
        Real* local_centrality = thread_centrality + thread_rank * N;

        scl::algo::fill(ws.distances, N, Index(-1));
        scl::algo::zero(ws.sigma, N);
        scl::algo::zero(ws.delta, N);

        ws.distances[s] = 0;
        ws.sigma[s] = Real(1);

        Size queue_head = 0, queue_tail = 0;
        Size stack_top = 0;
        ws.queue[queue_tail++] = static_cast<Index>(s);

        // BFS
        while (queue_head < queue_tail) {
            Index u = ws.queue[queue_head++];
            ws.stack[stack_top++] = u;

            auto indices = adjacency.primary_indices(u);
            const Index len = adjacency.primary_length(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (ws.distances[v] == -1) {
                    ws.distances[v] = ws.distances[u] + 1;
                    ws.queue[queue_tail++] = v;
                }

                if (ws.distances[v] == ws.distances[u] + 1) {
                    ws.sigma[v] += ws.sigma[u];
                }
            }
        }

        // Back-propagation
        while (stack_top > 0) {
            Index w = ws.stack[--stack_top];
            auto indices = adjacency.primary_indices(w);
            const Index len = adjacency.primary_length(w);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (ws.distances[v] == ws.distances[w] - 1) {
                    ws.delta[v] += (ws.sigma[v] / ws.sigma[w]) * (Real(1) + ws.delta[w]);
                }
            }

            if (w != static_cast<Index>(s)) {
                local_centrality[w] += ws.delta[w];
            }
        }
    });

    // Reduce thread results
    scl::algo::zero(centrality.ptr, N);
    for (size_t t = 0; t < n_threads; ++t) {
        Real* local = thread_centrality + t * N;
        for (Size i = 0; i < N; ++i) {
            centrality[i] += local[i];
        }
    }

    // Undirected graph: divide by 2
    detail::scale_simd(centrality.ptr, Real(0.5), N);

    if (normalize && n > 2) {
        Real norm = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        detail::scale_simd(centrality.ptr, norm, N);
    }

    // Cleanup
    for (size_t t = 0; t < n_threads; ++t) {
        scl::memory::aligned_free(workspaces[t].stack, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].queue, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].delta, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].sigma, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].distances, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(workspaces, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_centrality, SCL_ALIGNMENT);
}

// =============================================================================
// Sampled Betweenness (Approximate, Faster)
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n <= 2) {
        scl::algo::zero(centrality.ptr, N);
        return;
    }

    n_samples = scl::algo::min2(n_samples, n);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Generate sample indices
    Index* samples = scl::memory::aligned_alloc<Index>(n_samples, SCL_ALIGNMENT);
    bool* sampled = reinterpret_cast<bool*>(scl::memory::aligned_alloc<char>(N, SCL_ALIGNMENT));
    scl::algo::zero(sampled, N);

    detail::FastRNG rng(seed);
    Index samples_done = 0;

    while (samples_done < n_samples) {
        Index s = rng.bounded(n);
        if (!sampled[s]) {
            sampled[s] = true;
            samples[samples_done++] = s;
        }
    }

    // Per-thread centrality
    Real* thread_centrality = scl::memory::aligned_alloc<Real>(n_threads * N, SCL_ALIGNMENT);
    scl::algo::zero(thread_centrality, n_threads * N);

    // Per-thread workspace
    scl::threading::WorkspacePool<Index> dist_pool, queue_pool, stack_pool;
    scl::threading::WorkspacePool<Real> sigma_pool, delta_pool;

    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);
    stack_pool.init(n_threads, N);
    sigma_pool.init(n_threads, N);
    delta_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_samples), [&](size_t idx, size_t thread_rank) {
        Index s = samples[idx];

        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);
        Index* stack = stack_pool.get(thread_rank);
        Real* sigma = sigma_pool.get(thread_rank);
        Real* delta = delta_pool.get(thread_rank);

        Real* local_centrality = thread_centrality + thread_rank * N;

        scl::algo::fill(distances, N, Index(-1));
        scl::algo::zero(sigma, N);
        scl::algo::zero(delta, N);

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
                local_centrality[w] += delta[w];
            }
        }
    });

    // Reduce
    scl::algo::zero(centrality.ptr, N);
    for (size_t t = 0; t < n_threads; ++t) {
        Real* local = thread_centrality + t * N;
        for (Size i = 0; i < N; ++i) {
            centrality[i] += local[i];
        }
    }

    // Scale: n/n_samples for unbiased estimate, /2 for undirected
    Real scale = static_cast<Real>(n) / static_cast<Real>(n_samples) * Real(0.5);
    detail::scale_simd(centrality.ptr, scale, N);

    if (normalize && n > 2) {
        Real norm = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        detail::scale_simd(centrality.ptr, norm, N);
    }

    scl::memory::aligned_free(thread_centrality, SCL_ALIGNMENT);
    scl::memory::aligned_free(reinterpret_cast<char*>(sampled), SCL_ALIGNMENT);
    scl::memory::aligned_free(samples, SCL_ALIGNMENT);
}

// =============================================================================
// Harmonic Centrality (Variant of Closeness)
// =============================================================================

template <typename T, bool IsCSR>
void harmonic_centrality(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    bool normalize = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n <= 1) {
        scl::algo::zero(centrality.ptr, N);
        return;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Index> dist_pool;
    scl::threading::WorkspacePool<Index> queue_pool;
    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), N, [&](size_t s, size_t thread_rank) {
        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);

        scl::algo::fill(distances, N, Index(-1));

        Size queue_head = 0, queue_tail = 0;
        distances[s] = 0;
        queue[queue_tail++] = static_cast<Index>(s);

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

        // Harmonic mean: sum of 1/d(s,v)
        Real harmonic_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (i != s && distances[i] > 0) {
                harmonic_sum += Real(1) / static_cast<Real>(distances[i]);
            }
        }

        centrality[s] = harmonic_sum;
    });

    if (normalize && n > 1) {
        Real norm = Real(1) / static_cast<Real>(n - 1);
        detail::scale_simd(centrality.ptr, norm, N);
    }
}

// =============================================================================
// Current Flow Betweenness (Approximate via Random Walks)
// =============================================================================

template <typename T, bool IsCSR>
void current_flow_betweenness_approx(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> centrality,
    Index n_walks = 1000,
    Index walk_length = 100,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(centrality.len >= N, "Centrality: output buffer too small");

    if (n <= 2) {
        scl::algo::zero(centrality.ptr, N);
        return;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread visit counts
    Size* thread_visits = scl::memory::aligned_alloc<Size>(n_threads * N, SCL_ALIGNMENT);
    scl::algo::zero(thread_visits, n_threads * N);

    // Compute degrees for random neighbor selection
    Index* degree = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    for (Size i = 0; i < N; ++i) {
        degree[i] = adjacency.primary_length(static_cast<Index>(i));
    }

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_walks), [&](size_t walk_id, size_t thread_rank) {
        detail::FastRNG rng(seed ^ (walk_id * 0x9e3779b97f4a7c15ULL));

        Size* local_visits = thread_visits + thread_rank * N;

        // Random start
        Index current = rng.bounded(n);

        for (Index step = 0; step < walk_length; ++step) {
            ++local_visits[current];

            Index deg = degree[current];
            if (deg == 0) break;

            // Random neighbor
            Index k = rng.bounded(deg);
            current = adjacency.primary_indices(current)[k];
        }
    });

    // Reduce
    scl::algo::zero(centrality.ptr, N);
    for (size_t t = 0; t < n_threads; ++t) {
        Size* local = thread_visits + t * N;
        for (Size i = 0; i < N; ++i) {
            centrality[i] += static_cast<Real>(local[i]);
        }
    }

    // Normalize
    Real total = detail::sum_simd(centrality.ptr, N);
    if (total > config::MIN_SCORE) {
        detail::scale_simd(centrality.ptr, Real(1) / total, N);
    }

    scl::memory::aligned_free(degree, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_visits, SCL_ALIGNMENT);
}

} // namespace scl::kernel::centrality

