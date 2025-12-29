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
// C++20 Concepts
// =============================================================================

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    inline constexpr Real DEFAULT_DAMPING = Real(0.85);
    inline constexpr Index DEFAULT_MAX_ITER = 100;
    inline constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    inline constexpr Real MIN_SCORE = Real(1e-15);
    inline constexpr Size PARALLEL_THRESHOLD = 256;
    inline constexpr Size PREFETCH_DISTANCE = 4;
    inline constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// SIMD-Accelerated Vector Operations
// =============================================================================

namespace detail {

SCL_HOT SCL_FORCE_INLINE Real sum_simd(Array<const Real> x) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));
    const Size n = x.size();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, x.data() + k));
        v_sum1 = s::Add(v_sum1, s::Load(d, x.data() + k + lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, x.data() + k + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, x.data() + k + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        result += x[static_cast<Index>(k)];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(Array<const Real> x) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));
    const Size n = x.size();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto v0 = s::Load(d, x.data() + k);
        auto v1 = s::Load(d, x.data() + k + lanes);
        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        result += x[static_cast<Index>(k)] * x[static_cast<Index>(k)];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real l1_diff_simd(
    Array<const Real> a,
    Array<const Real> b
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));
    const Size n = a.size();

    auto v_sum = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        auto diff = s::Sub(s::Load(d, a.data() + static_cast<Index>(k)), s::Load(d, b.data() + static_cast<Index>(k)));
        v_sum = s::Add(v_sum, s::Abs(diff));
    }

    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        Real diff = a[static_cast<Index>(k)] - b[static_cast<Index>(k)];
        result += (diff >= Real(0)) ? diff : -diff;
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE void scale_simd(
    Array<Real> x,
    Real alpha
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));
    const Size n = x.size();

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x.data() + static_cast<Index>(k))), d, x.data() + static_cast<Index>(k));
        s::Store(s::Mul(v_alpha, s::Load(d, x.data() + static_cast<Index>(k + lanes))), d, x.data() + static_cast<Index>(k + lanes));
        s::Store(s::Mul(v_alpha, s::Load(d, x.data() + static_cast<Index>(k + 2*lanes))), d, x.data() + static_cast<Index>(k + 2*lanes));
        s::Store(s::Mul(v_alpha, s::Load(d, x.data() + static_cast<Index>(k + 3*lanes))), d, x.data() + static_cast<Index>(k + 3*lanes));
    }

    for (; k < n; ++k) {
        x[static_cast<Index>(k)] *= alpha;
    }
}

SCL_HOT SCL_FORCE_INLINE void axpby_simd(
    Real alpha,
    Array<const Real> x,
    Real beta,
    Array<Real> y
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const auto lanes = static_cast<Size>(s::Lanes(d));
    const Size n = x.size();

    auto v_alpha = s::Set(d, alpha);
    auto v_beta = s::Set(d, beta);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto y0 = s::Mul(v_beta, s::Load(d, y.data() + static_cast<Index>(k)));
        auto y1 = s::Mul(v_beta, s::Load(d, y.data() + static_cast<Index>(k + lanes)));
        y0 = s::MulAdd(v_alpha, s::Load(d, x.data() + static_cast<Index>(k)), y0);
        y1 = s::MulAdd(v_alpha, s::Load(d, x.data() + static_cast<Index>(k + lanes)), y1);
        s::Store(y0, d, y.data() + static_cast<Index>(k));
        s::Store(y1, d, y.data() + static_cast<Index>(k + lanes));
    }

    for (; k < n; ++k) {
        y[static_cast<Index>(k)] = alpha * x[static_cast<Index>(k)] + beta * y[static_cast<Index>(k)];
    }
}

// Normalize to sum = 1
SCL_FORCE_INLINE void normalize_l1(Array<Real> scores) {
    Real sum = sum_simd(scores);
    if (sum > config::MIN_SCORE) {
        scale_simd(scores, Real(1) / sum);
    }
}

// Normalize to L2 norm = 1
SCL_FORCE_INLINE void normalize_l2(Array<Real> scores) {
    Real norm_sq = norm_squared_simd(scores);
    if (norm_sq > config::MIN_SCORE) {
        scale_simd(scores, Real(1) / std::sqrt(norm_sq));
    }
}

// Check convergence
SCL_FORCE_INLINE bool check_convergence(
    Array<const Real> old_scores,
    Array<const Real> new_scores,
    Real tol
) {
    return l1_diff_simd(old_scores, new_scores) < tol;
}

// =============================================================================
// Fast PRNG
// =============================================================================

struct alignas(16) FastRNG {
    std::array<uint32_t, 4> s{};

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
        auto data_ptr = scl::memory::aligned_alloc<Index>(cap, SCL_ALIGNMENT);
        data = data_ptr.release();
        clear();
    }

    void destroy() {
        if (data) scl::memory::aligned_free(data, SCL_ALIGNMENT);
    }

    SCL_FORCE_INLINE Array<Index> array() noexcept {
        return {data, capacity};
    }

    SCL_FORCE_INLINE void clear() noexcept { head = 0; tail = 0; }
    [[nodiscard]] SCL_FORCE_INLINE bool empty() const noexcept { return head == tail; }
    [[nodiscard]] SCL_FORCE_INLINE Size size() const noexcept { return tail - head; }
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

template <Arithmetic T, bool IsCSR>
SCL_HOT void parallel_distribute(
    const Sparse<T, IsCSR>& adj,
    Array<const Real> scores,
    Array<const Real> out_degree_inv,
    Array<Real> scores_new,
    Real damping
) {
    const Size n = scores.size();
    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        // Use atomic accumulation for parallel distribution
        auto atomic_scores_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(n, SCL_ALIGNMENT);
        std::atomic<int64_t>* atomic_scores = atomic_scores_ptr.release();
        Array<std::atomic<int64_t>> atomic_scores_arr(atomic_scores, n);

        // Initialize
        for (Size i = 0; i < n; ++i) {
            atomic_scores_arr[static_cast<Index>(i)].store(0, std::memory_order_relaxed);
        }

        constexpr int64_t SCALE = 1000000000LL;

        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const auto idx = static_cast<Index>(i);
            Real contrib = damping * scores[static_cast<Index>(i)] * out_degree_inv[static_cast<Index>(i)];

            if (contrib <= config::MIN_SCORE) return;

            auto indices = adj.primary_indices_unsafe(idx);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(idx);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                auto delta = static_cast<std::atomic<int64_t>>(static_cast<int64_t>(contrib * static_cast<Real>(values[k])) * SCALE);
                atomic_scores_arr[j].fetch_add(delta, std::memory_order_relaxed);
            }
        });

        // Convert back
        Real inv_scale = Real(1) / static_cast<Real>(SCALE);
        for (Size i = 0; i < n; ++i) {
            scores_new[static_cast<Index>(i)] += static_cast<Real>(atomic_scores_arr[static_cast<Index>(i)].load()) * inv_scale;
        }

        scl::memory::aligned_free(reinterpret_cast<int64_t*>(atomic_scores), SCL_ALIGNMENT);
    } else {
        // Sequential distribution
        for (Size i = 0; i < n; ++i) {
            const auto idx = static_cast<Index>(i);
            Real contrib = damping * scores[static_cast<Index>(i)] * out_degree_inv[static_cast<Index>(i)];

            if (contrib <= config::MIN_SCORE) continue;

            auto indices = adj.primary_indices_unsafe(idx);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(idx);

            // 4-way unrolled
            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                scores_new[static_cast<Index>(indices[k + 0])] += contrib * static_cast<Real>(values[k + 0]);
                scores_new[static_cast<Index>(indices[k + 1])] += contrib * static_cast<Real>(values[k + 1]);
                scores_new[static_cast<Index>(indices[k + 2])] += contrib * static_cast<Real>(values[k + 2]);
                scores_new[static_cast<Index>(indices[k + 3])] += contrib * static_cast<Real>(values[k + 3]);
            }

            for (; k < len; ++k) {
                scores_new[static_cast<Index>(indices[k])] += contrib * static_cast<Real>(values[k]);
            }
        }
    }
}

// =============================================================================
// Parallel SpMV for centrality (y = A * x)
// =============================================================================

template <Arithmetic T, bool IsCSR>
SCL_HOT void spmv_centrality(
    const Sparse<T, IsCSR>& adj,
    Array<const Real> x,
    Array<Real> y
) {
    const Size n = x.size();
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const auto idx = static_cast<Index>(i);
            auto indices = adj.primary_indices_unsafe(idx);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(idx);

            Real sum = Real(0);

            // Prefetch
            if (SCL_LIKELY(i + 1 < n)) {
                SCL_PREFETCH_READ(adj.primary_indices_unsafe(idx + 1).ptr, 0);
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

            y[static_cast<Index>(i)] = sum;
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const auto idx = static_cast<Index>(i);
            auto indices = adj.primary_indices_unsafe(idx);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[static_cast<Index>(i)] = sum;
        }
    }
}

// Compute weighted out-degrees
template <Arithmetic T, bool IsCSR>
void compute_out_degrees(
    const Sparse<T, IsCSR>& adj,
    Array<Real> out_degree,
    Array<Real> out_degree_inv
) {
    const Size n = out_degree.size();
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const auto idx = static_cast<Index>(i);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(static_cast<Index>(i));

            Real deg = Real(0);
            for (Index k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }

            out_degree[static_cast<Index>(i)] = deg;
            out_degree_inv[static_cast<Index>(i)] = (deg > config::MIN_SCORE) ? Real(1) / deg : Real(0);
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const auto idx = static_cast<Index>(i);
            auto values = adj.primary_values_unsafe(idx);
            const Index len = adj.primary_length_unsafe(idx);

            Real deg = Real(0);
            for (Index k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }

            out_degree[static_cast<Index>(i)] = deg;
            out_degree_inv[static_cast<Index>(i)] = (deg > config::MIN_SCORE) ? Real(1) / deg : Real(0);
        }
    }
}

} // namespace detail

// =============================================================================
// Degree Centrality
// =============================================================================

template <Arithmetic T, bool IsCSR>
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
            centrality[static_cast<Index>(i)] = static_cast<Real>(adjacency.primary_length_unsafe(static_cast<Index>(i)));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            centrality[static_cast<Index>(i)] = static_cast<Real>(adjacency.primary_length_unsafe(static_cast<Index>(i)));
        }
    }

    if (normalize && n > 1) {
        Real inv_max = Real(1) / static_cast<Real>(n - 1);
        detail::scale_simd(centrality.first(N), inv_max);
    }
}

// =============================================================================
// Weighted Degree Centrality
// =============================================================================

template <Arithmetic T, bool IsCSR>
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
            const auto idx = static_cast<Index>(i);
            auto values = adjacency.primary_values_unsafe(idx);
            const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            centrality[static_cast<Index>(i)] = sum;
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            const auto idx = static_cast<Index>(i);
            auto values = adjacency.primary_values_unsafe(idx);
            const Index len = adjacency.primary_length_unsafe(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            centrality[static_cast<Index>(i)] = sum;
        }
    }

    if (normalize) {
        // Find max and normalize
        Real max_weight = Real(0);
        for (Size i = 0; i < N; ++i) {
            max_weight = scl::algo::max2(max_weight, centrality[static_cast<Index>(i)]);
        }

        if (max_weight > config::MIN_SCORE) {
            detail::scale_simd(centrality.first(N), Real(1) / max_weight);
        }
    }
}

// =============================================================================
// PageRank (Optimized)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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
    auto out_degree_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto out_degree_inv_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto scores_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    Array<Real> out_degree = out_degree_buf.array();
    Array<Real> out_degree_inv = out_degree_inv_buf.array();
    Array<Real> scores_new = scores_new_buf.array();

    // Compute out-degrees
    detail::compute_out_degrees(adjacency, out_degree, out_degree_inv);

    // Initialize scores uniformly
    Real init_score = Real(1) / static_cast<Real>(n);
    scl::algo::fill(scores.ptr, N, init_score);

    Real teleport = (Real(1) - damping) / static_cast<Real>(n);
    Real dangling_factor = damping / static_cast<Real>(n);

    // Power iteration
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Initialize with teleportation
        scl::algo::fill(scores_new.ptr, N, teleport);

        // Distribute via edges
        detail::parallel_distribute(adjacency, scores.first(N), out_degree_inv, 
                                    scores_new, damping);

        // Handle dangling nodes
        Real dangling_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (out_degree[static_cast<Index>(i)] <= config::MIN_SCORE) {
                dangling_sum += scores[static_cast<Index>(i)];
            }
        }

        if (dangling_sum > config::MIN_SCORE) {
            Real dangling_contrib = dangling_factor * dangling_sum;
            const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());
            
            if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
                scl::threading::parallel_for(Size(0), N, [&](Size i) {
                    scores_new[static_cast<Index>(i)] += dangling_contrib;
                });
            } else {
                for (Size i = 0; i < N; ++i) {
                    scores_new[static_cast<Index>(i)] += dangling_contrib;
                }
            }
        }

        // Check convergence
        if (detail::check_convergence(scores.first(N), scores_new, tol)) {
            scl::algo::copy(scores_new.ptr, scores.ptr, N);
            break;
        }

        scl::algo::swap(scores.ptr, scores_new.ptr);
    }
}

// =============================================================================
// Personalized PageRank
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    auto out_degree_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto out_degree_inv_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto scores_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto personalization_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    Array<Real> out_degree = out_degree_buf.array();
    Array<Real> out_degree_inv = out_degree_inv_buf.array();
    Array<Real> scores_new = scores_new_buf.array();
    Array<Real> personalization = personalization_buf.array();

    detail::compute_out_degrees(adjacency, out_degree, out_degree_inv);

    // Build personalization vector
    scl::algo::zero(personalization.ptr, N);
    Size valid_seeds = 0;
    for (Size s = 0; s < seed_nodes.len; ++s) {
        const auto idx = seed_nodes[static_cast<Index>(s)];
        if (idx >= 0 && idx < n) {
            personalization[idx] += Real(1);
            ++valid_seeds;
        }
    }

    if (valid_seeds > 0) {
        detail::scale_simd(personalization, Real(1) / static_cast<Real>(valid_seeds));
    }

    // Initialize
    scl::algo::copy(personalization.ptr, scores.ptr, N);

    Real one_minus_d = Real(1) - damping;

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Teleport to personalization
        for (Size i = 0; i < N; ++i) {
            scores_new[static_cast<Index>(i)] = one_minus_d * personalization[static_cast<Index>(i)];
        }

        // Distribute
        detail::parallel_distribute(adjacency, scores.first(N), out_degree_inv,
                                    scores_new, damping);

        // Dangling nodes
        Real dangling_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (out_degree[static_cast<Index>(i)] <= config::MIN_SCORE) {
                dangling_sum += scores[static_cast<Index>(i)];
            }
        }

        if (dangling_sum > config::MIN_SCORE) {
            detail::axpby_simd(damping * dangling_sum, personalization, 
                               Real(1), scores_new);
        }

        if (detail::check_convergence(scores.first(N), scores_new, tol)) {
            scl::algo::copy(scores_new.ptr, scores.ptr, N);
            break;
        }

        scl::algo::copy(scores_new.ptr, scores.ptr, N);
    }
}

// =============================================================================
// HITS Algorithm (Optimized)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    auto hub_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto auth_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    Array<Real> hub_new = hub_new_buf.array();
    Array<Real> auth_new = auth_new_buf.array();

    // Initialize
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    scl::algo::fill(hub_scores.ptr, N, init_score);
    scl::algo::fill(authority_scores.ptr, N, init_score);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    for (Index iter = 0; iter < max_iter; ++iter) {
        // Authority update: auth[j] = sum_i(hub[i] * A[i,j])
        scl::algo::zero(auth_new.ptr, N);

        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            // Parallel with atomics
            auto atomic_auth_ptr = scl::memory::aligned_alloc<int64_t>(N, SCL_ALIGNMENT);
            auto* atomic_auth = reinterpret_cast<std::atomic<int64_t>*>(
                atomic_auth_ptr.release());
            Array<std::atomic<int64_t>> atomic_auth_arr(atomic_auth, N);

            for (Size i = 0; i < N; ++i) {
                atomic_auth_arr[static_cast<Index>(i)].store(0, std::memory_order_relaxed);
            }

            constexpr int64_t SCALE = 1000000000LL;

            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
                auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
                const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

                Real hub_i = hub_scores[static_cast<Index>(i)];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    auto delta = static_cast<std::atomic<int64_t>>(static_cast<int64_t>(hub_i * static_cast<Real>(values[k])) * SCALE);
                    atomic_auth_arr[j].fetch_add(delta, std::memory_order_relaxed);
                }
            });

            Real inv_scale = Real(1) / static_cast<Real>(SCALE);
            for (Size i = 0; i < N; ++i) {
                auth_new[static_cast<Index>(i)] = static_cast<Real>(atomic_auth_arr[static_cast<Index>(i)].load()) * inv_scale;
            }

            scl::memory::aligned_free(reinterpret_cast<int64_t*>(atomic_auth), SCL_ALIGNMENT);
        } else {
            for (Index i = 0; i < n; ++i) {
                const auto idx = static_cast<Index>(i);
                auto indices = adjacency.primary_indices_unsafe(idx);
                auto values = adjacency.primary_values_unsafe(i);
                const Index len = adjacency.primary_length_unsafe(idx);

                Real hub_i = hub_scores[static_cast<Index>(i)];

                for (Index k = 0; k < len; ++k) {
                    auth_new[indices[k]] += hub_i * static_cast<Real>(values[k]);
                }
            }
        }

        // Hub update: hub[i] = sum_j(auth[j] * A[i,j])
        detail::spmv_centrality(adjacency, auth_new, hub_new);

        // Normalize
        detail::normalize_l2(auth_new);
        detail::normalize_l2(hub_new);

        // Check convergence
        bool converged = detail::check_convergence(authority_scores.first(N), auth_new, tol) &&
                         detail::check_convergence(hub_scores.first(N), hub_new, tol);

        scl::algo::copy(auth_new.ptr, authority_scores.ptr, N);
        scl::algo::copy(hub_new.ptr, hub_scores.ptr, N);

        if (converged) break;
    }
}

// =============================================================================
// Eigenvector Centrality (Optimized)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    auto centrality_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    Array<Real> centrality_new = centrality_new_buf.array();

    // Initialize
    Real init_score = Real(1) / std::sqrt(static_cast<Real>(n));
    scl::algo::fill(centrality.ptr, N, init_score);

    for (Index iter = 0; iter < max_iter; ++iter) {
        detail::spmv_centrality(adjacency, centrality.first(N), centrality_new);

        detail::normalize_l2(centrality_new);

        if (detail::check_convergence(centrality.first(N), centrality_new, tol)) {
            scl::algo::copy(centrality_new.ptr, centrality.ptr, N);
            break;
        }

        scl::algo::copy(centrality_new.ptr, centrality.ptr, N);
    }
}

// =============================================================================
// Katz Centrality (Optimized)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    auto centrality_new_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    auto temp_buf = scl::memory::AlignedBuffer<Real>(N, SCL_ALIGNMENT);
    Array<Real> centrality_new = centrality_new_buf.array();
    Array<Real> temp = temp_buf.array();

    // Initialize
    scl::algo::fill(centrality.ptr, N, beta);

    for (Index iter = 0; iter < max_iter; ++iter) {
        // temp = A * centrality
        detail::spmv_centrality(adjacency, centrality.first(N), temp);

        // centrality_new = alpha * temp + beta
        const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());
        
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](Size i) {
                centrality_new[static_cast<Index>(i)] = alpha * temp[static_cast<Index>(i)] + beta;
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                centrality_new[static_cast<Index>(i)] = alpha * temp[static_cast<Index>(i)] + beta;
            }
        }

        if (detail::check_convergence(centrality.first(N), centrality_new, tol)) {
            scl::algo::copy(centrality_new.ptr, centrality.ptr, N);
            break;
        }

        scl::algo::copy(centrality_new.ptr, centrality.ptr, N);
    }
}

// =============================================================================
// Closeness Centrality (Parallel)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    // Per-thread workspace
    scl::threading::WorkspacePool<Index> dist_pool;
    scl::threading::WorkspacePool<Index> queue_pool;
    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), N, [&](Size s, Size thread_rank) {
        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);
        Array<Index> distances_arr(distances, N);

        scl::algo::fill(distances, N, Index(-1));

        Size queue_head = 0, queue_tail = 0;
        distances_arr[static_cast<Index>(s)] = 0;
        queue[queue_tail++] = static_cast<Index>(s);

        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];

            // Prefetch
            if (queue_head + config::PREFETCH_DISTANCE < queue_tail) {
                Index next_u = queue[queue_head + config::PREFETCH_DISTANCE];
                SCL_PREFETCH_READ(adjacency.primary_indices_unsafe(next_u).ptr, 0);
            }

            auto indices = adjacency.primary_indices_unsafe(u);
            const Index len = adjacency.primary_length_unsafe(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];
                if (distances_arr[v] == -1) {
                    distances_arr[v] = distances_arr[u] + 1;
                    queue[static_cast<Index>(queue_tail++)] = v;
                }
            }
        }

        // Compute closeness
        int64_t total_dist = 0;
        Index reachable = 0;

        for (Size i = 0; i < N; ++i) {
            if (i != static_cast<Index>(s) && distances_arr[static_cast<Index>(i)] > 0) {
                total_dist += distances_arr[static_cast<Index>(i)];
                ++reachable;
            }
        }

        if (total_dist > 0) {
            centrality[static_cast<Index>(s)] = static_cast<Real>(reachable) / static_cast<Real>(total_dist);
            if (normalize && reachable < n - 1) {
                centrality[static_cast<Index>(s)] *= static_cast<Real>(reachable) / static_cast<Real>(n - 1);
            }
        } else {
            centrality[static_cast<Index>(s)] = Real(0);
        }
    });
}

// =============================================================================
// Betweenness Centrality (Parallel Brandes)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    // Per-thread partial centrality
    auto thread_centrality_buf = scl::memory::AlignedBuffer<Real>(n_threads * N, SCL_ALIGNMENT);
    Array<Real> thread_centrality = thread_centrality_buf.array();
    scl::algo::zero(thread_centrality.ptr, n_threads * N);

    // Per-thread workspace
    struct BrandesWorkspace {
        Index* distances;
        Real* sigma;
        Real* delta;
        Index* queue;
        Index* stack;
        Size n;

        [[nodiscard]] Array<Index> distances_arr() const noexcept { return {distances, n}; }
        [[nodiscard]] Array<Real> sigma_arr() const noexcept { return {sigma, n}; }
        [[nodiscard]] Array<Real> delta_arr() const noexcept { return {delta, n}; }
        [[nodiscard]] Array<Index> queue_arr() const noexcept { return {queue, n}; }
        [[nodiscard]] Array<Index> stack_arr() const noexcept { return {stack, n}; }
    };

    auto workspaces_ptr = scl::memory::aligned_alloc<BrandesWorkspace>(n_threads, SCL_ALIGNMENT);
    BrandesWorkspace* workspaces = workspaces_ptr.release();

    for (Size t = 0; t < n_threads; ++t) {
        auto dist_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
        auto sigma_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        auto delta_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        auto queue_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
        auto stack_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
        workspaces[t].distances = dist_ptr.release();
        workspaces[t].sigma = sigma_ptr.release();
        workspaces[t].delta = delta_ptr.release();
        workspaces[t].queue = queue_ptr.release();
        workspaces[t].stack = stack_ptr.release();
        workspaces[t].n = N;
    }

    scl::threading::parallel_for(Size(0), N, [&](Size s, Size thread_rank) {
        auto& ws = workspaces[thread_rank];
        Array<Real> local_centrality = thread_centrality.subspan(thread_rank * N, N);
        Array<Index> distances_arr = ws.distances_arr();
        Array<Real> sigma_arr = ws.sigma_arr();
        Array<Real> delta_arr = ws.delta_arr();
        Array<Index> queue_arr = ws.queue_arr();
        Array<Index> stack_arr = ws.stack_arr();

        scl::algo::fill(ws.distances, N, Index(-1));
        scl::algo::zero(ws.sigma, N);
        scl::algo::zero(ws.delta, N);

        distances_arr[static_cast<Index>(s)] = 0;
        sigma_arr[static_cast<Index>(s)] = Real(1);

        Size queue_head = 0, queue_tail = 0;
        Size stack_top = 0;
        queue_arr[static_cast<Index>(queue_tail++)] = static_cast<Index>(s);

        // BFS
        while (queue_head < queue_tail) {
            const auto u = static_cast<Index>(queue_arr[static_cast<Index>(queue_head++)]);
            stack_arr[static_cast<Index>(stack_top++)] = static_cast<Index>(u);

            auto indices = adjacency.primary_indices_unsafe(u);
            const Index len = adjacency.primary_length_unsafe(u);

            for (Index k = 0; k < len; ++k) {
                const auto v = static_cast<Index>(indices[k]);

                if (distances_arr[v] == -1) {
                    distances_arr[v] = distances_arr[static_cast<Index>(u)] + 1;
                    queue_arr[static_cast<Index>(queue_tail++)] = v;
                }

                if (distances_arr[v] == distances_arr[static_cast<Index>(u)] + 1) {
                    sigma_arr[v] += sigma_arr[static_cast<Index>(u)];
                }
            }
        }

        // Back-propagation
        while (stack_top > 0) {
            const auto w = static_cast<Index>(stack_arr[static_cast<Index>(--stack_top)]);
            auto indices = adjacency.primary_indices_unsafe(w);
            const Index len = adjacency.primary_length_unsafe(w);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (distances_arr[v] == distances_arr[w] - 1) {
                    delta_arr[v] += (sigma_arr[v] / sigma_arr[static_cast<Index>(w)]) * (Real(1) + delta_arr[static_cast<Index>(w)]);
                }
            }

            if (w != static_cast<Index>(s)) {
                local_centrality[static_cast<Index>(w)] += delta_arr[static_cast<Index>(w)];
            }
        }
    });

    // Reduce thread results
    scl::algo::zero(centrality.ptr, N);
    for (Size t = 0; t < n_threads; ++t) {
        Array<Real> local = thread_centrality.subspan(t * N, N);
        for (Size i = 0; i < N; ++i) {
            centrality[static_cast<Index>(i)] += local[static_cast<Index>(i)];
        }
    }

    // Undirected graph: divide by 2
    detail::scale_simd(centrality.first(static_cast<Index>(N)), Real(0.5));

    if (normalize && n > 2) {
        Real norm = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        detail::scale_simd(centrality.first(static_cast<Index>(N)), norm);
    }

    // Cleanup
    for (Size t = 0; t < n_threads; ++t) {
        scl::memory::aligned_free(workspaces[t].stack, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].queue, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].delta, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].sigma, SCL_ALIGNMENT);
        scl::memory::aligned_free(workspaces[t].distances, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(workspaces, SCL_ALIGNMENT);
}

// =============================================================================
// Sampled Betweenness (Approximate, Faster)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    // Generate sample indices
    auto samples_buf = scl::memory::AlignedBuffer<Index>(n_samples, SCL_ALIGNMENT);
    auto sampled_char_buf = scl::memory::AlignedBuffer<char>(N, SCL_ALIGNMENT);
    Array<Index> samples = samples_buf.array();
    bool* sampled = reinterpret_cast<bool*>(sampled_char_buf.get());
    scl::algo::zero(sampled, N);

    detail::FastRNG rng(seed);
    Index samples_done = 0;

    while (samples_done < n_samples) {
        const auto s = static_cast<Index>(rng.bounded(static_cast<Index>(n)));
        if (!sampled[s]) {
            sampled[s] = true;
            samples[static_cast<Index>(samples_done++)] = s;
        }
    }

    // Per-thread centrality
    auto thread_centrality_buf = scl::memory::AlignedBuffer<Real>(n_threads * N, SCL_ALIGNMENT);
    Array<Real> thread_centrality = thread_centrality_buf.array();
    scl::algo::zero(thread_centrality.ptr, n_threads * N);

    // Per-thread workspace
    scl::threading::WorkspacePool<Index> dist_pool, queue_pool, stack_pool;
    scl::threading::WorkspacePool<Real> sigma_pool, delta_pool;

    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);
    stack_pool.init(n_threads, N);
    sigma_pool.init(n_threads, N);
    delta_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_samples), [&](Size idx, Size thread_rank) {
        const auto s = static_cast<Index>(samples[static_cast<Index>(idx)]);

        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);
        Index* stack = stack_pool.get(thread_rank);
        Real* sigma = sigma_pool.get(thread_rank);
        Real* delta = delta_pool.get(thread_rank);
        Array<Index> distances_arr(distances, N);
        Array<Real> sigma_arr(sigma, N);
        Array<Real> delta_arr(delta, N);
        Array<Real> local_centrality = thread_centrality.subspan(thread_rank * N, N);

        scl::algo::fill(distances, N, Index(-1));
        scl::algo::zero(sigma, N);
        scl::algo::zero(delta, N);

        distances_arr[s] = 0;
        sigma_arr[s] = Real(1);

        Size queue_head = 0, queue_tail = 0;
        Size stack_top = 0;
        queue[queue_tail++] = s;

        while (queue_head < queue_tail) {
            Index u = queue[queue_head++];
            stack[stack_top++] = u;

            auto indices = adjacency.primary_indices_unsafe(u);
            const Index len = adjacency.primary_length_unsafe(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (distances_arr[v] == -1) {
                    distances_arr[v] = distances_arr[u] + 1;
                    queue[static_cast<Index>(queue_tail++)] = v;
                }

                if (distances_arr[v] == distances_arr[u] + 1) {
                    sigma_arr[v] += sigma_arr[u];
                }
            }
        }

        while (stack_top > 0) {
            Index w = stack[--stack_top];
            auto indices = adjacency.primary_indices_unsafe(w);
            const Index len = adjacency.primary_length_unsafe(w);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];

                if (distances_arr[v] == distances_arr[w] - 1) {
                    delta_arr[v] += (sigma_arr[v] / sigma_arr[w]) * (Real(1) + delta_arr[w]);
                }
            }

            if (w != s) {
                local_centrality[w] += delta_arr[w];
            }
        }
    });

    // Reduce
    scl::algo::zero(centrality.ptr, N);
    for (size_t t = 0; t < n_threads; ++t) {
        Array<Real> local = thread_centrality.subspan(t * N, N);
        for (Size i = 0; i < N; ++i) {
            centrality[static_cast<Index>(i)] += local[static_cast<Index>(i)];
        }
    }

    // Scale: n/n_samples for unbiased estimate, /2 for undirected
    Real scale = static_cast<Real>(n) / static_cast<Real>(n_samples) * Real(0.5);
    detail::scale_simd(centrality.first(static_cast<Index>(N)), scale);

    if (normalize && n > 2) {
        Real norm = Real(2) / (static_cast<Real>(n - 1) * static_cast<Real>(n - 2));
        detail::scale_simd(centrality.first(static_cast<Index>(N)), norm);
    }
}

// =============================================================================
// Harmonic Centrality (Variant of Closeness)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    scl::threading::WorkspacePool<Index> dist_pool;
    scl::threading::WorkspacePool<Index> queue_pool;
    dist_pool.init(n_threads, N);
    queue_pool.init(n_threads, N);

    scl::threading::parallel_for(Size(0), N, [&](Size s, Size thread_rank) {
        Index* distances = dist_pool.get(thread_rank);
        Index* queue = queue_pool.get(thread_rank);
        Array<Index> distances_arr(distances, N);

        scl::algo::fill(distances, N, Index(-1));

        Size queue_head = 0, queue_tail = 0;
        distances_arr[static_cast<Index>(s)] = 0;
        queue[queue_tail++] = static_cast<Index>(s);

        while (queue_head < queue_tail) {
            const auto u = static_cast<Index>(queue[static_cast<Index>(queue_head++)]);

            auto indices = adjacency.primary_indices_unsafe(u);
            const Index len = adjacency.primary_length_unsafe(u);

            for (Index k = 0; k < len; ++k) {
                const auto v = static_cast<Index>(indices[k]);
                if (distances_arr[v] == -1) {
                    distances_arr[v] = distances_arr[u] + 1;
                    queue[queue_tail++] = v;
                }
            }
        }

        // Harmonic mean: sum of 1/d(s,v)
        Real harmonic_sum = Real(0);
        for (Size i = 0; i < N; ++i) {
            if (i != s && distances_arr[static_cast<Index>(i)] > 0) {
                harmonic_sum += Real(1) / static_cast<Real>(distances_arr[static_cast<Index>(i)]);
            }
        }

        centrality[static_cast<Index>(s)] = harmonic_sum;
    });

    if (normalize && n > 1) {
        Real norm = Real(1) / static_cast<Real>(n - 1);
        detail::scale_simd(centrality.first(static_cast<Index>(N)), norm);
    }
}

// =============================================================================
// Current Flow Betweenness (Approximate via Random Walks)
// =============================================================================

template <Arithmetic T, bool IsCSR>
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

    const auto n_threads = static_cast<Size>(scl::threading::Scheduler::get_num_threads());

    // Per-thread visit counts
    auto thread_visits_buf = scl::memory::AlignedBuffer<Size>(n_threads * N, SCL_ALIGNMENT);
    Array<Size> thread_visits = thread_visits_buf.array();
    scl::algo::zero(thread_visits.ptr, n_threads * N);

    // Compute degrees for random neighbor selection
    auto degree_buf = scl::memory::AlignedBuffer<Index>(N, SCL_ALIGNMENT);
    Array<Index> degree = degree_buf.array();
    for (Size i = 0; i < N; ++i) {
        degree[static_cast<Index>(i)] = adjacency.primary_length_unsafe(static_cast<Index>(i));
    }

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_walks), [&](Size walk_id, Size thread_rank) {
        detail::FastRNG rng(seed ^ (walk_id * 0x9e3779b97f4a7c15ULL));

        Array<Size> local_visits = thread_visits.subspan(thread_rank * N, N);

        // Random start
        const auto current = static_cast<Index>(rng.bounded(static_cast<Index>(n)));

        for (Index step = 0; step < walk_length; ++step) {
            ++local_visits[static_cast<Index>(current)];

            const auto deg = degree[static_cast<Index>(current)];
            if (deg == 0) break;

            // Random neighbor
            const auto k = static_cast<Index>(rng.bounded(static_cast<Index>(deg)));
            current = adjacency.primary_indices_unsafe(static_cast<Index>(current))[static_cast<Index>(k)];
        }
    });

    // Reduce
    scl::algo::zero(centrality.ptr, N);
    for (size_t t = 0; t < n_threads; ++t) {
        Array<Size> local = thread_visits.subspan(t * N, N);
        for (Size i = 0; i < N; ++i) {
            centrality[static_cast<Index>(i)] += static_cast<Real>(local[static_cast<Index>(i)]);
        }
    }

    // Normalize
    Real total = detail::sum_simd(centrality.first(N));
    if (total > config::MIN_SCORE) {
        detail::scale_simd(centrality.first(N), Real(1) / total);
    }
}

} // namespace scl::kernel::centrality
