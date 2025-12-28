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
// FILE: scl/kernel/propagation.hpp
// BRIEF: Label propagation for semi-supervised learning on graphs (OPTIMIZED)
// =============================================================================

namespace scl::kernel::propagation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_ALPHA = Real(0.99);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Index UNLABELED = -1;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Simple PRNG for random node order
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

// Fisher-Yates shuffle
SCL_FORCE_INLINE void shuffle(Index* SCL_RESTRICT arr, Size n, FastRNG& rng) {
    for (Size i = n - 1; i > 0; --i) {
        Size j = rng.bounded(i + 1);
        Index tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// SIMD-optimized convergence check for integer labels
SCL_FORCE_INLINE bool check_convergence(
    const Index* SCL_RESTRICT labels_old,
    const Index* SCL_RESTRICT labels_new,
    Size n
) {
    // Use SIMD comparison with early exit
    namespace s = scl::simd;
    using IndexTag = s::IndexTag;
    const IndexTag di;
    const size_t lanes = Lanes(di);
    Size i = 0;

    // SIMD path with 4-way unrolling
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        auto v0_old = s::Load(di, labels_old + i + 0 * lanes);
        auto v0_new = s::Load(di, labels_new + i + 0 * lanes);
        auto v1_old = s::Load(di, labels_old + i + 1 * lanes);
        auto v1_new = s::Load(di, labels_new + i + 1 * lanes);
        auto v2_old = s::Load(di, labels_old + i + 2 * lanes);
        auto v2_new = s::Load(di, labels_new + i + 2 * lanes);
        auto v3_old = s::Load(di, labels_old + i + 3 * lanes);
        auto v3_new = s::Load(di, labels_new + i + 3 * lanes);

        auto ne0 = s::Ne(v0_old, v0_new);
        auto ne1 = s::Ne(v1_old, v1_new);
        auto ne2 = s::Ne(v2_old, v2_new);
        auto ne3 = s::Ne(v3_old, v3_new);

        auto any_ne = s::Or(s::Or(ne0, ne1), s::Or(ne2, ne3));
        if (SCL_UNLIKELY(!s::AllFalse(di, any_ne))) return false;
    }

    // Single SIMD pass
    for (; i + lanes <= n; i += lanes) {
        auto v_old = s::Load(di, labels_old + i);
        auto v_new = s::Load(di, labels_new + i);
        auto ne = s::Ne(v_old, v_new);
        if (SCL_UNLIKELY(!s::AllFalse(di, ne))) return false;
    }

    // Scalar cleanup
    for (; i < n; ++i) {
        if (SCL_UNLIKELY(labels_old[i] != labels_new[i])) return false;
    }

    return true;
}

// SIMD-optimized convergence check for real values with multi-accumulator
SCL_FORCE_INLINE bool check_convergence_real(
    const Real* SCL_RESTRICT vals_old,
    const Real* SCL_RESTRICT vals_new,
    Size n,
    Real tol
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Multi-accumulator pattern for L1 norm of difference
    auto v_diff0 = s::Zero(d);
    auto v_diff1 = s::Zero(d);
    auto v_diff2 = s::Zero(d);
    auto v_diff3 = s::Zero(d);

    Size i = 0;

    // 4-way unrolled SIMD loop
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        if (SCL_LIKELY(i + config::PREFETCH_DISTANCE * lanes < n)) {
            SCL_PREFETCH_READ(vals_old + i + config::PREFETCH_DISTANCE * lanes, 0);
            SCL_PREFETCH_READ(vals_new + i + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto old0 = s::Load(d, vals_old + i + 0 * lanes);
        auto new0 = s::Load(d, vals_new + i + 0 * lanes);
        auto old1 = s::Load(d, vals_old + i + 1 * lanes);
        auto new1 = s::Load(d, vals_new + i + 1 * lanes);
        auto old2 = s::Load(d, vals_old + i + 2 * lanes);
        auto new2 = s::Load(d, vals_new + i + 2 * lanes);
        auto old3 = s::Load(d, vals_old + i + 3 * lanes);
        auto new3 = s::Load(d, vals_new + i + 3 * lanes);

        v_diff0 = s::Add(v_diff0, s::Abs(s::Sub(new0, old0)));
        v_diff1 = s::Add(v_diff1, s::Abs(s::Sub(new1, old1)));
        v_diff2 = s::Add(v_diff2, s::Abs(s::Sub(new2, old2)));
        v_diff3 = s::Add(v_diff3, s::Abs(s::Sub(new3, old3)));
    }

    // Combine accumulators
    auto v_diff = s::Add(s::Add(v_diff0, v_diff1), s::Add(v_diff2, v_diff3));

    // Single SIMD pass for remainder
    for (; i + lanes <= n; i += lanes) {
        auto v_old = s::Load(d, vals_old + i);
        auto v_new = s::Load(d, vals_new + i);
        v_diff = s::Add(v_diff, s::Abs(s::Sub(v_new, v_old)));
    }

    Real diff = s::GetLane(s::SumOfLanes(d, v_diff));

    // Scalar cleanup
    for (; i < n; ++i) {
        Real delta = vals_new[i] - vals_old[i];
        diff += (delta >= Real(0)) ? delta : -delta;
    }

    return diff < tol;
}

// SIMD-optimized row sum computation with parallel processing
template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& adj,
    Real* SCL_RESTRICT row_sums
) {
    const Index n = adj.primary_dim();
    const Size N = static_cast<Size>(n);

    if (N >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto values = adj.primary_values_unsafe(static_cast<Index>(i));
            const Index len = adj.primary_length_unsafe(static_cast<Index>(i));

            if (SCL_UNLIKELY(len == 0)) {
                row_sums[i] = Real(0);
                return;
            }

            namespace s = scl::simd;
            using SimdTag = s::SimdTagFor<Real>;
            const SimdTag d;
            const size_t lanes = s::Lanes(d);

            // Multi-accumulator for longer rows
            if (static_cast<Size>(len) >= config::SIMD_THRESHOLD) {
                auto v_sum0 = s::Zero(d);
                auto v_sum1 = s::Zero(d);
                Index k = 0;

                for (; k + 2 * static_cast<Index>(lanes) <= len; k += 2 * static_cast<Index>(lanes)) {
                    v_sum0 = s::Add(v_sum0, s::LoadU(d, values.ptr + k));
                    v_sum1 = s::Add(v_sum1, s::LoadU(d, values.ptr + k + static_cast<Index>(lanes)));
                }

                auto v_sum = s::Add(v_sum0, v_sum1);
                Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

                for (; k < len; ++k) {
                    sum += static_cast<Real>(values[k]);
                }

                row_sums[i] = sum;
            } else {
                // Scalar path with unrolling for short rows
                Real sum = Real(0);
                Index k = 0;

                for (; k + 4 <= len; k += 4) {
                    sum += static_cast<Real>(values[k + 0]);
                    sum += static_cast<Real>(values[k + 1]);
                    sum += static_cast<Real>(values[k + 2]);
                    sum += static_cast<Real>(values[k + 3]);
                }

                for (; k < len; ++k) {
                    sum += static_cast<Real>(values[k]);
                }

                row_sums[i] = sum;
            }
        });
    } else {
        // Sequential path for small graphs
        for (Index i = 0; i < n; ++i) {
            auto values = adj.primary_values_unsafe(i);
            const Index len = adj.primary_length_unsafe(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }

            row_sums[i] = sum;
        }
    }
}

// SIMD-optimized argmax for class voting
SCL_FORCE_INLINE Index find_argmax(
    const Real* SCL_RESTRICT votes,
    Index n_classes
) {
    if (SCL_UNLIKELY(n_classes == 0)) return config::UNLABELED;
    if (SCL_UNLIKELY(n_classes == 1)) return 0;

    Index best_class = 0;
    Real best_votes = votes[0];

    // Unrolled scalar search (n_classes is typically small)
    Index c = 1;
    for (; c + 4 <= n_classes; c += 4) {
        if (votes[c + 0] > best_votes) { best_votes = votes[c + 0]; best_class = c + 0; }
        if (votes[c + 1] > best_votes) { best_votes = votes[c + 1]; best_class = c + 1; }
        if (votes[c + 2] > best_votes) { best_votes = votes[c + 2]; best_class = c + 2; }
        if (votes[c + 3] > best_votes) { best_votes = votes[c + 3]; best_class = c + 3; }
    }

    for (; c < n_classes; ++c) {
        if (votes[c] > best_votes) {
            best_votes = votes[c];
            best_class = c;
        }
    }

    return best_class;
}

// SIMD-optimized zero with prefetch
SCL_FORCE_INLINE void fast_zero(Real* SCL_RESTRICT arr, Size n) {
    std::memset(arr, 0, n * sizeof(Real));
}

// SIMD-optimized row normalization
SCL_FORCE_INLINE void normalize_row(Real* SCL_RESTRICT row, Index n_classes) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    // Compute sum
    Real sum = Real(0);
    Index c = 0;

    if (static_cast<Size>(n_classes) >= lanes) {
        auto v_sum = s::Zero(d);
        for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
            v_sum = s::Add(v_sum, s::Load(d, row + c));
        }
        sum = s::GetLane(s::SumOfLanes(d, v_sum));
    }

    for (; c < n_classes; ++c) {
        sum += row[c];
    }

    if (SCL_LIKELY(sum > Real(1e-15))) {
        Real inv_sum = Real(1) / sum;

        // SIMD division
        c = 0;
        if (static_cast<Size>(n_classes) >= lanes) {
            auto v_inv = s::Set(d, inv_sum);
            for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
                auto v = s::Load(d, row + c);
                s::Store(s::Mul(v, v_inv), d, row + c);
            }
        }

        for (; c < n_classes; ++c) {
            row[c] *= inv_sum;
        }
    }
}

} // namespace detail

// =============================================================================
// Label Propagation (Hard Labels) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void label_propagation(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Index max_iter = config::DEFAULT_MAX_ITER,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(labels.len >= N, "Propagation: labels buffer too small");

    if (SCL_UNLIKELY(n == 0)) return;

    // Find number of classes and count labeled nodes
    Index max_label = 0;
    Index n_labeled = 0;

    for (Index i = 0; i < n; ++i) {
        if (labels[i] != config::UNLABELED) {
            max_label = scl::algo::max2(max_label, labels[i]);
            ++n_labeled;
        }
    }

    if (SCL_UNLIKELY(n_labeled == 0)) return;
    if (SCL_UNLIKELY(n_labeled == n)) return;

    Index n_classes = max_label + 1;
    const Size n_classes_sz = static_cast<Size>(n_classes);

    // Allocate workspace
    Index* labels_new = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    Index* order = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);

    // Initialize with SIMD copy
    std::memcpy(labels_new, labels.ptr, N * sizeof(Index));

    for (Index i = 0; i < n; ++i) {
        order[i] = i;
    }

    // Per-thread workspace for class votes
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    scl::threading::WorkspacePool<Real> vote_pool;
    if (use_parallel) {
        vote_pool.init(n_threads, n_classes_sz);
    }

    // Single-thread workspace
    Real* class_votes_st = nullptr;
    if (!use_parallel) {
        class_votes_st = scl::memory::aligned_alloc<Real>(n_classes_sz, SCL_ALIGNMENT);
    }

    detail::FastRNG rng(seed);

    // Iterate
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Random node order
        detail::shuffle(order, N, rng);

        // Track changes atomically for parallel case
        std::atomic<bool> changed{false};

        if (use_parallel) {
            // Note: Parallel LP with random order requires careful handling
            // Here we parallelize in chunks to maintain some ordering
            const Size chunk_size = scl::algo::max2(Size(64), N / n_threads);
            const Size n_chunks = (N + chunk_size - 1) / chunk_size;

            for (Size chunk = 0; chunk < n_chunks; ++chunk) {
                Size start = chunk * chunk_size;
                Size end = scl::algo::min2(start + chunk_size, N);

                scl::threading::parallel_for(start, end, [&](size_t idx, size_t thread_rank) {
                    Index i = order[idx];

                    Real* class_votes = vote_pool.get(thread_rank);

                    auto indices = adjacency.primary_indices_unsafe(i);
                    auto values = adjacency.primary_values_unsafe(i);
                    const Index len = adjacency.primary_length_unsafe(i);

                    if (SCL_UNLIKELY(len == 0)) return;

                    // Zero votes
                    detail::fast_zero(class_votes, n_classes_sz);

                    // Count weighted votes
                    for (Index k = 0; k < len; ++k) {
                        Index j = indices[k];
                        Index lbl = labels_new[j];

                        if (SCL_LIKELY(lbl != config::UNLABELED && lbl < n_classes)) {
                            class_votes[lbl] += static_cast<Real>(values[k]);
                        }
                    }

                    // Find majority class
                    Index best_class = detail::find_argmax(class_votes, n_classes);
                    if (SCL_UNLIKELY(best_class != labels_new[i] && class_votes[best_class] > Real(0))) {
                        labels_new[i] = best_class;
                        changed.store(true, std::memory_order_relaxed);
                    }
                });
            }
        } else {
            // Sequential path
            bool local_changed = false;

            for (Index idx = 0; idx < n; ++idx) {
                Index i = order[idx];

                auto indices = adjacency.primary_indices_unsafe(i);
                auto values = adjacency.primary_values_unsafe(i);
                const Index len = adjacency.primary_length_unsafe(i);

                if (SCL_UNLIKELY(len == 0)) continue;

                detail::fast_zero(class_votes_st, n_classes_sz);

                // Unrolled vote accumulation
                Index k = 0;
                for (; k + 4 <= len; k += 4) {
                    Index j0 = indices[k + 0], j1 = indices[k + 1];
                    Index j2 = indices[k + 2], j3 = indices[k + 3];

                    Index lbl0 = labels_new[j0], lbl1 = labels_new[j1];
                    Index lbl2 = labels_new[j2], lbl3 = labels_new[j3];

                    if (lbl0 != config::UNLABELED && lbl0 < n_classes)
                        class_votes_st[lbl0] += static_cast<Real>(values[k + 0]);
                    if (lbl1 != config::UNLABELED && lbl1 < n_classes)
                        class_votes_st[lbl1] += static_cast<Real>(values[k + 1]);
                    if (lbl2 != config::UNLABELED && lbl2 < n_classes)
                        class_votes_st[lbl2] += static_cast<Real>(values[k + 2]);
                    if (lbl3 != config::UNLABELED && lbl3 < n_classes)
                        class_votes_st[lbl3] += static_cast<Real>(values[k + 3]);
                }

                for (; k < len; ++k) {
                    Index j = indices[k];
                    Index lbl = labels_new[j];

                    if (lbl != config::UNLABELED && lbl < n_classes) {
                        class_votes_st[lbl] += static_cast<Real>(values[k]);
                    }
                }

                Index best_class = detail::find_argmax(class_votes_st, n_classes);
                if (best_class != labels_new[i] && class_votes_st[best_class] > Real(0)) {
                    labels_new[i] = best_class;
                    local_changed = true;
                }
            }

            if (local_changed) changed.store(true, std::memory_order_relaxed);
        }

        // Copy back with SIMD
        std::memcpy(labels.ptr, labels_new, N * sizeof(Index));

        if (!changed.load(std::memory_order_relaxed)) break;
    }

    // Cleanup
    if (class_votes_st) {
        scl::memory::aligned_free(class_votes_st, SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(order, SCL_ALIGNMENT);
    scl::memory::aligned_free(labels_new, SCL_ALIGNMENT);
}

// =============================================================================
// Label Spreading (Regularized, Soft Labels) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void label_spreading(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> label_probs,
    Index n_classes,
    const bool* is_labeled,
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size n_classes_sz = static_cast<Size>(n_classes);
    const Size total_probs = N * n_classes_sz;

    SCL_CHECK_DIM(label_probs.len >= total_probs, "Spreading: label_probs buffer too small");

    if (SCL_UNLIKELY(n == 0 || n_classes == 0)) return;

    const Real one_minus_alpha = Real(1) - alpha;
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    // Compute normalized graph
    Real* row_sums = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    detail::compute_row_sums(adjacency, row_sums);

    // Compute D^(-1/2) with SIMD
    Real* d_inv_sqrt = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            d_inv_sqrt[i] = (row_sums[i] > Real(1e-15)) ?
                Real(1) / std::sqrt(row_sums[i]) : Real(0);
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            d_inv_sqrt[i] = (row_sums[i] > Real(1e-15)) ?
                Real(1) / std::sqrt(row_sums[i]) : Real(0);
        }
    }

    // Store initial labels
    Real* Y0 = scl::memory::aligned_alloc<Real>(total_probs, SCL_ALIGNMENT);
    Real* Y_new = scl::memory::aligned_alloc<Real>(total_probs, SCL_ALIGNMENT);
    std::memcpy(Y0, label_probs.ptr, total_probs * sizeof(Real));

    // Iterate
    for (Index iter = 0; iter < max_iter; ++iter) {
        if (use_parallel) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                Real* yi_new = Y_new + i * n_classes_sz;
                const Real* y0i = Y0 + i * n_classes_sz;

                // Initialize with (1-alpha) * Y0[i] or zero
                if (is_labeled[i]) {
                    namespace s = scl::simd;
                    using SimdTag = s::SimdTagFor<Real>;
                    const SimdTag d;
                    const size_t lanes = s::Lanes(d);
                    auto v_scale = s::Set(d, one_minus_alpha);
                    Index c = 0;

                    for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
                        auto v = s::Load(d, y0i + c);
                        s::Store(s::Mul(v, v_scale), d, yi_new + c);
                    }

                    for (; c < n_classes; ++c) {
                        yi_new[c] = one_minus_alpha * y0i[c];
                    }
                } else {
                    detail::fast_zero(yi_new, n_classes_sz);
                }

                // Add alpha * sum_j(S[i,j] * Y[j])
                auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
                auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
                const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));
                const Real d_i = d_inv_sqrt[i];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    Real w = static_cast<Real>(values[k]);
                    Real s_ij = alpha * d_i * w * d_inv_sqrt[j];
                    const Real* yj = label_probs.ptr + static_cast<Size>(j) * n_classes_sz;

                    // SIMD accumulation
                    namespace s = scl::simd;
                    using SimdTag = s::SimdTagFor<Real>;
                    const SimdTag d;
                    const size_t lanes = s::Lanes(d);
                    auto v_sij = s::Set(d, s_ij);
                    Index c = 0;

                    for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
                        auto v_yi = s::Load(d, yi_new + c);
                        auto v_yj = s::Load(d, yj + c);
                        s::Store(s::MulAdd(v_sij, v_yj, v_yi), d, yi_new + c);
                    }

                    for (; c < n_classes; ++c) {
                        yi_new[c] += s_ij * yj[c];
                    }
                }

                // Normalize
                detail::normalize_row(yi_new, n_classes);
            });
        } else {
            // Sequential path
            for (Index i = 0; i < n; ++i) {
                Real* yi_new = Y_new + static_cast<Size>(i) * n_classes_sz;
                const Real* y0i = Y0 + static_cast<Size>(i) * n_classes_sz;

                if (is_labeled[i]) {
                    for (Index c = 0; c < n_classes; ++c) {
                        yi_new[c] = one_minus_alpha * y0i[c];
                    }
                } else {
                    detail::fast_zero(yi_new, n_classes_sz);
                }

                auto indices = adjacency.primary_indices_unsafe(i);
                auto values = adjacency.primary_values_unsafe(i);
                const Index len = adjacency.primary_length_unsafe(i);
                const Real d_i = d_inv_sqrt[i];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    Real w = static_cast<Real>(values[k]);
                    Real s_ij = alpha * d_i * w * d_inv_sqrt[j];
                    const Real* yj = label_probs.ptr + static_cast<Size>(j) * n_classes_sz;

                    for (Index c = 0; c < n_classes; ++c) {
                        yi_new[c] += s_ij * yj[c];
                    }
                }

                detail::normalize_row(yi_new, n_classes);
            }
        }

        // Check convergence
        if (detail::check_convergence_real(label_probs.ptr, Y_new, total_probs, tol)) {
            std::memcpy(label_probs.ptr, Y_new, total_probs * sizeof(Real));
            break;
        }

        // Swap buffers
        std::memcpy(label_probs.ptr, Y_new, total_probs * sizeof(Real));
    }

    scl::memory::aligned_free(Y_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Y0, SCL_ALIGNMENT);
    scl::memory::aligned_free(d_inv_sqrt, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Inductive Label Transfer (Reference to Query) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void inductive_transfer(
    const Sparse<T, IsCSR>& ref_to_query,
    Array<const Index> reference_labels,
    Array<Index> query_labels,
    Index n_classes,
    Real confidence_threshold = Real(0.5)
) {
    const Index n_query = ref_to_query.rows();
    const Size N = static_cast<Size>(n_query);
    const Size n_classes_sz = static_cast<Size>(n_classes);

    SCL_CHECK_DIM(query_labels.len >= N, "Transfer: query_labels buffer too small");

    if (SCL_UNLIKELY(n_query == 0 || n_classes == 0)) return;

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> score_pool;
    if (use_parallel) {
        score_pool.init(n_threads, n_classes_sz);
    }

    auto process_query = [&](Index q, Real* class_scores) {
        auto indices = ref_to_query.row_indices_unsafe(q);
        auto values = ref_to_query.row_values_unsafe(q);
        const Index len = ref_to_query.row_length_unsafe(q);

        if (SCL_UNLIKELY(len == 0)) {
            query_labels[q] = config::UNLABELED;
            return;
        }

        detail::fast_zero(class_scores, n_classes_sz);

        Real total_weight = Real(0);

        // Weighted voting with unrolling
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int off = 0; off < 4; ++off) {
                Index ref_idx = indices[k + off];
                Index ref_label = reference_labels[ref_idx];
                Real w = static_cast<Real>(values[k + off]);

                if (SCL_LIKELY(ref_label >= 0 && ref_label < n_classes)) {
                    class_scores[ref_label] += w;
                    total_weight += w;
                }
            }
        }

        for (; k < len; ++k) {
            Index ref_idx = indices[k];
            Index ref_label = reference_labels[ref_idx];
            Real w = static_cast<Real>(values[k]);

            if (ref_label >= 0 && ref_label < n_classes) {
                class_scores[ref_label] += w;
                total_weight += w;
            }
        }

        // Find best class
        Index best_class = detail::find_argmax(class_scores, n_classes);
        Real best_score = (best_class >= 0) ? class_scores[best_class] : Real(0);

        // Apply threshold
        Real confidence = (total_weight > Real(1e-15)) ? best_score / total_weight : Real(0);
        query_labels[q] = (confidence >= confidence_threshold) ? best_class : config::UNLABELED;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t q, size_t thread_rank) {
            Real* class_scores = score_pool.get(thread_rank);
            process_query(static_cast<Index>(q), class_scores);
        });
    } else {
        Real* class_scores = scl::memory::aligned_alloc<Real>(n_classes_sz, SCL_ALIGNMENT);
        for (Index q = 0; q < n_query; ++q) {
            process_query(q, class_scores);
        }
        scl::memory::aligned_free(class_scores, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Confidence-Weighted Label Propagation - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void confidence_propagation(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Array<Real> confidence,
    Index n_classes,
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::DEFAULT_MAX_ITER
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size n_classes_sz = static_cast<Size>(n_classes);

    SCL_CHECK_DIM(labels.len >= N, "Propagation: labels buffer too small");
    SCL_CHECK_DIM(confidence.len >= N, "Propagation: confidence buffer too small");

    if (SCL_UNLIKELY(n == 0 || n_classes == 0)) return;

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    Index* labels_new = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    Real* conf_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    std::memcpy(labels_new, labels.ptr, N * sizeof(Index));
    std::memcpy(conf_new, confidence.ptr, N * sizeof(Real));

    scl::threading::WorkspacePool<Real> vote_pool;
    if (use_parallel) {
        vote_pool.init(n_threads, n_classes_sz);
    }

    for (Index iter = 0; iter < max_iter; ++iter) {
        std::atomic<bool> changed{false};

        auto process_node = [&](Index i, Real* class_votes) {
            auto indices = adjacency.primary_indices_unsafe(i);
            auto values = adjacency.primary_values_unsafe(i);
            const Index len = adjacency.primary_length_unsafe(i);

            if (SCL_UNLIKELY(len == 0)) return;

            detail::fast_zero(class_votes, n_classes_sz);

            Real total_conf = Real(0);

            // Confidence-weighted voting
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index lbl = labels[j];
                Real w = static_cast<Real>(values[k]);
                Real c = confidence[j];

                if (SCL_LIKELY(lbl >= 0 && lbl < n_classes)) {
                    Real wc = w * c;
                    class_votes[lbl] += wc;
                    total_conf += wc;
                }
            }

            // Include own label
            if (labels[i] >= 0 && labels[i] < n_classes) {
                Real own_vote = alpha * confidence[i];
                class_votes[labels[i]] += own_vote;
                total_conf += own_vote;
            }

            Index best_class = detail::find_argmax(class_votes, n_classes);
            Real best_votes = (best_class >= 0) ? class_votes[best_class] : Real(0);

            if (best_class != labels_new[i]) {
                labels_new[i] = best_class;
                changed.store(true, std::memory_order_relaxed);
            }

            conf_new[i] = (total_conf > Real(1e-15)) ? best_votes / total_conf : Real(0);
        };

        if (use_parallel) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
                Real* class_votes = vote_pool.get(thread_rank);
                process_node(static_cast<Index>(i), class_votes);
            });
        } else {
            Real* class_votes = scl::memory::aligned_alloc<Real>(n_classes_sz, SCL_ALIGNMENT);
            for (Index i = 0; i < n; ++i) {
                process_node(i, class_votes);
            }
            scl::memory::aligned_free(class_votes, SCL_ALIGNMENT);
        }

        // Copy back
        std::memcpy(labels.ptr, labels_new, N * sizeof(Index));
        std::memcpy(confidence.ptr, conf_new, N * sizeof(Real));

        if (!changed.load(std::memory_order_relaxed)) break;
    }

    scl::memory::aligned_free(conf_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(labels_new, SCL_ALIGNMENT);
}

// =============================================================================
// Harmonic Function Solution - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void harmonic_function(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> values,
    const bool* is_known,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(values.len >= N, "Harmonic: values buffer too small");

    if (SCL_UNLIKELY(n == 0)) return;

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    Real* row_sums = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* values_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    detail::compute_row_sums(adjacency, row_sums);
    std::memcpy(values_new, values.ptr, N * sizeof(Real));

    // Gauss-Seidel iteration
    for (Index iter = 0; iter < max_iter; ++iter) {
        std::atomic<Real> atomic_max_change{Real(0)};

        auto process_node = [&](Index i) {
            if (is_known[i]) return Real(0);

            auto indices = adjacency.primary_indices_unsafe(i);
            auto weights = adjacency.primary_values_unsafe(i);
            const Index len = adjacency.primary_length_unsafe(i);

            if (SCL_UNLIKELY(len == 0 || row_sums[i] <= Real(1e-15))) return Real(0);

            // SIMD summation for neighbors
            Real sum = Real(0);

            if (static_cast<Size>(len) >= config::SIMD_THRESHOLD) {
                // For sparse access, gather-based approach would be better
                // but standard approach for now
                Index k = 0;
                for (; k + 4 <= len; k += 4) {
                    sum += static_cast<Real>(weights[k + 0]) * values_new[indices[k + 0]];
                    sum += static_cast<Real>(weights[k + 1]) * values_new[indices[k + 1]];
                    sum += static_cast<Real>(weights[k + 2]) * values_new[indices[k + 2]];
                    sum += static_cast<Real>(weights[k + 3]) * values_new[indices[k + 3]];
                }

                for (; k < len; ++k) {
                    sum += static_cast<Real>(weights[k]) * values_new[indices[k]];
                }
            } else {
                for (Index k = 0; k < len; ++k) {
                    sum += static_cast<Real>(weights[k]) * values_new[indices[k]];
                }
            }

            Real new_val = sum / row_sums[i];
            Real change = (new_val >= values_new[i]) ?
                (new_val - values_new[i]) : (values_new[i] - new_val);

            values_new[i] = new_val;
            return change;
        };

        Real max_change = Real(0);

        if (use_parallel) {
            // Parallel Gauss-Seidel (red-black or Jacobi-like)
            // For simplicity, using Jacobi-style update
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                Real change = process_node(static_cast<Index>(i));

                // Atomic max update
                Real prev = atomic_max_change.load(std::memory_order_relaxed);
                while (change > prev &&
                       !atomic_max_change.compare_exchange_weak(prev, change,
                           std::memory_order_relaxed));
            });

            max_change = atomic_max_change.load(std::memory_order_relaxed);
        } else {
            for (Index i = 0; i < n; ++i) {
                Real change = process_node(i);
                max_change = scl::algo::max2(max_change, change);
            }
        }

        std::memcpy(values.ptr, values_new, N * sizeof(Real));

        if (max_change < tol) break;
    }

    scl::memory::aligned_free(values_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Get Predicted Labels from Soft Probabilities - Optimized
// =============================================================================

inline void get_hard_labels(
    Array<const Real> probs,
    Index n_nodes,
    Index n_classes,
    Array<Index> labels,
    Array<Real> max_probs = Array<Real>(nullptr, 0)
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size n_classes_sz = static_cast<Size>(n_classes);

    SCL_CHECK_DIM(labels.len >= N, "Labels: output buffer too small");

    const bool output_probs = (max_probs.ptr != nullptr && max_probs.len >= N);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        const Real* pi = probs.ptr + static_cast<Size>(i) * n_classes_sz;

        Index best_class = 0;
        Real best_prob = pi[0];

        // Unrolled argmax
        Index c = 1;
        for (; c + 4 <= n_classes; c += 4) {
            if (pi[c + 0] > best_prob) { best_prob = pi[c + 0]; best_class = c + 0; }
            if (pi[c + 1] > best_prob) { best_prob = pi[c + 1]; best_class = c + 1; }
            if (pi[c + 2] > best_prob) { best_prob = pi[c + 2]; best_class = c + 2; }
            if (pi[c + 3] > best_prob) { best_prob = pi[c + 3]; best_class = c + 3; }
        }

        for (; c < n_classes; ++c) {
            if (pi[c] > best_prob) {
                best_prob = pi[c];
                best_class = c;
            }
        }

        labels[i] = best_class;
        if (output_probs) {
            max_probs[i] = best_prob;
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n_nodes; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Initialize Soft Labels from Hard Labels - Optimized
// =============================================================================

inline void init_soft_labels(
    Array<const Index> hard_labels,
    Index n_classes,
    Array<Real> soft_labels,
    Real labeled_confidence = Real(1.0),
    Real unlabeled_prior = Real(0.0)
) {
    const Size N = hard_labels.len;
    const Size n_classes_sz = static_cast<Size>(n_classes);

    SCL_CHECK_DIM(soft_labels.len >= N * n_classes_sz, "SoftLabels: output buffer too small");

    const Real uniform_prob = (unlabeled_prior > Real(0)) ?
        unlabeled_prior : (Real(1) / static_cast<Real>(n_classes));

    const Real non_label_prob = (Real(1) - labeled_confidence) / static_cast<Real>(n_classes - 1);

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Size i) {
        Real* pi = soft_labels.ptr + i * n_classes_sz;
        Index label = hard_labels[i];

        if (label >= 0 && label < n_classes) {
            // Labeled node - vectorized fill
            namespace s = scl::simd;
            using SimdTag = s::SimdTagFor<Real>;
            const SimdTag d;
            const size_t lanes = s::Lanes(d);
            auto v_non_label = s::Set(d, non_label_prob);
            Index c = 0;

            for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
                s::Store(v_non_label, d, pi + c);
            }

            for (; c < n_classes; ++c) {
                pi[c] = non_label_prob;
            }

            pi[label] = labeled_confidence;
        } else {
            // Unlabeled node - vectorized fill
            namespace s = scl::simd;
            using SimdTag = s::SimdTagFor<Real>;
            const SimdTag d;
            const size_t lanes = s::Lanes(d);
            auto v_uniform = s::Set(d, uniform_prob);
            Index c = 0;

            for (; c + static_cast<Index>(lanes) <= n_classes; c += static_cast<Index>(lanes)) {
                s::Store(v_uniform, d, pi + c);
            }

            for (; c < n_classes; ++c) {
                pi[c] = uniform_prob;
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(i);
        });
    } else {
        for (Size i = 0; i < N; ++i) {
            process_node(i);
        }
    }
}

} // namespace scl::kernel::propagation
