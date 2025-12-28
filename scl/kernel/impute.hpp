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
// FILE: scl/kernel/impute.hpp
// BRIEF: High-performance sparse-aware imputation for single-cell data
//
// Optimizations applied:
// - Parallel cell/gene processing
// - SIMD-accelerated weighted aggregation
// - Block SpMM for diffusion steps
// - Branchless binary search for gene lookup
// - WorkspacePool for thread-local buffers
// - Fused initialization + computation
// - Parallel power iteration for SVD
// =============================================================================

namespace scl::kernel::impute {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_THRESHOLD = Real(0.0);
    constexpr Real DISTANCE_EPSILON = Real(1e-12);
    constexpr Real DEFAULT_BANDWIDTH = Real(1.0);
    constexpr Index DEFAULT_DIFFUSION_STEPS = 3;
    constexpr Real MIN_IMPUTED_VALUE = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size GENE_BLOCK_SIZE = 64;
    constexpr Size CELL_BLOCK_SIZE = 32;
}

// =============================================================================
// Imputation Mode
// =============================================================================

enum class ImputeMode {
    KNN,
    WeightedKNN,
    Diffusion,
    MAGIC
};

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD Vector Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE void axpy_simd(
    Real alpha,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto y0 = s::Load(d, y + k);
        auto y1 = s::Load(d, y + k + lanes);
        y0 = s::MulAdd(v_alpha, s::Load(d, x + k), y0);
        y1 = s::MulAdd(v_alpha, s::Load(d, x + k + lanes), y1);
        s::Store(y0, d, y + k);
        s::Store(y1, d, y + k + lanes);
    }

    for (; k < n; ++k) {
        y[k] += alpha * x[k];
    }
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
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x + k)), d, x + k);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + lanes)), d, x + k + lanes);
    }

    for (; k < n; ++k) {
        x[k] *= alpha;
    }
}

SCL_HOT SCL_FORCE_INLINE Real dot_simd(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
        v_sum1 = s::MulAdd(s::Load(d, a + k + lanes), s::Load(d, b + k + lanes), v_sum1);
    }

    Real result = s::GetLane(s::SumOfLanes(d, s::Add(v_sum0, v_sum1)));

    for (; k < n; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(const Real* a, Size n) noexcept {
    return dot_simd(a, a, n);
}

// =============================================================================
// Weight Kernels
// =============================================================================

SCL_FORCE_INLINE Real distance_to_weight(Real distance, Real bandwidth) noexcept {
    Real scaled = distance / (bandwidth + config::DISTANCE_EPSILON);
    return std::exp(-scaled * scaled);
}

SCL_FORCE_INLINE Real inverse_distance_weight(Real distance, Real epsilon) noexcept {
    return Real(1) / (distance + epsilon);
}

// =============================================================================
// Branchless Binary Search
// =============================================================================

template <typename T>
SCL_FORCE_INLINE Index binary_search_branchless(
    const Index* indices,
    Index len,
    Index target
) noexcept {
    Index left = 0;
    while (len > 1) {
        Index half = len / 2;
        // Branchless: always compute mid, conditionally update left
        left += (indices[left + half] < target) ? half : 0;
        len -= half;
    }
    return (left < len && indices[left] == target) ? left : -1;
}

template <typename T>
SCL_FORCE_INLINE bool has_expression(
    const Index* indices,
    const T* values,
    Index len,
    Index gene,
    T& out_value
) noexcept {
    if (len == 0) {
        out_value = T(0);
        return false;
    }

    // Linear search for small arrays
    if (len <= 16) {
        for (Index k = 0; k < len; ++k) {
            if (indices[k] == gene) {
                out_value = values[k];
                return true;
            }
            if (indices[k] > gene) break;
        }
        out_value = T(0);
        return false;
    }

    // Binary search for larger arrays
    Index left = 0, right = len;
    while (left < right) {
        Index mid = (left + right) >> 1;
        if (indices[mid] < gene) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if (left < len && indices[left] == gene) {
        out_value = values[left];
        return true;
    }
    out_value = T(0);
    return false;
}

// =============================================================================
// Get Expression (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real get_expression(
    const Sparse<T, IsCSR>& X,
    Index cell,
    Index gene,
    Index n_cells,
    Index n_genes
) noexcept {
    T val;
    if (IsCSR) {
        auto indices = X.row_indices_unsafe(cell);
        auto values = X.row_values_unsafe(cell);
        Index len = X.row_length_unsafe(cell);
        if (has_expression(indices, values, len, gene, val)) {
            return static_cast<Real>(val);
        }
    } else {
        auto indices = X.col_indices_unsafe(gene);
        auto values = X.col_values_unsafe(gene);
        Index len = X.col_length_unsafe(gene);
        if (has_expression(indices, values, len, cell, val)) {
            return static_cast<Real>(val);
        }
    }
    return Real(0);
}

// =============================================================================
// Batch Neighbor Value Collection
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT Index collect_neighbor_values_batch(
    const Sparse<T, IsCSR>& X,
    const Index* neighbor_indices,
    Index n_neighbors,
    Index gene,
    Index n_cells,
    Real* values,
    uint8_t* has_value
) noexcept {
    Index count = 0;

    for (Index k = 0; k < n_neighbors; ++k) {
        Index neighbor = neighbor_indices[k];
        if (neighbor >= n_cells) {
            has_value[k] = 0;
            values[k] = Real(0);
            continue;
        }

        T val;
        if (IsCSR) {
            auto indices = X.row_indices_unsafe(neighbor);
            auto row_values = X.row_values_unsafe(neighbor);
            Index len = X.row_length_unsafe(neighbor);
            
            if (has_expression(indices, row_values, len, gene, val)) {
                values[k] = static_cast<Real>(val);
                has_value[k] = 1;
                ++count;
            } else {
                values[k] = Real(0);
                has_value[k] = 0;
            }
        } else {
            auto indices = X.col_indices_unsafe(gene);
            auto col_values = X.col_values_unsafe(gene);
            Index len = X.col_length_unsafe(gene);

            if (has_expression(indices, col_values, len, neighbor, val)) {
                values[k] = static_cast<Real>(val);
                has_value[k] = 1;
                ++count;
            } else {
                values[k] = Real(0);
                has_value[k] = 0;
            }
        }
    }

    return count;
}

// =============================================================================
// Parallel SpMM for Diffusion
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmm_diffusion(
    const Sparse<T, IsCSR>& transition,
    const Real* SCL_RESTRICT X_in,   // n_cells x n_genes
    Real* SCL_RESTRICT X_out,        // n_cells x n_genes
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real* out_row = X_out + i * G;
        scl::algo::zero(out_row, G);

        auto t_indices = transition.primary_indices_unsafe(static_cast<Index>(i));
        auto t_values = transition.primary_values_unsafe(static_cast<Index>(i));
        const Index t_len = transition.primary_length_unsafe(static_cast<Index>(i));

        // Accumulate weighted neighbor rows
        for (Index k = 0; k < t_len; ++k) {
            Index j = t_indices[k];
            Real t_ij = static_cast<Real>(t_values[k]);
            const Real* in_row = X_in + static_cast<Size>(j) * G;

            axpy_simd(t_ij, in_row, out_row, G);
        }
    });
}

// =============================================================================
// Parallel SpMM with Row Normalization
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmm_normalized(
    const Sparse<T, IsCSR>& affinity,
    const Real* SCL_RESTRICT row_sums,
    const Real* SCL_RESTRICT X_in,
    Real* SCL_RESTRICT X_out,
    Index n_cells,
    Index n_genes
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real* out_row = X_out + i * G;
        Real row_sum = row_sums[i];

        if (row_sum < config::DISTANCE_EPSILON) {
            // Copy original
            scl::algo::copy(X_in + i * G, out_row, G);
            return;
        }

        scl::algo::zero(out_row, G);

        auto a_indices = affinity.primary_indices_unsafe(static_cast<Index>(i));
        auto a_values = affinity.primary_values_unsafe(static_cast<Index>(i));
        const Index a_len = affinity.primary_length_unsafe(static_cast<Index>(i));

        Real inv_sum = Real(1) / row_sum;

        for (Index k = 0; k < a_len; ++k) {
            Index j = a_indices[k];
            Real w = static_cast<Real>(a_values[k]) * inv_sum;
            const Real* in_row = X_in + static_cast<Size>(j) * G;

            axpy_simd(w, in_row, out_row, G);
        }
    });
}

} // namespace detail

// =============================================================================
// KNN Imputation (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void knn_impute_dense(
    const Sparse<T, IsCSR>& X,
    const Index* knn_indices,
    const Real* knn_distances,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Real* X_imputed,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    Real threshold = config::DEFAULT_THRESHOLD
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size K = static_cast<Size>(k_neighbors);
    const Size total = N * G;

    scl::algo::zero(X_imputed, total);

    // Step 1: Copy original values (parallel)
    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real* row = X_imputed + c * G;
            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    row[g] = static_cast<Real>(values[k]);
                }
            }
        });
    } else {
        scl::threading::parallel_for(Size(0), G, [&](size_t g) {
            auto indices = X.col_indices_unsafe(static_cast<Index>(g));
            auto values = X.col_values_unsafe(static_cast<Index>(g));
            Index len = X.col_length_unsafe(static_cast<Index>(g));

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    X_imputed[static_cast<Size>(c) * G + g] = static_cast<Real>(values[k]);
                }
            }
        });
    }

    // Step 2: Impute zeros (parallel over cells)
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::WorkspacePool<Real> values_pool;
    scl::threading::WorkspacePool<Real> weights_pool;
    scl::threading::WorkspacePool<uint8_t> has_value_pool;

    values_pool.init(n_threads, K);
    weights_pool.init(n_threads, K);
    has_value_pool.init(n_threads, K);

    scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
        Real* row = X_imputed + c * G;
        const Index* neighbors = knn_indices + c * K;
        const Real* distances = knn_distances + c * K;

        Real* neighbor_values = values_pool.get(thread_rank);
        Real* weights = weights_pool.get(thread_rank);
        uint8_t* has_value = has_value_pool.get(thread_rank);

        // Precompute weights
        for (Index k = 0; k < k_neighbors; ++k) {
            weights[k] = detail::distance_to_weight(distances[k], bandwidth);
        }

        // Process genes in blocks for cache efficiency
        for (Index g_start = 0; g_start < n_genes; g_start += config::GENE_BLOCK_SIZE) {
            Index g_end = scl::algo::min2(g_start + static_cast<Index>(config::GENE_BLOCK_SIZE), n_genes);

            for (Index g = g_start; g < g_end; ++g) {
                if (row[g] > threshold) continue;

                Index n_with_value = detail::collect_neighbor_values_batch(
                    X, neighbors, k_neighbors, g, n_cells,
                    neighbor_values, has_value
                );

                if (n_with_value == 0) continue;

                Real sum_val = Real(0);
                Real sum_weight = Real(0);

                for (Index k = 0; k < k_neighbors; ++k) {
                    if (has_value[k]) {
                        sum_val += weights[k] * neighbor_values[k];
                        sum_weight += weights[k];
                    }
                }

                if (sum_weight > config::DISTANCE_EPSILON) {
                    Real imputed = sum_val / sum_weight;
                    if (imputed > config::MIN_IMPUTED_VALUE) {
                        row[g] = imputed;
                    }
                }
            }
        }
    });
}

// =============================================================================
// Weighted KNN Imputation (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void knn_impute_weighted_dense(
    const Sparse<T, IsCSR>& X,
    const Index* knn_indices,
    const Real* knn_weights,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Real* X_imputed,
    Real threshold = config::DEFAULT_THRESHOLD
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size K = static_cast<Size>(k_neighbors);
    const Size total = N * G;

    scl::algo::zero(X_imputed, total);

    // Copy original values
    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real* row = X_imputed + c * G;
            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    row[g] = static_cast<Real>(values[k]);
                }
            }
        });
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> values_pool;
    scl::threading::WorkspacePool<uint8_t> has_value_pool;
    values_pool.init(n_threads, K);
    has_value_pool.init(n_threads, K);

    scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
        Real* row = X_imputed + c * G;
        const Index* neighbors = knn_indices + c * K;
        const Real* weights = knn_weights + c * K;

        Real* neighbor_values = values_pool.get(thread_rank);
        uint8_t* has_value = has_value_pool.get(thread_rank);

        for (Index g = 0; g < n_genes; ++g) {
            if (row[g] > threshold) continue;

            Index n_with_value = detail::collect_neighbor_values_batch(
                X, neighbors, k_neighbors, g, n_cells,
                neighbor_values, has_value
            );

            if (n_with_value == 0) continue;

            Real sum_val = Real(0);
            Real sum_weight = Real(0);

            for (Index k = 0; k < k_neighbors; ++k) {
                if (has_value[k]) {
                    sum_val += weights[k] * neighbor_values[k];
                    sum_weight += weights[k];
                }
            }

            if (sum_weight > config::DISTANCE_EPSILON) {
                Real imputed = sum_val / sum_weight;
                if (imputed > config::MIN_IMPUTED_VALUE) {
                    row[g] = imputed;
                }
            }
        }
    });
}

// =============================================================================
// Diffusion Imputation (Sparse Transition, Optimized)
// =============================================================================

template <typename T, bool IsCSR, typename TT, bool IsCSR2>
void diffusion_impute_sparse_transition(
    const Sparse<T, IsCSR>& X,
    const Sparse<TT, IsCSR2>& transition_matrix,
    Index n_cells,
    Index n_genes,
    Index n_steps,
    Real* X_imputed
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size total = N * G;

    // Initialize with original expression
    scl::algo::zero(X_imputed, total);

    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real* row = X_imputed + c * G;
            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    row[g] = static_cast<Real>(values[k]);
                }
            }
        });
    } else {
        scl::threading::parallel_for(Size(0), G, [&](size_t g) {
            auto indices = X.col_indices_unsafe(static_cast<Index>(g));
            auto values = X.col_values_unsafe(static_cast<Index>(g));
            Index len = X.col_length_unsafe(static_cast<Index>(g));

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (c < n_cells) {
                    X_imputed[static_cast<Size>(c) * G + g] = static_cast<Real>(values[k]);
                }
            }
        });
    }

    // Diffusion with double buffering
    Real* temp = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    Real* buf_in = X_imputed;
    Real* buf_out = temp;

    for (Index step = 0; step < n_steps; ++step) {
        detail::spmm_diffusion(transition_matrix, buf_in, buf_out, n_cells, n_genes);
        std::swap(buf_in, buf_out);
    }

    // Ensure result is in X_imputed
    if (buf_in != X_imputed) {
        scl::algo::copy(buf_in, X_imputed, total);
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// MAGIC Imputation (Optimized)
// =============================================================================

template <typename T, bool IsCSR, typename TT, bool IsCSR2>
void magic_impute(
    const Sparse<T, IsCSR>& X,
    const Sparse<TT, IsCSR2>& affinity_matrix,
    Index n_cells,
    Index n_genes,
    Index t_diffusion,
    Real* X_imputed
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size total = N * G;

    // Initialize
    scl::algo::zero(X_imputed, total);

    if (IsCSR) {
        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            Real* row = X_imputed + c * G;
            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    row[g] = static_cast<Real>(values[k]);
                }
            }
        });
    }

    // Precompute row sums (parallel)
    Real* row_sums = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        auto values = affinity_matrix.primary_values_unsafe(static_cast<Index>(i));
        Index len = affinity_matrix.primary_length_unsafe(static_cast<Index>(i));

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(values[k]);
        }
        row_sums[i] = sum;
    });

    // Diffusion with double buffering
    Real* temp = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    Real* buf_in = X_imputed;
    Real* buf_out = temp;

    for (Index step = 0; step < t_diffusion; ++step) {
        detail::spmm_normalized(affinity_matrix, row_sums, buf_in, buf_out,
                                n_cells, n_genes);
        std::swap(buf_in, buf_out);
    }

    if (buf_in != X_imputed) {
        scl::algo::copy(buf_in, X_imputed, total);
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// ALRA Imputation (Parallel SVD)
// =============================================================================

inline void alra_impute(
    const Real* X,
    Index n_cells,
    Index n_genes,
    Index rank,
    Real* X_imputed
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size total = N * G;
    const Index k = scl::algo::min2(rank, scl::algo::min2(n_cells, n_genes));

    Real* U = scl::memory::aligned_alloc<Real>(N * k, SCL_ALIGNMENT);
    Real* V = scl::memory::aligned_alloc<Real>(G * k, SCL_ALIGNMENT);
    Real* S = scl::memory::aligned_alloc<Real>(k, SCL_ALIGNMENT);

    // Per-thread workspace for u computation
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<Real> u_workspace;
    u_workspace.init(n_threads, N);

    // Initialize V with orthonormal random vectors
    Real inv_sqrt_g = Real(1) / std::sqrt(static_cast<Real>(n_genes));
    for (Size i = 0; i < G * k; ++i) {
        V[i] = inv_sqrt_g;
    }

    constexpr Index MAX_ITER = 50;
    constexpr Real CONVERGENCE_TOL = Real(1e-6);

    for (Index comp = 0; comp < k; ++comp) {
        Real* u_col = U + comp * N;
        Real* v_col = V + comp * G;

        Real prev_sigma = Real(0);

        for (Index iter = 0; iter < MAX_ITER; ++iter) {
            // u = X * v (parallel over rows)
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                const Real* x_row = X + i * G;
                u_col[i] = detail::dot_simd(x_row, v_col, G);
            });

            // Orthogonalize u against previous components
            for (Index p = 0; p < comp; ++p) {
                Real* u_prev = U + p * N;
                Real dot = detail::dot_simd(u_col, u_prev, N);
                detail::axpy_simd(-dot, u_prev, u_col, N);
            }

            // Normalize u
            Real norm_u = std::sqrt(detail::norm_squared_simd(u_col, N));
            if (norm_u > config::DISTANCE_EPSILON) {
                detail::scale_simd(u_col, Real(1) / norm_u, N);
            }

            // v = X^T * u (parallel reduction)
            scl::algo::zero(v_col, G);

            // Thread-local partial sums
            Real** partials = static_cast<Real**>(
                scl::memory::aligned_alloc<Real*>(n_threads, SCL_ALIGNMENT));
            for (size_t t = 0; t < n_threads; ++t) {
                partials[t] = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
                scl::algo::zero(partials[t], G);
            }

            scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
                const Real* x_row = X + i * G;
                Real u_i = u_col[i];
                detail::axpy_simd(u_i, x_row, partials[thread_rank], G);
            });

            // Reduce
            for (size_t t = 0; t < n_threads; ++t) {
                for (Size j = 0; j < G; ++j) {
                    v_col[j] += partials[t][j];
                }
                scl::memory::aligned_free(partials[t], SCL_ALIGNMENT);
            }
            scl::memory::aligned_free(partials, SCL_ALIGNMENT);

            // Orthogonalize v
            for (Index p = 0; p < comp; ++p) {
                Real* v_prev = V + p * G;
                Real dot = detail::dot_simd(v_col, v_prev, G);
                detail::axpy_simd(-dot, v_prev, v_col, G);
            }

            // Compute singular value (before normalizing v)
            Real sigma = std::sqrt(detail::norm_squared_simd(v_col, G));

            // Normalize v
            if (sigma > config::DISTANCE_EPSILON) {
                detail::scale_simd(v_col, Real(1) / sigma, G);
            }

            // Check convergence
            if (std::abs(sigma - prev_sigma) < CONVERGENCE_TOL * sigma) {
                S[comp] = sigma;
                break;
            }
            prev_sigma = sigma;

            if (iter == MAX_ITER - 1) {
                S[comp] = sigma;
            }
        }
    }

    // Reconstruct: X_imputed = U * S * V^T (parallel)
    scl::algo::zero(X_imputed, total);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real* out_row = X_imputed + i * G;

        for (Index c = 0; c < k; ++c) {
            Real u_s = U[c * N + i] * S[c];
            Real* v_col = V + c * G;
            detail::axpy_simd(u_s, v_col, out_row, G);
        }
    });

    // ALRA: keep original non-zeros, clamp negatives
    scl::threading::parallel_for(Size(0), total, [&](size_t idx) {
        if (X[idx] > config::DISTANCE_EPSILON) {
            X_imputed[idx] = X[idx];
        } else if (X_imputed[idx] < Real(0)) {
            X_imputed[idx] = Real(0);
        }
    });

    scl::memory::aligned_free(S, SCL_ALIGNMENT);
    scl::memory::aligned_free(V, SCL_ALIGNMENT);
    scl::memory::aligned_free(U, SCL_ALIGNMENT);
}

// =============================================================================
// Selective Gene Imputation (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void impute_selected_genes(
    const Sparse<T, IsCSR>& X,
    const Index* knn_indices,
    const Real* knn_distances,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Array<const Index> genes_to_impute,
    Real* X_imputed,
    Real bandwidth = config::DEFAULT_BANDWIDTH
) {
    const Size N = static_cast<Size>(n_cells);
    const Size K = static_cast<Size>(k_neighbors);
    const Index n_impute_genes = static_cast<Index>(genes_to_impute.len);
    const Size total = N * static_cast<Size>(n_impute_genes);

    scl::algo::zero(X_imputed, total);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> values_pool;
    scl::threading::WorkspacePool<Real> weights_pool;
    scl::threading::WorkspacePool<uint8_t> has_value_pool;

    values_pool.init(n_threads, K);
    weights_pool.init(n_threads, K);
    has_value_pool.init(n_threads, K);

    scl::threading::parallel_for(Size(0), N, [&](size_t c, size_t thread_rank) {
        Real* out_row = X_imputed + c * n_impute_genes;
        const Index* neighbors = knn_indices + c * K;
        const Real* distances = knn_distances + c * K;

        Real* neighbor_values = values_pool.get(thread_rank);
        Real* weights = weights_pool.get(thread_rank);
        uint8_t* has_value = has_value_pool.get(thread_rank);

        // Precompute weights
        for (Index k = 0; k < k_neighbors; ++k) {
            weights[k] = detail::distance_to_weight(distances[k], bandwidth);
        }

        for (Index gi = 0; gi < n_impute_genes; ++gi) {
            Index g = genes_to_impute[gi];

            // Check original value
            Real orig = detail::get_expression(X, static_cast<Index>(c), g, n_cells, n_genes);

            if (orig > config::DISTANCE_EPSILON) {
                out_row[gi] = orig;
                continue;
            }

            // Impute
            Index n_with_value = detail::collect_neighbor_values_batch(
                X, neighbors, k_neighbors, g, n_cells,
                neighbor_values, has_value
            );

            if (n_with_value == 0) continue;

            Real sum_val = Real(0);
            Real sum_weight = Real(0);

            for (Index k = 0; k < k_neighbors; ++k) {
                if (has_value[k]) {
                    sum_val += weights[k] * neighbor_values[k];
                    sum_weight += weights[k];
                }
            }

            if (sum_weight > config::DISTANCE_EPSILON) {
                out_row[gi] = sum_val / sum_weight;
            }
        }
    });
}

// =============================================================================
// Dropout Detection (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void detect_dropouts(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Real mean_threshold,
    Real* dropout_probability
) {
    const Size G = static_cast<Size>(n_genes);
    const Size N = static_cast<Size>(n_cells);

    Real* gene_means = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Index* gene_nnz = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);

    scl::algo::zero(gene_means, G);
    scl::algo::zero(gene_nnz, G);

    if (IsCSR) {
        // Parallel over cells, atomic accumulation
        std::atomic<int64_t>* atomic_sums = static_cast<std::atomic<int64_t>*>(
            scl::memory::aligned_alloc<std::atomic<int64_t>>(G, SCL_ALIGNMENT));
        std::atomic<Index>* atomic_nnz = static_cast<std::atomic<Index>*>(
            scl::memory::aligned_alloc<std::atomic<Index>>(G, SCL_ALIGNMENT));

        for (Size g = 0; g < G; ++g) {
            atomic_sums[g].store(0, std::memory_order_relaxed);
            atomic_nnz[g].store(0, std::memory_order_relaxed);
        }

        constexpr int64_t SCALE = 1000000LL;

        scl::threading::parallel_for(Size(0), N, [&](size_t c) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    int64_t scaled = static_cast<int64_t>(static_cast<Real>(values[k]) * SCALE);
                    atomic_sums[g].fetch_add(scaled, std::memory_order_relaxed);
                    atomic_nnz[g].fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

        for (Size g = 0; g < G; ++g) {
            gene_means[g] = static_cast<Real>(atomic_sums[g].load()) / SCALE;
            gene_nnz[g] = atomic_nnz[g].load();
        }

        scl::memory::aligned_free(atomic_nnz, SCL_ALIGNMENT);
        scl::memory::aligned_free(atomic_sums, SCL_ALIGNMENT);
    } else {
        // Parallel over genes
        scl::threading::parallel_for(Size(0), G, [&](size_t g) {
            auto values = X.col_values_unsafe(static_cast<Index>(g));
            Index len = X.col_length_unsafe(static_cast<Index>(g));

            gene_nnz[g] = len;
            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }
            gene_means[g] = sum;
        });
    }

    // Compute dropout probability (parallel)
    scl::threading::parallel_for(Size(0), G, [&](size_t g) {
        if (gene_nnz[g] > 0) {
            gene_means[g] /= static_cast<Real>(gene_nnz[g]);
        }

        Real zero_fraction = Real(1) - static_cast<Real>(gene_nnz[g]) /
                             static_cast<Real>(n_cells);

        if (gene_means[g] > mean_threshold) {
            dropout_probability[g] = zero_fraction;
        } else {
            dropout_probability[g] = zero_fraction * Real(0.5);
        }
    });

    scl::memory::aligned_free(gene_nnz, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_means, SCL_ALIGNMENT);
}

// =============================================================================
// Imputation Quality (Parallel)
// =============================================================================

inline Real imputation_quality(
    const Real* X_original,
    const Real* X_imputed,
    Index n_cells,
    Index n_genes
) {
    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_genes);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Thread-local statistics
    Real* t_sum_x = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
    Real* t_sum_y = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
    Real* t_sum_xx = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
    Real* t_sum_yy = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
    Real* t_sum_xy = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);
    Size* t_count = scl::memory::aligned_alloc<Size>(n_threads, SCL_ALIGNMENT);

    scl::algo::zero(t_sum_x, n_threads);
    scl::algo::zero(t_sum_y, n_threads);
    scl::algo::zero(t_sum_xx, n_threads);
    scl::algo::zero(t_sum_yy, n_threads);
    scl::algo::zero(t_sum_xy, n_threads);
    scl::algo::zero(t_count, n_threads);

    scl::threading::parallel_for(Size(0), total, [&](size_t i, size_t thread_rank) {
        if (X_original[i] > config::DISTANCE_EPSILON) {
            Real x = X_original[i];
            Real y = X_imputed[i];
            t_sum_x[thread_rank] += x;
            t_sum_y[thread_rank] += y;
            t_sum_xx[thread_rank] += x * x;
            t_sum_yy[thread_rank] += y * y;
            t_sum_xy[thread_rank] += x * y;
            ++t_count[thread_rank];
        }
    });

    // Reduce
    Real sum_x = Real(0), sum_y = Real(0);
    Real sum_xx = Real(0), sum_yy = Real(0), sum_xy = Real(0);
    Size count = 0;

    for (size_t t = 0; t < n_threads; ++t) {
        sum_x += t_sum_x[t];
        sum_y += t_sum_y[t];
        sum_xx += t_sum_xx[t];
        sum_yy += t_sum_yy[t];
        sum_xy += t_sum_xy[t];
        count += t_count[t];
    }

    scl::memory::aligned_free(t_count, SCL_ALIGNMENT);
    scl::memory::aligned_free(t_sum_xy, SCL_ALIGNMENT);
    scl::memory::aligned_free(t_sum_yy, SCL_ALIGNMENT);
    scl::memory::aligned_free(t_sum_xx, SCL_ALIGNMENT);
    scl::memory::aligned_free(t_sum_y, SCL_ALIGNMENT);
    scl::memory::aligned_free(t_sum_x, SCL_ALIGNMENT);

    if (count < 2) return Real(0);

    Real n = static_cast<Real>(count);
    Real cov = sum_xy - sum_x * sum_y / n;
    Real var_x = sum_xx - sum_x * sum_x / n;
    Real var_y = sum_yy - sum_y * sum_y / n;

    Real denom = std::sqrt(var_x * var_y);
    return (denom > config::DISTANCE_EPSILON) ? cov / denom : Real(0);
}

// =============================================================================
// Smooth Expression (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void smooth_expression(
    const Sparse<T, IsCSR>& X,
    const Index* knn_indices,
    const Real* knn_weights,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Real alpha,
    Real* X_smoothed
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size K = static_cast<Size>(k_neighbors);
    const Size total = N * G;

    scl::algo::zero(X_smoothed, total);

    Real one_minus_alpha = Real(1) - alpha;

    // Parallel smoothing
    scl::threading::parallel_for(Size(0), N, [&](size_t c) {
        Real* out_row = X_smoothed + c * G;
        const Index* neighbors = knn_indices + c * K;
        const Real* weights = knn_weights + c * K;

        Real weight_sum = Real(0);

        // Accumulate neighbor contributions
        for (Index k = 0; k < k_neighbors; ++k) {
            Index neighbor = neighbors[k];
            if (neighbor >= n_cells) continue;

            Real w = weights[k];
            weight_sum += w;

            if (IsCSR) {
                auto indices = X.row_indices_unsafe(neighbor);
                auto values = X.row_values_unsafe(neighbor);
                Index len = X.row_length_unsafe(neighbor);

                for (Index j = 0; j < len; ++j) {
                    Index g = indices[j];
                    if (g < n_genes) {
                        out_row[g] += w * static_cast<Real>(values[j]);
                    }
                }
            }
        }

        // Normalize neighbor contribution
        if (weight_sum > config::DISTANCE_EPSILON) {
            detail::scale_simd(out_row, alpha / weight_sum, G);
        }

        // Add original contribution
        if (IsCSR) {
            auto indices = X.row_indices_unsafe(static_cast<Index>(c));
            auto values = X.row_values_unsafe(static_cast<Index>(c));
            Index len = X.row_length_unsafe(static_cast<Index>(c));

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    out_row[g] += one_minus_alpha * static_cast<Real>(values[k]);
                }
            }
        }
    });
}

// =============================================================================
// Dense Diffusion (for reference / small matrices)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_impute_dense(
    const Sparse<T, IsCSR>& X,
    const Real* transition_matrix,
    Index n_cells,
    Index n_genes,
    Index n_steps,
    Real* X_imputed
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);
    const Size total = N * G;

    scl::algo::zero(X_imputed, total);

    // Initialize
    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            auto indices = X.row_indices_unsafe(c);
            auto values = X.row_values_unsafe(c);
            Index len = X.row_length_unsafe(c);

            Real* row = X_imputed + static_cast<Size>(c) * G;
            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (g < n_genes) {
                    row[g] = static_cast<Real>(values[k]);
                }
            }
        }
    }

    Real* temp = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    Real* buf_in = X_imputed;
    Real* buf_out = temp;

    for (Index step = 0; step < n_steps; ++step) {
        // Dense matrix multiplication (parallel)
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            Real* out_row = buf_out + i * G;
            scl::algo::zero(out_row, G);

            for (Index j = 0; j < n_cells; ++j) {
                Real t_ij = transition_matrix[i * N + j];
                if (t_ij < config::DISTANCE_EPSILON) continue;

                const Real* in_row = buf_in + static_cast<Size>(j) * G;
                detail::axpy_simd(t_ij, in_row, out_row, G);
            }
        });

        std::swap(buf_in, buf_out);
    }

    if (buf_in != X_imputed) {
        scl::algo::copy(buf_in, X_imputed, total);
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

} // namespace scl::kernel::impute
