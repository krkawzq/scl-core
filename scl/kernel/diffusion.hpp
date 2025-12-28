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
// FILE: scl/kernel/diffusion.hpp
// BRIEF: High-performance diffusion processes on sparse graphs
//
// Optimizations applied:
// - Parallel SpMV with SIMD-accelerated accumulation
// - Block-wise SpMM for cache efficiency
// - Modified Gram-Schmidt with reorthogonalization
// - Fused diffusion operations
// - Multi-accumulator patterns
// - Prefetching for sparse access
// =============================================================================

namespace scl::kernel::diffusion {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Real DEFAULT_ALPHA = Real(0.85);
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Index MAX_ITER = 100;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Real MIN_PROB = Real(1e-15);
    
    // Block sizes for cache efficiency
    constexpr Size SPMM_BLOCK_SIZE = 64;
    constexpr Size VECTOR_BLOCK_SIZE = 256;
    constexpr Size PREFETCH_DISTANCE = 4;
    
    // Gram-Schmidt
    constexpr Real REORTH_TOL = Real(0.7);  // Reorthogonalize if cos > 0.7
    constexpr Index MAX_REORTH = 2;
}

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD-Accelerated Vector Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE Real dot_product_simd(
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
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
        v_sum1 = s::MulAdd(s::Load(d, a + k + lanes), s::Load(d, b + k + lanes), v_sum1);
        v_sum2 = s::MulAdd(s::Load(d, a + k + 2*lanes), s::Load(d, b + k + 2*lanes), v_sum2);
        v_sum3 = s::MulAdd(s::Load(d, a + k + 3*lanes), s::Load(d, b + k + 3*lanes), v_sum3);
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    Real result = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(const Real* SCL_RESTRICT a, Size n) noexcept {
    return dot_product_simd(a, a, n);
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
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        auto y0 = s::Load(d, y + k);
        auto y1 = s::Load(d, y + k + lanes);
        auto y2 = s::Load(d, y + k + 2*lanes);
        auto y3 = s::Load(d, y + k + 3*lanes);

        y0 = s::MulAdd(v_alpha, s::Load(d, x + k), y0);
        y1 = s::MulAdd(v_alpha, s::Load(d, x + k + lanes), y1);
        y2 = s::MulAdd(v_alpha, s::Load(d, x + k + 2*lanes), y2);
        y3 = s::MulAdd(v_alpha, s::Load(d, x + k + 3*lanes), y3);

        s::Store(y0, d, y + k);
        s::Store(y1, d, y + k + lanes);
        s::Store(y2, d, y + k + 2*lanes);
        s::Store(y3, d, y + k + 3*lanes);
    }

    for (; k < n; ++k) {
        y[k] += alpha * x[k];
    }
}

SCL_HOT SCL_FORCE_INLINE void scale_simd(Real* SCL_RESTRICT x, Real alpha, Size n) noexcept {
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

SCL_HOT SCL_FORCE_INLINE void copy_simd(
    const Real* SCL_RESTRICT src,
    Real* SCL_RESTRICT dst,
    Size n
) noexcept {
    std::memcpy(dst, src, n * sizeof(Real));
}

// =============================================================================
// Parallel Sparse Matrix-Vector Multiply
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmv_parallel(
    const Sparse<T, IsCSR>& mat,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real sum = Real(0);

            // Prefetch next row's indices
            if (SCL_LIKELY(i + 1 < n)) {
                SCL_PREFETCH_READ(mat.primary_indices(idx + 1).ptr, 0);
            }

            // 4-way unrolled accumulation
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
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        }
    }
}

// =============================================================================
// Block Sparse Matrix-Dense Matrix Multiply (SpMM)
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmm_block(
    const Sparse<T, IsCSR>& mat,
    const Real* SCL_RESTRICT X,  // n x n_cols, row-major
    Real* SCL_RESTRICT Y,        // n x n_cols, row-major
    Size n,
    Size n_cols
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Zero output
    scl::algo::zero(Y, n * n_cols);

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real* Yi = Y + i * n_cols;

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real v = static_cast<Real>(values[k]);
                const Real* Xj = X + static_cast<Size>(j) * n_cols;

                // SIMD accumulation for this neighbor's contribution
                axpy_simd(v, Xj, Yi, n_cols);
            }
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real* Yi = Y + i * n_cols;

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real v = static_cast<Real>(values[k]);
                const Real* Xj = X + static_cast<Size>(j) * n_cols;

                for (Size c = 0; c < n_cols; ++c) {
                    Yi[c] += v * Xj[c];
                }
            }
        }
    }
}

// =============================================================================
// Fused SpMV with Linear Combination: y = alpha * T * x + beta * r
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmv_fused_linear(
    const Sparse<T, IsCSR>& mat,
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT r,
    Real* SCL_RESTRICT y,
    Size n,
    Real alpha,
    Real beta
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real sum = Real(0);
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

            y[i] = alpha * sum + beta * r[i];
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices(idx);
            auto values = mat.primary_values(idx);
            const Index len = mat.primary_length(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = alpha * sum + beta * r[i];
        }
    }
}

// =============================================================================
// Convergence Check with Early Exit
// =============================================================================

SCL_FORCE_INLINE bool check_convergence_simd(
    const Real* SCL_RESTRICT x_old,
    const Real* SCL_RESTRICT x_new,
    Size n,
    Real tol
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum = s::Zero(d);

    Size k = 0;
    for (; k + lanes <= n; k += lanes) {
        auto diff = s::Sub(s::Load(d, x_new + k), s::Load(d, x_old + k));
        v_sum = s::Add(v_sum, s::Abs(diff));
    }

    Real total = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < n; ++k) {
        Real d_val = x_new[k] - x_old[k];
        total += (d_val >= Real(0)) ? d_val : -d_val;
    }

    return total < tol;
}

// =============================================================================
// Modified Gram-Schmidt with Reorthogonalization
// =============================================================================

void orthogonalize_vectors(
    Real* Q,           // n x n_vecs, row-major
    Size n,
    Size n_vecs,
    Real* workspace    // temp storage of size n
) {
    for (Size v = 0; v < n_vecs; ++v) {
        Real* qv = Q + v;  // Column v (strided access)
        
        // Extract column to contiguous workspace
        for (Size i = 0; i < n; ++i) {
            workspace[i] = Q[i * n_vecs + v];
        }

        // Orthogonalize against previous vectors
        for (Index reorth = 0; reorth <= config::MAX_REORTH; ++reorth) {
            for (Size p = 0; p < v; ++p) {
                // Extract column p
                Real dot = Real(0);
                Real norm_p = Real(0);
                
                for (Size i = 0; i < n; ++i) {
                    Real qp_i = Q[i * n_vecs + p];
                    dot += workspace[i] * qp_i;
                    norm_p += qp_i * qp_i;
                }

                if (norm_p > config::MIN_PROB) {
                    Real coeff = dot / norm_p;
                    for (Size i = 0; i < n; ++i) {
                        workspace[i] -= coeff * Q[i * n_vecs + p];
                    }
                }
            }

            // Check if reorthogonalization needed
            Real norm_v = Real(0);
            for (Size i = 0; i < n; ++i) {
                norm_v += workspace[i] * workspace[i];
            }

            if (reorth == 0 && v > 0) {
                // Check orthogonality
                Real max_cos = Real(0);
                for (Size p = 0; p < v; ++p) {
                    Real dot = Real(0);
                    for (Size i = 0; i < n; ++i) {
                        dot += workspace[i] * Q[i * n_vecs + p];
                    }
                    Real cos_val = (norm_v > config::MIN_PROB) ? 
                                   std::abs(dot) / std::sqrt(norm_v) : Real(0);
                    max_cos = scl::algo::max2(max_cos, cos_val);
                }
                
                if (max_cos < config::REORTH_TOL) break;  // Good enough
            } else {
                break;
            }
        }

        // Normalize
        Real norm = Real(0);
        for (Size i = 0; i < n; ++i) {
            norm += workspace[i] * workspace[i];
        }
        
        if (norm > config::MIN_PROB) {
            Real inv_norm = Real(1) / std::sqrt(norm);
            for (Size i = 0; i < n; ++i) {
                Q[i * n_vecs + v] = workspace[i] * inv_norm;
            }
        } else {
            // Degenerate case: set to unit vector
            for (Size i = 0; i < n; ++i) {
                Q[i * n_vecs + v] = (i == v % n) ? Real(1) : Real(0);
            }
        }
    }
}

// =============================================================================
// Compute Row Sums (Degrees) with SIMD
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void compute_row_sums_parallel(
    const Sparse<T, IsCSR>& adj,
    Real* row_sums
) {
    const Index n = adj.primary_dim();
    const Size N = static_cast<Size>(n);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto values = adj.primary_values(static_cast<Index>(i));
            const Size len = static_cast<Size>(adj.primary_length(static_cast<Index>(i)));

            Real sum = Real(0);
            for (Size k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }
            row_sums[i] = sum;
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            auto values = adj.primary_values(i);
            const Index len = adj.primary_length(i);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]);
            }
            row_sums[i] = sum;
        }
    }
}

} // namespace detail

// =============================================================================
// Compute Transition Matrix (Row-Stochastic)
// =============================================================================

template <typename T, bool IsCSR>
void compute_transition_matrix(
    const Sparse<T, IsCSR>& adjacency,
    Real* transition_values,
    bool symmetric = true
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    Real* row_sums = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    detail::compute_row_sums_parallel(adjacency, row_sums);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (symmetric) {
        // Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        Real* d_inv_sqrt = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                d_inv_sqrt[i] = (row_sums[i] > config::MIN_PROB)
                    ? Real(1) / std::sqrt(row_sums[i]) : Real(0);
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                d_inv_sqrt[i] = (row_sums[i] > config::MIN_PROB)
                    ? Real(1) / std::sqrt(row_sums[i]) : Real(0);
            }
        }

        // Apply normalization
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            // Compute offsets for parallel access
            Size* offsets = scl::memory::aligned_alloc<Size>(N + 1, SCL_ALIGNMENT);
            offsets[0] = 0;
            for (Index i = 0; i < n; ++i) {
                offsets[i + 1] = offsets[i] + static_cast<Size>(adjacency.primary_length(i));
            }

            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                auto indices = adjacency.primary_indices(static_cast<Index>(i));
                const Index len = adjacency.primary_length(static_cast<Index>(i));
                Size base = offsets[i];
                Real di = d_inv_sqrt[i];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    transition_values[base + k] *= di * d_inv_sqrt[j];
                }
            });

            scl::memory::aligned_free(offsets, SCL_ALIGNMENT);
        } else {
            Size val_idx = 0;
            for (Index i = 0; i < n; ++i) {
                auto indices = adjacency.primary_indices(i);
                const Index len = adjacency.primary_length(i);
                Real di = d_inv_sqrt[i];

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    transition_values[val_idx++] *= di * d_inv_sqrt[j];
                }
            }
        }

        scl::memory::aligned_free(d_inv_sqrt, SCL_ALIGNMENT);
    } else {
        // Row normalization: D^(-1) * A
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            Size* offsets = scl::memory::aligned_alloc<Size>(N + 1, SCL_ALIGNMENT);
            offsets[0] = 0;
            for (Index i = 0; i < n; ++i) {
                offsets[i + 1] = offsets[i] + static_cast<Size>(adjacency.primary_length(i));
            }

            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                const Index len = adjacency.primary_length(static_cast<Index>(i));
                Size base = offsets[i];
                Real inv_sum = (row_sums[i] > config::MIN_PROB)
                    ? Real(1) / row_sums[i] : Real(0);

                for (Index k = 0; k < len; ++k) {
                    transition_values[base + k] *= inv_sum;
                }
            });

            scl::memory::aligned_free(offsets, SCL_ALIGNMENT);
        } else {
            Size val_idx = 0;
            for (Index i = 0; i < n; ++i) {
                const Index len = adjacency.primary_length(i);
                Real inv_sum = (row_sums[i] > config::MIN_PROB)
                    ? Real(1) / row_sums[i] : Real(0);

                for (Index k = 0; k < len; ++k) {
                    transition_values[val_idx++] *= inv_sum;
                }
            }
        }
    }

    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion on Dense Vector
// =============================================================================

template <typename T, bool IsCSR>
void diffuse_vector(
    const Sparse<T, IsCSR>& transition,
    Array<Real> x,
    Index n_steps
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(x.len >= N, "Diffusion: vector size mismatch");

    if (n == 0 || n_steps == 0) return;

    Real* x_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        detail::spmv_parallel(transition, x.ptr, x_new, N);
        std::memcpy(x.ptr, x_new, N * sizeof(Real));
    }

    scl::memory::aligned_free(x_new, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion on Dense Matrix (Optimized SpMM)
// =============================================================================

template <typename T, bool IsCSR>
void diffuse_matrix(
    const Sparse<T, IsCSR>& transition,
    Array<Real> X,
    Index n_nodes,
    Index n_features,
    Index n_steps
) {
    const Size N = static_cast<Size>(n_nodes);
    const Size F = static_cast<Size>(n_features);
    const Size total = N * F;

    SCL_CHECK_DIM(X.len >= total, "Diffusion: matrix size mismatch");

    if (n_nodes == 0 || n_features == 0 || n_steps == 0) return;

    Real* X_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        detail::spmm_block(transition, X.ptr, X_new, N, F);
        std::memcpy(X.ptr, X_new, total * sizeof(Real));
    }

    scl::memory::aligned_free(X_new, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion Pseudotime (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void compute_dpt(
    const Sparse<T, IsCSR>& transition,
    Index root_cell,
    Array<Real> pseudotime,
    Index max_iter = config::MAX_ITER,
    Real tol = config::CONVERGENCE_TOL
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(pseudotime.len >= N, "DPT: output buffer too small");
    SCL_CHECK_ARG(root_cell >= 0 && root_cell < n, "DPT: root_cell out of range");

    if (n == 0) return;

    // Allocate working arrays
    Real* p = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* p_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* hitting_time = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* visited_prob = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    scl::algo::zero(p, N);
    scl::algo::zero(hitting_time, N);
    scl::algo::zero(visited_prob, N);
    
    p[root_cell] = Real(1);
    visited_prob[root_cell] = Real(1);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Iterate diffusion
    for (Index iter = 1; iter <= max_iter; ++iter) {
        detail::spmv_parallel(transition, p, p_new, N);

        // Update hitting times in parallel
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                if (visited_prob[i] < Real(0.99) && p_new[i] > config::MIN_PROB) {
                    Real remaining = Real(1) - visited_prob[i];
                    Real new_prob = p_new[i] * remaining;
                    hitting_time[i] += static_cast<Real>(iter) * new_prob;
                    visited_prob[i] += new_prob;
                    visited_prob[i] = scl::algo::min2(visited_prob[i], Real(1));
                }
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                if (visited_prob[i] < Real(0.99) && p_new[i] > config::MIN_PROB) {
                    Real remaining = Real(1) - visited_prob[i];
                    Real new_prob = p_new[i] * remaining;
                    hitting_time[i] += static_cast<Real>(iter) * new_prob;
                    visited_prob[i] += new_prob;
                    visited_prob[i] = scl::algo::min2(visited_prob[i], Real(1));
                }
            }
        }

        // Check convergence
        Real max_unvisited = Real(0);
        for (Size i = 0; i < N; ++i) {
            max_unvisited = scl::algo::max2(max_unvisited, Real(1) - visited_prob[i]);
        }

        if (max_unvisited < tol) break;

        std::swap(p, p_new);
    }

    // Normalize to [0, 1]
    Real max_time = Real(0);
    for (Size i = 0; i < N; ++i) {
        if (visited_prob[i] > Real(0.5)) {
            hitting_time[i] /= visited_prob[i];
            max_time = scl::algo::max2(max_time, hitting_time[i]);
        }
    }

    Real inv_max = (max_time > Real(0)) ? Real(1) / max_time : Real(1);
    
    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            pseudotime[i] = (visited_prob[i] > Real(0.5))
                ? hitting_time[i] * inv_max : Real(1);
        });
    } else {
        for (Size i = 0; i < N; ++i) {
            pseudotime[i] = (visited_prob[i] > Real(0.5))
                ? hitting_time[i] * inv_max : Real(1);
        }
    }

    scl::memory::aligned_free(visited_prob, SCL_ALIGNMENT);
    scl::memory::aligned_free(hitting_time, SCL_ALIGNMENT);
    scl::memory::aligned_free(p_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(p, SCL_ALIGNMENT);
}

// =============================================================================
// Multi-Root DPT (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void compute_dpt_multi_root(
    const Sparse<T, IsCSR>& transition,
    Array<const Index> root_cells,
    Array<Real> pseudotime,
    Index max_iter = config::MAX_ITER
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(pseudotime.len >= N, "DPT: output buffer too small");

    if (n == 0 || root_cells.len == 0) return;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    Size n_roots = root_cells.len;

    // Parallel computation of DPT from each root
    Real* all_pt = scl::memory::aligned_alloc<Real>(N * n_roots, SCL_ALIGNMENT);
    Index* valid_roots = scl::memory::aligned_alloc<Index>(n_roots, SCL_ALIGNMENT);
    
    Index n_valid = 0;
    for (Size r = 0; r < n_roots; ++r) {
        if (root_cells[r] >= 0 && root_cells[r] < n) {
            valid_roots[n_valid++] = root_cells[r];
        }
    }

    if (n_valid == 0) {
        scl::algo::fill(pseudotime.ptr, N, Real(1));
        scl::memory::aligned_free(valid_roots, SCL_ALIGNMENT);
        scl::memory::aligned_free(all_pt, SCL_ALIGNMENT);
        return;
    }

    // Compute DPT for each root (can parallelize across roots for small n)
    for (Index r = 0; r < n_valid; ++r) {
        compute_dpt(transition, valid_roots[r], 
                    Array<Real>(all_pt + r * N, N), max_iter);
    }

    // Average across roots
    scl::algo::zero(pseudotime.ptr, N);
    
    for (Index r = 0; r < n_valid; ++r) {
        detail::axpy_simd(Real(1), all_pt + r * N, pseudotime.ptr, N);
    }

    Real inv_roots = Real(1) / static_cast<Real>(n_valid);
    detail::scale_simd(pseudotime.ptr, inv_roots, N);

    scl::memory::aligned_free(valid_roots, SCL_ALIGNMENT);
    scl::memory::aligned_free(all_pt, SCL_ALIGNMENT);
}

// =============================================================================
// Random Walk with Restart (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void random_walk_with_restart(
    const Sparse<T, IsCSR>& transition,
    Array<const Index> seed_nodes,
    Array<Real> scores,
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::MAX_ITER,
    Real tol = config::CONVERGENCE_TOL
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(scores.len >= N, "RWR: output buffer too small");

    if (n == 0) return;

    // Build and normalize restart vector
    Real* restart = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    scl::algo::zero(restart, N);

    Size n_seeds = 0;
    for (Size s = 0; s < seed_nodes.len; ++s) {
        Index idx = seed_nodes[s];
        if (idx >= 0 && idx < n) {
            restart[idx] = Real(1);
            ++n_seeds;
        }
    }

    if (n_seeds == 0) {
        scl::algo::zero(scores.ptr, N);
        scl::memory::aligned_free(restart, SCL_ALIGNMENT);
        return;
    }

    Real inv_seeds = Real(1) / static_cast<Real>(n_seeds);
    detail::scale_simd(restart, inv_seeds, N);

    // Initialize scores
    detail::copy_simd(restart, scores.ptr, N);

    Real* scores_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real beta = Real(1) - alpha;

    // Iterate with fused operation
    for (Index iter = 0; iter < max_iter; ++iter) {
        detail::spmv_fused_linear(transition, scores.ptr, restart, scores_new, N, alpha, beta);

        if (detail::check_convergence_simd(scores.ptr, scores_new, N, tol)) {
            detail::copy_simd(scores_new, scores.ptr, N);
            break;
        }

        std::swap(scores.ptr, scores_new);
    }

    // Ensure final result is in scores
    if (scores.ptr != scores_new) {
        // Already in correct place
    }

    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(restart, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion Map Embedding (Optimized Power Iteration)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_map_embedding(
    const Sparse<T, IsCSR>& transition,
    Array<Real> embedding,
    Index n_components,
    Index n_iter = 50
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size K = static_cast<Size>(n_components);
    const Size total = N * K;

    SCL_CHECK_DIM(embedding.len >= total, "DiffusionMap: embedding buffer too small");

    if (n == 0 || n_components == 0) return;

    // Allocate working arrays
    Real* Q = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* Q_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* workspace = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Deterministic initialization with good spread
    for (Size i = 0; i < N; ++i) {
        for (Size c = 0; c < K; ++c) {
            // Chebyshev nodes-inspired initialization
            Real t = static_cast<Real>(i) / static_cast<Real>(N);
            Real phase = static_cast<Real>(c + 1) * Real(3.14159265358979323846);
            Q[i * K + c] = std::cos(phase * t) + Real(0.1) * std::sin(phase * t * Real(2.7));
        }
    }

    // Initial orthogonalization
    detail::orthogonalize_vectors(Q, N, K, workspace);

    // Power iteration
    for (Index iter = 0; iter < n_iter; ++iter) {
        // Apply transition matrix: Q_new = T * Q
        detail::spmm_block(transition, Q, Q_new, N, K);

        // Orthogonalize
        detail::orthogonalize_vectors(Q_new, N, K, workspace);

        // Swap
        std::swap(Q, Q_new);
    }

    // Copy to output
    std::memcpy(embedding.ptr, Q, total * sizeof(Real));

    scl::memory::aligned_free(workspace, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q, SCL_ALIGNMENT);
}

// =============================================================================
// Heat Kernel Signature (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void heat_kernel_signature(
    const Sparse<T, IsCSR>& transition,
    Array<Real> signature,
    Real t = Real(1.0),
    Index n_steps = 10
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(signature.len >= N, "HKS: output buffer too small");

    if (n == 0) return;

    // Initialize with identity diagonal
    scl::algo::fill(signature.ptr, N, Real(1));

    Real dt = t / static_cast<Real>(n_steps);
    Real one_minus_dt = Real(1) - dt;

    Real* sig_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // sig_new = (1 - dt) * sig + dt * T * sig
    for (Index step = 0; step < n_steps; ++step) {
        detail::spmv_parallel(transition, signature.ptr, temp, N);

        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                signature[i] = one_minus_dt * signature[i] + dt * temp[i];
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                signature[i] = one_minus_dt * signature[i] + dt * temp[i];
            }
        }
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(sig_new, SCL_ALIGNMENT);
}

// =============================================================================
// MAGIC-style Imputation
// =============================================================================

template <typename T, bool IsCSR>
void magic_impute(
    const Sparse<T, IsCSR>& transition,
    Array<Real> X,
    Index n_nodes,
    Index n_features,
    Index t = config::DEFAULT_N_STEPS
) {
    diffuse_matrix(transition, X, n_nodes, n_features, t);
}

// =============================================================================
// Diffusion Distance (Optimized for Large Graphs)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_distance(
    const Sparse<T, IsCSR>& transition,
    Array<Real> distances,
    Index n_steps = config::DEFAULT_N_STEPS
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size total = N * N;

    SCL_CHECK_DIM(distances.len >= total, "DiffusionDist: output buffer too small");

    if (n == 0) return;

    // For large graphs, compute row by row to save memory
    if (N > 1000) {
        // Row-by-row computation
        Real* row_i = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        Real* row_j = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
        Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

        for (Size i = 0; i < N; ++i) {
            // Initialize row i as e_i
            scl::algo::zero(row_i, N);
            row_i[i] = Real(1);

            // Diffuse
            for (Index step = 0; step < n_steps; ++step) {
                detail::spmv_parallel(transition, row_i, temp, N);
                std::memcpy(row_i, temp, N * sizeof(Real));
            }

            // Compute distances from row i
            for (Size j = i; j < N; ++j) {
                if (i == j) {
                    distances[i * N + j] = Real(0);
                    continue;
                }

                // Initialize row j as e_j
                scl::algo::zero(row_j, N);
                row_j[j] = Real(1);

                // Diffuse
                for (Index step = 0; step < n_steps; ++step) {
                    detail::spmv_parallel(transition, row_j, temp, N);
                    std::memcpy(row_j, temp, N * sizeof(Real));
                }

                // Compute L2 distance
                Real dist_sq = Real(0);
                for (Size k = 0; k < N; ++k) {
                    Real diff = row_i[k] - row_j[k];
                    dist_sq += diff * diff;
                }

                Real dist = std::sqrt(dist_sq);
                distances[i * N + j] = dist;
                distances[j * N + i] = dist;
            }
        }

        scl::memory::aligned_free(temp, SCL_ALIGNMENT);
        scl::memory::aligned_free(row_j, SCL_ALIGNMENT);
        scl::memory::aligned_free(row_i, SCL_ALIGNMENT);
    } else {
        // Small graph: compute full T^t matrix
        Real* T_power = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
        Real* T_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

        // Initialize with identity
        scl::algo::zero(T_power, total);
        for (Size i = 0; i < N; ++i) {
            T_power[i * N + i] = Real(1);
        }

        // Compute T^t using SpMM
        for (Index step = 0; step < n_steps; ++step) {
            detail::spmm_block(transition, T_power, T_new, N, N);
            std::swap(T_power, T_new);
        }

        // Compute pairwise distances
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();

        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            distances[i * N + i] = Real(0);

            for (Size j = i + 1; j < N; ++j) {
                Real dist_sq = Real(0);
                for (Size k = 0; k < N; ++k) {
                    Real diff = T_power[i * N + k] - T_power[j * N + k];
                    dist_sq += diff * diff;
                }

                Real dist = std::sqrt(dist_sq);
                distances[i * N + j] = dist;
                distances[j * N + i] = dist;
            }
        });

        scl::memory::aligned_free(T_new, SCL_ALIGNMENT);
        scl::memory::aligned_free(T_power, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Personalized PageRank (Alias for RWR with single seed)
// =============================================================================

template <typename T, bool IsCSR>
void personalized_pagerank(
    const Sparse<T, IsCSR>& transition,
    Index seed_node,
    Array<Real> scores,
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::MAX_ITER,
    Real tol = config::CONVERGENCE_TOL
) {
    Index seeds[1] = { seed_node };
    random_walk_with_restart(transition, Array<const Index>(seeds, 1), 
                             scores, alpha, max_iter, tol);
}

// =============================================================================
// Lazy Random Walk (Slower diffusion with self-loops)
// =============================================================================

template <typename T, bool IsCSR>
void lazy_random_walk(
    const Sparse<T, IsCSR>& transition,
    Array<Real> x,
    Index n_steps,
    Real laziness = Real(0.5)  // Probability of staying
) {
    const Index n = transition.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(x.len >= N, "LazyRW: vector size mismatch");

    if (n == 0 || n_steps == 0) return;

    Real move_prob = Real(1) - laziness;
    Real* x_new = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        detail::spmv_parallel(transition, x.ptr, temp, N);

        // x_new = laziness * x + (1 - laziness) * T * x
        const size_t n_threads = scl::threading::Scheduler::get_num_threads();
        
        if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                x.ptr[i] = laziness * x.ptr[i] + move_prob * temp[i];
            });
        } else {
            for (Size i = 0; i < N; ++i) {
                x.ptr[i] = laziness * x.ptr[i] + move_prob * temp[i];
            }
        }
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(x_new, SCL_ALIGNMENT);
}

} // namespace scl::kernel::diffusion
