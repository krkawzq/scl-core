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
// BRIEF: Diffusion processes on sparse graphs for trajectory and imputation
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
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Real MIN_PROB = Real(1e-15);
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Check convergence
SCL_FORCE_INLINE bool check_convergence(
    const Real* vals_old,
    const Real* vals_new,
    Size n,
    Real tol
) {
    Real diff = Real(0);
    for (Size i = 0; i < n; ++i) {
        Real d = vals_new[i] - vals_old[i];
        diff += (d >= Real(0)) ? d : -d;
    }
    return diff < tol;
}

// Compute row sums
template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& adj,
    Real* row_sums
) {
    const Index n = adj.primary_dim();
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

// Sparse matrix-vector multiply: y = T * x
template <typename T, bool IsCSR>
void spmv(
    const Sparse<T, IsCSR>& T_mat,
    const Real* x,
    Real* y,
    Size n
) {
    scl::algo::zero(y, n);

    for (Index i = 0; i < static_cast<Index>(n); ++i) {
        auto indices = T_mat.primary_indices(i);
        auto values = T_mat.primary_values(i);
        const Index len = T_mat.primary_length(i);

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            sum += static_cast<Real>(values[k]) * x[j];
        }
        y[i] = sum;
    }
}

// Sparse matrix-vector multiply transpose: y = T^T * x
template <typename T, bool IsCSR>
void spmv_transpose(
    const Sparse<T, IsCSR>& T_mat,
    const Real* x,
    Real* y,
    Size n
) {
    scl::algo::zero(y, n);

    for (Index i = 0; i < static_cast<Index>(n); ++i) {
        auto indices = T_mat.primary_indices(i);
        auto values = T_mat.primary_values(i);
        const Index len = T_mat.primary_length(i);

        Real xi = x[i];
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            y[j] += static_cast<Real>(values[k]) * xi;
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
    Real* transition_values,  // In-place modification of values copy
    bool symmetric = true
) {
    const Index n = adjacency.primary_dim();

    // Compute row sums (degrees)
    Real* row_sums = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::compute_row_sums(adjacency, row_sums);

    if (symmetric) {
        // Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        Real* d_inv_sqrt = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
        for (Index i = 0; i < n; ++i) {
            d_inv_sqrt[i] = (row_sums[i] > config::MIN_PROB)
                ? Real(1) / std::sqrt(row_sums[i]) : Real(0);
        }

        Size val_idx = 0;
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            const Index len = adjacency.primary_length(i);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                transition_values[val_idx] *= d_inv_sqrt[i] * d_inv_sqrt[j];
                ++val_idx;
            }
        }

        scl::memory::aligned_free(d_inv_sqrt, SCL_ALIGNMENT);
    } else {
        // Row normalization: D^(-1) * A
        Size val_idx = 0;
        for (Index i = 0; i < n; ++i) {
            const Index len = adjacency.primary_length(i);
            Real inv_sum = (row_sums[i] > config::MIN_PROB)
                ? Real(1) / row_sums[i] : Real(0);

            for (Index k = 0; k < len; ++k) {
                transition_values[val_idx] *= inv_sum;
                ++val_idx;
            }
        }
    }

    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion on Dense Vector (Power Iteration)
// =============================================================================

template <typename T, bool IsCSR>
void diffuse_vector(
    const Sparse<T, IsCSR>& transition,
    Array<Real> x,  // In/Out: vector to diffuse
    Index n_steps
) {
    const Index n = transition.primary_dim();

    SCL_CHECK_DIM(x.len >= static_cast<Size>(n),
                  "Diffusion: vector size mismatch");

    if (n == 0 || n_steps == 0) return;

    Real* x_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        detail::spmv(transition, x.ptr, x_new, n);

        // Swap
        for (Index i = 0; i < n; ++i) {
            x[i] = x_new[i];
        }
    }

    scl::memory::aligned_free(x_new, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion on Dense Matrix (Each Column is a Feature)
// =============================================================================

template <typename T, bool IsCSR>
void diffuse_matrix(
    const Sparse<T, IsCSR>& transition,
    Array<Real> X,  // In/Out: n_nodes x n_features, row-major
    Index n_nodes,
    Index n_features,
    Index n_steps
) {
    SCL_CHECK_DIM(X.len >= static_cast<Size>(n_nodes) * static_cast<Size>(n_features),
                  "Diffusion: matrix size mismatch");

    if (n_nodes == 0 || n_features == 0 || n_steps == 0) return;

    Real* X_new = scl::memory::aligned_alloc<Real>(
        static_cast<Size>(n_nodes) * static_cast<Size>(n_features), SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        // For each feature column
        for (Index f = 0; f < n_features; ++f) {
            // Extract column f
            for (Index i = 0; i < n_nodes; ++i) {
                Real sum = Real(0);
                auto indices = transition.primary_indices(i);
                auto values = transition.primary_values(i);
                const Index len = transition.primary_length(i);

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    sum += static_cast<Real>(values[k]) *
                           X[static_cast<Size>(j) * n_features + f];
                }
                X_new[static_cast<Size>(i) * n_features + f] = sum;
            }
        }

        // Copy back
        Size total = static_cast<Size>(n_nodes) * static_cast<Size>(n_features);
        for (Size i = 0; i < total; ++i) {
            X[i] = X_new[i];
        }
    }

    scl::memory::aligned_free(X_new, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion Pseudotime (DPT)
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

    SCL_CHECK_DIM(pseudotime.len >= static_cast<Size>(n),
                  "DPT: output buffer too small");
    SCL_CHECK_ARG(root_cell >= 0 && root_cell < n,
                  "DPT: root_cell out of range");

    if (n == 0) return;

    // Initialize: probability starts at root
    Real* p = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* p_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* hitting_time = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    scl::algo::zero(p, static_cast<Size>(n));
    scl::algo::zero(hitting_time, static_cast<Size>(n));
    p[root_cell] = Real(1);

    // Compute expected hitting time via iteration
    // H[i] = E[first hit time to i starting from root]
    // Using mean first passage time computation

    Real* visited_prob = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    scl::algo::zero(visited_prob, static_cast<Size>(n));
    visited_prob[root_cell] = Real(1);

    for (Index iter = 1; iter <= max_iter; ++iter) {
        // Diffuse one step: p_new = T * p
        detail::spmv(transition, p, p_new, n);

        // Accumulate hitting time: nodes first visited at this step
        for (Index i = 0; i < n; ++i) {
            if (visited_prob[i] < Real(0.99) && p_new[i] > config::MIN_PROB) {
                Real new_prob = p_new[i] * (Real(1) - visited_prob[i]);
                hitting_time[i] += static_cast<Real>(iter) * new_prob;
                visited_prob[i] += new_prob;
            }
        }

        // Check convergence
        Real max_unvisited = Real(0);
        for (Index i = 0; i < n; ++i) {
            max_unvisited = scl::algo::max2(max_unvisited, Real(1) - visited_prob[i]);
        }

        if (max_unvisited < tol) break;

        // Swap
        for (Index i = 0; i < n; ++i) {
            p[i] = p_new[i];
        }
    }

    // Normalize pseudotime to [0, 1]
    Real max_time = Real(0);
    for (Index i = 0; i < n; ++i) {
        if (visited_prob[i] > Real(0.5)) {
            hitting_time[i] /= visited_prob[i];
            max_time = scl::algo::max2(max_time, hitting_time[i]);
        } else {
            hitting_time[i] = Real(-1);  // Unreachable
        }
    }

    if (max_time > Real(0)) {
        for (Index i = 0; i < n; ++i) {
            pseudotime[i] = (hitting_time[i] >= Real(0))
                ? hitting_time[i] / max_time : Real(1);
        }
    } else {
        for (Index i = 0; i < n; ++i) {
            pseudotime[i] = (i == root_cell) ? Real(0) : Real(1);
        }
    }

    scl::memory::aligned_free(visited_prob, SCL_ALIGNMENT);
    scl::memory::aligned_free(hitting_time, SCL_ALIGNMENT);
    scl::memory::aligned_free(p_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(p, SCL_ALIGNMENT);
}

// =============================================================================
// Multi-Root DPT (Average from Multiple Roots)
// =============================================================================

template <typename T, bool IsCSR>
void compute_dpt_multi_root(
    const Sparse<T, IsCSR>& transition,
    Array<const Index> root_cells,
    Array<Real> pseudotime,
    Index max_iter = config::MAX_ITER
) {
    const Index n = transition.primary_dim();

    SCL_CHECK_DIM(pseudotime.len >= static_cast<Size>(n),
                  "DPT: output buffer too small");

    if (n == 0 || root_cells.len == 0) return;

    Real* temp_pt = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    scl::algo::zero(pseudotime.ptr, static_cast<Size>(n));

    Index n_valid_roots = 0;
    for (Size r = 0; r < root_cells.len; ++r) {
        Index root = root_cells[r];
        if (root < 0 || root >= n) continue;

        compute_dpt(transition, root, Array<Real>(temp_pt, n), max_iter);

        for (Index i = 0; i < n; ++i) {
            pseudotime[i] += temp_pt[i];
        }
        ++n_valid_roots;
    }

    if (n_valid_roots > 0) {
        Real inv_roots = Real(1) / static_cast<Real>(n_valid_roots);
        for (Index i = 0; i < n; ++i) {
            pseudotime[i] *= inv_roots;
        }
    }

    scl::memory::aligned_free(temp_pt, SCL_ALIGNMENT);
}

// =============================================================================
// Random Walk with Restart
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

    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n),
                  "RWR: output buffer too small");

    if (n == 0) return;

    // Build restart vector
    Real* restart = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    scl::algo::zero(restart, static_cast<Size>(n));

    Size n_seeds = 0;
    for (Size s = 0; s < seed_nodes.len; ++s) {
        Index idx = seed_nodes[s];
        if (idx >= 0 && idx < n) {
            restart[idx] = Real(1);
            ++n_seeds;
        }
    }

    if (n_seeds == 0) {
        scl::algo::zero(scores.ptr, static_cast<Size>(n));
        scl::memory::aligned_free(restart, SCL_ALIGNMENT);
        return;
    }

    // Normalize restart vector
    Real inv_seeds = Real(1) / static_cast<Real>(n_seeds);
    for (Index i = 0; i < n; ++i) {
        restart[i] *= inv_seeds;
    }

    // Initialize scores with restart
    for (Index i = 0; i < n; ++i) {
        scores[i] = restart[i];
    }

    Real* scores_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    // Iterate: s = alpha * T * s + (1 - alpha) * r
    for (Index iter = 0; iter < max_iter; ++iter) {
        detail::spmv(transition, scores.ptr, scores_new, n);

        for (Index i = 0; i < n; ++i) {
            scores_new[i] = alpha * scores_new[i] + (Real(1) - alpha) * restart[i];
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

    scl::memory::aligned_free(scores_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(restart, SCL_ALIGNMENT);
}

// =============================================================================
// Diffusion Maps Embedding (Top k Eigenvectors of Transition Matrix)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_map_embedding(
    const Sparse<T, IsCSR>& transition,
    Array<Real> embedding,  // n_nodes x n_components, row-major
    Index n_components,
    Index n_iter = 50
) {
    const Index n = transition.primary_dim();
    const Size total = static_cast<Size>(n) * static_cast<Size>(n_components);

    SCL_CHECK_DIM(embedding.len >= total,
                  "DiffusionMap: embedding buffer too small");

    if (n == 0 || n_components == 0) return;

    // Power iteration for top eigenvectors
    // Initialize with random values
    Real* Q = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* Q_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    // Simple initialization
    for (Size i = 0; i < total; ++i) {
        // Deterministic pseudo-random initialization
        Q[i] = std::sin(static_cast<Real>(i) * Real(0.1)) + Real(0.5);
    }

    // Orthogonalize initial vectors (simple Gram-Schmidt)
    for (Index c = 0; c < n_components; ++c) {
        Real* qc = Q + static_cast<Size>(c);

        // Subtract projections onto previous vectors
        for (Index p = 0; p < c; ++p) {
            Real* qp = Q + static_cast<Size>(p);
            Real dot = Real(0);
            Real norm_p = Real(0);

            for (Index i = 0; i < n; ++i) {
                Size ic = static_cast<Size>(i) * n_components + c;
                Size ip = static_cast<Size>(i) * n_components + p;
                dot += Q[ic] * Q[ip];
                norm_p += Q[ip] * Q[ip];
            }

            if (norm_p > config::MIN_PROB) {
                Real coeff = dot / norm_p;
                for (Index i = 0; i < n; ++i) {
                    Size ic = static_cast<Size>(i) * n_components + c;
                    Size ip = static_cast<Size>(i) * n_components + p;
                    Q[ic] -= coeff * Q[ip];
                }
            }
        }

        // Normalize
        Real norm = Real(0);
        for (Index i = 0; i < n; ++i) {
            Size ic = static_cast<Size>(i) * n_components + c;
            norm += Q[ic] * Q[ic];
        }
        if (norm > config::MIN_PROB) {
            Real inv_norm = Real(1) / std::sqrt(norm);
            for (Index i = 0; i < n; ++i) {
                Size ic = static_cast<Size>(i) * n_components + c;
                Q[ic] *= inv_norm;
            }
        }
    }

    // Power iteration
    Real* temp = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index iter = 0; iter < n_iter; ++iter) {
        // Apply transition matrix to each column
        for (Index c = 0; c < n_components; ++c) {
            // Extract column c
            for (Index i = 0; i < n; ++i) {
                temp[i] = Q[static_cast<Size>(i) * n_components + c];
            }

            // Multiply: T * temp
            for (Index i = 0; i < n; ++i) {
                auto indices = transition.primary_indices(i);
                auto values = transition.primary_values(i);
                const Index len = transition.primary_length(i);

                Real sum = Real(0);
                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    sum += static_cast<Real>(values[k]) * temp[j];
                }
                Q_new[static_cast<Size>(i) * n_components + c] = sum;
            }
        }

        // Orthogonalize (Gram-Schmidt)
        for (Index c = 0; c < n_components; ++c) {
            for (Index p = 0; p < c; ++p) {
                Real dot = Real(0);
                Real norm_p = Real(0);

                for (Index i = 0; i < n; ++i) {
                    Size ic = static_cast<Size>(i) * n_components + c;
                    Size ip = static_cast<Size>(i) * n_components + p;
                    dot += Q_new[ic] * Q_new[ip];
                    norm_p += Q_new[ip] * Q_new[ip];
                }

                if (norm_p > config::MIN_PROB) {
                    Real coeff = dot / norm_p;
                    for (Index i = 0; i < n; ++i) {
                        Size ic = static_cast<Size>(i) * n_components + c;
                        Size ip = static_cast<Size>(i) * n_components + p;
                        Q_new[ic] -= coeff * Q_new[ip];
                    }
                }
            }

            // Normalize
            Real norm = Real(0);
            for (Index i = 0; i < n; ++i) {
                Size ic = static_cast<Size>(i) * n_components + c;
                norm += Q_new[ic] * Q_new[ic];
            }
            if (norm > config::MIN_PROB) {
                Real inv_norm = Real(1) / std::sqrt(norm);
                for (Index i = 0; i < n; ++i) {
                    Size ic = static_cast<Size>(i) * n_components + c;
                    Q_new[ic] *= inv_norm;
                }
            }
        }

        // Swap
        Real* tmp = Q;
        Q = Q_new;
        Q_new = tmp;
    }

    // Copy to output
    for (Size i = 0; i < total; ++i) {
        embedding[i] = Q[i];
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q, SCL_ALIGNMENT);
}

// =============================================================================
// Heat Kernel Signature
// =============================================================================

template <typename T, bool IsCSR>
void heat_kernel_signature(
    const Sparse<T, IsCSR>& transition,
    Array<Real> signature,  // Output: n_nodes
    Real t = Real(1.0),     // Diffusion time
    Index n_steps = 10
) {
    const Index n = transition.primary_dim();

    SCL_CHECK_DIM(signature.len >= static_cast<Size>(n),
                  "HKS: output buffer too small");

    if (n == 0) return;

    // HKS(x) = sum_i exp(-lambda_i * t) * phi_i(x)^2
    // Approximated by: diagonal of exp(-t * L) ≈ (I - t/n_steps * L)^n_steps

    // Initialize with identity diagonal
    for (Index i = 0; i < n; ++i) {
        signature[i] = Real(1);
    }

    Real dt = t / static_cast<Real>(n_steps);
    Real* sig_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    for (Index step = 0; step < n_steps; ++step) {
        // sig_new = (1 - dt) * sig + dt * T * sig
        detail::spmv(transition, signature.ptr, sig_new, n);

        for (Index i = 0; i < n; ++i) {
            signature[i] = (Real(1) - dt) * signature[i] + dt * sig_new[i];
        }
    }

    scl::memory::aligned_free(sig_new, SCL_ALIGNMENT);
}

// =============================================================================
// MAGIC-style Imputation (Diffusion on Feature Matrix)
// =============================================================================

template <typename T, bool IsCSR>
void magic_impute(
    const Sparse<T, IsCSR>& transition,
    Array<Real> X,  // In/Out: n_nodes x n_features, row-major
    Index n_nodes,
    Index n_features,
    Index t = config::DEFAULT_N_STEPS
) {
    diffuse_matrix(transition, X, n_nodes, n_features, t);
}

// =============================================================================
// Compute Diffusion Distance Between All Pairs (Dense Output)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_distance(
    const Sparse<T, IsCSR>& transition,
    Array<Real> distances,  // n_nodes x n_nodes, row-major
    Index n_steps = config::DEFAULT_N_STEPS
) {
    const Index n = transition.primary_dim();
    const Size total = static_cast<Size>(n) * static_cast<Size>(n);

    SCL_CHECK_DIM(distances.len >= total,
                  "DiffusionDist: output buffer too small");

    if (n == 0) return;

    // Compute T^t via repeated multiplication
    // Then distance[i,j] = ||T^t[i,:] - T^t[j,:]||_2

    // Initialize with identity
    Real* T_power = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* T_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    scl::algo::zero(T_power, total);
    for (Index i = 0; i < n; ++i) {
        T_power[static_cast<Size>(i) * n + i] = Real(1);
    }

    // Compute T^t
    for (Index step = 0; step < n_steps; ++step) {
        // T_new = T * T_power (each column)
        for (Index c = 0; c < n; ++c) {
            for (Index i = 0; i < n; ++i) {
                auto indices = transition.primary_indices(i);
                auto values = transition.primary_values(i);
                const Index len = transition.primary_length(i);

                Real sum = Real(0);
                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    sum += static_cast<Real>(values[k]) * T_power[static_cast<Size>(j) * n + c];
                }
                T_new[static_cast<Size>(i) * n + c] = sum;
            }
        }

        // Swap
        Real* tmp = T_power;
        T_power = T_new;
        T_new = tmp;
    }

    // Compute pairwise distances
    for (Index i = 0; i < n; ++i) {
        for (Index j = i; j < n; ++j) {
            Real dist_sq = Real(0);
            for (Index k = 0; k < n; ++k) {
                Real diff = T_power[static_cast<Size>(i) * n + k] -
                           T_power[static_cast<Size>(j) * n + k];
                dist_sq += diff * diff;
            }
            Real dist = std::sqrt(dist_sq);
            distances[static_cast<Size>(i) * n + j] = dist;
            distances[static_cast<Size>(j) * n + i] = dist;
        }
    }

    scl::memory::aligned_free(T_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(T_power, SCL_ALIGNMENT);
}

} // namespace scl::kernel::diffusion
