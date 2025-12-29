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
// FILE: scl/kernel/velocity.hpp
// BRIEF: RNA velocity analysis for single-cell transcriptomics (OPTIMIZED)
// =============================================================================

namespace scl::kernel::velocity {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_MIN_R2 = Real(0.01);
    constexpr Real DEFAULT_MIN_LIKELIHOOD = Real(0.001);
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Index DEFAULT_N_ITERATIONS = 100;
    constexpr Real DEFAULT_ALPHA = Real(0.05);
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real INF_VALUE = Real(1e30);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// Velocity Models
// =============================================================================

enum class VelocityModel {
    SteadyState,
    Dynamical,
    Stochastic
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Fast PRNG for stochastic sampling
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

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }
};

// SIMD-optimized linear regression: y = slope * x + intercept
SCL_HOT SCL_FORCE_INLINE void linear_regression(
    const Real* SCL_RESTRICT x,
    const Real* SCL_RESTRICT y,
    Size n,
    Real& slope,
    Real& intercept,
    Real& r2
) {
    if (SCL_UNLIKELY(n == 0)) {
        slope = Real(0);
        intercept = Real(0);
        r2 = Real(0);
        return;
    }

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    // Compute means with multi-accumulator
    auto v_sum_x0 = s::Zero(d), v_sum_x1 = s::Zero(d);
    auto v_sum_y0 = s::Zero(d), v_sum_y1 = s::Zero(d);

    Size i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        v_sum_x0 = s::Add(v_sum_x0, s::Load(d, x + i));
        v_sum_x1 = s::Add(v_sum_x1, s::Load(d, x + i + lanes));
        v_sum_y0 = s::Add(v_sum_y0, s::Load(d, y + i));
        v_sum_y1 = s::Add(v_sum_y1, s::Load(d, y + i + lanes));
    }

    auto v_sum_x = s::Add(v_sum_x0, v_sum_x1);
    auto v_sum_y = s::Add(v_sum_y0, v_sum_y1);

    Real sum_x = s::GetLane(s::SumOfLanes(d, v_sum_x));
    Real sum_y = s::GetLane(s::SumOfLanes(d, v_sum_y));

    for (; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
    }

    Real inv_n = Real(1) / static_cast<Real>(n);
    Real mean_x = sum_x * inv_n;
    Real mean_y = sum_y * inv_n;

    // Compute covariance and variance with SIMD
    auto v_mean_x = s::Set(d, mean_x);
    auto v_mean_y = s::Set(d, mean_y);
    auto v_cov0 = s::Zero(d), v_cov1 = s::Zero(d);
    auto v_var_x0 = s::Zero(d), v_var_x1 = s::Zero(d);
    auto v_var_y0 = s::Zero(d), v_var_y1 = s::Zero(d);

    i = 0;
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        auto dx0 = s::Sub(s::Load(d, x + i), v_mean_x);
        auto dx1 = s::Sub(s::Load(d, x + i + lanes), v_mean_x);
        auto dy0 = s::Sub(s::Load(d, y + i), v_mean_y);
        auto dy1 = s::Sub(s::Load(d, y + i + lanes), v_mean_y);

        v_cov0 = s::MulAdd(dx0, dy0, v_cov0);
        v_cov1 = s::MulAdd(dx1, dy1, v_cov1);
        v_var_x0 = s::MulAdd(dx0, dx0, v_var_x0);
        v_var_x1 = s::MulAdd(dx1, dx1, v_var_x1);
        v_var_y0 = s::MulAdd(dy0, dy0, v_var_y0);
        v_var_y1 = s::MulAdd(dy1, dy1, v_var_y1);
    }

    Real cov_xy = s::GetLane(s::SumOfLanes(d, s::Add(v_cov0, v_cov1)));
    Real var_x = s::GetLane(s::SumOfLanes(d, s::Add(v_var_x0, v_var_x1)));
    Real var_y = s::GetLane(s::SumOfLanes(d, s::Add(v_var_y0, v_var_y1)));

    for (; i < n; ++i) {
        Real dx = x[i] - mean_x;
        Real dy = y[i] - mean_y;
        cov_xy += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if (SCL_UNLIKELY(var_x < config::EPSILON)) {
        slope = Real(0);
        intercept = mean_y;
        r2 = Real(0);
        return;
    }

    slope = cov_xy / var_x;
    intercept = mean_y - slope * mean_x;

    // R-squared with SIMD
    if (SCL_UNLIKELY(var_y < config::EPSILON)) {
        r2 = Real(1);
    } else {
        auto v_slope = s::Set(d, slope);
        auto v_intercept = s::Set(d, intercept);
        auto v_ss_res0 = s::Zero(d), v_ss_res1 = s::Zero(d);

        i = 0;
        for (; i + 2 * lanes <= n; i += 2 * lanes) {
            auto v_x0 = s::Load(d, x + i);
            auto v_x1 = s::Load(d, x + i + lanes);
            auto v_y0 = s::Load(d, y + i);
            auto v_y1 = s::Load(d, y + i + lanes);

            auto v_pred0 = s::MulAdd(v_slope, v_x0, v_intercept);
            auto v_pred1 = s::MulAdd(v_slope, v_x1, v_intercept);
            auto v_res0 = s::Sub(v_y0, v_pred0);
            auto v_res1 = s::Sub(v_y1, v_pred1);

            v_ss_res0 = s::MulAdd(v_res0, v_res0, v_ss_res0);
            v_ss_res1 = s::MulAdd(v_res1, v_res1, v_ss_res1);
        }

        Real ss_res = s::GetLane(s::SumOfLanes(d, s::Add(v_ss_res0, v_ss_res1)));

        for (; i < n; ++i) {
            Real pred = slope * x[i] + intercept;
            Real res = y[i] - pred;
            ss_res += res * res;
        }

        r2 = Real(1) - ss_res / var_y;
        r2 = scl::algo::max2(Real(0), r2);
    }
}

// Fit gamma using steady-state model
SCL_FORCE_INLINE void fit_kinetics_steady_state(
    const Real* SCL_RESTRICT spliced,
    const Real* SCL_RESTRICT unspliced,
    Size n_cells,
    Real& gamma,
    Real& r2
) {
    Real slope = NAN, intercept = NAN;
    linear_regression(spliced, unspliced, n_cells, slope, intercept, r2);
    gamma = scl::algo::max2(slope, config::EPSILON);
}

// SIMD-optimized cosine similarity
SCL_HOT SCL_FORCE_INLINE Real cosine_similarity(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Index dim
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_dot0 = s::Zero(d), v_dot1 = s::Zero(d);
    auto v_norm_a0 = s::Zero(d), v_norm_a1 = s::Zero(d);
    auto v_norm_b0 = s::Zero(d), v_norm_b1 = s::Zero(d);

    Index i = 0;
    for (; i + static_cast<Index>(2 * lanes) <= dim; i += static_cast<Index>(2 * lanes)) {
        auto va0 = s::Load(d, a + i);
        auto va1 = s::Load(d, a + i + lanes);
        auto vb0 = s::Load(d, b + i);
        auto vb1 = s::Load(d, b + i + lanes);

        v_dot0 = s::MulAdd(va0, vb0, v_dot0);
        v_dot1 = s::MulAdd(va1, vb1, v_dot1);
        v_norm_a0 = s::MulAdd(va0, va0, v_norm_a0);
        v_norm_a1 = s::MulAdd(va1, va1, v_norm_a1);
        v_norm_b0 = s::MulAdd(vb0, vb0, v_norm_b0);
        v_norm_b1 = s::MulAdd(vb1, vb1, v_norm_b1);
    }

    Real dot = s::GetLane(s::SumOfLanes(d, s::Add(v_dot0, v_dot1)));
    Real norm_a = s::GetLane(s::SumOfLanes(d, s::Add(v_norm_a0, v_norm_a1)));
    Real norm_b = s::GetLane(s::SumOfLanes(d, s::Add(v_norm_b0, v_norm_b1)));

    // Scalar cleanup
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    Real denom = std::sqrt(norm_a * norm_b);
    return (SCL_LIKELY(denom > config::EPSILON)) ? dot / denom : Real(0);
}

// SIMD-optimized softmax normalization
SCL_HOT void softmax(Real* SCL_RESTRICT values, Size n) {
    if (SCL_UNLIKELY(n == 0)) return;

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    // Find max with SIMD
    Real max_val = values[0];
    auto v_max = s::Set(d, max_val);
    Size i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, values + i);
        v_max = s::Max(v_max, v);
    }
    max_val = s::GetLane(s::MaxOfLanes(d, v_max));

    for (; i < n; ++i) {
        max_val = scl::algo::max2(max_val, values[i]);
    }

    // Compute exp and sum with SIMD
    auto v_neg_max = s::Set(d, -max_val);
    auto v_sum = s::Zero(d);

    i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, values + i);
        auto v_exp = s::Exp(d, s::Add(v, v_neg_max));
        s::Store(v_exp, d, values + i);
        v_sum = s::Add(v_sum, v_exp);
    }

    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < n; ++i) {
        values[i] = std::exp(values[i] - max_val);
        sum += values[i];
    }

    // Normalize with SIMD
    if (SCL_LIKELY(sum > config::EPSILON)) {
        Real inv_sum = Real(1) / sum;
        auto v_inv_sum = s::Set(d, inv_sum);

        i = 0;
        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, values + i);
            s::Store(s::Mul(v, v_inv_sum), d, values + i);
        }

        for (; i < n; ++i) {
            values[i] *= inv_sum;
        }
    }
}

// SIMD-optimized vector difference: out = a - b
SCL_HOT SCL_FORCE_INLINE void vec_diff(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Real* SCL_RESTRICT out,
    Index dim
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    Index i = 0;
    for (; i + static_cast<Index>(lanes) <= dim; i += static_cast<Index>(lanes)) {
        auto va = s::Load(d, a + i);
        auto vb = s::Load(d, b + i);
        s::Store(s::Sub(va, vb), d, out + i);
    }

    for (; i < dim; ++i) {
        out[i] = a[i] - b[i];
    }
}

// SIMD-optimized weighted accumulation: out += weight * vec
SCL_HOT SCL_FORCE_INLINE void vec_accumulate(
    Real* SCL_RESTRICT out,
    const Real* SCL_RESTRICT vec,
    Real weight,
    Index dim
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_w = s::Set(d, weight);

    Index i = 0;
    for (; i + static_cast<Index>(lanes) <= dim; i += static_cast<Index>(lanes)) {
        auto v_out = s::Load(d, out + i);
        auto v_vec = s::Load(d, vec + i);
        s::Store(s::MulAdd(v_w, v_vec, v_out), d, out + i);
    }

    for (; i < dim; ++i) {
        out[i] += weight * vec[i];
    }
}

} // namespace detail

// =============================================================================
// Per-Gene Kinetics Fitting - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void fit_gene_kinetics(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Index n_cells,
    Index n_genes,
    Array<Real> gamma,
    Array<Real> r2,
    VelocityModel model = VelocityModel::SteadyState
) {
    SCL_CHECK_DIM(gamma.len >= static_cast<Size>(n_genes), "Velocity: gamma buffer too small");
    SCL_CHECK_DIM(r2.len >= static_cast<Size>(n_genes), "Velocity: r2 buffer too small");

    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_genes_sz = static_cast<Size>(n_genes);
    const bool use_parallel = (n_genes_sz >= config::PARALLEL_THRESHOLD);
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread workspace
    scl::threading::DualWorkspacePool<Real> workspace;
    if (use_parallel) {
        workspace.init(n_threads, n_cells_sz);
    }

    auto process_gene = [&](Index g, Real* s_vals, Real* u_vals) {
        std::memset(s_vals, 0, n_cells_sz * sizeof(Real));
        std::memset(u_vals, 0, n_cells_sz * sizeof(Real));

        // Extract gene values for all cells
        if (IsCSR) {
            for (Index c = 0; c < n_cells; ++c) {
                auto s_indices = spliced.row_indices_unsafe(c);
                auto s_values = spliced.row_values_unsafe(c);
                Index s_len = spliced.row_length_unsafe(c);

                // Binary search for gene index
                const Index* s_found = scl::algo::lower_bound(
                    s_indices.ptr, s_indices.ptr + s_len, g);

                if (s_found != s_indices.ptr + s_len && *s_found == g) {
                    auto k = static_cast<Index>(s_found - s_indices.ptr);
                    s_vals[c] = static_cast<Real>(s_values[k]);
                }

                auto u_indices = unspliced.row_indices_unsafe(c);
                auto u_values = unspliced.row_values_unsafe(c);
                Index u_len = unspliced.row_length_unsafe(c);

                const Index* u_found = scl::algo::lower_bound(
                    u_indices.ptr, u_indices.ptr + u_len, g);

                if (u_found != u_indices.ptr + u_len && *u_found == g) {
                    auto k = static_cast<Index>(u_found - u_indices.ptr);
                    u_vals[c] = static_cast<Real>(u_values[k]);
                }
            }
        } else {
            auto s_indices = spliced.col_indices_unsafe(g);
            auto s_values = spliced.col_values_unsafe(g);
            Index s_len = spliced.col_length_unsafe(g);

            for (Index k = 0; k < s_len; ++k) {
                Index c = s_indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    s_vals[c] = static_cast<Real>(s_values[k]);
                }
            }

            auto u_indices = unspliced.col_indices_unsafe(g);
            auto u_values = unspliced.col_values_unsafe(g);
            Index u_len = unspliced.col_length_unsafe(g);

            for (Index k = 0; k < u_len; ++k) {
                Index c = u_indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    u_vals[c] = static_cast<Real>(u_values[k]);
                }
            }
        }

        // Fit kinetics
        switch (model) {
            case VelocityModel::SteadyState:
            default:
                detail::fit_kinetics_steady_state(s_vals, u_vals, n_cells_sz, gamma[g], r2[g]);
                break;
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_genes_sz, [&](Size g, Size thread_rank) {
            Real* s_vals = workspace.get1(thread_rank);
            Real* u_vals = workspace.get2(thread_rank);
            process_gene(static_cast<Index>(g), s_vals, u_vals);
        });
    } else {
        auto s_vals_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);

        Real* s_vals = s_vals_ptr.release();
        auto u_vals_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);

        Real* u_vals = u_vals_ptr.release();

        for (Index g = 0; g < n_genes; ++g) {
            process_gene(g, s_vals, u_vals);
        }

        scl::memory::aligned_free(u_vals, SCL_ALIGNMENT);
        scl::memory::aligned_free(s_vals, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Compute Velocity (dS/dt) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void compute_velocity(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Array<const Real> gamma,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT velocity_out
) {
    SCL_CHECK_DIM(gamma.len >= static_cast<Size>(n_genes), "Velocity: gamma buffer too small");

    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_genes);
    std::memset(velocity_out, 0, total * sizeof(Real));

    const Size n_cells_sz = static_cast<Size>(n_cells);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);

    // velocity[c,g] = unspliced[c,g] - gamma[g] * spliced[c,g]
    if (IsCSR) {
        auto process_cell = [&](Index c) {
            Real* vel_row = velocity_out + static_cast<Size>(c) * n_genes;

            // Process spliced
            auto s_indices = spliced.row_indices_unsafe(c);
            auto s_values = spliced.row_values_unsafe(c);
            Index s_len = spliced.row_length_unsafe(c);

            for (Index k = 0; k < s_len; ++k) {
                Index g = s_indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    vel_row[g] -= gamma[g] * static_cast<Real>(s_values[k]);
                }
            }

            // Process unspliced
            auto u_indices = unspliced.row_indices_unsafe(c);
            auto u_values = unspliced.row_values_unsafe(c);
            Index u_len = unspliced.row_length_unsafe(c);

            for (Index k = 0; k < u_len; ++k) {
                Index g = u_indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    vel_row[g] += static_cast<Real>(u_values[k]);
                }
            }
        };

        if (use_parallel) {
            scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size c) {
                process_cell(static_cast<Index>(c));
            });
        } else {
            for (Index c = 0; c < n_cells; ++c) {
                process_cell(c);
            }
        }
    } else {
        // CSC format - process by gene (better for parallel)
        auto process_gene = [&](Index g) {
            Real gamma_g = gamma[g];

            auto s_indices = spliced.col_indices_unsafe(g);
            auto s_values = spliced.col_values_unsafe(g);
            Index s_len = spliced.col_length_unsafe(g);

            for (Index k = 0; k < s_len; ++k) {
                Index c = s_indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    velocity_out[static_cast<Size>(c) * n_genes + g] -=
                        gamma_g * static_cast<Real>(s_values[k]);
                }
            }

            auto u_indices = unspliced.col_indices_unsafe(g);
            auto u_values = unspliced.col_values_unsafe(g);
            Index u_len = unspliced.col_length_unsafe(g);

            for (Index k = 0; k < u_len; ++k) {
                Index c = u_indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    velocity_out[static_cast<Size>(c) * n_genes + g] +=
                        static_cast<Real>(u_values[k]);
                }
            }
        };

        const Size n_genes_sz = static_cast<Size>(n_genes);
        if (n_genes_sz >= config::PARALLEL_THRESHOLD) {
            scl::threading::parallel_for(Size(0), n_genes_sz, [&](Size g) {
                process_gene(static_cast<Index>(g));
            });
        } else {
            for (Index g = 0; g < n_genes; ++g) {
                process_gene(g);
            }
        }
    }
}

// =============================================================================
// Splice Ratio - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void splice_ratio(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT ratio_out
) {
    const Size total = static_cast<Size>(n_cells) * static_cast<Size>(n_genes);
    std::memset(ratio_out, 0, total * sizeof(Real));

    auto s_dense_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);


    Real* s_dense = s_dense_ptr.release();
    std::memset(s_dense, 0, total * sizeof(Real));

    // Fill spliced values
    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            auto indices = spliced.row_indices_unsafe(c);
            auto values = spliced.row_values_unsafe(c);
            Index len = spliced.row_length_unsafe(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    s_dense[static_cast<Size>(c) * n_genes + g] = static_cast<Real>(values[k]);
                }
            }
        }
    } else {
        for (Index g = 0; g < n_genes; ++g) {
            auto indices = spliced.col_indices_unsafe(g);
            auto values = spliced.col_values_unsafe(g);
            Index len = spliced.col_length_unsafe(g);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    s_dense[static_cast<Size>(c) * n_genes + g] = static_cast<Real>(values[k]);
                }
            }
        }
    }

    // Compute ratio with SIMD where possible
    const Real eps = config::EPSILON;

    if (IsCSR) {
        for (Index c = 0; c < n_cells; ++c) {
            auto indices = unspliced.row_indices_unsafe(c);
            auto values = unspliced.row_values_unsafe(c);
            Index len = unspliced.row_length_unsafe(c);

            for (Index k = 0; k < len; ++k) {
                Index g = indices[k];
                if (SCL_LIKELY(g < n_genes)) {
                    Size idx = static_cast<Size>(c) * n_genes + g;
                    Real u = static_cast<Real>(values[k]);
                    Real s = s_dense[idx];
                    ratio_out[idx] = u / (s + eps);
                }
            }
        }
    } else {
        for (Index g = 0; g < n_genes; ++g) {
            auto indices = unspliced.col_indices_unsafe(g);
            auto values = unspliced.col_values_unsafe(g);
            Index len = unspliced.col_length_unsafe(g);

            for (Index k = 0; k < len; ++k) {
                Index c = indices[k];
                if (SCL_LIKELY(c < n_cells)) {
                    Size idx = static_cast<Size>(c) * n_genes + g;
                    Real u = static_cast<Real>(values[k]);
                    Real s = s_dense[idx];
                    ratio_out[idx] = u / (s + eps);
                }
            }
        }
    }

    scl::memory::aligned_free(s_dense, SCL_ALIGNMENT);
}

// =============================================================================
// Velocity Graph (Transition Probabilities) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void velocity_graph(
    const Real* SCL_RESTRICT velocity,
    const Real* SCL_RESTRICT expression,
    const Sparse<T, IsCSR>& knn,
    Index n_cells,
    Index n_genes,
    Real* SCL_RESTRICT transition_probs,
    Index k_neighbors
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_genes_sz = static_cast<Size>(n_genes);
    const Size k_sz = static_cast<Size>(k_neighbors);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread delta buffer
    scl::threading::WorkspacePool<Real> delta_pool;
    if (use_parallel) {
        delta_pool.init(n_threads, n_genes_sz);
    }

    auto process_cell = [&](Index i, Real* delta) {
        const Real* vel_i = velocity + static_cast<Size>(i) * n_genes;
        const Real* expr_i = expression + static_cast<Size>(i) * n_genes;

        auto neighbor_indices = knn.primary_indices_unsafe(i);
        Index n_neighbors = knn.primary_length_unsafe(i);
        Index actual_k = scl::algo::min2(n_neighbors, k_neighbors);

        Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;
        std::memset(probs, 0, k_sz * sizeof(Real));

        for (Index k = 0; k < actual_k; ++k) {
            Index j = neighbor_indices[k];
            const Real* expr_j = expression + static_cast<Size>(j) * n_genes;

            // Compute delta = expr_j - expr_i (SIMD)
            detail::vec_diff(expr_j, expr_i, delta, n_genes);

            // Cosine similarity (SIMD)
            Real cos_sim = detail::cosine_similarity(vel_i, delta, n_genes);
            probs[k] = scl::algo::max2(cos_sim, Real(0));
        }

        // Softmax (SIMD)
        detail::softmax(probs, static_cast<Size>(actual_k));
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size i, Size thread_rank) {
            Real* delta = delta_pool.get(thread_rank);
            process_cell(static_cast<Index>(i), delta);
        });
    } else {
        auto delta_ptr = scl::memory::aligned_alloc<Real>(n_genes_sz, SCL_ALIGNMENT);

        Real* delta = delta_ptr.release();
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i, delta);
        }
        scl::memory::aligned_free(delta, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Velocity Graph with Cosine Kernel - Optimized
// =============================================================================

inline void velocity_graph_cosine(
    const Real* SCL_RESTRICT velocity,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Real* SCL_RESTRICT transition_probs
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);

    auto process_cell = [&](Index i) {
        const Real* vel_i = velocity + static_cast<Size>(i) * n_genes;
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
        Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;

        Real sum = Real(0);
        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_UNLIKELY(j < 0 || j >= n_cells)) {
                probs[k] = Real(0);
                continue;
            }

            const Real* vel_j = velocity + static_cast<Size>(j) * n_genes;
            Real cos_sim = detail::cosine_similarity(vel_i, vel_j, n_genes);
            probs[k] = (cos_sim + Real(1)) * Real(0.5);
            sum += probs[k];
        }

        // Normalize
        if (SCL_LIKELY(sum > config::EPSILON)) {
            Real inv_sum = Real(1) / sum;
            for (Index k = 0; k < k_neighbors; ++k) {
                probs[k] *= inv_sum;
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size i) {
            process_cell(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i);
        }
    }
}

// =============================================================================
// Velocity Embedding Projection - Optimized
// =============================================================================

inline void velocity_embedding(
    const Real* SCL_RESTRICT velocity,
    const Real* SCL_RESTRICT embedding,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index n_genes,
    Index n_dims,
    Index k_neighbors,
    Real* SCL_RESTRICT velocity_embedded
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_dims_sz = static_cast<Size>(n_dims);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Real> delta_pool;
    if (use_parallel) {
        delta_pool.init(n_threads, n_dims_sz);
    }

    auto process_cell = [&](Index i, Real* delta_emb) {
        const Real* vel_i = velocity + static_cast<Size>(i) * n_genes;
        const Real* emb_i = embedding + static_cast<Size>(i) * n_dims;
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
        Real* vel_emb_i = velocity_embedded + static_cast<Size>(i) * n_dims;

        std::memset(vel_emb_i, 0, n_dims_sz * sizeof(Real));

        Real weight_sum = Real(0);
        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_UNLIKELY(j < 0 || j >= n_cells || j == i)) continue;

            const Real* emb_j = embedding + static_cast<Size>(j) * n_dims;

            // Compute embedding direction
            detail::vec_diff(emb_j, emb_i, delta_emb, n_dims);

            // Weight from velocity (use norm as proxy for direction confidence)
            Real cos_weight = detail::cosine_similarity(vel_i, vel_i, n_genes);
            cos_weight = scl::algo::max2(cos_weight, Real(0));

            // Accumulate weighted direction
            detail::vec_accumulate(vel_emb_i, delta_emb, cos_weight, n_dims);
            weight_sum += cos_weight;
        }

        // Normalize
        if (SCL_LIKELY(weight_sum > config::EPSILON)) {
            Real inv_weight = Real(1) / weight_sum;
            for (Index d = 0; d < n_dims; ++d) {
                vel_emb_i[d] *= inv_weight;
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size i, Size thread_rank) {
            Real* delta_emb = delta_pool.get(thread_rank);
            process_cell(static_cast<Index>(i), delta_emb);
        });
    } else {
        auto delta_emb_ptr = scl::memory::aligned_alloc<Real>(n_dims_sz, SCL_ALIGNMENT);

        Real* delta_emb = delta_emb_ptr.release();
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i, delta_emb);
        }
        scl::memory::aligned_free(delta_emb, SCL_ALIGNMENT);
    }
}

// =============================================================================
// Grid-Based Velocity Visualization - Optimized
// =============================================================================

inline void velocity_grid(
    const Real* SCL_RESTRICT embedding,
    const Real* SCL_RESTRICT velocity_embedded,
    Index n_cells,
    Index n_dims,
    Index grid_size,
    Real* SCL_RESTRICT grid_coords,
    Real* SCL_RESTRICT grid_velocity
) {
    SCL_CHECK_ARG(n_dims == 2, "Velocity: grid only supports 2D embedding");

    const Index n_grid_points = grid_size * grid_size;
    const Size n_grid_sz = static_cast<Size>(n_grid_points);

    // Find embedding bounds
    Real min_x = embedding[0], max_x = embedding[0];
    Real min_y = embedding[1], max_y = embedding[1];

    // Strided access for x,y coordinates
    for (Index i = 0; i < n_cells; ++i) {
        Real x = embedding[static_cast<Size>(i) * 2];
        Real y = embedding[static_cast<Size>(i) * 2 + 1];
        min_x = scl::algo::min2(min_x, x);
        max_x = scl::algo::max2(max_x, x);
        min_y = scl::algo::min2(min_y, y);
        max_y = scl::algo::max2(max_y, y);
    }

    // Add margin
    Real margin_x = (max_x - min_x) * Real(0.05);
    Real margin_y = (max_y - min_y) * Real(0.05);
    min_x -= margin_x; max_x += margin_x;
    min_y -= margin_y; max_y += margin_y;

    Real inv_step_x = static_cast<Real>(grid_size - 1) / (max_x - min_x);
    Real inv_step_y = static_cast<Real>(grid_size - 1) / (max_y - min_y);
    Real step_x = Real(1) / inv_step_x;
    Real step_y = Real(1) / inv_step_y;

    // Initialize
    std::memset(grid_velocity, 0, n_grid_sz * 2 * sizeof(Real));

    auto counts_ptr = scl::memory::aligned_alloc<Index>(n_grid_sz, SCL_ALIGNMENT);


    Index* counts = counts_ptr.release();
    std::memset(counts, 0, n_grid_sz * sizeof(Index));

    // Generate grid coordinates
    for (Index gy = 0; gy < grid_size; ++gy) {
        for (Index gx = 0; gx < grid_size; ++gx) {
            Index gi = gy * grid_size + gx;
            grid_coords[static_cast<Size>(gi) * 2] = min_x + static_cast<Real>(gx) * step_x;
            grid_coords[static_cast<Size>(gi) * 2 + static_cast<Size>(1)] = min_y + static_cast<Real>(gy) * step_y;
        }
    }

    // Assign cells to grid and accumulate velocities
    for (Index i = 0; i < n_cells; ++i) {
        Real x = embedding[static_cast<Size>(i) * 2];
        Real y = embedding[static_cast<Size>(i) * 2 + 1];

        auto gx = static_cast<Index>((x - min_x) * inv_step_x);
        auto gy = static_cast<Index>((y - min_y) * inv_step_y);
        gx = scl::algo::clamp(gx, Index(0), grid_size - 1);
        gy = scl::algo::clamp(gy, Index(0), grid_size - 1);

        Index gi = gy * grid_size + gx;
        grid_velocity[static_cast<Size>(gi) * 2] += velocity_embedded[static_cast<Size>(i) * 2];
        grid_velocity[static_cast<Size>(gi) * 2 + 1] += velocity_embedded[static_cast<Size>(i) * 2 + 1];
        ++counts[gi];
    }

    // Average velocities
    for (Index gi = 0; gi < n_grid_points; ++gi) {
        if (counts[gi] > 0) {
            Real inv_count = Real(1) / static_cast<Real>(counts[gi]);
            grid_velocity[static_cast<Size>(gi) * 2] *= inv_count;
            grid_velocity[static_cast<Size>(gi) * 2 + 1] *= inv_count;
        }
    }

    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
}

// =============================================================================
// Velocity Confidence - Optimized
// =============================================================================

inline void velocity_confidence(
    const Real* SCL_RESTRICT velocity,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index n_genes,
    Index k_neighbors,
    Real* SCL_RESTRICT confidence
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);

    auto process_cell = [&](Index i) {
        const Real* vel_i = velocity + static_cast<Size>(i) * n_genes;
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;

        Real sum_cos = Real(0);
        Index valid_count = 0;

        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_UNLIKELY(j < 0 || j >= n_cells)) continue;

            const Real* vel_j = velocity + static_cast<Size>(j) * n_genes;
            sum_cos += detail::cosine_similarity(vel_i, vel_j, n_genes);
            ++valid_count;
        }

        confidence[i] = (valid_count > 0) ?
            sum_cos / static_cast<Real>(valid_count) : Real(0);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size i) {
            process_cell(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i);
        }
    }
}

// =============================================================================
// Latent Time Inference - Optimized
// =============================================================================

inline void latent_time(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index k_neighbors,
    Index root_cell,
    Real* SCL_RESTRICT latent_time_out
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);

    // Initialize with SIMD fill
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    auto v_inf = s::Set(d, config::INF_VALUE);

    Size i = 0;
    for (; i + lanes <= n_cells_sz; i += lanes) {
        s::Store(v_inf, d, latent_time_out + i);
    }

    for (; i < n_cells_sz; ++i) {
        latent_time_out[i] = config::INF_VALUE;
    }

    latent_time_out[root_cell] = Real(0);

    auto prev_time_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);


    Real* prev_time = prev_time_ptr.release();
    std::memcpy(prev_time, latent_time_out, n_cells_sz * sizeof(Real));

    const Real half_inf = config::INF_VALUE * Real(0.5);

    for (Index iter = 0; iter < config::DEFAULT_N_ITERATIONS; ++iter) {
        bool changed = false;

        // Outer loop index 'i' is scoped and does not shadow earlier declarations.
        for (Index cell_i = 0; cell_i < n_cells; ++cell_i) {
            if (SCL_UNLIKELY(prev_time[cell_i] >= half_inf)) continue;

            const Index* neighbors = knn_indices + static_cast<Size>(cell_i) * k_neighbors;
            const Real* probs = transition_probs + static_cast<Size>(cell_i) * k_neighbors;

            for (Index k = 0; k < k_neighbors; ++k) {
                Index j = neighbors[k];
                if (SCL_UNLIKELY(j < 0 || j >= n_cells)) continue;

                Real weight = (probs[k] > config::EPSILON) ?
                    -std::log(probs[k]) : config::INF_VALUE;

                Real new_time = prev_time[cell_i] + weight;

                if (new_time < latent_time_out[j]) {
                    latent_time_out[j] = new_time;
                    changed = true;
                }
            }
        }

        std::memcpy(prev_time, latent_time_out, n_cells_sz * sizeof(Real));
        if (!changed) break;
    }

    // Normalize to [0, 1]
    Real max_time = Real(0);
    for (Index cell_i = 0; cell_i < n_cells; ++cell_i) {
        if (latent_time_out[cell_i] < half_inf) {
            max_time = scl::algo::max2(max_time, latent_time_out[cell_i]);
        }
    }

    // Normalize latent times to [0, 1]
    if (SCL_LIKELY(max_time > config::EPSILON)) {
        const Real inv_max_time = Real(1) / max_time;
        for (Index cell_i = 0; cell_i < n_cells; ++cell_i) {
            if (latent_time_out[cell_i] < half_inf) {
                latent_time_out[cell_i] *= inv_max_time;
            } else {
                latent_time_out[cell_i] = Real(1);
            }
        }
    }

    scl::memory::aligned_free(prev_time, SCL_ALIGNMENT);
}

// =============================================================================
// Cell Fate Probability - Optimized
// =============================================================================

inline void cell_fate_probability(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index k_neighbors,
    Array<const Index> terminal_cells,
    Real* SCL_RESTRICT fate_probs
) {
    const auto n_terminal = static_cast<Index>(terminal_cells.len);
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_terminal_sz = static_cast<Size>(n_terminal);
    const Size total = n_cells_sz * n_terminal_sz;

    std::memset(fate_probs, 0, total * sizeof(Real));

    // Mark terminal cells
    auto is_terminal_ptr = scl::memory::aligned_alloc<bool>(n_cells_sz, SCL_ALIGNMENT);

    bool* is_terminal = is_terminal_ptr.release();
    std::memset(is_terminal, 0, n_cells_sz * sizeof(bool));

    for (Size t = 0; t < terminal_cells.len; ++t) {
        auto cell = terminal_cells[static_cast<Index>(t)];
        if (SCL_LIKELY(cell >= 0 && cell < n_cells)) {
            is_terminal[cell] = true;
            fate_probs[static_cast<Size>(cell) * n_terminal_sz + t] = Real(1);
        }
    }

    auto prev_probs_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);


    Real* prev_probs = prev_probs_ptr.release();

    for (Index iter = 0; iter < config::DEFAULT_N_ITERATIONS; ++iter) {
        std::memcpy(prev_probs, fate_probs, total * sizeof(Real));

        // Update non-terminal cells
        for (Index i = 0; i < n_cells; ++i) {
            if (is_terminal[i]) continue;

            const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
            const Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;

            Real* fate_i = fate_probs + static_cast<Size>(i) * n_terminal_sz;
            std::memset(fate_i, 0, n_terminal_sz * sizeof(Real));

            for (Index k = 0; k < k_neighbors; ++k) {
                Index j = neighbors[k];
                if (SCL_UNLIKELY(j < 0 || j >= n_cells)) continue;

                const Real* fate_j = prev_probs + static_cast<Size>(j) * n_terminal_sz;
                Real p_k = probs[k];

                // SIMD accumulation
                detail::vec_accumulate(fate_i, fate_j, p_k, n_terminal);
            }
        }
    }

    scl::memory::aligned_free(prev_probs, SCL_ALIGNMENT);
    scl::memory::aligned_free(is_terminal, SCL_ALIGNMENT);
}

// =============================================================================
// Random Walk Future State Prediction - Optimized
// =============================================================================

inline void predict_future_state(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index k_neighbors,
    Index n_steps,
    Real* SCL_RESTRICT future_probs
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size total = n_cells_sz * n_cells_sz;

    std::memset(future_probs, 0, total * sizeof(Real));

    // Initialize: diagonal = 1
    for (Index i = 0; i < n_cells; ++i) {
        future_probs[static_cast<Size>(i) * n_cells + i] = Real(1);
    }

    auto prev_probs_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);


    Real* prev_probs = prev_probs_ptr.release();

    for (Index step = 0; step < n_steps; ++step) {
        std::memcpy(prev_probs, future_probs, total * sizeof(Real));
        std::memset(future_probs, 0, total * sizeof(Real));

        // For each starting cell
        for (Index start = 0; start < n_cells; ++start) {
            Real* fut_start = future_probs + static_cast<Size>(start) * n_cells;
            const Real* prev_start = prev_probs + static_cast<Size>(start) * n_cells;

            // For each possible current position
            for (Index cur = 0; cur < n_cells; ++cur) {
                Real p_at_cur = prev_start[cur];
                if (SCL_UNLIKELY(p_at_cur < config::EPSILON)) continue;

                const Index* neighbors = knn_indices + static_cast<Size>(cur) * k_neighbors;
                const Real* trans = transition_probs + static_cast<Size>(cur) * k_neighbors;

                for (Index k = 0; k < k_neighbors; ++k) {
                    Index next = neighbors[k];
                    if (SCL_LIKELY(next >= 0 && next < n_cells)) {
                        fut_start[next] += p_at_cur * trans[k];
                    }
                }
            }
        }
    }

    scl::memory::aligned_free(prev_probs, SCL_ALIGNMENT);
}

// =============================================================================
// Velocity Gene Selection - Optimized
// =============================================================================

inline void select_velocity_genes(
    const Real* SCL_RESTRICT velocity,
    const Real* SCL_RESTRICT r2,
    Index n_cells,
    Index n_genes,
    Real min_r2,
    Real min_velocity_var,
    Array<Index> selected_genes,
    Index& n_selected
) {
    n_selected = 0;
    const Real inv_n = Real(1) / static_cast<Real>(n_cells);

    for (Index g = 0; g < n_genes; ++g) {
        if (r2[g] < min_r2) continue;

        // Compute variance for velocity of this gene
        Real sum = Real(0), sum_sq = Real(0);

        for (Index c = 0; c < n_cells; ++c) {
            Real v = velocity[static_cast<Size>(c) * n_genes + g];
            sum += v;
            sum_sq += v * v;
        }

        Real mean = sum * inv_n;
        Real var = sum_sq * inv_n - mean * mean;

        if (var < min_velocity_var) continue;

        if (n_selected < static_cast<Index>(selected_genes.len)) {
            selected_genes[n_selected++] = g;
        }
    }
}

// =============================================================================
// Velocity Pseudotime - Optimized
// =============================================================================

inline void velocity_pseudotime(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    const Real* SCL_RESTRICT velocity_magnitude,
    Index n_cells,
    Index k_neighbors,
    Index root_cell,
    Real* SCL_RESTRICT pseudotime
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);

    // First compute basic latent time
    latent_time(transition_probs, knn_indices, n_cells, k_neighbors, root_cell, pseudotime);

    auto refined_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);


    Real* refined = refined_ptr.release();

    for (Index iter = 0; iter < 10; ++iter) {
        for (Index i = 0; i < n_cells; ++i) {
            const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
            const Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;

            Real weighted_time = Real(0);
            Real weight_sum = Real(0);

            for (Index k = 0; k < k_neighbors; ++k) {
                Index j = neighbors[k];
                if (SCL_UNLIKELY(j < 0 || j >= n_cells)) continue;

                Real w = probs[k] * (Real(1) + velocity_magnitude[i]);
                weighted_time += w * pseudotime[j];
                weight_sum += w;
            }

            refined[i] = (SCL_LIKELY(weight_sum > config::EPSILON)) ?
                Real(0.5) * pseudotime[i] + Real(0.5) * weighted_time / weight_sum :
                pseudotime[i];
        }

        std::memcpy(pseudotime, refined, n_cells_sz * sizeof(Real));
    }

    // Renormalize
    Real min_t = pseudotime[0], max_t = pseudotime[0];
    for (Index i = 1; i < n_cells; ++i) {
        min_t = scl::algo::min2(min_t, pseudotime[i]);
        max_t = scl::algo::max2(max_t, pseudotime[i]);
    }

    Real range = max_t - min_t;
    if (SCL_LIKELY(range > config::EPSILON)) {
        Real inv_range = Real(1) / range;
        for (Index i = 0; i < n_cells; ++i) {
            pseudotime[i] = (pseudotime[i] - min_t) * inv_range;
        }
    }

    scl::memory::aligned_free(refined, SCL_ALIGNMENT);
}

// =============================================================================
// Velocity Stream Computation - Optimized
// =============================================================================

inline void velocity_stream(
    const Real* SCL_RESTRICT embedding,
    const Real* SCL_RESTRICT velocity_embedded,
    Index n_cells,
    Index start_cell,
    Index max_steps,
    Real step_size,
    Real* SCL_RESTRICT stream_points,
    Index& actual_steps
) {
    actual_steps = 0;

    Real x = embedding[static_cast<Size>(start_cell) * 2];
    Real y = embedding[static_cast<Size>(start_cell) * 2 + 1];

    for (Index step = 0; step < max_steps; ++step) {
        stream_points[static_cast<Size>(step) * 2] = x;
        stream_points[static_cast<Size>(step) * 2 + 1] = y;
        actual_steps = step + 1;

        // Find nearest cell
        Real min_dist = config::INF_VALUE;
        Index nearest = -1;

        for (Index i = 0; i < n_cells; ++i) {
            Real dx = embedding[static_cast<Size>(i) * 2] - x;
            Real dy = embedding[static_cast<Size>(i) * 2 + 1] - y;
            Real dist = dx * dx + dy * dy;

            if (dist < min_dist) {
                min_dist = dist;
                nearest = i;
            }
        }

        if (SCL_UNLIKELY(nearest < 0)) break;

        Real vx = velocity_embedded[static_cast<Size>(nearest) * 2];
        Real vy = velocity_embedded[static_cast<Size>(nearest) * 2 + 1];
        Real norm = std::sqrt(vx * vx + vy * vy);

        if (SCL_UNLIKELY(norm < config::EPSILON)) break;

        Real inv_norm = step_size / norm;
        x += vx * inv_norm;
        y += vy * inv_norm;
    }
}

// =============================================================================
// Velocity Divergence - Optimized
// =============================================================================

inline void velocity_divergence(
    const Real* SCL_RESTRICT velocity_embedded,
    const Real* SCL_RESTRICT embedding,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index n_dims,
    Index k_neighbors,
    Real* SCL_RESTRICT divergence
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const bool use_parallel = (n_cells_sz >= config::PARALLEL_THRESHOLD);

    auto process_cell = [&](Index i) {
        const Real* vel_i = velocity_embedded + static_cast<Size>(i) * n_dims;
        const Real* emb_i = embedding + static_cast<Size>(i) * n_dims;
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;

        Real div = Real(0);
        Index valid_count = 0;

        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_UNLIKELY(j < 0 || j >= n_cells)) continue;

            const Real* vel_j = velocity_embedded + static_cast<Size>(j) * n_dims;
            const Real* emb_j = embedding + static_cast<Size>(j) * n_dims;

            Real radial_dist = Real(0);
            Real radial_vel = Real(0);

            for (Index d = 0; d < n_dims; ++d) {
                Real r = emb_j[d] - emb_i[d];
                Real v = vel_j[d] - vel_i[d];
                radial_dist += r * r;
                radial_vel += r * v;
            }

            radial_dist = std::sqrt(radial_dist);

            if (SCL_LIKELY(radial_dist > config::EPSILON)) {
                div += radial_vel / radial_dist;
                ++valid_count;
            }
        }

        divergence[i] = (valid_count > 0) ?
            div / static_cast<Real>(valid_count) : Real(0);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), n_cells_sz, [&](Size i) {
            process_cell(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i);
        }
    }
}

// =============================================================================
// Root Cell Selection - Optimized
// =============================================================================

inline Index select_root_by_velocity(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    Index n_cells,
    Index k_neighbors
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    auto incoming_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);

    Real* incoming = incoming_ptr.release();
    std::memset(incoming, 0, n_cells_sz * sizeof(Real));

    for (Index i = 0; i < n_cells; ++i) {
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
        const Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;

        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_LIKELY(j >= 0 && j < n_cells)) {
                incoming[j] += probs[k];
            }
        }
    }

    // Find minimum
    Index root = 0;
    Real min_incoming = incoming[0];
    for (Index i = 1; i < n_cells; ++i) {
        if (incoming[i] < min_incoming) {
            min_incoming = incoming[i];
            root = i;
        }
    }

    scl::memory::aligned_free(incoming, SCL_ALIGNMENT);
    return root;
}

// =============================================================================
// Terminal State Detection - Optimized
// =============================================================================

inline Index detect_terminal_states(
    const Real* SCL_RESTRICT transition_probs,
    const Index* SCL_RESTRICT knn_indices,
    const Real* SCL_RESTRICT velocity_magnitude,
    Index n_cells,
    Index k_neighbors,
    Real magnitude_threshold,
    Array<Index> terminal_cells
) {
    const Size n_cells_sz = static_cast<Size>(n_cells);
    Index n_terminal = 0;

    auto outgoing_ptr = scl::memory::aligned_alloc<Real>(n_cells_sz, SCL_ALIGNMENT);


    Real* outgoing = outgoing_ptr.release();

    for (Index i = 0; i < n_cells; ++i) {
        const Index* neighbors = knn_indices + static_cast<Size>(i) * k_neighbors;
        const Real* probs = transition_probs + static_cast<Size>(i) * k_neighbors;

        Real total_out = Real(0);
        for (Index k = 0; k < k_neighbors; ++k) {
            Index j = neighbors[k];
            if (SCL_LIKELY(j >= 0 && j < n_cells && j != i)) {
                total_out += probs[k];
            }
        }
        outgoing[i] = total_out;
    }

    // Select terminal cells
    for (Index i = 0; i < n_cells; ++i) {
        bool low_velocity = velocity_magnitude[i] < magnitude_threshold;
        bool low_outgoing = outgoing[i] < Real(0.5);

        if (low_velocity && low_outgoing) {
            if (n_terminal < static_cast<Index>(terminal_cells.len)) {
                terminal_cells[n_terminal++] = i;
            }
        }
    }

    scl::memory::aligned_free(outgoing, SCL_ALIGNMENT);
    return n_terminal;
}

} // namespace scl::kernel::velocity
