#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/mmd.hpp
// BRIEF: Maximum Mean Discrepancy with RBF kernel
// =============================================================================

namespace scl::kernel::mmd {

namespace detail {

template <typename T>
SCL_FORCE_INLINE T unary_exp_sum_ultra(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_gamma = s::Set(d, gamma);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    auto v_sum4 = s::Zero(d);
    auto v_sum5 = s::Zero(d);
    auto v_sum6 = s::Zero(d);
    auto v_sum7 = s::Zero(d);

    size_t k = 0;

    for (; k + 8 * lanes <= nnz; k += 8 * lanes) {
        SCL_PREFETCH_READ(vals + k + 16 * lanes, 0);

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        auto sq0 = s::Mul(v0, v0);
        auto sq1 = s::Mul(v1, v1);
        auto sq2 = s::Mul(v2, v2);
        auto sq3 = s::Mul(v3, v3);
        auto sq4 = s::Mul(v4, v4);
        auto sq5 = s::Mul(v5, v5);
        auto sq6 = s::Mul(v6, v6);
        auto sq7 = s::Mul(v7, v7);

        auto exp0 = s::Exp(d, s::Neg(s::Mul(sq0, v_gamma)));
        auto exp1 = s::Exp(d, s::Neg(s::Mul(sq1, v_gamma)));
        auto exp2 = s::Exp(d, s::Neg(s::Mul(sq2, v_gamma)));
        auto exp3 = s::Exp(d, s::Neg(s::Mul(sq3, v_gamma)));
        auto exp4 = s::Exp(d, s::Neg(s::Mul(sq4, v_gamma)));
        auto exp5 = s::Exp(d, s::Neg(s::Mul(sq5, v_gamma)));
        auto exp6 = s::Exp(d, s::Neg(s::Mul(sq6, v_gamma)));
        auto exp7 = s::Exp(d, s::Neg(s::Mul(sq7, v_gamma)));

        s::Store(exp0, d, cache + k + 0 * lanes);
        s::Store(exp1, d, cache + k + 1 * lanes);
        s::Store(exp2, d, cache + k + 2 * lanes);
        s::Store(exp3, d, cache + k + 3 * lanes);
        s::Store(exp4, d, cache + k + 4 * lanes);
        s::Store(exp5, d, cache + k + 5 * lanes);
        s::Store(exp6, d, cache + k + 6 * lanes);
        s::Store(exp7, d, cache + k + 7 * lanes);

        v_sum0 = s::Add(v_sum0, exp0);
        v_sum1 = s::Add(v_sum1, exp1);
        v_sum2 = s::Add(v_sum2, exp2);
        v_sum3 = s::Add(v_sum3, exp3);
        v_sum4 = s::Add(v_sum4, exp4);
        v_sum5 = s::Add(v_sum5, exp5);
        v_sum6 = s::Add(v_sum6, exp6);
        v_sum7 = s::Add(v_sum7, exp7);
    }

    for (; k + 4 * lanes <= nnz; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        auto sq0 = s::Mul(v0, v0);
        auto sq1 = s::Mul(v1, v1);
        auto sq2 = s::Mul(v2, v2);
        auto sq3 = s::Mul(v3, v3);

        auto exp0 = s::Exp(d, s::Neg(s::Mul(sq0, v_gamma)));
        auto exp1 = s::Exp(d, s::Neg(s::Mul(sq1, v_gamma)));
        auto exp2 = s::Exp(d, s::Neg(s::Mul(sq2, v_gamma)));
        auto exp3 = s::Exp(d, s::Neg(s::Mul(sq3, v_gamma)));

        s::Store(exp0, d, cache + k + 0 * lanes);
        s::Store(exp1, d, cache + k + 1 * lanes);
        s::Store(exp2, d, cache + k + 2 * lanes);
        s::Store(exp3, d, cache + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, exp0);
        v_sum1 = s::Add(v_sum1, exp1);
        v_sum2 = s::Add(v_sum2, exp2);
        v_sum3 = s::Add(v_sum3, exp3);
    }

    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto sq = s::Mul(v, v);
        auto exp_v = s::Exp(d, s::Neg(s::Mul(sq, v_gamma)));
        s::Store(exp_v, d, cache + k);
        v_sum0 = s::Add(v_sum0, exp_v);
    }

    auto v_total = s::Add(
        s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3)),
        s::Add(s::Add(v_sum4, v_sum5), s::Add(v_sum6, v_sum7))
    );
    T sum = s::GetLane(s::SumOfLanes(d, v_total));

    for (; k < nnz; ++k) {
        T val = vals[k];
        T exp_term = std::exp(-gamma * val * val);
        cache[k] = exp_term;
        sum += exp_term;
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T self_kernel_sum_symmetric(
    const T* SCL_RESTRICT vals,
    Size nnz,
    Size N,
    T gamma,
    T sum_unary
) {
    const Size n_zeros = N - nnz;

    T sum = T(0);

    sum += static_cast<T>(n_zeros * n_zeros);

    if (n_zeros > 0) {
        sum += T(2) * static_cast<T>(n_zeros) * sum_unary;
    }

    sum += static_cast<T>(nnz);

    if (nnz <= 1) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T off_diag = T(0);

    for (size_t i = 0; i < nnz - 1; ++i) {
        const T vi = vals[i];
        const auto v_vi = s::Set(d, vi);

        auto v_row_sum0 = s::Zero(d);
        auto v_row_sum1 = s::Zero(d);

        size_t j = i + 1;

        for (; j + 2 * lanes <= nnz; j += 2 * lanes) {
            auto v_vj0 = s::Load(d, vals + j + 0 * lanes);
            auto v_vj1 = s::Load(d, vals + j + 1 * lanes);

            auto diff0 = s::Sub(v_vi, v_vj0);
            auto diff1 = s::Sub(v_vi, v_vj1);

            auto sq0 = s::Mul(diff0, diff0);
            auto sq1 = s::Mul(diff1, diff1);

            v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq0, v_gamma))));
            v_row_sum1 = s::Add(v_row_sum1, s::Exp(d, s::Neg(s::Mul(sq1, v_gamma))));
        }

        for (; j + lanes <= nnz; j += lanes) {
            auto v_vj = s::Load(d, vals + j);
            auto diff = s::Sub(v_vi, v_vj);
            auto sq = s::Mul(diff, diff);
            v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
        }

        off_diag += s::GetLane(s::SumOfLanes(d, s::Add(v_row_sum0, v_row_sum1)));

        for (; j < nnz; ++j) {
            T diff = vi - vals[j];
            off_diag += std::exp(-gamma * diff * diff);
        }
    }

    sum += T(2) * off_diag;

    return sum;
}

template <typename T>
SCL_FORCE_INLINE T cross_kernel_sum_blocked(
    const T* SCL_RESTRICT vals_x, Size nnz_x, Size N_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary
) {
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = T(0);

    sum += static_cast<T>(zeros_x * zeros_y);

    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    if (nnz_x == 0 || nnz_y == 0) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    constexpr size_t BLOCK_X = 64;
    constexpr size_t BLOCK_Y = 512;

    T cross_sum = T(0);

    for (size_t bx = 0; bx < nnz_x; bx += BLOCK_X) {
        size_t bx_end = (bx + BLOCK_X < static_cast<size_t>(nnz_x)) ? (bx + BLOCK_X) : static_cast<size_t>(nnz_x);

        for (size_t by = 0; by < nnz_y; by += BLOCK_Y) {
            size_t by_end = (by + BLOCK_Y < static_cast<size_t>(nnz_y)) ? (by + BLOCK_Y) : static_cast<size_t>(nnz_y);

            SCL_PREFETCH_READ(vals_y + by, 0);

            for (size_t i = bx; i < bx_end; ++i) {
                const T xi = vals_x[i];
                const auto v_xi = s::Set(d, xi);

                auto v_row_sum0 = s::Zero(d);
                auto v_row_sum1 = s::Zero(d);

                size_t j = by;

                for (; j + 2 * lanes <= by_end; j += 2 * lanes) {
                    auto v_yj0 = s::Load(d, vals_y + j + 0 * lanes);
                    auto v_yj1 = s::Load(d, vals_y + j + 1 * lanes);

                    auto diff0 = s::Sub(v_xi, v_yj0);
                    auto diff1 = s::Sub(v_xi, v_yj1);

                    auto sq0 = s::Mul(diff0, diff0);
                    auto sq1 = s::Mul(diff1, diff1);

                    v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq0, v_gamma))));
                    v_row_sum1 = s::Add(v_row_sum1, s::Exp(d, s::Neg(s::Mul(sq1, v_gamma))));
                }

                for (; j + lanes <= by_end; j += lanes) {
                    auto v_yj = s::Load(d, vals_y + j);
                    auto diff = s::Sub(v_xi, v_yj);
                    auto sq = s::Mul(diff, diff);
                    v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
                }

                cross_sum += s::GetLane(s::SumOfLanes(d, s::Add(v_row_sum0, v_row_sum1)));

                for (; j < by_end; ++j) {
                    T diff = xi - vals_y[j];
                    cross_sum += std::exp(-gamma * diff * diff);
                }
            }
        }
    }

    sum += cross_sum;
    return sum;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void unary_exp_sum(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache,
    T& out_sum
) {
    out_sum = detail::unary_exp_sum_ultra(vals, nnz, gamma, cache);
}

template <typename T>
SCL_FORCE_INLINE void cross_kernel_sum(
    const T* SCL_RESTRICT vals_x, Size nnz_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y,
    Size N_x, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary,
    T& out_sum
) {
    out_sum = detail::cross_kernel_sum_blocked(
        vals_x, nnz_x, N_x,
        vals_y, nnz_y, N_y,
        gamma, sum_x_unary, sum_y_unary
    );
}

template <typename T>
SCL_FORCE_INLINE void self_kernel_sum(
    const T* SCL_RESTRICT vals,
    Size nnz,
    Size N,
    T gamma,
    T sum_unary,
    T& out_sum
) {
    out_sum = detail::self_kernel_sum_symmetric(vals, nnz, N, gamma, sum_unary);
}

template <typename T, bool IsCSR>
void mmd_rbf(
    const Sparse<T, IsCSR>& mat_x,
    const Sparse<T, IsCSR>& mat_y,
    Array<T> output,
    T gamma = T(1)
) {
    const Index primary_dim = mat_x.primary_dim();
    const Size secondary_x = static_cast<Size>(mat_x.secondary_dim());
    const Size secondary_y = static_cast<Size>(mat_y.secondary_dim());

    SCL_CHECK_DIM(mat_y.primary_dim() == primary_dim, "MMD: Primary dimension mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "MMD: Output size mismatch");

    const T inv_Nx2 = T(1) / static_cast<T>(secondary_x * secondary_x);
    const T inv_Ny2 = T(1) / static_cast<T>(secondary_y * secondary_y);
    const T inv_NxNy = T(1) / static_cast<T>(secondary_x * secondary_y);

    // Find max nnz for buffer allocation
    Size max_nnz_x = 0, max_nnz_y = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size nnz_x = static_cast<Size>(mat_x.primary_length(i));
        Size nnz_y = static_cast<Size>(mat_y.primary_length(i));
        if (nnz_x > max_nnz_x) max_nnz_x = nnz_x;
        if (nnz_y > max_nnz_y) max_nnz_y = nnz_y;
    }

    // Pre-allocate cache pools for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Size max_cache = (max_nnz_x > max_nnz_y) ? max_nnz_x : max_nnz_y;

    scl::threading::DualWorkspacePool<T> cache_pool;
    if (max_cache > 0) {
        cache_pool.init(n_threads, max_cache);
    }

    scl::threading::parallel_for(Index(0), primary_dim, [&](size_t p_sz, size_t thread_rank) {
        const Index p = static_cast<Index>(p_sz);

        auto vals_x_arr = mat_x.primary_values(p);
        auto vals_y_arr = mat_y.primary_values(p);
        Size nnz_x = static_cast<Size>(mat_x.primary_length(p));
        Size nnz_y = static_cast<Size>(mat_y.primary_length(p));

        if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
            output[static_cast<Size>(p)] = T(0);
            return;
        }

        T* SCL_RESTRICT x_cache = cache_pool.get1(thread_rank);
        T* SCL_RESTRICT y_cache = cache_pool.get2(thread_rank);

        T sum_x_unary = (nnz_x > 0)
            ? detail::unary_exp_sum_ultra(vals_x_arr.ptr, nnz_x, gamma, x_cache)
            : T(0);
        T sum_y_unary = (nnz_y > 0)
            ? detail::unary_exp_sum_ultra(vals_y_arr.ptr, nnz_y, gamma, y_cache)
            : T(0);

        T sum_xx = detail::self_kernel_sum_symmetric(vals_x_arr.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
        T sum_yy = detail::self_kernel_sum_symmetric(vals_y_arr.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
        T sum_xy = detail::cross_kernel_sum_blocked(
            vals_x_arr.ptr, nnz_x, secondary_x,
            vals_y_arr.ptr, nnz_y, secondary_y,
            gamma, sum_x_unary, sum_y_unary
        );

        T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);

        output[static_cast<Size>(p)] = (mmd2 > T(0)) ? mmd2 : T(0);
    });
}

} // namespace scl::kernel::mmd

