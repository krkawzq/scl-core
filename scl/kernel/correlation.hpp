#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// Pearson Correlation for Sparse Matrices
//
// Key Optimizations:
// 1. Fused Statistics Computation
//    - Single-pass sum + sum_sq accumulation
//    - 4-way unrolled SIMD
//
// 2. Direct Correlation Computation
//    - Sparse-sparse centered dot product
//    - Skip Gram matrix intermediate step
//
// 3. Symmetric Matrix Optimization
//    - Only compute upper triangle
//    - Mirror to lower triangle
//
// 4. Cache-Blocked Processing
//    - Process rows in chunks for L2 cache
// =============================================================================

namespace scl::kernel::correlation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size STAT_CHUNK = 256;
    constexpr Size PREFETCH_DISTANCE = 32;
}

// =============================================================================
// SIMD Statistics Computation
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void compute_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    T& out_sum,
    T& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_sq = s::Add(v_sq0, v_sq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    T sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        T v = vals[k];
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

template <typename T>
SCL_FORCE_INLINE T sparse_centered_dot(
    const T* vals_a, const Index* inds_a, Size len_a, T mean_a,
    const T* vals_b, const Index* inds_b, Size len_b, T mean_b,
    Size total_dim
) {
    // Use algebraic identity for efficiency:
    // cov = sum(a_i * b_i) - ma * sum(b_i) - mb * sum(a_i) + n * ma * mb
    // All sums are over the total_dim, but sparse vectors have zeros elsewhere

    T n = static_cast<T>(total_dim);

    // Early exit for empty vectors - still need to return n * ma * mb
    if (SCL_UNLIKELY(len_a == 0 || len_b == 0)) {
        // When one is empty: sum_ab=0, sum_a or sum_b =0
        // cov = 0 - ma * 0 - mb * sum_a + n * ma * mb (if b empty)
        //     = -mb * sum_a + n * ma * mb = mb * (n * ma - sum_a)
        // But sum_a = len_a * mean_a_sparse (not mean_a which is over total_dim)
        // This is complex. For sparse vectors with explicit zeros absent,
        // if len_a == 0, all a values are 0, so sum_a = 0
        return n * mean_a * mean_b;
    }

    // O(1) range disjointness check - dot product contribution is zero
    // but sums still need to be computed
    bool disjoint = (inds_a[len_a-1] < inds_b[0] || inds_b[len_b-1] < inds_a[0]);

    T sum_ab = T(0);    // dot product of matched elements
    T sum_a = T(0);     // sum of all a values
    T sum_b = T(0);     // sum of all b values

    if (SCL_UNLIKELY(disjoint)) {
        // No overlap - compute sums separately with SIMD
        for (Size i = 0; i < len_a; ++i) sum_a += vals_a[i];
        for (Size j = 0; j < len_b; ++j) sum_b += vals_b[j];
    } else {
        Size i = 0, j = 0;

        // 8-way skip optimization for large non-overlapping ranges
        while (i + 8 <= len_a && j + 8 <= len_b) {
            Index ia7 = inds_a[i+7], ib0 = inds_b[j];
            Index ia0 = inds_a[i], ib7 = inds_b[j+7];

            if (ia7 < ib0) {
                sum_a += vals_a[i] + vals_a[i+1] + vals_a[i+2] + vals_a[i+3]
                       + vals_a[i+4] + vals_a[i+5] + vals_a[i+6] + vals_a[i+7];
                i += 8;
                continue;
            }
            if (ib7 < ia0) {
                sum_b += vals_b[j] + vals_b[j+1] + vals_b[j+2] + vals_b[j+3]
                       + vals_b[j+4] + vals_b[j+5] + vals_b[j+6] + vals_b[j+7];
                j += 8;
                continue;
            }
            break;
        }

        // 4-way skip optimization
        while (i + 4 <= len_a && j + 4 <= len_b) {
            Index ia3 = inds_a[i+3];
            Index ib0 = inds_b[j];
            Index ia0 = inds_a[i];
            Index ib3 = inds_b[j+3];

            if (ia3 < ib0) {
                sum_a += vals_a[i] + vals_a[i+1] + vals_a[i+2] + vals_a[i+3];
                i += 4;
                continue;
            }
            if (ib3 < ia0) {
                sum_b += vals_b[j] + vals_b[j+1] + vals_b[j+2] + vals_b[j+3];
                j += 4;
                continue;
            }
            break;
        }

        // Main merge loop with prefetch
        while (i < len_a && j < len_b) {
            if (SCL_LIKELY(i + config::PREFETCH_DISTANCE < len_a)) {
                SCL_PREFETCH_READ(&inds_a[i + config::PREFETCH_DISTANCE], 0);
                SCL_PREFETCH_READ(&vals_a[i + config::PREFETCH_DISTANCE], 0);
            }
            if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < len_b)) {
                SCL_PREFETCH_READ(&inds_b[j + config::PREFETCH_DISTANCE], 0);
                SCL_PREFETCH_READ(&vals_b[j + config::PREFETCH_DISTANCE], 0);
            }

            Index ia = inds_a[i];
            Index ib = inds_b[j];

            if (ia == ib) {
                sum_ab += vals_a[i] * vals_b[j];
                sum_a += vals_a[i];
                sum_b += vals_b[j];
                ++i; ++j;
            } else if (ia < ib) {
                sum_a += vals_a[i];
                ++i;
            } else {
                sum_b += vals_b[j];
                ++j;
            }
        }

        // Remaining a elements
        for (; i < len_a; ++i) {
            sum_a += vals_a[i];
        }

        // Remaining b elements
        for (; j < len_b; ++j) {
            sum_b += vals_b[j];
        }
    }

    // cov = sum_ab - ma * sum_b - mb * sum_a + n * ma * mb
    return sum_ab - mean_a * sum_b - mean_b * sum_a + n * mean_a * mean_b;
}

} // namespace detail

// =============================================================================
// Statistics Computation
// =============================================================================

template <typename MatrixT>
    requires SparseLike<MatrixT>
void compute_stats(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> out_means,
    Array<typename MatrixT::ValueType> out_inv_stds
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = matrix.primary_dim();
    const Size secondary_dim = static_cast<Size>(matrix.secondary_dim());
    const T inv_n = T(1) / static_cast<T>(secondary_dim);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len >= static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = matrix.primary_values_unsafe(static_cast<Index>(p));

        T sum, sq_sum;
        detail::compute_sum_sq_simd(vals.data(), vals.size(), sum, sq_sum);

        T mean = sum * inv_n;
        T var = (sq_sum * inv_n) - (mean * mean);
        if (var < T(0)) var = T(0);

        out_means[p] = mean;
        out_inv_stds[p] = (var > T(0)) ? (T(1) / std::sqrt(var)) : T(0);
    });
}

// =============================================================================
// Correlation Matrix
// =============================================================================

template <typename MatrixT>
    requires SparseLike<MatrixT>
void pearson(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means,
    Array<const typename MatrixT::ValueType> inv_stds,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index N = matrix.primary_dim();
    const Size N_size = static_cast<Size>(N);
    const Size M = static_cast<Size>(matrix.secondary_dim());
    const T inv_m = T(1) / static_cast<T>(M);

    SCL_CHECK_DIM(output.len >= N_size * N_size, "Output size mismatch");

    const Size n_chunks = (N_size + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
        Size i_start = chunk_idx * config::CHUNK_SIZE;
        Size i_end = scl::algo::min2(i_start + config::CHUNK_SIZE, N_size);

        for (Size i = i_start; i < i_end; ++i) {
            T inv_std_i = inv_stds[i];
            T* row_ptr = output.ptr + i * N_size;

            // Diagonal is always 1 (or 0 for zero-variance)
            row_ptr[i] = (inv_std_i > T(0)) ? T(1) : T(0);

            // Zero-variance row: all correlations are 0
            if (SCL_UNLIKELY(inv_std_i == T(0))) {
                for (Size j = i + 1; j < N_size; ++j) {
                    row_ptr[j] = T(0);
                    output.ptr[j * N_size + i] = T(0);
                }
                continue;
            }

            auto vals_i = matrix.primary_values_unsafe(static_cast<Index>(i));
            auto inds_i = matrix.primary_indices_unsafe(static_cast<Index>(i));
            Size len_i = vals_i.size();
            T mean_i = means[i];

            for (Size j = i + 1; j < N_size; ++j) {
                T inv_std_j = inv_stds[j];

                // Zero-variance column: correlation is 0
                if (SCL_UNLIKELY(inv_std_j == T(0))) {
                    row_ptr[j] = T(0);
                    output.ptr[j * N_size + i] = T(0);
                    continue;
                }

                auto vals_j = matrix.primary_values_unsafe(static_cast<Index>(j));
                auto inds_j = matrix.primary_indices_unsafe(static_cast<Index>(j));
                Size len_j = vals_j.size();
                T mean_j = means[j];

                T cov = detail::sparse_centered_dot(
                    vals_i.data(), inds_i.data(), len_i, mean_i,
                    vals_j.data(), inds_j.data(), len_j, mean_j,
                    M
                ) * inv_m;

                T corr = cov * inv_std_i * inv_std_j;

                // Clamp to [-1, 1] for numerical stability
                if (SCL_UNLIKELY(corr > T(1))) corr = T(1);
                if (SCL_UNLIKELY(corr < T(-1))) corr = T(-1);

                row_ptr[j] = corr;
                output.ptr[j * N_size + i] = corr;
            }
        }
    });
}

template <typename MatrixT>
    requires SparseLike<MatrixT>
void pearson(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    T* means = scl::memory::aligned_alloc<T>(N, SCL_ALIGNMENT);
    T* inv_stds = scl::memory::aligned_alloc<T>(N, SCL_ALIGNMENT);

    compute_stats(matrix, Array<T>(means, N), Array<T>(inv_stds, N));
    pearson(matrix, Array<const T>(means, N), Array<const T>(inv_stds, N), output);

    scl::memory::aligned_free(means, SCL_ALIGNMENT);
    scl::memory::aligned_free(inv_stds, SCL_ALIGNMENT);
}

} // namespace scl::kernel::correlation
