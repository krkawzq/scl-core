#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>

// =============================================================================
/// @file bbknn_mapped_impl.hpp
/// @brief BBKNN for Mapped Sparse Matrices
///
/// BBKNN (Batch-Balanced KNN) requires norm precomputation.
/// For Mapped matrices, we stream the data to compute norms.
///
/// Operations:
/// - compute_norms_bbknn_mapped: Streaming norm computation
// =============================================================================

namespace scl::kernel::bbknn::mapped {

// =============================================================================
// Norm Computation - Streaming Implementation
// =============================================================================

/// @brief Compute squared norms from mapped matrix (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_norms_bbknn_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len == static_cast<Size>(primary_dim), "Norms size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;

        const T* SCL_RESTRICT vals = matrix.data + start;

        // 4-way unrolled SIMD accumulation
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);

        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            v_sum0 = s::MulAdd(v0, v0, v_sum0);
            v_sum1 = s::MulAdd(v1, v1, v_sum1);
            v_sum2 = s::MulAdd(v2, v2, v_sum2);
            v_sum3 = s::MulAdd(v3, v3, v_sum3);
        }

        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::MulAdd(v, v, v_sum);
        }

        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; k < len; ++k) {
            T val = vals[k];
            sum_sq += val * val;
        }

        norms_sq[p] = sum_sq;
    });
}

/// @brief Compute squared norms from mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_norms_bbknn_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> norms_sq
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(norms_sq.len == static_cast<Size>(primary_dim), "Norms size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];

        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);

        // 4-way unrolled SIMD accumulation
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);

        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            v_sum0 = s::MulAdd(v0, v0, v_sum0);
            v_sum1 = s::MulAdd(v1, v1, v_sum1);
            v_sum2 = s::MulAdd(v2, v2, v_sum2);
            v_sum3 = s::MulAdd(v3, v3, v_sum3);
        }

        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::MulAdd(v, v, v_sum);
        }

        T sum_sq = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; k < len; ++k) {
            T val = vals[k];
            sum_sq += val * val;
        }

        norms_sq[p] = sum_sq;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_norms_bbknn_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> norms_sq
) {
    compute_norms_bbknn_mapped(matrix, norms_sq);
}

} // namespace scl::kernel::bbknn::mapped

