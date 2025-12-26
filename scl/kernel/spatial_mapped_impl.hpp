#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>

// =============================================================================
/// @file spatial_mapped_impl.hpp
/// @brief Spatial Statistics for Mapped Sparse Matrices
///
/// Spatial statistics (Moran's I) require weight summation.
/// For Mapped matrices, we stream the data to compute weight sums.
///
/// Operations:
/// - weight_sum_mapped: Streaming weight sum computation
/// - compute_spatial_lag_mapped: Compute spatial lag values
// =============================================================================

namespace scl::kernel::spatial::mapped {

// =============================================================================
// Weight Sum - Streaming Implementation
// =============================================================================

/// @brief Compute total weight sum from mapped graph (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void weight_sum_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& graph,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);
    const Index total_nnz = graph.indptr[primary_dim];

    if (total_nnz == 0) {
        out_sum = static_cast<T>(0);
        return;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const T* SCL_RESTRICT data = graph.data;

    // 4-way unrolled SIMD accumulation
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    size_t i = 0;
    for (; i + 4 * lanes <= total_nnz; i += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0 * lanes));
        v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1 * lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; i + lanes <= total_nnz; i += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, data + i));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < total_nnz; ++i) {
        sum += data[i];
    }

    out_sum = sum;
}

/// @brief Compute total weight sum from mapped graph (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void weight_sum_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& graph,
    Array<T> workspace,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(workspace.len >= static_cast<Size>(primary_dim), "Workspace too small");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Parallel reduction across rows
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = graph.lengths[p];

        if (len == 0) {
            workspace[p] = static_cast<T>(0);
            return;
        }

        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(graph.data_ptrs[p]);

        // 4-way unrolled SIMD accumulation
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);

        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
            v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
            v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
            v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
        }

        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }

        T sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; k < len; ++k) {
            sum += vals[k];
        }

        workspace[p] = sum;
    });

    // Final reduction
    T total_sum = static_cast<T>(0);
    for (Index p = 0; p < primary_dim; ++p) {
        total_sum += workspace[p];
    }

    out_sum = total_sum;
}

// =============================================================================
// Spatial Lag Computation
// =============================================================================

/// @brief Compute spatial lag values from mapped graph (MappedCustomSparse)
///
/// spatial_lag[i] = sum_j(w[i,j] * x[j])
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_spatial_lag_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& graph,
    Array<const T> x,
    Array<T> out_lag
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(out_lag.len == static_cast<Size>(primary_dim), "Output lag size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = graph.indptr[p];
        Index end = graph.indptr[p + 1];
        Index len = end - start;

        if (len == 0) {
            out_lag[p] = static_cast<T>(0);
            return;
        }

        const T* SCL_RESTRICT weights = graph.data + start;
        const Index* SCL_RESTRICT indices = graph.indices + start;

        T sum = static_cast<T>(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            sum += weights[k] * x[j];
        }

        out_lag[p] = sum;
    });
}

/// @brief Compute spatial lag values from mapped graph (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_spatial_lag_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& graph,
    Array<const T> x,
    Array<T> out_lag
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(out_lag.len == static_cast<Size>(primary_dim), "Output lag size mismatch");

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = graph.lengths[p];

        if (len == 0) {
            out_lag[p] = static_cast<T>(0);
            return;
        }

        const T* SCL_RESTRICT weights = static_cast<const T*>(graph.data_ptrs[p]);
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(graph.indices_ptrs[p]);

        T sum = static_cast<T>(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            sum += weights[k] * x[j];
        }

        out_lag[p] = sum;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void weight_sum_mapped_dispatch(
    const MatrixT& graph,
    Array<typename MatrixT::ValueType> workspace,
    typename MatrixT::ValueType& out_sum
) {
    if constexpr (kernel::mapped::detail::IsMappedCustomSparse<MatrixT>) {
        // MappedCustomSparse doesn't need workspace
        weight_sum_mapped(graph, out_sum);
    } else {
        weight_sum_mapped(graph, workspace, out_sum);
    }
}

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_spatial_lag_mapped_dispatch(
    const MatrixT& graph,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> out_lag
) {
    compute_spatial_lag_mapped(graph, x, out_lag);
}

} // namespace scl::kernel::spatial::mapped

