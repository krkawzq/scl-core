#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file spatial_mapped_impl.hpp
/// @brief Spatial Statistics for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Weight Sum
///    - SIMD accumulation on mapped data
///    - Prefetch hints for OS
///
/// 2. Optimized Centered Value Materialization
///    - Same SIMD fill/scatter as fast_impl
///
/// 3. Prefetch for Graph Traversal
///    - Hint upcoming pages
///
/// Performance: Near-RAM for cached, graceful degradation for cold
// =============================================================================

namespace scl::kernel::spatial::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 8;
}

// =============================================================================
// SECTION 2: SIMD Helpers
// =============================================================================

namespace detail {

/// @brief SIMD sum
template <typename T>
SCL_FORCE_INLINE T simd_sum_array(const T* SCL_RESTRICT data, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size i = 0;
    for (; i + 4 * lanes <= len; i += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0 * lanes));
        v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1 * lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; i + lanes <= len; i += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, data + i));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

/// @brief SIMD fill
template <typename T>
SCL_FORCE_INLINE void simd_fill(T* SCL_RESTRICT data, Size len, T val) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_val = s::Set(d, val);

    Size i = 0;
    for (; i + 4 * lanes <= len; i += 4 * lanes) {
        s::Store(v_val, d, data + i + 0 * lanes);
        s::Store(v_val, d, data + i + 1 * lanes);
        s::Store(v_val, d, data + i + 2 * lanes);
        s::Store(v_val, d, data + i + 3 * lanes);
    }

    for (; i + lanes <= len; i += lanes) {
        s::Store(v_val, d, data + i);
    }

    for (; i < len; ++i) {
        data[i] = val;
    }
}

/// @brief SIMD squared sum
template <typename T>
SCL_FORCE_INLINE T simd_squared_sum(const T* SCL_RESTRICT data, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size i = 0;
    for (; i + 2 * lanes <= len; i += 2 * lanes) {
        auto v0 = s::Load(d, data + i + 0 * lanes);
        auto v1 = s::Load(d, data + i + 1 * lanes);
        v_sum0 = s::MulAdd(v0, v0, v_sum0);
        v_sum1 = s::MulAdd(v1, v1, v_sum1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);

    for (; i + lanes <= len; i += lanes) {
        auto v = s::Load(d, data + i);
        v_sum = s::MulAdd(v, v, v_sum);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; i < len; ++i) {
        sum += data[i] * data[i];
    }

    return sum;
}

/// @brief Compute weighted product with prefetch
template <typename T>
SCL_FORCE_INLINE T compute_weighted_product(
    const T* SCL_RESTRICT weights,
    const Index* SCL_RESTRICT indices,
    Size len,
    T z_i,
    const T* SCL_RESTRICT z
) {
    T sum = T(0);

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (k + config::PREFETCH_DISTANCE < len) {
            SCL_PREFETCH_READ(&z[indices[k + config::PREFETCH_DISTANCE]], 0);
        }

        sum += weights[k + 0] * z_i * z[indices[k + 0]];
        sum += weights[k + 1] * z_i * z[indices[k + 1]];
        sum += weights[k + 2] * z_i * z[indices[k + 2]];
        sum += weights[k + 3] * z_i * z[indices[k + 3]];
    }

    for (; k < len; ++k) {
        sum += weights[k] * z_i * z[indices[k]];
    }

    return sum;
}

} // namespace detail

// =============================================================================
// SECTION 3: Weight Sum (MappedCustomSparse)
// =============================================================================

/// @brief Streaming weight sum (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void weight_sum_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& graph,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);
    const Size total_nnz = static_cast<Size>(graph.indptr[primary_dim]);

    if (total_nnz == 0) {
        out_sum = T(0);
        return;
    }

    kernel::mapped::hint_prefetch(graph);

    out_sum = detail::simd_sum_array(graph.data, total_nnz);
}

/// @brief Streaming weight sum (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void weight_sum_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& graph,
    Array<T> workspace,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(workspace.len >= static_cast<Size>(primary_dim), "Workspace too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Size len = static_cast<Size>(graph.lengths[p]);
        if (len == 0) {
            workspace[p] = T(0);
            return;
        }

        const T* vals = static_cast<const T*>(graph.data_ptrs[p]);
        workspace[p] = detail::simd_sum_array(vals, len);
    });

    out_sum = detail::simd_sum_array(workspace.ptr, static_cast<Size>(primary_dim));
}

// =============================================================================
// SECTION 4: Spatial Lag Computation
// =============================================================================

/// @brief Compute spatial lag (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_spatial_lag_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& graph,
    Array<const T> x,
    Array<T> out_lag
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(out_lag.len == static_cast<Size>(primary_dim), "Output lag size mismatch");

    kernel::mapped::hint_prefetch(graph);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = graph.indptr[p];
        Index end = graph.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) {
            out_lag[p] = T(0);
            return;
        }

        const T* SCL_RESTRICT weights = graph.data + start;
        const Index* SCL_RESTRICT indices = graph.indices + start;

        T sum = T(0);

        // Unrolled with prefetch
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&x[indices[k + config::PREFETCH_DISTANCE]], 0);
            }
            sum += weights[k + 0] * x[indices[k + 0]];
            sum += weights[k + 1] * x[indices[k + 1]];
            sum += weights[k + 2] * x[indices[k + 2]];
            sum += weights[k + 3] * x[indices[k + 3]];
        }

        for (; k < len; ++k) {
            sum += weights[k] * x[indices[k]];
        }

        out_lag[p] = sum;
    });
}

/// @brief Compute spatial lag (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_spatial_lag_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& graph,
    Array<const T> x,
    Array<T> out_lag
) {
    const Index primary_dim = scl::primary_size(graph);

    SCL_CHECK_DIM(out_lag.len == static_cast<Size>(primary_dim), "Output lag size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Size len = static_cast<Size>(graph.lengths[p]);

        if (len == 0) {
            out_lag[p] = T(0);
            return;
        }

        const T* SCL_RESTRICT weights = static_cast<const T*>(graph.data_ptrs[p]);
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(graph.indices_ptrs[p]);

        T sum = T(0);

        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&x[indices[k + config::PREFETCH_DISTANCE]], 0);
            }
            sum += weights[k + 0] * x[indices[k + 0]];
            sum += weights[k + 1] * x[indices[k + 1]];
            sum += weights[k + 2] * x[indices[k + 2]];
            sum += weights[k + 3] * x[indices[k + 3]];
        }

        for (; k < len; ++k) {
            sum += weights[k] * x[indices[k]];
        }

        out_lag[p] = sum;
    });
}

// =============================================================================
// SECTION 5: Full Moran's I (MappedCustomSparse)
// =============================================================================

/// @brief Full Moran's I (MappedCustomSparse graph and features)
template <typename T, bool GraphCSR, bool FeatCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, GraphCSR>, GraphCSR> &&
             kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, FeatCSR>, FeatCSR>
void morans_i_mapped(
    const scl::io::MappedCustomSparse<T, GraphCSR>& graph,
    const scl::io::MappedCustomSparse<T, FeatCSR>& features,
    Array<Real> output
) {
    const Index n_cells = scl::primary_size(graph);
    const Index n_features = scl::primary_size(features);

    SCL_CHECK_DIM(scl::secondary_size(graph) == n_cells, "Graph must be square");
    SCL_CHECK_DIM(scl::secondary_size(features) == n_cells, "Features dim mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(n_features), "Output size mismatch");

    kernel::mapped::hint_prefetch(graph);
    kernel::mapped::hint_prefetch(features);

    // Compute weight sum
    T W_sum;
    weight_sum_mapped(graph, W_sum);

    if (W_sum <= T(0)) {
        detail::simd_fill(output.ptr, output.len, Real(0));
        return;
    }

    const Real N = static_cast<Real>(n_cells);
    const Real N_over_W = N / static_cast<Real>(W_sum);

    // Parallel over features
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f) {
        Index f_idx = static_cast<Index>(f);

        auto feat_vals = scl::primary_values(features, f_idx);
        auto feat_inds = scl::primary_indices(features, f_idx);

        // Compute mean
        Real sum = (feat_vals.len > 0)
            ? static_cast<Real>(detail::simd_sum_array(feat_vals.ptr, feat_vals.len))
            : Real(0);
        Real mean = sum / N;

        // Materialize centered values
        thread_local std::vector<Real> z;
        z.resize(static_cast<size_t>(n_cells));

        detail::simd_fill(z.data(), static_cast<Size>(n_cells), -mean);

        for (Size k = 0; k < feat_vals.len; ++k) {
            z[feat_inds[k]] = static_cast<Real>(feat_vals[k]) - mean;
        }

        // Compute variance
        Real denom = detail::simd_squared_sum(z.data(), static_cast<Size>(n_cells));

        if (denom <= Real(0)) {
            output[f] = Real(0);
            return;
        }

        // Compute numerator
        Real numer = Real(0);

        for (Index i = 0; i < n_cells; ++i) {
            Index g_start = graph.indptr[i];
            Index g_end = graph.indptr[i + 1];
            Size g_len = static_cast<Size>(g_end - g_start);

            if (g_len == 0) continue;

            numer += detail::compute_weighted_product(
                graph.data + g_start,
                graph.indices + g_start,
                g_len,
                static_cast<T>(z[i]),
                reinterpret_cast<const T*>(z.data())
            );
        }

        output[f] = N_over_W * (numer / denom);
    });
}

// =============================================================================
// SECTION 6: Unified Dispatchers
// =============================================================================

/// @brief Weight sum dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void weight_sum_mapped_dispatch(
    const MatrixT& graph,
    Array<typename MatrixT::ValueType> workspace,
    typename MatrixT::ValueType& out_sum
) {
    if constexpr (kernel::mapped::detail::IsMappedCustomSparse<MatrixT>) {
        weight_sum_mapped(graph, out_sum);
    } else {
        weight_sum_mapped(graph, workspace, out_sum);
    }
}

/// @brief Spatial lag dispatcher
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
