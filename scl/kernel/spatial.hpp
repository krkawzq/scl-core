#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Include optimized backends
#include "scl/kernel/spatial_fast_impl.hpp"
#include "scl/kernel/spatial_mapped_impl.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file spatial.hpp
/// @brief Spatial Autocorrelation Statistics
///
/// ## Operations
///
/// - morans_i: Moran's I spatial autocorrelation statistic
/// - compute_spatial_lag: Spatial lag values (weighted sum of neighbors)
///
/// ## Formula
///
/// Moran's I = (N/W) * sum_ij(w_ij * (x_i - mean) * (x_j - mean)) / sum_i((x_i - mean)^2)
///
/// Where:
/// - N = number of cells
/// - W = sum of all weights
/// - w_ij = spatial weight between cells i and j
/// - x_i = feature value for cell i
///
/// ## Performance Optimizations
///
/// 1. SIMD Weight Sum
///    - 4-way unrolled accumulation
///    - Direct array access when possible
///
/// 2. Optimized Centered Value Materialization
///    - SIMD fill with -mean
///    - Sparse scatter for non-zeros
///
/// 3. Vectorized Variance
///    - SIMD squared sum with FMA
///
/// 4. Prefetch for Graph Traversal
///    - Random access to z[] vector optimized
///
/// ## Performance
///
/// - O(nnz_graph + nnz_feature) per feature
/// - ~5-10 GB/s per core
// =============================================================================

namespace scl::kernel::spatial {

// =============================================================================
// SECTION 1: SIMD Helpers
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
    constexpr Size PREFETCH_DIST = 8;
    T sum = T(0);

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (k + PREFETCH_DIST < len) {
            SCL_PREFETCH_READ(&z[indices[k + PREFETCH_DIST]], 0);
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
// SECTION 2: Moran's I (Generic Implementation)
// =============================================================================

/// @brief Compute Moran's I statistic (unified for all sparse types)
///
/// Dispatches to optimized backend based on matrix types.
///
/// @tparam GraphT Graph sparse matrix type (cells x cells)
/// @tparam FeatureT Feature sparse matrix type (features x cells or cells x features)
/// @param graph Spatial weight matrix (cells x cells), typically CSR
/// @param features Feature matrix, typically CSC (genes x cells)
/// @param output Output Moran's I values [size = n_features], PRE-ALLOCATED
template <typename GraphT, typename FeatureT>
    requires AnySparse<GraphT> && AnySparse<FeatureT>
void morans_i(
    const GraphT& graph,
    const FeatureT& features,
    Array<Real> output
) {
    using T = typename GraphT::ValueType;
    constexpr bool GraphCSR = std::is_same_v<typename GraphT::Tag, TagCSR>;
    constexpr bool FeatCSR = std::is_same_v<typename FeatureT::Tag, TagCSR>;

    const Index n_cells = scl::primary_size(graph);
    const Index n_features = scl::primary_size(features);

    SCL_CHECK_DIM(scl::secondary_size(graph) == n_cells, "Graph must be square");
    SCL_CHECK_DIM(scl::secondary_size(features) == n_cells, "Features dim mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(n_features), "Output size mismatch");

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<GraphT, GraphCSR> && CustomSparseLike<FeatureT, FeatCSR>) {
        fast::morans_i_custom(graph, features, output);
        return;
    }

    // Generic fallback
    // Compute weight sum
    Real W_sum = Real(0);
    for (Index i = 0; i < n_cells; ++i) {
        auto vals = scl::primary_values(graph, i);
        for (Size k = 0; k < vals.len; ++k) {
            W_sum += static_cast<Real>(vals[k]);
        }
    }

    if (W_sum <= Real(0)) {
        detail::simd_fill(output.ptr, output.len, Real(0));
        return;
    }

    const Real N = static_cast<Real>(n_cells);
    const Real N_over_W = N / W_sum;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f) {
        Index f_idx = static_cast<Index>(f);

        auto feat_vals = scl::primary_values(features, f_idx);
        auto feat_inds = scl::primary_indices(features, f_idx);

        // Compute mean
        Real sum = Real(0);
        for (Size k = 0; k < feat_vals.len; ++k) {
            sum += static_cast<Real>(feat_vals[k]);
        }
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
            auto graph_vals = scl::primary_values(graph, i);
            auto graph_inds = scl::primary_indices(graph, i);
            Size g_len = graph_vals.len;

            if (g_len == 0) continue;

            numer += detail::compute_weighted_product(
                graph_vals.ptr, graph_inds.ptr, g_len,
                static_cast<T>(z[i]),
                reinterpret_cast<const T*>(z.data())
            );
        }

        output[f] = N_over_W * (numer / denom);
    });
}

// =============================================================================
// SECTION 3: Spatial Lag
// =============================================================================

/// @brief Compute spatial lag values
///
/// spatial_lag[i] = sum_j(w[i,j] * x[j])
///
/// @tparam GraphT Graph sparse matrix type
/// @param graph Spatial weight matrix (cells x cells)
/// @param x Input values [size = n_cells]
/// @param out_lag Output lag values [size = n_cells], PRE-ALLOCATED
template <typename GraphT, typename T>
    requires AnySparse<GraphT>
void compute_spatial_lag(
    const GraphT& graph,
    Array<const T> x,
    Array<T> out_lag
) {
    constexpr bool IsCSR = std::is_same_v<typename GraphT::Tag, TagCSR>;
    const Index n_cells = scl::primary_size(graph);

    SCL_CHECK_DIM(x.len == static_cast<Size>(n_cells), "Input x size mismatch");
    SCL_CHECK_DIM(out_lag.len == static_cast<Size>(n_cells), "Output lag size mismatch");

    // Dispatch to mapped backend if applicable
    if constexpr (kernel::mapped::MappedSparseLike<GraphT, IsCSR>) {
        mapped::compute_spatial_lag_mapped(graph, x, out_lag);
        return;
    }

    // Generic with prefetch optimization
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells), [&](size_t i) {
        auto vals = scl::primary_values(graph, static_cast<Index>(i));
        auto inds = scl::primary_indices(graph, static_cast<Index>(i));
        Size len = vals.len;

        if (len == 0) {
            out_lag[i] = T(0);
            return;
        }

        out_lag[i] = detail::compute_weighted_product(
            vals.ptr, inds.ptr, len,
            T(1),  // z_i = 1 for simple weighted sum
            x.ptr
        );
    });
}

// =============================================================================
// SECTION 4: Weight Sum
// =============================================================================

/// @brief Compute total weight sum
///
/// @tparam GraphT Graph sparse matrix type
/// @param graph Spatial weight matrix
/// @return Total sum of all weights
template <typename GraphT>
    requires AnySparse<GraphT>
typename GraphT::ValueType compute_weight_sum(const GraphT& graph) {
    using T = typename GraphT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename GraphT::Tag, TagCSR>;

    const Index n_cells = scl::primary_size(graph);

    // Dispatch to optimized backends
    if constexpr (CustomSparseLike<GraphT, IsCSR>) {
        T sum;
        fast::weight_sum_custom(graph, sum);
        return sum;
    } else if constexpr (kernel::mapped::MappedSparseLike<GraphT, IsCSR>) {
        if constexpr (kernel::mapped::detail::IsMappedCustomSparse<GraphT>) {
            T sum;
            mapped::weight_sum_mapped(graph, sum);
            return sum;
        }
    }

    // Generic fallback
    T total = T(0);
    for (Index i = 0; i < n_cells; ++i) {
        auto vals = scl::primary_values(graph, i);
        for (Size k = 0; k < vals.len; ++k) {
            total += vals[k];
        }
    }
    return total;
}

} // namespace scl::kernel::spatial
