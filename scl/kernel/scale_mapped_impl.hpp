#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file scale_mapped_impl.hpp
/// @brief Extreme Performance Scaling for Memory-Mapped Sparse Matrices
///
/// ## Key Insight
///
/// Mapped data is READ-ONLY. For in-place operations we must materialize.
/// Optimization: FUSE materialization with transformation to save bandwidth.
///
/// ## Key Optimizations
///
/// 1. Fused Materialize + Transform
///    - Single pass: read source, transform, write to output
///    - Saves one memory traversal vs materialize-then-transform
///
/// 2. Chunk-Based Processing
///    - Process rows in L2-friendly chunks
///    - Prefetch next chunk while processing current
///
/// 3. 4-Way Unrolled SIMD
///    - Fused (x - mu) * inv_sigma operations
///    - Optional clipping fused with scaling
///
/// 4. Streaming Writes
///    - Sequential writes to output for cache efficiency
///
/// Performance: Near in-memory performance with minimal I/O overhead
// =============================================================================

namespace scl::kernel::scale::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 256;
}

// =============================================================================
// SECTION 2: Fused Transform Helpers
// =============================================================================

namespace detail {

/// @brief Fused copy + standardize (SIMD)
template <typename T>
SCL_FORCE_INLINE void fused_standardize(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_mu = s::Set(d, mu);
    const auto v_inv_sigma = s::Set(d, inv_sigma);
    const auto v_max = s::Set(d, max_val);
    const auto v_min = s::Set(d, -max_val);

    Size k = 0;

    // 4-way unrolled with prefetch
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        // Prefetch ahead
        SCL_PREFETCH_READ(src + k + 8 * lanes, 0);

        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        if (zero_center) {
            v0 = s::Sub(v0, v_mu);
            v1 = s::Sub(v1, v_mu);
            v2 = s::Sub(v2, v_mu);
            v3 = s::Sub(v3, v_mu);
        }

        v0 = s::Mul(v0, v_inv_sigma);
        v1 = s::Mul(v1, v_inv_sigma);
        v2 = s::Mul(v2, v_inv_sigma);
        v3 = s::Mul(v3, v_inv_sigma);

        if (do_clip) {
            v0 = s::Min(s::Max(v0, v_min), v_max);
            v1 = s::Min(s::Max(v1, v_min), v_max);
            v2 = s::Min(s::Max(v2, v_min), v_max);
            v3 = s::Min(s::Max(v3, v_min), v_max);
        }

        s::Store(v0, d, dst + k + 0 * lanes);
        s::Store(v1, d, dst + k + 1 * lanes);
        s::Store(v2, d, dst + k + 2 * lanes);
        s::Store(v3, d, dst + k + 3 * lanes);
    }

    // Single vector tail
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        if (zero_center) v = s::Sub(v, v_mu);
        v = s::Mul(v, v_inv_sigma);
        if (do_clip) v = s::Min(s::Max(v, v_min), v_max);
        s::Store(v, d, dst + k);
    }

    // Scalar tail
    for (; k < len; ++k) {
        T v = src[k];
        if (zero_center) v -= mu;
        v *= inv_sigma;
        if (do_clip) {
            if (v > max_val) v = max_val;
            if (v < -max_val) v = -max_val;
        }
        dst[k] = v;
    }
}

/// @brief Fused copy + scale
template <typename T>
SCL_FORCE_INLINE void fused_scale(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size len,
    T scale
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_scale = s::Set(d, scale);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        s::Store(s::Mul(v0, v_scale), d, dst + k + 0 * lanes);
        s::Store(s::Mul(v1, v_scale), d, dst + k + 1 * lanes);
        s::Store(s::Mul(v2, v_scale), d, dst + k + 2 * lanes);
        s::Store(s::Mul(v3, v_scale), d, dst + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        s::Store(s::Mul(v, v_scale), d, dst + k);
    }

    for (; k < len; ++k) {
        dst[k] = src[k] * scale;
    }
}

/// @brief Simple copy (when scale == 1)
template <typename T>
SCL_FORCE_INLINE void copy_values(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size len
) {
    std::memcpy(dst, src, len * sizeof(T));
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse Operations
// =============================================================================

/// @brief Fused materialize + standardize for MappedCustomSparse
///
/// Single pass: read source, transform, write to output.
/// Saves one memory traversal vs materialize-then-transform.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> stds,
    T max_value,
    bool zero_center
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Allocate output
    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);

    // Copy indptr and indices (structure unchanged)
    std::memcpy(out_indptr.data(), matrix.indptr(), (primary_dim + 1) * sizeof(Index));
    std::memcpy(out_indices.data(), matrix.indices(), nnz * sizeof(Index));

    const bool do_clip = (max_value > T(0));

    // Chunk-based processing
    const Size n_chunks = (static_cast<Size>(primary_dim) + config::CHUNK_SIZE - 1) / config::CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * config::CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(config::CHUNK_SIZE), primary_dim);

        // Prefetch next chunk
        if (chunk_id + 1 < n_chunks) {
            Index next_start = static_cast<Index>((chunk_id + 1) * config::CHUNK_SIZE);
            auto vals_next = scl::primary_values(matrix, next_start);
            SCL_PREFETCH_READ(vals_next.ptr, 0);
        }

        // Parallel within chunk
        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            T sigma = stds[p];
            Index start = out_indptr[p];
            Index end = out_indptr[p + 1];
            Size len = static_cast<Size>(end - start);

            if (len == 0) return;

            auto src_vals = scl::primary_values(matrix, p);
            T* dst_vals = out_data.data() + start;

            if (sigma == T(0)) {
                // Zero-std: just copy
                detail::copy_values(src_vals.ptr, dst_vals, len);
            } else {
                T mu = means[p];
                T inv_sigma = T(1) / sigma;
                detail::fused_standardize(
                    src_vals.ptr, dst_vals, len,
                    mu, inv_sigma, max_value,
                    zero_center, do_clip
                );
            }
        });
    }

    // Build OwnedSparse
    if constexpr (IsCSR) {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            primary_dim, matrix.cols
        );
    } else {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            matrix.rows, primary_dim
        );
    }
}

/// @brief Fused materialize + scale_rows for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");

    kernel::mapped::hint_prefetch(matrix);

    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);

    std::memcpy(out_indptr.data(), matrix.indptr(), (primary_dim + 1) * sizeof(Index));
    std::memcpy(out_indices.data(), matrix.indices(), nnz * sizeof(Index));

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        T scale = scales[p];
        Index start = out_indptr[p];
        Index end = out_indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        T* dst_vals = out_data.data() + start;

        if (scale == T(1)) {
            detail::copy_values(src_vals.ptr, dst_vals, len);
        } else {
            detail::fused_scale(src_vals.ptr, dst_vals, len, scale);
        }
    });

    if constexpr (IsCSR) {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            primary_dim, matrix.cols
        );
    } else {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            matrix.rows, primary_dim
        );
    }
}

// =============================================================================
// SECTION 4: MappedVirtualSparse Operations
// =============================================================================

/// @brief Fused materialize + standardize for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> standardize_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> stds,
    T max_value,
    bool zero_center
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");

    // Compute total nnz
    Index nnz = 0;
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index len = scl::primary_length(matrix, p);
        out_indptr[p + 1] = out_indptr[p] + len;
        nnz += len;
    }

    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));

    const bool do_clip = (max_value > T(0));

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        T sigma = stds[p];
        Index start = out_indptr[p];
        Size len = static_cast<Size>(out_indptr[p + 1] - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        auto src_inds = scl::primary_indices(matrix, p);
        T* dst_vals = out_data.data() + start;
        Index* dst_inds = out_indices.data() + start;

        // Copy indices
        std::memcpy(dst_inds, src_inds.ptr, len * sizeof(Index));

        if (sigma == T(0)) {
            detail::copy_values(src_vals.ptr, dst_vals, len);
        } else {
            T mu = means[p];
            T inv_sigma = T(1) / sigma;
            detail::fused_standardize(
                src_vals.ptr, dst_vals, len,
                mu, inv_sigma, max_value,
                zero_center, do_clip
            );
        }
    });

    Index out_rows = IsCSR ? primary_dim : matrix.rows;
    Index out_cols = IsCSR ? matrix.cols : primary_dim;

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(out_indptr),
        out_rows, out_cols
    );
}

/// @brief Fused materialize + scale_rows for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");

    Index nnz = 0;
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);
    out_indptr[0] = 0;
    for (Index p = 0; p < primary_dim; ++p) {
        Index len = scl::primary_length(matrix, p);
        out_indptr[p + 1] = out_indptr[p] + len;
        nnz += len;
    }

    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        T scale = scales[p];
        Index start = out_indptr[p];
        Size len = static_cast<Size>(out_indptr[p + 1] - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        auto src_inds = scl::primary_indices(matrix, p);
        T* dst_vals = out_data.data() + start;
        Index* dst_inds = out_indices.data() + start;

        std::memcpy(dst_inds, src_inds.ptr, len * sizeof(Index));

        if (scale == T(1)) {
            detail::copy_values(src_vals.ptr, dst_vals, len);
        } else {
            detail::fused_scale(src_vals.ptr, dst_vals, len, scale);
        }
    });

    Index out_rows = IsCSR ? primary_dim : matrix.rows;
    Index out_cols = IsCSR ? matrix.cols : primary_dim;

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(out_indptr),
        out_rows, out_cols
    );
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Unified standardize dispatcher for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> standardize_mapped_dispatch(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> means,
    Array<const typename MatrixT::ValueType> stds,
    typename MatrixT::ValueType max_value,
    bool zero_center
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        return standardize_mapped_custom(matrix, means, stds, max_value, zero_center);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        return standardize_mapped_virtual(matrix, means, stds, max_value, zero_center);
    } else {
        // Generic fallback: materialize then transform
        auto owned = matrix.materialize();
        // Apply standardization in-place on owned data
        const Index primary_dim = IsCSR ? owned.rows : owned.cols;
        const bool do_clip = (max_value > T(0));

        for (Index p = 0; p < primary_dim; ++p) {
            T sigma = stds[p];
            if (sigma == T(0)) continue;

            Index start = owned.indptr[p];
            Index end = owned.indptr[p + 1];
            Size len = static_cast<Size>(end - start);
            if (len == 0) continue;

            T mu = means[p];
            T inv_sigma = T(1) / sigma;
            T* vals = owned.data.data() + start;

            for (Size k = 0; k < len; ++k) {
                T v = vals[k];
                if (zero_center) v -= mu;
                v *= inv_sigma;
                if (do_clip) {
                    if (v > max_value) v = max_value;
                    if (v < -max_value) v = -max_value;
                }
                vals[k] = v;
            }
        }

        return owned;
    }
}

/// @brief Unified scale_rows dispatcher for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> scale_rows_mapped_dispatch(
    const MatrixT& matrix,
    Array<const typename MatrixT::ValueType> scales
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        return scale_rows_mapped_custom(matrix, scales);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        return scale_rows_mapped_virtual(matrix, scales);
    } else {
        auto owned = matrix.materialize();
        const Index primary_dim = IsCSR ? owned.rows : owned.cols;

        for (Index p = 0; p < primary_dim; ++p) {
            T scale = scales[p];
            if (scale == T(1)) continue;

            Index start = owned.indptr[p];
            Index end = owned.indptr[p + 1];
            T* vals = owned.data.data() + start;

            for (Index k = start; k < end; ++k) {
                vals[k - start] *= scale;
            }
        }

        return owned;
    }
}

} // namespace scl::kernel::scale::mapped
