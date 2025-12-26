#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <limits>

// =============================================================================
/// @file softmax_mapped_impl.hpp
/// @brief Extreme Performance Softmax for Memory-Mapped Sparse Matrices
///
/// ## Key Insight
///
/// Mapped data is READ-ONLY. For in-place operations we must materialize.
/// Optimization: FUSE materialization with softmax transformation.
///
/// ## Key Optimizations
///
/// 1. Fused Materialize + Transform
///    - Copy indices once
///    - Apply softmax while copying values
///    - Single memory traversal for values
///
/// 2. Two-Pass Per-Row Algorithm
///    - Pass 1: Read source, find max (SIMD)
///    - Pass 2: Read source again, compute exp-sum, write normalized
///    - Avoids intermediate storage
///
/// 3. Chunk-Based Processing
///    - Process rows in L2-friendly chunks
///    - Prefetch next chunk
///
/// 4. 4-Way Unrolled SIMD
///    - Fused exp + sum operations
///    - Optimized normalize pass
///
/// Performance: Near in-memory performance with minimal I/O overhead
// =============================================================================

namespace scl::kernel::softmax::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 256;
}

// =============================================================================
// SECTION 2: Softmax Helpers
// =============================================================================

namespace detail {

/// @brief Find max in source array (SIMD)
template <typename T>
SCL_FORCE_INLINE T find_max(const T* SCL_RESTRICT src, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_max = s::Set(d, -std::numeric_limits<T>::infinity());

    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        v_max = s::Max(v_max, v);
    }

    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));

    for (; k < len; ++k) {
        if (src[k] > max_val) max_val = src[k];
    }

    return max_val;
}

/// @brief Fused copy + exp + sum + normalize
///
/// Single read of source, writes normalized softmax to destination.
template <typename T>
SCL_FORCE_INLINE void fused_softmax(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size len,
    T max_val
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);

    // First: compute exp and sum
    auto v_sum = s::Zero(d);
    Size k = 0;

    // 4-way unrolled
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        v0 = s::Exp(d, s::Sub(v0, v_max));
        v1 = s::Exp(d, s::Sub(v1, v_max));
        v2 = s::Exp(d, s::Sub(v2, v_max));
        v3 = s::Exp(d, s::Sub(v3, v_max));

        s::Store(v0, d, dst + k + 0 * lanes);
        s::Store(v1, d, dst + k + 1 * lanes);
        s::Store(v2, d, dst + k + 2 * lanes);
        s::Store(v3, d, dst + k + 3 * lanes);

        v_sum = s::Add(v_sum, s::Add(s::Add(v0, v1), s::Add(v2, v3)));
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        v = s::Exp(d, s::Sub(v, v_max));
        s::Store(v, d, dst + k);
        v_sum = s::Add(v_sum, v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        T v = std::exp(src[k] - max_val);
        dst[k] = v;
        sum += v;
    }

    // Normalize
    if (sum > T(0)) {
        T inv_sum = T(1) / sum;
        const auto v_inv_sum = s::Set(d, inv_sum);

        k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, dst + k + 0 * lanes);
            auto v1 = s::Load(d, dst + k + 1 * lanes);
            auto v2 = s::Load(d, dst + k + 2 * lanes);
            auto v3 = s::Load(d, dst + k + 3 * lanes);

            s::Store(s::Mul(v0, v_inv_sum), d, dst + k + 0 * lanes);
            s::Store(s::Mul(v1, v_inv_sum), d, dst + k + 1 * lanes);
            s::Store(s::Mul(v2, v_inv_sum), d, dst + k + 2 * lanes);
            s::Store(s::Mul(v3, v_inv_sum), d, dst + k + 3 * lanes);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, dst + k);
            s::Store(s::Mul(v, v_inv_sum), d, dst + k);
        }

        for (; k < len; ++k) {
            dst[k] *= inv_sum;
        }
    }
}

/// @brief Fused copy + log_softmax
template <typename T>
SCL_FORCE_INLINE void fused_log_softmax(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size len,
    T max_val
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_max = s::Set(d, max_val);

    // Compute sum(exp(x - max))
    auto v_sum = s::Zero(d);
    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        v_sum = s::Add(v_sum, s::Exp(d, s::Sub(v, v_max)));
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < len; ++k) {
        sum += std::exp(src[k] - max_val);
    }

    // log_softmax = x - max - log(sum)
    T log_sum = std::log(sum);
    T offset = max_val + log_sum;
    const auto v_offset = s::Set(d, offset);

    k = 0;
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, src + k + 0 * lanes);
        auto v1 = s::Load(d, src + k + 1 * lanes);
        auto v2 = s::Load(d, src + k + 2 * lanes);
        auto v3 = s::Load(d, src + k + 3 * lanes);

        s::Store(s::Sub(v0, v_offset), d, dst + k + 0 * lanes);
        s::Store(s::Sub(v1, v_offset), d, dst + k + 1 * lanes);
        s::Store(s::Sub(v2, v_offset), d, dst + k + 2 * lanes);
        s::Store(s::Sub(v3, v_offset), d, dst + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, src + k);
        s::Store(s::Sub(v, v_offset), d, dst + k);
    }

    for (; k < len; ++k) {
        dst[k] = src[k] - offset;
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse Operations
// =============================================================================

/// @brief Fused materialize + softmax for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Allocate output
    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);

    // Copy structure
    std::memcpy(out_indptr.data(), matrix.indptr(), (primary_dim + 1) * sizeof(Index));
    std::memcpy(out_indices.data(), matrix.indices(), nnz * sizeof(Index));

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

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            Index start = out_indptr[p];
            Index end = out_indptr[p + 1];
            Size len = static_cast<Size>(end - start);

            if (len == 0) return;

            auto src_vals = scl::primary_values(matrix, p);
            T* dst_vals = out_data.data() + start;

            // Find max
            T max_val = detail::find_max(src_vals.ptr, len);

            // Fused softmax
            detail::fused_softmax(src_vals.ptr, dst_vals, len, max_val);
        });
    }

    // Build OwnedSparse
    if constexpr (IsCSR) {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            primary_dim, matrix.cols()
        );
    } else {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            matrix.rows(), primary_dim
        );
    }
}

/// @brief Fused materialize + log_softmax for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log_softmax_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    kernel::mapped::hint_prefetch(matrix);

    std::vector<T> out_data(static_cast<size_t>(nnz));
    std::vector<Index> out_indices(static_cast<size_t>(nnz));
    std::vector<Index> out_indptr(static_cast<size_t>(primary_dim) + 1);

    std::memcpy(out_indptr.data(), matrix.indptr(), (primary_dim + 1) * sizeof(Index));
    std::memcpy(out_indices.data(), matrix.indices(), nnz * sizeof(Index));

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        Index start = out_indptr[p];
        Index end = out_indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        T* dst_vals = out_data.data() + start;

        T max_val = detail::find_max(src_vals.ptr, len);
        detail::fused_log_softmax(src_vals.ptr, dst_vals, len, max_val);
    });

    if constexpr (IsCSR) {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            primary_dim, matrix.cols()
        );
    } else {
        return scl::io::OwnedSparse<T, IsCSR>(
            std::move(out_data),
            std::move(out_indices),
            std::move(out_indptr),
            matrix.rows(), primary_dim
        );
    }
}

// =============================================================================
// SECTION 4: MappedVirtualSparse Operations
// =============================================================================

/// @brief Fused materialize + softmax for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    const Index primary_dim = scl::primary_size(matrix);

    // Compute total nnz and indptr
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
        Index start = out_indptr[p];
        Size len = static_cast<Size>(out_indptr[p + 1] - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        auto src_inds = scl::primary_indices(matrix, p);
        T* dst_vals = out_data.data() + start;
        Index* dst_inds = out_indices.data() + start;

        // Copy indices
        std::memcpy(dst_inds, src_inds.ptr, len * sizeof(Index));

        // Find max and apply softmax
        T max_val = detail::find_max(src_vals.ptr, len);
        detail::fused_softmax(src_vals.ptr, dst_vals, len, max_val);
    });

    Index out_rows = IsCSR ? primary_dim : matrix.rows();
    Index out_cols = IsCSR ? matrix.cols() : primary_dim;

    return scl::io::OwnedSparse<T, IsCSR>(
        std::move(out_data),
        std::move(out_indices),
        std::move(out_indptr),
        out_rows, out_cols
    );
}

/// @brief Fused materialize + log_softmax for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log_softmax_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    const Index primary_dim = scl::primary_size(matrix);

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
        Index start = out_indptr[p];
        Size len = static_cast<Size>(out_indptr[p + 1] - start);

        if (len == 0) return;

        auto src_vals = scl::primary_values(matrix, p);
        auto src_inds = scl::primary_indices(matrix, p);
        T* dst_vals = out_data.data() + start;
        Index* dst_inds = out_indices.data() + start;

        std::memcpy(dst_inds, src_inds.ptr, len * sizeof(Index));

        T max_val = detail::find_max(src_vals.ptr, len);
        detail::fused_log_softmax(src_vals.ptr, dst_vals, len, max_val);
    });

    Index out_rows = IsCSR ? primary_dim : matrix.rows();
    Index out_cols = IsCSR ? matrix.cols() : primary_dim;

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

/// @brief Unified softmax dispatcher for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> softmax_mapped_dispatch(
    const MatrixT& matrix
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        return softmax_mapped_custom(matrix);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        return softmax_mapped_virtual(matrix);
    } else {
        // Generic fallback
        auto owned = matrix.materialize();
        const Index primary_dim = IsCSR ? owned.rows : owned.cols;

        for (Index p = 0; p < primary_dim; ++p) {
            Index start = owned.indptr[p];
            Index end = owned.indptr[p + 1];
            Size len = static_cast<Size>(end - start);
            if (len == 0) continue;

            T* vals = owned.data.data() + start;

            // Find max
            T max_val = vals[0];
            for (Size k = 1; k < len; ++k) {
                if (vals[k] > max_val) max_val = vals[k];
            }

            // Exp + sum
            T sum = T(0);
            for (Size k = 0; k < len; ++k) {
                T v = std::exp(vals[k] - max_val);
                vals[k] = v;
                sum += v;
            }

            // Normalize
            if (sum > T(0)) {
                T inv_sum = T(1) / sum;
                for (Size k = 0; k < len; ++k) {
                    vals[k] *= inv_sum;
                }
            }
        }

        return owned;
    }
}

/// @brief Unified log_softmax dispatcher for mapped matrices
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> log_softmax_mapped_dispatch(
    const MatrixT& matrix
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        return log_softmax_mapped_custom(matrix);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        return log_softmax_mapped_virtual(matrix);
    } else {
        auto owned = matrix.materialize();
        const Index primary_dim = IsCSR ? owned.rows : owned.cols;

        for (Index p = 0; p < primary_dim; ++p) {
            Index start = owned.indptr[p];
            Index end = owned.indptr[p + 1];
            Size len = static_cast<Size>(end - start);
            if (len == 0) continue;

            T* vals = owned.data.data() + start;

            // Find max
            T max_val = vals[0];
            for (Size k = 1; k < len; ++k) {
                if (vals[k] > max_val) max_val = vals[k];
            }

            // Sum(exp)
            T sum = T(0);
            for (Size k = 0; k < len; ++k) {
                sum += std::exp(vals[k] - max_val);
            }

            // log_softmax = x - max - log(sum)
            T offset = max_val + std::log(sum);
            for (Size k = 0; k < len; ++k) {
                vals[k] -= offset;
            }
        }

        return owned;
    }
}

} // namespace scl::kernel::softmax::mapped
