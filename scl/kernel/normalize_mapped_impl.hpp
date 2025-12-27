#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>

// =============================================================================
/// @file normalize_mapped_impl.hpp
/// @brief Normalization for Memory-Mapped Sparse Matrices
///
/// ## Design
///
/// Read-only operations (row_sums, scales): Stream directly from mapped data
/// Write operations (scale_rows): Materialize first, then apply
///
/// ## Key Optimizations
///
/// 1. SIMD Sum Accumulation (4-way unrolled)
/// 2. Fused Copy + Scale (for write operations)
/// 3. Full Parallelization (no serial chunks)
/// 4. Prefetch Hints for Page Cache
// =============================================================================

namespace scl::kernel::normalize::mapped {

// =============================================================================
// SECTION 1: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD sum (4-way unrolled)
template <typename T>
SCL_FORCE_INLINE T sum_simd(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;

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

    return sum;
}

/// @brief Fused copy + scale
template <typename T>
SCL_FORCE_INLINE void copy_scale_simd(
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

/// @brief Masked sum
template <typename T>
SCL_FORCE_INLINE T sum_masked_simd(
    const T* SCL_RESTRICT vals,
    const Index* SCL_RESTRICT indices,
    Size len,
    const Byte* SCL_RESTRICT mask
) {
    T sum = T(0);

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        T v0 = (mask[indices[k + 0]] == 0) ? vals[k + 0] : T(0);
        T v1 = (mask[indices[k + 1]] == 0) ? vals[k + 1] : T(0);
        T v2 = (mask[indices[k + 2]] == 0) ? vals[k + 2] : T(0);
        T v3 = (mask[indices[k + 3]] == 0) ? vals[k + 3] : T(0);
        sum += v0 + v1 + v2 + v3;
    }

    for (; k < len; ++k) {
        if (mask[indices[k]] == 0) {
            sum += vals[k];
        }
    }

    return sum;
}

} // namespace detail

// =============================================================================
// SECTION 2: MappedCustomSparse Read-Only Operations
// =============================================================================

/// @brief Compute row sums (streaming)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_row_sums_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n_primary), "Output size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        output[p] = (values.len > 0) ? detail::sum_simd(values.ptr, values.len) : T(0);
    });
}

/// @brief Compute normalization scales
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_normalization_scales_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> scales,
    T target_sum = T(1)
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(n_primary), "Scales size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        T sum = (values.len > 0) ? detail::sum_simd(values.ptr, values.len) : T(0);
        scales[p] = (sum != T(0)) ? (target_sum / sum) : T(0);
    });
}

/// @brief Masked sum
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void primary_sums_masked_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n_primary), "Output size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));

        if (values.len == 0) {
            output[p] = Real(0);
            return;
        }

        output[p] = detail::sum_masked_simd(values.ptr, indices.ptr, values.len, mask.ptr);
    });
}

/// @brief Detect highly expressed
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void detect_highly_expressed_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index n_primary = scl::primary_size(matrix);

    scl::memory::zero(out_mask);

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Real total = row_sums[p];
        if (total <= Real(0)) return;

        Real threshold = total * max_fraction;

        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));

        for (Size k = 0; k < values.len; ++k) {
            if (static_cast<Real>(values[k]) > threshold) {
                __atomic_store_n(&out_mask.ptr[indices[k]], 1, __ATOMIC_RELAXED);
            }
        }
    });
}

// =============================================================================
// SECTION 3: MappedCustomSparse Write Operations
// =============================================================================

/// @brief Scale rows - returns materialized result with fused copy+scale
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index n_primary = scl::primary_size(matrix);
    const Index nnz = matrix.nnz();

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(n_primary), "Scales dim mismatch");

    // Allocate owned storage
    scl::io::OwnedSparse<T, IsCSR> owned(matrix.rows, matrix.cols, nnz);

    // Copy structure
    std::copy(matrix.indptr(), matrix.indptr() + n_primary + 1, owned.indptr.begin());
    std::copy(matrix.indices(), matrix.indices() + nnz, owned.indices.begin());

    kernel::mapped::hint_prefetch(matrix);

    // Fused copy + scale
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr()[p];
        Index end = matrix.indptr()[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        T scale = scales[p];
        const T* src = matrix.data() + start;
        T* dst = owned.data.data() + start;

        if (scale == T(1)) {
            std::copy(src, src + len, dst);
        } else {
            detail::copy_scale_simd(src, dst, len, scale);
        }
    });

    return owned;
}

/// @brief Normalize rows - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> normalize_rows_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    T target_sum = T(1)
) {
    const Index n_primary = scl::primary_size(matrix);

    // Compute scales
    std::vector<T> scales(n_primary);
    Array<T> scales_arr(scales.data(), scales.size());
    compute_normalization_scales_mapped(matrix, scales_arr, target_sum);

    // Apply fused copy + scale
    return scale_rows_mapped(matrix, Array<const T>(scales.data(), scales.size()));
}

// =============================================================================
// SECTION 4: MappedVirtualSparse
// =============================================================================

/// @brief Compute row sums for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_row_sums_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));

        output[p] = (values.len > 0) ? detail::sum_simd(values.ptr, values.len) : T(0);
    });
}

/// @brief Masked sum for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void primary_sums_masked_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(n_primary), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));

        if (values.len == 0) {
            output[p] = Real(0);
            return;
        }

        output[p] = detail::sum_masked_simd(values.ptr, indices.ptr, values.len, mask.ptr);
    });
}

/// @brief Detect highly expressed for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void detect_highly_expressed_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index n_primary = scl::primary_size(matrix);

    scl::memory::zero(out_mask);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Real total = row_sums[p];
        if (total <= Real(0)) return;

        Real threshold = total * max_fraction;

        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));

        for (Size k = 0; k < values.len; ++k) {
            if (static_cast<Real>(values[k]) > threshold) {
                __atomic_store_n(&out_mask.ptr[indices[k]], 1, __ATOMIC_RELAXED);
            }
        }
    });
}

/// @brief Scale rows for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> scale_rows_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        T scale = scales[p];
        if (scale == T(1)) return;

        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        T* vals = owned.data.data() + start;

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        const auto v_scale = s::Set(d, scale);

        Size k = 0;
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_scale), d, vals + k);
        }

        for (; k < len; ++k) {
            vals[k] *= scale;
        }
    });

    return owned;
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch row sums
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_row_sums_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    compute_row_sums_mapped(matrix, output);
}

/// @brief Auto-dispatch masked sums
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void primary_sums_masked_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    primary_sums_masked_mapped(matrix, mask, output);
}

/// @brief Auto-dispatch highly expressed detection
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void detect_highly_expressed_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    detect_highly_expressed_mapped(matrix, row_sums, max_fraction, out_mask);
}

} // namespace scl::kernel::normalize::mapped
