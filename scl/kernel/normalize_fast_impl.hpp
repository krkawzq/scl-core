#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/normalize_mapped_impl.hpp"

#include <algorithm>

// =============================================================================
/// @file normalize_fast_impl.hpp
/// @brief Extreme Performance Normalization Operations
///
/// ## Key Optimizations
///
/// 1. 4-Way SIMD Unrolling
///    - Process 4 vectors per iteration
///    - Better instruction-level parallelism
///
/// 2. Fused Operations
///    - Compute sums and scales in single pass
///    - Reduce memory bandwidth
///
/// 3. SIMD Threshold Detection
///    - Vectorized comparison for highly_expressed
///
/// 4. Thread-Local Atomic Writes
///    - Lock-free mask updates for detect_highly_expressed
///
/// Performance Target: 3-5x faster than generic
// =============================================================================

namespace scl::kernel::normalize::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD sum accumulation (4-way unrolled)
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

/// @brief SIMD scaling (4-way unrolled)
template <typename T>
SCL_FORCE_INLINE void scale_simd(T* SCL_RESTRICT vals, Size len, T scale) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_scale = s::Set(d, scale);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        s::Store(s::Mul(v0, v_scale), d, vals + k + 0 * lanes);
        s::Store(s::Mul(v1, v_scale), d, vals + k + 1 * lanes);
        s::Store(s::Mul(v2, v_scale), d, vals + k + 2 * lanes);
        s::Store(s::Mul(v3, v_scale), d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        s::Store(s::Mul(v, v_scale), d, vals + k);
    }

    for (; k < len; ++k) {
        vals[k] *= scale;
    }
}

/// @brief SIMD masked sum (exclude masked indices)
template <typename T>
SCL_FORCE_INLINE T sum_masked_simd(
    const T* SCL_RESTRICT vals,
    const Index* SCL_RESTRICT indices,
    Size len,
    const Byte* SCL_RESTRICT mask
) {
    T sum = T(0);

    // Note: Cannot vectorize well due to indirect mask access
    // Use 4-way scalar unrolling instead
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
// SECTION 3: CustomSparse Fast Path
// =============================================================================

/// @brief Compute row sums for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void compute_row_sums_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        output[p] = (len > 0) ? detail::sum_simd(matrix.data + start, len) : T(0);
    });
}

/// @brief Scale rows for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void scale_primary_custom(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Real> scales
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(primary_dim), "Scales dim mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == Real(1)) return;

        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len > 0) {
            detail::scale_simd(matrix.data + start, len, static_cast<T>(scale));
        }
    });
}

/// @brief Masked sum for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void primary_sums_masked_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) {
            output[p] = Real(0);
            return;
        }

        output[p] = detail::sum_masked_simd(
            matrix.data + start,
            matrix.indices + start,
            len,
            mask.ptr
        );
    });
}

/// @brief Detect highly expressed for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void detect_highly_expressed_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);

    // Zero initialize mask
    scl::memory::zero(out_mask);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real total = row_sums[p];
        if (total <= Real(0)) return;

        Real threshold = total * max_fraction;

        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];

        const T* vals = matrix.data + start;
        const Index* inds = matrix.indices + start;
        Size len = static_cast<Size>(end - start);

        // 4-way unrolled comparison
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            if (static_cast<Real>(vals[k + 0]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 0]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 1]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 1]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 2]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 2]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 3]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 3]], 1, __ATOMIC_RELAXED);
            }
        }

        for (; k < len; ++k) {
            if (static_cast<Real>(vals[k]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k]], 1, __ATOMIC_RELAXED);
            }
        }
    });
}

// =============================================================================
// SECTION 4: VirtualSparse Fast Path
// =============================================================================

/// @brief Compute row sums for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void compute_row_sums_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);

        output[p] = (len > 0) ? detail::sum_simd(vals, static_cast<Size>(len)) : T(0);
    });
}

/// @brief Scale rows for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void scale_primary_virtual(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> scales
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(primary_dim), "Scales dim mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == Real(1)) return;

        Index len = matrix.lengths[p];
        if (len == 0) return;

        T* vals = static_cast<T*>(matrix.data_ptrs[p]);
        detail::scale_simd(vals, static_cast<Size>(len), static_cast<T>(scale));
    });
}

/// @brief Masked sum for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void primary_sums_masked_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];

        if (len == 0) {
            output[p] = Real(0);
            return;
        }

        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* inds = static_cast<const Index*>(matrix.indices_ptrs[p]);

        output[p] = detail::sum_masked_simd(vals, inds, static_cast<Size>(len), mask.ptr);
    });
}

/// @brief Detect highly expressed for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void detect_highly_expressed_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);

    scl::memory::zero(out_mask);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real total = row_sums[p];
        if (total <= Real(0)) return;

        Real threshold = total * max_fraction;

        Index len = matrix.lengths[p];
        if (len == 0) return;

        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* inds = static_cast<const Index*>(matrix.indices_ptrs[p]);

        Size k = 0;
        for (; k + 4 <= static_cast<Size>(len); k += 4) {
            if (static_cast<Real>(vals[k + 0]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 0]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 1]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 1]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 2]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 2]], 1, __ATOMIC_RELAXED);
            }
            if (static_cast<Real>(vals[k + 3]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k + 3]], 1, __ATOMIC_RELAXED);
            }
        }

        for (; k < static_cast<Size>(len); ++k) {
            if (static_cast<Real>(vals[k]) > threshold) {
                __atomic_store_n(&out_mask.ptr[inds[k]], 1, __ATOMIC_RELAXED);
            }
        }
    });
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Auto-dispatch row sums
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void compute_row_sums_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::normalize::mapped::compute_row_sums_mapped_dispatch<MatrixT, IsCSR>(matrix, output);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        compute_row_sums_custom(matrix, output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        compute_row_sums_virtual(matrix, output);
    }
}

/// @brief Auto-dispatch scale primary
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void scale_primary_fast(
    MatrixT& matrix,
    Array<const Real> scales
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        scale_primary_custom(matrix, scales);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        scale_primary_virtual(matrix, scales);
    }
    // Note: Mapped matrices require materialize, handled separately
}

/// @brief Auto-dispatch masked sums
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void primary_sums_masked_fast(
    const MatrixT& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::normalize::mapped::primary_sums_masked_mapped_dispatch<MatrixT, IsCSR>(matrix, mask, output);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        primary_sums_masked_custom(matrix, mask, output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        primary_sums_masked_virtual(matrix, mask, output);
    }
}

/// @brief Auto-dispatch highly expressed detection
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void detect_highly_expressed_fast(
    const MatrixT& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::normalize::mapped::detect_highly_expressed_mapped_dispatch<MatrixT, IsCSR>(
            matrix, row_sums, max_fraction, out_mask);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        detect_highly_expressed_custom(matrix, row_sums, max_fraction, out_mask);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        detect_highly_expressed_virtual(matrix, row_sums, max_fraction, out_mask);
    }
}

} // namespace scl::kernel::normalize::fast
