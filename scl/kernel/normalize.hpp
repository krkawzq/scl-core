#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>
#include <atomic>

// =============================================================================
// FILE: scl/kernel/normalize.hpp
// BRIEF: Normalization operations with SIMD optimization
// =============================================================================

namespace scl::kernel::normalize {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

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

template <typename T>
SCL_FORCE_INLINE SCL_HOT T sum_masked_simd(
    const T* SCL_RESTRICT vals,
    const Index* SCL_RESTRICT indices,
    Size len,
    const Byte* SCL_RESTRICT mask
) {
    // Multi-accumulator pattern to hide latency
    T sum0 = T(0), sum1 = T(0);

    Size k = 0;

    for (; k + 8 <= len; k += 8) {
        // Prefetch ahead for indirect mask access
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&mask[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&mask[indices[k + config::PREFETCH_DISTANCE + 4]], 0);
        }

        T v0 = (mask[indices[k + 0]] == 0) ? vals[k + 0] : T(0);
        T v1 = (mask[indices[k + 1]] == 0) ? vals[k + 1] : T(0);
        T v2 = (mask[indices[k + 2]] == 0) ? vals[k + 2] : T(0);
        T v3 = (mask[indices[k + 3]] == 0) ? vals[k + 3] : T(0);
        T v4 = (mask[indices[k + 4]] == 0) ? vals[k + 4] : T(0);
        T v5 = (mask[indices[k + 5]] == 0) ? vals[k + 5] : T(0);
        T v6 = (mask[indices[k + 6]] == 0) ? vals[k + 6] : T(0);
        T v7 = (mask[indices[k + 7]] == 0) ? vals[k + 7] : T(0);

        sum0 += v0 + v1 + v2 + v3;
        sum1 += v4 + v5 + v6 + v7;
    }

    T sum = sum0 + sum1;

    // Handle remaining 4-element chunk
    if (k + 4 <= len) {
        T v0 = (mask[indices[k + 0]] == 0) ? vals[k + 0] : T(0);
        T v1 = (mask[indices[k + 1]] == 0) ? vals[k + 1] : T(0);
        T v2 = (mask[indices[k + 2]] == 0) ? vals[k + 2] : T(0);
        T v3 = (mask[indices[k + 3]] == 0) ? vals[k + 3] : T(0);
        sum += v0 + v1 + v2 + v3;
        k += 4;
    }

    // Scalar cleanup
    for (; k < len; ++k) {
        if (mask[indices[k]] == 0) {
            sum += vals[k];
        }
    }

    return sum;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) {
            output[p] = T(0);
            return;
        }

        auto values = matrix.primary_values_unsafe(idx);
        output[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz));
    });
}

template <typename T, bool IsCSR>
void scale_primary(
    Sparse<T, IsCSR>& matrix,
    Array<const Real> scales
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(scales.len >= static_cast<Size>(primary_dim), "Scales dim mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == Real(1)) return;

        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        if (len == 0) return;

        auto values = matrix.primary_values_unsafe(idx);
        detail::scale_simd(values.ptr, static_cast<Size>(len), static_cast<T>(scale));
    });
}

template <typename T, bool IsCSR>
void primary_sums_masked(
    const Sparse<T, IsCSR>& matrix,
    Array<const Byte> mask,
    Array<Real> output
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(output.len >= static_cast<Size>(primary_dim), "Output size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) {
            output[p] = Real(0);
            return;
        }

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        output[p] = detail::sum_masked_simd(
            values.ptr,
            indices.ptr,
            len_sz,
            mask.ptr
        );
    });
}

template <typename T, bool IsCSR>
void detect_highly_expressed(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> row_sums,
    Real max_fraction,
    Array<Byte> out_mask
) {
    const Index primary_dim = matrix.primary_dim();

    scl::memory::zero(out_mask);

    // Cast to atomic for thread-safe writes
    std::atomic<Byte>* atomic_mask = reinterpret_cast<std::atomic<Byte>*>(out_mask.ptr);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real total = row_sums[p];
        if (SCL_UNLIKELY(total <= Real(0))) return;

        Real threshold = total * max_fraction;

        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) return;

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            // Prefetch ahead for indirect mask write
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_WRITE(&out_mask.ptr[indices.ptr[k + config::PREFETCH_DISTANCE]], 0);
            }

            if (static_cast<Real>(values.ptr[k + 0]) > threshold) {
                atomic_mask[indices.ptr[k + 0]].store(1, std::memory_order_relaxed);
            }
            if (static_cast<Real>(values.ptr[k + 1]) > threshold) {
                atomic_mask[indices.ptr[k + 1]].store(1, std::memory_order_relaxed);
            }
            if (static_cast<Real>(values.ptr[k + 2]) > threshold) {
                atomic_mask[indices.ptr[k + 2]].store(1, std::memory_order_relaxed);
            }
            if (static_cast<Real>(values.ptr[k + 3]) > threshold) {
                atomic_mask[indices.ptr[k + 3]].store(1, std::memory_order_relaxed);
            }
        }

        for (; k < len_sz; ++k) {
            if (static_cast<Real>(values.ptr[k]) > threshold) {
                atomic_mask[indices.ptr[k]].store(1, std::memory_order_relaxed);
            }
        }
    });
}

} // namespace scl::kernel::normalize

