#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/scale.hpp
// BRIEF: Scaling operations with SIMD optimization
// =============================================================================

namespace scl::kernel::scale {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size SHORT_THRESHOLD = 16;
    constexpr Size MEDIUM_THRESHOLD = 128;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// SIMD Helpers
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void standardize_short(
    T* SCL_RESTRICT vals,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    for (Size k = 0; k < len; ++k) {
        T v = vals[k];
        if (zero_center) v -= mu;
        v *= inv_sigma;
        if (do_clip) {
            if (v > max_val) v = max_val;
            if (v < -max_val) v = -max_val;
        }
        vals[k] = v;
    }
}

template <typename T>
SCL_FORCE_INLINE void standardize_medium(
    T* SCL_RESTRICT vals,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_mu = s::Set(d, mu);
    const auto v_inv_sigma = s::Set(d, inv_sigma);
    const auto v_max = s::Set(d, max_val);
    const auto v_min = s::Set(d, -max_val);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

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

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        if (zero_center) v = s::Sub(v, v_mu);
        v = s::Mul(v, v_inv_sigma);
        if (do_clip) v = s::Min(s::Max(v, v_min), v_max);
        s::Store(v, d, vals + k);
    }

    for (; k < len; ++k) {
        T v = vals[k];
        if (zero_center) v -= mu;
        v *= inv_sigma;
        if (do_clip) {
            if (v > max_val) v = max_val;
            if (v < -max_val) v = -max_val;
        }
        vals[k] = v;
    }
}

template <typename T>
SCL_FORCE_INLINE void standardize_long(
    T* SCL_RESTRICT vals,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_mu = s::Set(d, mu);
    const auto v_inv_sigma = s::Set(d, inv_sigma);
    const auto v_max = s::Set(d, max_val);
    const auto v_min = s::Set(d, -max_val);

    Size k = 0;

    for (; k + 8 * lanes <= len; k += 8 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        if (zero_center) {
            v0 = s::Sub(v0, v_mu);
            v1 = s::Sub(v1, v_mu);
            v2 = s::Sub(v2, v_mu);
            v3 = s::Sub(v3, v_mu);
            v4 = s::Sub(v4, v_mu);
            v5 = s::Sub(v5, v_mu);
            v6 = s::Sub(v6, v_mu);
            v7 = s::Sub(v7, v_mu);
        }

        v0 = s::Mul(v0, v_inv_sigma);
        v1 = s::Mul(v1, v_inv_sigma);
        v2 = s::Mul(v2, v_inv_sigma);
        v3 = s::Mul(v3, v_inv_sigma);
        v4 = s::Mul(v4, v_inv_sigma);
        v5 = s::Mul(v5, v_inv_sigma);
        v6 = s::Mul(v6, v_inv_sigma);
        v7 = s::Mul(v7, v_inv_sigma);

        if (do_clip) {
            v0 = s::Min(s::Max(v0, v_min), v_max);
            v1 = s::Min(s::Max(v1, v_min), v_max);
            v2 = s::Min(s::Max(v2, v_min), v_max);
            v3 = s::Min(s::Max(v3, v_min), v_max);
            v4 = s::Min(s::Max(v4, v_min), v_max);
            v5 = s::Min(s::Max(v5, v_min), v_max);
            v6 = s::Min(s::Max(v6, v_min), v_max);
            v7 = s::Min(s::Max(v7, v_min), v_max);
        }

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
        s::Store(v4, d, vals + k + 4 * lanes);
        s::Store(v5, d, vals + k + 5 * lanes);
        s::Store(v6, d, vals + k + 6 * lanes);
        s::Store(v7, d, vals + k + 7 * lanes);
    }

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

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

        s::Store(v0, d, vals + k + 0 * lanes);
        s::Store(v1, d, vals + k + 1 * lanes);
        s::Store(v2, d, vals + k + 2 * lanes);
        s::Store(v3, d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        if (zero_center) v = s::Sub(v, v_mu);
        v = s::Mul(v, v_inv_sigma);
        if (do_clip) v = s::Min(s::Max(v, v_min), v_max);
        s::Store(v, d, vals + k);
    }

    for (; k < len; ++k) {
        T v = vals[k];
        if (zero_center) v -= mu;
        v *= inv_sigma;
        if (do_clip) {
            if (v > max_val) v = max_val;
            if (v < -max_val) v = -max_val;
        }
        vals[k] = v;
    }
}

template <typename T>
SCL_FORCE_INLINE void standardize_adaptive(
    T* SCL_RESTRICT vals,
    Size len,
    T mu,
    T inv_sigma,
    T max_val,
    bool zero_center,
    bool do_clip
) {
    if (len < config::SHORT_THRESHOLD) {
        standardize_short(vals, len, mu, inv_sigma, max_val, zero_center, do_clip);
    } else if (len < config::MEDIUM_THRESHOLD) {
        standardize_medium(vals, len, mu, inv_sigma, max_val, zero_center, do_clip);
    } else {
        standardize_long(vals, len, mu, inv_sigma, max_val, zero_center, do_clip);
    }
}

template <typename T>
SCL_FORCE_INLINE void scale_values(
    T* SCL_RESTRICT vals,
    Size len,
    T scale
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_scale = s::Set(d, scale);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

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
SCL_FORCE_INLINE void shift_values(
    T* SCL_RESTRICT vals,
    Size len,
    T offset
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_offset = s::Set(d, offset);

    Size k = 0;

    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        s::Store(s::Add(v0, v_offset), d, vals + k + 0 * lanes);
        s::Store(s::Add(v1, v_offset), d, vals + k + 1 * lanes);
        s::Store(s::Add(v2, v_offset), d, vals + k + 2 * lanes);
        s::Store(s::Add(v3, v_offset), d, vals + k + 3 * lanes);
    }

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        s::Store(s::Add(v, v_offset), d, vals + k);
    }

    for (; k < len; ++k) {
        vals[k] += offset;
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void standardize(
    Sparse<T, IsCSR>& matrix,
    Array<const T> means,
    Array<const T> stds,
    T max_value,
    bool zero_center
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");

    const bool do_clip = (max_value > T(0));

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        T sigma = stds[p];
        if (sigma == T(0)) return;

        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        if (len == 0) return;

        T mu = means[p];
        T inv_sigma = T(1) / sigma;
        auto values = matrix.primary_values_unsafe(idx);

        detail::standardize_adaptive(
            values.ptr, static_cast<Size>(len),
            mu, inv_sigma, max_value,
            zero_center, do_clip
        );
    });
}

template <typename T, bool IsCSR>
void scale_rows(
    Sparse<T, IsCSR>& matrix,
    Array<const T> scales
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        T scale = scales[p];
        if (scale == T(1)) return;

        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        if (len == 0) return;

        auto values = matrix.primary_values_unsafe(idx);
        detail::scale_values(values.ptr, static_cast<Size>(len), scale);
    });
}

template <typename T, bool IsCSR>
void shift_rows(
    Sparse<T, IsCSR>& matrix,
    Array<const T> offsets
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(offsets.len == static_cast<Size>(primary_dim), "Offsets dim mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        T offset = offsets[p];
        if (offset == T(0)) return;

        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        if (len == 0) return;

        auto values = matrix.primary_values_unsafe(idx);
        detail::shift_values(values.ptr, static_cast<Size>(len), offset);
    });
}

} // namespace scl::kernel::scale

