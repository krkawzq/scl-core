#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file softmax_mapped_impl.hpp
/// @brief Mapped Backend Softmax Operations
///
/// Key insight: Mapped data is READ-ONLY (memory-mapped files).
/// For in-place softmax, we must:
/// 1. Materialize to OwnedSparse
/// 2. Apply softmax on owned data
/// 3. Return OwnedSparse (caller takes ownership)
// =============================================================================

namespace scl::kernel::softmax::mapped {

// =============================================================================
// MappedCustomSparse Softmax
// =============================================================================

/// @brief Apply softmax to mapped matrix - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    // Materialize to owned storage
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Apply softmax in-place on owned data
    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        // Pass 1: Find max
        T max_val = vals[0];
        for (Index k = 1; k < len; ++k) {
            if (vals[k] > max_val) max_val = vals[k];
        }

        const auto v_max = s::Set(d, max_val);

        // Pass 2: Compute exp(x - max) and sum (fused)
        auto v_sum = s::Zero(d);
        Index k = 0;

        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);

            v0 = s::Exp(d, s::Sub(v0, v_max));
            v1 = s::Exp(d, s::Sub(v1, v_max));
            v2 = s::Exp(d, s::Sub(v2, v_max));
            v3 = s::Exp(d, s::Sub(v3, v_max));

            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);

            v_sum = s::Add(v_sum, v0);
            v_sum = s::Add(v_sum, v1);
            v_sum = s::Add(v_sum, v2);
            v_sum = s::Add(v_sum, v3);
        }

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v = s::Exp(d, s::Sub(v, v_max));
            s::Store(v, d, vals + k);
            v_sum = s::Add(v_sum, v);
        }

        T sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; k < len; ++k) {
            T v = std::exp(vals[k] - max_val);
            vals[k] = v;
            sum += v;
        }

        // Pass 3: Normalize
        if (sum > 0) {
            T inv_sum = 1.0 / sum;
            const auto v_inv_sum = s::Set(d, inv_sum);

            k = 0;
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Mul(v, v_inv_sum), d, vals + k);
            }

            for (; k < len; ++k) {
                vals[k] *= inv_sum;
            }
        }
    });

    return owned;
}

// =============================================================================
// MappedVirtualSparse Softmax
// =============================================================================

/// @brief Apply softmax to MappedVirtualSparse - returns materialized result
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    auto owned = matrix.materialize();

    const Index n_primary = scl::primary_size(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index start = owned.indptr[p];
        Index end = owned.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* SCL_RESTRICT vals = owned.data.data() + start;

        // Pass 1: Find max
        T max_val = vals[0];
        for (Index k = 1; k < len; ++k) {
            if (vals[k] > max_val) max_val = vals[k];
        }

        const auto v_max = s::Set(d, max_val);

        // Pass 2: Compute exp(x - max) and sum
        auto v_sum = s::Zero(d);
        Index k = 0;

        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v = s::Exp(d, s::Sub(v, v_max));
            s::Store(v, d, vals + k);
            v_sum = s::Add(v_sum, v);
        }

        T sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; k < len; ++k) {
            T v = std::exp(vals[k] - max_val);
            vals[k] = v;
            sum += v;
        }

        // Pass 3: Normalize
        if (sum > 0) {
            T inv_sum = 1.0 / sum;
            const auto v_inv_sum = s::Set(d, inv_sum);

            k = 0;
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Mul(v, v_inv_sum), d, vals + k);
            }

            for (; k < len; ++k) {
                vals[k] *= inv_sum;
            }
        }
    });

    return owned;
}

} // namespace scl::kernel::softmax::mapped
