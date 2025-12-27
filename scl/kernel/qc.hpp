#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/qc.hpp
// BRIEF: Quality control metrics with SIMD optimization
// =============================================================================

namespace scl::kernel::qc {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = Real(100);
}

// =============================================================================
// SIMD Helpers
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE Real simd_sum_4way(const T* SCL_RESTRICT vals, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        SCL_PREFETCH_READ(vals + k + config::PREFETCH_DISTANCE * lanes, 0);
        v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
        v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
    }

    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= len; k += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, vals + k));
    }

    Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));

    for (; k < len; ++k) {
        sum += static_cast<Real>(vals[k]);
    }

    return sum;
}

template <typename T>
SCL_FORCE_INLINE void fused_total_subset_sum(
    const T* SCL_RESTRICT vals,
    const Index* SCL_RESTRICT indices,
    const uint8_t* SCL_RESTRICT mask,
    Size len,
    Real& out_total,
    Real& out_subset
) {
    Real total = Real(0);
    Real subset = Real(0);

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        Real v0 = static_cast<Real>(vals[k + 0]);
        Real v1 = static_cast<Real>(vals[k + 1]);
        Real v2 = static_cast<Real>(vals[k + 2]);
        Real v3 = static_cast<Real>(vals[k + 3]);

        total += v0 + v1 + v2 + v3;

        Real m0 = static_cast<Real>(mask[indices[k + 0]] != 0);
        Real m1 = static_cast<Real>(mask[indices[k + 1]] != 0);
        Real m2 = static_cast<Real>(mask[indices[k + 2]] != 0);
        Real m3 = static_cast<Real>(mask[indices[k + 3]] != 0);

        subset += v0 * m0 + v1 * m1 + v2 * m2 + v3 * m3;
    }

    for (; k < len; ++k) {
        Real v = static_cast<Real>(vals[k]);
        total += v;
        subset += v * static_cast<Real>(mask[indices[k]] != 0);
    }

    out_total = total;
    out_subset = subset;
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void compute_basic_qc(
    const Sparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(primary_dim), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(primary_dim), "total_counts size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        out_n_genes[p] = len;

        if (len_sz == 0) {
            out_total_counts[p] = Real(0);
            return;
        }

        auto values = matrix.primary_values(idx);
        out_total_counts[p] = detail::simd_sum_4way(values.ptr, len_sz);
    });
}

template <typename T, bool IsCSR>
void compute_subset_pct(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(primary_dim), "pcts size mismatch");

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            out_pcts[p] = Real(0);
            return;
        }

        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);

        Real total, subset;
        detail::fused_total_subset_sum(values.ptr, indices.ptr, mask, len_sz, total, subset);

        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

template <typename T, bool IsCSR>
void compute_fused_qc(
    const Sparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pcts
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(primary_dim), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(primary_dim), "total_counts size mismatch");
    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(primary_dim), "pcts size mismatch");

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        out_n_genes[p] = len;

        if (len_sz == 0) {
            out_total_counts[p] = Real(0);
            out_pcts[p] = Real(0);
            return;
        }

        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);

        Real total, subset;
        detail::fused_total_subset_sum(values.ptr, indices.ptr, mask, len_sz, total, subset);

        out_total_counts[p] = total;
        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

} // namespace scl::kernel::qc

