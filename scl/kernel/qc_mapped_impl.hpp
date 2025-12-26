#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>
#include <vector>

// =============================================================================
/// @file qc_mapped_impl.hpp
/// @brief Mapped Backend QC Metrics
///
/// ## Key Optimizations
///
/// 1. Streaming Access
///    - Sequential row access for page cache efficiency
///    - Prefetch hints for OS
///
/// 2. 4-Way SIMD Accumulation
///    - Same vectorization as fast path
///
/// 3. Fused Multi-Metric Computation
///    - Single pass for n_genes + total_counts + subset_pct
///
/// 4. Efficient Top-N Computation
///    - nth_element O(n) instead of partial_sort O(n log k)
///
/// Performance: Near-RAM for hot cache, graceful degradation for cold
// =============================================================================

namespace scl::kernel::qc::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = Real(100);
}

// =============================================================================
// SECTION 2: SIMD Helpers
// =============================================================================

namespace detail {

/// @brief 4-way unrolled SIMD sum
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

/// @brief Fused subset sum with mask (branchless)
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

/// @brief Compute sum of top-K values using nth_element (O(n) average)
template <typename T>
SCL_FORCE_INLINE Real top_k_sum(const T* vals, Size len, Size k) {
    if (len <= k) {
        // Sum all
        Real sum = Real(0);
        for (Size i = 0; i < len; ++i) {
            sum += static_cast<Real>(vals[i]);
        }
        return sum;
    }

    // Copy to temp buffer
    thread_local std::vector<Real> temp;
    temp.resize(len);
    for (Size i = 0; i < len; ++i) {
        temp[i] = static_cast<Real>(vals[i]);
    }

    // nth_element puts the k-th largest at position k-1
    std::nth_element(temp.begin(), temp.begin() + k, temp.end(), std::greater<Real>());

    // Sum top k
    Real sum = Real(0);
    for (Size i = 0; i < k; ++i) {
        sum += temp[i];
    }

    return sum;
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse QC
// =============================================================================

/// @brief Basic QC for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_basic_qc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = matrix.data + start;
        out_total_counts[p] = detail::simd_sum_4way(vals, len);
    });
}

/// @brief Subset percentage for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_subset_pct_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(n_primary), "pcts size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        if (len == 0) {
            out_pcts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = matrix.data + start;
        const Index* SCL_RESTRICT inds = matrix.indices + start;

        Real total, subset;
        detail::fused_total_subset_sum(vals, inds, mask, len, total, subset);

        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

/// @brief Fused QC for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_fused_qc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pcts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");
    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(n_primary), "pcts size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            out_pcts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = matrix.data + start;
        const Index* SCL_RESTRICT inds = matrix.indices + start;

        Real total, subset;
        detail::fused_total_subset_sum(vals, inds, mask, len, total, subset);

        out_total_counts[p] = total;
        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

/// @brief Extended QC with top-N percentage
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void compute_extended_qc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pct_top,
    Size n_top
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");
    SCL_CHECK_DIM(out_pct_top.len == static_cast<Size>(n_primary), "pct_top size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            out_pct_top[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = matrix.data + start;

        Real total = detail::simd_sum_4way(vals, len);
        out_total_counts[p] = total;

        if (len <= n_top) {
            out_pct_top[p] = config::PCT_SCALE;
        } else {
            Real top_sum = detail::top_k_sum(vals, len, n_top);
            out_pct_top[p] = (total > Real(0)) ? (top_sum / total * config::PCT_SCALE) : Real(0);
        }
    });
}

// =============================================================================
// SECTION 4: MappedVirtualSparse QC
// =============================================================================

/// @brief Basic QC for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_basic_qc_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        out_total_counts[p] = detail::simd_sum_4way(vals, len);
    });
}

/// @brief Subset percentage for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_subset_pct_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(n_primary), "pcts size mismatch");

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        if (len == 0) {
            out_pcts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* SCL_RESTRICT inds = static_cast<const Index*>(matrix.indices_ptrs[p]);

        Real total, subset;
        detail::fused_total_subset_sum(vals, inds, mask, len, total, subset);

        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

/// @brief Fused QC for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void compute_fused_qc_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const uint8_t> subset_mask,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pcts
) {
    const Index n_primary = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(n_primary), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(n_primary), "total_counts size mismatch");
    SCL_CHECK_DIM(out_pcts.len == static_cast<Size>(n_primary), "pcts size mismatch");

    const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            out_pcts[p] = Real(0);
            return;
        }

        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* SCL_RESTRICT inds = static_cast<const Index*>(matrix.indices_ptrs[p]);

        Real total, subset;
        detail::fused_total_subset_sum(vals, inds, mask, len, total, subset);

        out_total_counts[p] = total;
        out_pcts[p] = (total > Real(0)) ? (subset / total * config::PCT_SCALE) : Real(0);
    });
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Basic QC dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_basic_qc_mapped_dispatch(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    compute_basic_qc_mapped(matrix, out_n_genes, out_total_counts);
}

/// @brief Subset percentage dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_subset_pct_mapped_dispatch(
    const MatrixT& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    compute_subset_pct_mapped(matrix, subset_mask, out_pcts);
}

/// @brief Fused QC dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void compute_fused_qc_mapped_dispatch(
    const MatrixT& matrix,
    Array<const uint8_t> subset_mask,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pcts
) {
    compute_fused_qc_mapped(matrix, subset_mask, out_n_genes, out_total_counts, out_pcts);
}

} // namespace scl::kernel::qc::mapped
