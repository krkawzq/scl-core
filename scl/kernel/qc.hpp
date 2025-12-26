#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Include optimized backends
#include "scl/kernel/qc_fast_impl.hpp"
#include "scl/kernel/qc_mapped_impl.hpp"

#include <cmath>

// =============================================================================
/// @file qc.hpp
/// @brief Quality Control Metrics
///
/// ## Operations
///
/// - compute_basic_qc: n_genes (nnz per row), total_counts (row sums)
/// - compute_subset_pct: Percentage from specific gene sets (e.g., MT%, RB%)
/// - compute_fused_qc: All metrics in single pass
///
/// ## Use Cases
///
/// - Mitochondrial %: Identify dying cells
/// - Ribosomal %: Protein synthesis activity
/// - Hemoglobin %: Blood contamination
///
/// ## Performance
///
/// - O(nnz) complexity
/// - ~15-20 GB/s per core
/// - Backend dispatch: CustomSparse/VirtualSparse -> fast, Mapped -> mapped
// =============================================================================

namespace scl::kernel::qc {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Real PCT_SCALE = Real(100);
}

// =============================================================================
// SECTION 2: Basic QC
// =============================================================================

/// @brief Compute basic QC metrics (unified for all sparse types)
///
/// Dispatches to optimized backend based on matrix type:
/// - CustomSparse/VirtualSparse -> fast path with 4-way SIMD
/// - MappedSparse -> mapped path with prefetch hints
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse concept)
/// @param matrix Input sparse matrix
/// @param out_n_genes Number of detected elements [size = primary_dim], PRE-ALLOCATED
/// @param out_total_counts Total counts [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_basic_qc(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.size() == static_cast<Size>(primary_dim),
                  "QC: n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.size() == static_cast<Size>(primary_dim),
                  "QC: total_counts size mismatch");

    // Dispatch to optimized backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::mapped::compute_basic_qc_mapped_dispatch<MatrixT, IsCSR>(
            matrix, out_n_genes, out_total_counts
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_basic_qc_custom(
            matrix, out_n_genes, out_total_counts
        );
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_basic_qc_virtual(
            matrix, out_n_genes, out_total_counts
        );
    } else {
        // Generic fallback
        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(matrix, idx);

            out_n_genes[p] = static_cast<Index>(vals.size());

            namespace s = scl::simd;
            const s::Tag d;
            const size_t lanes = s::lanes();

            auto v_sum = s::Zero(d);
            size_t k = 0;

            for (; k + lanes <= vals.size(); k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, vals.ptr + k));
            }

            Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));

            for (; k < vals.size(); ++k) {
                sum += static_cast<Real>(vals[k]);
            }

            out_total_counts[p] = sum;
        });
    }
}

// =============================================================================
// SECTION 3: Subset Percentage
// =============================================================================

/// @brief Compute subset percentage (e.g., MT%, RB%)
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse concept)
/// @param matrix Input sparse matrix
/// @param subset_mask Binary mask marking subset genes [size = secondary_dim]
/// @param out_pcts Output percentages [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_subset_pct(
    const MatrixT& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);
    const Index secondary_dim = scl::secondary_size(matrix);

    SCL_CHECK_DIM(subset_mask.size() == static_cast<Size>(secondary_dim),
                  "QC: Subset mask size mismatch");
    SCL_CHECK_DIM(out_pcts.size() == static_cast<Size>(primary_dim),
                  "QC: Output pcts size mismatch");

    // Dispatch to optimized backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::mapped::compute_subset_pct_mapped_dispatch<MatrixT, IsCSR>(
            matrix, subset_mask, out_pcts
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_subset_pct_custom(
            matrix, subset_mask, out_pcts
        );
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_subset_pct_virtual(
            matrix, subset_mask, out_pcts
        );
    } else {
        // Generic fallback
        const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(matrix, idx);
            auto inds = scl::primary_indices(matrix, idx);

            Real subset_sum = Real(0);
            Real total_sum = Real(0);

            for (size_t k = 0; k < vals.size(); ++k) {
                Real val = static_cast<Real>(vals[k]);
                total_sum += val;
                subset_sum += val * static_cast<Real>(mask[inds[k]] != 0);
            }

            out_pcts[p] = (total_sum > Real(0))
                ? (subset_sum / total_sum * config::PCT_SCALE)
                : Real(0);
        });
    }
}

// =============================================================================
// SECTION 4: Fused QC
// =============================================================================

/// @brief Fused QC: basic metrics + subset percentage in single pass
///
/// More efficient than calling compute_basic_qc + compute_subset_pct separately.
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse concept)
/// @param matrix Input sparse matrix
/// @param subset_mask Binary mask marking subset genes [size = secondary_dim]
/// @param out_n_genes Number of detected elements [size = primary_dim], PRE-ALLOCATED
/// @param out_total_counts Total counts [size = primary_dim], PRE-ALLOCATED
/// @param out_pcts Output percentages [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_fused_qc(
    const MatrixT& matrix,
    Array<const uint8_t> subset_mask,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pcts
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);
    const Index secondary_dim = scl::secondary_size(matrix);

    SCL_CHECK_DIM(subset_mask.size() == static_cast<Size>(secondary_dim),
                  "QC: Subset mask size mismatch");
    SCL_CHECK_DIM(out_n_genes.size() == static_cast<Size>(primary_dim),
                  "QC: n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.size() == static_cast<Size>(primary_dim),
                  "QC: total_counts size mismatch");
    SCL_CHECK_DIM(out_pcts.size() == static_cast<Size>(primary_dim),
                  "QC: Output pcts size mismatch");

    // Dispatch to optimized backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::mapped::compute_fused_qc_mapped_dispatch<MatrixT, IsCSR>(
            matrix, subset_mask, out_n_genes, out_total_counts, out_pcts
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_fused_qc_custom(
            matrix, subset_mask, out_n_genes, out_total_counts, out_pcts
        );
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        scl::kernel::qc::fast::compute_fused_qc_virtual(
            matrix, subset_mask, out_n_genes, out_total_counts, out_pcts
        );
    } else {
        // Generic fallback
        const uint8_t* SCL_RESTRICT mask = subset_mask.ptr;

        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            Index idx = static_cast<Index>(p);
            auto vals = scl::primary_values(matrix, idx);
            auto inds = scl::primary_indices(matrix, idx);

            out_n_genes[p] = static_cast<Index>(vals.size());

            Real subset_sum = Real(0);
            Real total_sum = Real(0);

            for (size_t k = 0; k < vals.size(); ++k) {
                Real val = static_cast<Real>(vals[k]);
                total_sum += val;
                subset_sum += val * static_cast<Real>(mask[inds[k]] != 0);
            }

            out_total_counts[p] = total_sum;
            out_pcts[p] = (total_sum > Real(0))
                ? (subset_sum / total_sum * config::PCT_SCALE)
                : Real(0);
        });
    }
}

// =============================================================================
// SECTION 5: Extended QC
// =============================================================================

/// @brief Extended QC: basic metrics + top-N percentage
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse concept)
/// @param matrix Input sparse matrix
/// @param out_n_genes Number of detected elements [size = primary_dim], PRE-ALLOCATED
/// @param out_total_counts Total counts [size = primary_dim], PRE-ALLOCATED
/// @param out_pct_top Percentage from top N genes [size = primary_dim], PRE-ALLOCATED
/// @param n_top Number of top genes to consider
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_extended_qc(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts,
    Array<Real> out_pct_top,
    Size n_top
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    const Index primary_dim = scl::primary_size(matrix);

    SCL_CHECK_DIM(out_n_genes.size() == static_cast<Size>(primary_dim),
                  "QC: n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.size() == static_cast<Size>(primary_dim),
                  "QC: total_counts size mismatch");
    SCL_CHECK_DIM(out_pct_top.size() == static_cast<Size>(primary_dim),
                  "QC: pct_top size mismatch");

    // Dispatch to mapped backend if available
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        // MappedCustomSparse has extended QC
        if constexpr (requires { scl::kernel::qc::mapped::compute_extended_qc_mapped(
            matrix, out_n_genes, out_total_counts, out_pct_top, n_top); }) {
            scl::kernel::qc::mapped::compute_extended_qc_mapped(
                matrix, out_n_genes, out_total_counts, out_pct_top, n_top
            );
            return;
        }
    }

    // Generic fallback with nth_element
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(matrix, idx);
        Size len = vals.size();

        out_n_genes[p] = static_cast<Index>(len);

        if (len == 0) {
            out_total_counts[p] = Real(0);
            out_pct_top[p] = Real(0);
            return;
        }

        // Compute total
        Real total = Real(0);
        for (size_t k = 0; k < len; ++k) {
            total += static_cast<Real>(vals[k]);
        }
        out_total_counts[p] = total;

        // Compute top-N sum
        if (len <= n_top) {
            out_pct_top[p] = config::PCT_SCALE;
        } else {
            thread_local std::vector<Real> temp;
            temp.resize(len);
            for (size_t k = 0; k < len; ++k) {
                temp[k] = static_cast<Real>(vals[k]);
            }

            std::nth_element(temp.begin(), temp.begin() + n_top, temp.end(),
                           std::greater<Real>());

            Real top_sum = Real(0);
            for (Size i = 0; i < n_top; ++i) {
                top_sum += temp[i];
            }

            out_pct_top[p] = (total > Real(0))
                ? (top_sum / total * config::PCT_SCALE)
                : Real(0);
        }
    });
}

} // namespace scl::kernel::qc
