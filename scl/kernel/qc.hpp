#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file qc.hpp
/// @brief Quality Control Metrics
///
/// Computes per-cell statistics:
/// - n_genes: Number of detected genes (nnz per row)
/// - total_counts: Library size (sum per row)
/// - subset_pcts: Percentage from specific gene sets (e.g., MT%)
///
/// Use Cases:
/// - Mitochondrial %: Identify dying cells
/// - Ribosomal %: Protein synthesis activity
/// - Hemoglobin %: Blood contamination
///
/// Performance: O(nnz), ~10-15 GB/s per core
// =============================================================================

namespace scl::kernel::qc {

/// @brief Compute basic QC metrics (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param out_n_genes Number of detected elements [size = primary_dim]
/// @param out_total_counts Total counts [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_basic_qc(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_n_genes.size() == static_cast<Size>(primary_dim), 
                  "QC: n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.size() == static_cast<Size>(primary_dim), 
                  "QC: total_counts size mismatch");

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(matrix, idx);
        
        out_n_genes[p] = static_cast<Index>(vals.size());
        
        // Sum with SIMD
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        size_t k = 0;
        
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
        }
        
        Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));
        
        for (; k < vals.size(); ++k) {
            sum += static_cast<Real>(vals[k]);
        }
        
        out_total_counts[p] = sum;
    });
}

/// @brief Compute subset percentage (e.g., MT%, RB%)
///
/// @param matrix Input sparse matrix
/// @param subset_mask Binary mask marking subset genes [size = secondary_dim]
/// @param out_pcts Output percentages [size = primary_dim]
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_subset_pct(
    const MatrixT& matrix,
    Array<const uint8_t> subset_mask,
    Array<Real> out_pcts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Index secondary_dim = scl::secondary_size(matrix);
    
    SCL_CHECK_DIM(subset_mask.size() == static_cast<Size>(secondary_dim),
                  "QC: Subset mask size mismatch");
    SCL_CHECK_DIM(out_pcts.size() == static_cast<Size>(primary_dim),
                  "QC: Output pcts size mismatch");

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(matrix, idx);
        auto inds = scl::primary_indices(matrix, idx);
        
        Real subset_sum = 0.0;
        Real total_sum = 0.0;
        
        for (size_t k = 0; k < vals.size(); ++k) {
            Real val = static_cast<Real>(vals[k]);
            total_sum += val;
            
            if (subset_mask[inds[k]] != 0) {
                subset_sum += val;
            }
        }
        
        if (total_sum > 0) {
            out_pcts[p] = (subset_sum / total_sum) * 100.0;
        } else {
            out_pcts[p] = 0.0;
        }
    });
}

} // namespace scl::kernel::qc
