#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file spatial.hpp
/// @brief Spatial Autocorrelation Statistics
///
/// Implements Moran's I for spatial transcriptomics.
/// Moran's I = (N/W) * sum_ij(w_ij * (x_i - mean) * (x_j - mean)) / sum_i((x_i - mean)^2)
///
/// Performance: O(nnz_graph + nnz_feature) per feature
// =============================================================================

namespace scl::kernel::spatial {

/// @brief Compute Moran's I statistic (unified for CSR/CSC)
///
/// @param graph Spatial weight matrix (cells x cells), typically CSR
/// @param features Feature matrix (cells x genes), typically CSC
/// @param output Output Moran's I values [size = n_features]
template <typename GraphT, typename FeatureT>
    requires AnySparse<GraphT> && AnySparse<FeatureT>
void morans_i(
    const GraphT& graph,
    const FeatureT& features,
    Array<Real> output
) {
    const Index n_cells = scl::primary_size(graph);
    const Index n_features = scl::primary_size(features);
    
    SCL_CHECK_DIM(scl::secondary_size(graph) == n_cells, "Graph must be square");
    SCL_CHECK_DIM(scl::secondary_size(features) == n_cells, "Features dim mismatch");
    SCL_CHECK_DIM(output.size() == static_cast<Size>(n_features), "Output size mismatch");
    
    // Compute weight sum
    Real W_sum = 0.0;
    for (Index i = 0; i < n_cells; ++i) {
        auto vals = scl::primary_values(graph, i);
        for (Size k = 0; k < vals.size(); ++k) {
            W_sum += vals[k];
        }
    }
    
    const Real N = static_cast<Real>(n_cells);
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_features), [&](size_t f) {
        auto feat_vals = scl::primary_values(features, static_cast<Index>(f));
        auto feat_inds = scl::primary_indices(features, static_cast<Index>(f));
        
        // Compute mean
        Real sum = 0.0;
        for (Size k = 0; k < feat_vals.size(); ++k) {
            sum += feat_vals[k];
        }
        Real mean = sum / N;
        
        // Materialize centered values
        std::vector<Real> z(n_cells, -mean);
        for (Size k = 0; k < feat_vals.size(); ++k) {
            z[feat_inds[k]] = feat_vals[k] - mean;
        }
        
        // Compute numerator and denominator
        Real numer = 0.0;
        Real denom = 0.0;
        
        for (Index i = 0; i < n_cells; ++i) {
            auto graph_vals = scl::primary_values(graph, i);
            auto graph_inds = scl::primary_indices(graph, i);
            
            for (Size k = 0; k < graph_vals.size(); ++k) {
                Index j = graph_inds[k];
                Real w_ij = graph_vals[k];
                numer += w_ij * z[i] * z[j];
            }
            
            denom += z[i] * z[i];
        }
        
        if (denom > 0 && W_sum > 0) {
            output[f] = (N / W_sum) * (numer / denom);
        } else {
            output[f] = 0.0;
        }
    });
}

} // namespace scl::kernel::spatial
