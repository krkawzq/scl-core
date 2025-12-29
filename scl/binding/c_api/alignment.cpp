// =============================================================================
// FILE: scl/binding/c_api/alignment.cpp
// BRIEF: C API implementation for multi-modal data alignment
// =============================================================================

#include "scl/binding/c_api/alignment.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/alignment.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// MNN Pairs
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_mnn_pairs(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t* n_pairs) {
    
    SCL_C_API_CHECK_NULL(data1, "Data1 matrix is null");
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(mnn_cell1, "Output mnn_cell1 array is null");
    SCL_C_API_CHECK_NULL(mnn_cell2, "Output mnn_cell2 array is null");
    SCL_C_API_CHECK_NULL(n_pairs, "Output n_pairs pointer is null");
    
    SCL_C_API_TRY
        Size n_pairs_result = 0;
        
        data1->visit([&](auto& m1) {
            data2->visit([&](auto& m2) {
                scl::kernel::alignment::mnn_pairs(
                    m1, m2,
                    k,
                    reinterpret_cast<Index*>(mnn_cell1),
                    reinterpret_cast<Index*>(mnn_cell2),
                    n_pairs_result
                );
            });
        });
        
        *n_pairs = n_pairs_result;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Anchor Finding
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_find_anchors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    scl_real_t* anchor_scores,
    scl_size_t* n_anchors) {
    
    SCL_C_API_CHECK_NULL(data1, "Data1 matrix is null");
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(anchor_cell1, "Output anchor_cell1 array is null");
    SCL_C_API_CHECK_NULL(anchor_cell2, "Output anchor_cell2 array is null");
    SCL_C_API_CHECK_NULL(anchor_scores, "Output anchor_scores array is null");
    SCL_C_API_CHECK_NULL(n_anchors, "Output n_anchors pointer is null");
    
    SCL_C_API_TRY
        Size n_anchors_result = 0;
        
        data1->visit([&](auto& m1) {
            data2->visit([&](auto& m2) {
                scl::kernel::alignment::find_anchors(
                    m1, m2,
                    k,
                    reinterpret_cast<Index*>(anchor_cell1),
                    reinterpret_cast<Index*>(anchor_cell2),
                    reinterpret_cast<Real*>(anchor_scores),
                    n_anchors_result
                );
            });
        });
        
        *n_anchors = n_anchors_result;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Label Transfer
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_transfer_labels(
    const scl_index_t* anchor_cell1,
    const scl_index_t* anchor_cell2,
    const scl_real_t* anchor_weights,
    const scl_size_t n_anchors,
    const scl_index_t* source_labels,
    const scl_size_t n_source,
    const scl_size_t n_target,
    scl_index_t* target_labels,
    scl_real_t* transfer_confidence) {
    
    SCL_C_API_CHECK_NULL(anchor_cell1, "Anchor cell1 array is null");
    SCL_C_API_CHECK_NULL(anchor_cell2, "Anchor cell2 array is null");
    SCL_C_API_CHECK_NULL(anchor_weights, "Anchor weights array is null");
    SCL_C_API_CHECK_NULL(source_labels, "Source labels array is null");
    SCL_C_API_CHECK_NULL(target_labels, "Output target labels array is null");
    SCL_C_API_CHECK_NULL(transfer_confidence, "Output confidence array is null");
    
    SCL_C_API_TRY
        scl::kernel::alignment::transfer_labels(
            reinterpret_cast<const Index*>(anchor_cell1),
            reinterpret_cast<const Index*>(anchor_cell2),
            reinterpret_cast<const Real*>(anchor_weights),
            n_anchors,
            Array<const Index>(
                reinterpret_cast<const Index*>(source_labels),
                n_source
            ),
            n_target,
            Array<Index>(
                reinterpret_cast<Index*>(target_labels),
                n_target
            ),
            Array<Real>(
                reinterpret_cast<Real*>(transfer_confidence),
                n_target
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Integration Quality
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_integration_score(
    scl_sparse_t integrated_data,
    const scl_index_t* batch_labels,
    const scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* score) {
    
    SCL_C_API_CHECK_NULL(integrated_data, "Integrated data matrix is null");
    SCL_C_API_CHECK_NULL(batch_labels, "Batch labels array is null");
    SCL_C_API_CHECK_NULL(neighbors, "Neighbors graph is null");
    SCL_C_API_CHECK_NULL(score, "Output score pointer is null");
    
    SCL_C_API_TRY
        Real score_result = Real(0);
        
        integrated_data->visit([&](auto& data) {
            using DataType = std::remove_reference_t<decltype(data)>;
            using T = typename DataType::ValueType;
            constexpr bool IsCSR_Data = DataType::is_csr;
            
            neighbors->visit([&](auto& neigh) {
                using NeighborType = std::remove_reference_t<decltype(neigh)>;
                constexpr bool IsCSR_Neighbors = NeighborType::is_csr;
                
                // PERFORMANCE: Cast to Index type sparse matrix
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                const auto& neigh_index = reinterpret_cast<
                    const Sparse<Index, IsCSR_Neighbors>&>(neigh);
                
                score_result = scl::kernel::alignment::integration_score<T, IsCSR_Data, IsCSR_Neighbors>(
                    data,
                    Array<const Index>(
                        reinterpret_cast<const Index*>(batch_labels),
                        n_cells
                    ),
                    neigh_index
                );
            });
        });
        
        *score = static_cast<scl_real_t>(score_result);
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_alignment_batch_mixing(
    const scl_index_t* batch_labels,
    const scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* mixing_scores) {
    
    SCL_C_API_CHECK_NULL(batch_labels, "Batch labels array is null");
    SCL_C_API_CHECK_NULL(neighbors, "Neighbors graph is null");
    SCL_C_API_CHECK_NULL(mixing_scores, "Output mixing scores array is null");
    
    SCL_C_API_TRY
        neighbors->visit([&](auto& neigh) {
            using NeighborType = std::remove_reference_t<decltype(neigh)>;
            constexpr bool IsCSR = NeighborType::is_csr;
            
            // PERFORMANCE: Cast to Index type sparse matrix
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            const auto& neigh_index = reinterpret_cast<
                const Sparse<Index, IsCSR>&>(neigh);
            
            scl::kernel::alignment::batch_mixing<IsCSR>(
                Array<const Index>(
                    reinterpret_cast<const Index*>(batch_labels),
                    n_cells
                ),
                neigh_index,
                Array<Real>(
                    reinterpret_cast<Real*>(mixing_scores),
                    n_cells
                )
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_alignment_kbet_score(
    scl_sparse_t neighbors,
    const scl_index_t* batch_labels,
    const scl_size_t n_cells,
    scl_real_t* score) {
    
    SCL_C_API_CHECK_NULL(neighbors, "Neighbors graph is null");
    SCL_C_API_CHECK_NULL(batch_labels, "Batch labels array is null");
    SCL_C_API_CHECK_NULL(score, "Output score pointer is null");
    
    SCL_C_API_TRY
        Real score_result = Real(0);
        
        neighbors->visit([&](auto& neigh) {
            using NeighborType = std::remove_reference_t<decltype(neigh)>;
            constexpr bool IsCSR = NeighborType::is_csr;
            
            // PERFORMANCE: Cast to Index type sparse matrix
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            const auto& neigh_index = reinterpret_cast<
                const Sparse<Index, IsCSR>&>(neigh);
            
            score_result = scl::kernel::alignment::kbet_score<IsCSR>(
                neigh_index,
                Array<const Index>(
                    reinterpret_cast<const Index*>(batch_labels),
                    n_cells
                )
            );
        });
        
        *score = static_cast<scl_real_t>(score_result);
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Correction Vectors
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_compute_correction_vectors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t* mnn_cell1,
    const scl_index_t* mnn_cell2,
    const scl_size_t n_pairs,
    scl_real_t* correction_vectors,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(data1, "Data1 matrix is null");
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(mnn_cell1, "MNN cell1 array is null");
    SCL_C_API_CHECK_NULL(mnn_cell2, "MNN cell2 array is null");
    SCL_C_API_CHECK_NULL(correction_vectors, "Output correction vectors array is null");
    
    SCL_C_API_TRY
        data1->visit([&](auto& m1) {
            data2->visit([&](auto& m2) {
                scl::kernel::alignment::compute_correction_vectors(
                    m1, m2,
                    reinterpret_cast<const Index*>(mnn_cell1),
                    reinterpret_cast<const Index*>(mnn_cell2),
                    n_pairs,
                    reinterpret_cast<Real*>(correction_vectors),
                    n_features
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_alignment_smooth_correction_vectors(
    scl_sparse_t data2,
    scl_real_t* correction_vectors,
    const scl_size_t n_features,
    const scl_real_t sigma) {
    
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(correction_vectors, "Correction vectors array is null");
    
    SCL_C_API_TRY
        data2->visit([&](auto& m) {
            scl::kernel::alignment::smooth_correction_vectors(
                m,
                reinterpret_cast<Real*>(correction_vectors),
                n_features,
                static_cast<Real>(sigma)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// CCA Projection
// =============================================================================

SCL_EXPORT scl_error_t scl_alignment_cca_projection(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_size_t n_components,
    scl_real_t* projection1,
    scl_real_t* projection2) {
    
    SCL_C_API_CHECK_NULL(data1, "Data1 matrix is null");
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(projection1, "Output projection1 array is null");
    SCL_C_API_CHECK_NULL(projection2, "Output projection2 array is null");
    
    SCL_C_API_TRY
        data1->visit([&](auto& m1) {
            data2->visit([&](auto& m2) {
                scl::kernel::alignment::cca_projection(
                    m1, m2,
                    n_components,
                    reinterpret_cast<Real*>(projection1),
                    reinterpret_cast<Real*>(projection2)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
