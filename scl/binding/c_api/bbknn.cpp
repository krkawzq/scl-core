// =============================================================================
// FILE: scl/binding/c_api/bbknn.cpp
// BRIEF: C API implementation for Batch Balanced KNN
// =============================================================================

#include "scl/binding/c_api/bbknn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/bbknn.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Compute Norms
// =============================================================================

SCL_EXPORT scl_error_t scl_bbknn_compute_norms(
    scl_sparse_t matrix,
    scl_real_t* norms_sq,
    const scl_size_t n_cells) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(norms_sq, "Output norms_sq array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        matrix->visit([&](auto& m) {
            Array<Real> norms_arr(
                reinterpret_cast<Real*>(norms_sq),
                n_cells
            );
            scl::kernel::bbknn::compute_norms(m, norms_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// BBKNN
// =============================================================================

SCL_EXPORT scl_error_t scl_bbknn(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    const scl_size_t n_cells,
    const scl_size_t n_batches,
    const scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(batch_labels, "Batch labels array is null");
    SCL_C_API_CHECK_NULL(out_indices, "Output indices array is null");
    SCL_C_API_CHECK_NULL(out_distances, "Output distances array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(n_batches > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of batches must be positive");
    SCL_C_API_CHECK(k > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of neighbors per batch must be positive");
    
    SCL_C_API_TRY
        const Size neighbors_per_cell = n_batches * k;
        const Size total_neighbors = n_cells * neighbors_per_cell;
        
        matrix->visit([&](auto& m) {
            const Array<const int32_t> batch_arr(batch_labels, n_cells);
            Array<Index> indices_arr(
                reinterpret_cast<Index*>(out_indices),
                total_neighbors
            );
            Array<Real> distances_arr(
                reinterpret_cast<Real*>(out_distances),
                total_neighbors
            );
            
            scl::kernel::bbknn::bbknn(
                m, batch_arr, n_batches, k,
                indices_arr, distances_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// BBKNN with Precomputed Norms
// =============================================================================

SCL_EXPORT scl_error_t scl_bbknn_with_norms(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    const scl_size_t n_cells,
    const scl_size_t n_batches,
    const scl_size_t k,
    const scl_real_t* norms_sq,
    scl_index_t* out_indices,
    scl_real_t* out_distances) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(batch_labels, "Batch labels array is null");
    SCL_C_API_CHECK_NULL(norms_sq, "Precomputed norms array is null");
    SCL_C_API_CHECK_NULL(out_indices, "Output indices array is null");
    SCL_C_API_CHECK_NULL(out_distances, "Output distances array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    SCL_C_API_CHECK(n_batches > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of batches must be positive");
    SCL_C_API_CHECK(k > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of neighbors per batch must be positive");
    
    SCL_C_API_TRY
        const Size neighbors_per_cell = n_batches * k;
        const Size total_neighbors = n_cells * neighbors_per_cell;
        
        matrix->visit([&](auto& m) {
            const Array<const int32_t> batch_arr(batch_labels, n_cells);
            const Array<const Real> norms_arr(
                reinterpret_cast<const Real*>(norms_sq),
                n_cells
            );
            Array<Index> indices_arr(
                reinterpret_cast<Index*>(out_indices),
                total_neighbors
            );
            Array<Real> distances_arr(
                reinterpret_cast<Real*>(out_distances),
                total_neighbors
            );
            
            scl::kernel::bbknn::bbknn(
                m, batch_arr, n_batches, k,
                indices_arr, distances_arr, norms_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
