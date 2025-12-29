// =============================================================================
// FILE: scl/binding/c_api/lineage/lineage.cpp
// BRIEF: C API implementation for lineage tracing
// =============================================================================

#include "scl/binding/c_api/lineage.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/lineage.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Lineage Coupling
// =============================================================================

SCL_EXPORT scl_error_t scl_lineage_coupling(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* coupling_matrix,
    const scl_size_t n_cells,
    const scl_size_t n_clones,
    const scl_size_t n_types) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(cell_types, "Cell types array is null");
    SCL_C_API_CHECK_NULL(coupling_matrix, "Output coupling matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_clones > 0 && n_types > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        Array<const Index> clone_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        Array<const Index> type_arr(
            reinterpret_cast<const Index*>(cell_types),
            n_cells
        );

        scl::kernel::lineage::lineage_coupling(
            clone_arr, type_arr,
            reinterpret_cast<Real*>(coupling_matrix),
            n_clones, n_types
        );

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Fate Bias
// =============================================================================

SCL_EXPORT scl_error_t scl_lineage_fate_bias(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* bias_scores,
    const scl_size_t n_cells,
    const scl_size_t n_types) {
    
    SCL_C_API_CHECK_NULL(clone_ids, "Clone IDs array is null");
    SCL_C_API_CHECK_NULL(cell_types, "Cell types array is null");
    SCL_C_API_CHECK_NULL(bias_scores, "Output bias scores array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        Array<const Index> clone_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        Array<const Index> type_arr(
            reinterpret_cast<const Index*>(cell_types),
            n_cells
        );

        // Find max clone ID
        Index max_clone = -1;
        for (scl_size_t i = 0; i < n_cells; ++i) {
            if (clone_ids[i] > max_clone) {
                max_clone = clone_ids[i];
            }
        }
        const Size n_clones = static_cast<Size>(max_clone + 1);

        Array<Real> bias_arr(
            reinterpret_cast<Real*>(bias_scores),
            n_clones
        );

        scl::kernel::lineage::fate_bias(clone_arr, type_arr, bias_arr, n_types);

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
