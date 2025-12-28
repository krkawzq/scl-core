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
using namespace scl::kernel::lineage;

extern "C" {

scl_error_t scl_lineage_coupling(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* coupling_matrix,
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types)
{
    if (!clone_ids || !cell_types || !coupling_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        Array<const Index> clone_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        Array<const Index> type_arr(
            reinterpret_cast<const Index*>(cell_types),
            n_cells
        );

        lineage_coupling(
            clone_arr, type_arr,
            reinterpret_cast<Real*>(coupling_matrix),
            n_clones, n_types
        );

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_fate_bias(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* bias_scores,
    scl_size_t n_cells,
    scl_size_t n_types)
{
    if (!clone_ids || !cell_types || !bias_scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
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

        if (max_clone < 0) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "No valid clone IDs");
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl_size_t n_clones = static_cast<scl_size_t>(max_clone + 1);

        Array<Real> bias_arr(
            reinterpret_cast<Real*>(bias_scores),
            n_clones
        );

        fate_bias(clone_arr, type_arr, bias_arr, n_types);

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_build_tree(
    const scl_index_t* clone_ids,
    const scl_real_t* pseudotime,
    scl_index_t* parent,
    scl_size_t n_cells,
    scl_size_t n_clones)
{
    if (!clone_ids || !pseudotime || !parent) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        Array<const Index> clone_arr(
            reinterpret_cast<const Index*>(clone_ids),
            n_cells
        );
        Array<const Real> time_arr(
            reinterpret_cast<const Real*>(pseudotime),
            n_cells
        );

        build_lineage_tree(
            clone_arr, time_arr,
            reinterpret_cast<Index*>(parent),
            n_clones
        );

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

