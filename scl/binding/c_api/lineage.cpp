// =============================================================================
// FILE: scl/binding/c_api/lineage.cpp
// BRIEF: C API implementation for lineage tracing
// =============================================================================

#include "scl/binding/c_api/lineage.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/lineage.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>
#include <cstring>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_lineage_coupling(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* coupling_matrix,
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !coupling_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::kernel::lineage::lineage_coupling(
            clone_arr, type_arr,
            reinterpret_cast<scl::Real*>(coupling_matrix),
            n_clones, n_types
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_fate_bias(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* bias_scores,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !bias_scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        
        // Find max clone ID to determine output size
        scl::Index max_clone = -1;
        for (scl::Size i = 0; i < n_cells; ++i) {
            if (clone_ids[i] > max_clone) max_clone = clone_ids[i];
        }
        if (max_clone < 0) return SCL_ERROR_OK;
        
        scl::Size n_clones = static_cast<scl::Size>(max_clone + 1);
        scl::Array<scl::Real> bias_arr(reinterpret_cast<scl::Real*>(bias_scores), n_clones);
        
        scl::kernel::lineage::fate_bias(clone_arr, type_arr, bias_arr, n_types);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_build_lineage_tree(
    const scl_index_t* clone_ids,
    const scl_real_t* pseudotime,
    scl_index_t* parent,
    scl_size_t n_cells,
    scl_size_t n_clones
) {
    if (!clone_ids || !pseudotime || !parent) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Real> time_arr(reinterpret_cast<const scl::Real*>(pseudotime), n_cells);
        scl::kernel::lineage::build_lineage_tree(
            clone_arr, time_arr,
            reinterpret_cast<scl::Index*>(parent),
            n_clones
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_distance(
    const scl_index_t* parent,
    scl_real_t* distance_matrix,
    scl_size_t n_clones
) {
    if (!parent || !distance_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::kernel::lineage::lineage_distance(
            reinterpret_cast<const scl::Index*>(parent),
            n_clones,
            reinterpret_cast<scl::Real*>(distance_matrix)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_barcode_clone_assignment(
    const uint64_t* barcode_hashes,
    scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t* n_unique_clones
) {
    if (!barcode_hashes || !clone_ids || !n_unique_clones) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Size n_clones = 0;
        scl::kernel::lineage::barcode_clone_assignment(
            barcode_hashes, n_cells,
            reinterpret_cast<scl::Index*>(clone_ids),
            n_clones
        );
        *n_unique_clones = n_clones;
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clonal_fate_probability(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* fate_probs,
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !fate_probs) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::kernel::lineage::clonal_fate_probability(
            clone_arr, type_arr,
            reinterpret_cast<scl::Real*>(fate_probs),
            n_clones, n_types
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_fate_bias_per_type(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* bias_per_type,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !bias_per_type) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::Array<scl::Real> bias_arr(reinterpret_cast<scl::Real*>(bias_per_type), n_types);
        scl::kernel::lineage::fate_bias_per_type(clone_arr, type_arr, bias_arr, n_types);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_sharing(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* sharing_matrix,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !sharing_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::kernel::lineage::lineage_sharing(
            clone_arr, type_arr,
            reinterpret_cast<scl::Real*>(sharing_matrix),
            n_types
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_commitment(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* commitment_scores,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !commitment_scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(commitment_scores), n_cells);
        scl::kernel::lineage::lineage_commitment(clone_arr, type_arr, scores_arr, n_types);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_progenitor_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    const scl_real_t* pseudotime,
    scl_real_t* progenitor_scores,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !pseudotime || !progenitor_scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::Array<const scl::Real> time_arr(reinterpret_cast<const scl::Real*>(pseudotime), n_cells);
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(progenitor_scores), n_cells);
        scl::kernel::lineage::progenitor_score(clone_arr, type_arr, time_arr, scores_arr, n_types);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_lineage_transition_probability(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    const scl_real_t* pseudotime,
    scl_real_t* transition_prob,
    scl_size_t n_cells,
    scl_size_t n_types
) {
    if (!clone_ids || !cell_types || !pseudotime || !transition_prob) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> clone_arr(reinterpret_cast<const scl::Index*>(clone_ids), n_cells);
        scl::Array<const scl::Index> type_arr(reinterpret_cast<const scl::Index*>(cell_types), n_cells);
        scl::Array<const scl::Real> time_arr(reinterpret_cast<const scl::Real*>(pseudotime), n_cells);
        scl::kernel::lineage::lineage_transition_probability(
            clone_arr, type_arr, time_arr,
            reinterpret_cast<scl::Real*>(transition_prob),
            n_types
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clone_generation(
    const scl_index_t* parent,
    scl_index_t* generation,
    scl_size_t n_clones
) {
    if (!parent || !generation) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::kernel::lineage::clone_generation(
            reinterpret_cast<const scl::Index*>(parent),
            n_clones,
            reinterpret_cast<scl::Index*>(generation)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

