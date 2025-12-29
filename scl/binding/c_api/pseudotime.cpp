// =============================================================================
// FILE: scl/binding/c_api/pseudotime/pseudotime.cpp
// BRIEF: C API implementation for pseudotime inference
// =============================================================================

#include "scl/binding/c_api/pseudotime.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/pseudotime.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_pseudotime_compute(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_index_t n,
    scl_pseudotime_method_t method,
    scl_index_t n_dcs)
{
    if (!adjacency || !pseudotime) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (root_cell < 0 || root_cell >= n) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Root cell index out of bounds");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Real> pseudotime_arr(
            reinterpret_cast<Real*>(pseudotime),
            static_cast<Size>(n)
        );
        
        scl::kernel::pseudotime::PseudotimeMethod method_enum{};
        switch (method) {
            case SCL_PSEUDOTIME_DIFFUSION:
                method_enum = scl::kernel::pseudotime::PseudotimeMethod::DiffusionPseudotime;
                break;
            case SCL_PSEUDOTIME_SHORTEST_PATH:
                method_enum = scl::kernel::pseudotime::PseudotimeMethod::ShortestPath;
                break;
            case SCL_PSEUDOTIME_GRAPH_DISTANCE:
                method_enum = scl::kernel::pseudotime::PseudotimeMethod::GraphDistance;
                break;
            case SCL_PSEUDOTIME_WATERSHED:
                method_enum = scl::kernel::pseudotime::PseudotimeMethod::WatershedDescent;
                break;
            default:
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid pseudotime method");
                return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::pseudotime::compute_pseudotime(
                adj,
                root_cell,
                pseudotime_arr,
                method_enum,
                n_dcs
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_pseudotime_diffusion(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_index_t n,
    scl_index_t n_dcs)
{
    if (!adjacency || !pseudotime) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (root_cell < 0 || root_cell >= n) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Root cell index out of bounds");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Real> pseudotime_arr(
            reinterpret_cast<Real*>(pseudotime),
            static_cast<Size>(n)
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::pseudotime::diffusion_pseudotime(
                adj,
                root_cell,
                pseudotime_arr,
                n_dcs
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_pseudotime_graph(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_index_t n)
{
    if (!adjacency || !pseudotime) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (root_cell < 0 || root_cell >= n) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Root cell index out of bounds");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Real> pseudotime_arr(
            reinterpret_cast<Real*>(pseudotime),
            static_cast<Size>(n)
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::pseudotime::graph_pseudotime(
                adj,
                root_cell,
                pseudotime_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_pseudotime_multi_source(
    scl_sparse_t adjacency,
    const scl_index_t* source_cells,
    scl_index_t n_sources,
    scl_real_t* distances)
{
    if (!adjacency || !source_cells || !distances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_sources == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Number of sources must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Index> source_cells_arr(
            reinterpret_cast<const Index*>(source_cells),
            static_cast<Size>(n_sources)
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::pseudotime::dijkstra_multi_source(
                adj,
                source_cells_arr,
                reinterpret_cast<Real*>(distances)  // Kernel expects Real*, not Array
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

