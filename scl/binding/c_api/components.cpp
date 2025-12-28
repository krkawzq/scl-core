// =============================================================================
// FILE: scl/binding/c_api/components.cpp
// BRIEF: C API implementation for connected components and graph connectivity
// =============================================================================

#include "scl/binding/c_api/components.h"
#include "scl/kernel/components.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

extern "C" {

// Internal helper to convert C++ exception to error code
static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_components_connected_components(
    scl_sparse_matrix_t adjacency,
    scl_index_t* component_labels,
    scl_index_t* n_components
) {
    if (!adjacency || !component_labels || !n_components) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Index> labels_arr(component_labels, static_cast<scl::Size>(n));
        scl::Index n_comp;
        
        scl::kernel::components::connected_components(*sparse, labels_arr, n_comp);
        *n_components = n_comp;
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

int scl_components_is_connected(scl_sparse_matrix_t adjacency) {
    if (!adjacency) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        return scl::kernel::components::is_connected(*sparse) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_components_largest_component(
    scl_sparse_matrix_t adjacency,
    scl_index_t* node_mask,
    scl_index_t* component_size
) {
    if (!adjacency || !node_mask || !component_size) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Index> mask_arr(node_mask, static_cast<scl::Size>(n));
        scl::Index comp_size;
        
        scl::kernel::components::largest_component(*sparse, mask_arr, comp_size);
        *component_size = comp_size;
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_components_component_sizes(
    scl_sparse_matrix_t adjacency,
    scl_index_t* sizes,
    scl_index_t* n_components,
    scl_index_t max_components
) {
    if (!adjacency || !sizes || !n_components) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        scl::Index n_comp;
        
        // First get labels to determine n_components
        const scl::Index n = sparse->primary_dim();
        scl::Index* labels = new scl::Index[n];
        scl::Array<scl::Index> labels_arr(labels, static_cast<scl::Size>(n));
        scl::kernel::components::connected_components(*sparse, labels_arr, n_comp);
        
        if (static_cast<scl::Size>(n_comp) > static_cast<scl::Size>(max_components)) {
            delete[] labels;
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        scl::Array<scl::Index> sizes_arr(sizes, static_cast<scl::Size>(max_components));
        scl::kernel::components::component_sizes(*sparse, sizes_arr, n_comp);
        *n_components = n_comp;
        
        delete[] labels;
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_components_bfs(
    scl_sparse_matrix_t adjacency,
    scl_index_t source,
    scl_index_t* distances,
    scl_index_t* predecessors
) {
    if (!adjacency || !distances) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Index> dist_arr(distances, static_cast<scl::Size>(n));
        scl::Array<scl::Index> pred_arr(predecessors, predecessors ? static_cast<scl::Size>(n) : 0);
        
        scl::kernel::components::bfs(*sparse, source, dist_arr, pred_arr);
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_components_multi_source_bfs(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* sources,
    scl_size_t n_sources,
    scl_index_t* distances
) {
    if (!adjacency || !sources || !distances) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<const scl::Index> src_arr(sources, n_sources);
        scl::Array<scl::Index> dist_arr(distances, static_cast<scl::Size>(n));
        
        scl::kernel::components::multi_source_bfs(*sparse, src_arr, dist_arr);
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_components_parallel_bfs(
    scl_sparse_matrix_t adjacency,
    scl_index_t source,
    scl_index_t* distances
) {
    if (!adjacency || !distances) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Index> dist_arr(distances, static_cast<scl::Size>(n));
        
        scl::kernel::components::parallel_bfs(*sparse, source, dist_arr);
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_index_t scl_components_graph_diameter(scl_sparse_matrix_t adjacency) {
    if (!adjacency) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        return scl::kernel::components::graph_diameter(*sparse);
    } catch (...) {
        return 0;
    }
}

scl_real_t scl_components_average_path_length(
    scl_sparse_matrix_t adjacency,
    scl_size_t max_samples
) {
    if (!adjacency) {
        return 0.0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        return scl::kernel::components::average_path_length(*sparse, max_samples);
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_components_clustering_coefficient(
    scl_sparse_matrix_t adjacency,
    scl_real_t* coefficients
) {
    if (!adjacency || !coefficients) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Real> coeff_arr(coefficients, static_cast<scl::Size>(n));
        
        scl::kernel::components::clustering_coefficient(*sparse, coeff_arr);
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_components_global_clustering_coefficient(scl_sparse_matrix_t adjacency) {
    if (!adjacency) {
        return 0.0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        return scl::kernel::components::global_clustering_coefficient(*sparse);
    } catch (...) {
        return 0.0;
    }
}

scl_size_t scl_components_count_triangles(scl_sparse_matrix_t adjacency) {
    if (!adjacency) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        return scl::kernel::components::count_triangles(*sparse);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_components_degree_sequence(
    scl_sparse_matrix_t adjacency,
    scl_index_t* degrees
) {
    if (!adjacency || !degrees) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        scl::Array<scl::Index> deg_arr(degrees, static_cast<scl::Size>(n));
        
        scl::kernel::components::degree_sequence(*sparse, deg_arr);
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_components_degree_statistics(
    scl_sparse_matrix_t adjacency,
    scl_real_t* mean_degree,
    scl_real_t* max_degree,
    scl_real_t* min_degree,
    scl_real_t* std_degree
) {
    if (!adjacency || !mean_degree || !max_degree || !min_degree || !std_degree) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        scl::Real mean, max, min, std;
        
        scl::kernel::components::degree_statistics(*sparse, mean, max, min, std);
        
        *mean_degree = mean;
        *max_degree = max;
        *min_degree = min;
        *std_degree = std;
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::components::connected_components<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    scl::Index&
);

template bool scl::kernel::components::is_connected<scl::Real, true>(const scl::CSR&);

template void scl::kernel::components::largest_component<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    scl::Index&
);

template void scl::kernel::components::component_sizes<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    scl::Index&
);

template void scl::kernel::components::bfs<scl::Real, true>(
    const scl::CSR&,
    scl::Index,
    scl::Array<scl::Index>,
    scl::Array<scl::Index>
);

template void scl::kernel::components::multi_source_bfs<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Index>
);

template void scl::kernel::components::parallel_bfs<scl::Real, true>(
    const scl::CSR&,
    scl::Index,
    scl::Array<scl::Index>
);

template scl::Index scl::kernel::components::graph_diameter<scl::Real, true>(const scl::CSR&);

template scl::Real scl::kernel::components::average_path_length<scl::Real, true>(
    const scl::CSR&,
    scl::Size
);

template void scl::kernel::components::clustering_coefficient<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>
);

template scl::Real scl::kernel::components::global_clustering_coefficient<scl::Real, true>(
    const scl::CSR&
);

template scl::Size scl::kernel::components::count_triangles<scl::Real, true>(const scl::CSR&);

template void scl::kernel::components::degree_sequence<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>
);

template void scl::kernel::components::degree_statistics<scl::Real, true>(
    const scl::CSR&,
    scl::Real&,
    scl::Real&,
    scl::Real&,
    scl::Real&
);

} // extern "C"
