// =============================================================================
// FILE: scl/binding/c_api/components/components.cpp
// BRIEF: C API implementation for graph components analysis
// =============================================================================

#include "scl/binding/c_api/components.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/components.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Connected Components
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_connected_components(
    scl_sparse_t adjacency,
    scl_index_t* component_labels,
    scl_index_t* n_components) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(component_labels, "Output component labels array is null");
    SCL_C_API_CHECK_NULL(n_components, "Output n_components pointer is null");
    
    SCL_C_API_TRY
        Index n_comp = 0;
        
        adjacency->visit([&](auto& m) {
            Array<Index> labels(component_labels, static_cast<Size>(m.primary_dim()));
            scl::kernel::components::connected_components(m, labels, n_comp);
        });
        
        *n_components = n_comp;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Graph Connectivity
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_is_connected(
    scl_sparse_t adjacency,
    int* is_connected) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(is_connected, "Output is_connected pointer is null");
    
    SCL_C_API_TRY
        bool connected = false;
        
        adjacency->visit([&](auto& m) {
            connected = scl::kernel::components::is_connected(m);
        });
        
        *is_connected = connected ? 1 : 0;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_largest_component(
    scl_sparse_t adjacency,
    scl_index_t* node_mask,
    scl_index_t* component_size) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(node_mask, "Output node mask array is null");
    SCL_C_API_CHECK_NULL(component_size, "Output component size pointer is null");
    
    SCL_C_API_TRY
        Index size = 0;
        
        adjacency->visit([&](auto& m) {
            Array<Index> mask(node_mask, static_cast<Size>(m.primary_dim()));
            scl::kernel::components::largest_component(m, mask, size);
        });
        
        *component_size = size;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_component_sizes(
    scl_sparse_t adjacency,
    scl_index_t* sizes,
    scl_index_t* n_components) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(sizes, "Output sizes array is null");
    SCL_C_API_CHECK_NULL(n_components, "Output n_components pointer is null");
    
    SCL_C_API_TRY
        Index n_comp = 0;
        
        adjacency->visit([&](auto& m) {
            // Size determined internally by kernel
            Array<Index> sizes_arr(sizes, 0);
            scl::kernel::components::component_sizes(m, sizes_arr, n_comp);
        });
        
        *n_components = n_comp;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Breadth-First Search
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_bfs(
    scl_sparse_t adjacency,
    const scl_index_t source,
    scl_index_t* distances,
    scl_index_t* predecessors) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(distances, "Output distances array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Index> dist(distances, static_cast<Size>(n_nodes));
            Array<Index> pred;
            if (predecessors) {
                pred = Array<Index>(predecessors, static_cast<Size>(n_nodes));
            }
            scl::kernel::components::bfs(m, source, dist, pred);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_multi_source_bfs(
    scl_sparse_t adjacency,
    const scl_index_t* sources,
    const scl_size_t n_sources,
    scl_index_t* distances) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(sources, "Sources array is null");
    SCL_C_API_CHECK_NULL(distances, "Output distances array is null");
    SCL_C_API_CHECK(n_sources > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of sources must be positive");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<const Index> srcs(sources, n_sources);
            Array<Index> dist(distances, static_cast<Size>(n_nodes));
            scl::kernel::components::multi_source_bfs(m, srcs, dist);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_parallel_bfs(
    scl_sparse_t adjacency,
    const scl_index_t source,
    scl_index_t* distances) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(distances, "Output distances array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Index> dist(distances, static_cast<Size>(n_nodes));
            scl::kernel::components::parallel_bfs(m, source, dist);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Depth-First Search
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_dfs(
    scl_sparse_t adjacency,
    const scl_index_t source,
    scl_index_t* discovery_time,
    scl_index_t* finish_time) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(discovery_time, "Output discovery time array is null");
    SCL_C_API_CHECK_NULL(finish_time, "Output finish time array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Index> disc(discovery_time, static_cast<Size>(n_nodes));
            Array<Index> fin(finish_time, static_cast<Size>(n_nodes));
            scl::kernel::components::dfs(m, source, disc, fin);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Topological Sort
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_topological_sort(
    scl_sparse_t adjacency,
    scl_index_t* order,
    int* is_valid) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(order, "Output order array is null");
    SCL_C_API_CHECK_NULL(is_valid, "Output is_valid pointer is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        bool valid = false;
        
        adjacency->visit([&](auto& m) {
            Array<Index> ord(order, static_cast<Size>(n_nodes));
            valid = scl::kernel::components::topological_sort(m, ord);
        });
        
        *is_valid = valid ? 1 : 0;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Graph Metrics
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_graph_diameter(
    scl_sparse_t adjacency,
    scl_index_t* diameter) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(diameter, "Output diameter pointer is null");
    
    SCL_C_API_TRY
        Index diam = 0;
        
        adjacency->visit([&](auto& m) {
            diam = scl::kernel::components::graph_diameter(m);
        });
        
        *diameter = diam;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_average_path_length(
    scl_sparse_t adjacency,
    const scl_size_t max_samples,
    scl_real_t* avg_length) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(avg_length, "Output average length pointer is null");
    
    SCL_C_API_TRY
        Real length = Real(0);
        
        adjacency->visit([&](auto& m) {
            length = scl::kernel::components::average_path_length(m, max_samples);
        });
        
        *avg_length = static_cast<scl_real_t>(length);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clustering Coefficient
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficients) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(coefficients, "Output coefficients array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Real> coeffs(
                reinterpret_cast<Real*>(coefficients),
                static_cast<Size>(n_nodes)
            );
            scl::kernel::components::clustering_coefficient(m, coeffs);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_global_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficient) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(coefficient, "Output coefficient pointer is null");
    
    SCL_C_API_TRY
        Real coeff = Real(0);
        
        adjacency->visit([&](auto& m) {
            coeff = scl::kernel::components::global_clustering_coefficient(m);
        });
        
        *coefficient = static_cast<scl_real_t>(coeff);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Triangle Counting
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_count_triangles(
    scl_sparse_t adjacency,
    scl_size_t* n_triangles) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(n_triangles, "Output n_triangles pointer is null");
    
    SCL_C_API_TRY
        Size count = 0;
        
        adjacency->visit([&](auto& m) {
            count = scl::kernel::components::count_triangles(m);
        });
        
        *n_triangles = count;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Degree Statistics
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_degree_sequence(
    scl_sparse_t adjacency,
    scl_index_t* degrees) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(degrees, "Output degrees array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Index> degs(degrees, static_cast<Size>(n_nodes));
            scl::kernel::components::degree_sequence(m, degs);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_degree_statistics(
    scl_sparse_t adjacency,
    scl_real_t* mean_degree,
    scl_real_t* max_degree,
    scl_real_t* min_degree,
    scl_real_t* std_degree) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(mean_degree, "Output mean degree pointer is null");
    SCL_C_API_CHECK_NULL(max_degree, "Output max degree pointer is null");
    SCL_C_API_CHECK_NULL(min_degree, "Output min degree pointer is null");
    SCL_C_API_CHECK_NULL(std_degree, "Output std degree pointer is null");
    
    SCL_C_API_TRY
        Real mean = Real(0);
        Real max = Real(0);
        Real min = Real(0);
        Real std = Real(0);
        
        adjacency->visit([&](auto& m) {
            scl::kernel::components::degree_statistics(m, mean, max, min, std);
        });
        
        *mean_degree = static_cast<scl_real_t>(mean);
        *max_degree = static_cast<scl_real_t>(max);
        *min_degree = static_cast<scl_real_t>(min);
        *std_degree = static_cast<scl_real_t>(std);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_degree_distribution(
    scl_sparse_t adjacency,
    scl_size_t* histogram,
    const scl_index_t max_degree) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(histogram, "Output histogram array is null");
    SCL_C_API_CHECK(max_degree >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Max degree must be non-negative");
    
    SCL_C_API_TRY
        const Size hist_size = static_cast<Size>(max_degree + 1);
        
        adjacency->visit([&](auto& m) {
            Array<Size> hist(histogram, hist_size);
            scl::kernel::components::degree_distribution(m, hist, max_degree);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Graph Properties
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_graph_density(
    scl_sparse_t adjacency,
    scl_real_t* density) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(density, "Output density pointer is null");
    
    SCL_C_API_TRY
        Real dens = Real(0);
        
        adjacency->visit([&](auto& m) {
            dens = scl::kernel::components::graph_density(m);
        });
        
        *density = static_cast<scl_real_t>(dens);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// K-Core Decomposition
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_kcore_decomposition(
    scl_sparse_t adjacency,
    scl_index_t* core_numbers) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(core_numbers, "Output core numbers array is null");
    
    SCL_C_API_TRY
        const Index n_nodes = adjacency->rows();
        
        adjacency->visit([&](auto& m) {
            Array<Index> cores(core_numbers, static_cast<Size>(n_nodes));
            scl::kernel::components::kcore_decomposition(m, cores);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
