// =============================================================================
// FILE: scl/binding/c_api/components/components.cpp
// BRIEF: C API implementation for graph components analysis
// =============================================================================

#include "scl/binding/c_api/components.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/components.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_comp_connected_components(
    scl_sparse_t adjacency,
    scl_index_t* component_labels,
    scl_index_t* n_components
) {
    if (!adjacency || !component_labels || !n_components) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_comp = 0;

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> labels(component_labels, static_cast<scl::Size>(m.primary_dim()));
            scl::kernel::components::connected_components(m, labels, n_comp);
        });

        *n_components = n_comp;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_is_connected(
    scl_sparse_t adjacency,
    int* is_connected
) {
    if (!adjacency || !is_connected) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        bool connected = false;

        sparse->visit([&](auto& m) {
            connected = scl::kernel::components::is_connected(m);
        });

        *is_connected = connected ? 1 : 0;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_largest_component(
    scl_sparse_t adjacency,
    scl_index_t* node_mask,
    scl_index_t* component_size
) {
    if (!adjacency || !node_mask || !component_size) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index size = 0;

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> mask(node_mask, static_cast<scl::Size>(m.primary_dim()));
            scl::kernel::components::largest_component(m, mask, size);
        });

        *component_size = size;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_component_sizes(
    scl_sparse_t adjacency,
    scl_index_t* sizes,
    scl_index_t* n_components
) {
    if (!adjacency || !sizes || !n_components) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_comp = 0;

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> sizes_arr(sizes, 0);
            scl::kernel::components::component_sizes(m, sizes_arr, n_comp);
        });

        *n_components = n_comp;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_bfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* distances,
    scl_index_t* predecessors
) {
    if (!adjacency || !distances) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> dist(distances, static_cast<scl::Size>(n_nodes));
            scl::Array<scl::Index> pred;
            if (predecessors) {
                pred = scl::Array<scl::Index>(predecessors, static_cast<scl::Size>(n_nodes));
            }
            scl::kernel::components::bfs(m, source, dist, pred);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_multi_source_bfs(
    scl_sparse_t adjacency,
    const scl_index_t* sources,
    scl_size_t n_sources,
    scl_index_t* distances
) {
    if (!adjacency || !sources || !distances) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> srcs(sources, n_sources);
            scl::Array<scl::Index> dist(distances, static_cast<scl::Size>(n_nodes));
            scl::kernel::components::multi_source_bfs(m, srcs, dist);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_parallel_bfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* distances
) {
    if (!adjacency || !distances) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> dist(distances, static_cast<scl::Size>(n_nodes));
            scl::kernel::components::parallel_bfs(m, source, dist);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_dfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* discovery_time,
    scl_index_t* finish_time
) {
    if (!adjacency || !discovery_time || !finish_time) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> disc(discovery_time, static_cast<scl::Size>(n_nodes));
            scl::Array<scl::Index> fin(finish_time, static_cast<scl::Size>(n_nodes));
            scl::kernel::components::dfs(m, source, disc, fin);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_topological_sort(
    scl_sparse_t adjacency,
    scl_index_t* order,
    int* is_valid
) {
    if (!adjacency || !order || !is_valid) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();
        bool valid = false;

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> ord(order, static_cast<scl::Size>(n_nodes));
            valid = scl::kernel::components::topological_sort(m, ord);
        });

        *is_valid = valid ? 1 : 0;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_graph_diameter(
    scl_sparse_t adjacency,
    scl_index_t* diameter
) {
    if (!adjacency || !diameter) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index diam = 0;

        sparse->visit([&](auto& m) {
            diam = scl::kernel::components::graph_diameter(m);
        });

        *diameter = diam;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_average_path_length(
    scl_sparse_t adjacency,
    scl_size_t max_samples,
    scl_real_t* avg_length
) {
    if (!adjacency || !avg_length) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Real length = 0;

        sparse->visit([&](auto& m) {
            length = scl::kernel::components::average_path_length(m, max_samples);
        });

        *avg_length = static_cast<scl_real_t>(length);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficients
) {
    if (!adjacency || !coefficients) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> coeffs(reinterpret_cast<scl::Real*>(coefficients),
                                        static_cast<scl::Size>(n_nodes));
            scl::kernel::components::clustering_coefficient(m, coeffs);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_global_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficient
) {
    if (!adjacency || !coefficient) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Real coeff = 0;

        sparse->visit([&](auto& m) {
            coeff = scl::kernel::components::global_clustering_coefficient(m);
        });

        *coefficient = static_cast<scl_real_t>(coeff);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_count_triangles(
    scl_sparse_t adjacency,
    scl_size_t* n_triangles
) {
    if (!adjacency || !n_triangles) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Size count = 0;

        sparse->visit([&](auto& m) {
            count = scl::kernel::components::count_triangles(m);
        });

        *n_triangles = count;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_degree_sequence(
    scl_sparse_t adjacency,
    scl_index_t* degrees
) {
    if (!adjacency || !degrees) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> degs(degrees, static_cast<scl::Size>(n_nodes));
            scl::kernel::components::degree_sequence(m, degs);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_degree_statistics(
    scl_sparse_t adjacency,
    scl_real_t* mean_degree,
    scl_real_t* max_degree,
    scl_real_t* min_degree,
    scl_real_t* std_degree
) {
    if (!adjacency || !mean_degree || !max_degree || !min_degree || !std_degree) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Real mean = 0, max = 0, min = 0, std = 0;

        sparse->visit([&](auto& m) {
            scl::kernel::components::degree_statistics(m, mean, max, min, std);
        });

        *mean_degree = static_cast<scl_real_t>(mean);
        *max_degree = static_cast<scl_real_t>(max);
        *min_degree = static_cast<scl_real_t>(min);
        *std_degree = static_cast<scl_real_t>(std);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_degree_distribution(
    scl_sparse_t adjacency,
    scl_size_t* histogram,
    scl_index_t max_degree
) {
    if (!adjacency || !histogram) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Size hist_size = static_cast<scl::Size>(max_degree + 1);

        sparse->visit([&](auto& m) {
            scl::Array<scl::Size> hist(histogram, hist_size);
            scl::kernel::components::degree_distribution(m, hist, max_degree);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_graph_density(
    scl_sparse_t adjacency,
    scl_real_t* density
) {
    if (!adjacency || !density) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Real dens = 0;

        sparse->visit([&](auto& m) {
            dens = scl::kernel::components::graph_density(m);
        });

        *density = static_cast<scl_real_t>(dens);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_kcore_decomposition(
    scl_sparse_t adjacency,
    scl_index_t* core_numbers
) {
    if (!adjacency || !core_numbers) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(adjacency);
        scl::Index n_nodes = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Index> cores(core_numbers, static_cast<scl::Size>(n_nodes));
            scl::kernel::components::kcore_decomposition(m, cores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
