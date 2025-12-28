#include "diffusion.h"
#include "scl/kernel/diffusion.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include <cstring>

// Helper to convert error code
static scl_error_t convert_error(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

// Helper to unwrap sparse matrix
static scl::Sparse<scl::Real, true>* unwrap_matrix(scl_sparse_matrix_t mat) {
    return static_cast<scl::Sparse<scl::Real, true>*>(mat);
}

extern "C" {

scl_error_t scl_diffusion_compute_transition_matrix(
    scl_sparse_matrix_t adjacency,
    scl_real_t* transition_values,
    int symmetric
) {
    try {
        auto* mat = unwrap_matrix(adjacency);
        if (!mat || !transition_values) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::kernel::diffusion::compute_transition_matrix(
            *mat, transition_values, symmetric != 0
        );
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_diffuse_vector(
    scl_sparse_matrix_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !x) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<scl::Real> x_arr(x, n_nodes);
        scl::kernel::diffusion::diffuse_vector(*mat, x_arr, n_steps);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_diffuse_matrix(
    scl_sparse_matrix_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t n_steps
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !X) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_features);
        scl::Array<scl::Real> X_arr(X, total);
        scl::kernel::diffusion::diffuse_matrix(*mat, X_arr, n_nodes, n_features, n_steps);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_compute_dpt(
    scl_sparse_matrix_t transition,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !pseudotime) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<scl::Real> pt_arr(pseudotime, n_nodes);
        scl::kernel::diffusion::compute_dpt(*mat, root_cell, pt_arr, max_iter, tol);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_compute_dpt_multi_root(
    scl_sparse_matrix_t transition,
    const scl_index_t* root_cells,
    scl_size_t n_roots,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !root_cells || !pseudotime) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<const scl::Index> roots_arr(root_cells, n_roots);
        scl::Array<scl::Real> pt_arr(pseudotime, n_nodes);
        scl::kernel::diffusion::compute_dpt_multi_root(*mat, roots_arr, pt_arr, max_iter);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_random_walk_with_restart(
    scl_sparse_matrix_t transition,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !seed_nodes || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<const scl::Index> seeds_arr(seed_nodes, n_seeds);
        scl::Array<scl::Real> scores_arr(scores, n_nodes);
        scl::kernel::diffusion::random_walk_with_restart(
            *mat, seeds_arr, scores_arr, alpha, max_iter, tol
        );
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_diffusion_map_embedding(
    scl_sparse_matrix_t transition,
    scl_real_t* embedding,
    scl_size_t n_nodes,
    scl_index_t n_components,
    scl_index_t n_iter
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !embedding) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Size total = n_nodes * static_cast<scl::Size>(n_components);
        scl::Array<scl::Real> emb_arr(embedding, total);
        scl::kernel::diffusion::diffusion_map_embedding(*mat, emb_arr, n_components, n_iter);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_heat_kernel_signature(
    scl_sparse_matrix_t transition,
    scl_real_t* signature,
    scl_size_t n_nodes,
    scl_real_t t,
    scl_index_t n_steps
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !signature) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<scl::Real> sig_arr(signature, n_nodes);
        scl::kernel::diffusion::heat_kernel_signature(*mat, sig_arr, t, n_steps);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_magic_impute(
    scl_sparse_matrix_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t t
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !X) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_features);
        scl::Array<scl::Real> X_arr(X, total);
        scl::kernel::diffusion::magic_impute(*mat, X_arr, n_nodes, n_features, t);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_diffusion_distance(
    scl_sparse_matrix_t transition,
    scl_real_t* distances,
    scl_size_t n_nodes,
    scl_index_t n_steps
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !distances) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Size total = n_nodes * n_nodes;
        scl::Array<scl::Real> dist_arr(distances, total);
        scl::kernel::diffusion::diffusion_distance(*mat, dist_arr, n_steps);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_personalized_pagerank(
    scl_sparse_matrix_t transition,
    scl_index_t seed_node,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<scl::Real> scores_arr(scores, n_nodes);
        scl::kernel::diffusion::personalized_pagerank(
            *mat, seed_node, scores_arr, alpha, max_iter, tol
        );
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_lazy_random_walk(
    scl_sparse_matrix_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps,
    scl_real_t laziness
) {
    try {
        auto* mat = unwrap_matrix(transition);
        if (!mat || !x) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        scl::Array<scl::Real> x_arr(x, n_nodes);
        scl::kernel::diffusion::lazy_random_walk(*mat, x_arr, n_steps, laziness);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

} // extern "C"
