// =============================================================================
// FILE: scl/binding/c_api/diffusion/diffusion.cpp
// BRIEF: C API implementation for diffusion operations
// =============================================================================

#include "scl/binding/c_api/diffusion.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/diffusion.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

namespace {
    using namespace scl::kernel::diffusion;
    constexpr scl::Real DEFAULT_ALPHA = scl::Real(0.85);
    constexpr scl::Real DEFAULT_TOL = scl::Real(1e-6);
    constexpr scl::Index DEFAULT_MAX_ITER = 100;
    constexpr scl::Index DEFAULT_N_STEPS = 3;
    constexpr scl::Index DEFAULT_N_ITER = 50;
    constexpr scl::Real DEFAULT_T = scl::Real(1.0);
    constexpr scl::Index DEFAULT_N_STEPS_HKS = 10;
    constexpr scl::Real DEFAULT_LAZINESS = scl::Real(0.5);
}

scl_error_t scl_diffusion_compute_transition_matrix(
    scl_sparse_t adjacency,
    int symmetric
) {
    if (!adjacency) return SCL_ERROR_NULL_POINTER;
    try {
        adjacency->visit([&](auto& mat) {
            using MatType = std::remove_reference_t<decltype(mat)>;
            using T = typename MatType::ValueType;
            constexpr bool IsCSR = MatType::is_csr;
            
            // compute_transition_matrix modifies matrix in-place
            scl::kernel::diffusion::compute_transition_matrix<T, IsCSR>(mat, symmetric != 0);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_diffuse_vector(
    scl_sparse_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps
) {
    if (!transition || !x) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> x_arr(reinterpret_cast<scl::Real*>(x), n_nodes);
        transition->visit([&](auto& mat) {
            diffuse_vector(mat, x_arr, n_steps);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_diffuse_matrix(
    scl_sparse_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t n_steps
) {
    if (!transition || !X) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_features);
        scl::Array<scl::Real> X_arr(reinterpret_cast<scl::Real*>(X), total);
        transition->visit([&](auto& mat) {
            diffuse_matrix(mat, X_arr, n_nodes, n_features, n_steps);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_compute_dpt(
    scl_sparse_t transition,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!transition || !pseudotime) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> pt_arr(reinterpret_cast<scl::Real*>(pseudotime), n_nodes);
        scl::Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        scl::Real tolerance = (tol == scl::Real(0)) ? DEFAULT_TOL : tol;
        transition->visit([&](auto& mat) {
            compute_dpt(mat, root_cell, pt_arr, max_it, tolerance);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_compute_dpt_multi_root(
    scl_sparse_t transition,
    const scl_index_t* root_cells,
    scl_size_t n_roots,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter
) {
    if (!transition || !root_cells || !pseudotime) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Index> roots(
            reinterpret_cast<const scl::Index*>(root_cells), n_roots
        );
        scl::Array<scl::Real> pt_arr(reinterpret_cast<scl::Real*>(pseudotime), n_nodes);
        scl::Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        transition->visit([&](auto& mat) {
            compute_dpt_multi_root(mat, roots, pt_arr, max_it);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_random_walk_with_restart(
    scl_sparse_t transition,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!transition || !seed_nodes || !scores) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Index> seeds(
            reinterpret_cast<const scl::Index*>(seed_nodes), n_seeds
        );
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(scores), n_nodes);
        scl::Real a = (alpha == scl::Real(0)) ? DEFAULT_ALPHA : alpha;
        scl::Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        scl::Real tolerance = (tol == scl::Real(0)) ? DEFAULT_TOL : tol;
        transition->visit([&](auto& mat) {
            random_walk_with_restart(mat, seeds, scores_arr, a, max_it, tolerance);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_personalized_pagerank(
    scl_sparse_t transition,
    scl_index_t seed_node,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!transition || !scores) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> scores_arr(reinterpret_cast<scl::Real*>(scores), n_nodes);
        scl::Real a = (alpha == scl::Real(0)) ? DEFAULT_ALPHA : alpha;
        scl::Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        scl::Real tolerance = (tol == scl::Real(0)) ? DEFAULT_TOL : tol;
        transition->visit([&](auto& mat) {
            personalized_pagerank(mat, seed_node, scores_arr, a, max_it, tolerance);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_diffusion_map_embedding(
    scl_sparse_t transition,
    scl_real_t* embedding,
    scl_index_t n_nodes,
    scl_index_t n_components,
    scl_index_t n_iter
) {
    if (!transition || !embedding) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_components);
        scl::Array<scl::Real> emb_arr(reinterpret_cast<scl::Real*>(embedding), total);
        scl::Index n_it = (n_iter == 0) ? DEFAULT_N_ITER : n_iter;
        transition->visit([&](auto& mat) {
            diffusion_map_embedding(mat, emb_arr, n_components, n_it);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_heat_kernel_signature(
    scl_sparse_t transition,
    scl_real_t* signature,
    scl_size_t n_nodes,
    scl_real_t t,
    scl_index_t n_steps
) {
    if (!transition || !signature) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> sig_arr(reinterpret_cast<scl::Real*>(signature), n_nodes);
        scl::Real time = (t == scl::Real(0)) ? DEFAULT_T : t;
        scl::Index steps = (n_steps == 0) ? DEFAULT_N_STEPS_HKS : n_steps;
        transition->visit([&](auto& mat) {
            heat_kernel_signature(mat, sig_arr, time, steps);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_magic_impute(
    scl_sparse_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t t
) {
    if (!transition || !X) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_features);
        scl::Array<scl::Real> X_arr(reinterpret_cast<scl::Real*>(X), total);
        scl::Index steps = (t == 0) ? DEFAULT_N_STEPS : t;
        transition->visit([&](auto& mat) {
            magic_impute(mat, X_arr, n_nodes, n_features, steps);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_diffusion_distance(
    scl_sparse_t transition,
    scl_real_t* distances,
    scl_size_t n_nodes,
    scl_index_t n_steps
) {
    if (!transition || !distances) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Size total = static_cast<scl::Size>(n_nodes) * static_cast<scl::Size>(n_nodes);
        scl::Array<scl::Real> dist_arr(reinterpret_cast<scl::Real*>(distances), total);
        scl::Index steps = (n_steps == 0) ? DEFAULT_N_STEPS : n_steps;
        transition->visit([&](auto& mat) {
            diffusion_distance(mat, dist_arr, steps);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_diffusion_lazy_random_walk(
    scl_sparse_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps,
    scl_real_t laziness
) {
    if (!transition || !x) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> x_arr(reinterpret_cast<scl::Real*>(x), n_nodes);
        scl::Real laz = (laziness == scl::Real(0)) ? DEFAULT_LAZINESS : laziness;
        transition->visit([&](auto& mat) {
            lazy_random_walk(mat, x_arr, n_steps, laz);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
