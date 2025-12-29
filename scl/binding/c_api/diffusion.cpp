// =============================================================================
// FILE: scl/binding/c_api/diffusion/diffusion.cpp
// BRIEF: C API implementation for diffusion operations
// =============================================================================

#include "scl/binding/c_api/diffusion.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/diffusion.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Configuration Constants
// =============================================================================

namespace {
    constexpr Real DEFAULT_ALPHA = Real(0.85);
    constexpr Real DEFAULT_TOL = Real(1e-6);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Index DEFAULT_N_ITER = 50;
    constexpr Real DEFAULT_T = Real(1.0);
    constexpr Index DEFAULT_N_STEPS_HKS = 10;
    constexpr Real DEFAULT_LAZINESS = Real(0.5);
} // anonymous namespace

extern "C" {

// =============================================================================
// Transition Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_compute_transition_matrix(
    scl_sparse_t adjacency,
    const int symmetric) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& mat) {
            using MatType = std::remove_reference_t<decltype(mat)>;
            using T = typename MatType::ValueType;
            constexpr bool IsCSR = MatType::is_csr;
            
            // Note: compute_transition_matrix requires output buffer
            // This API modifies the adjacency matrix values in-place
            // by passing contiguous_data() as the output buffer
            Real* values = mat.contiguous_data();
            SCL_CHECK_ARG(values != nullptr, 
                         "Matrix must be contiguous for transition matrix computation");
            
            scl::kernel::diffusion::compute_transition_matrix<T, IsCSR>(
                mat, values, symmetric != 0
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Vector Diffusion
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_diffuse_vector(
    scl_sparse_t transition,
    scl_real_t* x,
    const scl_size_t n_nodes,
    const scl_index_t n_steps) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(x, "Vector x is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_steps > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Real> x_arr(reinterpret_cast<Real*>(x), n_nodes);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::diffuse_vector(mat, x_arr, n_steps);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Matrix Diffusion
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_diffuse_matrix(
    scl_sparse_t transition,
    scl_real_t* X,
    const scl_index_t n_nodes,
    const scl_index_t n_features,
    const scl_index_t n_steps) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(X, "Matrix X is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_features > 0 && n_steps > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total = static_cast<Size>(n_nodes) * static_cast<Size>(n_features);
        Array<Real> X_arr(reinterpret_cast<Real*>(X), total);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::diffuse_matrix(mat, X_arr, n_nodes, n_features, n_steps);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Diffusion Pseudotime (DPT)
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_compute_dpt(
    scl_sparse_t transition,
    const scl_index_t root_cell,
    scl_real_t* pseudotime,
    const scl_size_t n_nodes,
    const scl_index_t max_iter,
    const scl_real_t tol) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(pseudotime, "Output pseudotime array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        Array<Real> pt_arr(reinterpret_cast<Real*>(pseudotime), n_nodes);
        const Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        const Real tolerance = (tol == static_cast<scl_real_t>(0)) ? DEFAULT_TOL : static_cast<Real>(tol);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::compute_dpt(mat, root_cell, pt_arr, max_it, tolerance);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_diffusion_compute_dpt_multi_root(
    scl_sparse_t transition,
    const scl_index_t* root_cells,
    const scl_size_t n_roots,
    scl_real_t* pseudotime,
    const scl_size_t n_nodes,
    const scl_index_t max_iter) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(root_cells, "Root cells array is null");
    SCL_C_API_CHECK_NULL(pseudotime, "Output pseudotime array is null");
    SCL_C_API_CHECK(n_roots > 0 && n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> roots(root_cells, n_roots);
        Array<Real> pt_arr(reinterpret_cast<Real*>(pseudotime), n_nodes);
        const Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::compute_dpt_multi_root(mat, roots, pt_arr, max_it);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Random Walk with Restart
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_random_walk_with_restart(
    scl_sparse_t transition,
    const scl_index_t* seed_nodes,
    const scl_size_t n_seeds,
    scl_real_t* scores,
    const scl_size_t n_nodes,
    const scl_real_t alpha,
    const scl_index_t max_iter,
    const scl_real_t tol) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(seed_nodes, "Seed nodes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_seeds > 0 && n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    SCL_C_API_CHECK(alpha >= 0 && alpha <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "Alpha must be in [0, 1]");
    
    SCL_C_API_TRY
        Array<const Index> seeds(seed_nodes, n_seeds);
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), n_nodes);
        const Real alpha_val = (alpha == static_cast<scl_real_t>(0)) ? DEFAULT_ALPHA : static_cast<Real>(alpha);
        const Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        const Real tolerance = (tol == static_cast<scl_real_t>(0)) ? DEFAULT_TOL : static_cast<Real>(tol);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::random_walk_with_restart(
                mat, seeds, scores_arr, alpha_val, max_it, tolerance
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Personalized PageRank
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_personalized_pagerank(
    scl_sparse_t transition,
    const scl_index_t seed_node,
    scl_real_t* scores,
    const scl_size_t n_nodes,
    const scl_real_t alpha,
    const scl_index_t max_iter,
    const scl_real_t tol) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(alpha >= 0 && alpha <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "Alpha must be in [0, 1]");
    
    SCL_C_API_TRY
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), n_nodes);
        const Real alpha_val = (alpha == static_cast<scl_real_t>(0)) ? DEFAULT_ALPHA : static_cast<Real>(alpha);
        const Index max_it = (max_iter == 0) ? DEFAULT_MAX_ITER : max_iter;
        const Real tolerance = (tol == static_cast<scl_real_t>(0)) ? DEFAULT_TOL : static_cast<Real>(tol);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::personalized_pagerank(
                mat, seed_node, scores_arr, alpha_val, max_it, tolerance
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Diffusion Map Embedding
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_diffusion_map_embedding(
    scl_sparse_t transition,
    scl_real_t* embedding,
    const scl_index_t n_nodes,
    const scl_index_t n_components,
    const scl_index_t n_iter) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(embedding, "Output embedding array is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_components > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size embed_size = static_cast<Size>(n_nodes) * static_cast<Size>(n_components);
        Array<Real> embed_arr(reinterpret_cast<Real*>(embedding), embed_size);
        const Index n_it = (n_iter == 0) ? DEFAULT_N_ITER : n_iter;
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::diffusion_map_embedding(
                mat, embed_arr, n_components, n_it
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Heat Kernel Signature
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_heat_kernel_signature(
    scl_sparse_t transition,
    scl_real_t* signature,
    const scl_size_t n_nodes,
    const scl_real_t t,
    const scl_index_t n_steps) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(signature, "Output signature array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(t > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Time parameter t must be positive");
    
    SCL_C_API_TRY
        Array<Real> sig_arr(reinterpret_cast<Real*>(signature), n_nodes);
        const Real time_val = (t == static_cast<scl_real_t>(0)) ? DEFAULT_T : static_cast<Real>(t);
        const Index steps = (n_steps == 0) ? DEFAULT_N_STEPS_HKS : n_steps;
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::heat_kernel_signature(
                mat, sig_arr, time_val, steps
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// MAGIC Imputation
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_magic_impute(
    scl_sparse_t transition,
    scl_real_t* X,
    const scl_index_t n_nodes,
    const scl_index_t n_features,
    const scl_index_t t) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(X, "Matrix X is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total = static_cast<Size>(n_nodes) * static_cast<Size>(n_features);
        Array<Real> X_arr(reinterpret_cast<Real*>(X), total);
        const Index steps = (t == 0) ? DEFAULT_N_STEPS : t;
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::magic_impute(mat, X_arr, n_nodes, n_features, steps);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Diffusion Distance
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_diffusion_distance(
    scl_sparse_t transition,
    scl_real_t* distances,
    const scl_size_t n_nodes,
    const scl_index_t n_steps) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(distances, "Output distances array is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_steps > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total = static_cast<Size>(n_nodes) * static_cast<Size>(n_nodes);
        Array<Real> dist_arr(reinterpret_cast<Real*>(distances), total);
        const Index steps = (n_steps == 0) ? DEFAULT_N_STEPS : n_steps;
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::diffusion_distance(mat, dist_arr, steps);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Lazy Random Walk
// =============================================================================

SCL_EXPORT scl_error_t scl_diffusion_lazy_random_walk(
    scl_sparse_t transition,
    scl_real_t* x,
    const scl_size_t n_nodes,
    const scl_index_t n_steps,
    const scl_real_t laziness) {
    
    SCL_C_API_CHECK_NULL(transition, "Transition matrix is null");
    SCL_C_API_CHECK_NULL(x, "Vector x is null");
    SCL_C_API_CHECK(n_nodes > 0 && n_steps > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    SCL_C_API_CHECK(laziness >= 0 && laziness <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "Laziness must be in [0, 1]");
    
    SCL_C_API_TRY
        Array<Real> x_arr(reinterpret_cast<Real*>(x), n_nodes);
        const Real lazy_val = (laziness == static_cast<scl_real_t>(0)) ? DEFAULT_LAZINESS : static_cast<Real>(laziness);
        
        transition->visit([&](auto& mat) {
            scl::kernel::diffusion::lazy_random_walk(mat, x_arr, n_steps, lazy_val);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
