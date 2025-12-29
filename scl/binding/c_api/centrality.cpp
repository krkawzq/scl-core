// =============================================================================
// FILE: scl/binding/c_api/centrality.cpp
// BRIEF: C API implementation for graph centrality measures
// =============================================================================

#include "scl/binding/c_api/centrality.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/centrality.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Degree Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_degree(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::degree_centrality(
                m, cent_arr, normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_centrality_weighted_degree(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::weighted_degree_centrality(
                m, cent_arr, normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// PageRank
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_pagerank(
    scl_sparse_t adjacency,
    scl_real_t* scores,
    const scl_size_t n_nodes,
    const scl_real_t damping,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(damping >= 0 && damping <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "Damping factor must be in [0, 1]");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> scores_arr(
                reinterpret_cast<Real*>(scores),
                n_nodes
            );
            scl::kernel::centrality::pagerank(
                m, scores_arr,
                static_cast<Real>(damping),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_centrality_personalized_pagerank(
    scl_sparse_t adjacency,
    const scl_index_t* seed_nodes,
    const scl_size_t n_seeds,
    scl_real_t* scores,
    const scl_size_t n_nodes,
    const scl_real_t damping,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(seed_nodes, "Seed nodes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(n_seeds > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of seed nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            const Array<const Index> seeds_arr(
                reinterpret_cast<const Index*>(seed_nodes),
                n_seeds
            );
            Array<Real> scores_arr(
                reinterpret_cast<Real*>(scores),
                n_nodes
            );
            scl::kernel::centrality::personalized_pagerank(
                m, seeds_arr, scores_arr,
                static_cast<Real>(damping),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// HITS Algorithm
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_hits(
    scl_sparse_t adjacency,
    scl_real_t* hub_scores,
    scl_real_t* authority_scores,
    const scl_size_t n_nodes,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(hub_scores, "Output hub scores array is null");
    SCL_C_API_CHECK_NULL(authority_scores, "Output authority scores array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> hub_arr(
                reinterpret_cast<Real*>(hub_scores),
                n_nodes
            );
            Array<Real> auth_arr(
                reinterpret_cast<Real*>(authority_scores),
                n_nodes
            );
            scl::kernel::centrality::hits(
                m, hub_arr, auth_arr,
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Eigenvector Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_eigenvector(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::eigenvector_centrality(
                m, cent_arr, max_iter, static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Katz Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_katz(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_real_t alpha,
    const scl_real_t beta,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::katz_centrality(
                m, cent_arr,
                static_cast<Real>(alpha),
                static_cast<Real>(beta),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Closeness Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_closeness(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::closeness_centrality(
                m, cent_arr, normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Betweenness Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_betweenness(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::betweenness_centrality(
                m, cent_arr, normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_centrality_betweenness_sampled(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_index_t n_samples,
    const scl_bool_t normalize,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(n_samples > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of samples must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::betweenness_centrality_sampled(
                m, cent_arr, n_samples, normalize != SCL_FALSE, seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Harmonic Centrality
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_harmonic(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_bool_t normalize) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::harmonic_centrality(
                m, cent_arr, normalize != SCL_FALSE
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Current Flow Betweenness (Approximate)
// =============================================================================

SCL_EXPORT scl_error_t scl_centrality_current_flow_approx(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    const scl_size_t n_nodes,
    const scl_index_t n_walks,
    const scl_index_t walk_length,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(centrality, "Output centrality array is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");
    SCL_C_API_CHECK(n_walks > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of walks must be positive");
    SCL_C_API_CHECK(walk_length > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Walk length must be positive");
    
    SCL_C_API_TRY
        adjacency->visit([&](auto& m) {
            Array<Real> cent_arr(
                reinterpret_cast<Real*>(centrality),
                n_nodes
            );
            scl::kernel::centrality::current_flow_betweenness_approx(
                m, cent_arr, n_walks, walk_length, seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
