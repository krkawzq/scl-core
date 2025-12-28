// =============================================================================
// FILE: scl/binding/c_api/centrality/centrality.cpp
// BRIEF: C API implementation for centrality measures
// =============================================================================

#include "scl/binding/c_api/centrality.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/centrality.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::centrality;

extern "C" {

scl_error_t scl_degree_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            degree_centrality(m, cent_arr, normalize != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_weighted_degree_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            weighted_degree_centrality(m, cent_arr, normalize != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_pagerank(
    scl_sparse_t adjacency,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            pagerank(m, scores_arr, damping, max_iter, tolerance);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_personalized_pagerank(
    scl_sparse_t adjacency,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !seed_nodes || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<const Index> seeds_arr(reinterpret_cast<const Index*>(seed_nodes), n_seeds);
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            personalized_pagerank(m, seeds_arr, scores_arr, damping, max_iter, tolerance);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hits(
    scl_sparse_t adjacency,
    scl_real_t* hub_scores,
    scl_real_t* authority_scores,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !hub_scores || !authority_scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> hub_arr(reinterpret_cast<Real*>(hub_scores), static_cast<Size>(n));
        Array<Real> auth_arr(reinterpret_cast<Real*>(authority_scores), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            hits(m, hub_arr, auth_arr, max_iter, tolerance);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_eigenvector_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            eigenvector_centrality(m, cent_arr, max_iter, tolerance);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_katz_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_real_t alpha,
    scl_real_t beta,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            katz_centrality(m, cent_arr, alpha, beta, max_iter, tolerance);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_closeness_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            closeness_centrality(m, cent_arr, normalize != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_betweenness_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            betweenness_centrality(m, cent_arr, normalize != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_betweenness_centrality_sampled(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_index_t n_samples,
    int normalize,
    uint64_t seed)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            betweenness_centrality_sampled(m, cent_arr, n_samples, normalize != 0, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_harmonic_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize)
{
    if (!adjacency || !centrality) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(adjacency);
        const Index n = wrapper->rows();
        Array<Real> cent_arr(reinterpret_cast<Real*>(centrality), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            harmonic_centrality(m, cent_arr, normalize != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

