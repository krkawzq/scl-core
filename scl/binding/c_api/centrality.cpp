// =============================================================================
// FILE: scl/binding/c_api/centrality.cpp
// BRIEF: C API implementation for graph centrality measures
// =============================================================================

#include "scl/binding/c_api/centrality.h"
#include "scl/kernel/centrality.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

extern "C" {

scl_error_t scl_degree_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);
        scl::kernel::centrality::degree_centrality(*sparse, centrality_arr, normalize != 0);

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_weighted_degree_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);
        scl::kernel::centrality::weighted_degree_centrality(*sparse, centrality_arr, normalize != 0);

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_pagerank(
    scl_sparse_matrix_t adjacency,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        if (!adjacency || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> scores_arr(scores, N);
        scl::kernel::centrality::pagerank(
            *sparse,
            scores_arr,
            static_cast<scl::Real>(damping),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_personalized_pagerank(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        if (!adjacency || !seed_nodes || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<const scl::Index> seeds_arr(
            reinterpret_cast<const scl::Index*>(seed_nodes),
            n_seeds
        );
        scl::Array<scl::Real> scores_arr(scores, N);

        scl::kernel::centrality::personalized_pagerank(
            *sparse,
            seeds_arr,
            scores_arr,
            static_cast<scl::Real>(damping),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_hits(
    scl_sparse_matrix_t adjacency,
    scl_real_t* hub_scores,
    scl_real_t* authority_scores,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        if (!adjacency || !hub_scores || !authority_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> hub_arr(hub_scores, N);
        scl::Array<scl::Real> auth_arr(authority_scores, N);

        scl::kernel::centrality::hits(
            *sparse,
            hub_arr,
            auth_arr,
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_eigenvector_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::eigenvector_centrality(
            *sparse,
            centrality_arr,
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_katz_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_real_t alpha,
    scl_real_t beta,
    scl_index_t max_iter,
    scl_real_t tol
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::katz_centrality(
            *sparse,
            centrality_arr,
            static_cast<scl::Real>(alpha),
            static_cast<scl::Real>(beta),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_closeness_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::closeness_centrality(
            *sparse,
            centrality_arr,
            normalize != 0
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_betweenness_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::betweenness_centrality(
            *sparse,
            centrality_arr,
            normalize != 0
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_betweenness_centrality_sampled(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_index_t n_samples,
    int normalize,
    uint64_t seed
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::betweenness_centrality_sampled(
            *sparse,
            centrality_arr,
            static_cast<scl::Index>(n_samples),
            normalize != 0,
            seed
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_harmonic_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
) {
    try {
        if (!adjacency || !centrality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);

        scl::Array<scl::Real> centrality_arr(centrality, N);

        scl::kernel::centrality::harmonic_centrality(
            *sparse,
            centrality_arr,
            normalize != 0
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"

