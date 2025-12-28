// =============================================================================
// FILE: scl/binding/c_api/kernels/metrics.cpp
// BRIEF: C API implementation for quality metrics
// =============================================================================

#include "metrics.h"
#include "scl/kernel/metrics.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/type.hpp"

namespace {

inline scl_error_t from_error_code(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

inline scl::CSR* to_sparse(scl_sparse_matrix_t handle) {
    return static_cast<scl::CSR*>(handle);
}

inline scl::Sparse<scl::Index, true>* to_sparse_index(scl_sparse_matrix_t handle) {
    return static_cast<scl::Sparse<scl::Index, true>*>(handle);
}

} // anonymous namespace

extern "C" {

scl_error_t scl_metrics_silhouette_score(
    scl_sparse_matrix_t distances,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_real_t* score
) {
    try {
        scl::CSR* dist = to_sparse(distances);
        if (!dist || !labels || !score) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_cells));
        *score = scl::kernel::metrics::silhouette_score(*dist, labels_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_silhouette_samples(
    scl_sparse_matrix_t distances,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_real_t* scores
) {
    try {
        scl::CSR* dist = to_sparse(distances);
        if (!dist || !labels || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_array(scores, static_cast<scl::Size>(n_cells));
        scl::kernel::metrics::silhouette_samples(*dist, labels_array, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_adjusted_rand_index(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_index_t n,
    scl_real_t* ari
) {
    try {
        if (!labels1 || !labels2 || !ari) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels1_array(labels1, static_cast<scl::Size>(n));
        scl::Array<const scl::Index> labels2_array(labels2, static_cast<scl::Size>(n));
        *ari = scl::kernel::metrics::adjusted_rand_index(labels1_array, labels2_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_normalized_mutual_information(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_index_t n,
    scl_real_t* nmi
) {
    try {
        if (!labels1 || !labels2 || !nmi) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels1_array(labels1, static_cast<scl::Size>(n));
        scl::Array<const scl::Index> labels2_array(labels2, static_cast<scl::Size>(n));
        *nmi = scl::kernel::metrics::normalized_mutual_information(labels1_array, labels2_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_graph_connectivity(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_real_t* connectivity
) {
    try {
        scl::CSR* adj = to_sparse(adjacency);
        if (!adj || !labels || !connectivity) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_cells));
        *connectivity = scl::kernel::metrics::graph_connectivity(*adj, labels_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_batch_entropy(
    scl_sparse_matrix_t neighbors,
    const scl_index_t* batch_labels,
    scl_index_t n_cells,
    scl_real_t* entropy_scores
) {
    try {
        scl::Sparse<scl::Index, true>* nbrs = to_sparse_index(neighbors);
        if (!nbrs || !batch_labels || !entropy_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(batch_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_array(entropy_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::metrics::batch_entropy<true>(*nbrs, labels_array, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_lisi(
    scl_sparse_matrix_t neighbors,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_real_t* lisi_scores
) {
    try {
        scl::Sparse<scl::Index, true>* nbrs = to_sparse_index(neighbors);
        if (!nbrs || !labels || !lisi_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores_array(lisi_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::metrics::lisi<true>(*nbrs, labels_array, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_fowlkes_mallows_index(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_index_t n,
    scl_real_t* fmi
) {
    try {
        if (!labels1 || !labels2 || !fmi) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels1_array(labels1, static_cast<scl::Size>(n));
        scl::Array<const scl::Index> labels2_array(labels2, static_cast<scl::Size>(n));
        *fmi = scl::kernel::metrics::fowlkes_mallows_index(labels1_array, labels2_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_v_measure(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    scl_index_t n,
    scl_real_t beta,
    scl_real_t* v_measure
) {
    try {
        if (!labels_true || !labels_pred || !v_measure) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> true_array(labels_true, static_cast<scl::Size>(n));
        scl::Array<const scl::Index> pred_array(labels_pred, static_cast<scl::Size>(n));
        *v_measure = scl::kernel::metrics::v_measure(true_array, pred_array, beta);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_metrics_purity_score(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    scl_index_t n,
    scl_real_t* purity
) {
    try {
        if (!labels_true || !labels_pred || !purity) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> true_array(labels_true, static_cast<scl::Size>(n));
        scl::Array<const scl::Index> pred_array(labels_pred, static_cast<scl::Size>(n));
        *purity = scl::kernel::metrics::purity_score(true_array, pred_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"
