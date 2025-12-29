// =============================================================================
// FILE: scl/binding/c_api/metrics.cpp
// BRIEF: C API implementation for quality metrics
// =============================================================================

#include "scl/binding/c_api/metrics.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/metrics.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Silhouette Score
// =============================================================================

SCL_EXPORT scl_error_t scl_metrics_silhouette_score(
    scl_sparse_t distances,
    const scl_index_t* labels,
    const scl_size_t n_cells,
    scl_real_t* score) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(labels, "Labels pointer is null");
    SCL_C_API_CHECK_NULL(score, "Score output pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(labels), n_cells_sz);

        Real score_result = Real(0);
        
        distances->visit([&](auto& dist) {
            score_result = scl::kernel::metrics::silhouette_score(dist, labels_arr);
        });
        
        *score = static_cast<scl_real_t>(score_result);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Silhouette Samples
// =============================================================================

SCL_EXPORT scl_error_t scl_metrics_silhouette_samples(
    scl_sparse_t distances,
    const scl_index_t* labels,
    const scl_size_t n_cells,
    scl_real_t* scores) {
    
    SCL_C_API_CHECK_NULL(distances, "Distances matrix handle is null");
    SCL_C_API_CHECK_NULL(labels, "Labels pointer is null");
    SCL_C_API_CHECK_NULL(scores, "Scores output pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(labels), n_cells_sz);
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), n_cells_sz);

        distances->visit([&](auto& dist) {
            scl::kernel::metrics::silhouette_samples(dist, labels_arr, scores_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Adjusted Rand Index
// =============================================================================

scl_error_t scl_metrics_adjusted_rand_index(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* ari)
{
    SCL_C_API_CHECK_NULL(labels1, "Labels1 pointer is null");
    SCL_C_API_CHECK_NULL(labels2, "Labels2 pointer is null");
    SCL_C_API_CHECK_NULL(ari, "ARI output pointer is null");

    SCL_C_API_TRY {
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels1_arr(reinterpret_cast<const Index*>(labels1), n_cells_sz);
        Array<const Index> labels2_arr(reinterpret_cast<const Index*>(labels2), n_cells_sz);

        const Real ari_result = scl::kernel::metrics::adjusted_rand_index(labels1_arr, labels2_arr);
        *ari = static_cast<scl_real_t>(ari_result);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Normalized Mutual Information
// =============================================================================

scl_error_t scl_metrics_normalized_mutual_information(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* nmi)
{
    SCL_C_API_CHECK_NULL(labels1, "Labels1 pointer is null");
    SCL_C_API_CHECK_NULL(labels2, "Labels2 pointer is null");
    SCL_C_API_CHECK_NULL(nmi, "NMI output pointer is null");

    SCL_C_API_TRY {
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels1_arr(reinterpret_cast<const Index*>(labels1), n_cells_sz);
        Array<const Index> labels2_arr(reinterpret_cast<const Index*>(labels2), n_cells_sz);

        const Real nmi_result = scl::kernel::metrics::normalized_mutual_information(labels1_arr, labels2_arr);
        *nmi = static_cast<scl_real_t>(nmi_result);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Homogeneity Score
// =============================================================================

SCL_EXPORT scl_error_t scl_metrics_homogeneity_score(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    const scl_size_t n_cells,
    scl_real_t* homogeneity) {
    
    SCL_C_API_CHECK_NULL(labels_true, "True labels pointer is null");
    SCL_C_API_CHECK_NULL(labels_pred, "Predicted labels pointer is null");
    SCL_C_API_CHECK_NULL(homogeneity, "Homogeneity output pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels_true_arr(reinterpret_cast<const Index*>(labels_true), n_cells_sz);
        Array<const Index> labels_pred_arr(reinterpret_cast<const Index*>(labels_pred), n_cells_sz);

        const Real h = scl::kernel::metrics::homogeneity_score(labels_true_arr, labels_pred_arr);
        *homogeneity = static_cast<scl_real_t>(h);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Completeness Score
// =============================================================================

SCL_EXPORT scl_error_t scl_metrics_completeness_score(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    const scl_size_t n_cells,
    scl_real_t* completeness) {
    
    SCL_C_API_CHECK_NULL(labels_true, "True labels pointer is null");
    SCL_C_API_CHECK_NULL(labels_pred, "Predicted labels pointer is null");
    SCL_C_API_CHECK_NULL(completeness, "Completeness output pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels_true_arr(reinterpret_cast<const Index*>(labels_true), n_cells_sz);
        Array<const Index> labels_pred_arr(reinterpret_cast<const Index*>(labels_pred), n_cells_sz);

        const Real c = scl::kernel::metrics::completeness_score(labels_true_arr, labels_pred_arr);
        *completeness = static_cast<scl_real_t>(c);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// V-Measure Score
// =============================================================================

SCL_EXPORT scl_error_t scl_metrics_v_measure(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    const scl_size_t n_cells,
    scl_real_t* v_measure) {
    
    SCL_C_API_CHECK_NULL(labels_true, "True labels pointer is null");
    SCL_C_API_CHECK_NULL(labels_pred, "Predicted labels pointer is null");
    SCL_C_API_CHECK_NULL(v_measure, "V-measure output pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Size n_cells_sz = static_cast<Size>(n_cells);
        Array<const Index> labels_true_arr(reinterpret_cast<const Index*>(labels_true), n_cells_sz);
        Array<const Index> labels_pred_arr(reinterpret_cast<const Index*>(labels_pred), n_cells_sz);

        const Real v = scl::kernel::metrics::v_measure(labels_true_arr, labels_pred_arr, Real(1.0));
        *v_measure = static_cast<scl_real_t>(v);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
