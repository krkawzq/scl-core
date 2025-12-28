// =============================================================================
// FILE: scl/binding/c_api/entropy.cpp
// BRIEF: C API implementation for information theory measures
// =============================================================================

#include "scl/binding/c_api/entropy.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/entropy.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

// Internal helper to convert C++ exception to error code
static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_real_t scl_entropy_discrete_entropy(
    const scl_real_t* probabilities,
    scl_size_t n_probs,
    int use_log2
) {
    if (!probabilities || n_probs == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> probs_arr(
            reinterpret_cast<const scl::Real*>(probabilities),
            n_probs
        );
        return scl::kernel::entropy::discrete_entropy(probs_arr, use_log2 != 0);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_count_entropy(
    const scl_index_t* counts,
    scl_size_t n_counts,
    int use_log2
) {
    if (!counts || n_counts == 0) {
        return 0.0;
    }
    
    try {
        return scl::kernel::entropy::count_entropy(
            reinterpret_cast<const scl::Index*>(counts),
            n_counts,
            use_log2 != 0
        );
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_entropy_row_entropy(
    scl_sparse_matrix_t X,
    scl_real_t* entropies,
    scl_size_t n_rows,
    int normalize,
    int use_log2
) {
    if (!X || !entropies) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(X);
        scl::Array<scl::Real> entropies_arr(
            reinterpret_cast<scl::Real*>(entropies),
            n_rows
        );
        scl::kernel::entropy::row_entropy(
            *sparse,
            entropies_arr,
            normalize != 0,
            use_log2 != 0
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_entropy_kl_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
) {
    if (!p || !q || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p),
            n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q),
            n
        );
        return scl::kernel::entropy::kl_divergence(p_arr, q_arr, use_log2 != 0);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_js_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
) {
    if (!p || !q || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p),
            n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q),
            n
        );
        return scl::kernel::entropy::js_divergence(p_arr, q_arr, use_log2 != 0);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_symmetric_kl(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
) {
    if (!p || !q || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p),
            n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q),
            n
        );
        return scl::kernel::entropy::symmetric_kl(p_arr, q_arr, use_log2 != 0);
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_entropy_discretize_equal_width(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
) {
    if (!values || !binned || n == 0 || n_bins <= 0) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> values_arr(
            reinterpret_cast<const scl::Real*>(values),
            n
        );
        scl::Array<scl::Index> binned_arr(
            reinterpret_cast<scl::Index*>(binned),
            n
        );
        scl::kernel::entropy::discretize_equal_width(
            values_arr,
            static_cast<scl::Index>(n_bins),
            binned_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_entropy_discretize_equal_frequency(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
) {
    if (!values || !binned || n == 0 || n_bins <= 0) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> values_arr(
            reinterpret_cast<const scl::Real*>(values),
            n
        );
        scl::Array<scl::Index> binned_arr(
            reinterpret_cast<scl::Index*>(binned),
            n
        );
        scl::kernel::entropy::discretize_equal_frequency(
            values_arr,
            static_cast<scl::Index>(n_bins),
            binned_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_entropy_histogram_2d(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    scl_size_t* counts
) {
    if (!x_binned || !y_binned || !counts || n == 0 || n_bins_x <= 0 || n_bins_y <= 0) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> x_arr(
            reinterpret_cast<const scl::Index*>(x_binned),
            n
        );
        scl::Array<const scl::Index> y_arr(
            reinterpret_cast<const scl::Index*>(y_binned),
            n
        );
        scl::Array<scl::Size> counts_arr(
            reinterpret_cast<scl::Size*>(counts),
            static_cast<scl::Size>(n_bins_x) * static_cast<scl::Size>(n_bins_y)
        );
        scl::kernel::entropy::histogram_2d(
            x_arr,
            y_arr,
            static_cast<scl::Index>(n_bins_x),
            static_cast<scl::Index>(n_bins_y),
            counts_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_entropy_joint_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
) {
    if (!x_binned || !y_binned || n == 0 || n_bins_x <= 0 || n_bins_y <= 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> x_arr(
            reinterpret_cast<const scl::Index*>(x_binned),
            n
        );
        scl::Array<const scl::Index> y_arr(
            reinterpret_cast<const scl::Index*>(y_binned),
            n
        );
        return scl::kernel::entropy::joint_entropy(
            x_arr,
            y_arr,
            static_cast<scl::Index>(n_bins_x),
            static_cast<scl::Index>(n_bins_y),
            use_log2 != 0
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_marginal_entropy(
    const scl_index_t* binned,
    scl_size_t n,
    scl_index_t n_bins,
    int use_log2
) {
    if (!binned || n == 0 || n_bins <= 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> binned_arr(
            reinterpret_cast<const scl::Index*>(binned),
            n
        );
        return scl::kernel::entropy::marginal_entropy(
            binned_arr,
            static_cast<scl::Index>(n_bins),
            use_log2 != 0
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_conditional_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
) {
    if (!x_binned || !y_binned || n == 0 || n_bins_x <= 0 || n_bins_y <= 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> x_arr(
            reinterpret_cast<const scl::Index*>(x_binned),
            n
        );
        scl::Array<const scl::Index> y_arr(
            reinterpret_cast<const scl::Index*>(y_binned),
            n
        );
        return scl::kernel::entropy::conditional_entropy(
            x_arr,
            y_arr,
            static_cast<scl::Index>(n_bins_x),
            static_cast<scl::Index>(n_bins_y),
            use_log2 != 0
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_mutual_information(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
) {
    if (!x_binned || !y_binned || n == 0 || n_bins_x <= 0 || n_bins_y <= 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> x_arr(
            reinterpret_cast<const scl::Index*>(x_binned),
            n
        );
        scl::Array<const scl::Index> y_arr(
            reinterpret_cast<const scl::Index*>(y_binned),
            n
        );
        return scl::kernel::entropy::mutual_information(
            x_arr,
            y_arr,
            static_cast<scl::Index>(n_bins_x),
            static_cast<scl::Index>(n_bins_y),
            use_log2 != 0
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_normalized_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2
) {
    if (!labels1 || !labels2 || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> labels1_arr(
            reinterpret_cast<const scl::Index*>(labels1),
            n
        );
        scl::Array<const scl::Index> labels2_arr(
            reinterpret_cast<const scl::Index*>(labels2),
            n
        );
        return scl::kernel::entropy::normalized_mi(
            labels1_arr,
            labels2_arr,
            static_cast<scl::Index>(n_clusters1),
            static_cast<scl::Index>(n_clusters2)
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_adjusted_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2
) {
    if (!labels1 || !labels2 || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> labels1_arr(
            reinterpret_cast<const scl::Index*>(labels1),
            n
        );
        scl::Array<const scl::Index> labels2_arr(
            reinterpret_cast<const scl::Index*>(labels2),
            n
        );
        return scl::kernel::entropy::adjusted_mi(
            labels1_arr,
            labels2_arr,
            static_cast<scl::Index>(n_clusters1),
            static_cast<scl::Index>(n_clusters2)
        );
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_entropy_select_features_mi(
    scl_sparse_matrix_t X,
    const scl_index_t* target,
    scl_size_t n_samples,
    scl_index_t n_features,
    scl_index_t n_to_select,
    scl_index_t* selected_features,
    scl_real_t* mi_scores,
    scl_index_t n_bins
) {
    if (!X || !target || !selected_features || !mi_scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(X);
        scl::Array<const scl::Index> target_arr(
            reinterpret_cast<const scl::Index*>(target),
            n_samples
        );
        scl::Array<scl::Index> selected_arr(
            reinterpret_cast<scl::Index*>(selected_features),
            static_cast<scl::Size>(n_to_select)
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(mi_scores),
            static_cast<scl::Size>(n_features)
        );
        scl::kernel::entropy::select_features_mi(
            *sparse,
            target_arr,
            static_cast<scl::Index>(n_features),
            static_cast<scl::Index>(n_to_select),
            selected_arr,
            scores_arr,
            static_cast<scl::Index>(n_bins)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_entropy_mrmr_selection(
    scl_sparse_matrix_t X,
    const scl_index_t* target,
    scl_size_t n_samples,
    scl_index_t n_features,
    scl_index_t n_to_select,
    scl_index_t* selected_features,
    scl_index_t n_bins
) {
    if (!X || !target || !selected_features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(X);
        scl::Array<const scl::Index> target_arr(
            reinterpret_cast<const scl::Index*>(target),
            n_samples
        );
        scl::Array<scl::Index> selected_arr(
            reinterpret_cast<scl::Index*>(selected_features),
            static_cast<scl::Size>(n_to_select)
        );
        scl::kernel::entropy::mrmr_selection(
            *sparse,
            target_arr,
            static_cast<scl::Index>(n_features),
            static_cast<scl::Index>(n_to_select),
            selected_arr,
            static_cast<scl::Index>(n_bins)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_entropy_cross_entropy(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n
) {
    if (!true_probs || !pred_probs || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> true_arr(
            reinterpret_cast<const scl::Real*>(true_probs),
            n
        );
        scl::Array<const scl::Real> pred_arr(
            reinterpret_cast<const scl::Real*>(pred_probs),
            n
        );
        return scl::kernel::entropy::cross_entropy(true_arr, pred_arr);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_perplexity(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n
) {
    if (!true_probs || !pred_probs || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> true_arr(
            reinterpret_cast<const scl::Real*>(true_probs),
            n
        );
        scl::Array<const scl::Real> pred_arr(
            reinterpret_cast<const scl::Real*>(pred_probs),
            n
        );
        return scl::kernel::entropy::perplexity(true_arr, pred_arr);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_information_gain(
    const scl_index_t* feature_binned,
    const scl_index_t* target,
    scl_size_t n,
    scl_index_t n_feature_bins,
    scl_index_t n_target_classes
) {
    if (!feature_binned || !target || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Index> feature_arr(
            reinterpret_cast<const scl::Index*>(feature_binned),
            n
        );
        scl::Array<const scl::Index> target_arr(
            reinterpret_cast<const scl::Index*>(target),
            n
        );
        return scl::kernel::entropy::information_gain(
            feature_arr,
            target_arr,
            static_cast<scl::Index>(n_feature_bins),
            static_cast<scl::Index>(n_target_classes)
        );
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_entropy_gini_impurity(
    const scl_real_t* probabilities,
    scl_size_t n
) {
    if (!probabilities || n == 0) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> probs_arr(
            reinterpret_cast<const scl::Real*>(probabilities),
            n
        );
        return scl::kernel::entropy::gini_impurity(probs_arr);
    } catch (...) {
        return 0.0;
    }
}

// Explicit instantiation
template void scl::kernel::entropy::row_entropy<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>,
    bool,
    bool
);

template void scl::kernel::entropy::select_features_mi<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Index,
    scl::Index,
    scl::Array<scl::Index>,
    scl::Array<scl::Real>,
    scl::Index
);

template void scl::kernel::entropy::mrmr_selection<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Index,
    scl::Index,
    scl::Array<scl::Index>,
    scl::Index
);

} // extern "C"

