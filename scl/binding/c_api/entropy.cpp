// =============================================================================
// FILE: scl/binding/c_api/entropy/entropy.cpp
// BRIEF: C API implementation for information theory measures
// =============================================================================

#include "scl/binding/c_api/entropy.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/entropy.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

namespace {
    using namespace scl::kernel::entropy;
}

// =============================================================================
// Entropy Measures
// =============================================================================

scl_error_t scl_entropy_discrete_entropy(
    const scl_real_t* probabilities,
    scl_size_t n,
    int use_log2,
    scl_real_t* entropy_out
) {
    if (!probabilities || !entropy_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> probs_arr(
            reinterpret_cast<const scl::Real*>(probabilities), n
        );
        *entropy_out = static_cast<scl_real_t>(
            discrete_entropy(probs_arr, use_log2 != 0)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_count_entropy(
    const scl_real_t* counts,
    scl_size_t n,
    int use_log2,
    scl_real_t* entropy_out
) {
    if (!counts || !entropy_out) return SCL_ERROR_NULL_POINTER;
    try {
        *entropy_out = static_cast<scl_real_t>(
            count_entropy(
                reinterpret_cast<const scl::Real*>(counts),
                n, use_log2 != 0
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_row_entropy(
    scl_sparse_t X,
    scl_real_t* entropies,
    scl_size_t n_rows,
    int normalize,
    int use_log2
) {
    if (!X || !entropies) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<scl::Real> ent_arr(
            reinterpret_cast<scl::Real*>(entropies), n_rows
        );
        X->visit([&](auto& mat) {
            row_entropy(mat, ent_arr, normalize != 0, use_log2 != 0);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Divergence Measures
// =============================================================================

scl_error_t scl_entropy_kl_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* kl_out
) {
    if (!p || !q || !kl_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p), n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q), n
        );
        *kl_out = static_cast<scl_real_t>(
            kl_divergence(p_arr, q_arr, use_log2 != 0)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_js_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* js_out
) {
    if (!p || !q || !js_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p), n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q), n
        );
        *js_out = static_cast<scl_real_t>(
            js_divergence(p_arr, q_arr, use_log2 != 0)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_symmetric_kl(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* sym_kl_out
) {
    if (!p || !q || !sym_kl_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> p_arr(
            reinterpret_cast<const scl::Real*>(p), n
        );
        scl::Array<const scl::Real> q_arr(
            reinterpret_cast<const scl::Real*>(q), n
        );
        *sym_kl_out = static_cast<scl_real_t>(
            symmetric_kl(p_arr, q_arr, use_log2 != 0)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Mutual Information
// =============================================================================

scl_error_t scl_entropy_mutual_information(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* mi_out
) {
    if (!x_binned || !y_binned || !mi_out) return SCL_ERROR_NULL_POINTER;
    try {
        *mi_out = static_cast<scl_real_t>(
            mutual_information(
                reinterpret_cast<const scl::Index*>(x_binned),
                reinterpret_cast<const scl::Index*>(y_binned),
                n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_joint_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* joint_entropy_out
) {
    if (!x_binned || !y_binned || !joint_entropy_out) {
        return SCL_ERROR_NULL_POINTER;
    }
    try {
        *joint_entropy_out = static_cast<scl_real_t>(
            joint_entropy(
                reinterpret_cast<const scl::Index*>(x_binned),
                reinterpret_cast<const scl::Index*>(y_binned),
                n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_conditional_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* cond_entropy_out
) {
    if (!x_binned || !y_binned || !cond_entropy_out) {
        return SCL_ERROR_NULL_POINTER;
    }
    try {
        *cond_entropy_out = static_cast<scl_real_t>(
            conditional_entropy(
                reinterpret_cast<const scl::Index*>(x_binned),
                reinterpret_cast<const scl::Index*>(y_binned),
                n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_marginal_entropy(
    const scl_index_t* binned,
    scl_size_t n,
    scl_index_t n_bins,
    int use_log2,
    scl_real_t* marginal_entropy_out
) {
    if (!binned || !marginal_entropy_out) return SCL_ERROR_NULL_POINTER;
    try {
        *marginal_entropy_out = static_cast<scl_real_t>(
            marginal_entropy(
                reinterpret_cast<const scl::Index*>(binned),
                n, n_bins, use_log2 != 0
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Normalized Mutual Information
// =============================================================================

scl_error_t scl_entropy_normalized_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* nmi_out
) {
    if (!labels1 || !labels2 || !nmi_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Index> l1_arr(
            reinterpret_cast<const scl::Index*>(labels1), n
        );
        scl::Array<const scl::Index> l2_arr(
            reinterpret_cast<const scl::Index*>(labels2), n
        );
        *nmi_out = static_cast<scl_real_t>(
            normalized_mi(l1_arr, l2_arr, n_clusters1, n_clusters2)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_adjusted_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* ami_out
) {
    if (!labels1 || !labels2 || !ami_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Index> l1_arr(
            reinterpret_cast<const scl::Index*>(labels1), n
        );
        scl::Array<const scl::Index> l2_arr(
            reinterpret_cast<const scl::Index*>(labels2), n
        );
        *ami_out = static_cast<scl_real_t>(
            adjusted_mi(l1_arr, l2_arr, n_clusters1, n_clusters2)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Discretization
// =============================================================================

scl_error_t scl_entropy_discretize_equal_width(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
) {
    if (!values || !binned) return SCL_ERROR_NULL_POINTER;
    try {
        discretize_equal_width(
            reinterpret_cast<const scl::Real*>(values),
            n, n_bins, reinterpret_cast<scl::Index*>(binned)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_discretize_equal_frequency(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
) {
    if (!values || !binned) return SCL_ERROR_NULL_POINTER;
    try {
        discretize_equal_frequency(
            reinterpret_cast<const scl::Real*>(values),
            n, n_bins, reinterpret_cast<scl::Index*>(binned)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Other Measures
// =============================================================================

scl_error_t scl_entropy_cross_entropy(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n,
    scl_real_t* cross_entropy_out
) {
    if (!true_probs || !pred_probs || !cross_entropy_out) {
        return SCL_ERROR_NULL_POINTER;
    }
    try {
        scl::Array<const scl::Real> true_arr(
            reinterpret_cast<const scl::Real*>(true_probs), n
        );
        scl::Array<const scl::Real> pred_arr(
            reinterpret_cast<const scl::Real*>(pred_probs), n
        );
        *cross_entropy_out = static_cast<scl_real_t>(
            cross_entropy(true_arr, pred_arr)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_entropy_gini_impurity(
    const scl_real_t* probabilities,
    scl_size_t n,
    scl_real_t* gini_out
) {
    if (!probabilities || !gini_out) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> probs_arr(
            reinterpret_cast<const scl::Real*>(probabilities), n
        );
        *gini_out = static_cast<scl_real_t>(
            gini_impurity(probs_arr)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
