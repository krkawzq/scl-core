// =============================================================================
// FILE: scl/binding/c_api/permutation/permutation.cpp
// BRIEF: C API implementation for permutation testing
// =============================================================================

#include "scl/binding/c_api/permutation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/permutation.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_perm_correlation_test(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t n,
    scl_real_t observed_correlation,
    scl_size_t n_permutations,
    scl_real_t* p_value,
    uint64_t seed
) {
    if (!x || !y || !p_value) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> x_arr(reinterpret_cast<const scl::Real*>(x), n);
        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y), n);

        scl::Real pval = scl::kernel::permutation::permutation_correlation_test(
            x_arr, y_arr, static_cast<scl::Real>(observed_correlation),
            n_permutations, seed
        );

        *p_value = static_cast<scl_real_t>(pval);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_fdr_correction_bh(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* q_values
) {
    if (!p_values || !q_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Array<scl::Real> qvals(reinterpret_cast<scl::Real*>(q_values), n);

        scl::kernel::permutation::fdr_correction_bh(pvals, qvals);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_fdr_correction_by(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* q_values
) {
    if (!p_values || !q_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Array<scl::Real> qvals(reinterpret_cast<scl::Real*>(q_values), n);

        scl::kernel::permutation::fdr_correction_by(pvals, qvals);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_bonferroni_correction(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* adjusted
) {
    if (!p_values || !adjusted) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Array<scl::Real> adj(reinterpret_cast<scl::Real*>(adjusted), n);

        scl::kernel::permutation::bonferroni_correction(pvals, adj);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_holm_correction(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* adjusted
) {
    if (!p_values || !adjusted) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Array<scl::Real> adj(reinterpret_cast<scl::Real*>(adjusted), n);

        scl::kernel::permutation::holm_correction(pvals, adj);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_count_significant(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t alpha,
    scl_size_t* n_significant
) {
    if (!p_values || !n_significant) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Size count = scl::kernel::permutation::count_significant(
            pvals, static_cast<scl::Real>(alpha)
        );

        *n_significant = count;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_get_significant_indices(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t alpha,
    scl_index_t* indices,
    scl_size_t max_results,
    scl_size_t* n_results
) {
    if (!p_values || !indices || !n_results) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> pvals(reinterpret_cast<const scl::Real*>(p_values), n);
        scl::Array<scl::Index> idx_arr(indices, max_results);
        scl::Size n_res = 0;

        scl::kernel::permutation::get_significant_indices(
            pvals, static_cast<scl::Real>(alpha), idx_arr, n_res
        );

        *n_results = n_res;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_perm_batch_test(
    scl_sparse_t matrix,
    const scl_index_t* group_labels,
    scl_size_t n_permutations,
    scl_real_t* p_values,
    uint64_t seed
) {
    if (!matrix || !group_labels || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_cols = sparse->cols();
        scl::Index n_rows = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> labels(group_labels, static_cast<scl::Size>(n_cols));
            scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values),
                                       static_cast<scl::Size>(n_rows));
            scl::kernel::permutation::batch_permutation_test(
                m, labels, n_permutations, pvals, seed
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
