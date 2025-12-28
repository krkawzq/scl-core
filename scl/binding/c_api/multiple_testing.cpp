// =============================================================================
// FILE: scl/binding/c_api/multiple_testing.cpp
// BRIEF: C API implementation for Multiple Testing Correction
// =============================================================================

#include "scl/binding/c_api/multiple_testing.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/multiple_testing.hpp"
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

scl_error_t scl_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_real_t* adjusted_p_values,
    scl_size_t n,
    scl_real_t fdr_level
) {
    if (!p_values || !adjusted_p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> adj_arr(adjusted_p_values, n);
        scl::kernel::multiple_testing::benjamini_hochberg(p_arr, adj_arr, static_cast<scl::Real>(fdr_level));
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_bonferroni(
    const scl_real_t* p_values,
    scl_real_t* adjusted_p_values,
    scl_size_t n
) {
    if (!p_values || !adjusted_p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> adj_arr(adjusted_p_values, n);
        scl::kernel::multiple_testing::bonferroni(p_arr, adj_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_storey_qvalue(
    const scl_real_t* p_values,
    scl_real_t* q_values,
    scl_size_t n,
    scl_real_t lambda
) {
    if (!p_values || !q_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> q_arr(q_values, n);
        scl::kernel::multiple_testing::storey_qvalue(p_arr, q_arr, static_cast<scl::Real>(lambda));
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_local_fdr(
    const scl_real_t* p_values,
    scl_real_t* lfdr,
    scl_size_t n
) {
    if (!p_values || !lfdr) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> lfdr_arr(lfdr, n);
        scl::kernel::multiple_testing::local_fdr(p_arr, lfdr_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_benjamini_yekutieli(
    const scl_real_t* p_values,
    scl_real_t* adjusted_p_values,
    scl_size_t n
) {
    if (!p_values || !adjusted_p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> adj_arr(adjusted_p_values, n);
        scl::kernel::multiple_testing::benjamini_yekutieli(p_arr, adj_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_holm_bonferroni(
    const scl_real_t* p_values,
    scl_real_t* adjusted_p_values,
    scl_size_t n
) {
    if (!p_values || !adjusted_p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> adj_arr(adjusted_p_values, n);
        scl::kernel::multiple_testing::holm_bonferroni(p_arr, adj_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hochberg(
    const scl_real_t* p_values,
    scl_real_t* adjusted_p_values,
    scl_size_t n
) {
    if (!p_values || !adjusted_p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> adj_arr(adjusted_p_values, n);
        scl::kernel::multiple_testing::hochberg(p_arr, adj_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_size_t scl_count_significant(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t threshold
) {
    if (!p_values) {
        return 0;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        return scl::kernel::multiple_testing::count_significant(p_arr, static_cast<scl::Real>(threshold));
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_significant_indices(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t threshold,
    scl_index_t* out_indices,
    scl_size_t* out_count
) {
    if (!p_values || !out_indices || !out_count) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Size count = 0;
        scl::kernel::multiple_testing::significant_indices(
            p_arr,
            static_cast<scl::Real>(threshold),
            out_indices,
            count
        );
        *out_count = count;
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_neglog10_pvalues(
    const scl_real_t* p_values,
    scl_real_t* neglog_p,
    scl_size_t n
) {
    if (!p_values || !neglog_p) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        scl::Array<scl::Real> neglog_arr(neglog_p, n);
        scl::kernel::multiple_testing::neglog10_pvalues(p_arr, neglog_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_fisher_combine(
    const scl_real_t* p_values,
    scl_size_t n
) {
    if (!p_values) {
        return static_cast<scl_real_t>(1.0);
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        return static_cast<scl_real_t>(scl::kernel::multiple_testing::fisher_combine(p_arr));
    } catch (...) {
        return static_cast<scl_real_t>(1.0);
    }
}

scl_real_t scl_stouffer_combine(
    const scl_real_t* p_values,
    const scl_real_t* weights,
    scl_size_t n
) {
    if (!p_values) {
        return static_cast<scl_real_t>(0.0);
    }

    try {
        scl::Array<const scl::Real> p_arr(p_values, n);
        if (weights) {
            scl::Array<const scl::Real> w_arr(weights, n);
            return static_cast<scl_real_t>(scl::kernel::multiple_testing::stouffer_combine(p_arr, w_arr));
        } else {
            scl::Array<const scl::Real> empty_weights(nullptr, 0);
            return static_cast<scl_real_t>(scl::kernel::multiple_testing::stouffer_combine(p_arr, empty_weights));
        }
    } catch (...) {
        return static_cast<scl_real_t>(0.0);
    }
}

} // extern "C"

