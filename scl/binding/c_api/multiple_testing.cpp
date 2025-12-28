// =============================================================================
// FILE: scl/binding/c_api/multiple_testing/multiple_testing.cpp
// BRIEF: C API implementation for multiple testing correction
// =============================================================================

#include "scl/binding/c_api/multiple_testing.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/multiple_testing.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>
#include <vector>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::multiple_testing;

extern "C" {

scl_error_t scl_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values,
    scl_real_t fdr_level)
{
    if (!p_values || !adjusted_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        benjamini_hochberg(p_arr, adj_arr, fdr_level);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values)
{
    if (!p_values || !adjusted_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        bonferroni(p_arr, adj_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_benjamini_yekutieli(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values)
{
    if (!p_values || !adjusted_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        benjamini_yekutieli(p_arr, adj_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_holm_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values)
{
    if (!p_values || !adjusted_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        holm_bonferroni(p_arr, adj_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hochberg(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values)
{
    if (!p_values || !adjusted_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        hochberg(p_arr, adj_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_storey_qvalue(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* q_values,
    scl_real_t lambda)
{
    if (!p_values || !q_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> q_arr(reinterpret_cast<Real*>(q_values), n_tests);
        storey_qvalue(p_arr, q_arr, lambda);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_local_fdr(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* lfdr)
{
    if (!p_values || !lfdr) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> lfdr_arr(reinterpret_cast<Real*>(lfdr), n_tests);
        local_fdr(p_arr, lfdr_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_count_significant(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t threshold,
    scl_size_t* count)
{
    if (!p_values || !count) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        *count = count_significant(p_arr, threshold);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_significant_indices(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t threshold,
    scl_index_t* out_indices,
    scl_size_t* out_count,
    scl_size_t max_count)
{
    if (!p_values || !out_indices || !out_count) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), max_count);
        significant_indices(p_arr, threshold, indices_arr.ptr, *out_count);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_neglog10_pvalues(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* neglog_p)
{
    if (!p_values || !neglog_p) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> neglog_arr(reinterpret_cast<Real*>(neglog_p), n_tests);
        neglog10_pvalues(p_arr, neglog_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_fisher_combine(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* chi2_stat)
{
    if (!p_values || !chi2_stat) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        *chi2_stat = fisher_combine(p_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_stouffer_combine(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    const scl_real_t* weights,
    scl_real_t* z_score)
{
    if (!p_values || !z_score) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<const Real> weights_arr;
        if (weights) {
            weights_arr = Array<const Real>(reinterpret_cast<const Real*>(weights), n_tests);
        }
        *z_score = stouffer_combine(p_arr, weights_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

