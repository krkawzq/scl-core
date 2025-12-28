// =============================================================================
// FILE: scl/binding/c_api/transition.cpp
// BRIEF: C API implementation for cell state transition analysis
// =============================================================================

#include "scl/binding/c_api/transition.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/transition.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

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

static scl::kernel::transition::TransitionType convert_transition_type(scl_transition_type_t type) {
    switch (type) {
        case SCL_TRANSITION_FORWARD: return scl::kernel::transition::TransitionType::Forward;
        case SCL_TRANSITION_BACKWARD: return scl::kernel::transition::TransitionType::Backward;
        case SCL_TRANSITION_SYMMETRIC: return scl::kernel::transition::TransitionType::Symmetric;
        default: return scl::kernel::transition::TransitionType::Forward;
    }
}

scl_error_t scl_transition_normalize(
    scl_sparse_matrix_t transition_mat,
    scl_index_t n,
    scl_real_t* output
) {
    if (!transition_mat || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(transition_mat);
        scl::kernel::transition::normalize_transition(
            *sparse,
            static_cast<scl::Index>(n),
            reinterpret_cast<scl::Real*>(output)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_transition_stationary_distribution(
    scl_sparse_matrix_t transition_mat,
    scl_index_t n,
    scl_real_t* stationary,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!transition_mat || !stationary) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(transition_mat);
        scl::Array<scl::Real> stat_arr(
            reinterpret_cast<scl::Real*>(stationary),
            static_cast<scl::Size>(n)
        );
        scl::kernel::transition::stationary_distribution(
            *sparse,
            static_cast<scl::Index>(n),
            stat_arr,
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_transition_absorption_probability(
    scl_sparse_matrix_t transition_mat,
    const scl_index_t* terminal_states,
    scl_index_t n,
    scl_index_t n_terminal,
    scl_real_t* absorption_probs,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!transition_mat || !terminal_states || !absorption_probs) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(transition_mat);
        scl::Array<const scl::Index> term_arr(
            reinterpret_cast<const scl::Index*>(terminal_states),
            static_cast<scl::Size>(n_terminal)
        );
        scl::Array<scl::Real> prob_arr(
            reinterpret_cast<scl::Real*>(absorption_probs),
            static_cast<scl::Size>(n) * static_cast<scl::Size>(n_terminal)
        );
        scl::kernel::transition::absorption_probability(
            *sparse,
            term_arr,
            static_cast<scl::Index>(n),
            prob_arr,
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::transition::normalize_transition<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Real*);
template void scl::kernel::transition::stationary_distribution<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Array<scl::Real>, scl::Index, scl::Real);
template void scl::kernel::transition::absorption_probability<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Index>, scl::Index, scl::Array<scl::Real>, scl::Index, scl::Real);

} // extern "C"
