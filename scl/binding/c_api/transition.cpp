// =============================================================================
// FILE: scl/binding/c_api/transition.cpp
// BRIEF: C API implementation for cell state transition analysis
// =============================================================================

#include "scl/binding/c_api/transition.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/transition.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t get_sparse_matrix(
    scl_sparse_t handle,
    scl::binding::SparseWrapper*& wrapper
) {
    if (!handle) {
        return SCL_ERROR_NULL_POINTER;
    }
    wrapper = static_cast<scl::binding::SparseWrapper*>(handle);
    if (!wrapper->valid()) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    return SCL_OK;
}

scl_error_t scl_transition_matrix_from_velocity(
    scl_sparse_t velocity_graph,
    scl_index_t n,
    scl_real_t* row_stochastic_out
) {
    if (!velocity_graph || !row_stochastic_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(velocity_graph, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& graph) {
            scl::kernel::transition::transition_matrix_from_velocity(
                graph,
                static_cast<scl::Index>(n),
                reinterpret_cast<scl::Real*>(row_stochastic_out)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_transition_row_normalize_to_stochastic(
    scl_sparse_t input,
    scl_index_t n,
    scl_real_t* output_values,
    scl_size_t output_size
) {
    if (!input || !output_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(input, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& in) {
            scl::kernel::transition::row_normalize_to_stochastic(
                in,
                static_cast<scl::Index>(n),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(output_values),
                    static_cast<scl::Size>(output_size)
                )
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_transition_symmetrize(
    scl_sparse_t transition_mat,
    scl_index_t n,
    scl_real_t* symmetric_out
) {
    if (!transition_mat || !symmetric_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(transition_mat, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& mat) {
            scl::kernel::transition::symmetrize_transition(
                mat,
                static_cast<scl::Index>(n),
                reinterpret_cast<scl::Real*>(symmetric_out)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_transition_stationary_distribution(
    scl_sparse_t transition_mat,
    scl_index_t n,
    scl_real_t* stationary,
    scl_index_t max_iter,
    scl_real_t tolerance
) {
    if (!transition_mat || !stationary) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(transition_mat, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& mat) {
            scl::kernel::transition::stationary_distribution(
                mat,
                static_cast<scl::Index>(n),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(stationary),
                    static_cast<scl::Size>(n)
                ),
                static_cast<scl::Index>(max_iter),
                static_cast<scl::Real>(tolerance)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
