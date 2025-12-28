// =============================================================================
// FILE: scl/binding/c_api/algebra.cpp
// BRIEF: C API implementation for sparse linear algebra operations
// =============================================================================

#include "scl/binding/c_api/algebra.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/algebra.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

// =============================================================================
// Helper: Convert sparse handle to C++ matrix reference
// =============================================================================

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

// =============================================================================
// SpMV: y = alpha * A * x + beta * y
// =============================================================================

scl_error_t scl_algebra_spmv(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
) {
    if (!A || !x || !y) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<const scl::Real> x_arr(
                reinterpret_cast<const scl::Real*>(x),
                x_size
            );
            scl::Array<scl::Real> y_arr(
                reinterpret_cast<scl::Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv(
                matrix, x_arr, y_arr,
                static_cast<scl::Real>(alpha),
                static_cast<scl::Real>(beta)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_algebra_spmv_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
) {
    return scl_algebra_spmv(A, x, x_size, y, y_size, 1.0f, 0.0f);
}

scl_error_t scl_algebra_spmv_scaled(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha
) {
    return scl_algebra_spmv(A, x, x_size, y, y_size, alpha, 0.0f);
}

scl_error_t scl_algebra_spmv_add(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
) {
    return scl_algebra_spmv(A, x, x_size, y, y_size, 1.0f, 1.0f);
}

// =============================================================================
// Transposed SpMV
// =============================================================================

scl_error_t scl_algebra_spmv_transpose(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
) {
    if (!A || !x || !y) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<const scl::Real> x_arr(
                reinterpret_cast<const scl::Real*>(x),
                x_size
            );
            scl::Array<scl::Real> y_arr(
                reinterpret_cast<scl::Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv_transpose(
                matrix, x_arr, y_arr,
                static_cast<scl::Real>(alpha),
                static_cast<scl::Real>(beta)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_algebra_spmv_transpose_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
) {
    return scl_algebra_spmv_transpose(A, x, x_size, y, y_size, 1.0f, 0.0f);
}

// =============================================================================
// SpMM: Y = alpha * A * X + beta * Y
// =============================================================================

scl_error_t scl_algebra_spmm(
    scl_sparse_t A,
    const scl_real_t* X,
    scl_index_t n_cols,
    scl_real_t* Y,
    scl_real_t alpha,
    scl_real_t beta
) {
    if (!A || !X || !Y) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::kernel::algebra::spmm(
                matrix,
                reinterpret_cast<const scl::Real*>(X),
                static_cast<scl::Index>(n_cols),
                reinterpret_cast<scl::Real*>(Y),
                static_cast<scl::Real>(alpha),
                static_cast<scl::Real>(beta)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Fused SpMV
// =============================================================================

scl_error_t scl_algebra_spmv_fused_linear(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    const scl_real_t* z,
    scl_size_t z_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta,
    scl_real_t gamma
) {
    if (!A || !x || !z || !y) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<const scl::Real> x_arr(
                reinterpret_cast<const scl::Real*>(x),
                x_size
            );
            scl::Array<const scl::Real> z_arr(
                reinterpret_cast<const scl::Real*>(z),
                z_size
            );
            scl::Array<scl::Real> y_arr(
                reinterpret_cast<scl::Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv_fused_linear(
                matrix, x_arr, z_arr, y_arr,
                static_cast<scl::Real>(alpha),
                static_cast<scl::Real>(beta),
                static_cast<scl::Real>(gamma)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Row Operations
// =============================================================================

scl_error_t scl_algebra_row_norms(
    scl_sparse_t A,
    scl_real_t* norms,
    scl_size_t norms_size
) {
    if (!A || !norms) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<scl::Real> norms_arr(
                reinterpret_cast<scl::Real*>(norms),
                norms_size
            );
            scl::kernel::algebra::row_norms(matrix, norms_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_algebra_row_sums(
    scl_sparse_t A,
    scl_real_t* sums,
    scl_size_t sums_size
) {
    if (!A || !sums) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<scl::Real> sums_arr(
                reinterpret_cast<scl::Real*>(sums),
                sums_size
            );
            scl::kernel::algebra::row_sums(matrix, sums_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_algebra_extract_diagonal(
    scl_sparse_t A,
    scl_real_t* diag,
    scl_size_t diag_size
) {
    if (!A || !diag) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<scl::Real> diag_arr(
                reinterpret_cast<scl::Real*>(diag),
                diag_size
            );
            scl::kernel::algebra::extract_diagonal(matrix, diag_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_algebra_scale_rows(
    scl_sparse_t A,
    const scl_real_t* scale_factors,
    scl_size_t scale_factors_size
) {
    if (!A || !scale_factors) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(A, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& matrix) {
            scl::Array<const scl::Real> scale_arr(
                reinterpret_cast<const scl::Real*>(scale_factors),
                scale_factors_size
            );
            scl::kernel::algebra::scale_rows(matrix, scale_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"

