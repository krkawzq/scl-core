// =============================================================================
// FILE: scl/binding/c_api/algebra.cpp
// BRIEF: C API implementation for sparse linear algebra operations
// =============================================================================

#include "algebra.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/algebra.hpp"

#include <cstring>

// =============================================================================
// Error Code Conversion Helper
// =============================================================================

namespace {
    inline scl_error_t convert_error_code(scl::ErrorCode code) {
        return static_cast<scl_error_t>(code);
    }

    inline scl_error_t catch_exception() {
        try {
            throw;
        } catch (const scl::DimensionError&) {
            return SCL_ERROR_DIMENSION_MISMATCH;
        } catch (const scl::ValueError&) {
            return SCL_ERROR_INVALID_ARGUMENT;
        } catch (const scl::TypeError&) {
            return SCL_ERROR_TYPE_ERROR;
        } catch (const scl::DomainError&) {
            return SCL_ERROR_DOMAIN_ERROR;
        } catch (const scl::InternalError&) {
            return SCL_ERROR_INTERNAL;
        } catch (const scl::IOError&) {
            return SCL_ERROR_IO_ERROR;
        } catch (const scl::NotImplementedError&) {
            return SCL_ERROR_NOT_IMPLEMENTED;
        } catch (const scl::Exception& e) {
            return convert_error_code(e.code());
        } catch (...) {
            return SCL_ERROR_UNKNOWN;
        }
    }

    template <typename T>
    inline scl::Sparse<T, true>* get_matrix(scl_sparse_matrix_t handle) {
        return reinterpret_cast<scl::Sparse<T, true>*>(handle);
    }
}

// =============================================================================
// SpMV: y = alpha * A * x + beta * y
// =============================================================================

extern "C" scl_error_t scl_spmv_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !x || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const float> x_arr(x, x_size);
        scl::Array<float> y_arr(y, y_size);

        scl::kernel::algebra::spmv(*matrix, x_arr, y_arr, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_spmv_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !x || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const double> x_arr(x, x_size);
        scl::Array<double> y_arr(y, y_size);

        scl::kernel::algebra::spmv(*matrix, x_arr, y_arr, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// SpMV Transpose: y = alpha * A^T * x + beta * y
// =============================================================================

extern "C" scl_error_t scl_spmv_transpose_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !x || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const float> x_arr(x, x_size);
        scl::Array<float> y_arr(y, y_size);

        scl::kernel::algebra::spmv_transpose(*matrix, x_arr, y_arr, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_spmv_transpose_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !x || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const double> x_arr(x, x_size);
        scl::Array<double> y_arr(y, y_size);

        scl::kernel::algebra::spmv_transpose(*matrix, x_arr, y_arr, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// SpMM: Y = alpha * A * X + beta * Y
// =============================================================================

extern "C" scl_error_t scl_spmm_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* X,
    scl_size_t n_cols_X,
    scl_real_t* Y,
    scl_real_t alpha,
    scl_real_t beta
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !X || !Y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::algebra::spmm(*matrix, X, static_cast<scl::Index>(n_cols_X), Y, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_spmm_f64_csr(
    scl_sparse_matrix_t A,
    const double* X,
    scl_size_t n_cols_X,
    double* Y,
    double alpha,
    double beta
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !X || !Y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::algebra::spmm(*matrix, X, static_cast<scl::Index>(n_cols_X), Y, alpha, beta);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// Fused SpMV: y = alpha * A * x + beta * A * z + gamma * y
// =============================================================================

extern "C" scl_error_t scl_spmv_fused_linear_f32_csr(
    scl_sparse_matrix_t A,
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
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !x || !z || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const float> x_arr(x, x_size);
        scl::Array<const float> z_arr(z, z_size);
        scl::Array<float> y_arr(y, y_size);

        scl::kernel::algebra::spmv_fused_linear(*matrix, x_arr, z_arr, y_arr, alpha, beta, gamma);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_spmv_fused_linear_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    const double* z,
    scl_size_t z_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta,
    double gamma
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !x || !z || !y) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const double> x_arr(x, x_size);
        scl::Array<const double> z_arr(z, z_size);
        scl::Array<double> y_arr(y, y_size);

        scl::kernel::algebra::spmv_fused_linear(*matrix, x_arr, z_arr, y_arr, alpha, beta, gamma);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// Row Norms
// =============================================================================

extern "C" scl_error_t scl_row_norms_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* norms,
    scl_size_t norms_size
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !norms) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<float> norms_arr(norms, norms_size);
        scl::kernel::algebra::row_norms(*matrix, norms_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_row_norms_f64_csr(
    scl_sparse_matrix_t A,
    double* norms,
    scl_size_t norms_size
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !norms) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<double> norms_arr(norms, norms_size);
        scl::kernel::algebra::row_norms(*matrix, norms_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// Row Sums
// =============================================================================

extern "C" scl_error_t scl_row_sums_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* sums,
    scl_size_t sums_size
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !sums) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<float> sums_arr(sums, sums_size);
        scl::kernel::algebra::row_sums(*matrix, sums_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_row_sums_f64_csr(
    scl_sparse_matrix_t A,
    double* sums,
    scl_size_t sums_size
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !sums) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<double> sums_arr(sums, sums_size);
        scl::kernel::algebra::row_sums(*matrix, sums_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// Extract Diagonal
// =============================================================================

extern "C" scl_error_t scl_extract_diagonal_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* diag,
    scl_size_t diag_size
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !diag) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<float> diag_arr(diag, diag_size);
        scl::kernel::algebra::extract_diagonal(*matrix, diag_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_extract_diagonal_f64_csr(
    scl_sparse_matrix_t A,
    double* diag,
    scl_size_t diag_size
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !diag) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<double> diag_arr(diag, diag_size);
        scl::kernel::algebra::extract_diagonal(*matrix, diag_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

// =============================================================================
// Scale Rows
// =============================================================================

extern "C" scl_error_t scl_scale_rows_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* scale_factors,
    scl_size_t scale_factors_size
) {
    try {
        auto* matrix = get_matrix<float>(A);
        if (!matrix || !scale_factors) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const float> scale_arr(scale_factors, scale_factors_size);
        scl::kernel::algebra::scale_rows(*matrix, scale_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_scale_rows_f64_csr(
    scl_sparse_matrix_t A,
    const double* scale_factors,
    scl_size_t scale_factors_size
) {
    try {
        auto* matrix = get_matrix<double>(A);
        if (!matrix || !scale_factors) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const double> scale_arr(scale_factors, scale_factors_size);
        scl::kernel::algebra::scale_rows(*matrix, scale_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}
