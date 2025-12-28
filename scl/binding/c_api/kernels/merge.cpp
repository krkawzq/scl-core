// =============================================================================
// FILE: scl/binding/c_api/kernels/merge.cpp
// BRIEF: C API implementation for matrix merging
// =============================================================================

#include "merge.h"
#include "scl/kernel/merge.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/type.hpp"
#include "scl/core/registry.hpp"

namespace {

inline scl_error_t from_error_code(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

inline scl::CSR* to_sparse(scl_sparse_matrix_t handle) {
    return static_cast<scl::CSR*>(handle);
}

} // anonymous namespace

extern "C" {

scl_error_t scl_merge_vstack(
    scl_sparse_matrix_t matrix1,
    scl_sparse_matrix_t matrix2,
    scl_sparse_matrix_t* result
) {
    try {
        scl::CSR* m1 = to_sparse(matrix1);
        scl::CSR* m2 = to_sparse(matrix2);
        if (!m1 || !m2 || !result) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::CSR merged = scl::kernel::merge::vstack(*m1, *m2);
        
        // Allocate new CSR for result
        scl::CSR* result_ptr = new scl::CSR(std::move(merged));
        *result = static_cast<scl_sparse_matrix_t>(result_ptr);
        
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_merge_hstack(
    scl_sparse_matrix_t matrix1,
    scl_sparse_matrix_t matrix2,
    scl_sparse_matrix_t* result
) {
    try {
        scl::CSR* m1 = to_sparse(matrix1);
        scl::CSR* m2 = to_sparse(matrix2);
        if (!m1 || !m2 || !result) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::CSR merged = scl::kernel::merge::hstack(*m1, *m2);
        
        // Allocate new CSR for result
        scl::CSR* result_ptr = new scl::CSR(std::move(merged));
        *result = static_cast<scl_sparse_matrix_t>(result_ptr);
        
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"
