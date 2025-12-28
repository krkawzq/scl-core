// =============================================================================
// FILE: scl/binding/c_api/merge.cpp
// BRIEF: C API implementation for matrix merging
// =============================================================================

#include "scl/binding/c_api/merge.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/kernel/merge.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/registry.hpp"

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

scl_error_t scl_merge_vstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result
) {
    if (!matrix1 || !matrix2 || !result) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(matrix1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(matrix2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        // Both matrices must have the same format
        if (wrapper1->is_csr != wrapper2->is_csr) {
            return SCL_ERROR_TYPE_MISMATCH;
        }

        bool is_csr = wrapper1->is_csr;
        if (is_csr) {
            auto& m1 = std::get<scl::CSR>(wrapper1->matrix);
            auto& m2 = std::get<scl::CSR>(wrapper2->matrix);
            auto merged = scl::kernel::merge::vstack(m1, m2);
            auto* new_wrapper = scl::get_registry().new_object<scl::binding::SparseWrapper>(
                scl::binding::SparseWrapper(std::move(merged))
            );
            *result = reinterpret_cast<scl_sparse_t>(new_wrapper);
        } else {
            auto& m1 = std::get<scl::CSC>(wrapper1->matrix);
            auto& m2 = std::get<scl::CSC>(wrapper2->matrix);
            auto merged = scl::kernel::merge::vstack(m1, m2);
            auto* new_wrapper = scl::get_registry().new_object<scl::binding::SparseWrapper>(
                scl::binding::SparseWrapper(std::move(merged))
            );
            *result = reinterpret_cast<scl_sparse_t>(new_wrapper);
        }

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_merge_hstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result
) {
    if (!matrix1 || !matrix2 || !result) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(matrix1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(matrix2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        // Both matrices must have the same format
        if (wrapper1->is_csr != wrapper2->is_csr) {
            return SCL_ERROR_TYPE_MISMATCH;
        }

        bool is_csr = wrapper1->is_csr;
        if (is_csr) {
            auto& m1 = std::get<scl::CSR>(wrapper1->matrix);
            auto& m2 = std::get<scl::CSR>(wrapper2->matrix);
            auto merged = scl::kernel::merge::hstack(m1, m2);
            auto* new_wrapper = scl::get_registry().new_object<scl::binding::SparseWrapper>(
                scl::binding::SparseWrapper(std::move(merged))
            );
            *result = reinterpret_cast<scl_sparse_t>(new_wrapper);
        } else {
            auto& m1 = std::get<scl::CSC>(wrapper1->matrix);
            auto& m2 = std::get<scl::CSC>(wrapper2->matrix);
            auto merged = scl::kernel::merge::hstack(m1, m2);
            auto* new_wrapper = scl::get_registry().new_object<scl::binding::SparseWrapper>(
                scl::binding::SparseWrapper(std::move(merged))
            );
            *result = reinterpret_cast<scl_sparse_t>(new_wrapper);
        }

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
