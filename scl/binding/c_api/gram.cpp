// =============================================================================
// FILE: scl/binding/c_api/gram/gram.cpp
// BRIEF: C API implementation for Gram matrix computation
// =============================================================================

#include "scl/binding/c_api/gram.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/gram.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::gram;

extern "C" {

// =============================================================================
// Gram Matrix Computation
// =============================================================================

scl_error_t scl_gram_compute(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t n_rows)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index primary_dim = wrapper->rows();
        scl_size_t expected_size = static_cast<scl_size_t>(primary_dim) * static_cast<scl_size_t>(primary_dim);
        
        if (n_rows != expected_size) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Output buffer size mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            n_rows
        );

        wrapper->visit([&](auto& m) {
            gram(m, output_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

