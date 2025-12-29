// =============================================================================
// FILE: scl/binding/c_api/gram.cpp
// BRIEF: C API implementation for Gram matrix computation
// =============================================================================

#include "scl/binding/c_api/gram.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/gram.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Gram Matrix Computation
// =============================================================================

SCL_EXPORT scl_error_t scl_gram_compute(
    scl_sparse_t matrix,
    scl_real_t* output,
    const scl_size_t n_rows) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size expected_size = primary_dim_sz * primary_dim_sz;
        
        SCL_C_API_CHECK(static_cast<Size>(n_rows) == expected_size,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Output buffer size mismatch");

        Array<Real> output_arr(reinterpret_cast<Real*>(output), expected_size);

        matrix->visit([&](auto& m) {
            scl::kernel::gram::gram(m, output_arr);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
