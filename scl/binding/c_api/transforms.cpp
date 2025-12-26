// =============================================================================
/// @file transforms.cpp
/// @brief Log Transforms and Mathematical Transformations (log1p.hpp)
///
/// Provides logarithmic and exponential transformation functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Log Transforms
// =============================================================================

int scl_log1p_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log1p_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_log1p_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::log1p_inplace(matrix);
    )
}

int scl_log2p1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log2p1_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_expm1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::expm1_inplace(scl::Array<scl::Real>(data, size));
    )
}

} // extern "C"
