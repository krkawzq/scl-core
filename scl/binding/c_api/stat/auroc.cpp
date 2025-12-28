// =============================================================================
// FILE: scl/binding/c_api/stat/auroc.cpp
// BRIEF: C API implementation for AUROC statistics
// =============================================================================

#include "scl/binding/c_api/stat/auroc.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/stat/auroc.hpp"
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

scl_error_t scl_stat_auroc(
    const scl_real_t* scores,
    const uint8_t* labels,
    scl_size_t n,
    scl_real_t* auroc
) {
    if (!scores || !labels || !auroc) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(scores),
            n
        );
        scl::Array<const scl::Byte> labels_arr(
            reinterpret_cast<const scl::Byte*>(labels),
            n
        );
        *auroc = static_cast<scl_real_t>(
            scl::kernel::stat::auroc::compute_auroc(scores_arr, labels_arr)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"
