// =============================================================================
/// @file version_info.cpp
/// @brief Version and Type Information Exports
///
/// Provides version strings and type metadata for C-ABI consumers.
// =============================================================================

#include "scl/kernel/core.hpp"
#include "scl/version.hpp"

extern "C" {

// =============================================================================
// Version and Type Information
// =============================================================================

const char* scl_version() {
    return SCL_VERSION;
}

int scl_precision_type() {
    return scl::DTYPE_CODE;
}

const char* scl_precision_name() {
    return scl::DTYPE_NAME;
}

int scl_index_type() {
    return scl::INDEX_DTYPE_CODE;
}

const char* scl_index_name() {
    return scl::INDEX_DTYPE_NAME;
}

} // extern "C"
