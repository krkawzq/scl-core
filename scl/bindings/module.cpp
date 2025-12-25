#include "scl/error.hpp"
#include "scl/version.hpp"
#include <cstring>
#include <exception>
#include <stdexcept>

// =============================================================================
/// @file module.cpp
/// @brief Module-level C-ABI Functions
///
/// This file provides module initialization, version information, and
/// other module-level operations exposed through the C-ABI.
///
/// =============================================================================

extern "C" {

/// @brief Get library version string
///
/// @return Version string (e.g., "0.1.0")
const char* scl_get_version() {
    return SCL_VERSION;
}

/// @brief Get library author string
///
/// @return Author string
const char* scl_get_author() {
    return SCL_AUTHOR;
}

/// @brief Initialize the SCL module
///
/// Performs any necessary initialization (thread pools, etc.)
///
/// @return nullptr on success, error instance on failure
scl_error_t scl_init() {
    try {
        // Module initialization logic here
        // For now, just return success
        return nullptr;
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

/// @brief Shutdown the SCL module
///
/// Performs cleanup operations
///
/// @return nullptr on success, error instance on failure
scl_error_t scl_shutdown() {
    try {
        // Module shutdown logic here
        return nullptr;
    } catch (const std::exception& e) {
        return scl::error::exception_to_error(e);
    } catch (...) {
        return scl::error::unknown_exception_to_error();
    }
}

} // extern "C"

