/// @file scl/api/scl_core.cpp
/// @brief C API (FFI) for core SCL functionality
///
/// This file exports C-compatible functions for use in Python bindings
/// and other language FFI interfaces.

#include "scl/core/error.hpp"
#include "scl/version.hpp"

#include <cstdint>

extern "C" {
  
/// @brief Get last error code from thread-local storage
/// @return Error code as int32_t
std::int32_t scl_get_last_error_code() {
    return static_cast<std::int32_t>(scl::get_thread_error().code());
}

/// @brief Get last error message from thread-local storage
/// @return Error message string (null-terminated)
const char* scl_get_last_error_message() {
    return scl::get_thread_error().message();
}

/// @brief Clear thread-local error state
void scl_clear_error() {
    scl::clear_thread_error();
}

}  // extern "C"

