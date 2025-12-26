// =============================================================================
/// @file error.hpp
/// @brief C ABI Error Handling Infrastructure
///
/// Provides error handling utilities for C-ABI wrapper functions.
///
/// This module implements the error barrier between C++ exceptions and C-ABI.
/// All C-ABI functions must use SCL_C_API_WRAPPER macro to catch exceptions
/// and convert them to error codes with messages stored in thread-local buffer.
///
/// Error Protocol:
///
/// 1. Success: Return 0
/// 2. Failure: Return -1 and store error message in thread-local buffer
/// 3. Python side retrieves error via scl_get_last_error()
///
/// Thread Safety:
///
/// - Error buffer is thread-local, safe for concurrent use
/// - Each thread maintains its own error state
// =============================================================================

#pragma once

#include <cstring>
#include <exception>

namespace scl::binding::detail {

/// @brief Thread-local error message buffer (512 bytes)
inline thread_local char g_error_buffer[512] = {0};

/// @brief Store exception message for later retrieval
///
/// @param e Standard exception to extract message from
inline void store_error(const std::exception& e) noexcept {
    std::strncpy(g_error_buffer, e.what(), sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

/// @brief Store generic error message
///
/// @param msg C-string error message
inline void store_error(const char* msg) noexcept {
    std::strncpy(g_error_buffer, msg, sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

/// @brief Clear error state (reset buffer to empty string)
inline void clear_error() noexcept {
    g_error_buffer[0] = '\0';
}

} // namespace scl::binding::detail

extern "C" {

/// @brief Retrieve last error message from thread-local buffer
///
/// @return Pointer to null-terminated error string (empty if no error)
inline const char* scl_get_last_error() {
    return scl::binding::detail::g_error_buffer;
}

/// @brief Clear the error state
inline void scl_clear_error() {
    scl::binding::detail::clear_error();
}

} // extern "C"

/// @brief Helper macro for C-ABI exception barrier
///
/// Usage:
///
///     extern "C" int some_function(...) {
///         SCL_C_API_WRAPPER(
///             // C++ code that may throw
///             some_cpp_function();
///         );
///     }
///
/// Returns:
///
/// - 0 on success
/// - -1 on failure (error message stored in buffer)
///
/// Note: Uses variadic macro (__VA_ARGS__) to handle code containing commas
/// such as template arguments like MappedArray<scl::Real> or function calls
/// with multiple parameters.
#define SCL_C_API_WRAPPER(...) \
    try { \
        scl::binding::detail::clear_error(); \
        __VA_ARGS__; \
        return 0; \
    } catch (const std::exception& e) { \
        scl::binding::detail::store_error(e); \
        return -1; \
    } catch (...) { \
        scl::binding::detail::store_error("Unknown error"); \
        return -1; \
    }
