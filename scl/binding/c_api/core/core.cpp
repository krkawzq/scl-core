// =============================================================================
// FILE: scl/binding/c_api/core/core.cpp
// BRIEF: Core C API implementation with thread-safe error handling
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/error.hpp"

#include <array>
#include <cstring>
#include <exception>
#include <new>
#include <string_view>
#include <sstream>

namespace scl::binding {

// =============================================================================
// Thread-Local Error State (Zero-Overhead When No Errors)
// =============================================================================

namespace {

// Thread-local error state with fixed-size buffer for zero-allocation error handling
// Buffer size chosen to handle typical error messages without truncation
constexpr std::size_t ERROR_MESSAGE_BUFFER_SIZE = 512;

// PERFORMANCE: Thread-local storage for error state
// Zero overhead for error-free execution (just TLS access)
// No allocations during error reporting (fixed buffer)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local scl_error_t g_last_error_code = SCL_OK;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::array<char, ERROR_MESSAGE_BUFFER_SIZE> g_last_error_message = {};

} // anonymous namespace

// =============================================================================
// Internal Error Management Functions
// =============================================================================

// Set last error with code and message
// Thread-safe via thread-local storage
// noexcept guarantee: error reporting must never throw
// NOTE: Not inline - must export symbols for Python bindings
void set_last_error(scl_error_t code, const char* message) noexcept {
    g_last_error_code = code;
    
    if (SCL_LIKELY(message != nullptr)) [[likely]] {
        // Use std::strncpy for safety with explicit null termination
        std::strncpy(g_last_error_message.data(), message, 
                    ERROR_MESSAGE_BUFFER_SIZE - 1);
        g_last_error_message[ERROR_MESSAGE_BUFFER_SIZE - 1] = '\0';
    } else [[unlikely]] {
        g_last_error_message[0] = '\0';
    }
}

// Set last error with std::string_view (C++20)
// Convenience overload for modern C++ code
// NOTE: Not inline - must export symbols for Python bindings
void set_last_error(scl_error_t code, std::string_view message) noexcept {
    g_last_error_code = code;
    
    const auto copy_len = std::min(message.size(), 
                                   ERROR_MESSAGE_BUFFER_SIZE - 1);
    std::memcpy(g_last_error_message.data(), message.data(), copy_len);
    g_last_error_message[copy_len] = '\0';
}

// Clear error state
// NOTE: Not inline - must export symbols for Python bindings
void clear_last_error() noexcept {
    g_last_error_code = SCL_OK;
    g_last_error_message[0] = '\0';
}

// Get error message
// NOTE: Not inline - must export symbols for Python bindings
auto get_last_error_message() noexcept -> const char* {
    if (SCL_LIKELY(g_last_error_message[0] != '\0')) [[likely]] {
        return g_last_error_message.data();
    }
    return "No error";
}

// Get error code
// NOTE: Not inline - must export symbols for Python bindings
auto get_last_error_code() noexcept -> scl_error_t {
    return g_last_error_code;
}

// =============================================================================
// Exception to Error Code Conversion (Zero-Overhead Try-Catch)
// =============================================================================

// Convert active C++ exception to C error code
// Uses multiple catch blocks for precise error mapping
// noexcept: must handle all exceptions internally
[[nodiscard]] auto handle_exception() noexcept -> scl_error_t {
    try {
        throw;
    }
    // Specific SCL exceptions (most specific first to avoid catching by base class)
    catch (const IndexOutOfBoundsError& e) {
        set_last_error(SCL_ERROR_INDEX_OUT_OF_BOUNDS, e.what());
        return SCL_ERROR_INDEX_OUT_OF_BOUNDS;
    }
    catch (const DimensionError& e) {
        set_last_error(SCL_ERROR_DIMENSION_MISMATCH, e.what());
        return SCL_ERROR_DIMENSION_MISMATCH;
    }
    catch (const DomainError& e) {
        set_last_error(SCL_ERROR_DOMAIN_ERROR, e.what());
        return SCL_ERROR_DOMAIN_ERROR;
    }
    catch (const RangeError& e) {
        set_last_error(SCL_ERROR_RANGE_ERROR, e.what());
        return SCL_ERROR_RANGE_ERROR;
    }
    catch (const ValueError& e) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, e.what());
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    // Type errors
    catch (const TypeMismatchError& e) {
        set_last_error(SCL_ERROR_TYPE_MISMATCH, e.what());
        return SCL_ERROR_TYPE_MISMATCH;
    }
    catch (const TypeError& e) {
        set_last_error(SCL_ERROR_TYPE_ERROR, e.what());
        return SCL_ERROR_TYPE_ERROR;
    }
    // Memory and pointer errors
    catch (const NullPointerError& e) {
        set_last_error(SCL_ERROR_NULL_POINTER, e.what());
        return SCL_ERROR_NULL_POINTER;
    }
    catch (const OutOfMemoryError& e) {
        set_last_error(SCL_ERROR_OUT_OF_MEMORY, e.what());
        return SCL_ERROR_OUT_OF_MEMORY;
    }
    catch (const UnregisteredPointerError& e) {
        set_last_error(SCL_ERROR_UNREGISTERED_POINTER, e.what());
        return SCL_ERROR_UNREGISTERED_POINTER;
    }
    catch (const BufferNotFoundError& e) {
        set_last_error(SCL_ERROR_BUFFER_NOT_FOUND, e.what());
        return SCL_ERROR_BUFFER_NOT_FOUND;
    }
    // I/O errors
    catch (const FileNotFoundError& e) {
        set_last_error(SCL_ERROR_FILE_NOT_FOUND, e.what());
        return SCL_ERROR_FILE_NOT_FOUND;
    }
    catch (const ReadError& e) {
        set_last_error(SCL_ERROR_READ_ERROR, e.what());
        return SCL_ERROR_READ_ERROR;
    }
    catch (const WriteError& e) {
        set_last_error(SCL_ERROR_WRITE_ERROR, e.what());
        return SCL_ERROR_WRITE_ERROR;
    }
    catch (const PermissionDeniedError& e) {
        set_last_error(SCL_ERROR_PERMISSION_DENIED, e.what());
        return SCL_ERROR_PERMISSION_DENIED;
    }
    catch (const IOError& e) {
        set_last_error(SCL_ERROR_IO_ERROR, e.what());
        return SCL_ERROR_IO_ERROR;
    }
    // Numerical errors
    catch (const DivisionByZeroError& e) {
        set_last_error(SCL_ERROR_DIVISION_BY_ZERO, e.what());
        return SCL_ERROR_DIVISION_BY_ZERO;
    }
    catch (const OverflowError& e) {
        set_last_error(SCL_ERROR_OVERFLOW, e.what());
        return SCL_ERROR_OVERFLOW;
    }
    catch (const UnderflowError& e) {
        set_last_error(SCL_ERROR_UNDERFLOW, e.what());
        return SCL_ERROR_UNDERFLOW;
    }
    catch (const ConvergenceError& e) {
        set_last_error(SCL_ERROR_CONVERGENCE_ERROR, e.what());
        return SCL_ERROR_CONVERGENCE_ERROR;
    }
    catch (const NumericalError& e) {
        set_last_error(SCL_ERROR_NUMERICAL_ERROR, e.what());
        return SCL_ERROR_NUMERICAL_ERROR;
    }
    // Feature errors
    catch (const NotImplementedError& e) {
        set_last_error(SCL_ERROR_NOT_IMPLEMENTED, e.what());
        return SCL_ERROR_NOT_IMPLEMENTED;
    }
    catch (const FeatureUnavailableError& e) {
        set_last_error(SCL_ERROR_FEATURE_UNAVAILABLE, e.what());
        return SCL_ERROR_FEATURE_UNAVAILABLE;
    }
    // Internal errors
    catch (const InternalError& e) {
        set_last_error(SCL_ERROR_INTERNAL, e.what());
        return SCL_ERROR_INTERNAL;
    }
    // Base SCL exception
    catch (const Exception& e) {
        set_last_error(SCL_ERROR_UNKNOWN, e.what());
        return SCL_ERROR_UNKNOWN;
    }
    // Standard exceptions
    catch (const std::bad_alloc&) {
        set_last_error(SCL_ERROR_OUT_OF_MEMORY, 
                      "Memory allocation failed (std::bad_alloc)");
        return SCL_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::bad_array_new_length&) {
        set_last_error(SCL_ERROR_OUT_OF_MEMORY, 
                      "Invalid array length (std::bad_array_new_length)");
        return SCL_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::length_error& e) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, e.what());
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    catch (const std::out_of_range& e) {
        set_last_error(SCL_ERROR_RANGE_ERROR, e.what());
        return SCL_ERROR_RANGE_ERROR;
    }
    catch (const std::domain_error& e) {
        set_last_error(SCL_ERROR_DOMAIN_ERROR, e.what());
        return SCL_ERROR_DOMAIN_ERROR;
    }
    catch (const std::logic_error& e) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, e.what());
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    catch (const std::runtime_error& e) {
        set_last_error(SCL_ERROR_UNKNOWN, e.what());
        return SCL_ERROR_UNKNOWN;
    }
    catch (const std::exception& e) {
        set_last_error(SCL_ERROR_UNKNOWN, e.what());
        return SCL_ERROR_UNKNOWN;
    }
    catch (...) {
        set_last_error(SCL_ERROR_UNKNOWN, 
                      "Unknown exception (not derived from std::exception)");
        return SCL_ERROR_UNKNOWN;
    }
}

} // namespace scl::binding

// =============================================================================
// C API Implementation (Stable ABI)
// =============================================================================

extern "C" {

// =============================================================================
// Version Information
// =============================================================================

SCL_EXPORT const char* scl_get_version(void) {
    // Version string constructed at compile time
    return "1.0.0";
}

SCL_EXPORT const char* scl_get_build_config(void) {
    // Build configuration string
    static const char* config_str = 
        SCL_REAL_TYPE_NAME "+" SCL_INDEX_TYPE_NAME
#if defined(__AVX512F__)
        "+avx512"
#elif defined(__AVX2__)
        "+avx2"
#elif defined(__AVX__)
        "+avx"
#elif defined(__SSE4_2__)
        "+sse4.2"
#elif defined(__SSE4_1__)
        "+sse4.1"
#elif defined(__SSSE3__)
        "+ssse3"
#elif defined(__SSE3__)
        "+sse3"
#elif defined(__SSE2__)
        "+sse2"
#else
        "+scalar"
#endif
#if defined(_OPENMP)
        "+openmp"
#endif
        ;
    return config_str;
}

// =============================================================================
// Error Handling
// =============================================================================

SCL_EXPORT const char* scl_get_last_error(void) {
    return scl::binding::get_last_error_message();
}

SCL_EXPORT scl_error_t scl_get_last_error_code(void) {
    return scl::binding::get_last_error_code();
}

SCL_EXPORT void scl_clear_error(void) {
    scl::binding::clear_last_error();
}

SCL_EXPORT scl_bool_t scl_is_ok(scl_error_t code) {
    return (code == SCL_OK) ? SCL_TRUE : SCL_FALSE;
}

SCL_EXPORT scl_bool_t scl_is_error(scl_error_t code) {
    return (code != SCL_OK) ? SCL_TRUE : SCL_FALSE;
}

} // extern "C"
