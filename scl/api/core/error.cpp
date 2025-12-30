/// @file scl/api/core/error.cpp
/// @brief C-API bindings for SCL error handling
///
/// This file provides C-ABI compatible functions for:
///   - Error code queries and conversions
///   - Thread-local error state access
///   - Error message retrieval
///
/// ## Usage from C/Python:
///
/// ```c
/// // Check last error after an operation
/// int32_t code = scl_get_last_error();
/// if (code != 0) {
///     const char* msg = scl_get_error_message();
///     printf("Error %d: %s\n", code, msg);
///     scl_clear_error();
/// }
/// ```

#include "scl/core/error.hpp"

#include <cstdint>
#include <cstring>

// =============================================================================
// C-ABI Export Macros
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef SCL_BUILDING_LIBRARY
        #define SCL_API extern "C" __declspec(dllexport)
    #else
        #define SCL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define SCL_API extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// SECTION 1: Error Code Queries
// =============================================================================

/// @brief Get human-readable name for an error code
/// @param[in] code Error code value
/// @return Static string with error name (never null)
/// @note Thread-safe, no memory allocation
SCL_API
auto scl_error_code_name(std::int32_t code) -> const char* {
    return scl::error_code_name(static_cast<scl::ErrorCode>(code));
}

/// @brief Get error category name for an error code
/// @param[in] code Error code value
/// @return Static string with category name (e.g., "Memory", "Value")
/// @note Thread-safe, no memory allocation
SCL_API
auto scl_error_code_category(std::int32_t code) -> const char* {
    return scl::error_code_category(static_cast<scl::ErrorCode>(code));
}

/// @brief Check if error code represents success
/// @param[in] code Error code value
/// @return 1 if success (code == 0), 0 otherwise
SCL_API
auto scl_is_success(std::int32_t code) -> std::int32_t {
    return scl::is_success(static_cast<scl::ErrorCode>(code)) ? 1 : 0;
}

/// @brief Check if error code represents a failure
/// @param[in] code Error code value
/// @return 1 if error (code != 0), 0 otherwise
SCL_API
auto scl_is_error(std::int32_t code) -> std::int32_t {
    return scl::is_error(static_cast<scl::ErrorCode>(code)) ? 1 : 0;
}

/// @brief Check if error is recoverable (not internal/hardware)
/// @param[in] code Error code value
/// @return 1 if recoverable, 0 otherwise
SCL_API
auto scl_is_recoverable(std::int32_t code) -> std::int32_t {
    return scl::is_recoverable(static_cast<scl::ErrorCode>(code)) ? 1 : 0;
}

// =============================================================================
// SECTION 2: Thread-Local Error State
// =============================================================================

/// @brief Get the last error code for the current thread
/// @return Error code (0 = success, non-zero = error)
/// @note Thread-safe, each thread has its own error state
SCL_API
auto scl_get_last_error() -> std::int32_t {
    return static_cast<std::int32_t>(scl::get_thread_error().code());
}

/// @brief Get the last error message for the current thread
/// @return Pointer to error message string (valid until next error or clear)
/// @note Thread-safe, returns empty string if no error
/// @warning Do not free the returned pointer
SCL_API
auto scl_get_error_message() -> const char* {
    return scl::get_thread_error().message();
}

/// @brief Clear the thread-local error state
/// @note Thread-safe, only affects the calling thread
SCL_API
auto scl_clear_error() -> void {
    scl::clear_thread_error();
}

/// @brief Set a custom error for the current thread
/// @param[in] code Error code value
/// @param[in] message Error message (can be null, will use default)
/// @note Thread-safe, message is copied internally
SCL_API
auto scl_set_error(std::int32_t code, const char* message) -> void {
    scl::set_thread_error(static_cast<scl::ErrorCode>(code), message);
}

/// @brief Check if an error is currently set for this thread
/// @return 1 if error is set, 0 otherwise
SCL_API
auto scl_has_error() -> std::int32_t {
    return scl::get_thread_error().has_error() ? 1 : 0;
}

// =============================================================================
// SECTION 3: Error Code Constants
// =============================================================================

// These constants allow C code to reference error codes without casting

/// @brief Success code
SCL_API
auto scl_error_success() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::Success);
}

/// @brief Unknown/unspecified error
SCL_API
auto scl_error_unknown() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

/// @brief Not implemented error
SCL_API
auto scl_error_not_implemented() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::NotImplemented);
}

/// @brief Out of memory error
SCL_API
auto scl_error_out_of_memory() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::OutOfMemory);
}

/// @brief Null pointer error
SCL_API
auto scl_error_null_pointer() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::NullPointer);
}

/// @brief Invalid argument error
SCL_API
auto scl_error_invalid_argument() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::InvalidArgument);
}

/// @brief Index out of bounds error
SCL_API
auto scl_error_index_out_of_bounds() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::IndexOutOfBounds);
}

/// @brief Dimension mismatch error
SCL_API
auto scl_error_dimension_mismatch() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::DimensionMismatch);
}

/// @brief Shape mismatch error
SCL_API
auto scl_error_shape_mismatch() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::ShapeMismatch);
}

/// @brief Type mismatch error
SCL_API
auto scl_error_type_mismatch() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::TypeMismatch);
}

/// @brief Division by zero error
SCL_API
auto scl_error_division_by_zero() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::DivisionByZero);
}

/// @brief NaN encountered error
SCL_API
auto scl_error_nan() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::NaN);
}

/// @brief Numerical overflow error
SCL_API
auto scl_error_overflow() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::Overflow);
}

/// @brief Computation error (general)
SCL_API
auto scl_error_computation() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::ComputationError);
}

/// @brief Convergence error
SCL_API
auto scl_error_convergence() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::ConvergenceError);
}

/// @brief Singular matrix error
SCL_API
auto scl_error_singular_matrix() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::SingularMatrix);
}

/// @brief IO error (general)
SCL_API
auto scl_error_io() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::IoError);
}

/// @brief File not found error
SCL_API
auto scl_error_file_not_found() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::FileNotFound);
}

/// @brief Internal error
SCL_API
auto scl_error_internal() -> std::int32_t {
    return static_cast<std::int32_t>(scl::ErrorCode::InternalError);
}

// =============================================================================
// SECTION 4: Error Code Range Queries
// =============================================================================

/// @brief Get the minimum value for memory error codes
SCL_API
auto scl_error_memory_min() -> std::int32_t {
    return 100;  // ErrorCode::OutOfMemory
}

/// @brief Get the maximum value for memory error codes
SCL_API
auto scl_error_memory_max() -> std::int32_t {
    return 199;
}

/// @brief Get the minimum value for dimension error codes
SCL_API
auto scl_error_dimension_min() -> std::int32_t {
    return 200;  // ErrorCode::DimensionMismatch
}

/// @brief Get the maximum value for dimension error codes
SCL_API
auto scl_error_dimension_max() -> std::int32_t {
    return 299;
}

/// @brief Get the minimum value for type error codes
SCL_API
auto scl_error_type_min() -> std::int32_t {
    return 300;  // ErrorCode::TypeMismatch
}

/// @brief Get the maximum value for type error codes
SCL_API
auto scl_error_type_max() -> std::int32_t {
    return 399;
}

/// @brief Get the minimum value for value/argument error codes
SCL_API
auto scl_error_value_min() -> std::int32_t {
    return 400;  // ErrorCode::InvalidArgument
}

/// @brief Get the maximum value for value/argument error codes
SCL_API
auto scl_error_value_max() -> std::int32_t {
    return 499;
}

/// @brief Get the minimum value for IO error codes
SCL_API
auto scl_error_io_min() -> std::int32_t {
    return 500;  // ErrorCode::IoError
}

/// @brief Get the maximum value for IO error codes
SCL_API
auto scl_error_io_max() -> std::int32_t {
    return 599;
}

/// @brief Get the minimum value for computation error codes
SCL_API
auto scl_error_compute_min() -> std::int32_t {
    return 600;  // ErrorCode::ComputationError
}

/// @brief Get the maximum value for computation error codes
SCL_API
auto scl_error_compute_max() -> std::int32_t {
    return 699;
}

/// @brief Get the minimum value for threading error codes
SCL_API
auto scl_error_thread_min() -> std::int32_t {
    return 700;  // ErrorCode::ThreadError
}

/// @brief Get the maximum value for threading error codes
SCL_API
auto scl_error_thread_max() -> std::int32_t {
    return 799;
}

/// @brief Get the minimum value for internal error codes
SCL_API
auto scl_error_internal_min() -> std::int32_t {
    return 900;  // ErrorCode::InternalError
}

/// @brief Get the maximum value for internal error codes
SCL_API
auto scl_error_internal_max() -> std::int32_t {
    return 999;
}

// =============================================================================
// SECTION 5: Error Information Queries
// =============================================================================

/// @brief Get full error information as formatted string
/// @param[out] buffer Output buffer for error string
/// @param[in] buffer_size Size of output buffer in bytes
/// @return Number of bytes written (excluding null terminator), or required size if buffer too small
/// @note Format: "[CODE_NAME] message"
SCL_API
auto scl_get_error_info(char* buffer, std::size_t buffer_size) -> std::size_t {
    const auto& state = scl::get_thread_error();
    const char* code_name = scl::error_code_name(state.code());
    const char* message = state.message();
    
    const std::size_t code_len = std::strlen(code_name);
    const std::size_t msg_len = std::strlen(message);
    // Format: "[CODE_NAME] message\0" = 1 + code_len + 2 + msg_len + 1
    const std::size_t required_size = code_len + msg_len + 4;
    
    if (buffer == nullptr || buffer_size == 0) {
        // Return required size
        return required_size;
    }
    
    // Build the formatted string manually (avoid varargs)
    std::size_t pos = 0;
    
    // "["
    if (pos < buffer_size - 1) {
        buffer[pos++] = '[';
    }
    
    // CODE_NAME
    for (std::size_t i = 0; i < code_len && pos < buffer_size - 1; ++i) {
        buffer[pos++] = code_name[i];
    }
    
    // "] "
    if (pos < buffer_size - 1) {
        buffer[pos++] = ']';
    }
    if (pos < buffer_size - 1) {
        buffer[pos++] = ' ';
    }
    
    // message
    for (std::size_t i = 0; i < msg_len && pos < buffer_size - 1; ++i) {
        buffer[pos++] = message[i];
    }
    
    // null terminator
    buffer[pos] = '\0';
    
    return pos;
}

/// @brief Copy error message to a user-provided buffer
/// @param[out] buffer Output buffer for message
/// @param[in] buffer_size Size of output buffer in bytes
/// @return Number of bytes written (excluding null terminator)
/// @note Safer alternative to scl_get_error_message for multi-threaded use
SCL_API
auto scl_copy_error_message(char* buffer, std::size_t buffer_size) -> std::size_t {
    if (buffer == nullptr || buffer_size == 0) {
        return 0;
    }
    
    const char* message = scl::get_thread_error().message();
    const std::size_t msg_len = std::strlen(message);
    const std::size_t copy_len = (msg_len < buffer_size - 1) ? msg_len : (buffer_size - 1);
    
    std::memcpy(buffer, message, copy_len);
    buffer[copy_len] = '\0';
    
    return copy_len;
}

// =============================================================================
// SECTION 6: Version and Library Info
// =============================================================================

/// @brief Get library version string
/// @return Static string with version (e.g., "2.0.0")
SCL_API
auto scl_get_version() -> const char* {
    return "2.0.0";
}

/// @brief Get library version as integer components
/// @param[out] major Major version number
/// @param[out] minor Minor version number  
/// @param[out] patch Patch version number
SCL_API
auto scl_get_version_components(
    std::int32_t* major, 
    std::int32_t* minor, 
    std::int32_t* patch
) -> void {
    if (major) *major = 2;
    if (minor) *minor = 0;
    if (patch) *patch = 0;
}

