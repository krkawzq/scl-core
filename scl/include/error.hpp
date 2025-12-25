#pragma once

#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <new>

// =============================================================================
/// @file error.hpp
/// @brief Error Handling System for C-ABI Boundary
///
/// This header defines the error handling mechanism for the C-ABI firewall.
/// C++ exceptions are caught at the boundary and converted into Error Instances
/// that can be safely passed to Python.
///
/// @section Architecture
///
/// Error handling follows this pattern:
/// 1. C++ exceptions are caught in C-ABI wrappers
/// 2. Exceptions are converted to ErrorInstance structures
/// 3. Error instances are registered in a global registry
/// 4. Python receives a pointer to the error instance
/// 5. Python looks up the exception type and raises it
///
/// =============================================================================

namespace scl {
namespace error {

/// @brief Error code enumeration
enum class ErrorCode : std::uint32_t {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_MEMORY = 2,
    RUNTIME_ERROR = 3,
    NOT_IMPLEMENTED = 4,
    UNKNOWN_ERROR = 999
};

/// @brief Error instance structure (C-compatible)
///
/// This structure can be safely passed across the C-ABI boundary.
/// The message buffer is pre-allocated to avoid dynamic allocation.
struct ErrorInstance {
    ErrorCode code;
    char message[256];  // Fixed-size buffer for error message
    
    ErrorInstance() : code(ErrorCode::SUCCESS) {
        message[0] = '\0';
    }
    
    ErrorInstance(ErrorCode c, const char* msg) : code(c) {
        std::strncpy(message, msg, sizeof(message) - 1);
        message[sizeof(message) - 1] = '\0';
    }
};

/// @brief Global error registry
///
/// Thread-safe registry for error instances. Errors are stored here
/// and referenced by pointer from Python.
class ErrorRegistry {
public:
    /// @brief Register an error instance
    ///
    /// @param code Error code
    /// @param message Error message
    /// @return Pointer to registered error instance
    static const ErrorInstance* register_error(ErrorCode code, const char* message);
    
    /// @brief Get error instance by pointer
    ///
    /// @param ptr Pointer to error instance
    /// @return Error instance, or nullptr if invalid
    static const ErrorInstance* get_error(const ErrorInstance* ptr);
    
    /// @brief Clear all registered errors (for testing)
    static void clear();
};

/// @brief Convert C++ exception to ErrorInstance
///
/// @param e Exception reference
/// @return Pointer to registered error instance
const ErrorInstance* exception_to_error(const std::exception& e);

/// @brief Convert unknown exception to ErrorInstance
///
/// @return Pointer to registered error instance
const ErrorInstance* unknown_exception_to_error();

} // namespace error
} // namespace scl

// =============================================================================
// C-ABI Interface
// =============================================================================

extern "C" {

/// @brief C-compatible error code type
typedef std::uint32_t scl_error_code_t;

/// @brief C-compatible error instance pointer
/// @note nullptr indicates success, non-null indicates error
typedef const scl::error::ErrorInstance* scl_error_t;

/// @brief Get error code from error instance
///
/// @param err Error instance pointer
/// @return Error code, or 0 if err is nullptr
scl_error_code_t scl_error_get_code(scl_error_t err);

/// @brief Get error message from error instance
///
/// @param err Error instance pointer
/// @return Error message string, or empty string if err is nullptr
const char* scl_error_get_message(scl_error_t err);

} // extern "C"

