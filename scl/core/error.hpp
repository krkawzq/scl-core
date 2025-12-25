#pragma once

#include "scl/core/macros.hpp"
#include <exception>
#include <string>
#include <utility>

// =============================================================================
/// @file error.hpp
/// @brief SCL Core Exception System
///
/// Defines the standardized exception hierarchy for the library.
/// All exceptions thrown by SCL MUST inherit from `scl::Exception` and carry
/// a unique `ErrorCode` to facilitate C-ABI translation to Python exceptions.
///
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Error Codes (C-ABI Compatible)
// =============================================================================

/// @brief Numeric identifiers for error types.
///
/// These codes map directly to Python exception types in the binding layer.
enum class ErrorCode : int32_t {
    OK = 0,                 ///< Success (no error)
    
    // --- Runtime Errors (RuntimeError) ---
    UNKNOWN = 1,            ///< Generic error
    INTERNAL_ERROR = 2,     ///< Internal logic assertion failed
    
    // --- Input Validation (ValueError) ---
    INVALID_ARGUMENT = 10,  ///< Argument value is invalid (e.g., negative size)
    DIMENSION_MISMATCH = 11,///< Tensor/Matrix dimensions do not match
    DOMAIN_ERROR = 12,      ///< Mathematical domain error (e.g., log(-1))
    
    // --- Type Errors (TypeError) ---
    TYPE_ERROR = 20,        ///< Data type mismatch (e.g., f32 vs f64)
    
    // --- System/IO (IOError) ---
    IO_ERROR = 30,          ///< File read/write failed
    
    // --- Feature Status ---
    NOT_IMPLEMENTED = 40,   ///< Feature not yet implemented
};

// =============================================================================
// SECTION 2: Base Exception Class
// =============================================================================

/// @brief Base class for all SCL exceptions.
///
/// Holds the error code and message. Inherits from std::exception for 
/// compatibility with standard C++ catch blocks.
class SCL_EXPORT Exception : public std::exception {
public:
    /// @brief Construct with code and message
    explicit Exception(ErrorCode code, std::string msg) 
        : _code(code), _msg(std::move(msg)) {}

    /// @brief Get the standard C-style error message
    const char* what() const noexcept override {
        return _msg.c_str();
    }

    /// @brief Get the SCL error code
    ErrorCode code() const noexcept {
        return _code;
    }

protected:
    ErrorCode _code;
    std::string _msg;
};

// =============================================================================
// SECTION 3: Specialized Exception Classes
// =============================================================================

// --- Runtime Errors ---

/// @brief Generic runtime failure (Maps to Python `RuntimeError`)
class RuntimeError : public Exception {
public:
    explicit RuntimeError(const std::string& msg) 
        : Exception(ErrorCode::UNKNOWN, msg) {}
    
    explicit RuntimeError(ErrorCode code, const std::string& msg)
        : Exception(code, msg) {}
};

/// @brief Internal library failure (Maps to Python `RuntimeError` or `AssertionError`)
class InternalError : public RuntimeError {
public:
    explicit InternalError(const std::string& msg) 
        : RuntimeError(ErrorCode::INTERNAL_ERROR, "Internal SCL Error: " + msg) {}
};

// --- Value Errors ---

/// @brief Invalid argument value (Maps to Python `ValueError`)
class ValueError : public Exception {
public:
    explicit ValueError(const std::string& msg) 
        : Exception(ErrorCode::INVALID_ARGUMENT, msg) {}
    
protected:
    // Allow subclasses to set specific codes
    ValueError(ErrorCode code, std::string msg) 
        : Exception(code, std::move(msg)) {}
};

/// @brief Dimension mismatch in tensor ops (Maps to Python `ValueError`)
class DimensionError : public ValueError {
public:
    explicit DimensionError(const std::string& msg) 
        : ValueError(ErrorCode::DIMENSION_MISMATCH, msg) {}
};

// --- Other Errors ---

/// @brief Type mismatch (Maps to Python `TypeError`)
class TypeError : public Exception {
public:
    explicit TypeError(const std::string& msg) 
        : Exception(ErrorCode::TYPE_ERROR, msg) {}
};

/// @brief IO failure (Maps to Python `IOError` / `OSError`)
class IOError : public Exception {
public:
    explicit IOError(const std::string& msg) 
        : Exception(ErrorCode::IO_ERROR, msg) {}
};

/// @brief Feature not implemented (Maps to Python `NotImplementedError`)
class NotImplementedError : public Exception {
public:
    explicit NotImplementedError(const std::string& msg = "Not implemented yet") 
        : Exception(ErrorCode::NOT_IMPLEMENTED, msg) {}
};

// =============================================================================
// SECTION 4: Helper Macros
// =============================================================================

/// @brief Assertion macro that throws InternalError on failure.
///
/// Use this for internal invariant checking. It remains active in Release builds
/// (unlike assert) because library safety is paramount.
#define SCL_ASSERT(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::InternalError(std::string(msg) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
        } \
    } while(0)

/// @brief Validation macro that throws ValueError on failure.
///
/// Use this for checking user inputs (API boundaries).
#define SCL_CHECK_ARG(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::ValueError(msg); \
        } \
    } while(0)

/// @brief Validation macro for dimensions.
#define SCL_CHECK_DIM(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::DimensionError(msg); \
        } \
    } while(0)

} // namespace scl
