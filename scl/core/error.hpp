#pragma once

#include "scl/core/macros.hpp"
#include <exception>
#include <string>
#include <utility>
#include <sstream>
#include <iomanip>
#include <cstdint>

// =============================================================================
// FILE: scl/core/error.hpp
// BRIEF: SCL Core Exception System
// =============================================================================

namespace scl {

// =============================================================================
// Error Codes (C-ABI Compatible)
// =============================================================================

enum class ErrorCode : std::int32_t {
    OK = 0,

    // General errors
    UNKNOWN = 1,
    INTERNAL_ERROR = 2,
    OUT_OF_MEMORY = 3,
    NULL_POINTER = 4,

    // Argument errors
    INVALID_ARGUMENT = 10,
    DIMENSION_MISMATCH = 11,
    DOMAIN_ERROR = 12,
    RANGE_ERROR = 13,
    INDEX_OUT_OF_BOUNDS = 14,

    // Type errors
    TYPE_ERROR = 20,
    TYPE_MISMATCH = 21,

    // I/O errors
    IO_ERROR = 30,
    FILE_NOT_FOUND = 31,
    PERMISSION_DENIED = 32,
    READ_ERROR = 33,
    WRITE_ERROR = 34,

    // Registry errors
    UNREGISTERED_POINTER = 35,
    BUFFER_NOT_FOUND = 36,

    // Feature errors
    NOT_IMPLEMENTED = 40,
    FEATURE_UNAVAILABLE = 41,

    // Numerical errors
    NUMERICAL_ERROR = 50,
    DIVISION_BY_ZERO = 51,
    OVERFLOW = 52,
    UNDERFLOW = 53,
    CONVERGENCE_ERROR = 54,
};

// =============================================================================
// Base Exception Class
// =============================================================================

class SCL_EXPORT Exception : public std::exception {
public:
    explicit Exception(ErrorCode code, std::string msg)
        : code_(code), msg_(std::move(msg)) {}

    [[nodiscard]] auto what() const noexcept -> const char* override {
        return msg_.c_str();
    }

    [[nodiscard]] auto code() const noexcept -> ErrorCode {
        return code_;
    }

    [[nodiscard]] auto message() const noexcept -> const std::string& {
        return msg_;
    }

protected:
    // Protected members for derived classes to access and modify
    // This is intentional design to allow derived classes to set error codes
    // NOLINTNEXTLINE(*-non-private-member-variables-in-classes)
    ErrorCode code_;
    // NOLINTNEXTLINE(*-non-private-member-variables-in-classes)
    std::string msg_;
};

// =============================================================================
// Specialized Exception Classes
// =============================================================================

class RuntimeError : public Exception {
public:
    explicit RuntimeError(const std::string& msg)
        : Exception(ErrorCode::UNKNOWN, msg) {}

    explicit RuntimeError(ErrorCode code, const std::string& msg)
        : Exception(code, msg) {}
};

class OutOfMemoryError : public RuntimeError {
public:
    explicit OutOfMemoryError(const std::string& msg = "Out of memory")
        : RuntimeError(ErrorCode::OUT_OF_MEMORY, msg) {}
};

class NullPointerError : public RuntimeError {
public:
    explicit NullPointerError(const std::string& msg = "Null pointer encountered")
        : RuntimeError(ErrorCode::NULL_POINTER, msg) {}
};

class InternalError : public RuntimeError {
public:
    explicit InternalError(const std::string& msg)
        : RuntimeError(ErrorCode::INTERNAL_ERROR, "Internal SCL Error: " + msg) {}
};

class ValueError : public Exception {
public:
    explicit ValueError(const std::string& msg)
        : Exception(ErrorCode::INVALID_ARGUMENT, msg) {}

protected:
    ValueError(ErrorCode code, std::string msg)
        : Exception(code, std::move(msg)) {}
};

class DimensionError : public ValueError {
public:
    explicit DimensionError(const std::string& msg)
        : ValueError(ErrorCode::DIMENSION_MISMATCH, msg) {}
};

class DomainError : public ValueError {
public:
    explicit DomainError(const std::string& msg)
        : ValueError(ErrorCode::DOMAIN_ERROR, msg) {}
};

class RangeError : public ValueError {
public:
    explicit RangeError(const std::string& msg)
        : ValueError(ErrorCode::RANGE_ERROR, msg) {}
};

class IndexOutOfBoundsError : public ValueError {
public:
    explicit IndexOutOfBoundsError(const std::string& msg)
        : ValueError(ErrorCode::INDEX_OUT_OF_BOUNDS, msg) {}
};

class TypeError : public Exception {
public:
    explicit TypeError(const std::string& msg)
        : Exception(ErrorCode::TYPE_ERROR, msg) {}

protected:
    explicit TypeError(ErrorCode code, const std::string& msg)
        : Exception(code, msg) {}
};

class TypeMismatchError : public TypeError {
public:
    explicit TypeMismatchError(const std::string& msg)
        : TypeError(ErrorCode::TYPE_MISMATCH, msg) {}
};

class IOError : public Exception {
public:
    explicit IOError(const std::string& msg)
        : Exception(ErrorCode::IO_ERROR, msg) {}

protected:
    explicit IOError(ErrorCode code, const std::string& msg)
        : Exception(code, msg) {}
};

class FileNotFoundError : public IOError {
public:
    explicit FileNotFoundError(const std::string& path)
        : IOError(ErrorCode::FILE_NOT_FOUND, "File not found: " + path) {}
};

class PermissionDeniedError : public IOError {
public:
    explicit PermissionDeniedError(const std::string& msg)
        : IOError(ErrorCode::PERMISSION_DENIED, msg) {}
};

class ReadError : public IOError {
public:
    explicit ReadError(const std::string& msg)
        : IOError(ErrorCode::READ_ERROR, msg) {}
};

class WriteError : public IOError {
public:
    explicit WriteError(const std::string& msg)
        : IOError(ErrorCode::WRITE_ERROR, msg) {}
};

class NotImplementedError : public Exception {
public:
    explicit NotImplementedError(const std::string& msg = "Not implemented yet")
        : Exception(ErrorCode::NOT_IMPLEMENTED, msg) {}
};

class FeatureUnavailableError : public Exception {
public:
    explicit FeatureUnavailableError(const std::string& msg)
        : Exception(ErrorCode::FEATURE_UNAVAILABLE, msg) {}
};

class UnregisteredPointerError : public RuntimeError {
public:
    explicit UnregisteredPointerError(const void* ptr)
        : RuntimeError(ErrorCode::UNREGISTERED_POINTER,
                      "Attempted to access unregistered pointer: " + ptr_to_string(ptr)) {}

    explicit UnregisteredPointerError(const std::string& msg)
        : RuntimeError(ErrorCode::UNREGISTERED_POINTER, msg) {}

private:
    static constexpr std::size_t PTR_BUFFER_SIZE = 32;
    
    static auto ptr_to_string(const void* ptr) -> std::string {
        std::ostringstream oss;
        oss << std::hex << std::showbase << ptr;
        return oss.str();
    }
};

class BufferNotFoundError : public RuntimeError {
public:
    explicit BufferNotFoundError(const std::string& msg)
        : RuntimeError(ErrorCode::BUFFER_NOT_FOUND, msg) {}
};

// =============================================================================
// Numerical Errors
// =============================================================================

class NumericalError : public Exception {
public:
    explicit NumericalError(const std::string& msg)
        : Exception(ErrorCode::NUMERICAL_ERROR, msg) {}

protected:
    explicit NumericalError(ErrorCode code, const std::string& msg)
        : Exception(code, msg) {}
};

class DivisionByZeroError : public NumericalError {
public:
    explicit DivisionByZeroError(const std::string& msg = "Division by zero")
        : NumericalError(ErrorCode::DIVISION_BY_ZERO, msg) {}
};

class OverflowError : public NumericalError {
public:
    explicit OverflowError(const std::string& msg = "Numerical overflow")
        : NumericalError(ErrorCode::OVERFLOW, msg) {}
};

class UnderflowError : public NumericalError {
public:
    explicit UnderflowError(const std::string& msg = "Numerical underflow")
        : NumericalError(ErrorCode::UNDERFLOW, msg) {}
};

class ConvergenceError : public NumericalError {
public:
    explicit ConvergenceError(const std::string& msg = "Algorithm did not converge")
        : NumericalError(ErrorCode::CONVERGENCE_ERROR, msg) {}
};

// =============================================================================
// Helper Macros
// =============================================================================

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
// Assertion for internal invariants (active in all builds)
// Must remain as macro for __FILE__ and __LINE__ expansion
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_ASSERT(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::InternalError(std::string(msg) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
        } \
    } while(0)

// Validation for user inputs
// Must remain as macro for consistent error handling
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_CHECK_ARG(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::ValueError(msg); \
        } \
    } while(0)

// Validation for dimension mismatches
// Must remain as macro for consistent error handling
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_CHECK_DIM(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::DimensionError(msg); \
        } \
    } while(0)

// Validation for null pointers
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_CHECK_NULL(ptr, msg) \
    do { \
        if (SCL_UNLIKELY((ptr) == nullptr)) { \
            throw scl::NullPointerError(msg); \
        } \
    } while(0)

// Validation for index bounds
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_CHECK_BOUNDS(index, size, msg) \
    do { \
        if (SCL_UNLIKELY((index) < 0 || static_cast<std::size_t>(index) >= (size))) { \
            throw scl::IndexOutOfBoundsError(msg); \
        } \
    } while(0)

// Validation for range errors
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_CHECK_RANGE(value, min_val, max_val, msg) \
    do { \
        if (SCL_UNLIKELY((value) < (min_val) || (value) > (max_val))) { \
            throw scl::RangeError(msg); \
        } \
    } while(0)
// NOLINTEND(cppcoreguidelines-macro-usage)

} // namespace scl
