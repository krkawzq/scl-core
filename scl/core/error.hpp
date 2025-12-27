#pragma once

#include "scl/core/macros.hpp"
#include <exception>
#include <string>
#include <utility>

// =============================================================================
// FILE: scl/core/error.hpp
// BRIEF: SCL Core Exception System
// =============================================================================

namespace scl {

// =============================================================================
// Error Codes (C-ABI Compatible)
// =============================================================================

enum class ErrorCode : int32_t {
    OK = 0,

    UNKNOWN = 1,
    INTERNAL_ERROR = 2,

    INVALID_ARGUMENT = 10,
    DIMENSION_MISMATCH = 11,
    DOMAIN_ERROR = 12,

    TYPE_ERROR = 20,

    IO_ERROR = 30,

    UNREGISTERED_POINTER = 35,

    NOT_IMPLEMENTED = 40,
};

// =============================================================================
// Base Exception Class
// =============================================================================

class SCL_EXPORT Exception : public std::exception {
public:
    explicit Exception(ErrorCode code, std::string msg)
        : _code(code), _msg(std::move(msg)) {}

    const char* what() const noexcept override {
        return _msg.c_str();
    }

    ErrorCode code() const noexcept {
        return _code;
    }

protected:
    ErrorCode _code;
    std::string _msg;
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

class TypeError : public Exception {
public:
    explicit TypeError(const std::string& msg)
        : Exception(ErrorCode::TYPE_ERROR, msg) {}
};

class IOError : public Exception {
public:
    explicit IOError(const std::string& msg)
        : Exception(ErrorCode::IO_ERROR, msg) {}
};

class NotImplementedError : public Exception {
public:
    explicit NotImplementedError(const std::string& msg = "Not implemented yet")
        : Exception(ErrorCode::NOT_IMPLEMENTED, msg) {}
};

class UnregisteredPointerError : public RuntimeError {
public:
    explicit UnregisteredPointerError(const void* ptr)
        : RuntimeError(ErrorCode::UNREGISTERED_POINTER,
                      "Attempted to access unregistered pointer: " + ptr_to_string(ptr)) {}

    explicit UnregisteredPointerError(const std::string& msg)
        : RuntimeError(ErrorCode::UNREGISTERED_POINTER, msg) {}

private:
    static std::string ptr_to_string(const void* ptr) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%p", ptr);
        return std::string(buf);
    }
};

// =============================================================================
// Helper Macros
// =============================================================================

// Assertion for internal invariants (active in all builds)
#define SCL_ASSERT(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::InternalError(std::string(msg) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
        } \
    } while(0)

// Validation for user inputs
#define SCL_CHECK_ARG(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::ValueError(msg); \
        } \
    } while(0)

// Validation for dimension mismatches
#define SCL_CHECK_DIM(condition, msg) \
    do { \
        if (SCL_UNLIKELY(!(condition))) { \
            throw scl::DimensionError(msg); \
        } \
    } while(0)

} // namespace scl
