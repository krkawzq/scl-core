// =============================================================================
// FILE: scl/core/error.h
// BRIEF: API reference for SCL Core Exception System
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <exception>
#include <string>

namespace scl {

// =============================================================================
// ERROR CODES
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: ErrorCode
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Numeric identifiers for exception types, compatible with C-ABI.
 *
 * PURPOSE:
 *     These codes enable translation between C++ exceptions and Python
 *     exceptions across the C-ABI boundary. Each code maps to a specific
 *     Python exception type in the binding layer.
 *
 * VALUES:
 *     OK                   (0)  - No error (success status)
 *     UNKNOWN              (1)  - Generic runtime error
 *     INTERNAL_ERROR       (2)  - Internal library logic failure
 *     INVALID_ARGUMENT     (10) - Invalid parameter value
 *     DIMENSION_MISMATCH   (11) - Tensor/matrix dimension incompatibility
 *     DOMAIN_ERROR         (12) - Mathematical domain violation
 *     TYPE_ERROR           (20) - Data type mismatch
 *     IO_ERROR             (30) - File I/O failure
 *     UNREGISTERED_POINTER (35) - Access to unregistered memory pointer
 *     NOT_IMPLEMENTED      (40) - Feature not yet implemented
 *
 * PYTHON MAPPING:
 *     OK                   -> No exception
 *     UNKNOWN              -> RuntimeError
 *     INTERNAL_ERROR       -> RuntimeError or AssertionError
 *     INVALID_ARGUMENT     -> ValueError
 *     DIMENSION_MISMATCH   -> ValueError
 *     DOMAIN_ERROR         -> ValueError
 *     TYPE_ERROR           -> TypeError
 *     IO_ERROR             -> IOError/OSError
 *     UNREGISTERED_POINTER -> RuntimeError
 *     NOT_IMPLEMENTED      -> NotImplementedError
 *
 * INVARIANTS:
 *     - Codes are stable across versions (ABI compatibility)
 *     - OK is always zero
 *     - Related errors are grouped by tens (10-19 for validation, etc.)
 * -------------------------------------------------------------------------- */
enum class ErrorCode : int32_t {
    OK = 0,                     // Success

    UNKNOWN = 1,                // Generic error
    INTERNAL_ERROR = 2,         // Internal logic failure

    INVALID_ARGUMENT = 10,      // Invalid parameter value
    DIMENSION_MISMATCH = 11,    // Dimension incompatibility
    DOMAIN_ERROR = 12,          // Mathematical domain violation

    TYPE_ERROR = 20,            // Data type mismatch

    IO_ERROR = 30,              // File I/O failure

    UNREGISTERED_POINTER = 35,  // Unregistered pointer access

    NOT_IMPLEMENTED = 40,       // Feature not implemented
};

// =============================================================================
// BASE EXCEPTION CLASS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Exception
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Base class for all SCL exceptions.
 *
 * INHERITANCE:
 *     std::exception (for standard C++ compatibility)
 *
 * PURPOSE:
 *     Provides a unified exception interface with both error code (for C-ABI)
 *     and message (for human readability). All SCL exceptions inherit from
 *     this class.
 *
 * MEMBERS:
 *     _code [protected] - ErrorCode enum value
 *     _msg  [protected] - Human-readable error description
 *
 * PRECONDITIONS:
 *     - code must be a valid ErrorCode value
 *     - msg should be descriptive without being verbose
 *
 * POSTCONDITIONS:
 *     - Exception is ready to throw
 *     - what() returns valid C-string
 *     - code() returns the original code
 *
 * THREAD SAFETY:
 *     Safe - all methods are const or noexcept
 *
 * USAGE NOTES:
 *     - Do not throw Exception directly; use specialized subclasses
 *     - Code enables programmatic error handling across language boundaries
 * -------------------------------------------------------------------------- */
class Exception : public std::exception {
public:
    // Construct with code and message
    explicit Exception(
        ErrorCode code,         // Error classification code
        std::string msg         // Descriptive error message
    );

    // Get C-style error message
    const char* what() const noexcept override;

    // Get SCL error code
    ErrorCode code() const noexcept;

protected:
    ErrorCode _code;
    std::string _msg;
};

// =============================================================================
// RUNTIME ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: RuntimeError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic runtime failure exception.
 *
 * INHERITANCE:
 *     Exception
 *
 * PURPOSE:
 *     Represents errors that occur during execution but are not necessarily
 *     programmer errors. Maps to Python RuntimeError.
 *
 * PRECONDITIONS:
 *     - msg should describe what went wrong
 *
 * POSTCONDITIONS:
 *     - code() returns UNKNOWN (default) or specified code
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Unexpected state encountered during execution
 *     - External conditions prevent operation completion
 *     - Not for programmer errors (use InternalError)
 *     - Not for invalid inputs (use ValueError)
 * -------------------------------------------------------------------------- */
class RuntimeError : public Exception {
public:
    // Create with UNKNOWN code
    explicit RuntimeError(
        const std::string& msg  // Error description
    );

    // Create with specific code (for subclasses)
    explicit RuntimeError(
        ErrorCode code,         // Specific error code
        const std::string& msg  // Error description
    );
};

/* -----------------------------------------------------------------------------
 * CLASS: InternalError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Internal library logic failure.
 *
 * INHERITANCE:
 *     RuntimeError
 *
 * PURPOSE:
 *     Indicates a bug in the SCL library itself. These exceptions should
 *     never occur in correct usage and represent assertion failures or
 *     violated invariants.
 *
 * PRECONDITIONS:
 *     - msg describes the violated invariant
 *
 * POSTCONDITIONS:
 *     - code() returns INTERNAL_ERROR
 *     - what() is prefixed with "Internal SCL Error: "
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Internal invariant violated
 *     - Unreachable code path executed
 *     - Data structure corruption detected
 *     - Never for user input errors
 *
 * DEBUGGING:
 *     - These indicate library bugs that should be reported
 *     - SCL_ASSERT macro throws this exception type
 * -------------------------------------------------------------------------- */
class InternalError : public RuntimeError {
public:
    // Create internal error
    explicit InternalError(
        const std::string& msg  // Invariant violation description
    );
};

// =============================================================================
// VALUE ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: ValueError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Invalid argument value exception.
 *
 * INHERITANCE:
 *     Exception
 *
 * PURPOSE:
 *     Indicates that a function received an argument with an invalid value,
 *     even if the type is correct. Maps to Python ValueError.
 *
 * PRECONDITIONS:
 *     - msg describes what is invalid and why
 *
 * POSTCONDITIONS:
 *     - code() returns INVALID_ARGUMENT (default) or specified code
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Negative size where positive required
 *     - Out-of-range index
 *     - Invalid enum value
 *     - Null pointer where non-null required
 *     - Invalid combination of arguments
 *
 * EXAMPLES OF INVALID VALUES:
 *     - n_components = -5
 *     - alpha = inf or nan
 *     - empty array where non-empty required
 * -------------------------------------------------------------------------- */
class ValueError : public Exception {
public:
    // Create with INVALID_ARGUMENT code
    explicit ValueError(
        const std::string& msg  // Validation failure description
    );

protected:
    // Allow subclasses to set specific codes
    ValueError(
        ErrorCode code,         // Specific value error code
        std::string msg         // Error description
    );
};

/* -----------------------------------------------------------------------------
 * CLASS: DimensionError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Tensor or matrix dimension mismatch exception.
 *
 * INHERITANCE:
 *     ValueError
 *
 * PURPOSE:
 *     Specialized ValueError for dimension-related errors in matrix and
 *     tensor operations. Maps to Python ValueError.
 *
 * PRECONDITIONS:
 *     - msg includes actual and expected dimensions
 *
 * POSTCONDITIONS:
 *     - code() returns DIMENSION_MISMATCH
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Matrix multiplication with incompatible shapes
 *     - Broadcasting failure
 *     - Output buffer size mismatch
 *     - Axis out of bounds
 *
 * MESSAGE GUIDELINES:
 *     Include actual and expected dimensions:
 *     "Matrix shape mismatch: expected (100, 50), got (100, 60)"
 * -------------------------------------------------------------------------- */
class DimensionError : public ValueError {
public:
    // Create dimension error
    explicit DimensionError(
        const std::string& msg  // Dimension incompatibility description
    );
};

// =============================================================================
// TYPE ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: TypeError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Data type mismatch exception.
 *
 * INHERITANCE:
 *     Exception
 *
 * PURPOSE:
 *     Indicates an operation received data of an incompatible type.
 *     Maps to Python TypeError.
 *
 * PRECONDITIONS:
 *     - msg describes expected vs actual type
 *
 * POSTCONDITIONS:
 *     - code() returns TYPE_ERROR
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - float32 operation receives float64 data
 *     - CSR matrix operation receives COO matrix
 *     - Sparse operation receives dense matrix
 *     - Unsupported data type for operation
 * -------------------------------------------------------------------------- */
class TypeError : public Exception {
public:
    // Create type error
    explicit TypeError(
        const std::string& msg  // Type incompatibility description
    );
};

// =============================================================================
// IO ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: IOError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     File or I/O operation failure exception.
 *
 * INHERITANCE:
 *     Exception
 *
 * PURPOSE:
 *     Represents failures in file reading, writing, or other I/O operations.
 *     Maps to Python IOError/OSError.
 *
 * PRECONDITIONS:
 *     - msg includes file path and error details when available
 *
 * POSTCONDITIONS:
 *     - code() returns IO_ERROR
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - File not found
 *     - Permission denied
 *     - Disk full
 *     - Read/write failure
 *     - Corrupt file format
 * -------------------------------------------------------------------------- */
class IOError : public Exception {
public:
    // Create IO error
    explicit IOError(
        const std::string& msg  // I/O failure description
    );
};

// =============================================================================
// FEATURE STATUS ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: NotImplementedError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Feature not yet implemented exception.
 *
 * INHERITANCE:
 *     Exception
 *
 * PURPOSE:
 *     Indicates functionality that is planned but not yet available.
 *     Maps to Python NotImplementedError.
 *
 * PRECONDITIONS:
 *     - None
 *
 * POSTCONDITIONS:
 *     - code() returns NOT_IMPLEMENTED
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Feature stubbed out for future implementation
 *     - Platform-specific code not available on current platform
 *     - Optional algorithm variant not yet coded
 *
 * USAGE NOTES:
 *     - Should be temporary; remove once feature implemented
 *     - Consider documenting timeline or alternative approaches
 * -------------------------------------------------------------------------- */
class NotImplementedError : public Exception {
public:
    // Create not-implemented error
    explicit NotImplementedError(
        const std::string& msg = "Not implemented yet"  // Optional description
    );
};

// =============================================================================
// MEMORY MANAGEMENT ERRORS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: UnregisteredPointerError
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Attempted access to unregistered memory pointer.
 *
 * INHERITANCE:
 *     RuntimeError
 *
 * PURPOSE:
 *     Thrown when code attempts to access a pointer that is not registered
 *     in the HandlerRegistry. This indicates a logic error in memory
 *     lifecycle management.
 *
 * PRECONDITIONS:
 *     - ptr is the actual unregistered pointer (for first constructor)
 *     - msg describes the context (for second constructor)
 *
 * POSTCONDITIONS:
 *     - code() returns UNREGISTERED_POINTER
 *     - what() includes pointer address in hex (first constructor)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * WHEN TO USE:
 *     - Accessing deallocated memory
 *     - Using pointer before registration
 *     - Double-free detection
 *     - Stale pointer access
 *
 * DEBUGGING:
 *     Indicates memory management bug that should be fixed.
 * -------------------------------------------------------------------------- */
class UnregisteredPointerError : public RuntimeError {
public:
    // Create with pointer address
    explicit UnregisteredPointerError(
        const void* ptr  // The unregistered pointer
    );

    // Create with custom message
    explicit UnregisteredPointerError(
        const std::string& msg  // Custom error description
    );

private:
    static std::string ptr_to_string(const void* ptr);
};

// =============================================================================
// HELPER MACROS
// =============================================================================

/* -----------------------------------------------------------------------------
 * MACRO: SCL_ASSERT
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assertion macro for internal invariant checking.
 *
 * SIGNATURE:
 *     SCL_ASSERT(condition, msg)
 *
 * PARAMETERS:
 *     condition [in] - Boolean expression to check
 *     msg       [in] - Error message if assertion fails
 *
 * BEHAVIOR:
 *     If condition evaluates to false, throws InternalError with:
 *         - The provided message
 *         - Source file name
 *         - Line number
 *
 * PRECONDITIONS:
 *     - msg must be convertible to std::string
 *
 * POSTCONDITIONS:
 *     If condition is false:
 *         - InternalError is thrown
 *         - Execution does not continue
 *     If condition is true:
 *         - No observable effect (no-op)
 *
 * PERFORMANCE:
 *     Active in all build configurations (Debug and Release).
 *     Use for critical safety checks only.
 *
 * THREAD SAFETY:
 *     Safe - evaluation is local to calling thread
 *
 * WHEN TO USE:
 *     - Verify internal data structure invariants
 *     - Check algorithm preconditions that should never fail
 *     - Detect impossible states
 *     - NOT for user input validation (use SCL_CHECK_ARG instead)
 *
 * DESIGN RATIONALE:
 *     Unlike standard assert(), this remains active in release builds
 *     because library correctness is paramount. It throws rather than
 *     aborting to allow graceful error handling across language boundaries.
 * -------------------------------------------------------------------------- */
#define SCL_ASSERT(condition, msg)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_CHECK_ARG
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Validation macro for user input arguments.
 *
 * SIGNATURE:
 *     SCL_CHECK_ARG(condition, msg)
 *
 * PARAMETERS:
 *     condition [in] - Boolean expression to validate
 *     msg       [in] - Error message describing validation failure
 *
 * BEHAVIOR:
 *     If condition evaluates to false, throws ValueError with the
 *     provided message.
 *
 * PRECONDITIONS:
 *     - msg must be convertible to std::string
 *
 * POSTCONDITIONS:
 *     If condition is false:
 *         - ValueError is thrown
 *         - Function does not proceed
 *     If condition is true:
 *         - No observable effect (no-op)
 *
 * PERFORMANCE:
 *     Active in all build configurations. Input validation is not optional.
 *
 * THREAD SAFETY:
 *     Safe - evaluation is local to calling thread
 *
 * WHEN TO USE:
 *     - Validate function parameters at API boundaries
 *     - Check for null pointers
 *     - Verify numeric ranges
 *     - Ensure non-empty containers
 *     - Validate enum values
 *
 * MESSAGE GUIDELINES:
 *     Be specific about what is invalid:
 *     GOOD: "n_neighbors must be positive, got -5"
 *     BAD:  "Invalid argument"
 * -------------------------------------------------------------------------- */
#define SCL_CHECK_ARG(condition, msg)

/* -----------------------------------------------------------------------------
 * MACRO: SCL_CHECK_DIM
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Validation macro for dimension compatibility.
 *
 * SIGNATURE:
 *     SCL_CHECK_DIM(condition, msg)
 *
 * PARAMETERS:
 *     condition [in] - Boolean expression checking dimension compatibility
 *     msg       [in] - Error message describing dimension mismatch
 *
 * BEHAVIOR:
 *     If condition evaluates to false, throws DimensionError with the
 *     provided message.
 *
 * PRECONDITIONS:
 *     - msg must be convertible to std::string
 *
 * POSTCONDITIONS:
 *     If condition is false:
 *         - DimensionError is thrown
 *         - Function does not proceed
 *     If condition is true:
 *         - No observable effect (no-op)
 *
 * PERFORMANCE:
 *     Active in all build configurations. Dimension validation prevents
 *     memory corruption and segmentation faults.
 *
 * THREAD SAFETY:
 *     Safe - evaluation is local to calling thread
 *
 * WHEN TO USE:
 *     - Matrix multiplication dimension checks
 *     - Output buffer size validation
 *     - Broadcasting compatibility
 *     - Axis bounds checking
 *
 * MESSAGE GUIDELINES:
 *     Include actual and expected dimensions:
 *     GOOD: "Output size mismatch: expected 100, got 50"
 *     BAD:  "Dimension error"
 * -------------------------------------------------------------------------- */
#define SCL_CHECK_DIM(condition, msg)

} // namespace scl
