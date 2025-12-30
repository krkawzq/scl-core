#pragma once

/// @file scl/core/error.hpp
/// @brief Comprehensive error handling and diagnostics system for SCL.
///
/// This header provides:
///   - Error code enumeration (C-ABI compatible)
///   - Exception type hierarchy with error codes
///   - Runtime assertion macros (SCL_CHECK, SCL_CHECK_ARG, etc.)
///   - Compile-time check macros (SCL_STATIC_CHECK, etc.)
///   - Debug-only assertion macros (SCL_DEBUG_ASSERT)
///   - Utilities for error code <-> exception conversion
///   - Thread-local error state support for C-ABI boundary
///
/// @note Requires scl/core/macro.hpp for source location support
///
/// ============================================================================
/// EXCEPTION HIERARCHY
/// ============================================================================
///
/// [Most frequently used: argument/index checks]
///   - ValueError             : Argument value error (base/general)
///   - IndexError             : Index out of bounds (index, size)
///   - RangeError             : Value exceeds valid range
///
/// [Common: dimension and shape checks]
///   - DimensionError         : Dimension-related error (base)
///   - ShapeMismatchError     : Tensor shape mismatch
///
/// [Common: memory-related]
///   - MemoryError            : Memory error (base)
///   - OutOfMemoryError       : Allocation failure (can pass size)
///   - NullPointerError       : Null pointer error (can pass pointer name)
///
/// [Common: type checks]
///   - TypeError              : Type error (base)
///   - DtypeMismatchError     : Dtype mismatch (expected, actual)
///
/// [Numeric computation errors]
///   - DivisionByZeroError    : Division by zero
///   - NaNError               : NaN encountered (may accept context)
///   - OverflowError          : Numeric overflow
///
/// [Computation/Algorithm errors]
///   - ComputationError           : Computation error (base)
///   - ConvergenceError           : Algorithm did not converge (algorithm, iterations)
///   - SingularMatrixError        : Singular matrix
///   - NumericalInstabilityError  : Numerical instability
///
/// [Not implemented & internal errors]
///   - NotImplementedError    : Functionality not implemented (feature name)
///   - InternalError          : Internal error (base)
///   - AssertionError         : Assertion failure (condition)
///
/// [I/O errors]
///   - IoError                : I/O error (base)
///   - FileNotFoundError      : File not found (path)
///
/// [Less commonly used]
///   - AlignmentError         : Misaligned memory (required, actual)
///   - BufferSizeError        : Buffer too small (required, actual)
///   - BroadcastError         : Broadcast failure
///   - InvalidAxisError       : Invalid axis (axis, ndim)
///   - UnsupportedTypeError   : Unsupported type (type name)
///   - ThreadingError         : Threading error (base)
///
/// ============================================================================
/// UTILITIES (functions & macros)
/// ============================================================================
///
/// [Error code utilities]
///   - error_code_name(code):        Convert error code to string name
///   - error_code_category(code):    Query code category (General/Memory/Dimension/etc)
///   - is_success(code):             constexpr: true if success
///   - is_error(code):               constexpr: true if error
///   - is_recoverable(code):         constexpr: true if recoverable
///
/// [Thread-local error state (for C-ABI)]
///   - ThreadErrorState:             Per-thread error state container
///   - get_thread_error():           Get thread-local error state reference
///   - clear_thread_error():         Clear thread-local error state
///   - set_thread_error(code, msg):  Set thread-local error info
///
/// [Exception conversion utilities]
///   - catch_to_error_code(func):    Wrap and map exception to error code
///   - catch_to_error_code_with_result<T>(func, default):
///                                   As above, but returns pair<T, ErrorCode>
///   - ErrorStateGuard:              RAII error state scope guard (reset on exit)
///
/// [Compile-time check macros]
///   - SCL_STATIC_CHECK(cond, msg):          Compile-time assertion
///   - SCL_STATIC_CHECK_FLOATING(T):         Check floating point type
///   - SCL_STATIC_CHECK_INTEGRAL(T):         Check integer type
///   - SCL_STATIC_CHECK_ARITHMETIC(T):       Check arithmetic type
///   - SCL_STATIC_CHECK_SIGNED(T):           Check signed type
///   - SCL_STATIC_CHECK_UNSIGNED(T):         Check unsigned type
///   - SCL_STATIC_CHECK_SIZE(T, min):        Check type size is at least min
///   - SCL_STATIC_CHECK_SAME(T, U):          Check types are the same
///
/// [Runtime check macros (throw exceptions)]
///   - SCL_CHECK(cond, ExType, ...):         General: throw ExType if condition fails
///   - SCL_CHECK_ARG(cond, ...):             Parameter check (throws ValueError)
///   - SCL_CHECK_DIM(cond, ...):             Dimension check (throws DimensionError)
///   - SCL_CHECK_RANGE(cond, ...):           Range check (throws RangeError)
///   - SCL_CHECK_MEM(cond, ...):             Memory check (throws MemoryError)
///   - SCL_CHECK_TYPE(cond, ...):            Type check (throws TypeError)
///   - SCL_CHECK_IO(cond, ...):              IO check (throws IoError)
///   - SCL_CHECK_COMPUTE(cond, ...):         Computation check (throws ComputationError)
///
/// [Specialized check macros]
///   - SCL_CHECK_NOT_NULL(ptr):              Throws NullPointerError if ptr is null
///   - SCL_CHECK_INDEX(index, size):         Throws IndexError on OOB
///   - SCL_CHECK_SIZE_MATCH(a, b):           Throws DimensionError for size mismatch
///   - SCL_CHECK_POSITIVE(val, name):        Throws ValueError if value is not > 0
///   - SCL_CHECK_NON_NEGATIVE(val, name):    Throws ValueError if value < 0
///   - SCL_CHECK_ALIGNMENT(ptr, align):      Throws AlignmentError if misaligned
///   - SCL_CHECK_FINITE(val):                Throws NaNError/ValueError if not finite
///
/// [Debug assertion macros (debug mode only)]
///   - SCL_DEBUG_ASSERT(cond):               Debug assertion (aborts on failure)
///   - SCL_DEBUG_ASSERT_MSG(cond, msg):      Debug assertion with message
///
/// [Exception throw macros]
///   - SCL_THROW(ExType, code, ...):         Throw specific exception and error code
///   - SCL_NOT_IMPLEMENTED(feature):         Throw NotImplementedError
///   - SCL_UNREACHABLE_CODE():               Throw UnreachableCode error
///
/// [C-ABI wrapping macros]
///   - SCL_C_API_BEGIN / SCL_C_API_END:           C-ABI wrapper for error-code functions
///   - SCL_C_API_BEGIN_VOID / SCL_C_API_END_VOID: C-ABI wrapper for void-returning functions
///
/// ============================================================================
/// ERROR CODE DESIGN
/// ============================================================================
///   - 0         : Success
///   - 1-99      : General errors
///   - 100-199   : Memory errors
///   - 200-299   : Dimension/shape errors
///   - 300-399   : Type/precision errors
///   - 400-499   : Value/argument errors
///   - 500-599   : I/O errors
///   - 600-699   : Algorithm/computation errors
///   - 700-799   : Threading/concurrency errors
///   - 800-899   : Hardware/platform errors
///   - 900-999   : Internal errors

#include "scl/core/macro.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <functional>
#include <optional>

// Feature test macros
#if __has_include(<version>)
    #include <version>
#endif

// C++20 std::format detection
#if __has_include(<format>) && defined(__cpp_lib_format)
    #include <format>
    #ifndef SCL_HAS_FORMAT
        #define SCL_HAS_FORMAT 1
    #endif
#else
    #ifndef SCL_HAS_FORMAT
        #define SCL_HAS_FORMAT 0
    #endif
#endif

// =============================================================================
// SECTION 1: Error Codes (C-ABI Compatible)
// =============================================================================

namespace scl {

/// @brief Error code enumeration for C-ABI compatibility
/// @note Values are stable and should not be changed once released
enum class ErrorCode : std::int32_t {
    // -------------------------------------------------------------------------
    // Success (0)
    // -------------------------------------------------------------------------
    Success = 0,
    
    // -------------------------------------------------------------------------
    // General Errors (1-99)
    // -------------------------------------------------------------------------
    Unknown = 1,                    ///< Unknown/unspecified error
    NotImplemented = 2,             ///< Feature not implemented
    Unsupported = 3,                ///< Operation not supported
    InvalidOperation = 4,           ///< Invalid operation in current state
    Cancelled = 5,                  ///< Operation was cancelled
    Timeout = 6,                    ///< Operation timed out
    Interrupted = 7,                ///< Operation was interrupted
    
    // -------------------------------------------------------------------------
    // Memory Errors (100-199)
    // -------------------------------------------------------------------------
    OutOfMemory = 100,              ///< Memory allocation failed
    AllocationFailed = 101,         ///< Specific allocation failed
    DeallocationFailed = 102,       ///< Deallocation failed
    NullPointer = 103,              ///< Null pointer dereference
    InvalidPointer = 104,           ///< Invalid pointer
    AlignmentError = 105,           ///< Memory alignment error
    BufferTooSmall = 106,           ///< Buffer size insufficient
    BufferOverflow = 107,           ///< Buffer overflow detected
    MemoryCorruption = 108,         ///< Memory corruption detected
    DoubleFree = 109,               ///< Double free detected
    UseAfterFree = 110,             ///< Use after free detected
    StackOverflow = 111,            ///< Stack overflow
    MmapFailed = 112,               ///< Memory mapping failed
    MunmapFailed = 113,             ///< Memory unmapping failed
    
    // -------------------------------------------------------------------------
    // Dimension/Shape Errors (200-299)
    // -------------------------------------------------------------------------
    DimensionMismatch = 200,        ///< Tensor dimensions don't match
    ShapeMismatch = 201,            ///< Tensor shapes don't match
    RankMismatch = 202,             ///< Tensor ranks don't match
    InvalidShape = 203,             ///< Invalid shape specification
    InvalidDimension = 204,         ///< Invalid dimension value
    EmptyTensor = 205,              ///< Tensor is empty
    NonContiguous = 206,            ///< Tensor is not contiguous
    BroadcastError = 207,           ///< Broadcasting failed
    StrideError = 208,              ///< Invalid stride configuration
    InvalidAxis = 209,              ///< Invalid axis specified
    AxisOutOfRange = 210,           ///< Axis out of valid range
    
    // -------------------------------------------------------------------------
    // Type/Precision Errors (300-399)
    // -------------------------------------------------------------------------
    TypeMismatch = 300,             ///< Type mismatch
    InvalidType = 301,              ///< Invalid type
    UnsupportedType = 302,          ///< Type not supported
    PrecisionLoss = 303,            ///< Precision loss detected
    InvalidPrecision = 304,         ///< Invalid precision specification
    CastError = 305,                ///< Type cast failed
    InvalidDtype = 306,             ///< Invalid data type
    DtypeMismatch = 307,            ///< Data type mismatch
    
    // -------------------------------------------------------------------------
    // Value/Argument Errors (400-499)
    // -------------------------------------------------------------------------
    InvalidArgument = 400,          ///< Invalid argument
    OutOfRange = 401,               ///< Value out of range
    IndexOutOfBounds = 402,         ///< Index out of bounds
    InvalidIndex = 403,             ///< Invalid index
    NegativeValue = 404,            ///< Unexpected negative value
    ZeroValue = 405,                ///< Unexpected zero value
    NaN = 406,                      ///< NaN encountered
    Infinity = 407,                 ///< Infinity encountered
    Underflow = 408,                ///< Numerical underflow
    Overflow = 409,                 ///< Numerical overflow
    DivisionByZero = 410,           ///< Division by zero
    InvalidRange = 411,             ///< Invalid range specification
    EmptyInput = 412,               ///< Empty input not allowed
    InvalidSize = 413,              ///< Invalid size
    SizeMismatch = 414,             ///< Size mismatch
    InvalidConfig = 415,            ///< Invalid configuration
    MissingArgument = 416,          ///< Required argument missing
    ExtraArgument = 417,            ///< Unexpected extra argument
    
    // -------------------------------------------------------------------------
    // I/O Errors (500-599)
    // -------------------------------------------------------------------------
    IoError = 500,                  ///< General I/O error
    FileNotFound = 501,             ///< File not found
    FileExists = 502,               ///< File already exists
    PermissionDenied = 503,         ///< Permission denied
    ReadError = 504,                ///< Read operation failed
    WriteError = 505,               ///< Write operation failed
    SeekError = 506,                ///< Seek operation failed
    EndOfFile = 507,                ///< Unexpected end of file
    InvalidPath = 508,              ///< Invalid file path
    DirectoryError = 509,           ///< Directory operation failed
    DiskFull = 510,                 ///< Disk is full
    InvalidFormat = 511,            ///< Invalid file format
    CorruptedFile = 512,            ///< File is corrupted
    
    // -------------------------------------------------------------------------
    // Algorithm/Computation Errors (600-699)
    // -------------------------------------------------------------------------
    ComputationError = 600,         ///< General computation error
    ConvergenceError = 601,         ///< Algorithm failed to converge
    SingularMatrix = 602,           ///< Matrix is singular
    NotPositiveDefinite = 603,      ///< Matrix not positive definite
    IllConditioned = 604,           ///< Matrix is ill-conditioned
    NumericalInstability = 605,     ///< Numerical instability detected
    MaxIterationsReached = 606,     ///< Maximum iterations exceeded
    InvalidAlgorithm = 607,         ///< Invalid algorithm choice
    KernelError = 608,              ///< Kernel execution error
    ReductionError = 609,           ///< Reduction operation error
    SortError = 610,                ///< Sorting operation error
    SearchError = 611,              ///< Search operation error
    FFTError = 612,                 ///< FFT operation error
    ConvolutionError = 613,         ///< Convolution operation error
    PoolingError = 614,             ///< Pooling operation error
    NormalizationError = 615,       ///< Normalization operation error
    ActivationError = 616,          ///< Activation function error
    LossError = 617,                ///< Loss computation error
    GradientError = 618,            ///< Gradient computation error
    
    // -------------------------------------------------------------------------
    // Threading/Concurrency Errors (700-799)
    // -------------------------------------------------------------------------
    ThreadError = 700,              ///< Thread operation error
    ThreadCreationFailed = 701,     ///< Thread creation failed
    ThreadJoinFailed = 702,         ///< Thread join failed
    MutexError = 703,               ///< Mutex operation error
    DeadlockDetected = 704,         ///< Potential deadlock detected
    RaceCondition = 705,            ///< Race condition detected
    SynchronizationError = 706,     ///< Synchronization error
    ThreadPoolError = 707,          ///< Thread pool error
    TaskError = 708,                ///< Task execution error
    FutureError = 709,              ///< Future/promise error
    AtomicError = 710,              ///< Atomic operation error
    
    // -------------------------------------------------------------------------
    // Hardware/Platform Errors (800-899)
    // -------------------------------------------------------------------------
    HardwareError = 800,            ///< Hardware error
    DeviceNotFound = 801,           ///< Device not found
    DeviceError = 802,              ///< Device error
    DriverError = 803,              ///< Driver error
    SimdNotSupported = 804,         ///< SIMD instruction not supported
    CpuFeatureNotSupported = 805,   ///< CPU feature not supported
    InstructionError = 806,         ///< Illegal instruction
    CacheError = 807,               ///< Cache-related error
    
    // -------------------------------------------------------------------------
    // Internal Errors (900-999)
    // -------------------------------------------------------------------------
    InternalError = 900,            ///< Internal error
    AssertionFailed = 901,          ///< Assertion failed
    InvariantViolation = 902,       ///< Invariant violation
    StateCorruption = 903,          ///< Internal state corruption
    UnreachableCode = 904,          ///< Unreachable code reached
    LogicError = 905,               ///< Logic error
    Uninitialized = 906,            ///< Uninitialized data access
    AlreadyInitialized = 907,       ///< Already initialized
    NotInitialized = 908,           ///< Not initialized
    InvalidState = 909,             ///< Invalid internal state
};

/// @brief Convert error code to string
/// @param code Error code
/// @return String representation of error code
[[nodiscard]] constexpr auto error_code_name(ErrorCode code) noexcept -> const char* {
    switch (code) {
        // Success
        case ErrorCode::Success: return "Success";
        
        // General
        case ErrorCode::Unknown: return "Unknown";
        case ErrorCode::NotImplemented: return "NotImplemented";
        case ErrorCode::Unsupported: return "Unsupported";
        case ErrorCode::InvalidOperation: return "InvalidOperation";
        case ErrorCode::Cancelled: return "Cancelled";
        case ErrorCode::Timeout: return "Timeout";
        case ErrorCode::Interrupted: return "Interrupted";
        
        // Memory
        case ErrorCode::OutOfMemory: return "OutOfMemory";
        case ErrorCode::AllocationFailed: return "AllocationFailed";
        case ErrorCode::DeallocationFailed: return "DeallocationFailed";
        case ErrorCode::NullPointer: return "NullPointer";
        case ErrorCode::InvalidPointer: return "InvalidPointer";
        case ErrorCode::AlignmentError: return "AlignmentError";
        case ErrorCode::BufferTooSmall: return "BufferTooSmall";
        case ErrorCode::BufferOverflow: return "BufferOverflow";
        case ErrorCode::MemoryCorruption: return "MemoryCorruption";
        case ErrorCode::DoubleFree: return "DoubleFree";
        case ErrorCode::UseAfterFree: return "UseAfterFree";
        case ErrorCode::StackOverflow: return "StackOverflow";
        case ErrorCode::MmapFailed: return "MmapFailed";
        case ErrorCode::MunmapFailed: return "MunmapFailed";
        
        // Dimension
        case ErrorCode::DimensionMismatch: return "DimensionMismatch";
        case ErrorCode::ShapeMismatch: return "ShapeMismatch";
        case ErrorCode::RankMismatch: return "RankMismatch";
        case ErrorCode::InvalidShape: return "InvalidShape";
        case ErrorCode::InvalidDimension: return "InvalidDimension";
        case ErrorCode::EmptyTensor: return "EmptyTensor";
        case ErrorCode::NonContiguous: return "NonContiguous";
        case ErrorCode::BroadcastError: return "BroadcastError";
        case ErrorCode::StrideError: return "StrideError";
        case ErrorCode::InvalidAxis: return "InvalidAxis";
        case ErrorCode::AxisOutOfRange: return "AxisOutOfRange";
        
        // Type
        case ErrorCode::TypeMismatch: return "TypeMismatch";
        case ErrorCode::InvalidType: return "InvalidType";
        case ErrorCode::UnsupportedType: return "UnsupportedType";
        case ErrorCode::PrecisionLoss: return "PrecisionLoss";
        case ErrorCode::InvalidPrecision: return "InvalidPrecision";
        case ErrorCode::CastError: return "CastError";
        case ErrorCode::InvalidDtype: return "InvalidDtype";
        case ErrorCode::DtypeMismatch: return "DtypeMismatch";
        
        // Value
        case ErrorCode::InvalidArgument: return "InvalidArgument";
        case ErrorCode::OutOfRange: return "OutOfRange";
        case ErrorCode::IndexOutOfBounds: return "IndexOutOfBounds";
        case ErrorCode::InvalidIndex: return "InvalidIndex";
        case ErrorCode::NegativeValue: return "NegativeValue";
        case ErrorCode::ZeroValue: return "ZeroValue";
        case ErrorCode::NaN: return "NaN";
        case ErrorCode::Infinity: return "Infinity";
        case ErrorCode::Underflow: return "Underflow";
        case ErrorCode::Overflow: return "Overflow";
        case ErrorCode::DivisionByZero: return "DivisionByZero";
        case ErrorCode::InvalidRange: return "InvalidRange";
        case ErrorCode::EmptyInput: return "EmptyInput";
        case ErrorCode::InvalidSize: return "InvalidSize";
        case ErrorCode::SizeMismatch: return "SizeMismatch";
        case ErrorCode::InvalidConfig: return "InvalidConfig";
        case ErrorCode::MissingArgument: return "MissingArgument";
        case ErrorCode::ExtraArgument: return "ExtraArgument";
        
        // I/O
        case ErrorCode::IoError: return "IoError";
        case ErrorCode::FileNotFound: return "FileNotFound";
        case ErrorCode::FileExists: return "FileExists";
        case ErrorCode::PermissionDenied: return "PermissionDenied";
        case ErrorCode::ReadError: return "ReadError";
        case ErrorCode::WriteError: return "WriteError";
        case ErrorCode::SeekError: return "SeekError";
        case ErrorCode::EndOfFile: return "EndOfFile";
        case ErrorCode::InvalidPath: return "InvalidPath";
        case ErrorCode::DirectoryError: return "DirectoryError";
        case ErrorCode::DiskFull: return "DiskFull";
        case ErrorCode::InvalidFormat: return "InvalidFormat";
        case ErrorCode::CorruptedFile: return "CorruptedFile";
        
        // Algorithm
        case ErrorCode::ComputationError: return "ComputationError";
        case ErrorCode::ConvergenceError: return "ConvergenceError";
        case ErrorCode::SingularMatrix: return "SingularMatrix";
        case ErrorCode::NotPositiveDefinite: return "NotPositiveDefinite";
        case ErrorCode::IllConditioned: return "IllConditioned";
        case ErrorCode::NumericalInstability: return "NumericalInstability";
        case ErrorCode::MaxIterationsReached: return "MaxIterationsReached";
        case ErrorCode::InvalidAlgorithm: return "InvalidAlgorithm";
        case ErrorCode::KernelError: return "KernelError";
        case ErrorCode::ReductionError: return "ReductionError";
        case ErrorCode::SortError: return "SortError";
        case ErrorCode::SearchError: return "SearchError";
        case ErrorCode::FFTError: return "FFTError";
        case ErrorCode::ConvolutionError: return "ConvolutionError";
        case ErrorCode::PoolingError: return "PoolingError";
        case ErrorCode::NormalizationError: return "NormalizationError";
        case ErrorCode::ActivationError: return "ActivationError";
        case ErrorCode::LossError: return "LossError";
        case ErrorCode::GradientError: return "GradientError";
        
        // Threading
        case ErrorCode::ThreadError: return "ThreadError";
        case ErrorCode::ThreadCreationFailed: return "ThreadCreationFailed";
        case ErrorCode::ThreadJoinFailed: return "ThreadJoinFailed";
        case ErrorCode::MutexError: return "MutexError";
        case ErrorCode::DeadlockDetected: return "DeadlockDetected";
        case ErrorCode::RaceCondition: return "RaceCondition";
        case ErrorCode::SynchronizationError: return "SynchronizationError";
        case ErrorCode::ThreadPoolError: return "ThreadPoolError";
        case ErrorCode::TaskError: return "TaskError";
        case ErrorCode::FutureError: return "FutureError";
        case ErrorCode::AtomicError: return "AtomicError";
        
        // Hardware
        case ErrorCode::HardwareError: return "HardwareError";
        case ErrorCode::DeviceNotFound: return "DeviceNotFound";
        case ErrorCode::DeviceError: return "DeviceError";
        case ErrorCode::DriverError: return "DriverError";
        case ErrorCode::SimdNotSupported: return "SimdNotSupported";
        case ErrorCode::CpuFeatureNotSupported: return "CpuFeatureNotSupported";
        case ErrorCode::InstructionError: return "InstructionError";
        case ErrorCode::CacheError: return "CacheError";
        
        // Internal
        case ErrorCode::InternalError: return "InternalError";
        case ErrorCode::AssertionFailed: return "AssertionFailed";
        case ErrorCode::InvariantViolation: return "InvariantViolation";
        case ErrorCode::StateCorruption: return "StateCorruption";
        case ErrorCode::UnreachableCode: return "UnreachableCode";
        case ErrorCode::LogicError: return "LogicError";
        case ErrorCode::Uninitialized: return "Uninitialized";
        case ErrorCode::AlreadyInitialized: return "AlreadyInitialized";
        case ErrorCode::NotInitialized: return "NotInitialized";
        case ErrorCode::InvalidState: return "InvalidState";
        
        default: return "Unknown";
    }
}

/// @brief Get error category from code
/// @param code Error code
/// @return Category name
[[nodiscard]] constexpr auto error_code_category(ErrorCode code) noexcept -> const char* {
    const auto val = static_cast<std::int32_t>(code);
    if (val == 0) return "Success";
    if (val < 100) return "General";
    if (val < 200) return "Memory";
    if (val < 300) return "Dimension";
    if (val < 400) return "Type";
    if (val < 500) return "Value";
    if (val < 600) return "IO";
    if (val < 700) return "Algorithm";
    if (val < 800) return "Threading";
    if (val < 900) return "Hardware";
    return "Internal";
}

/// @brief Check if error code represents success
[[nodiscard]] constexpr auto is_success(ErrorCode code) noexcept -> bool {
    return code == ErrorCode::Success;
}

/// @brief Check if error code represents failure
[[nodiscard]] constexpr auto is_error(ErrorCode code) noexcept -> bool {
    return code != ErrorCode::Success;
}

/// @brief Check if error is recoverable (not internal/hardware)
[[nodiscard]] constexpr auto is_recoverable(ErrorCode code) noexcept -> bool {
    const auto val = static_cast<std::int32_t>(code);
    return val > 0 && val < 800;
}

}  // namespace scl

// =============================================================================
// SECTION 2: Exception Hierarchy
// =============================================================================

namespace scl {

/// @brief Base class for all SCL exceptions
class Error : public std::exception {
protected:
    ErrorCode code_;
    std::string message_;
    source_location location_;
    std::string full_message_;  // Built in constructor for thread-safety

public:
    explicit Error(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : code_(code),
        message_(std::move(message)),
        location_(loc),
        full_message_(build_full_message()) {}  // Build immediately

    explicit Error(
        std::string message,
        source_location loc = source_location::current()
    ) : code_(ErrorCode::Unknown),
        message_(std::move(message)),
        location_(loc),
        full_message_(build_full_message()) {}  // Build immediately

    virtual ~Error() = default;

    [[nodiscard]] auto what() const noexcept -> const char* override {
        return full_message_.c_str();
    }

    [[nodiscard]] auto code() const noexcept -> ErrorCode { return code_; }
    [[nodiscard]] auto message() const noexcept -> const std::string& { return message_; }
    [[nodiscard]] auto location() const noexcept -> const source_location& { return location_; }

    [[nodiscard]] auto file() const noexcept -> const char* {
        return filename_only(location_.file_name());
    }
    [[nodiscard]] auto line() const noexcept -> std::uint32_t { return location_.line(); }
    [[nodiscard]] auto function() const noexcept -> const char* {
        return location_.function_name();
    }

protected:
    [[nodiscard]] virtual auto build_full_message() const -> std::string {
        char buffer[2048];
        std::snprintf(buffer, sizeof(buffer), "[%s] %s:%u in %s: %s",
                     error_code_name(code_),
                     file(), line(), function(),
                     message_.c_str());
        return buffer;
    }
};

// -----------------------------------------------------------------------------
// Memory Errors (100-199)
// -----------------------------------------------------------------------------

/// @brief Memory-related error base class
class MemoryError : public Error {
public:
    explicit MemoryError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit MemoryError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::OutOfMemory, std::move(message), loc) {}

    ~MemoryError() override = default;
};

/// @brief Out of memory error
class OutOfMemoryError : public MemoryError {
public:
    explicit OutOfMemoryError(
        std::string message = "Out of memory",
        source_location loc = source_location::current()
    ) : MemoryError(ErrorCode::OutOfMemory, std::move(message), loc) {}
    
    explicit OutOfMemoryError(
        std::size_t requested_size,
        source_location loc = source_location::current()
    ) : MemoryError(ErrorCode::OutOfMemory, 
                    "Failed to allocate " + std::to_string(requested_size) + " bytes", loc) {}
};

/// @brief Null pointer error
class NullPointerError : public MemoryError {
public:
    explicit NullPointerError(
        std::string name = "pointer",
        source_location loc = source_location::current()
    ) : MemoryError(ErrorCode::NullPointer, name + " is null", loc) {}
};

/// @brief Alignment error
class AlignmentError : public MemoryError {
public:
    explicit AlignmentError(
        std::size_t required_alignment,
        std::size_t misalignment_offset,
        source_location loc = source_location::current()
    ) : MemoryError(ErrorCode::AlignmentError,
                    "Alignment error: required " + std::to_string(required_alignment) +
                    "-byte alignment, offset from alignment: " + std::to_string(misalignment_offset), loc) {}
};

/// @brief Buffer size error
class BufferSizeError : public MemoryError {
public:
    explicit BufferSizeError(
        std::size_t required,
        std::size_t actual,
        source_location loc = source_location::current()
    ) : MemoryError(ErrorCode::BufferTooSmall,
                    "Buffer too small: required " + std::to_string(required) + 
                    ", got " + std::to_string(actual), loc) {}
};

// -----------------------------------------------------------------------------
// Dimension/Shape Errors (200-299)
// -----------------------------------------------------------------------------

/// @brief Dimension-related error base class
class DimensionError : public Error {
public:
    explicit DimensionError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit DimensionError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::DimensionMismatch, std::move(message), loc) {}

    ~DimensionError() override = default;
};

/// @brief Shape mismatch error
class ShapeMismatchError : public DimensionError {
public:
    explicit ShapeMismatchError(
        std::string message,
        source_location loc = source_location::current()
    ) : DimensionError(ErrorCode::ShapeMismatch, std::move(message), loc) {}
};

/// @brief Broadcast error
class BroadcastError : public DimensionError {
public:
    explicit BroadcastError(
        std::string message,
        source_location loc = source_location::current()
    ) : DimensionError(ErrorCode::BroadcastError, std::move(message), loc) {}
};

/// @brief Invalid axis error
class InvalidAxisError : public DimensionError {
public:
    explicit InvalidAxisError(
        std::int64_t axis,
        std::int64_t ndim,
        source_location loc = source_location::current()
    ) : DimensionError(ErrorCode::AxisOutOfRange,
                       "Axis " + std::to_string(axis) + " out of range for " + 
                       std::to_string(ndim) + "-dimensional tensor", loc) {}
};

// -----------------------------------------------------------------------------
// Type/Precision Errors (300-399)
// -----------------------------------------------------------------------------

/// @brief Type-related error base class
class TypeError : public Error {
public:
    explicit TypeError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit TypeError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::TypeMismatch, std::move(message), loc) {}

    ~TypeError() override = default;
};

/// @brief Unsupported type error
class UnsupportedTypeError : public TypeError {
public:
    explicit UnsupportedTypeError(
        std::string type_name,
        source_location loc = source_location::current()
    ) : TypeError(ErrorCode::UnsupportedType, 
                  "Unsupported type: " + type_name, loc) {}
};

/// @brief Data type mismatch error
class DtypeMismatchError : public TypeError {
public:
    explicit DtypeMismatchError(
        std::string expected,
        std::string actual,
        source_location loc = source_location::current()
    ) : TypeError(ErrorCode::DtypeMismatch,
                  "Dtype mismatch: expected " + expected + ", got " + actual, loc) {}
};

// -----------------------------------------------------------------------------
// Value/Argument Errors (400-499)
// -----------------------------------------------------------------------------

/// @brief Value-related error base class
class ValueError : public Error {
public:
    explicit ValueError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit ValueError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::InvalidArgument, std::move(message), loc) {}

    ~ValueError() override = default;
};

/// @brief Index out of bounds error
class IndexError : public ValueError {
public:
    explicit IndexError(
        std::int64_t index,
        std::int64_t size,
        source_location loc = source_location::current()
    ) : ValueError(ErrorCode::IndexOutOfBounds,
                   "Index " + std::to_string(index) + " out of bounds for size " + 
                   std::to_string(size), loc) {}
};

/// @brief Range error
class RangeError : public ValueError {
public:
    explicit RangeError(
        std::string message,
        source_location loc = source_location::current()
    ) : ValueError(ErrorCode::OutOfRange, std::move(message), loc) {}
};

/// @brief Division by zero error
class DivisionByZeroError : public ValueError {
public:
    explicit DivisionByZeroError(
        source_location loc = source_location::current()
    ) : ValueError(ErrorCode::DivisionByZero, "Division by zero", loc) {}
};

/// @brief NaN error
class NaNError : public ValueError {
public:
    explicit NaNError(
        std::string context = "",
        source_location loc = source_location::current()
    ) : ValueError(ErrorCode::NaN, 
                   context.empty() ? "NaN encountered" : "NaN encountered in " + context, loc) {}
};

/// @brief Overflow error
class OverflowError : public ValueError {
public:
    explicit OverflowError(
        std::string message = "Numerical overflow",
        source_location loc = source_location::current()
    ) : ValueError(ErrorCode::Overflow, std::move(message), loc) {}
};

// -----------------------------------------------------------------------------
// I/O Errors (500-599)
// -----------------------------------------------------------------------------

/// @brief I/O error base class
class IoError : public Error {
public:
    explicit IoError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit IoError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::IoError, std::move(message), loc) {}

    ~IoError() override = default;
};

/// @brief File not found error
class FileNotFoundError : public IoError {
public:
    explicit FileNotFoundError(
        std::string path,
        source_location loc = source_location::current()
    ) : IoError(ErrorCode::FileNotFound, "File not found: " + path, loc) {}
};

// -----------------------------------------------------------------------------
// Algorithm/Computation Errors (600-699)
// -----------------------------------------------------------------------------

/// @brief Computation error base class
class ComputationError : public Error {
public:
    explicit ComputationError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit ComputationError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::ComputationError, std::move(message), loc) {}

    ~ComputationError() override = default;
};

/// @brief Convergence error
class ConvergenceError : public ComputationError {
public:
    explicit ConvergenceError(
        std::string algorithm,
        std::size_t iterations,
        source_location loc = source_location::current()
    ) : ComputationError(ErrorCode::ConvergenceError,
                         algorithm + " failed to converge after " + 
                         std::to_string(iterations) + " iterations", loc) {}
};

/// @brief Singular matrix error
class SingularMatrixError : public ComputationError {
public:
    explicit SingularMatrixError(
        source_location loc = source_location::current()
    ) : ComputationError(ErrorCode::SingularMatrix, "Matrix is singular", loc) {}
};

/// @brief Numerical instability error
class NumericalInstabilityError : public ComputationError {
public:
    explicit NumericalInstabilityError(
        std::string message,
        source_location loc = source_location::current()
    ) : ComputationError(ErrorCode::NumericalInstability, std::move(message), loc) {}
};

// -----------------------------------------------------------------------------
// Threading Errors (700-799)
// -----------------------------------------------------------------------------

/// @brief Threading error base class
class ThreadingError : public Error {
public:
    explicit ThreadingError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit ThreadingError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::ThreadError, std::move(message), loc) {}

    ~ThreadingError() override = default;
};

// -----------------------------------------------------------------------------
// Internal Errors (900-999)
// -----------------------------------------------------------------------------

/// @brief Internal error base class
class InternalError : public Error {
public:
    explicit InternalError(
        ErrorCode code,
        std::string message,
        source_location loc = source_location::current()
    ) : Error(code, std::move(message), loc) {}

    explicit InternalError(
        std::string message,
        source_location loc = source_location::current()
    ) : Error(ErrorCode::InternalError, std::move(message), loc) {}

    ~InternalError() override = default;
};

/// @brief Not implemented error
class NotImplementedError : public InternalError {
public:
    explicit NotImplementedError(
        std::string feature = "",
        source_location loc = source_location::current()
    ) : InternalError(ErrorCode::NotImplemented,
                      feature.empty() ? "Not implemented" : feature + " is not implemented", loc) {}
};

/// @brief Assertion failed error
class AssertionError : public InternalError {
public:
    explicit AssertionError(
        std::string condition,
        source_location loc = source_location::current()
    ) : InternalError(ErrorCode::AssertionFailed, 
                      "Assertion failed: " + condition, loc) {}
};

}  // namespace scl

// =============================================================================
// SECTION 3: Error Formatting Utilities
// =============================================================================

namespace scl::detail {

#if SCL_HAS_FORMAT
/// @brief Format error message with source location (C++20 std::format version)
template<typename... Args>
[[nodiscard]] auto format_error(
    source_location loc, 
    std::format_string<Args...> fmt, 
    Args&&... args
) -> std::string {
    return std::format(fmt, std::forward<Args>(args)...);
}
#else
/// @brief Format error message (fallback version using snprintf)
template<typename... Args>
[[nodiscard]] auto format_error(
    source_location /*loc*/, 
    const char* fmt, 
    Args&&... args
) -> std::string {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    return buffer;
}
#endif

/// @brief Simple string format without source location
template<typename... Args>
[[nodiscard]] auto format_message(const char* fmt, Args&&... args) -> std::string {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
    return buffer;
}

/// @brief Debug assertion failure handler
[[noreturn]] inline void debug_assert_fail(
    const char* expr,
    source_location loc = source_location::current()
) {
    std::fprintf(stderr, 
        "SCL_DEBUG_ASSERT failed: %s\n  at %s:%u in %s\n",
        expr, 
        filename_only(loc.file_name()), 
        loc.line(), 
        loc.function_name());
    std::abort();
}

/// @brief Debug assertion failure with message
[[noreturn]] inline void debug_assert_fail_msg(
    const char* expr,
    const char* msg,
    source_location loc = source_location::current()
) {
    std::fprintf(stderr, 
        "SCL_DEBUG_ASSERT failed: %s\n  Message: %s\n  at %s:%u in %s\n",
        expr, msg,
        filename_only(loc.file_name()), 
        loc.line(), 
        loc.function_name());
    std::abort();
}

}  // namespace scl::detail

// =============================================================================
// SECTION 4: Thread-Local Error State (for C-ABI)
// =============================================================================

namespace scl {

/// @brief Thread-local error state for C-ABI functions
class ThreadErrorState {
    static constexpr std::size_t MAX_MESSAGE_LENGTH = 1024;
    
    ErrorCode code_ = ErrorCode::Success;
    char message_[MAX_MESSAGE_LENGTH] = {};
    
public:
    /// @brief Get thread-local error state instance
    [[nodiscard]] static auto instance() noexcept -> ThreadErrorState& {
        thread_local ThreadErrorState state;
        return state;
    }
    
    /// @brief Clear error state
    void clear() noexcept {
        code_ = ErrorCode::Success;
        message_[0] = '\0';
    }
    
    /// @brief Set error state
    void set(ErrorCode code, const char* message = nullptr) noexcept {
        code_ = code;
        if (message) {
            std::strncpy(message_, message, MAX_MESSAGE_LENGTH - 1);
            message_[MAX_MESSAGE_LENGTH - 1] = '\0';
        } else {
            std::strncpy(message_, error_code_name(code), MAX_MESSAGE_LENGTH - 1);
            message_[MAX_MESSAGE_LENGTH - 1] = '\0';
        }
    }
    
    /// @brief Set error from exception
    void set_from_exception(const Error& e) noexcept {
        set(e.code(), e.message().c_str());
    }
    
    /// @brief Set error from std::exception
    void set_from_std_exception(const std::exception& e) noexcept {
        set(ErrorCode::Unknown, e.what());
    }
    
    /// @brief Get error code
    [[nodiscard]] auto code() const noexcept -> ErrorCode { return code_; }
    
    /// @brief Get error message
    [[nodiscard]] auto message() const noexcept -> const char* { return message_; }
    
    /// @brief Check if error is set
    [[nodiscard]] auto has_error() const noexcept -> bool { 
        return code_ != ErrorCode::Success; 
    }
};

/// @brief Get thread-local error state
[[nodiscard]] inline auto get_thread_error() noexcept -> ThreadErrorState& {
    return ThreadErrorState::instance();
}

/// @brief Clear thread-local error state
inline void clear_thread_error() noexcept {
    ThreadErrorState::instance().clear();
}

/// @brief Set thread-local error state
inline void set_thread_error(ErrorCode code, const char* message = nullptr) noexcept {
    ThreadErrorState::instance().set(code, message);
}

}  // namespace scl

// =============================================================================
// SECTION 5: Exception to Error Code Conversion
// =============================================================================

namespace scl {

/// @brief Execute function and convert exceptions to error codes
/// @tparam F Callable type
/// @param func Function to execute
/// @return Error code (Success if no exception)
template<typename F>
[[nodiscard]] auto catch_to_error_code(F&& func) noexcept -> ErrorCode {
    try {
        func();
        return ErrorCode::Success;
    } catch (const Error& e) {
        set_thread_error(e.code(), e.message().c_str());
        return e.code();
    } catch (const std::bad_alloc&) {
        set_thread_error(ErrorCode::OutOfMemory, "Memory allocation failed");
        return ErrorCode::OutOfMemory;
    } catch (const std::out_of_range& e) {
        set_thread_error(ErrorCode::OutOfRange, e.what());
        return ErrorCode::OutOfRange;
    } catch (const std::invalid_argument& e) {
        set_thread_error(ErrorCode::InvalidArgument, e.what());
        return ErrorCode::InvalidArgument;
    } catch (const std::exception& e) {
        set_thread_error(ErrorCode::Unknown, e.what());
        return ErrorCode::Unknown;
    } catch (...) {
        set_thread_error(ErrorCode::Unknown, "Unknown exception");
        return ErrorCode::Unknown;
    }
}

/// @brief Execute function and convert exceptions to error codes (with return value)
/// @tparam T Return value type
/// @tparam F Callable type
/// @param func Function to execute
/// @param default_value Value to return on error
/// @return Pair of (result, error_code)
template<typename T, typename F>
[[nodiscard]] auto catch_to_error_code_with_result(
    F&& func,
    T default_value = T{}
) noexcept -> std::pair<T, ErrorCode> {
    try {
        return {func(), ErrorCode::Success};
    } catch (const Error& e) {
        set_thread_error(e.code(), e.message().c_str());
        return {default_value, e.code()};
    } catch (const std::bad_alloc&) {
        set_thread_error(ErrorCode::OutOfMemory, "Memory allocation failed");
        return {default_value, ErrorCode::OutOfMemory};
    } catch (const std::exception& e) {
        set_thread_error(ErrorCode::Unknown, e.what());
        return {default_value, ErrorCode::Unknown};
    } catch (...) {
        set_thread_error(ErrorCode::Unknown, "Unknown exception");
        return {default_value, ErrorCode::Unknown};
    }
}

/// @brief RAII guard for clearing error state
class ErrorStateGuard {
public:
    ErrorStateGuard() noexcept { clear_thread_error(); }
    ~ErrorStateGuard() = default;
    
    ErrorStateGuard(const ErrorStateGuard&) = delete;
    ErrorStateGuard& operator=(const ErrorStateGuard&) = delete;
};

}  // namespace scl

// =============================================================================
// SECTION 6: Check Macros
// =============================================================================

// -----------------------------------------------------------------------------
// SCL_STATIC_CHECK - Compile-time check
// -----------------------------------------------------------------------------

#define SCL_STATIC_CHECK(cond, msg) static_assert(cond, msg)

#define SCL_STATIC_CHECK_FLOATING(T) \
    SCL_STATIC_CHECK(std::is_floating_point_v<T>, "Type must be floating-point")

#define SCL_STATIC_CHECK_INTEGRAL(T) \
    SCL_STATIC_CHECK(std::is_integral_v<T>, "Type must be integral")

#define SCL_STATIC_CHECK_ARITHMETIC(T) \
    SCL_STATIC_CHECK(std::is_arithmetic_v<T>, "Type must be arithmetic")

#define SCL_STATIC_CHECK_SIGNED(T) \
    SCL_STATIC_CHECK(std::is_signed_v<T>, "Type must be signed")

#define SCL_STATIC_CHECK_UNSIGNED(T) \
    SCL_STATIC_CHECK(std::is_unsigned_v<T>, "Type must be unsigned")

#define SCL_STATIC_CHECK_SIZE(T, min_bytes) \
    SCL_STATIC_CHECK(sizeof(T) >= (min_bytes), "Type size too small")

#define SCL_STATIC_CHECK_SAME(T, U) \
    SCL_STATIC_CHECK((std::is_same_v<T, U>), "Types must be the same")

// -----------------------------------------------------------------------------
// SCL_CHECK - Runtime check (throws exception)
// -----------------------------------------------------------------------------

#define SCL_CHECK(cond, ExType, ...)                                           \
    do {                                                                       \
        if (!(cond)) [[unlikely]] {                                            \
            throw ::scl::ExType(                                               \
                ::scl::detail::format_error(                                   \
                    ::scl::source_location::current(), __VA_ARGS__),           \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

// Convenience wrappers
#define SCL_CHECK_ARG(cond, ...) \
    SCL_CHECK(cond, ValueError, __VA_ARGS__)

#define SCL_CHECK_DIM(cond, ...) \
    SCL_CHECK(cond, DimensionError, __VA_ARGS__)

#define SCL_CHECK_RANGE(cond, ...) \
    SCL_CHECK(cond, RangeError, __VA_ARGS__)

#define SCL_CHECK_MEM(cond, ...) \
    SCL_CHECK(cond, MemoryError, __VA_ARGS__)

#define SCL_CHECK_TYPE(cond, ...) \
    SCL_CHECK(cond, TypeError, __VA_ARGS__)

#define SCL_CHECK_IO(cond, ...) \
    SCL_CHECK(cond, IoError, __VA_ARGS__)

#define SCL_CHECK_COMPUTE(cond, ...) \
    SCL_CHECK(cond, ComputationError, __VA_ARGS__)

// Specific checks
#define SCL_CHECK_NOT_NULL(ptr) \
    do {                                                                       \
        if ((ptr) == nullptr) [[unlikely]] {                                   \
            throw ::scl::NullPointerError(#ptr,                                \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_INDEX(index, size) \
    do {                                                                       \
        const auto _idx = static_cast<std::int64_t>(index);                    \
        const auto _sz = static_cast<std::int64_t>(size);                      \
        if (_idx < 0 || _idx >= _sz) [[unlikely]] {                            \
            throw ::scl::IndexError(_idx, _sz,                                 \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_SIZE_MATCH(a, b) \
    do {                                                                       \
        if ((a) != (b)) [[unlikely]] {                                         \
            throw ::scl::DimensionError(::scl::ErrorCode::SizeMismatch,        \
                ::scl::detail::format_message("Size mismatch: %zu vs %zu",     \
                    static_cast<std::size_t>(a), static_cast<std::size_t>(b)), \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_POSITIVE(val, name) \
    do {                                                                       \
        if ((val) <= 0) [[unlikely]] {                                         \
            throw ::scl::ValueError(::scl::ErrorCode::InvalidArgument,         \
                std::string(name) + " must be positive",                       \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_NON_NEGATIVE(val, name) \
    do {                                                                       \
        if ((val) < 0) [[unlikely]] {                                          \
            throw ::scl::ValueError(::scl::ErrorCode::NegativeValue,           \
                std::string(name) + " must be non-negative",                   \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_ALIGNMENT(ptr, align) \
    do {                                                                       \
        static_assert(((align) & ((align) - 1)) == 0,                          \
                      "Alignment must be a power of 2");                       \
        auto _addr = reinterpret_cast<std::uintptr_t>(ptr);                    \
        if ((_addr & ((align) - 1)) != 0) [[unlikely]] {                       \
            throw ::scl::AlignmentError((align), _addr % (align),              \
                ::scl::source_location::current());                            \
        }                                                                      \
    } while (0)

#define SCL_CHECK_FINITE(val) \
    do {                                                                       \
        if (!std::isfinite(val)) [[unlikely]] {                                \
            if (std::isnan(val)) {                                             \
                throw ::scl::NaNError("",                                      \
                    ::scl::source_location::current());                        \
            } else {                                                           \
                throw ::scl::ValueError(::scl::ErrorCode::Infinity,            \
                    "Infinity encountered",                                    \
                    ::scl::source_location::current());                        \
            }                                                                  \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// SCL_DEBUG_ASSERT - Debug-only assertion
// -----------------------------------------------------------------------------

#ifdef NDEBUG
    #define SCL_DEBUG_ASSERT(cond) ((void)0)
    #define SCL_DEBUG_ASSERT_MSG(cond, msg) ((void)0)
#else
    #define SCL_DEBUG_ASSERT(cond)                                             \
        do {                                                                   \
            if (!(cond)) [[unlikely]] {                                        \
                ::scl::detail::debug_assert_fail(#cond,                        \
                    ::scl::source_location::current());                        \
            }                                                                  \
        } while (0)
    
    #define SCL_DEBUG_ASSERT_MSG(cond, msg)                                    \
        do {                                                                   \
            if (!(cond)) [[unlikely]] {                                        \
                ::scl::detail::debug_assert_fail_msg(#cond, msg,               \
                    ::scl::source_location::current());                        \
            }                                                                  \
        } while (0)
#endif

// -----------------------------------------------------------------------------
// SCL_THROW - Throw with error code
// -----------------------------------------------------------------------------

#define SCL_THROW(ExType, code, ...) \
    throw ::scl::ExType(::scl::ErrorCode::code, \
        ::scl::detail::format_error(::scl::source_location::current(), __VA_ARGS__), \
        ::scl::source_location::current())

#define SCL_NOT_IMPLEMENTED(feature) \
    throw ::scl::NotImplementedError(feature, ::scl::source_location::current())

#define SCL_UNREACHABLE_CODE() \
    do {                                                                       \
        throw ::scl::InternalError(::scl::ErrorCode::UnreachableCode,          \
            "Unreachable code reached",                                        \
            ::scl::source_location::current());                                \
    } while (0)

// =============================================================================
// SECTION 7: C-ABI Error Handling Macros
// =============================================================================

/// @brief Wrapper for C-ABI functions that returns error code
/// @note Use this macro to wrap C++ code in C-ABI functions
#define SCL_C_API_BEGIN \
    ::scl::clear_thread_error(); \
    try {

#define SCL_C_API_END \
        return static_cast<std::int32_t>(::scl::ErrorCode::Success); \
    } catch (const ::scl::Error& _e) { \
        ::scl::set_thread_error(_e.code(), _e.message().c_str()); \
        return static_cast<std::int32_t>(_e.code()); \
    } catch (const std::bad_alloc&) { \
        ::scl::set_thread_error(::scl::ErrorCode::OutOfMemory); \
        return static_cast<std::int32_t>(::scl::ErrorCode::OutOfMemory); \
    } catch (const std::exception& _e) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown, _e.what()); \
        return static_cast<std::int32_t>(::scl::ErrorCode::Unknown); \
    } catch (...) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown); \
        return static_cast<std::int32_t>(::scl::ErrorCode::Unknown); \
    }

/// @brief Wrapper for C-ABI functions that returns value through pointer
#define SCL_C_API_BEGIN_VOID \
    ::scl::clear_thread_error(); \
    try {

#define SCL_C_API_END_VOID \
    } catch (const ::scl::Error& _e) { \
        ::scl::set_thread_error(_e.code(), _e.message().c_str()); \
    } catch (const std::bad_alloc&) { \
        ::scl::set_thread_error(::scl::ErrorCode::OutOfMemory); \
    } catch (const std::exception& _e) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown, _e.what()); \
    } catch (...) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown); \
    }

/// @brief Wrapper for C-ABI functions that returns a handle (pointer)
/// @param null_value The null value to return on error (e.g., SCL_NULL_SPARSE)
#define SCL_C_API_END_HANDLE(null_value) \
    } catch (const ::scl::Error& _e) { \
        ::scl::set_thread_error(_e.code(), _e.message().c_str()); \
        return null_value; \
    } catch (const std::bad_alloc&) { \
        ::scl::set_thread_error(::scl::ErrorCode::OutOfMemory); \
        return null_value; \
    } catch (const std::exception& _e) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown, _e.what()); \
        return null_value; \
    } catch (...) { \
        ::scl::set_thread_error(::scl::ErrorCode::Unknown); \
        return null_value; \
    }
