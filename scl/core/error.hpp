#pragma once

/**
 * @file scl/core/error.hpp
 * @brief Comprehensive error handling and diagnostics system for SCL.
 *
 * This header provides:
 *   - Error code enumeration
 *   - Exception type hierarchy with error codes
 *   - Thread-local error state management (singleton pattern)
 *   - Runtime assertion macros (SCL_CHECK, SCL_CHECK_ARG, etc.)
 *   - Compile-time check macros (SCL_STATIC_CHECK, etc.)
 *   - Debug-only assertion macros (SCL_DEBUG_ASSERT)
 *
 * @note Requires scl/core/macro.hpp for source location support
 *
 * ============================================================================
 * EXCEPTION HIERARCHY
 * ============================================================================
 *
 * [Most frequently used: argument/index checks]
 *   - ValueError             : Argument value error (base/general)
 *   - IndexError             : Index out of bounds (index, size)
 *   - RangeError             : Value exceeds valid range
 *
 * [Common: dimension and shape checks]
 *   - DimensionError         : Dimension-related error (base)
 *   - ShapeMismatchError     : Tensor shape mismatch
 *
 * [Common: memory-related]
 *   - MemoryError            : Memory error (base)
 *   - OutOfMemoryError       : Allocation failure (can pass size)
 *   - NullPointerError       : Null pointer error (can pass pointer name)
 *
 * [Common: type checks]
 *   - TypeError              : Type error (base)
 *   - DtypeMismatchError     : Dtype mismatch (expected, actual)
 *
 * [Numeric computation errors]
 *   - DivisionByZeroError    : Division by zero
 *   - NaNError               : NaN encountered (may accept context)
 *   - OverflowError          : Numeric overflow
 *
 * [Computation/Algorithm errors]
 *   - ComputationError           : Computation error (base)
 *   - ConvergenceError           : Algorithm did not converge (algorithm, iterations)
 *   - SingularMatrixError        : Singular matrix
 *   - NumericalInstabilityError  : Numerical instability
 *
 * [Not implemented & internal errors]
 *   - NotImplementedError    : Functionality not implemented (feature name)
 *   - InternalError          : Internal error (base)
 *   - AssertionError         : Assertion failure (condition)
 *
 * [I/O errors]
 *   - IoError                : I/O error (base)
 *   - FileNotFoundError      : File not found (path)
 *
 * [Less commonly used]
 *   - AlignmentError         : Misaligned memory (required, actual)
 *   - BufferSizeError        : Buffer too small (required, actual)
 *   - BroadcastError         : Broadcast failure
 *   - InvalidAxisError       : Invalid axis (axis, ndim)
 *   - UnsupportedTypeError   : Unsupported type (type name)
 *   - ThreadingError         : Threading error (base)
 *
 * ============================================================================
 * UTILITIES (functions & macros)
 * ============================================================================
 *
 * [Error code utilities]
 *   - error_code_name(code):        Convert error code to string name
 *   - error_code_category(code):    Query code category (General/Memory/Dimension/etc)
 *   - is_success(code):             constexpr: true if success
 *   - is_error(code):               constexpr: true if error
 *   - is_recoverable(code):         constexpr: true if recoverable
 *
 * [Thread-local error state utilities]
 *   - ThreadErrorState::instance(): Get thread-local singleton instance
 *   - get_thread_error():           Get thread-local error state reference
 *   - clear_thread_error():         Clear thread-local error state
 *   - set_thread_error():           Set error code and message
 *   - set_thread_error_from_exception(): Set error from exception
 *   - thread_has_error():           Check if thread has error
 *   - get_thread_error_code():      Get current thread error code
 *   - get_thread_error_message():   Get current thread error message
 *
 * [Compile-time check macros]
 *   - SCL_STATIC_CHECK(cond, msg):          Compile-time assertion
 *   - SCL_STATIC_CHECK_FLOATING(T):         Check floating point type
 *   - SCL_STATIC_CHECK_INTEGRAL(T):         Check integer type
 *   - SCL_STATIC_CHECK_ARITHMETIC(T):       Check arithmetic type
 *   - SCL_STATIC_CHECK_SIGNED(T):           Check signed type
 *   - SCL_STATIC_CHECK_UNSIGNED(T):         Check unsigned type
 *   - SCL_STATIC_CHECK_SIZE(T, min):        Check type size is at least min
 *   - SCL_STATIC_CHECK_SAME(T, U):          Check types are the same
 *
 * [Runtime check macros (throw exceptions)]
 *   - SCL_CHECK(cond, ExType, ...):         General: throw ExType if condition fails
 *   - SCL_CHECK_ARG(cond, ...):             Parameter check (throws ValueError)
 *   - SCL_CHECK_DIM(cond, ...):             Dimension check (throws DimensionError)
 *   - SCL_CHECK_RANGE(cond, ...):           Range check (throws RangeError)
 *   - SCL_CHECK_MEM(cond, ...):             Memory check (throws MemoryError)
 *   - SCL_CHECK_TYPE(cond, ...):            Type check (throws TypeError)
 *   - SCL_CHECK_IO(cond, ...):              IO check (throws IoError)
 *   - SCL_CHECK_COMPUTE(cond, ...):         Computation check (throws ComputationError)
 *
 * [Specialized check macros]
 *   - SCL_CHECK_NOT_NULL(ptr):              Throws NullPointerError if ptr is null
 *   - SCL_CHECK_INDEX(index, size):         Throws IndexError on OOB
 *   - SCL_CHECK_SIZE_MATCH(a, b):           Throws DimensionError for size mismatch
 *   - SCL_CHECK_POSITIVE(val, name):        Throws ValueError if value is not > 0
 *   - SCL_CHECK_NON_NEGATIVE(val, name):    Throws ValueError if value < 0
 *   - SCL_CHECK_ALIGNMENT(ptr, align):      Throws AlignmentError if misaligned
 *   - SCL_CHECK_FINITE(val):                Throws NaNError/ValueError if not finite
 *
 * [Debug assertion macros (debug mode only)]
 *   - SCL_DEBUG_ASSERT(cond):               Debug assertion (aborts on failure)
 *   - SCL_DEBUG_ASSERT_MSG(cond, msg):      Debug assertion with message
 *
 * [Exception throw macros]
 *   - SCL_THROW(ExType, code, ...):         Throw specific exception and error code
 *   - SCL_NOT_IMPLEMENTED(feature):         Throw NotImplementedError
 *   - SCL_UNREACHABLE_CODE():               Throw UnreachableCode error
 *
 * ============================================================================
 * ERROR CODE DESIGN
 * ============================================================================
 *   - 0         : Success
 *   - 1-99      : General errors
 *   - 100-199   : Memory errors
 *   - 200-299   : Dimension/shape errors
 *   - 300-399   : Type/precision errors
 *   - 400-499   : Value/argument errors
 *   - 500-599   : I/O errors
 *   - 600-699   : Algorithm/computation errors
 *   - 700-799   : Threading/concurrency errors
 *   - 800-899   : Hardware/platform errors
 *   - 900-999   : Internal errors
 */

#include "scl/core/source_location.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

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
  Unknown = 1,             ///< Unknown/unspecified error
  NotImplemented = 2,      ///< Feature not implemented
  Unsupported = 3,         ///< Operation not supported
  InvalidOperation = 4,    ///< Invalid operation in current state
  Cancelled = 5,           ///< Operation was cancelled
  Timeout = 6,             ///< Operation timed out
  Interrupted = 7,         ///< Operation was interrupted

  // -------------------------------------------------------------------------
  // Memory Errors (100-199)
  // -------------------------------------------------------------------------
  OutOfMemory = 100,       ///< Memory allocation failed
  AllocationFailed = 101,  ///< Specific allocation failed
  DeallocationFailed = 102,  ///< Deallocation failed
  NullPointer = 103,       ///< Null pointer dereference
  InvalidPointer = 104,    ///< Invalid pointer
  AlignmentError = 105,    ///< Memory alignment error
  BufferTooSmall = 106,    ///< Buffer size insufficient
  BufferOverflow = 107,    ///< Buffer overflow detected
  MemoryCorruption = 108,  ///< Memory corruption detected
  DoubleFree = 109,        ///< Double free detected
  UseAfterFree = 110,      ///< Use after free detected
  StackOverflow = 111,     ///< Stack overflow
  MmapFailed = 112,        ///< Memory mapping failed
  MunmapFailed = 113,      ///< Memory unmapping failed

  // -------------------------------------------------------------------------
  // Dimension/Shape Errors (200-299)
  // -------------------------------------------------------------------------
  DimensionMismatch = 200,  ///< Tensor dimensions don't match
  ShapeMismatch = 201,      ///< Tensor shapes don't match
  RankMismatch = 202,       ///< Tensor ranks don't match
  InvalidShape = 203,       ///< Invalid shape specification
  InvalidDimension = 204,   ///< Invalid dimension value
  EmptyTensor = 205,        ///< Tensor is empty
  NonContiguous = 206,      ///< Tensor is not contiguous
  BroadcastError = 207,     ///< Broadcasting failed
  StrideError = 208,        ///< Invalid stride configuration
  InvalidAxis = 209,        ///< Invalid axis specified
  AxisOutOfRange = 210,     ///< Axis out of valid range

  // -------------------------------------------------------------------------
  // Type/Precision Errors (300-399)
  // -------------------------------------------------------------------------
  TypeMismatch = 300,      ///< Type mismatch
  InvalidType = 301,       ///< Invalid type
  UnsupportedType = 302,   ///< Type not supported
  PrecisionLoss = 303,     ///< Precision loss detected
  InvalidPrecision = 304,  ///< Invalid precision specification
  CastError = 305,         ///< Type cast failed
  InvalidDtype = 306,      ///< Invalid data type
  DtypeMismatch = 307,     ///< Data type mismatch

  // -------------------------------------------------------------------------
  // Value/Argument Errors (400-499)
  // -------------------------------------------------------------------------
  InvalidArgument = 400,    ///< Invalid argument
  OutOfRange = 401,         ///< Value out of range
  IndexOutOfBounds = 402,   ///< Index out of bounds
  InvalidIndex = 403,       ///< Invalid index
  NegativeValue = 404,      ///< Unexpected negative value
  ZeroValue = 405,          ///< Unexpected zero value
  NaN = 406,                ///< NaN encountered
  Infinity = 407,           ///< Infinity encountered
  Underflow = 408,          ///< Numerical underflow
  Overflow = 409,           ///< Numerical overflow
  DivisionByZero = 410,     ///< Division by zero
  InvalidRange = 411,       ///< Invalid range specification
  EmptyInput = 412,         ///< Empty input not allowed
  InvalidSize = 413,        ///< Invalid size
  SizeMismatch = 414,       ///< Size mismatch
  InvalidConfig = 415,      ///< Invalid configuration
  MissingArgument = 416,    ///< Required argument missing
  ExtraArgument = 417,      ///< Unexpected extra argument

  // -------------------------------------------------------------------------
  // I/O Errors (500-599)
  // -------------------------------------------------------------------------
  IoError = 500,          ///< General I/O error
  FileNotFound = 501,     ///< File not found
  FileExists = 502,       ///< File already exists
  PermissionDenied = 503, ///< Permission denied
  ReadError = 504,        ///< Read operation failed
  WriteError = 505,       ///< Write operation failed
  SeekError = 506,        ///< Seek operation failed
  EndOfFile = 507,        ///< Unexpected end of file
  InvalidPath = 508,      ///< Invalid file path
  DirectoryError = 509,   ///< Directory operation failed
  DiskFull = 510,         ///< Disk is full
  InvalidFormat = 511,    ///< Invalid file format
  CorruptedFile = 512,    ///< File is corrupted

  // -------------------------------------------------------------------------
  // Algorithm/Computation Errors (600-699)
  // -------------------------------------------------------------------------
  ComputationError = 600,       ///< General computation error
  ConvergenceError = 601,       ///< Algorithm failed to converge
  SingularMatrix = 602,         ///< Matrix is singular
  NotPositiveDefinite = 603,    ///< Matrix not positive definite
  IllConditioned = 604,         ///< Matrix is ill-conditioned
  NumericalInstability = 605,   ///< Numerical instability detected
  MaxIterationsReached = 606,   ///< Maximum iterations exceeded
  InvalidAlgorithm = 607,       ///< Invalid algorithm choice
  KernelError = 608,            ///< Kernel execution error
  ReductionError = 609,         ///< Reduction operation error
  SortError = 610,              ///< Sorting operation error
  SearchError = 611,            ///< Search operation error
  FFTError = 612,               ///< FFT operation error
  ConvolutionError = 613,       ///< Convolution operation error
  PoolingError = 614,           ///< Pooling operation error
  NormalizationError = 615,     ///< Normalization operation error
  ActivationError = 616,        ///< Activation function error
  LossError = 617,              ///< Loss computation error
  GradientError = 618,          ///< Gradient computation error

  // -------------------------------------------------------------------------
  // Threading/Concurrency Errors (700-799)
  // -------------------------------------------------------------------------
  ThreadError = 700,            ///< Thread operation error
  ThreadCreationFailed = 701,   ///< Thread creation failed
  ThreadJoinFailed = 702,       ///< Thread join failed
  MutexError = 703,             ///< Mutex operation error
  DeadlockDetected = 704,       ///< Potential deadlock detected
  RaceCondition = 705,          ///< Race condition detected
  SynchronizationError = 706,   ///< Synchronization error
  ThreadPoolError = 707,        ///< Thread pool error
  TaskError = 708,              ///< Task execution error
  FutureError = 709,            ///< Future/promise error
  AtomicError = 710,            ///< Atomic operation error

  // -------------------------------------------------------------------------
  // Hardware/Platform Errors (800-899)
  // -------------------------------------------------------------------------
  HardwareError = 800,           ///< Hardware error
  DeviceNotFound = 801,          ///< Device not found
  DeviceError = 802,             ///< Device error
  DriverError = 803,             ///< Driver error
  SimdNotSupported = 804,        ///< SIMD instruction not supported
  CpuFeatureNotSupported = 805,  ///< CPU feature not supported
  InstructionError = 806,        ///< Illegal instruction
  CacheError = 807,              ///< Cache-related error

  // -------------------------------------------------------------------------
  // Internal Errors (900-999)
  // -------------------------------------------------------------------------
  InternalError = 900,        ///< Internal error
  AssertionFailed = 901,      ///< Assertion failed
  InvariantViolation = 902,   ///< Invariant violation
  StateCorruption = 903,      ///< Internal state corruption
  UnreachableCode = 904,      ///< Unreachable code reached
  LogicError = 905,           ///< Logic error
  Uninitialized = 906,        ///< Uninitialized data access
  AlreadyInitialized = 907,   ///< Already initialized
  NotInitialized = 908,       ///< Not initialized
  InvalidState = 909,         ///< Invalid internal state
};

/// @brief Convert error code to string
/// @param code Error code
/// @return String representation of error code
[[nodiscard]]
constexpr
auto error_code_name(ErrorCode code) noexcept -> const char* {
  switch (code) {
    // Success
    case ErrorCode::Success:
      return "Success";

    // General
    case ErrorCode::Unknown:
      return "Unknown";
    case ErrorCode::NotImplemented:
      return "NotImplemented";
    case ErrorCode::Unsupported:
      return "Unsupported";
    case ErrorCode::InvalidOperation:
      return "InvalidOperation";
    case ErrorCode::Cancelled:
      return "Cancelled";
    case ErrorCode::Timeout:
      return "Timeout";
    case ErrorCode::Interrupted:
      return "Interrupted";

    // Memory
    case ErrorCode::OutOfMemory:
      return "OutOfMemory";
    case ErrorCode::AllocationFailed:
      return "AllocationFailed";
    case ErrorCode::DeallocationFailed:
      return "DeallocationFailed";
    case ErrorCode::NullPointer:
      return "NullPointer";
    case ErrorCode::InvalidPointer:
      return "InvalidPointer";
    case ErrorCode::AlignmentError:
      return "AlignmentError";
    case ErrorCode::BufferTooSmall:
      return "BufferTooSmall";
    case ErrorCode::BufferOverflow:
      return "BufferOverflow";
    case ErrorCode::MemoryCorruption:
      return "MemoryCorruption";
    case ErrorCode::DoubleFree:
      return "DoubleFree";
    case ErrorCode::UseAfterFree:
      return "UseAfterFree";
    case ErrorCode::StackOverflow:
      return "StackOverflow";
    case ErrorCode::MmapFailed:
      return "MmapFailed";
    case ErrorCode::MunmapFailed:
      return "MunmapFailed";

    // Dimension
    case ErrorCode::DimensionMismatch:
      return "DimensionMismatch";
    case ErrorCode::ShapeMismatch:
      return "ShapeMismatch";
    case ErrorCode::RankMismatch:
      return "RankMismatch";
    case ErrorCode::InvalidShape:
      return "InvalidShape";
    case ErrorCode::InvalidDimension:
      return "InvalidDimension";
    case ErrorCode::EmptyTensor:
      return "EmptyTensor";
    case ErrorCode::NonContiguous:
      return "NonContiguous";
    case ErrorCode::BroadcastError:
      return "BroadcastError";
    case ErrorCode::StrideError:
      return "StrideError";
    case ErrorCode::InvalidAxis:
      return "InvalidAxis";
    case ErrorCode::AxisOutOfRange:
      return "AxisOutOfRange";

    // Type
    case ErrorCode::TypeMismatch:
      return "TypeMismatch";
    case ErrorCode::InvalidType:
      return "InvalidType";
    case ErrorCode::UnsupportedType:
      return "UnsupportedType";
    case ErrorCode::PrecisionLoss:
      return "PrecisionLoss";
    case ErrorCode::InvalidPrecision:
      return "InvalidPrecision";
    case ErrorCode::CastError:
      return "CastError";
    case ErrorCode::InvalidDtype:
      return "InvalidDtype";
    case ErrorCode::DtypeMismatch:
      return "DtypeMismatch";

    // Value
    case ErrorCode::InvalidArgument:
      return "InvalidArgument";
    case ErrorCode::OutOfRange:
      return "OutOfRange";
    case ErrorCode::IndexOutOfBounds:
      return "IndexOutOfBounds";
    case ErrorCode::InvalidIndex:
      return "InvalidIndex";
    case ErrorCode::NegativeValue:
      return "NegativeValue";
    case ErrorCode::ZeroValue:
      return "ZeroValue";
    case ErrorCode::NaN:
      return "NaN";
    case ErrorCode::Infinity:
      return "Infinity";
    case ErrorCode::Underflow:
      return "Underflow";
    case ErrorCode::Overflow:
      return "Overflow";
    case ErrorCode::DivisionByZero:
      return "DivisionByZero";
    case ErrorCode::InvalidRange:
      return "InvalidRange";
    case ErrorCode::EmptyInput:
      return "EmptyInput";
    case ErrorCode::InvalidSize:
      return "InvalidSize";
    case ErrorCode::SizeMismatch:
      return "SizeMismatch";
    case ErrorCode::InvalidConfig:
      return "InvalidConfig";
    case ErrorCode::MissingArgument:
      return "MissingArgument";
    case ErrorCode::ExtraArgument:
      return "ExtraArgument";

    // I/O
    case ErrorCode::IoError:
      return "IoError";
    case ErrorCode::FileNotFound:
      return "FileNotFound";
    case ErrorCode::FileExists:
      return "FileExists";
    case ErrorCode::PermissionDenied:
      return "PermissionDenied";
    case ErrorCode::ReadError:
      return "ReadError";
    case ErrorCode::WriteError:
      return "WriteError";
    case ErrorCode::SeekError:
      return "SeekError";
    case ErrorCode::EndOfFile:
      return "EndOfFile";
    case ErrorCode::InvalidPath:
      return "InvalidPath";
    case ErrorCode::DirectoryError:
      return "DirectoryError";
    case ErrorCode::DiskFull:
      return "DiskFull";
    case ErrorCode::InvalidFormat:
      return "InvalidFormat";
    case ErrorCode::CorruptedFile:
      return "CorruptedFile";

    // Algorithm
    case ErrorCode::ComputationError:
      return "ComputationError";
    case ErrorCode::ConvergenceError:
      return "ConvergenceError";
    case ErrorCode::SingularMatrix:
      return "SingularMatrix";
    case ErrorCode::NotPositiveDefinite:
      return "NotPositiveDefinite";
    case ErrorCode::IllConditioned:
      return "IllConditioned";
    case ErrorCode::NumericalInstability:
      return "NumericalInstability";
    case ErrorCode::MaxIterationsReached:
      return "MaxIterationsReached";
    case ErrorCode::InvalidAlgorithm:
      return "InvalidAlgorithm";
    case ErrorCode::KernelError:
      return "KernelError";
    case ErrorCode::ReductionError:
      return "ReductionError";
    case ErrorCode::SortError:
      return "SortError";
    case ErrorCode::SearchError:
      return "SearchError";
    case ErrorCode::FFTError:
      return "FFTError";
    case ErrorCode::ConvolutionError:
      return "ConvolutionError";
    case ErrorCode::PoolingError:
      return "PoolingError";
    case ErrorCode::NormalizationError:
      return "NormalizationError";
    case ErrorCode::ActivationError:
      return "ActivationError";
    case ErrorCode::LossError:
      return "LossError";
    case ErrorCode::GradientError:
      return "GradientError";

    // Threading
    case ErrorCode::ThreadError:
      return "ThreadError";
    case ErrorCode::ThreadCreationFailed:
      return "ThreadCreationFailed";
    case ErrorCode::ThreadJoinFailed:
      return "ThreadJoinFailed";
    case ErrorCode::MutexError:
      return "MutexError";
    case ErrorCode::DeadlockDetected:
      return "DeadlockDetected";
    case ErrorCode::RaceCondition:
      return "RaceCondition";
    case ErrorCode::SynchronizationError:
      return "SynchronizationError";
    case ErrorCode::ThreadPoolError:
      return "ThreadPoolError";
    case ErrorCode::TaskError:
      return "TaskError";
    case ErrorCode::FutureError:
      return "FutureError";
    case ErrorCode::AtomicError:
      return "AtomicError";

    // Hardware
    case ErrorCode::HardwareError:
      return "HardwareError";
    case ErrorCode::DeviceNotFound:
      return "DeviceNotFound";
    case ErrorCode::DeviceError:
      return "DeviceError";
    case ErrorCode::DriverError:
      return "DriverError";
    case ErrorCode::SimdNotSupported:
      return "SimdNotSupported";
    case ErrorCode::CpuFeatureNotSupported:
      return "CpuFeatureNotSupported";
    case ErrorCode::InstructionError:
      return "InstructionError";
    case ErrorCode::CacheError:
      return "CacheError";

    // Internal
    case ErrorCode::InternalError:
      return "InternalError";
    case ErrorCode::AssertionFailed:
      return "AssertionFailed";
    case ErrorCode::InvariantViolation:
      return "InvariantViolation";
    case ErrorCode::StateCorruption:
      return "StateCorruption";
    case ErrorCode::UnreachableCode:
      return "UnreachableCode";
    case ErrorCode::LogicError:
      return "LogicError";
    case ErrorCode::Uninitialized:
      return "Uninitialized";
    case ErrorCode::AlreadyInitialized:
      return "AlreadyInitialized";
    case ErrorCode::NotInitialized:
      return "NotInitialized";
    case ErrorCode::InvalidState:
      return "InvalidState";

    default:
      return "Unknown";
  }
}

/// @brief Get error category from code
/// @param code Error code
/// @return Category name
[[nodiscard]]
constexpr
auto error_code_category(ErrorCode code) noexcept -> const char* {
  const auto val = static_cast<std::int32_t>(code);
  if (val == 0) {
    return "Success";
  }
  if (val < 100) { // NOLINT(readability-magic-numbers)
    return "General";
  }
  if (val < 200) { // NOLINT(readability-magic-numbers)
    return "Memory";
  }
  if (val < 300) { // NOLINT(readability-magic-numbers)
    return "Dimension";
  }
  if (val < 400) { // NOLINT(readability-magic-numbers)
    return "Type";
  }
  if (val < 500) { // NOLINT(readability-magic-numbers)
    return "Value";
  }
  if (val < 600) { // NOLINT(readability-magic-numbers)
    return "IO";
  }
  if (val < 700) { // NOLINT(readability-magic-numbers)
    return "Algorithm";
  }
  if (val < 800) { // NOLINT(readability-magic-numbers)
    return "Threading";
  }
  if (val < 900) { // NOLINT(readability-magic-numbers)
    return "Hardware";
  }
  return "Internal";
}

/// @brief Check if error code represents success
[[nodiscard]]
constexpr
auto is_success(ErrorCode code) noexcept -> bool {
  return code == ErrorCode::Success;
}

/// @brief Check if error code represents failure
[[nodiscard]]
constexpr
auto is_error(ErrorCode code) noexcept -> bool {
  return code != ErrorCode::Success;
}

/// @brief Check if error is recoverable (not internal/hardware)
[[nodiscard]]
constexpr
auto is_recoverable(ErrorCode code) noexcept -> bool {
  const auto val = static_cast<std::int32_t>(code);
  return val > 0 && val < 800; // NOLINT(readability-magic-numbers)
}

}  // namespace scl

// =============================================================================
// SECTION 2: Exception Hierarchy
// =============================================================================

namespace scl {

/// @brief Base class for all SCL exceptions
class Error : public std::exception { // NOLINT(readability-magic-numbers)
 protected:
  ErrorCode code_;
  std::string message_;
  source_location location_;
  std::string full_message_;  // Built in constructor for thread-safety

 public:
  explicit Error(ErrorCode code, std::string message,
                 source_location loc = source_location::current())
      : code_(code),
        message_(std::move(message)),
        location_(loc),
        full_message_(build_full_message()) {}  // Build immediately

  explicit Error(std::string message,
                 source_location loc = source_location::current())
      : code_(ErrorCode::Unknown),
        message_(std::move(message)),
        location_(loc),
        full_message_(build_full_message()) {}  // Build immediately

  // Special member functions
  Error(const Error& other)
      : code_(other.code_),
        message_(other.message_),
        location_(other.location_),
        full_message_(other.full_message_) {}

  Error& operator=(const Error& other) {
    if (this != &other) {
      code_ = other.code_;
      message_ = other.message_;
      location_ = other.location_;
      full_message_ = other.full_message_;
    }
    return *this;
  }

  Error(Error&& other) noexcept
      : code_(other.code_),
        message_(std::move(other.message_)),
        location_(other.location_),
        full_message_(std::move(other.full_message_)) {}

  Error& operator=(Error&& other) noexcept {
    if (this != &other) {
      code_ = other.code_;
      message_ = std::move(other.message_);
      location_ = other.location_;
      full_message_ = std::move(other.full_message_);
    }
    return *this;
  }

  ~Error() override = default;

  [[nodiscard]]
  auto what() const noexcept -> const char* override {
    return full_message_.c_str();
  }

  [[nodiscard]]
  auto code() const noexcept -> ErrorCode {
    return code_;
  }
  [[nodiscard]]
  auto message() const noexcept -> const std::string& {
    return message_;
  }
  [[nodiscard]]
  auto location() const noexcept -> const source_location& {
    return location_;
  }

  [[nodiscard]]
  auto file() const noexcept -> const char* {
    return filename_only(location_.file_name());
  }
  [[nodiscard]]
  auto line() const noexcept -> std::uint32_t {
    return location_.line();
  }
  [[nodiscard]]
  auto function() const noexcept -> const char* {
    return location_.function_name();
  }

 protected:
  [[nodiscard]]
  virtual
  auto build_full_message() const -> std::string {
#if SCL_HAS_FORMAT
    return std::format("[{}] {}:{} in {}: {}", error_code_name(code_), file(),
                       line(), function(), message_);
#else
    // NOLINT: snprintf is necessary for pre-C++20 formatting
    static constexpr std::size_t kBufferSize = 2048;
    std::array<char, kBufferSize> buffer{};
    std::snprintf(buffer.data(), buffer.size(),  // NOLINT(cppcoreguidelines-pro-type-vararg)
                  "[%s] %s:%u in %s: %s", error_code_name(code_), file(),
                  line(), function(), message_.c_str());
    return buffer.data();
#endif
  }
};

// -----------------------------------------------------------------------------
// Memory Errors (100-199)
// -----------------------------------------------------------------------------

/// @brief Memory-related error base class
class MemoryError : public Error {
 public:
  explicit MemoryError(ErrorCode code, std::string message,
                       source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit MemoryError(std::string message,
                       source_location loc = source_location::current())
      : Error(ErrorCode::OutOfMemory, std::move(message), loc) {}

  // Special member functions
  MemoryError(const MemoryError&) = default;
  auto operator=(const MemoryError&) -> MemoryError& = default;
  MemoryError(MemoryError&&) noexcept = default;
  auto operator=(MemoryError&&) noexcept -> MemoryError& = default;

  ~MemoryError() override = default;
};

/// @brief Out of memory error
class OutOfMemoryError : public MemoryError {
 public:
  explicit OutOfMemoryError(
      std::string message = "Out of memory",
      source_location loc = source_location::current())
      : MemoryError(ErrorCode::OutOfMemory, std::move(message), loc) {}

  explicit OutOfMemoryError(std::size_t requested_size,
                            source_location loc = source_location::current())
      : MemoryError(ErrorCode::OutOfMemory,
                    "Failed to allocate " + std::to_string(requested_size) +
                        " bytes",
                    loc) {}
};

/// @brief Null pointer error
class NullPointerError : public MemoryError {
 public:
  explicit NullPointerError(const std::string& name = "pointer",
                            source_location loc = source_location::current())
      : MemoryError(ErrorCode::NullPointer, name + " is null", loc) {}
};

/// @brief Alignment error
class AlignmentError : public MemoryError {
 public:
  explicit AlignmentError(std::size_t required_alignment,
                          std::size_t misalignment_offset,
                          source_location loc = source_location::current())
      : MemoryError(ErrorCode::AlignmentError,
                    "Alignment error: required " +
                        std::to_string(required_alignment) +
                        "-byte alignment, offset from alignment: " +
                        std::to_string(misalignment_offset),
                    loc) {}
};

/// @brief Buffer size error
class BufferSizeError : public MemoryError {
 public:
  explicit BufferSizeError(std::size_t required, std::size_t actual,
                           source_location loc = source_location::current())
      : MemoryError(ErrorCode::BufferTooSmall,
                    "Buffer too small: required " + std::to_string(required) +
                        ", got " + std::to_string(actual),
                    loc) {}
};

// -----------------------------------------------------------------------------
// Dimension/Shape Errors (200-299)
// -----------------------------------------------------------------------------

/// @brief Dimension-related error base class
class DimensionError : public Error {
 public:
  explicit DimensionError(ErrorCode code, std::string message,
                          source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit DimensionError(std::string message,
                          source_location loc = source_location::current())
      : Error(ErrorCode::DimensionMismatch, std::move(message), loc) {}

  // Special member functions
  DimensionError(const DimensionError&) = default;
  auto operator=(const DimensionError&) -> DimensionError& = default;
  DimensionError(DimensionError&&) noexcept = default;
  auto operator=(DimensionError&&) noexcept -> DimensionError& = default;

  ~DimensionError() override = default;
};

/// @brief Shape mismatch error
class ShapeMismatchError : public DimensionError {
 public:
  explicit ShapeMismatchError(std::string message,
                              source_location loc = source_location::current())
      : DimensionError(ErrorCode::ShapeMismatch, std::move(message), loc) {}
};

/// @brief Broadcast error
class BroadcastError : public DimensionError {
 public:
  explicit BroadcastError(std::string message,
                          source_location loc = source_location::current())
      : DimensionError(ErrorCode::BroadcastError, std::move(message), loc) {}
};

/// @brief Invalid axis error
class InvalidAxisError : public DimensionError {
 public:
  explicit InvalidAxisError(std::int64_t axis, std::int64_t ndim,
                            source_location loc = source_location::current())
      : DimensionError(ErrorCode::AxisOutOfRange,
                       "Axis " + std::to_string(axis) +
                           " out of range for " + std::to_string(ndim) +
                           "-dimensional tensor",
                       loc) {}
};

// -----------------------------------------------------------------------------
// Type/Precision Errors (300-399)
// -----------------------------------------------------------------------------

/// @brief Type-related error base class
class TypeError : public Error {
 public:
  explicit TypeError(ErrorCode code, std::string message,
                     source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit TypeError(std::string message,
                     source_location loc = source_location::current())
      : Error(ErrorCode::TypeMismatch, std::move(message), loc) {}

  // Special member functions
  TypeError(const TypeError&) = default;
  auto operator=(const TypeError&) -> TypeError& = default;
  TypeError(TypeError&&) noexcept = default;
  auto operator=(TypeError&&) noexcept -> TypeError& = default;

  ~TypeError() override = default;
};

/// @brief Unsupported type error
class UnsupportedTypeError : public TypeError {
 public:
  explicit UnsupportedTypeError(
      const std::string& type_name,
      source_location loc = source_location::current())
      : TypeError(ErrorCode::UnsupportedType, "Unsupported type: " + type_name,
                  loc) {}
};

/// @brief Data type mismatch error
class DtypeMismatchError : public TypeError {
 public:
  explicit DtypeMismatchError(const std::string& expected,
                              const std::string& actual,
                              source_location loc = source_location::current())
      : TypeError(ErrorCode::DtypeMismatch,
                  "Dtype mismatch: expected " + expected + ", got " + actual,
                  loc) {}
};

// -----------------------------------------------------------------------------
// Value/Argument Errors (400-499)
// -----------------------------------------------------------------------------

/// @brief Value-related error base class
class ValueError : public Error {
 public:
  explicit ValueError(ErrorCode code, std::string message,
                      source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit ValueError(std::string message,
                      source_location loc = source_location::current())
      : Error(ErrorCode::InvalidArgument, std::move(message), loc) {}

  // Special member functions
  ValueError(const ValueError&) = default;
  auto operator=(const ValueError&) -> ValueError& = default;
  ValueError(ValueError&&) noexcept = default;
  auto operator=(ValueError&&) noexcept -> ValueError& = default;

  ~ValueError() override = default;
};

/// @brief Index out of bounds error
class IndexError : public ValueError {
 public:
  explicit IndexError(std::int64_t index, std::int64_t size,
                      source_location loc = source_location::current())
      : ValueError(ErrorCode::IndexOutOfBounds,
                   "Index " + std::to_string(index) + " out of bounds for size " +
                       std::to_string(size),
                   loc) {}
};

/// @brief Range error
class RangeError : public ValueError {
 public:
  explicit RangeError(std::string message,
                      source_location loc = source_location::current())
      : ValueError(ErrorCode::OutOfRange, std::move(message), loc) {}
};

/// @brief Division by zero error
class DivisionByZeroError : public ValueError {
 public:
  explicit DivisionByZeroError(
      source_location loc = source_location::current())
      : ValueError(ErrorCode::DivisionByZero, "Division by zero", loc) {}
};

/// @brief NaN error
class NaNError : public ValueError {
 public:
  explicit NaNError(const std::string& context = "",
                    source_location loc = source_location::current())
      : ValueError(ErrorCode::NaN,
                   context.empty() ? "NaN encountered"
                                   : "NaN encountered in " + context,
                   loc) {}
};

/// @brief Overflow error
class OverflowError : public ValueError {
 public:
  explicit OverflowError(std::string message = "Numerical overflow",
                         source_location loc = source_location::current())
      : ValueError(ErrorCode::Overflow, std::move(message), loc) {}
};

// -----------------------------------------------------------------------------
// I/O Errors (500-599)
// -----------------------------------------------------------------------------

/// @brief I/O error base class
class IoError : public Error {
 public:
  explicit IoError(ErrorCode code, std::string message,
                   source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit IoError(std::string message,
                   source_location loc = source_location::current())
      : Error(ErrorCode::IoError, std::move(message), loc) {}

  // Special member functions
  IoError(const IoError&) = default;
  auto operator=(const IoError&) -> IoError& = default;
  IoError(IoError&&) noexcept = default;
  auto operator=(IoError&&) noexcept -> IoError& = default;

  ~IoError() override = default;
};

/// @brief File not found error
class FileNotFoundError : public IoError {
 public:
  explicit FileNotFoundError(const std::string& path,
                             source_location loc = source_location::current())
      : IoError(ErrorCode::FileNotFound, "File not found: " + path, loc) {}
};

// -----------------------------------------------------------------------------
// Algorithm/Computation Errors (600-699)
// -----------------------------------------------------------------------------

/// @brief Computation error base class
class ComputationError : public Error {
 public:
  explicit ComputationError(ErrorCode code, std::string message,
                            source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit ComputationError(std::string message,
                            source_location loc = source_location::current())
      : Error(ErrorCode::ComputationError, std::move(message), loc) {}

  // Special member functions
  ComputationError(const ComputationError&) = default;
  auto operator=(const ComputationError&) -> ComputationError& = default;
  ComputationError(ComputationError&&) noexcept = default;
  auto operator=(ComputationError&&) noexcept -> ComputationError& = default;

  ~ComputationError() override = default;
};

/// @brief Convergence error
class ConvergenceError : public ComputationError {
 public:
  explicit ConvergenceError(const std::string& algorithm,
                            std::size_t iterations,
                            source_location loc = source_location::current())
      : ComputationError(ErrorCode::ConvergenceError,
                         algorithm + " failed to converge after " +
                             std::to_string(iterations) + " iterations",
                         loc) {}
};

/// @brief Singular matrix error
class SingularMatrixError : public ComputationError {
 public:
  explicit SingularMatrixError(
      source_location loc = source_location::current())
      : ComputationError(ErrorCode::SingularMatrix, "Matrix is singular", loc) {
  }
};

/// @brief Numerical instability error
class NumericalInstabilityError : public ComputationError {
 public:
  explicit NumericalInstabilityError(
      std::string message, source_location loc = source_location::current())
      : ComputationError(ErrorCode::NumericalInstability, std::move(message),
                         loc) {}
};

// -----------------------------------------------------------------------------
// Threading Errors (700-799)
// -----------------------------------------------------------------------------

/// @brief Threading error base class
class ThreadingError : public Error {
 public:
  explicit ThreadingError(ErrorCode code, std::string message,
                          source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit ThreadingError(std::string message,
                          source_location loc = source_location::current())
      : Error(ErrorCode::ThreadError, std::move(message), loc) {}

  // Special member functions
  ThreadingError(const ThreadingError&) = default;
  auto operator=(const ThreadingError&) -> ThreadingError& = default;
  ThreadingError(ThreadingError&&) noexcept = default;
  auto operator=(ThreadingError&&) noexcept -> ThreadingError& = default;

  ~ThreadingError() override = default;
};

// -----------------------------------------------------------------------------
// Internal Errors (900-999)
// -----------------------------------------------------------------------------

/// @brief Internal error base class
class InternalError : public Error {
 public:
  explicit InternalError(ErrorCode code, std::string message,
                         source_location loc = source_location::current())
      : Error(code, std::move(message), loc) {}

  explicit InternalError(std::string message,
                         source_location loc = source_location::current())
      : Error(ErrorCode::InternalError, std::move(message), loc) {}

  // Special member functions
  InternalError(const InternalError&) = default;
  auto operator=(const InternalError&) -> InternalError& = default;
  InternalError(InternalError&&) noexcept = default;
  auto operator=(InternalError&&) noexcept -> InternalError& = default;

  ~InternalError() override = default;
};

/// @brief Not implemented error
class NotImplementedError : public InternalError {
 public:
  explicit NotImplementedError(
      const std::string& feature = "",
      source_location loc = source_location::current())
      : InternalError(ErrorCode::NotImplemented,
                      feature.empty() ? "Not implemented"
                                      : feature + " is not implemented",
                      loc) {}
};

/// @brief Assertion failed error
class AssertionError : public InternalError {
 public:
  explicit AssertionError(const std::string& condition,
                          source_location loc = source_location::current())
      : InternalError(ErrorCode::AssertionFailed,
                      "Assertion failed: " + condition, loc) {}
};

}  // namespace scl

// =============================================================================
// SECTION 3: Error Formatting Utilities
// =============================================================================

namespace scl::detail {

#if SCL_HAS_FORMAT
/// @brief Format error message with source location (C++20 std::format
/// version)
template <typename... Args>
[[nodiscard]]
auto format_error(source_location loc, std::format_string<Args...> fmt,
                  Args&&... args) -> std::string {
  return std::format(fmt, std::forward<Args>(args)...);
}
#else
/// @brief Format error message (fallback version using snprintf)
template <typename... Args>
[[nodiscard]]
auto format_error(source_location /*loc*/, const char* fmt, Args&&... args)
    -> std::string {
  static constexpr std::size_t kBufferSize = 1024;
  std::array<char, kBufferSize> buffer{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::snprintf(buffer.data(), buffer.size(), fmt,
                std::forward<Args>(args)...);
  return buffer.data();
}
#endif

/// @brief Simple string format without source location
template <typename... Args>
[[nodiscard]]
auto format_message(const char* fmt, Args&&... args) -> std::string {
  static constexpr std::size_t kBufferSize = 1024;
  std::array<char, kBufferSize> buffer{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::snprintf(buffer.data(), buffer.size(), fmt,
                std::forward<Args>(args)...);
  return buffer.data();
}

/// @brief Debug assertion failure handler
[[noreturn]] inline void debug_assert_fail(
    const char* expr, source_location loc = source_location::current()) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::fprintf(stderr, "SCL_DEBUG_ASSERT failed: %s\n  at %s:%u in %s\n", expr,
               filename_only(loc.file_name()), loc.line(), loc.function_name());
  std::abort();
}

/// @brief Debug assertion failure with message
[[noreturn]] inline void debug_assert_fail_msg(
    const char* expr, const char* msg,
    source_location loc = source_location::current()) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  std::fprintf(stderr,
               "SCL_DEBUG_ASSERT failed: %s\n  Message: %s\n  at %s:%u in %s\n",
               expr, msg, filename_only(loc.file_name()), loc.line(),
               loc.function_name());
  std::abort();
}

}  // namespace scl::detail

// =============================================================================
// SECTION 4: Thread-Local Error State
// =============================================================================

namespace scl {

/// @brief Thread-local error state for error tracking and reporting
/// @note This class uses a singleton pattern with thread_local storage
///       to provide per-thread error state without explicit instantiation.
class ThreadErrorState {
 public:
  static constexpr std::size_t kMaxMessageLength = 1024;

  // Delete copy and move operations (singleton pattern)
  ThreadErrorState(const ThreadErrorState&) = delete;
  auto operator=(const ThreadErrorState&) -> ThreadErrorState& = delete;
  ThreadErrorState(ThreadErrorState&&) = delete;
  auto operator=(ThreadErrorState&&) -> ThreadErrorState& = delete;
  ~ThreadErrorState() = default;

  /// @brief Get the thread-local instance (singleton)
  /// @return Reference to thread-local error state
  [[nodiscard]]
  static
  auto instance() noexcept -> ThreadErrorState& {
    thread_local ThreadErrorState state;
    return state;
  }

  /// @brief Clear error state (reset to success)
  auto clear() noexcept -> void {
    code_ = ErrorCode::Success;
    message_ = {};
  }

  /// @brief Set error state with code and optional message
  /// @param code Error code to set
  /// @param message Optional error message (uses error code name if nullptr)
  auto set(ErrorCode code, const char* message = nullptr) noexcept -> void {
    code_ = code;
    if (message != nullptr) {
      const std::size_t len =
          std::min(std::strlen(message), kMaxMessageLength - 1);
      std::strncpy(message_.data(), message, len);
      message_[len] = '\0';
    } else {
      const char* code_name = error_code_name(code);
      const std::size_t len =
          std::min(std::strlen(code_name), kMaxMessageLength - 1);
      std::strncpy(message_.data(), code_name, len);
      message_[len] = '\0';
    }
  }

  /// @brief Set error state from SCL exception
  /// @param error The error to capture
  auto set_from_exception(const Error& error) noexcept -> void {
    set(error.code(), error.message().c_str());
  }

  /// @brief Set error state from std::exception
  /// @param exception The exception to capture
  auto set_from_std_exception(const std::exception& exception) noexcept
      -> void {
    set(ErrorCode::Unknown, exception.what());
  }

  /// @brief Get current error code
  /// @return Current error code
  [[nodiscard]]
  auto code() const noexcept -> ErrorCode {
    return code_;
  }

  /// @brief Get error message as C-string
  /// @return Pointer to null-terminated error message
  [[nodiscard]]
  auto message() const noexcept -> const char* {
    return message_.data();
  }

  /// @brief Get error message as std::string_view
  /// @return String view of error message
  [[nodiscard]]
  auto message_view() const noexcept -> std::string_view {
    return {message_.data()};
  }

  /// @brief Check if an error is currently set
  /// @return true if error state is not Success
  [[nodiscard]]
  auto has_error() const noexcept -> bool {
    return code_ != ErrorCode::Success;
  }

  /// @brief Get error category
  /// @return Category name string
  [[nodiscard]]
  auto category() const noexcept -> const char* {
    return error_code_category(code_);
  }

  /// @brief Check if current error is recoverable
  /// @return true if error is recoverable
  [[nodiscard]]
  auto is_recoverable() const noexcept -> bool {
    return scl::is_recoverable(code_);
  }

 private:
  // Private constructor for singleton pattern
  ThreadErrorState() noexcept = default;

  ErrorCode code_ = ErrorCode::Success;
  std::array<char, kMaxMessageLength> message_{};
};

// -----------------------------------------------------------------------------
// Convenience Functions
// -----------------------------------------------------------------------------

/// @brief Get thread-local error state instance
/// @return Reference to thread-local error state
[[nodiscard]] inline auto get_thread_error() noexcept -> ThreadErrorState& {
  return ThreadErrorState::instance();
}

/// @brief Clear thread-local error state
inline auto clear_thread_error() noexcept -> void {
  ThreadErrorState::instance().clear();
}

/// @brief Set thread-local error state
/// @param code Error code to set
/// @param message Optional error message
inline auto set_thread_error(ErrorCode code,
                              const char* message = nullptr) noexcept -> void {
  ThreadErrorState::instance().set(code, message);
}

/// @brief Set thread-local error from exception
/// @param error The error to capture
inline auto set_thread_error_from_exception(const Error& error) noexcept
    -> void {
  ThreadErrorState::instance().set_from_exception(error);
}

/// @brief Check if thread has an error set
/// @return true if error state is not Success
[[nodiscard]] inline auto thread_has_error() noexcept -> bool {
  return ThreadErrorState::instance().has_error();
}

/// @brief Get current thread error code
/// @return Current error code
[[nodiscard]] inline auto get_thread_error_code() noexcept -> ErrorCode {
  return ThreadErrorState::instance().code();
}

/// @brief Get current thread error message
/// @return Pointer to null-terminated error message
[[nodiscard]] inline auto get_thread_error_message() noexcept -> const char* {
  return ThreadErrorState::instance().message();
}

}  // namespace scl

// =============================================================================
// SECTION 5: Compile-Time Check Functions
// =============================================================================

namespace scl::error {

/// @brief Compile-time type check for floating-point types
template <typename T>
constexpr auto check_floating() noexcept -> void {
  static_assert(std::is_floating_point_v<T>, "Type must be floating-point");
}

/// @brief Compile-time type check for integral types
template <typename T>
constexpr auto check_integral() noexcept -> void {
  static_assert(std::is_integral_v<T>, "Type must be integral");
}

/// @brief Compile-time type check for arithmetic types
template <typename T>
constexpr auto check_arithmetic() noexcept -> void {
  static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
}

/// @brief Compile-time type check for signed types
template <typename T>
constexpr auto check_signed() noexcept -> void {
  static_assert(std::is_signed_v<T>, "Type must be signed");
}

/// @brief Compile-time type check for unsigned types
template <typename T>
constexpr auto check_unsigned() noexcept -> void {
  static_assert(std::is_unsigned_v<T>, "Type must be unsigned");
}

/// @brief Compile-time type size check
template <typename T, std::size_t MinBytes>
constexpr auto check_size() noexcept -> void {
  static_assert(sizeof(T) >= MinBytes, "Type size too small");
}

/// @brief Compile-time type equality check
template <typename T, typename U>
constexpr auto check_same() noexcept -> void {
  static_assert(std::is_same_v<T, U>, "Types must be the same");
}

}  // namespace scl::error

// =============================================================================
// SECTION 6: Runtime Check Functions
// =============================================================================

namespace scl::error {

/// @brief Check argument condition, throw ValueError if false
template <typename... Args>
inline auto check_arg(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw ValueError(detail::format_error(source_location::current(),
                                          std::forward<Args>(args)...),
                     source_location::current());
  }
}

/// @brief Check dimension condition, throw DimensionError if false
template <typename... Args>
inline auto check_dim(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw DimensionError(detail::format_error(source_location::current(),
                                              std::forward<Args>(args)...),
                         source_location::current());
  }
}

/// @brief Check range condition, throw RangeError if false
template <typename... Args>
inline auto check_range(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw RangeError(detail::format_error(source_location::current(),
                                          std::forward<Args>(args)...),
                     source_location::current());
  }
}

/// @brief Check memory condition, throw MemoryError if false
template <typename... Args>
inline auto check_mem(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw MemoryError(detail::format_error(source_location::current(),
                                           std::forward<Args>(args)...),
                      source_location::current());
  }
}

/// @brief Check type condition, throw TypeError if false
template <typename... Args>
inline auto check_type(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw TypeError(detail::format_error(source_location::current(),
                                         std::forward<Args>(args)...),
                    source_location::current());
  }
}

/// @brief Check IO condition, throw IoError if false
template <typename... Args>
inline auto check_io(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw IoError(detail::format_error(source_location::current(),
                                       std::forward<Args>(args)...),
                  source_location::current());
  }
}

/// @brief Check computation condition, throw ComputationError if false
template <typename... Args>
inline auto check_compute(bool condition, Args&&... args) -> void {
  if (!condition) [[unlikely]] {
    throw ComputationError(detail::format_error(source_location::current(),
                                                 std::forward<Args>(args)...),
                           source_location::current());
  }
}

/// @brief Check pointer is not null, throw NullPointerError if null
template <typename T>
inline auto check_not_null(const T* ptr, const char* name = "pointer") -> void {
  if (ptr == nullptr) [[unlikely]] {
    throw NullPointerError(name, source_location::current());
  }
}

/// @brief Check index is in bounds, throw IndexError if out of bounds
template <typename IndexT, typename SizeT>
inline auto check_index(IndexT index, SizeT size) -> void {
  const auto idx = static_cast<std::int64_t>(index);
  const auto sz = static_cast<std::int64_t>(size);
  if (idx < 0 || idx >= sz) [[unlikely]] {
    throw IndexError(idx, sz, source_location::current());
  }
}

/// @brief Check sizes match, throw DimensionError if mismatch
template <typename T1, typename T2>
inline auto check_size_match(T1 size_a, T2 size_b) -> void {
  if (size_a != size_b) [[unlikely]] {
    throw DimensionError(
        ErrorCode::SizeMismatch,
        detail::format_message("Size mismatch: %zu vs %zu",
                               static_cast<std::size_t>(size_a),
                               static_cast<std::size_t>(size_b)),
        source_location::current());
  }
}

/// @brief Check value is positive, throw ValueError if not
template <typename T>
inline auto check_positive(T value, const char* name) -> void {
  if (value <= 0) [[unlikely]] {
    throw ValueError(ErrorCode::InvalidArgument,
                     std::string(name) + " must be positive",
                     source_location::current());
  }
}

/// @brief Check value is non-negative, throw ValueError if negative
template <typename T>
inline auto check_non_negative(T value, const char* name) -> void {
  if (value < 0) [[unlikely]] {
    throw ValueError(ErrorCode::NegativeValue,
                     std::string(name) + " must be non-negative",
                     source_location::current());
  }
}

/// @brief Check pointer alignment, throw AlignmentError if misaligned
template <typename T, std::size_t Alignment>
inline auto check_alignment(const T* ptr) -> void {
  static_assert((Alignment & (Alignment - 1)) == 0,
                "Alignment must be a power of 2");
  auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  if ((addr & (Alignment - 1)) != 0) [[unlikely]] {
    throw AlignmentError(Alignment, addr % Alignment,
                         source_location::current());
  }
}

/// @brief Check value is finite, throw NaNError or ValueError if not
template <typename T>
inline auto check_finite(T value) -> void {
  if (!std::isfinite(value)) [[unlikely]] {
    if (std::isnan(value)) {
      throw NaNError("", source_location::current());
    }
    throw ValueError(ErrorCode::Infinity, "Infinity encountered",
                     source_location::current());
  }
}

/// @brief Throw not implemented error
[[noreturn]]
inline auto not_implemented(const std::string& feature = "") -> void {
  throw NotImplementedError(feature, source_location::current());
}

/// @brief Throw unreachable code error
[[noreturn]]
inline auto unreachable() -> void {
  throw InternalError(ErrorCode::UnreachableCode, "Unreachable code reached",
                      source_location::current());
}

}  // namespace scl::error

// =============================================================================
// SECTION 7: Backward Compatibility Macros
// =============================================================================
//
// NOTE: These macros are provided for backward compatibility only.
//       Prefer using the template functions in scl::error:: namespace.
//       Example: scl::error::check_arg(...) instead of SCL_CHECK_ARG(...)
//
// NOLINTBEGIN(cppcoreguidelines-macro-usage)

// Compile-time checks
#define SCL_STATIC_CHECK(cond, msg) static_assert(cond, msg)
#define SCL_STATIC_CHECK_FLOATING(T) ::scl::error::check_floating<T>()
#define SCL_STATIC_CHECK_INTEGRAL(T) ::scl::error::check_integral<T>()
#define SCL_STATIC_CHECK_ARITHMETIC(T) ::scl::error::check_arithmetic<T>()
#define SCL_STATIC_CHECK_SIGNED(T) ::scl::error::check_signed<T>()
#define SCL_STATIC_CHECK_UNSIGNED(T) ::scl::error::check_unsigned<T>()
#define SCL_STATIC_CHECK_SIZE(T, min_bytes) \
  ::scl::error::check_size<T, min_bytes>()
#define SCL_STATIC_CHECK_SAME(T, U) ::scl::error::check_same<T, U>()

// Runtime checks
#define SCL_CHECK_ARG(cond, ...) ::scl::error::check_arg((cond), __VA_ARGS__)
#define SCL_CHECK_DIM(cond, ...) ::scl::error::check_dim((cond), __VA_ARGS__)
#define SCL_CHECK_RANGE(cond, ...) ::scl::error::check_range((cond), __VA_ARGS__)
#define SCL_CHECK_MEM(cond, ...) ::scl::error::check_mem((cond), __VA_ARGS__)
#define SCL_CHECK_TYPE(cond, ...) ::scl::error::check_type((cond), __VA_ARGS__)
#define SCL_CHECK_IO(cond, ...) ::scl::error::check_io((cond), __VA_ARGS__)
#define SCL_CHECK_COMPUTE(cond, ...) \
  ::scl::error::check_compute((cond), __VA_ARGS__)

// Specific checks
#define SCL_CHECK_NOT_NULL(ptr) ::scl::error::check_not_null((ptr), #ptr)
#define SCL_CHECK_INDEX(index, size) ::scl::error::check_index((index), (size))
#define SCL_CHECK_SIZE_MATCH(a, b) ::scl::error::check_size_match((a), (b))
#define SCL_CHECK_POSITIVE(val, name) \
  ::scl::error::check_positive((val), (name))
#define SCL_CHECK_NON_NEGATIVE(val, name) \
  ::scl::error::check_non_negative((val), (name))
#define SCL_CHECK_ALIGNMENT(ptr, align) \
  ::scl::error::check_alignment<std::remove_pointer_t<decltype(ptr)>, align>(ptr)
#define SCL_CHECK_FINITE(val) ::scl::error::check_finite((val))


// Debug assertions (must remain as macros for condition stringification)
#ifdef NDEBUG
  #define SCL_DEBUG_ASSERT(cond) ((void)0)
  #define SCL_DEBUG_ASSERT_MSG(cond, msg) ((void)0)
#else
  #define SCL_DEBUG_ASSERT(cond)                                           \
    do {                                                                   \
      if (!(cond)) [[unlikely]] {                                          \
        ::scl::detail::debug_assert_fail(#cond,                            \
                                         ::scl::source_location::current()); \
      }                                                                    \
    } while (0)

  #define SCL_DEBUG_ASSERT_MSG(cond, msg)                              \
    do {                                                               \
      if (!(cond)) [[unlikely]] {                                      \
        ::scl::detail::debug_assert_fail_msg(                          \
            #cond, msg, ::scl::source_location::current());            \
      }                                                                \
    } while (0)
#endif

// Throw shortcuts
#define SCL_NOT_IMPLEMENTED(feature) ::scl::error::not_implemented(feature)
#define SCL_UNREACHABLE_CODE() ::scl::error::unreachable()

// NOLINTEND(cppcoreguidelines-macro-usage)
