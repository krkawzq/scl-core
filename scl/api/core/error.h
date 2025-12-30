/**
 * @file scl/api/core/error.h
 * @brief SCL Error Handling C-API
 *
 * This header provides C-ABI compatible functions for error handling in SCL.
 *
 * ## Thread Safety
 *
 * All functions in this header are thread-safe. Each thread maintains its own
 * error state, so errors from one thread do not affect other threads.
 *
 * ## Error Handling Pattern
 *
 * ```c
 * // Call an SCL function
 * int32_t result = scl_some_operation(...);
 *
 * // Check for errors
 * if (scl_is_error(result)) {
 *     // Get detailed error information
 *     const char* msg = scl_get_error_message();
 *     const char* name = scl_error_code_name(result);
 *     printf("Error [%s]: %s\n", name, msg);
 *
 *     // Clear error state for next operation
 *     scl_clear_error();
 * }
 * ```
 *
 * ## Error Code Ranges
 *
 * | Range     | Category     | Examples                          |
 * |-----------|--------------|-----------------------------------|
 * | 0         | Success      | No error                          |
 * | 1-99      | General      | Unknown, NotImplemented           |
 * | 100-199   | Memory       | OutOfMemory, NullPointer          |
 * | 200-299   | Dimension    | DimensionMismatch, ShapeMismatch  |
 * | 300-399   | Type         | TypeMismatch, InvalidType         |
 * | 400-499   | Value        | InvalidArgument, IndexOutOfBounds |
 * | 500-599   | I/O          | IoError, FileNotFound             |
 * | 600-699   | Computation  | ComputationError, Convergence     |
 * | 700-799   | Threading    | ThreadError, Deadlock             |
 * | 800-899   | Hardware     | DeviceError, SimdNotSupported     |
 * | 900-999   | Internal     | InternalError, AssertionFailed    |
 */

#ifndef SCL_API_CORE_ERROR_H_
#define SCL_API_CORE_ERROR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: Error Code Queries
 * ============================================================================ */

/**
 * @brief Get human-readable name for an error code
 * @param code Error code value
 * @return Static string with error name (never null)
 */
const char* scl_error_code_name(int32_t code);

/**
 * @brief Get error category name for an error code
 * @param code Error code value
 * @return Static string with category name (e.g., "Memory", "Value")
 */
const char* scl_error_code_category(int32_t code);

/**
 * @brief Check if error code represents success
 * @param code Error code value
 * @return 1 if success (code == 0), 0 otherwise
 */
int32_t scl_is_success(int32_t code);

/**
 * @brief Check if error code represents a failure
 * @param code Error code value
 * @return 1 if error (code != 0), 0 otherwise
 */
int32_t scl_is_error(int32_t code);

/**
 * @brief Check if error is recoverable (not internal/hardware)
 * @param code Error code value
 * @return 1 if recoverable, 0 otherwise
 */
int32_t scl_is_recoverable(int32_t code);

/* ============================================================================
 * SECTION 2: Thread-Local Error State
 * ============================================================================ */

/**
 * @brief Get the last error code for the current thread
 * @return Error code (0 = success, non-zero = error)
 */
int32_t scl_get_last_error(void);

/**
 * @brief Get the last error message for the current thread
 * @return Pointer to error message string (valid until next error or clear)
 * @warning Do not free the returned pointer
 */
const char* scl_get_error_message(void);

/**
 * @brief Clear the thread-local error state
 */
void scl_clear_error(void);

/**
 * @brief Set a custom error for the current thread
 * @param code Error code value
 * @param message Error message (can be NULL, will use default)
 */
void scl_set_error(int32_t code, const char* message);

/**
 * @brief Check if an error is currently set for this thread
 * @return 1 if error is set, 0 otherwise
 */
int32_t scl_has_error(void);

/* ============================================================================
 * SECTION 3: Common Error Code Constants
 * ============================================================================ */

/** @brief Success code (0) */
int32_t scl_error_success(void);

/** @brief Unknown/unspecified error (1) */
int32_t scl_error_unknown(void);

/** @brief Feature not implemented (2) */
int32_t scl_error_not_implemented(void);

/** @brief Memory allocation failed (100) */
int32_t scl_error_out_of_memory(void);

/** @brief Null pointer dereference (103) */
int32_t scl_error_null_pointer(void);

/** @brief Invalid argument (400) */
int32_t scl_error_invalid_argument(void);

/** @brief Index out of bounds (402) */
int32_t scl_error_index_out_of_bounds(void);

/** @brief Dimension mismatch (200) */
int32_t scl_error_dimension_mismatch(void);

/** @brief Shape mismatch (201) */
int32_t scl_error_shape_mismatch(void);

/** @brief Type mismatch (300) */
int32_t scl_error_type_mismatch(void);

/** @brief Division by zero (410) */
int32_t scl_error_division_by_zero(void);

/** @brief NaN encountered (406) */
int32_t scl_error_nan(void);

/** @brief Numerical overflow (409) */
int32_t scl_error_overflow(void);

/** @brief General computation error (600) */
int32_t scl_error_computation(void);

/** @brief Algorithm failed to converge (601) */
int32_t scl_error_convergence(void);

/** @brief Matrix is singular (602) */
int32_t scl_error_singular_matrix(void);

/** @brief General I/O error (500) */
int32_t scl_error_io(void);

/** @brief File not found (501) */
int32_t scl_error_file_not_found(void);

/** @brief Internal error (900) */
int32_t scl_error_internal(void);

/* ============================================================================
 * SECTION 4: Error Code Range Queries
 * ============================================================================ */

/** @brief Get minimum value for memory error codes (100) */
int32_t scl_error_memory_min(void);

/** @brief Get maximum value for memory error codes (199) */
int32_t scl_error_memory_max(void);

/** @brief Get minimum value for dimension error codes (200) */
int32_t scl_error_dimension_min(void);

/** @brief Get maximum value for dimension error codes (299) */
int32_t scl_error_dimension_max(void);

/** @brief Get minimum value for type error codes (300) */
int32_t scl_error_type_min(void);

/** @brief Get maximum value for type error codes (399) */
int32_t scl_error_type_max(void);

/** @brief Get minimum value for value/argument error codes (400) */
int32_t scl_error_value_min(void);

/** @brief Get maximum value for value/argument error codes (499) */
int32_t scl_error_value_max(void);

/** @brief Get minimum value for I/O error codes (500) */
int32_t scl_error_io_min(void);

/** @brief Get maximum value for I/O error codes (599) */
int32_t scl_error_io_max(void);

/** @brief Get minimum value for computation error codes (600) */
int32_t scl_error_compute_min(void);

/** @brief Get maximum value for computation error codes (699) */
int32_t scl_error_compute_max(void);

/** @brief Get minimum value for threading error codes (700) */
int32_t scl_error_thread_min(void);

/** @brief Get maximum value for threading error codes (799) */
int32_t scl_error_thread_max(void);

/** @brief Get minimum value for internal error codes (900) */
int32_t scl_error_internal_min(void);

/** @brief Get maximum value for internal error codes (999) */
int32_t scl_error_internal_max(void);

/* ============================================================================
 * SECTION 5: Error Information Queries
 * ============================================================================ */

/**
 * @brief Get full error information as formatted string
 * @param buffer Output buffer for error string
 * @param buffer_size Size of output buffer in bytes
 * @return Number of bytes written, or required size if buffer is NULL/too small
 *
 * Format: "[CODE_NAME] message"
 *
 * Example:
 * ```c
 * char buf[256];
 * scl_get_error_info(buf, sizeof(buf));
 * printf("%s\n", buf);  // "[InvalidArgument] size must be positive"
 * ```
 */
size_t scl_get_error_info(char* buffer, size_t buffer_size);

/**
 * @brief Copy error message to a user-provided buffer
 * @param buffer Output buffer for message
 * @param buffer_size Size of output buffer in bytes
 * @return Number of bytes written (excluding null terminator)
 *
 * Safer alternative to scl_get_error_message() for multi-threaded use.
 */
size_t scl_copy_error_message(char* buffer, size_t buffer_size);

/* ============================================================================
 * SECTION 6: Version and Library Info
 * ============================================================================ */

/**
 * @brief Get library version string
 * @return Static string with version (e.g., "2.0.0")
 */
const char* scl_get_version(void);

/**
 * @brief Get library version as integer components
 * @param major Output for major version number (can be NULL)
 * @param minor Output for minor version number (can be NULL)
 * @param patch Output for patch version number (can be NULL)
 */
void scl_get_version_components(int32_t* major, int32_t* minor, int32_t* patch);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SCL_API_CORE_ERROR_H_ */

