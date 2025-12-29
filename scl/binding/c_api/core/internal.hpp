#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/internal.hpp
// BRIEF: Internal C++ wrapper structures for C API binding layer
// =============================================================================
//
// WARNING: This header is INTERNAL to the C API binding layer
// NOT part of the public API - do not include from user code
//
// PURPOSE:
//   - Bridge C opaque handles to C++ objects
//   - Provide type-safe wrappers for sparse/dense matrices
//   - Implement exception-to-error-code conversion
//   - Manage lifecycle via Registry
//
// DESIGN:
//   - Wrappers use std::variant for CSR/CSC polymorphism (C++17)
//   - Zero-overhead: wrappers are thin shells around C++ objects
//   - Move semantics: efficient transfer of ownership
//   - Registry integration: automatic memory management
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/type.hpp"
// Note: registry.hpp is included via sparse.hpp

#include <variant>
#include <string>
#include <utility>
#include <type_traits>

namespace scl::binding {

// =============================================================================
// Internal Sparse Matrix Wrapper (supports both CSR and CSC)
// =============================================================================

/// @brief Wrapper for sparse matrices supporting both CSR and CSC formats
/// @details Uses std::variant for runtime polymorphism without virtual dispatch
///          Provides unified interface via visitor pattern
struct SparseWrapper {
    // Storage for either CSR or CSC matrix
    std::variant<CSR, CSC> matrix;
    
    // Format flag (redundant with variant index, but faster to check)
    bool is_csr;
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /// @brief Default constructor (empty invalid matrix)
    constexpr SparseWrapper() noexcept 
        : matrix(), is_csr(true) {}
    
    /// @brief Construct from CSR matrix (move semantics)
    explicit SparseWrapper(CSR&& m) noexcept
        : matrix(std::move(m)), is_csr(true) {}
    
    /// @brief Construct from CSC matrix (move semantics)
    explicit SparseWrapper(CSC&& m) noexcept
        : matrix(std::move(m)), is_csr(false) {}
    
    // Delete copy to prevent accidental copies
    SparseWrapper(const SparseWrapper&) = delete;
    SparseWrapper& operator=(const SparseWrapper&) = delete;
    
    // Move is allowed
    SparseWrapper(SparseWrapper&&) noexcept = default;
    SparseWrapper& operator=(SparseWrapper&&) noexcept = default;
    
    // Default destructor (matrix variant handles cleanup)
    ~SparseWrapper() = default;
    
    // =========================================================================
    // Property Accessors (Zero-Overhead Dispatch via variant.index())
    // =========================================================================
    
    /// @brief Get number of rows
    [[nodiscard]] SCL_FORCE_INLINE auto rows() const noexcept -> Index {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return std::get<CSR>(matrix).rows();
        }
        return std::get<CSC>(matrix).rows();
    }
    
    /// @brief Get number of columns
    [[nodiscard]] SCL_FORCE_INLINE auto cols() const noexcept -> Index {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return std::get<CSR>(matrix).cols();
        }
        return std::get<CSC>(matrix).cols();
    }
    
    /// @brief Get number of non-zeros
    [[nodiscard]] SCL_FORCE_INLINE auto nnz() const noexcept -> Index {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return std::get<CSR>(matrix).nnz();
        }
        return std::get<CSC>(matrix).nnz();
    }
    
    /// @brief Check if matrix is in valid state
    [[nodiscard]] SCL_FORCE_INLINE auto valid() const noexcept -> bool {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return std::get<CSR>(matrix).valid();
        }
        return std::get<CSC>(matrix).valid();
    }
    
    /// @brief Check if matrix is in CSR format
    [[nodiscard]] constexpr auto is_csr_format() const noexcept -> bool {
        return is_csr;
    }
    
    /// @brief Check if matrix is in CSC format
    [[nodiscard]] constexpr auto is_csc_format() const noexcept -> bool {
        return !is_csr;
    }
    
    // =========================================================================
    // Visitor Pattern (Type-Safe Polymorphic Dispatch)
    // =========================================================================
    
    /// @brief Visit pattern for const operations
    /// @tparam Func Callable object accepting const CSR& or const CSC&
    /// @param func Visitor function
    /// @return Result of visitor function
    template <typename Func>
    [[nodiscard]] SCL_FORCE_INLINE auto visit(Func&& func) const 
        -> decltype(auto) {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return func(std::get<CSR>(matrix));
        }
        return func(std::get<CSC>(matrix));
    }
    
    /// @brief Visit pattern for mutable operations
    /// @tparam Func Callable object accepting CSR& or CSC&
    /// @param func Visitor function
    /// @return Result of visitor function
    template <typename Func>
    [[nodiscard]] SCL_FORCE_INLINE auto visit(Func&& func) -> decltype(auto) {
        if (SCL_LIKELY(is_csr)) [[likely]] {
            return func(std::get<CSR>(matrix));
        }
        return func(std::get<CSC>(matrix));
    }
    
    // =========================================================================
    // Type Traits
    // =========================================================================
    
    /// @brief Get reference to CSR matrix (throws if not CSR)
    [[nodiscard]] auto as_csr() -> CSR& {
        SCL_CHECK_ARG(is_csr, "Matrix is not in CSR format");
        return std::get<CSR>(matrix);
    }
    
    /// @brief Get const reference to CSR matrix (throws if not CSR)
    [[nodiscard]] auto as_csr() const -> const CSR& {
        SCL_CHECK_ARG(is_csr, "Matrix is not in CSR format");
        return std::get<CSR>(matrix);
    }
    
    /// @brief Get reference to CSC matrix (throws if not CSC)
    [[nodiscard]] auto as_csc() -> CSC& {
        SCL_CHECK_ARG(!is_csr, "Matrix is not in CSC format");
        return std::get<CSC>(matrix);
    }
    
    /// @brief Get const reference to CSC matrix (throws if not CSC)
    [[nodiscard]] auto as_csc() const -> const CSC& {
        SCL_CHECK_ARG(!is_csr, "Matrix is not in CSC format");
        return std::get<CSC>(matrix);
    }
};

// Static assertions for wrapper properties
static_assert(!std::is_copy_constructible_v<SparseWrapper>, 
              "SparseWrapper must not be copyable");
static_assert(std::is_move_constructible_v<SparseWrapper>, 
              "SparseWrapper must be movable");

// =============================================================================
// Internal Dense View (Pure View - No Memory Ownership)
// =============================================================================

/// @brief View into dense matrix (row-major layout)
/// @details Pure view - never owns data, caller manages lifetime
///          This library does not allocate dense matrices internally
struct DenseView {
    Real* data;           // Pointer to row-major data (external, not owned)
    Index rows;           // Number of rows
    Index cols;           // Number of columns
    Index stride;         // Row stride (for sub-matrices)
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /// @brief Default constructor (empty invalid view)
    constexpr DenseView() noexcept
        : data(nullptr), rows(0), cols(0), stride(0) {}
    
    /// @brief Construct view from existing data (caller manages lifetime)
    constexpr DenseView(Real* ptr, Index r, Index c, Index s) noexcept
        : data(ptr), rows(r), cols(c), stride(s) {}
    
    // Delete copy to prevent accidental copies
    DenseView(const DenseView&) = delete;
    DenseView& operator=(const DenseView&) = delete;
    
    // Move constructor
    DenseView(DenseView&& other) noexcept
        : data(other.data), rows(other.rows), cols(other.cols),
          stride(other.stride) {
        other.data = nullptr;
        other.rows = other.cols = other.stride = 0;
    }
    
    // Move assignment
    auto operator=(DenseView&& other) noexcept -> DenseView& {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
            stride = other.stride;
            other.data = nullptr;
            other.rows = other.cols = other.stride = 0;
        }
        return *this;
    }
    
    // Default destructor (no resources to release - pure view)
    ~DenseView() = default;
    
    // =========================================================================
    // Property Accessors
    // =========================================================================
    
    /// @brief Check if view is in valid state
    [[nodiscard]] constexpr auto valid() const noexcept -> bool {
        return data != nullptr && rows > 0 && cols > 0;
    }
    
    /// @brief Get total number of elements
    [[nodiscard]] constexpr auto size() const noexcept -> Size {
        return static_cast<Size>(rows) * static_cast<Size>(cols);
    }
    
    /// @brief Always returns true (pure view, never owns data)
    [[nodiscard]] constexpr auto is_view() const noexcept -> bool {
        return true;
    }
};

// Static assertions for wrapper properties
static_assert(!std::is_copy_constructible_v<DenseView>, 
              "DenseView must not be copyable");
static_assert(std::is_move_constructible_v<DenseView>, 
              "DenseView must be movable");

// =============================================================================
// Thread-Local Error State Management
// =============================================================================

// Set last error with code and message (noexcept guarantee)
void set_last_error(scl_error_t code, const char* message) noexcept;

// Clear last error state
void clear_last_error() noexcept;

// Get last error message (thread-local)
[[nodiscard]] auto get_last_error_message() noexcept -> const char*;

// Get last error code (thread-local)
[[nodiscard]] auto get_last_error_code() noexcept -> scl_error_t;

// =============================================================================
// Exception Handling
// =============================================================================

// Convert active C++ exception to C error code
// Must be called from within a catch block
// noexcept: all exceptions are caught internally
[[nodiscard]] auto handle_exception() noexcept -> scl_error_t;

// =============================================================================
// Convenience Macros for Error Handling
// =============================================================================

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
// These macros simplify error handling in C API functions
// Macros are necessary here for:
// - Avoiding code duplication across hundreds of functions
// - Automatic return from calling function
// - Preserving line numbers in error messages

/// @brief Check pointer argument and return error if null
/// @param ptr Pointer to check
/// @param msg Error message
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_C_API_CHECK_NULL(ptr, msg) \
    do { \
        if (SCL_UNLIKELY((ptr) == nullptr)) { \
            scl::binding::set_last_error(SCL_ERROR_NULL_POINTER, (msg)); \
            return SCL_ERROR_NULL_POINTER; \
        } \
    } while(0)

/// @brief Check condition and return error if false
/// @param cond Condition to check
/// @param code Error code to return
/// @param msg Error message
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SCL_C_API_CHECK(cond, code, msg) \
    do { \
        if (SCL_UNLIKELY(!(cond))) { \
            scl::binding::set_last_error((code), (msg)); \
            return (code); \
        } \
    } while(0)

/// @brief Try block wrapper for C API functions
/// @details Catches all C++ exceptions and converts to error codes
#define SCL_C_API_TRY try {

/// @brief Catch block wrapper for C API functions
#define SCL_C_API_CATCH \
    } catch (...) { \
        return scl::binding::handle_exception(); \
    }

/// @brief Clear error and return success
#define SCL_C_API_RETURN_OK \
    do { \
        scl::binding::clear_last_error(); \
        return SCL_OK; \
    } while(0)

// NOLINTEND(cppcoreguidelines-macro-usage)

} // namespace scl::binding

// =============================================================================
// Opaque Handle Definitions (C ABI Compatibility)
// =============================================================================

// These structs complete the forward declarations in core.h
// They inherit from the wrappers to ensure proper layout and lifetime
//
// DESIGN RATIONALE:
//   - Using inheritance ensures the handle types are complete types in C++
//   - The C API only sees opaque pointers, never the internal structure
//   - This pattern allows type-safe allocation: new_object<scl_sparse_matrix>()
//   - No downcasting needed: allocated type matches handle type exactly

/// @brief Opaque handle for sparse matrices (C ABI)
/// @details Inherits all functionality from SparseWrapper
///          Must be allocated as scl_sparse_matrix, not as SparseWrapper
struct scl_sparse_matrix : scl::binding::SparseWrapper {
    // Inherit all constructors
    using SparseWrapper::SparseWrapper;
    
    // Default destructor (variant handles cleanup)
    ~scl_sparse_matrix() = default;
    
    // Ensure proper move semantics
    scl_sparse_matrix(scl_sparse_matrix&&) noexcept = default;
    scl_sparse_matrix& operator=(scl_sparse_matrix&&) noexcept = default;
    
    // Delete copy (inherited from SparseWrapper, explicit for clarity)
    scl_sparse_matrix(const scl_sparse_matrix&) = delete;
    scl_sparse_matrix& operator=(const scl_sparse_matrix&) = delete;
};

/// @brief Opaque handle for dense matrices (C ABI)
/// @details Inherits all functionality from DenseView
///          Must be allocated as scl_dense_matrix, not as DenseView
struct scl_dense_matrix : scl::binding::DenseView {
    // Inherit all constructors
    using DenseView::DenseView;
    
    // Default destructor (wrapper handles cleanup)
    ~scl_dense_matrix() = default;
    
    // Ensure proper move semantics
    scl_dense_matrix(scl_dense_matrix&&) noexcept = default;
    scl_dense_matrix& operator=(scl_dense_matrix&&) noexcept = default;
    
    // Delete copy (inherited from DenseView, explicit for clarity)
    scl_dense_matrix(const scl_dense_matrix&) = delete;
    scl_dense_matrix& operator=(const scl_dense_matrix&) = delete;
};

// =============================================================================
// Static Assertions for Type Safety
// =============================================================================

// Verify inheritance relationship (ensures safe pointer conversion)
static_assert(std::is_base_of_v<scl::binding::SparseWrapper, scl_sparse_matrix>,
              "scl_sparse_matrix must inherit from SparseWrapper");
static_assert(std::is_base_of_v<scl::binding::DenseView, scl_dense_matrix>,
              "scl_dense_matrix must inherit from DenseView");

// Verify move semantics are available
static_assert(std::is_move_constructible_v<scl_sparse_matrix>,
              "scl_sparse_matrix must be move constructible");
static_assert(std::is_move_constructible_v<scl_dense_matrix>,
              "scl_dense_matrix must be move constructible");

// Verify copy is disabled (prevents accidental copies)
static_assert(!std::is_copy_constructible_v<scl_sparse_matrix>,
              "scl_sparse_matrix must not be copyable");
static_assert(!std::is_copy_constructible_v<scl_dense_matrix>,
              "scl_dense_matrix must not be copyable");

// NOTE: We intentionally do NOT assert std::is_standard_layout because:
//   1. SparseWrapper contains std::variant which is not standard layout
//   2. The C API only uses opaque pointers - layout is irrelevant to C code
//   3. What matters is that the handle type is the actual allocated type
