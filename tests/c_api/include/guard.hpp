#pragma once

// =============================================================================
// SCL Core - RAII Guards for C API Handles
// =============================================================================
//
// Smart RAII wrappers for C API opaque handles, ensuring automatic cleanup.
//
// Usage:
//   Sparse mat;
//   scl_sparse_create(mat.ptr(), ...);
//   // Automatically destroyed when mat goes out of scope
//
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/dense.h"

#include <utility>

namespace scl::test {

// =============================================================================
// Sparse Matrix Guard
// =============================================================================

/// RAII wrapper for scl_sparse_t
class Sparse {
public:
    Sparse() = default;
    
    explicit Sparse(scl_sparse_t h) : handle_(h) {}
    
    // Non-copyable
    Sparse(const Sparse&) = delete;
    Sparse& operator=(const Sparse&) = delete;
    
    // Movable
    Sparse(Sparse&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    Sparse& operator=(Sparse&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    ~Sparse() {
        reset();
    }
    
    /// Get raw handle (const)
    [[nodiscard]] scl_sparse_t get() const noexcept {
        return handle_;
    }
    
    /// Get pointer to handle (for output parameters)
    [[nodiscard]] scl_sparse_t* ptr() noexcept {
        return &handle_;
    }
    
    /// Release ownership and return raw handle
    [[nodiscard]] scl_sparse_t release() noexcept {
        scl_sparse_t h = handle_;
        handle_ = nullptr;
        return h;
    }
    
    /// Reset to new handle (destroys old one)
    void reset(scl_sparse_t h = nullptr) {
        if (handle_ && handle_ != h) {
            scl_sparse_destroy(&handle_);
        }
        handle_ = h;
    }
    
    /// Check if valid
    [[nodiscard]] bool valid() const noexcept {
        return handle_ != nullptr;
    }
    
    /// Implicit conversion to raw handle (for function calls)
    operator scl_sparse_t() const noexcept {
        return handle_;
    }
    
    /// Boolean conversion
    explicit operator bool() const noexcept {
        return valid();
    }

private:
    scl_sparse_t handle_ = nullptr;
};

// =============================================================================
// Dense Matrix Guard
// =============================================================================

/// RAII wrapper for scl_dense_t
class Dense {
public:
    Dense() = default;
    
    explicit Dense(scl_dense_t h) : handle_(h) {}
    
    // Non-copyable
    Dense(const Dense&) = delete;
    Dense& operator=(const Dense&) = delete;
    
    // Movable
    Dense(Dense&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    Dense& operator=(Dense&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    ~Dense() {
        reset();
    }
    
    [[nodiscard]] scl_dense_t get() const noexcept {
        return handle_;
    }
    
    [[nodiscard]] scl_dense_t* ptr() noexcept {
        return &handle_;
    }
    
    [[nodiscard]] scl_dense_t release() noexcept {
        scl_dense_t h = handle_;
        handle_ = nullptr;
        return h;
    }
    
    void reset(scl_dense_t h = nullptr) {
        if (handle_ && handle_ != h) {
            scl_dense_destroy(&handle_);
        }
        handle_ = h;
    }
    
    [[nodiscard]] bool valid() const noexcept {
        return handle_ != nullptr;
    }
    
    operator scl_dense_t() const noexcept {
        return handle_;
    }
    
    explicit operator bool() const noexcept {
        return valid();
    }

private:
    scl_dense_t handle_ = nullptr;
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Create sparse matrix from arrays with automatic cleanup
inline Sparse make_sparse_csr(
    scl_index_t rows, scl_index_t cols, scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data
) {
    Sparse mat;
    scl_error_t err = scl_sparse_create(
        mat.ptr(), rows, cols, nnz,
        indptr, indices, data, SCL_TRUE
    );
    if (err != SCL_OK) {
        throw std::runtime_error(
            std::string("Failed to create sparse matrix: ") + scl_get_last_error()
        );
    }
    return mat;
}

/// Create sparse matrix with strategy
inline Sparse make_sparse_csr_with_strategy(
    scl_index_t rows, scl_index_t cols, scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    scl_block_strategy_t strategy
) {
    Sparse mat;
    scl_error_t err = scl_sparse_create_with_strategy(
        mat.ptr(), rows, cols, nnz,
        indptr, indices, data, SCL_TRUE, strategy
    );
    if (err != SCL_OK) {
        throw std::runtime_error(
            std::string("Failed to create sparse matrix with strategy: ") + scl_get_last_error()
        );
    }
    return mat;
}

/// Wrap sparse matrix (zero-copy)
inline Sparse wrap_sparse_csr(
    scl_index_t rows, scl_index_t cols, scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data
) {
    Sparse mat;
    scl_error_t err = scl_sparse_wrap(
        mat.ptr(), rows, cols, nnz,
        indptr, indices, data, SCL_TRUE
    );
    if (err != SCL_OK) {
        throw std::runtime_error(
            std::string("Failed to wrap sparse matrix: ") + scl_get_last_error()
        );
    }
    return mat;
}

/// Wrap dense matrix (view)
inline Dense wrap_dense(
    scl_index_t rows, scl_index_t cols,
    scl_real_t* data,
    scl_index_t stride = 0
) {
    Dense mat;
    if (stride == 0) stride = cols;
    scl_error_t err = scl_dense_wrap(mat.ptr(), rows, cols, data, stride);
    if (err != SCL_OK) {
        throw std::runtime_error(
            std::string("Failed to wrap dense matrix: ") + scl_get_last_error()
        );
    }
    return mat;
}

} // namespace scl::test

