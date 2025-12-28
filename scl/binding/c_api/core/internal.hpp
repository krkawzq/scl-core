#pragma once

// =============================================================================
// FILE: scl/binding/c_api/internal.hpp
// BRIEF: Internal C++ wrapper structures for C API (NOT part of public API)
// =============================================================================

#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"
#include "scl/core/type.hpp"
#include "scl/binding/c_api/core/core.h"

#include <variant>
#include <string>

namespace scl::binding {

// =============================================================================
// Internal Wrapper for Sparse Matrix (supports both CSR and CSC)
// =============================================================================

struct SparseWrapper {
    std::variant<CSR, CSC> matrix;
    bool is_csr;
    
    SparseWrapper() : is_csr(true) {}
    
    explicit SparseWrapper(CSR&& m) 
        : matrix(std::move(m)), is_csr(true) {}
    
    explicit SparseWrapper(CSC&& m) 
        : matrix(std::move(m)), is_csr(false) {}
    
    // Helper accessors
    Index rows() const {
        if (is_csr) {
            return std::get<CSR>(matrix).rows();
        } else {
            return std::get<CSC>(matrix).rows();
        }
    }
    
    Index cols() const {
        if (is_csr) {
            return std::get<CSR>(matrix).cols();
        } else {
            return std::get<CSC>(matrix).cols();
        }
    }
    
    Index nnz() const {
        if (is_csr) {
            return std::get<CSR>(matrix).nnz();
        } else {
            return std::get<CSC>(matrix).nnz();
        }
    }
    
    bool valid() const {
        if (is_csr) {
            return std::get<CSR>(matrix).valid();
        } else {
            return std::get<CSC>(matrix).valid();
        }
    }
    
    // Dynamic dispatch template
    template<typename Func>
    auto visit(Func&& func) -> decltype(auto) {
        if (is_csr) {
            return func(std::get<CSR>(matrix));
        } else {
            return func(std::get<CSC>(matrix));
        }
    }
    
    template<typename Func>
    auto visit(Func&& func) const -> decltype(auto) {
        if (is_csr) {
            return func(std::get<CSR>(matrix));
        } else {
            return func(std::get<CSC>(matrix));
        }
    }
};

// =============================================================================
// Thread-Local Error State
// =============================================================================

void set_last_error(scl_error_t code, const char* message);
void clear_last_error();
const char* get_last_error_message();

// =============================================================================
// Exception to Error Code Conversion
// =============================================================================

scl_error_t handle_exception();

} // namespace scl::binding

// =============================================================================
// Opaque Handle Definitions
// =============================================================================

// The opaque handle points to this wrapper
struct scl_sparse_matrix : scl::binding::SparseWrapper {};

// Dense matrix wrapper (placeholder for now)
struct scl_dense_matrix {
    // Will be implemented in Phase 5
};
