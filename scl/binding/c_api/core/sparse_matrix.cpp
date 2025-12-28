// =============================================================================
// FILE: scl/binding/c_api/sparse_matrix.cpp
// BRIEF: Sparse matrix C API implementation
// =============================================================================

#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"

#include <span>
#include <vector>

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Lifecycle Management
// =============================================================================

scl_error_t scl_sparse_create(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    int is_csr)
{
    if (!out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Output pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!indptr || !indices || !data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Input data pointers are null");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = (is_csr != 0);
        
        Index pdim = is_csr ? rows : cols;
        std::span<const Index> offsets_span(
            reinterpret_cast<const Index*>(indptr), 
            static_cast<size_t>(pdim + 1)
        );
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(indices), 
            static_cast<size_t>(nnz)
        );
        std::span<const Real> data_span(
            reinterpret_cast<const Real*>(data), 
            static_cast<size_t>(nnz)
        );
        
        if (is_csr) {
            wrapper->matrix = CSR::from_traditional(
                rows, cols, data_span, indices_span, offsets_span
            );
        } else {
            wrapper->matrix = CSC::from_traditional(
                rows, cols, data_span, indices_span, offsets_span
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_wrap(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    int is_csr)
{
    if (!out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Output pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!indptr || !indices || !data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Input data pointers are null");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = (is_csr != 0);
        
        Index pdim = is_csr ? rows : cols;
        std::span<const Index> offsets_span(
            reinterpret_cast<Index*>(indptr), 
            static_cast<size_t>(pdim + 1)
        );
        
        if (is_csr) {
            wrapper->matrix = CSR::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        } else {
            wrapper->matrix = CSC::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to wrap sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_clone(scl_sparse_t src, scl_sparse_t* out) {
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = src->is_csr;
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).clone();
        } else {
            wrapper->matrix = std::get<CSC>(src->matrix).clone();
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to clone sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_destroy(scl_sparse_t* matrix) {
    if (!matrix || !*matrix) {
        return SCL_OK;  // Already null, nothing to do
    }
    
    try {
        auto& reg = get_registry();
        reg.unregister_ptr(*matrix);
        *matrix = nullptr;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Property Queries
// =============================================================================

scl_error_t scl_sparse_rows(scl_sparse_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        *out = static_cast<scl_index_t>(matrix->rows());
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_cols(scl_sparse_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        *out = static_cast<scl_index_t>(matrix->cols());
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_nnz(scl_sparse_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        *out = static_cast<scl_index_t>(matrix->nnz());
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_is_csr(scl_sparse_t matrix, int* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        *out = matrix->is_csr ? 1 : 0;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_is_valid(scl_sparse_t matrix, int* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        *out = matrix->valid() ? 1 : 0;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Data Export
// =============================================================================

scl_error_t scl_sparse_export(
    scl_sparse_t matrix,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data)
{
    if (!matrix || !indptr || !indices || !data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto trad = matrix->visit([](auto& m) {
            return m.to_traditional();
        });
        
        Index pdim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index nnz = matrix->nnz();
        
        for (Index i = 0; i <= pdim; ++i) {
            indptr[i] = static_cast<scl_index_t>(trad.offsets[i]);
        }
        
        for (Index i = 0; i < nnz; ++i) {
            indices[i] = static_cast<scl_index_t>(trad.indices[i]);
            data[i] = static_cast<scl_real_t>(trad.values[i]);
        }
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Format Conversion
// =============================================================================

scl_error_t scl_sparse_transpose(scl_sparse_t src, scl_sparse_t* out) {
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = !src->is_csr;
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).transpose();
        } else {
            wrapper->matrix = std::get<CSC>(src->matrix).transpose();
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to transpose sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_to_contiguous(scl_sparse_t src, scl_sparse_t* out) {
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = src->is_csr;
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).to_contiguous();
        } else {
            wrapper->matrix = std::get<CSC>(src->matrix).to_contiguous();
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to convert to contiguous");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

