// =============================================================================
// FILE: scl/binding/c_api/sparse_matrix.cpp
// BRIEF: Sparse matrix C API implementation
// =============================================================================

#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"
#include "scl/kernel/sparse.hpp"

#include <span>
#include <vector>

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions (C++ scope)
// =============================================================================

namespace {

BlockStrategy convert_block_strategy(scl_block_strategy_t strategy) {
    switch (strategy) {
        case SCL_BLOCK_STRATEGY_CONTIGUOUS:
            return BlockStrategy::contiguous();
        case SCL_BLOCK_STRATEGY_SMALL:
            return BlockStrategy::small_blocks();
        case SCL_BLOCK_STRATEGY_LARGE:
            return BlockStrategy::large_blocks();
        case SCL_BLOCK_STRATEGY_ADAPTIVE:
        default:
            return BlockStrategy::adaptive();
    }
}

} // anonymous namespace

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
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
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
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
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
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
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
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
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
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Advanced Lifecycle Management
// =============================================================================

scl_error_t scl_sparse_wrap_and_own(
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
        
        Index pdim = is_csr ? rows : cols;
        
        // Register data arrays with registry as buffer+aliases
        BufferID data_buf = reg.create_buffer(
            data,
            static_cast<std::size_t>(nnz) * sizeof(Real),
            AllocType::ArrayNew
        );
        BufferID indices_buf = reg.create_buffer(
            indices,
            static_cast<std::size_t>(nnz) * sizeof(Index),
            AllocType::ArrayNew
        );
        BufferID indptr_buf = reg.create_buffer(
            indptr,
            static_cast<std::size_t>(pdim + 1) * sizeof(Index),
            AllocType::ArrayNew
        );
        
        if (!data_buf || !indices_buf || !indptr_buf) {
            // Buffer IDs are automatically managed, no need to manually decref here
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to register buffers");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        // Create aliases for each row/column's data slice
        for (Index i = 0; i < pdim; ++i) {
            Index start = indptr[i];
            Index len = indptr[i + 1] - start;
            if (len > 0) {
                Real* data_ptr = data + start;
                Index* idx_ptr = indices + start;
                
                reg.create_alias(data_ptr, data_buf, start * sizeof(Real));
                reg.create_alias(idx_ptr, indices_buf, start * sizeof(Index));
            }
        }
        
        // Create wrapper
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = (is_csr != 0);
        
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
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_slice_rows(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_sparse_t* out)
{
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!row_indices && n_rows > 0) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Row indices are null");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!src->is_csr) {
            set_last_error(SCL_ERROR_TYPE_MISMATCH, 
                "Row slicing requires CSR format - use scl_sparse_slice_cols for CSC");
            return SCL_ERROR_TYPE_MISMATCH;
        }
        
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = true;
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(row_indices),
            n_rows
        );
        
        wrapper->matrix = std::get<CSR>(src->matrix).row_slice_view(indices_span);
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create row slice");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_slice_cols(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    scl_size_t n_cols,
    scl_sparse_t* out)
{
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!col_indices && n_cols > 0) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Column indices are null");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (src->is_csr) {
            set_last_error(SCL_ERROR_TYPE_MISMATCH,
                "Column slicing requires CSC format - use scl_sparse_slice_rows for CSR");
            return SCL_ERROR_TYPE_MISMATCH;
        }
        
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = false;
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(col_indices),
            n_cols
        );
        
        wrapper->matrix = std::get<CSC>(src->matrix).col_slice_view(indices_span);
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create column slice");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Block Strategy Support
// =============================================================================

scl_error_t scl_sparse_create_with_strategy(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    int is_csr,
    scl_block_strategy_t strategy)
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
        
        BlockStrategy block_strat = convert_block_strategy(strategy);
        
        if (is_csr) {
            wrapper->matrix = CSR::from_traditional(
                rows, cols, data_span, indices_span, offsets_span, block_strat
            );
        } else {
            wrapper->matrix = CSC::from_traditional(
                rows, cols, data_span, indices_span, offsets_span, block_strat
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create sparse matrix");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// COO Format Support
// =============================================================================

scl_error_t scl_sparse_from_coo(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* row_indices,
    const scl_index_t* col_indices,
    const scl_real_t* values,
    int is_csr,
    scl_block_strategy_t strategy)
{
    if (!out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Output pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!row_indices || !col_indices || !values) {
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
        
        std::span<const Index> row_span(
            reinterpret_cast<const Index*>(row_indices), 
            static_cast<size_t>(nnz)
        );
        std::span<const Index> col_span(
            reinterpret_cast<const Index*>(col_indices), 
            static_cast<size_t>(nnz)
        );
        std::span<const Real> val_span(
            reinterpret_cast<const Real*>(values), 
            static_cast<size_t>(nnz)
        );
        
        BlockStrategy block_strat = convert_block_strategy(strategy);
        
        if (is_csr) {
            wrapper->matrix = CSR::from_coo(
                rows, cols, row_span, col_span, val_span, block_strat
            );
        } else {
            wrapper->matrix = CSC::from_coo(
                rows, cols, row_span, col_span, val_span, block_strat
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create sparse matrix from COO");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_to_coo(
    scl_sparse_t matrix,
    scl_index_t** row_indices,
    scl_index_t** col_indices,
    scl_real_t** values,
    scl_index_t* nnz)
{
    if (!matrix || !row_indices || !col_indices || !values || !nnz) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        wrapper->visit([&](const auto& m) {
            auto coo = scl::kernel::sparse::to_coo_arrays(m);
            
            *row_indices = reinterpret_cast<scl_index_t*>(coo.row_indices);
            *col_indices = reinterpret_cast<scl_index_t*>(coo.col_indices);
            *values = reinterpret_cast<scl_real_t*>(coo.values);
            *nnz = coo.nnz;
        });
        
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Layout Information
// =============================================================================

scl_error_t scl_sparse_layout_info(
    scl_sparse_t matrix,
    scl_sparse_layout_info_t* info)
{
    if (!matrix || !info) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        wrapper->visit([&](const auto& m) {
            auto layout = m.layout_info();
            
            info->data_block_count = layout.data_block_count;
            info->index_block_count = layout.index_block_count;
            info->data_bytes = layout.data_bytes;
            info->index_bytes = layout.index_bytes;
            info->metadata_bytes = layout.metadata_bytes;
            info->is_contiguous = layout.is_contiguous ? 1 : 0;
            info->is_traditional_format = layout.is_traditional_format ? 1 : 0;
        });
        
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_is_contiguous(
    scl_sparse_t matrix,
    int* out)
{
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        wrapper->visit([&](const auto& m) {
            *out = m.is_contiguous() ? 1 : 0;
        });
        
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Advanced Slicing
// =============================================================================

scl_error_t scl_sparse_row_range_view(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_sparse_t* out)
{
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
            wrapper->matrix = std::get<CSR>(src->matrix).row_range_view(start, end);
        } else {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, 
                          "row_range_view only supported for CSR matrices");
            reg.unregister_ptr(wrapper);
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create row range view");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_row_range_copy(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_block_strategy_t strategy,
    scl_sparse_t* out)
{
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
        BlockStrategy block_strat = convert_block_strategy(strategy);
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).row_range_copy(
                start, end, block_strat
            );
        } else {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, 
                          "row_range_copy only supported for CSR matrices");
            reg.unregister_ptr(wrapper);
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create row range copy");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_row_slice_copy(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_block_strategy_t strategy,
    scl_sparse_t* out)
{
    if (!src || !out || !row_indices) {
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
        BlockStrategy block_strat = convert_block_strategy(strategy);
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(row_indices),
            n_rows
        );
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).row_slice_copy(
                indices_span, block_strat
            );
        } else {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, 
                          "row_slice_copy only supported for CSR matrices");
            reg.unregister_ptr(wrapper);
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create row slice copy");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_col_slice(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    scl_size_t n_cols,
    scl_block_strategy_t strategy,
    scl_sparse_t* out)
{
    if (!src || !out || !col_indices) {
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
        BlockStrategy block_strat = convert_block_strategy(strategy);
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(col_indices),
            n_cols
        );
        
        if (src->is_csr) {
            wrapper->matrix = std::get<CSR>(src->matrix).col_slice(
                indices_span, block_strat
            );
        } else {
            wrapper->matrix = std::get<CSC>(src->matrix).col_slice_copy(
                indices_span, block_strat
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create column slice");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Batch Operations
// =============================================================================

scl_error_t scl_sparse_vstack(
    const scl_sparse_t* matrices,
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out)
{
    if (!matrices || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_matrices == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Empty matrix array");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto& reg = get_registry();
        
        // Check all matrices are CSR
        for (size_t i = 0; i < n_matrices; ++i) {
            if (!matrices[i]) {
                set_last_error(SCL_ERROR_NULL_POINTER, "Null matrix in array");
                return SCL_ERROR_NULL_POINTER;
            }
            if (!matrices[i]->is_csr) {
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, 
                              "vstack requires all matrices to be CSR");
                return SCL_ERROR_INVALID_ARGUMENT;
            }
        }
        
        // Collect CSR matrices
        std::vector<CSR> csr_vec;
        csr_vec.reserve(n_matrices);
        for (size_t i = 0; i < n_matrices; ++i) {
            csr_vec.push_back(std::get<CSR>(matrices[i]->matrix));
        }
        
        BlockStrategy block_strat = convert_block_strategy(strategy);
        CSR result = scl::vstack<Real>(csr_vec, block_strat);
        
        if (!result.valid()) {
            set_last_error(SCL_ERROR_INTERNAL, "Failed to vstack matrices");
            return SCL_ERROR_INTERNAL;
        }
        
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = true;
        wrapper->matrix = std::move(result);
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_hstack(
    const scl_sparse_t* matrices,
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out)
{
    if (!matrices || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_matrices == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Empty matrix array");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto& reg = get_registry();
        
        // Check all matrices are CSC
        for (size_t i = 0; i < n_matrices; ++i) {
            if (!matrices[i]) {
                set_last_error(SCL_ERROR_NULL_POINTER, "Null matrix in array");
                return SCL_ERROR_NULL_POINTER;
            }
            if (matrices[i]->is_csr) {
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, 
                              "hstack requires all matrices to be CSC");
                return SCL_ERROR_INVALID_ARGUMENT;
            }
        }
        
        // Collect CSC matrices
        std::vector<CSC> csc_vec;
        csc_vec.reserve(n_matrices);
        for (size_t i = 0; i < n_matrices; ++i) {
            csc_vec.push_back(std::get<CSC>(matrices[i]->matrix));
        }
        
        BlockStrategy block_strat = convert_block_strategy(strategy);
        CSC result = scl::hstack<Real>(csc_vec, block_strat);
        
        if (!result.valid()) {
            set_last_error(SCL_ERROR_INTERNAL, "Failed to hstack matrices");
            return SCL_ERROR_INTERNAL;
        }
        
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = false;
        wrapper->matrix = std::move(result);
        
        *out = reinterpret_cast<scl_sparse_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

