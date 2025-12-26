// =============================================================================
/// @file advanced.cpp
/// @brief Advanced Operations (softmax.hpp, mmd.hpp, spatial.hpp, algebra.hpp)
///
/// Provides advanced computational functions including softmax, MMD, and spatial statistics.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/softmax_mapped_impl.hpp"
#include "scl/kernel/algebra_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// Softmax - Standard
// =============================================================================

int scl_softmax_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::softmax::softmax_inplace(matrix);
    )
}

// =============================================================================
// MMD (Maximum Mean Discrepancy) - Standard
// =============================================================================

int scl_mmd_rbf_csc(
    const scl::Real* data_x,
    const scl::Index* indices_x,
    const scl::Index* indptr_x,
    scl::Index rows_x,
    scl::Index cols,
    const scl::Real* data_y,
    const scl::Index* indices_y,
    const scl::Index* indptr_y,
    scl::Index rows_y,
    scl::Real* output,
    scl::Real gamma
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC mat_x(
            const_cast<scl::Real*>(data_x),
            const_cast<scl::Index*>(indices_x),
            const_cast<scl::Index*>(indptr_x),
            rows_x, cols
        );
        scl::CustomCSC mat_y(
            const_cast<scl::Real*>(data_y),
            const_cast<scl::Index*>(indices_y),
            const_cast<scl::Index*>(indptr_y),
            rows_y, cols
        );
        scl::kernel::mmd::mmd_rbf(
            mat_x,
            mat_y,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)),
            gamma
        );
    )
}

// =============================================================================
// Spatial Statistics - Standard
// =============================================================================

int scl_morans_i(
    const scl::Real* graph_data,
    const scl::Index* graph_indices,
    const scl::Index* graph_indptr,
    scl::Index n_cells,
    const scl::Real* features_data,
    const scl::Index* features_indices,
    const scl::Index* features_indptr,
    scl::Index n_genes,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR graph(
            const_cast<scl::Real*>(graph_data),
            const_cast<scl::Index*>(graph_indices),
            const_cast<scl::Index*>(graph_indptr),
            n_cells, n_cells
        );
        scl::CustomCSC features(
            const_cast<scl::Real*>(features_data),
            const_cast<scl::Index*>(features_indices),
            const_cast<scl::Index*>(features_indptr),
            n_cells, n_genes
        );
        scl::kernel::spatial::morans_i(
            graph,
            features,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(n_genes))
        );
    )
}

// =============================================================================
// Linear Algebra - Standard
// =============================================================================

int scl_spmv_csr(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols
        );
        scl::kernel::algebra::spmv(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_cols)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_rows)),
            alpha,
            beta
        );
    )
}

int scl_spmv_trans_csc(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols
        );
        scl::kernel::algebra::spmv(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_rows)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_cols)),
            alpha,
            beta
        );
    )
}

// =============================================================================
// Softmax - Mapped
// =============================================================================

int scl_softmax_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_data,
    scl::Index* out_indices,
    scl::Index* out_indptr
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(rows + 1));
        
        scl::io::MappedCustomSparse<scl::Real, true> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        auto result = scl::kernel::softmax::mapped::softmax_mapped_custom(matrix);
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

// =============================================================================
// Linear Algebra - Mapped
// =============================================================================

int scl_spmv_mapped_csr(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    scl::Index A_nnz,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(A_data), static_cast<scl::Size>(A_nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(A_indices), static_cast<scl::Size>(A_nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(A_indptr), static_cast<scl::Size>(A_rows + 1));
        
        scl::io::MappedCustomSparse<scl::Real, true> A(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            A_rows, A_cols
        );
        
        scl::kernel::algebra::mapped::spmv_mapped(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_cols)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_rows)),
            alpha,
            beta
        );
    )
}

int scl_spmv_trans_mapped_csc(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    scl::Index A_nnz,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(A_data), static_cast<scl::Size>(A_nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(A_indices), static_cast<scl::Size>(A_nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(A_indptr), static_cast<scl::Size>(A_cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> A(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            A_rows, A_cols
        );
        
        scl::kernel::algebra::mapped::spmv_mapped(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_rows)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_cols)),
            alpha,
            beta
        );
    )
}

} // extern "C"
