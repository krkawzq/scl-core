#pragma once

// =============================================================================
// SCL Core - Oracle (Eigen Reference Implementation)
// =============================================================================
//
// Eigen-based reference implementation for correctness verification.
// "Oracle" = source of truth for numerical results.
//
// Provides:
//   - Conversion between SCL C API and Eigen types
//   - Reference implementations of operations
//   - Numerical comparison utilities
//
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/dense.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <vector>
#include <cmath>
#include <algorithm>

namespace scl::test {

// =============================================================================
// Type Aliases
// =============================================================================

using EigenCSR = Eigen::SparseMatrix<scl_real_t, Eigen::RowMajor, scl_index_t>;
using EigenCSC = Eigen::SparseMatrix<scl_real_t, Eigen::ColMajor, scl_index_t>;
using EigenDense = Eigen::Matrix<scl_real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenVector = Eigen::Matrix<scl_real_t, Eigen::Dynamic, 1>;

// =============================================================================
// Conversion: C API → Eigen
// =============================================================================

/// Convert SCL sparse matrix to Eigen CSR
inline EigenCSR to_eigen_csr(scl_sparse_t mat) {
    scl_index_t rows, cols, nnz;
    scl_bool_t is_csr;
    
    scl_sparse_rows(mat, &rows);
    scl_sparse_cols(mat, &cols);
    scl_sparse_nnz(mat, &nnz);
    scl_sparse_is_csr(mat, &is_csr);
    
    if (!is_csr) {
        throw std::runtime_error("Expected CSR matrix");
    }
    
    // Export data
    std::vector<scl_index_t> indptr(rows + 1);
    std::vector<scl_index_t> indices(nnz);
    std::vector<scl_real_t> data(nnz);
    
    scl_sparse_export(mat, indptr.data(), indices.data(), data.data());
    
    // Build Eigen matrix
    EigenCSR result(rows, cols);
    result.reserve(Eigen::VectorXi::Constant(rows, nnz / rows + 1));
    
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = indptr[i]; j < indptr[i + 1]; ++j) {
            result.insert(i, indices[j]) = data[j];
        }
    }
    
    result.makeCompressed();
    return result;
}

/// Convert SCL sparse matrix to Eigen CSC
inline EigenCSC to_eigen_csc(scl_sparse_t mat) {
    scl_index_t rows, cols, nnz;
    scl_bool_t is_csr;
    
    scl_sparse_rows(mat, &rows);
    scl_sparse_cols(mat, &cols);
    scl_sparse_nnz(mat, &nnz);
    scl_sparse_is_csr(mat, &is_csr);
    
    if (is_csr) {
        throw std::runtime_error("Expected CSC matrix");
    }
    
    // Export data
    std::vector<scl_index_t> indptr(cols + 1);
    std::vector<scl_index_t> indices(nnz);
    std::vector<scl_real_t> data(nnz);
    
    scl_sparse_export(mat, indptr.data(), indices.data(), data.data());
    
    // Build Eigen matrix
    EigenCSC result(rows, cols);
    result.reserve(Eigen::VectorXi::Constant(cols, nnz / cols + 1));
    
    for (scl_index_t j = 0; j < cols; ++j) {
        for (scl_index_t k = indptr[j]; k < indptr[j + 1]; ++k) {
            result.insert(indices[k], j) = data[k];
        }
    }
    
    result.makeCompressed();
    return result;
}

/// Convert SCL dense matrix to Eigen (creates a copy)
inline EigenDense to_eigen_dense(scl_dense_t mat) {
    scl_index_t rows, cols, stride;
    
    scl_dense_rows(mat, &rows);
    scl_dense_cols(mat, &cols);
    scl_dense_stride(mat, &stride);
    
    EigenDense result(rows, cols);
    
    // Copy data
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            scl_real_t value;
            scl_dense_get(mat, i, j, &value);
            result(i, j) = value;
        }
    }
    
    return result;
}

// =============================================================================
// Conversion: Eigen → C API Data
// =============================================================================

/// Extract CSR arrays from Eigen sparse matrix
struct CSRArrays {
    std::vector<scl_index_t> indptr;
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    scl_index_t rows;
    scl_index_t cols;
    scl_index_t nnz;
};

inline CSRArrays from_eigen_csr(const EigenCSR& mat) {
    CSRArrays result;
    result.rows = mat.rows();
    result.cols = mat.cols();
    result.nnz = mat.nonZeros();
    
    result.indptr.resize(result.rows + 1);
    result.indices.reserve(result.nnz);
    result.data.reserve(result.nnz);
    
    result.indptr[0] = 0;
    for (scl_index_t i = 0; i < result.rows; ++i) {
        for (typename EigenCSR::InnerIterator it(mat, i); it; ++it) {
            result.indices.push_back(it.col());
            result.data.push_back(it.value());
        }
        result.indptr[i + 1] = static_cast<scl_index_t>(result.indices.size());
    }
    
    return result;
}

/// Extract CSC arrays from Eigen sparse matrix
struct CSCArrays {
    std::vector<scl_index_t> indptr;
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    scl_index_t rows;
    scl_index_t cols;
    scl_index_t nnz;
};

inline CSCArrays from_eigen_csc(const EigenCSC& mat) {
    CSCArrays result;
    result.rows = mat.rows();
    result.cols = mat.cols();
    result.nnz = mat.nonZeros();
    
    result.indptr.resize(result.cols + 1);
    result.indices.reserve(result.nnz);
    result.data.reserve(result.nnz);
    
    result.indptr[0] = 0;
    for (scl_index_t j = 0; j < result.cols; ++j) {
        for (typename EigenCSC::InnerIterator it(mat, j); it; ++it) {
            result.indices.push_back(it.row());
            result.data.push_back(it.value());
        }
        result.indptr[j + 1] = static_cast<scl_index_t>(result.indices.size());
    }
    
    return result;
}

// =============================================================================
// Reference Operations (Eigen-based)
// =============================================================================

namespace oracle {

/// Transpose CSR → CSC (using Eigen)
inline EigenCSC transpose_csr_to_csc(const EigenCSR& mat) {
    return mat.transpose();
}

/// Transpose CSC → CSR (using Eigen)
inline EigenCSR transpose_csc_to_csr(const EigenCSC& mat) {
    return mat.transpose();
}

/// Clone sparse matrix
inline EigenCSR clone_csr(const EigenCSR& mat) {
    return EigenCSR(mat);
}

inline EigenCSC clone_csc(const EigenCSC& mat) {
    return EigenCSC(mat);
}

/// SpMV: y = alpha * A * x + beta * y
inline void spmv_csr(
    const EigenCSR& A,
    const Eigen::Ref<const EigenVector>& x,
    Eigen::Ref<EigenVector> y,
    scl_real_t alpha = 1.0,
    scl_real_t beta = 0.0
) {
    if (beta == 0.0) {
        y = alpha * (A * x);
    } else {
        y = alpha * (A * x) + beta * y;
    }
}

/// SpMV: y = alpha * A^T * x + beta * y
inline void spmv_csr_transpose(
    const EigenCSR& A,
    const Eigen::Ref<const EigenVector>& x,
    Eigen::Ref<EigenVector> y,
    scl_real_t alpha = 1.0,
    scl_real_t beta = 0.0
) {
    if (beta == 0.0) {
        y = alpha * (A.transpose() * x);
    } else {
        y = alpha * (A.transpose() * x) + beta * y;
    }
}

/// Element-wise operations
inline EigenCSR add_csr(const EigenCSR& A, const EigenCSR& B) {
    return A + B;
}

inline EigenCSR subtract_csr(const EigenCSR& A, const EigenCSR& B) {
    return A - B;
}

inline EigenCSR scale_csr(const EigenCSR& A, scl_real_t alpha) {
    return alpha * A;
}

/// Matrix norms
inline scl_real_t frobenius_norm_csr(const EigenCSR& mat) {
    return mat.norm();
}

inline scl_real_t max_abs_csr(const EigenCSR& mat) {
    scl_real_t max_val = 0;
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (typename EigenCSR::InnerIterator it(mat, k); it; ++it) {
            max_val = std::max(max_val, std::abs(it.value()));
        }
    }
    return max_val;
}

} // namespace oracle

// =============================================================================
// Numerical Comparison Utilities
// =============================================================================

/// Compare two sparse matrices (CSR format)
inline bool matrices_equal(
    const EigenCSR& A,
    const EigenCSR& B,
    scl_real_t rtol = 1e-9,
    scl_real_t atol = 1e-12
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    
    if (A.nonZeros() != B.nonZeros()) {
        return false;
    }
    
    EigenCSR diff = A - B;
    scl_real_t max_diff = 0;
    scl_real_t max_val = std::max(oracle::max_abs_csr(A), oracle::max_abs_csr(B));
    
    for (int k = 0; k < diff.outerSize(); ++k) {
        for (typename EigenCSR::InnerIterator it(diff, k); it; ++it) {
            max_diff = std::max(max_diff, std::abs(it.value()));
        }
    }
    
    return max_diff <= atol + rtol * max_val;
}

/// Compare two sparse matrices (CSC format)
inline bool matrices_equal(
    const EigenCSC& A,
    const EigenCSC& B,
    scl_real_t rtol = 1e-9,
    scl_real_t atol = 1e-12
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    
    if (A.nonZeros() != B.nonZeros()) {
        return false;
    }
    
    EigenCSC diff = A - B;
    scl_real_t max_diff = 0;
    scl_real_t max_val = 0;
    
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename EigenCSC::InnerIterator it(A, k); it; ++it) {
            max_val = std::max(max_val, std::abs(it.value()));
        }
        for (typename EigenCSC::InnerIterator it(B, k); it; ++it) {
            max_val = std::max(max_val, std::abs(it.value()));
        }
    }
    
    for (int k = 0; k < diff.outerSize(); ++k) {
        for (typename EigenCSC::InnerIterator it(diff, k); it; ++it) {
            max_diff = std::max(max_diff, std::abs(it.value()));
        }
    }
    
    return max_diff <= atol + rtol * max_val;
}

/// Compare dense matrices
inline bool matrices_equal(
    const EigenDense& A,
    const EigenDense& B,
    scl_real_t rtol = 1e-9,
    scl_real_t atol = 1e-12
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    
    scl_real_t max_val = std::max(A.cwiseAbs().maxCoeff(), B.cwiseAbs().maxCoeff());
    scl_real_t max_diff = (A - B).cwiseAbs().maxCoeff();
    
    return max_diff <= atol + rtol * max_val;
}

/// Compare vectors
inline bool vectors_equal(
    const EigenVector& a,
    const EigenVector& b,
    scl_real_t rtol = 1e-9,
    scl_real_t atol = 1e-12
) {
    if (a.size() != b.size()) {
        return false;
    }
    
    scl_real_t max_val = std::max(a.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff());
    scl_real_t max_diff = (a - b).cwiseAbs().maxCoeff();
    
    return max_diff <= atol + rtol * max_val;
}

/// Check if matrix is approximately zero
inline bool is_zero(const EigenCSR& mat, scl_real_t atol = 1e-12) {
    return oracle::max_abs_csr(mat) <= atol;
}

inline bool is_zero(const EigenDense& mat, scl_real_t atol = 1e-12) {
    return mat.cwiseAbs().maxCoeff() <= atol;
}

/// Check if vector is approximately zero
inline bool is_zero(const EigenVector& vec, scl_real_t atol = 1e-12) {
    return vec.cwiseAbs().maxCoeff() <= atol;
}

// =============================================================================
// Matrix Information
// =============================================================================

/// Print sparse matrix info
inline void print_sparse_info(const EigenCSR& mat, const char* name = "Matrix") {
    std::printf("  %s: %lldx%lld, nnz=%lld, density=%.4f%%\n",
        name,
        static_cast<long long>(mat.rows()),
        static_cast<long long>(mat.cols()),
        static_cast<long long>(mat.nonZeros()),
        100.0 * mat.nonZeros() / (static_cast<double>(mat.rows()) * mat.cols())
    );
}

/// Print dense matrix info
inline void print_dense_info(const EigenDense& mat, const char* name = "Matrix") {
    std::printf("  %s: %lldx%lld\n",
        name,
        static_cast<long long>(mat.rows()),
        static_cast<long long>(mat.cols())
    );
}

/// Print matrix difference statistics
inline void print_diff_stats(const EigenCSR& A, const EigenCSR& B, const char* name = "Diff") {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::printf("  %s: dimension mismatch (%lldx%lld vs %lldx%lld)\n",
            name,
            static_cast<long long>(A.rows()), static_cast<long long>(A.cols()),
            static_cast<long long>(B.rows()), static_cast<long long>(B.cols())
        );
        return;
    }
    
    EigenCSR diff = A - B;
    scl_real_t max_diff = oracle::max_abs_csr(diff);
    scl_real_t frob_diff = diff.norm();
    scl_real_t max_a = oracle::max_abs_csr(A);
    scl_real_t max_b = oracle::max_abs_csr(B);
    scl_real_t rel_err = max_diff / std::max(max_a, max_b);
    
    std::printf("  %s: max_diff=%.3e, frob_diff=%.3e, rel_err=%.3e\n",
        name, max_diff, frob_diff, rel_err);
}

inline void print_diff_stats(const EigenDense& A, const EigenDense& B, const char* name = "Diff") {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        std::printf("  %s: dimension mismatch\n", name);
        return;
    }
    
    scl_real_t max_diff = (A - B).cwiseAbs().maxCoeff();
    scl_real_t frob_diff = (A - B).norm();
    scl_real_t max_val = std::max(A.cwiseAbs().maxCoeff(), B.cwiseAbs().maxCoeff());
    scl_real_t rel_err = max_diff / max_val;
    
    std::printf("  %s: max_diff=%.3e, frob_diff=%.3e, rel_err=%.3e\n",
        name, max_diff, frob_diff, rel_err);
}

} // namespace scl::test

