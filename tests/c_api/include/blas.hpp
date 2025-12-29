#pragma once

// =============================================================================
// SCL Core - BLAS Reference Implementation
// =============================================================================
//
// BLAS-based reference implementation for numerical verification.
// Provides gold standard for correctness testing.
//
// Requires: libblas or libopenblas
//
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#include <vector>
#include <cmath>
#include <algorithm>

// BLAS declarations (avoid full BLAS header dependency)
extern "C" {
    // Level 1: Vector operations
    double ddot_(const int* n, const double* x, const int* incx, 
                 const double* y, const int* incy);
    
    void daxpy_(const int* n, const double* alpha, const double* x, const int* incx,
                double* y, const int* incy);
    
    double dnrm2_(const int* n, const double* x, const int* incx);
    
    // Level 2: Matrix-vector operations
    void dgemv_(const char* trans, const int* m, const int* n,
                const double* alpha, const double* a, const int* lda,
                const double* x, const int* incx,
                const double* beta, double* y, const int* incy);
    
    // Level 3: Matrix-matrix operations
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
}

namespace scl::test::blas {

// =============================================================================
// BLAS Wrapper Functions
// =============================================================================

/// Vector dot product: x^T * y
inline double dot(const std::vector<scl_real_t>& x, const std::vector<scl_real_t>& y) {
    int n = static_cast<int>(x.size());
    int inc = 1;
    return ddot_(&n, x.data(), &inc, y.data(), &inc);
}

/// Vector norm: ||x||_2
inline double norm2(const std::vector<scl_real_t>& x) {
    int n = static_cast<int>(x.size());
    int inc = 1;
    return dnrm2_(&n, x.data(), &inc);
}

/// Dense matrix-vector multiply: y = alpha * A * x + beta * y
inline void gemv(
    bool transpose,
    int m, int n,
    double alpha,
    const std::vector<scl_real_t>& A,
    const std::vector<scl_real_t>& x,
    double beta,
    std::vector<scl_real_t>& y
) {
    char trans = transpose ? 'T' : 'N';
    int lda = n;  // Row-major
    int incx = 1;
    int incy = 1;
    
    dgemv_(&trans, &m, &n, &alpha, A.data(), &lda, x.data(), &incx, &beta, y.data(), &incy);
}

/// Dense matrix-matrix multiply: C = alpha * A * B + beta * C
inline void gemm(
    bool transpose_a, bool transpose_b,
    int m, int n, int k,
    double alpha,
    const std::vector<scl_real_t>& A,
    const std::vector<scl_real_t>& B,
    double beta,
    std::vector<scl_real_t>& C
) {
    char transa = transpose_a ? 'T' : 'N';
    char transb = transpose_b ? 'T' : 'N';
    int lda = transpose_a ? m : k;
    int ldb = transpose_b ? k : n;
    int ldc = n;
    
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C.data(), &ldc);
}

} // namespace scl::test::blas

