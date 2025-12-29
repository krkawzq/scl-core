#pragma once

// =============================================================================
// SCL Core - Test Data Generators
// =============================================================================
//
// Random and structured test data generation for matrix operations.
//
// Features:
//   - Random sparse matrices (CSR/CSC) with controlled density
//   - Random dense matrices and vectors
//   - Special patterns (diagonal, identity, zero, ones)
//   - Reproducible with seed control
//
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace scl::test {

// Type aliases for convenience
using EigenCSR = Eigen::SparseMatrix<scl_real_t, Eigen::RowMajor, scl_index_t>;
using EigenCSC = Eigen::SparseMatrix<scl_real_t, Eigen::ColMajor, scl_index_t>;
using EigenDense = Eigen::Matrix<scl_real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenVector = Eigen::Matrix<scl_real_t, Eigen::Dynamic, 1>;

// =============================================================================
// Random Number Generator
// =============================================================================

class Random {
public:
    explicit Random(uint64_t seed = 42) : rng_(seed) {}
    
    /// Uniform random double in [min, max]
    [[nodiscard]] double uniform(double min = 0.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng_);
    }
    
    /// Uniform random integer in [min, max]
    [[nodiscard]] int64_t uniform_int(int64_t min, int64_t max) {
        std::uniform_int_distribution<int64_t> dist(min, max);
        return dist(rng_);
    }
    
    /// Standard normal (mean=0, stddev=1)
    [[nodiscard]] double normal(double mean = 0.0, double stddev = 1.0) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(rng_);
    }
    
    /// Bernoulli (true with probability p)
    [[nodiscard]] bool bernoulli(double p = 0.5) {
        std::bernoulli_distribution dist(p);
        return dist(rng_);
    }
    
    /// Get underlying engine
    [[nodiscard]] std::mt19937_64& engine() { return rng_; }

private:
    std::mt19937_64 rng_;
};

// =============================================================================
// Global Random Generator
// =============================================================================

/// Get global random generator (lazily initialized)
inline Random& global_rng() {
    static Random rng(42);  // Default seed
    return rng;
}

/// Set global random seed
inline void set_seed(uint64_t seed) {
    global_rng() = Random(seed);
}

// =============================================================================
// Random Shape Generation
// =============================================================================

/// Generate random shape within bounds
inline std::pair<scl_index_t, scl_index_t> random_shape(
    scl_index_t min_dim = 1,
    scl_index_t max_dim = 100,
    Random& rng = global_rng()
) {
    scl_index_t rows = rng.uniform_int(min_dim, max_dim);
    scl_index_t cols = rng.uniform_int(min_dim, max_dim);
    return {rows, cols};
}

/// Generate random square shape
inline scl_index_t random_square_size(
    scl_index_t min_size = 1,
    scl_index_t max_size = 100,
    Random& rng = global_rng()
) {
    return rng.uniform_int(min_size, max_size);
}

/// Generate random density
inline double random_density(
    double min_density = 0.01,
    double max_density = 0.2,
    Random& rng = global_rng()
) {
    return rng.uniform(min_density, max_density);
}

// =============================================================================
// Random Sparse Matrix Generation
// =============================================================================

/// Generate random sparse CSR matrix
inline EigenCSR random_sparse_csr(
    scl_index_t rows,
    scl_index_t cols,
    double density,
    Random& rng = global_rng()
) {
    EigenCSR mat(rows, cols);
    
    scl_index_t expected_nnz = static_cast<scl_index_t>(rows * cols * density);
    mat.reserve(Eigen::VectorXi::Constant(rows, expected_nnz / rows + 1));
    
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            if (rng.bernoulli(density)) {
                mat.insert(i, j) = rng.normal(0.0, 1.0);
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate random sparse CSC matrix
inline EigenCSC random_sparse_csc(
    scl_index_t rows,
    scl_index_t cols,
    double density,
    Random& rng = global_rng()
) {
    EigenCSC mat(rows, cols);
    
    scl_index_t expected_nnz = static_cast<scl_index_t>(rows * cols * density);
    mat.reserve(Eigen::VectorXi::Constant(cols, expected_nnz / cols + 1));
    
    for (scl_index_t j = 0; j < cols; ++j) {
        for (scl_index_t i = 0; i < rows; ++i) {
            if (rng.bernoulli(density)) {
                mat.insert(i, j) = rng.normal(0.0, 1.0);
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate random sparse matrix with specified nnz per row
inline EigenCSR random_sparse_csr_nnz_per_row(
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz_per_row,
    Random& rng = global_rng()
) {
    EigenCSR mat(rows, cols);
    mat.reserve(Eigen::VectorXi::Constant(rows, nnz_per_row));
    
    for (scl_index_t i = 0; i < rows; ++i) {
        std::vector<scl_index_t> col_indices(cols);
        std::iota(col_indices.begin(), col_indices.end(), 0);
        std::shuffle(col_indices.begin(), col_indices.end(), rng.engine());
        
        for (scl_index_t k = 0; k < std::min(nnz_per_row, cols); ++k) {
            mat.insert(i, col_indices[k]) = rng.normal(0.0, 1.0);
        }
    }
    
    mat.makeCompressed();
    return mat;
}

// =============================================================================
// Random Dense Matrix/Vector Generation
// =============================================================================

/// Generate random dense matrix
inline EigenDense random_dense(
    scl_index_t rows,
    scl_index_t cols,
    Random& rng = global_rng()
) {
    EigenDense mat(rows, cols);
    
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            mat(i, j) = rng.normal(0.0, 1.0);
        }
    }
    
    return mat;
}

/// Generate random vector
inline EigenVector random_vector(
    scl_index_t size,
    Random& rng = global_rng()
) {
    EigenVector vec(size);
    
    for (scl_index_t i = 0; i < size; ++i) {
        vec(i) = rng.normal(0.0, 1.0);
    }
    
    return vec;
}

/// Generate random positive definite matrix (for special tests)
inline EigenDense random_positive_definite(
    scl_index_t size,
    Random& rng = global_rng()
) {
    EigenDense A = random_dense(size, size, rng);
    return A * A.transpose() + EigenDense::Identity(size, size);
}

// =============================================================================
// Structured Matrix Generation
// =============================================================================

/// Generate diagonal sparse matrix
inline EigenCSR diagonal_csr(const EigenVector& diag) {
    scl_index_t n = diag.size();
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        if (std::abs(diag(i)) > 1e-15) {
            mat.insert(i, i) = diag(i);
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate identity matrix (sparse)
inline EigenCSR identity_csr(scl_index_t n) {
    EigenVector ones = EigenVector::Ones(n);
    return diagonal_csr(ones);
}

/// Generate zero matrix (sparse)
inline EigenCSR zero_csr(scl_index_t rows, scl_index_t cols) {
    EigenCSR mat(rows, cols);
    mat.makeCompressed();
    return mat;
}

/// Generate matrix of all ones (sparse)
inline EigenCSR ones_csr(scl_index_t rows, scl_index_t cols) {
    EigenCSR mat(rows, cols);
    mat.reserve(Eigen::VectorXi::Constant(rows, cols));
    
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            mat.insert(i, j) = 1.0;
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate banded matrix (sparse)
inline EigenCSR banded_csr(
    scl_index_t n,
    scl_index_t bandwidth,
    Random& rng = global_rng()
) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, 2 * bandwidth + 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        scl_index_t j_start = std::max<scl_index_t>(0, i - bandwidth);
        scl_index_t j_end = std::min<scl_index_t>(n, i + bandwidth + 1);
        
        for (scl_index_t j = j_start; j < j_end; ++j) {
            mat.insert(i, j) = rng.normal(0.0, 1.0);
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate block diagonal matrix
inline EigenCSR block_diagonal_csr(
    const std::vector<EigenDense>& blocks,
    Random& rng = global_rng()
) {
    (void)rng;  // Reserved for future use
    
    scl_index_t total_size = 0;
    for (const auto& block : blocks) {
        total_size += block.rows();
    }
    
    EigenCSR mat(total_size, total_size);
    
    scl_index_t offset = 0;
    for (const auto& block : blocks) {
        scl_index_t block_size = block.rows();
        for (scl_index_t i = 0; i < block_size; ++i) {
            for (scl_index_t j = 0; j < block_size; ++j) {
                if (std::abs(block(i, j)) > 1e-15) {
                    mat.insert(offset + i, offset + j) = block(i, j);
                }
            }
        }
        offset += block_size;
    }
    
    mat.makeCompressed();
    return mat;
}

// =============================================================================
// Test Data Fixtures (Common Test Matrices)
// =============================================================================

namespace fixture {

/// Tiny 3x3 matrix for quick tests
inline std::tuple<std::vector<scl_index_t>, std::vector<scl_index_t>, std::vector<scl_real_t>>
tiny_3x3_csr() {
    // Matrix:
    //   [1.0  0.0  2.0]
    //   [0.0  3.0  0.0]
    //   [4.0  5.0  6.0]
    
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    return {indptr, indices, data};
}

/// Small diagonal matrix 10x10
inline EigenCSR small_diagonal() {
    EigenVector diag = EigenVector::LinSpaced(10, 1.0, 10.0);
    return diagonal_csr(diag);
}

/// Medium sparse matrix (100x100, 5% density)
inline EigenCSR medium_sparse(Random& rng = global_rng()) {
    return random_sparse_csr(100, 100, 0.05, rng);
}

/// Large sparse matrix (1000x500, 2% density)
inline EigenCSR large_sparse(Random& rng = global_rng()) {
    return random_sparse_csr(1000, 500, 0.02, rng);
}

/// Rectangular matrix (tall: 1000x100)
inline EigenCSR tall_sparse(Random& rng = global_rng()) {
    return random_sparse_csr(1000, 100, 0.05, rng);
}

/// Rectangular matrix (wide: 100x1000)
inline EigenCSR wide_sparse(Random& rng = global_rng()) {
    return random_sparse_csr(100, 1000, 0.05, rng);
}

/// Single row matrix
inline EigenCSR single_row(scl_index_t cols, Random& rng = global_rng()) {
    return random_sparse_csr(1, cols, 0.5, rng);
}

/// Single column matrix
inline EigenCSR single_col(scl_index_t rows, Random& rng = global_rng()) {
    return random_sparse_csr(rows, 1, 0.5, rng);
}

/// Extremely sparse (< 0.1% density)
inline EigenCSR extremely_sparse(Random& rng = global_rng()) {
    return random_sparse_csr(1000, 1000, 0.0005, rng);
}

/// Moderately dense (50% density)
inline EigenCSR moderately_dense(Random& rng = global_rng()) {
    return random_sparse_csr(50, 50, 0.5, rng);
}

} // namespace fixture

// =============================================================================
// Size and Density Presets
// =============================================================================

namespace preset {

/// Sizes for different test scales
struct Sizes {
    static constexpr scl_index_t TINY = 3;
    static constexpr scl_index_t SMALL = 10;
    static constexpr scl_index_t MEDIUM = 100;
    static constexpr scl_index_t LARGE = 1000;
    static constexpr scl_index_t HUGE = 10000;
};

/// Density presets
struct Density {
    static constexpr double EXTREMELY_SPARSE = 0.0001;  // 0.01%
    static constexpr double VERY_SPARSE = 0.001;        // 0.1%
    static constexpr double SPARSE = 0.01;              // 1%
    static constexpr double MODERATE = 0.05;            // 5%
    static constexpr double DENSE = 0.2;                // 20%
    static constexpr double VERY_DENSE = 0.5;           // 50%
};

/// Generate matrix by preset size and density
inline EigenCSR random_by_preset(
    const char* size_preset,
    const char* density_preset,
    Random& rng = global_rng()
) {
    scl_index_t size = Sizes::MEDIUM;
    if (std::strcmp(size_preset, "tiny") == 0) size = Sizes::TINY;
    else if (std::strcmp(size_preset, "small") == 0) size = Sizes::SMALL;
    else if (std::strcmp(size_preset, "medium") == 0) size = Sizes::MEDIUM;
    else if (std::strcmp(size_preset, "large") == 0) size = Sizes::LARGE;
    else if (std::strcmp(size_preset, "huge") == 0) size = Sizes::HUGE;
    
    double density = Density::SPARSE;
    if (std::strcmp(density_preset, "extremely_sparse") == 0) density = Density::EXTREMELY_SPARSE;
    else if (std::strcmp(density_preset, "very_sparse") == 0) density = Density::VERY_SPARSE;
    else if (std::strcmp(density_preset, "sparse") == 0) density = Density::SPARSE;
    else if (std::strcmp(density_preset, "moderate") == 0) density = Density::MODERATE;
    else if (std::strcmp(density_preset, "dense") == 0) density = Density::DENSE;
    else if (std::strcmp(density_preset, "very_dense") == 0) density = Density::VERY_DENSE;
    
    return random_sparse_csr(size, size, density, rng);
}

} // namespace preset

// =============================================================================
// Matrix Property Generators
// =============================================================================

namespace property {

/// Generate symmetric matrix (CSR)
inline EigenCSR symmetric(scl_index_t n, double density, Random& rng = global_rng()) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, static_cast<int>(n * density) + 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        for (scl_index_t j = i; j < n; ++j) {
            if (rng.bernoulli(density)) {
                scl_real_t value = rng.normal(0.0, 1.0);
                mat.insert(i, j) = value;
                if (i != j) {
                    mat.insert(j, i) = value;
                }
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate lower triangular matrix
inline EigenCSR lower_triangular(scl_index_t n, double density, Random& rng = global_rng()) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, static_cast<int>(n * density / 2) + 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        for (scl_index_t j = 0; j <= i; ++j) {
            if (rng.bernoulli(density)) {
                mat.insert(i, j) = rng.normal(0.0, 1.0);
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate upper triangular matrix
inline EigenCSR upper_triangular(scl_index_t n, double density, Random& rng = global_rng()) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, static_cast<int>(n * density / 2) + 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        for (scl_index_t j = i; j < n; ++j) {
            if (rng.bernoulli(density)) {
                mat.insert(i, j) = rng.normal(0.0, 1.0);
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Generate strictly diagonally dominant matrix (for stability tests)
inline EigenCSR diagonally_dominant(scl_index_t n, Random& rng = global_rng()) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, n / 10 + 2));
    
    for (scl_index_t i = 0; i < n; ++i) {
        scl_real_t row_sum = 0;
        
        // Off-diagonal elements
        for (scl_index_t j = 0; j < n; ++j) {
            if (i != j && rng.bernoulli(0.1)) {
                scl_real_t value = rng.uniform(0.0, 1.0);
                mat.insert(i, j) = value;
                row_sum += std::abs(value);
            }
        }
        
        // Diagonal element (strictly dominant)
        mat.insert(i, i) = row_sum + rng.uniform(1.0, 2.0);
    }
    
    mat.makeCompressed();
    return mat;
}

} // namespace property

// =============================================================================
// Edge Case Generators
// =============================================================================

namespace edge {

/// Empty matrix (0x0)
inline EigenCSR empty_matrix() {
    return EigenCSR(0, 0);
}

/// Single element matrix (1x1)
inline EigenCSR single_element(scl_real_t value = 1.0) {
    EigenCSR mat(1, 1);
    mat.insert(0, 0) = value;
    mat.makeCompressed();
    return mat;
}

/// All zeros (explicit)
inline EigenCSR zero_matrix(scl_index_t rows, scl_index_t cols) {
    return zero_csr(rows, cols);
}

/// Matrix with single non-zero
inline EigenCSR single_nonzero(
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t row,
    scl_index_t col,
    scl_real_t value = 1.0
) {
    EigenCSR mat(rows, cols);
    if (row < rows && col < cols) {
        mat.insert(row, col) = value;
    }
    mat.makeCompressed();
    return mat;
}

/// Matrix with all same value
inline EigenCSR constant_matrix(
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t value,
    double density = 0.1,
    Random& rng = global_rng()
) {
    EigenCSR mat(rows, cols);
    mat.reserve(Eigen::VectorXi::Constant(rows, static_cast<int>(cols * density) + 1));
    
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            if (rng.bernoulli(density)) {
                mat.insert(i, j) = value;
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

/// Matrix with very large values (overflow test)
inline EigenCSR large_values(scl_index_t n = 10) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        mat.insert(i, i) = 1e100;
    }
    
    mat.makeCompressed();
    return mat;
}

/// Matrix with very small values (underflow test)
inline EigenCSR small_values(scl_index_t n = 10) {
    EigenCSR mat(n, n);
    mat.reserve(Eigen::VectorXi::Constant(n, 1));
    
    for (scl_index_t i = 0; i < n; ++i) {
        mat.insert(i, i) = 1e-100;
    }
    
    mat.makeCompressed();
    return mat;
}

} // namespace edge

// =============================================================================
// Batch Generators (for Parameterized Tests)
// =============================================================================

/// Generate multiple matrices for parameterized tests
inline std::vector<EigenCSR> batch_random_sparse(
    const std::vector<std::tuple<scl_index_t, scl_index_t, double>>& configs,
    Random& rng = global_rng()
) {
    std::vector<EigenCSR> matrices;
    matrices.reserve(configs.size());
    
    for (const auto& [rows, cols, density] : configs) {
        matrices.push_back(random_sparse_csr(rows, cols, density, rng));
    }
    
    return matrices;
}

/// Generate matrices with varying sizes
inline std::vector<EigenCSR> batch_varying_sizes(
    const std::vector<scl_index_t>& sizes,
    double density = 0.05,
    Random& rng = global_rng()
) {
    std::vector<EigenCSR> matrices;
    matrices.reserve(sizes.size());
    
    for (scl_index_t size : sizes) {
        matrices.push_back(random_sparse_csr(size, size, density, rng));
    }
    
    return matrices;
}

/// Generate matrices with varying densities
inline std::vector<EigenCSR> batch_varying_densities(
    scl_index_t size,
    const std::vector<double>& densities,
    Random& rng = global_rng()
) {
    std::vector<EigenCSR> matrices;
    matrices.reserve(densities.size());
    
    for (double density : densities) {
        matrices.push_back(random_sparse_csr(size, size, density, rng));
    }
    
    return matrices;
}

/// Generate random sparse matrix with random shape
inline EigenCSR random_sparse_random_shape(
    scl_index_t min_dim = 10,
    scl_index_t max_dim = 100,
    double min_density = 0.01,
    double max_density = 0.1,
    Random& rng = global_rng()
) {
    auto [rows, cols] = random_shape(min_dim, max_dim, rng);
    double density = random_density(min_density, max_density, rng);
    return random_sparse_csr(rows, cols, density, rng);
}

/// Generate batch of matrices with random shapes
inline std::vector<EigenCSR> batch_random_shapes(
    size_t count,
    scl_index_t min_dim = 10,
    scl_index_t max_dim = 100,
    double density = 0.05,
    Random& rng = global_rng()
) {
    std::vector<EigenCSR> matrices;
    matrices.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        auto [rows, cols] = random_shape(min_dim, max_dim, rng);
        matrices.push_back(random_sparse_csr(rows, cols, density, rng));
    }
    
    return matrices;
}

} // namespace scl::test

