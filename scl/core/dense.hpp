#pragma once

#include "scl/core/type.hpp"
#include <vector>
#include <stdexcept>

// =============================================================================
/// @file dense.hpp
/// @brief Dense Matrix Implementation
///
/// Contains:
/// - IDense<T>: Virtual interface for polymorphic dense matrices
/// - Dense<T>: Concrete dense matrix implementation
/// - Utility functions for dense matrices
// =============================================================================

namespace scl {

// =============================================================================
// Virtual Interface: IDense
// =============================================================================

/// @brief Abstract base for dense matrices (for polymorphism)
template <typename T>
struct IDense {
    using ValueType = T;
    using Tag = TagDense;
    
    virtual ~IDense() = default;
    
    virtual Index rows() const = 0;
    virtual Index cols() const = 0;
    
    virtual const T* data() const { return nullptr; }
    virtual T* data() { return nullptr; }
    
    virtual T operator()(Index i, Index j) const = 0;
    
    virtual std::vector<T> row(Index i) const {
        Index n_cols = cols();
        std::vector<T> result(n_cols);
        for (Index j = 0; j < n_cols; ++j) {
            result[j] = (*this)(i, j);
        }
        return result;
    }
};

static_assert(DenseLike<IDense<float>>, "IDense must satisfy DenseLike");

// =============================================================================
// Dense: Concrete Dense Matrix
// =============================================================================

/// @brief Dense row-major matrix (non-owning view)
///
/// Memory Layout: Single contiguous array (rows × cols elements)
/// Indexing: ptr[r * cols + c]
///
/// Example:
///
/// std::vector<float> data(100);
/// Dense<float> mat(data.data(), 10, 10);
/// mat(0, 0) = 1.0f;
template <typename T>
struct Dense {
    using ValueType = T;
    using Tag = TagDense;
    
    T* ptr;
    Index rows;
    Index cols;
    
    constexpr Dense() noexcept : ptr(nullptr), rows(0), cols(0) {}
    constexpr Dense(T* p, Index r, Index c) noexcept : ptr(p), rows(r), cols(c) {}
    
    SCL_NODISCARD SCL_FORCE_INLINE T& operator()(Index r, Index c) const {
#if !defined(NDEBUG)
        assert(r >= 0 && r < rows && "Dense: Row out of bounds");
        assert(c >= 0 && c < cols && "Dense: Col out of bounds");
#endif
        return ptr[r * cols + c];
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Array<T> row(Index r) const {
#if !defined(NDEBUG)
        assert(r >= 0 && r < rows && "Dense: Row out of bounds");
#endif
        return Array<T>(ptr + (r * cols), static_cast<Size>(cols));
    }
};

static_assert(DenseLike<Dense<float>>, "Dense must satisfy DenseLike");

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Get element count
template <DenseLike M>
SCL_NODISCARD constexpr Size element_count(const M& mat) {
    return static_cast<Size>(scl::rows(mat)) * static_cast<Size>(scl::cols(mat));
}

/// @brief Check if indices are valid
template <typename M>
SCL_NODISCARD inline bool is_valid_index(const M& mat, Index i, Index j) {
    return i >= 0 && i < scl::rows(mat) && j >= 0 && j < scl::cols(mat);
}

/// @brief Safe element access with bounds checking
template <DenseLike M>
SCL_NODISCARD inline typename M::ValueType safe_at(const M& mat, Index i, Index j) {
    if (!is_valid_index(mat, i, j)) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return mat(i, j);
}

} // namespace scl
