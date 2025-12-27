#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

// =============================================================================
/// @file callback_sparse.hpp
/// @brief Callback-Based Sparse Matrix Implementation
///
/// Enables Python (or other languages) to implement custom sparse matrix
/// access patterns via C-style function pointers. This allows:
///
/// - Lazy loading: Load data from disk on demand
/// - Remote data: Fetch data from network/database
/// - Virtual views: Transform data dynamically
/// - Custom storage: Implement arbitrary storage formats
///
/// Design:
///
/// 1. VTable Structure: Holds C function pointers for all required operations
/// 2. CallbackSparse<T, IsCSR>: Inherits from ISparse and delegates to callbacks
/// 3. Zero C++ overhead: Pure C function pointer dispatch
///
/// Usage from Python:
///
/// from ctypes import CFUNCTYPE
///
/// GetRowsFunc = CFUNCTYPE(c_int64, c_void_p)
///
/// @GetRowsFunc
/// def get_rows(ctx):
///     return self.shape[0]
///
/// vtable = SparseCallbackVTable(get_rows=get_rows, ...)
/// handle = scl_create_callback_csr(id(self), byref(vtable))
///
/// Warning:
///
/// - Performance: Each access crosses the Python-C++ boundary
/// - GIL: Callbacks hold the GIL, preventing parallel execution
/// - Best suited for I/O-bound scenarios, not compute-intensive loops
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: C-Style Callback Type Definitions
// =============================================================================

/// @brief Error codes returned by callbacks
enum class CallbackError : int {
    OK = 0,              ///< Success
    INVALID_INDEX = 1,   ///< Index out of bounds
    IO_ERROR = 2,        ///< I/O operation failed
    MEMORY_ERROR = 3,    ///< Memory allocation failed
    UNKNOWN = 99         ///< Unknown error
};

/// @brief Callback to get number of rows
/// @param context User-provided context pointer (e.g., Python object ID)
/// @return Number of rows
typedef Index (*GetRowsCallback)(void* context);

/// @brief Callback to get number of columns
/// @param context User-provided context pointer
/// @return Number of columns
typedef Index (*GetColsCallback)(void* context);

/// @brief Callback to get total number of non-zero elements
/// @param context User-provided context pointer
/// @return Total nnz
typedef Index (*GetNnzCallback)(void* context);

/// @brief Callback to get values for primary dimension i
/// @param context User-provided context pointer
/// @param i Primary dimension index (row for CSR, col for CSC)
/// @param out_data Output: pointer to data array
/// @param out_len Output: length of data array
/// @return 0 on success, error code on failure
typedef int (*GetPrimaryValuesCallback)(
    void* context, 
    Index i, 
    Real** out_data, 
    Index* out_len
);

/// @brief Callback to get indices for primary dimension i
/// @param context User-provided context pointer
/// @param i Primary dimension index
/// @param out_indices Output: pointer to indices array
/// @param out_len Output: length of indices array
/// @return 0 on success, error code on failure
typedef int (*GetPrimaryIndicesCallback)(
    void* context, 
    Index i, 
    Index** out_indices, 
    Index* out_len
);

/// @brief Callback to get length of primary dimension i (optional optimization)
/// @param context User-provided context pointer
/// @param i Primary dimension index
/// @return Length of the i-th row/column
typedef Index (*GetPrimaryLengthCallback)(void* context, Index i);

/// @brief Callback for batch prefetch (optional optimization)
/// @param context User-provided context pointer
/// @param start Start index
/// @param end End index (exclusive)
/// @return 0 on success, error code on failure
typedef int (*PrefetchRangeCallback)(void* context, Index start, Index end);

/// @brief Callback to release resources for primary dimension i (optional)
/// @param context User-provided context pointer
/// @param i Primary dimension index
/// @return 0 on success, error code on failure
typedef int (*ReleasePrimaryCallback)(void* context, Index i);

// =============================================================================
// SECTION 2: VTable Structure
// =============================================================================

/// @brief Virtual function table for callback-based sparse matrices
///
/// All required callbacks must be non-null. Optional callbacks can be null.
///
/// Memory Contract:
///
/// - Data returned by get_primary_values/get_primary_indices must remain valid
///   until the next call to the same callback or until release_primary is called
/// - The Python side is responsible for keeping data alive (e.g., caching arrays)
struct SparseCallbackVTable {
    // Required callbacks (must be non-null)
    GetRowsCallback get_rows;                   ///< Get number of rows
    GetColsCallback get_cols;                   ///< Get number of columns
    GetNnzCallback get_nnz;                     ///< Get total nnz
    GetPrimaryValuesCallback get_primary_values;   ///< Get values for primary[i]
    GetPrimaryIndicesCallback get_primary_indices; ///< Get indices for primary[i]
    
    // Optional callbacks (can be null)
    GetPrimaryLengthCallback get_primary_length;   ///< Fast length query (optional)
    PrefetchRangeCallback prefetch_range;          ///< Batch prefetch (optional)
    ReleasePrimaryCallback release_primary;        ///< Release resources (optional)
    
    /// @brief Validate that all required callbacks are set
    bool is_valid() const {
        return get_rows != nullptr 
            && get_cols != nullptr 
            && get_nnz != nullptr
            && get_primary_values != nullptr 
            && get_primary_indices != nullptr;
    }
};

// =============================================================================
// SECTION 3: CallbackSparse Implementation
// =============================================================================

/// @brief Sparse matrix implemented via callback functions
///
/// This class inherits from ISparse<T, IsCSR> and delegates all operations
/// to user-provided callbacks. This allows Python users to implement custom
/// data access patterns that integrate seamlessly with SCL operators.
///
/// Template Parameters:
/// - T: Element type (usually Real)
/// - IsCSR: true for CSR layout, false for CSC layout
///
/// Example (Lazy HDF5 CSR):
///
/// class HDF5LazyCSR(CallbackCSR):
///     def get_row_data(self, i):
///         return self.h5file[f'row_{i}'][:]
template <typename T, bool IsCSR>
class CallbackSparse : public ISparse<T, IsCSR> {
private:
    void* context_;                         ///< User context (e.g., Python object ID)
    const SparseCallbackVTable* vtable_;    ///< Callback function table
    
    // Cached dimension values (computed once)
    mutable Index cached_rows_ = -1;
    mutable Index cached_cols_ = -1;
    mutable Index cached_nnz_ = -1;
    
public:
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    /// @brief Construct callback sparse matrix
    /// @param context User context pointer (passed to all callbacks)
    /// @param vtable Pointer to callback function table (must remain valid)
    CallbackSparse(void* context, const SparseCallbackVTable* vtable)
        : context_(context), vtable_(vtable) {}
    
    // -------------------------------------------------------------------------
    // ISparse Interface Implementation
    // -------------------------------------------------------------------------
    
    /// @brief Get number of rows
    Index rows() const override {
        if (cached_rows_ < 0) {
            cached_rows_ = vtable_->get_rows(context_);
        }
        return cached_rows_;
    }
    
    /// @brief Get number of columns
    Index cols() const override {
        if (cached_cols_ < 0) {
            cached_cols_ = vtable_->get_cols(context_);
        }
        return cached_cols_;
    }
    
    /// @brief Get total number of non-zero elements
    Index nnz() const override {
        if (cached_nnz_ < 0) {
            cached_nnz_ = vtable_->get_nnz(context_);
        }
        return cached_nnz_;
    }
    
    /// @brief Get values for primary dimension i
    /// @param i Primary dimension index (row for CSR, col for CSC)
    /// @return Array view of values (valid until next call or release)
    Array<T> primary_values(Index i) const override {
        T* data = nullptr;
        Index len = 0;
        int err = vtable_->get_primary_values(context_, i, &data, &len);
        if (err != 0) {
            // Return empty array on error
            // The caller should check for empty arrays
            return Array<T>(nullptr, 0);
        }
        return Array<T>(data, static_cast<Size>(len));
    }
    
    /// @brief Get indices for primary dimension i
    /// @param i Primary dimension index
    /// @return Array view of indices (valid until next call or release)
    Array<Index> primary_indices(Index i) const override {
        Index* indices = nullptr;
        Index len = 0;
        int err = vtable_->get_primary_indices(context_, i, &indices, &len);
        if (err != 0) {
            return Array<Index>(nullptr, 0);
        }
        return Array<Index>(indices, static_cast<Size>(len));
    }
    
    /// @brief Get length of primary dimension i
    /// @param i Primary dimension index
    /// @return Number of elements in row/column i
    Index primary_length(Index i) const override {
        // Use optimized callback if available
        if (vtable_->get_primary_length != nullptr) {
            return vtable_->get_primary_length(context_, i);
        }
        // Fallback: get values and return size
        return static_cast<Index>(primary_values(i).size());
    }
    
    // -------------------------------------------------------------------------
    // Optional Operations
    // -------------------------------------------------------------------------
    
    /// @brief Prefetch a range of rows/columns for better performance
    /// @param start Start index (inclusive)
    /// @param end End index (exclusive)
    /// @return true if prefetch was performed, false otherwise
    bool prefetch(Index start, Index end) const {
        if (vtable_->prefetch_range != nullptr) {
            return vtable_->prefetch_range(context_, start, end) == 0;
        }
        return false;
    }
    
    /// @brief Release resources for row/column i
    /// @param i Primary dimension index
    /// @return true if release was performed, false otherwise
    bool release(Index i) const {
        if (vtable_->release_primary != nullptr) {
            return vtable_->release_primary(context_, i) == 0;
        }
        return false;
    }
    
    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    
    /// @brief Get the user context pointer
    void* context() const { return context_; }
    
    /// @brief Get the vtable pointer
    const SparseCallbackVTable* vtable() const { return vtable_; }
    
    /// @brief Invalidate cached dimensions (call after Python object changes)
    void invalidate_cache() {
        cached_rows_ = -1;
        cached_cols_ = -1;
        cached_nnz_ = -1;
    }
};

// =============================================================================
// SECTION 4: Type Aliases
// =============================================================================

/// @brief Callback-based CSR matrix
template <typename T = Real>
using CallbackCSR = CallbackSparse<T, true>;

/// @brief Callback-based CSC matrix
template <typename T = Real>
using CallbackCSC = CallbackSparse<T, false>;

// Verify concepts
static_assert(CSRLike<CallbackCSR<Real>>, "CallbackCSR must satisfy CSRLike");
static_assert(CSCLike<CallbackCSC<Real>>, "CallbackCSC must satisfy CSCLike");

// =============================================================================
// SECTION 5: Factory Functions
// =============================================================================

/// @brief Create a callback CSR matrix
/// @param context User context pointer
/// @param vtable Callback function table
/// @return Pointer to new CallbackCSR (caller owns memory)
template <typename T = Real>
inline CallbackCSR<T>* create_callback_csr(void* context, const SparseCallbackVTable* vtable) {
    if (!vtable || !vtable->is_valid()) {
        return nullptr;
    }
    return new CallbackCSR<T>(context, vtable);
}

/// @brief Create a callback CSC matrix
/// @param context User context pointer
/// @param vtable Callback function table
/// @return Pointer to new CallbackCSC (caller owns memory)
template <typename T = Real>
inline CallbackCSC<T>* create_callback_csc(void* context, const SparseCallbackVTable* vtable) {
    if (!vtable || !vtable->is_valid()) {
        return nullptr;
    }
    return new CallbackCSC<T>(context, vtable);
}

/// @brief Destroy a callback sparse matrix
/// @param mat Pointer to CallbackSparse (will be deleted)
template <typename T, bool IsCSR>
inline void destroy_callback_sparse(CallbackSparse<T, IsCSR>* mat) {
    delete mat;
}

} // namespace scl

