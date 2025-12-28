#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"
#include "scl/core/registry.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <span>
#include <vector>
#include <optional>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// =============================================================================
// FILE: scl/core/sparse.hpp
// BRIEF: Sparse matrix with block-allocated discontiguous storage
//
// DESIGN NOTES:
// - Uses row-pointer (not row-offset) design for efficient slicing
// - Block allocation to balance memory reuse and fragmentation
// - Reference counting for shared sub-matrices
// - Compatible with traditional CSR when using contiguous allocation
//
// INVARIANT:
// - Indices within each row (CSR) or column (CSC) are strictly ascending
//   (sorted). This is enforced by all factory methods and maintained by
//   all operations. All element access methods assume this invariant.
//
// OWNERSHIP SEMANTICS:
// - Factory methods (create, from_traditional, etc.) -> owns data
// - wrap_traditional -> does NOT own data (external lifetime management)
// - row_slice_view/col_slice_view -> shares data via reference counting
// - clone()/copy() -> deep copy with independent ownership
// - Move -> transfers ownership, source becomes empty
// =============================================================================

namespace scl {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T, bool IsCSR>
struct Sparse;

template <typename T>
using CSRMatrix = Sparse<T, true>;

template <typename T>
using CSCMatrix = Sparse<T, false>;

using CSR = CSRMatrix<Real>;
using CSC = CSCMatrix<Real>;

// =============================================================================
// Block Allocation Strategy
// =============================================================================

/// @brief Configuration for block allocation strategy
struct BlockStrategy {
    /// @brief Minimum elements per block (to avoid too many small blocks)
    Index min_block_elements = 4096;
    
    /// @brief Maximum elements per block (to enable partial release)
    Index max_block_elements = 262144;  // 256K elements = ~1MB for float
    
    /// @brief Target number of blocks (0 = auto, based on hardware concurrency)
    Index target_block_count = 0;
    
    /// @brief Force single contiguous block (traditional CSR compatible)
    bool force_contiguous = false;
    
    // Predefined strategies
    static constexpr BlockStrategy contiguous() {
        return {0, 0, 0, true};
    }
    
    static constexpr BlockStrategy small_blocks() {
        return {1024, 16384, 0, false};
    }
    
    static constexpr BlockStrategy large_blocks() {
        return {65536, 1048576, 0, false};
    }
    
    static constexpr BlockStrategy adaptive() {
        return {4096, 262144, 0, false};
    }
    
    /// @brief Compute optimal block size for given data
    [[nodiscard]] Index compute_block_size(Index total_nnz, Index primary_dim) const {
        if (force_contiguous || total_nnz == 0) {
            return total_nnz;
        }
        
        Index target_blocks = target_block_count;
        if (target_blocks <= 0) {
            // Auto: aim for reasonable parallelism
            target_blocks = std::max(Index{4}, 
                static_cast<Index>(std::thread::hardware_concurrency()));
        }
        
        Index ideal_size = (total_nnz + target_blocks - 1) / target_blocks;
        return std::clamp(ideal_size, min_block_elements, max_block_elements);
    }
};

// =============================================================================
// Memory Layout Information
// =============================================================================

/// @brief Information about the memory layout of a sparse matrix
struct SparseLayoutInfo {
    /// @brief Total number of allocated data blocks
    Index data_block_count = 0;
    
    /// @brief Total number of allocated index blocks  
    Index index_block_count = 0;
    
    /// @brief Total bytes used for values
    std::size_t data_bytes = 0;
    
    /// @brief Total bytes used for indices
    std::size_t index_bytes = 0;
    
    /// @brief Total bytes used for metadata (pointers, lengths)
    std::size_t metadata_bytes = 0;
    
    /// @brief Whether all data is in a single contiguous block
    bool is_contiguous = false;
    
    /// @brief Whether layout is compatible with traditional CSR/CSC
    bool is_traditional_format = false;
    
    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return data_bytes + index_bytes + metadata_bytes;
    }
};

// =============================================================================
// Sparse Matrix Implementation
// =============================================================================

template <typename T, bool IsCSR>
struct Sparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    using SelfType = Sparse<T, IsCSR>;
    using TransposeType = Sparse<T, !IsCSR>;

    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

    // =========================================================================
    // Data Members
    // =========================================================================

    Pointer* data_ptrs;      // Array of pointers to row/col data
    Pointer* indices_ptrs;   // Array of pointers to row/col indices
    Index* lengths;          // Length of each row/col
    Index rows_;
    Index cols_;
    Index nnz_;
    bool owns_data_;         // Whether this matrix owns the data (false for wrapped)
    bool is_view_;           // Whether this is a view (shares pointers with another matrix)

    // =========================================================================
    // Constructors
    // =========================================================================

    constexpr Sparse() noexcept
        : data_ptrs(nullptr), indices_ptrs(nullptr), lengths(nullptr),
          rows_(0), cols_(0), nnz_(0), owns_data_(true), is_view_(false) {}

    constexpr Sparse(Pointer* dp, Pointer* ip, Index* len,
                     Index r, Index c, Index n, bool owns = true, bool is_view = false) noexcept
        : data_ptrs(dp), indices_ptrs(ip), lengths(len),
          rows_(r), cols_(c), nnz_(n), owns_data_(owns), is_view_(is_view) {
        SCL_ASSERT(r >= 0, "Sparse: rows must be non-negative");
        SCL_ASSERT(c >= 0, "Sparse: cols must be non-negative");
        SCL_ASSERT(n >= 0, "Sparse: nnz must be non-negative");
    }

    ~Sparse() {
        release_resources();
    }

    // Disable copy (use clone() for deep copy)
    Sparse(const Sparse&) = delete;
    Sparse& operator=(const Sparse&) = delete;

    // Move constructor: transfer ownership
    Sparse(Sparse&& other) noexcept
        : data_ptrs(other.data_ptrs)
        , indices_ptrs(other.indices_ptrs)
        , lengths(other.lengths)
        , rows_(other.rows_)
        , cols_(other.cols_)
        , nnz_(other.nnz_)
        , owns_data_(other.owns_data_)
        , is_view_(other.is_view_)
    {
        other.data_ptrs = nullptr;
        other.indices_ptrs = nullptr;
        other.lengths = nullptr;
        other.rows_ = other.cols_ = other.nnz_ = 0;
        other.owns_data_ = false;
        other.is_view_ = false;
    }

    // Move assignment: release current resources and transfer ownership
    Sparse& operator=(Sparse&& other) noexcept {
        if (this != &other) {
            release_resources();

            data_ptrs = other.data_ptrs;
            indices_ptrs = other.indices_ptrs;
            lengths = other.lengths;
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            owns_data_ = other.owns_data_;
            is_view_ = other.is_view_;

            other.data_ptrs = nullptr;
            other.indices_ptrs = nullptr;
            other.lengths = nullptr;
            other.rows_ = other.cols_ = other.nnz_ = 0;
            other.owns_data_ = false;
            other.is_view_ = false;
        }
        return *this;
    }
    
    /// @brief Explicit deep copy (since copy constructor is deleted)
    [[nodiscard]] Sparse copy() const {
        return clone(BlockStrategy::adaptive());
    }

    // =========================================================================
    // Basic Queries
    // =========================================================================

    [[nodiscard]] SCL_FORCE_INLINE bool valid() const noexcept {
        return data_ptrs != nullptr && indices_ptrs != nullptr && lengths != nullptr;
    }

    [[nodiscard]] SCL_FORCE_INLINE explicit operator bool() const noexcept {
        return valid();
    }

    [[nodiscard]] SCL_FORCE_INLINE Index rows() const noexcept { return rows_; }
    [[nodiscard]] SCL_FORCE_INLINE Index cols() const noexcept { return cols_; }
    [[nodiscard]] SCL_FORCE_INLINE Index nnz() const noexcept { return nnz_; }

    [[nodiscard]] SCL_FORCE_INLINE Index primary_dim() const noexcept {
        return IsCSR ? rows_ : cols_;
    }

    [[nodiscard]] SCL_FORCE_INLINE Index secondary_dim() const noexcept {
        return IsCSR ? cols_ : rows_;
    }

    [[nodiscard]] SCL_FORCE_INLINE bool empty() const noexcept {
        return rows_ == 0 || cols_ == 0 || nnz_ == 0;
    }
    
    /// @brief Calculate sparsity (fraction of zeros)
    [[nodiscard]] double sparsity() const noexcept {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        return 1.0 - static_cast<double>(nnz_) / (static_cast<double>(rows_) * cols_);
    }
    
    /// @brief Calculate density (fraction of non-zeros)
    [[nodiscard]] double density() const noexcept {
        return 1.0 - sparsity();
    }

    // =========================================================================
    // Row/Column Access (CSR/CSC specific)
    // =========================================================================

    [[nodiscard]] SCL_FORCE_INLINE Array<T> row_values(Index i) const noexcept 
        requires (IsCSR) {
        SCL_ASSERT(i >= 0 && i < rows_, "row_values: index out of bounds");
        return {static_cast<T*>(data_ptrs[i]), static_cast<Size>(lengths[i])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Array<Index> row_indices(Index i) const noexcept 
        requires (IsCSR) {
        SCL_ASSERT(i >= 0 && i < rows_, "row_indices: index out of bounds");
        return {static_cast<Index*>(indices_ptrs[i]), static_cast<Size>(lengths[i])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Index row_length(Index i) const noexcept 
        requires (IsCSR) {
        SCL_ASSERT(i >= 0 && i < rows_, "row_length: index out of bounds");
        return lengths[i];
    }

    [[nodiscard]] SCL_FORCE_INLINE Array<T> col_values(Index j) const noexcept 
        requires (!IsCSR) {
        SCL_ASSERT(j >= 0 && j < cols_, "col_values: index out of bounds");
        return {static_cast<T*>(data_ptrs[j]), static_cast<Size>(lengths[j])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Array<Index> col_indices(Index j) const noexcept 
        requires (!IsCSR) {
        SCL_ASSERT(j >= 0 && j < cols_, "col_indices: index out of bounds");
        return {static_cast<Index*>(indices_ptrs[j]), static_cast<Size>(lengths[j])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Index col_length(Index j) const noexcept 
        requires (!IsCSR) {
        SCL_ASSERT(j >= 0 && j < cols_, "col_length: index out of bounds");
        return lengths[j];
    }

    // Unified access for generic code
    [[nodiscard]] SCL_FORCE_INLINE Array<T> primary_values(Index i) const noexcept {
        SCL_ASSERT(i >= 0 && i < primary_dim(), "primary_values: index out of bounds");
        return {static_cast<T*>(data_ptrs[i]), static_cast<Size>(lengths[i])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Array<Index> primary_indices(Index i) const noexcept {
        SCL_ASSERT(i >= 0 && i < primary_dim(), "primary_indices: index out of bounds");
        return {static_cast<Index*>(indices_ptrs[i]), static_cast<Size>(lengths[i])};
    }

    [[nodiscard]] SCL_FORCE_INLINE Index primary_length(Index i) const noexcept {
        SCL_ASSERT(i >= 0 && i < primary_dim(), "primary_length: index out of bounds");
        return lengths[i];
    }

    // =========================================================================
    // Element Access
    // =========================================================================
    
    /// @brief Get value at (row, col), returns 0 if not found
    /// @note O(log n) - indices are guaranteed sorted per CSR/CSC specification
    [[nodiscard]] T at(Index row, Index col) const noexcept {
        if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return T{0};
        }
        
        const Index primary_idx = IsCSR ? row : col;
        const Index secondary_idx = IsCSR ? col : row;
        
        const auto indices = primary_indices(primary_idx);
        if (indices.empty()) return T{0};
        
        // Binary search (indices guaranteed sorted)
        const auto it = std::lower_bound(indices.begin(), indices.end(), secondary_idx);
        if (it != indices.end() && *it == secondary_idx) {
            return primary_values(primary_idx)[it - indices.begin()];
        }
        return T{0};
    }
    
    /// @brief Check if element exists at (row, col)
    /// @note O(log n) - indices are guaranteed sorted per CSR/CSC specification
    [[nodiscard]] bool exists(Index row, Index col) const noexcept {
        if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return false;
        }
        
        const Index primary_idx = IsCSR ? row : col;
        const Index secondary_idx = IsCSR ? col : row;
        
        const auto indices = primary_indices(primary_idx);
        if (indices.empty()) return false;
        
        // Binary search (indices guaranteed sorted)
        const auto it = std::lower_bound(indices.begin(), indices.end(), secondary_idx);
        return it != indices.end() && *it == secondary_idx;
    }

    // =========================================================================
    // Layout Information
    // =========================================================================
    
    /// @brief Check if data is stored in a single contiguous block
    [[nodiscard]] bool is_contiguous() const noexcept {
        if (!valid() || nnz_ == 0) return true;
        
        const Index pdim = primary_dim();
        
        // Find first non-empty row
        Index first_nonempty = -1;
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] > 0) {
                first_nonempty = i;
                break;
            }
        }
        
        if (first_nonempty < 0) return true;
        
        // Check if all data is contiguous
        T* expected_data = static_cast<T*>(data_ptrs[first_nonempty]);
        Index* expected_indices = static_cast<Index*>(indices_ptrs[first_nonempty]);
        
        for (Index i = first_nonempty; i < pdim; ++i) {
            if (lengths[i] == 0) continue;
            
            if (static_cast<T*>(data_ptrs[i]) != expected_data ||
                static_cast<Index*>(indices_ptrs[i]) != expected_indices) {
                return false;
            }
            
            expected_data += lengths[i];
            expected_indices += lengths[i];
        }
        
        return true;
    }
    
    /// @brief Check if layout is compatible with traditional CSR/CSC format
    /// Traditional format: contiguous data + row_ptr array derivable from lengths
    [[nodiscard]] bool is_traditional_format() const noexcept {
        return is_contiguous();
    }
    
    /// @brief Get detailed layout information
    [[nodiscard]] SparseLayoutInfo layout_info() const noexcept {
        SparseLayoutInfo info;
        
        if (!valid()) return info;
        
        const Index pdim = primary_dim();
        auto& reg = get_registry();
        
        // Count unique data blocks using unordered_set for O(1) insertion
        std::unordered_set<BufferID> seen_data_blocks;
        std::unordered_set<BufferID> seen_index_blocks;
        
        for (Index i = 0; i < pdim; ++i) {
            if (data_ptrs[i]) {
                BufferID id = reg.get_buffer_id(data_ptrs[i]);
                if (id != 0) {
                    seen_data_blocks.insert(id);
                }
            }
            if (indices_ptrs[i]) {
                BufferID id = reg.get_buffer_id(indices_ptrs[i]);
                if (id != 0) {
                    seen_index_blocks.insert(id);
                }
            }
        }
        
        info.data_block_count = static_cast<Index>(seen_data_blocks.size());
        info.index_block_count = static_cast<Index>(seen_index_blocks.size());
        info.data_bytes = static_cast<std::size_t>(nnz_) * sizeof(T);
        info.index_bytes = static_cast<std::size_t>(nnz_) * sizeof(Index);
        info.metadata_bytes = static_cast<std::size_t>(pdim) * 
                              (2 * sizeof(Pointer) + sizeof(Index));
        info.is_contiguous = is_contiguous();
        info.is_traditional_format = info.is_contiguous;
        
        return info;
    }
    
    /// @brief Get base data pointer (only valid if contiguous)
    [[nodiscard]] T* contiguous_data() const noexcept {
        if (!valid() || !is_contiguous()) return nullptr;
        
        // Find first non-empty row
        const Index pdim = primary_dim();
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] > 0) {
                return static_cast<T*>(data_ptrs[i]);
            }
        }
        return nullptr;
    }
    
    /// @brief Get base indices pointer (only valid if contiguous)
    [[nodiscard]] Index* contiguous_indices() const noexcept {
        if (!valid() || !is_contiguous()) return nullptr;
        
        const Index pdim = primary_dim();
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] > 0) {
                return static_cast<Index*>(indices_ptrs[i]);
            }
        }
        return nullptr;
    }
    
    /// @brief Build traditional row_ptr/col_ptr array
    [[nodiscard]] std::vector<Index> build_offset_array() const {
        const Index pdim = primary_dim();
        std::vector<Index> offsets(pdim + 1);
        
        offsets[0] = 0;
        for (Index i = 0; i < pdim; ++i) {
            offsets[i + 1] = offsets[i] + (valid() ? lengths[i] : 0);
        }
        
        return offsets;
    }

    // =========================================================================
    // Factory: Create Empty/Zero Matrix
    // =========================================================================
    
    [[nodiscard]] static Sparse zeros(Index rows, Index cols) {
        if (rows <= 0 || cols <= 0) return {};
        
        const Index pdim = IsCSR ? rows : cols;
        auto& reg = get_registry();
        
        auto* dp = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* ip = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* len = reg.new_array<Index>(static_cast<size_t>(pdim));
        
        if (!dp || !ip || !len) {
            if (dp) reg.unregister_ptr(dp);
            if (ip) reg.unregister_ptr(ip);
            if (len) reg.unregister_ptr(len);
            return {};
        }
        
        std::memset(dp, 0, sizeof(Pointer) * pdim);
        std::memset(ip, 0, sizeof(Pointer) * pdim);
        std::memset(len, 0, sizeof(Index) * pdim);
        
        return Sparse(dp, ip, len, rows, cols, 0, true);
    }

    // =========================================================================
    // Factory: Create from Row/Column NNZ Counts
    // =========================================================================
    
    [[nodiscard]] static Sparse create(
        Index rows, Index cols,
        std::span<const Index> primary_nnzs,
        BlockStrategy strategy = BlockStrategy::adaptive())
    {
        SCL_CHECK_ARG(rows >= 0 && cols >= 0, "dimensions must be non-negative");

        const Index pdim = IsCSR ? rows : cols;
        SCL_CHECK_ARG(static_cast<Index>(primary_nnzs.size()) == pdim,
                      "nnz array size mismatch");

        if (rows == 0 || cols == 0) return {};

        // Calculate total nnz
        Index total_nnz = 0;
        for (Index i = 0; i < pdim; ++i) {
            SCL_CHECK_ARG(primary_nnzs[i] >= 0, "nnz counts must be non-negative");
            total_nnz += primary_nnzs[i];
        }
        
        if (total_nnz == 0) {
            return zeros(rows, cols);
        }

        auto& reg = get_registry();

        // Allocate metadata arrays
        auto* dp = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* ip = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* len = reg.new_array<Index>(static_cast<size_t>(pdim));

        if (!dp || !ip || !len) {
            if (dp) reg.unregister_ptr(dp);
            if (ip) reg.unregister_ptr(ip);
            if (len) reg.unregister_ptr(len);
            return {};
        }

        // Copy lengths
        for (Index i = 0; i < pdim; ++i) {
            len[i] = primary_nnzs[i];
        }

        // Compute block size
        Index block_size = strategy.compute_block_size(total_nnz, pdim);

        // Allocate data blocks
        bool success = allocate_blocks_impl(dp, ip, len, pdim, 
                                            primary_nnzs, block_size, reg);
        
        if (!success) {
            cleanup_partial_impl(dp, ip, pdim, reg);
            reg.unregister_ptr(dp);
            reg.unregister_ptr(ip);
            reg.unregister_ptr(len);
            return {};
        }

        return Sparse(dp, ip, len, rows, cols, total_nnz, true);
    }

    // =========================================================================
    // Factory: Create from Traditional CSR/CSC Arrays
    // =========================================================================
    
    /// @brief Create from traditional CSR/CSC format (copies data)
    [[nodiscard]] static Sparse from_traditional(
        Index rows, Index cols,
        std::span<const T> values,
        std::span<const Index> indices,
        std::span<const Index> offsets,
        BlockStrategy strategy = BlockStrategy::adaptive())
    {
        const Index pdim = IsCSR ? rows : cols;
        SCL_CHECK_ARG(static_cast<Index>(offsets.size()) == pdim + 1,
                      "offsets size must be primary_dim + 1");
        
        const Index nnz = offsets[pdim];
        SCL_CHECK_ARG(static_cast<Index>(values.size()) >= nnz &&
                      static_cast<Index>(indices.size()) >= nnz,
                      "values/indices size mismatch");

        // Extract row lengths
        std::vector<Index> primary_nnzs(pdim);
        for (Index i = 0; i < pdim; ++i) {
            primary_nnzs[i] = offsets[i + 1] - offsets[i];
        }

        // Create matrix
        Sparse result = create(rows, cols, primary_nnzs, strategy);
        if (!result.valid()) return {};

        // Verify indices are sorted (debug mode only)
#ifdef SCL_DEBUG
        for (Index i = 0; i < pdim; ++i) {
            Index start = offsets[i];
            Index len = offsets[i + 1] - start;
            for (Index k = 1; k < len; ++k) {
                SCL_ASSERT(indices[start + k - 1] < indices[start + k],
                           "from_traditional: indices must be sorted within each row/column");
            }
        }
#endif

        // Copy data
        for (Index i = 0; i < pdim; ++i) {
            if (result.lengths[i] > 0) {
                Index start = offsets[i];
                Index len = result.lengths[i];
                
                std::memcpy(result.data_ptrs[i], &values[start], len * sizeof(T));
                std::memcpy(result.indices_ptrs[i], &indices[start], len * sizeof(Index));
            }
        }

        return result;
    }
    
    /// @brief Wrap existing traditional CSR/CSC arrays (zero-copy, external ownership)
    /// @warning Caller must ensure arrays outlive the Sparse object
    [[nodiscard]] static Sparse wrap_traditional(
        Index rows, Index cols,
        T* values,
        Index* indices, 
        std::span<const Index> offsets)
    {
        const Index pdim = IsCSR ? rows : cols;
        SCL_CHECK_ARG(static_cast<Index>(offsets.size()) == pdim + 1,
                      "offsets size must be primary_dim + 1");
        SCL_CHECK_ARG(values != nullptr && indices != nullptr,
                      "values and indices must not be null");

        const Index nnz = offsets[pdim];
        auto& reg = get_registry();

        // Allocate only metadata (pointers to external data)
        auto* dp = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* ip = reg.new_array<Pointer>(static_cast<size_t>(pdim));
        auto* len = reg.new_array<Index>(static_cast<size_t>(pdim));

        if (!dp || !ip || !len) {
            if (dp) reg.unregister_ptr(dp);
            if (ip) reg.unregister_ptr(ip);
            if (len) reg.unregister_ptr(len);
            return {};
        }

        // Set up pointers into external arrays
        for (Index i = 0; i < pdim; ++i) {
            Index start = offsets[i];
            len[i] = offsets[i + 1] - start;
            dp[i] = len[i] > 0 ? (values + start) : nullptr;
            ip[i] = len[i] > 0 ? (indices + start) : nullptr;
        }

        // Note: data is NOT registered, so unregister_all() will only free metadata
        return Sparse(dp, ip, len, rows, cols, nnz, false);
    }

    // =========================================================================
    // Factory: Create from COO Format
    // =========================================================================
    
    /// @brief Create sparse matrix from COO (Coordinate) format
    /// @note Duplicate coordinates are NOT summed - each (row, col) entry
    ///       is stored separately. If duplicates exist, the last value for
    ///       each coordinate will be retained after sorting.
    /// @note For COO with duplicates that should be summed, preprocess the
    ///       input to aggregate values before calling this function.
    [[nodiscard]] static Sparse from_coo(
        Index rows, Index cols,
        std::span<const Index> row_indices,
        std::span<const Index> col_indices,
        std::span<const T> values,
        BlockStrategy strategy = BlockStrategy::adaptive())
    {
        const Index nnz = static_cast<Index>(values.size());
        SCL_CHECK_ARG(static_cast<Index>(row_indices.size()) == nnz &&
                      static_cast<Index>(col_indices.size()) == nnz,
                      "COO arrays size mismatch");

        if (nnz == 0) return zeros(rows, cols);

        const Index pdim = IsCSR ? rows : cols;

        // Count nnz per row/col
        std::vector<Index> primary_nnzs(pdim, 0);
        for (Index i = 0; i < nnz; ++i) {
            Index pidx = IsCSR ? row_indices[i] : col_indices[i];
            SCL_CHECK_ARG(pidx >= 0 && pidx < pdim, "index out of bounds");
            ++primary_nnzs[pidx];
        }

        // Create matrix
        Sparse result = create(rows, cols, primary_nnzs, strategy);
        if (!result.valid()) return {};

        // Track insertion position for each row/col
        std::vector<Index> insert_pos(pdim, 0);

        // Fill data (first pass: unsorted)
        for (Index i = 0; i < nnz; ++i) {
            Index pidx = IsCSR ? row_indices[i] : col_indices[i];
            Index sidx = IsCSR ? col_indices[i] : row_indices[i];
            Index pos = insert_pos[pidx]++;
            
            static_cast<T*>(result.data_ptrs[pidx])[pos] = values[i];
            static_cast<Index*>(result.indices_ptrs[pidx])[pos] = sidx;
        }

        // Sort each row/col by secondary index
        result.sort_indices();

        return result;
    }

    // =========================================================================
    // Factory: Create from Dense Matrix
    // =========================================================================
    
    /// @brief Create sparse matrix from dense matrix (row-major layout)
    template <typename Pred = std::nullptr_t>
    [[nodiscard]] static Sparse from_dense(
        Index rows, Index cols,
        std::span<const T> data,
        Pred&& is_nonzero = nullptr,
        BlockStrategy strategy = BlockStrategy::adaptive())
    {
        SCL_CHECK_ARG(static_cast<Index>(data.size()) >= rows * cols,
                      "dense data size mismatch");

        const Index pdim = IsCSR ? rows : cols;
        const Index sdim = IsCSR ? cols : rows;

        // Determine which elements are non-zero
        auto check_nonzero = [&](T val) -> bool {
            if constexpr (std::is_same_v<std::decay_t<Pred>, std::nullptr_t>) {
                return val != T{0};
            } else {
                return is_nonzero(val);
            }
        };

        // Count nnz per row/col
        std::vector<Index> primary_nnzs(pdim, 0);
        
        if constexpr (IsCSR) {
            // Row-major dense -> CSR
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    if (check_nonzero(data[i * cols + j])) {
                        ++primary_nnzs[i];
                    }
                }
            }
        } else {
            // Row-major dense -> CSC
            for (Index j = 0; j < cols; ++j) {
                for (Index i = 0; i < rows; ++i) {
                    if (check_nonzero(data[i * cols + j])) {
                        ++primary_nnzs[j];
                    }
                }
            }
        }

        // Create matrix
        Sparse result = create(rows, cols, primary_nnzs, strategy);
        if (!result.valid()) return {};

        // Fill data
        std::vector<Index> insert_pos(pdim, 0);
        
        if constexpr (IsCSR) {
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    T val = data[i * cols + j];
                    if (check_nonzero(val)) {
                        Index pos = insert_pos[i]++;
                        static_cast<T*>(result.data_ptrs[i])[pos] = val;
                        static_cast<Index*>(result.indices_ptrs[i])[pos] = j;
                    }
                }
            }
        } else {
            for (Index j = 0; j < cols; ++j) {
                for (Index i = 0; i < rows; ++i) {
                    T val = data[i * cols + j];
                    if (check_nonzero(val)) {
                        Index pos = insert_pos[j]++;
                        static_cast<T*>(result.data_ptrs[j])[pos] = val;
                        static_cast<Index*>(result.indices_ptrs[j])[pos] = i;
                    }
                }
            }
        }

        return result;
    }

    // =========================================================================
    // Factory: Identity Matrix
    // =========================================================================
    
    [[nodiscard]] static Sparse identity(Index n, BlockStrategy strategy = BlockStrategy::adaptive()) {
        if (n <= 0) return {};

        std::vector<Index> nnzs(n, 1);
        Sparse result = create(n, n, nnzs, strategy);
        if (!result.valid()) return {};

        for (Index i = 0; i < n; ++i) {
            static_cast<T*>(result.data_ptrs[i])[0] = T{1};
            static_cast<Index*>(result.indices_ptrs[i])[0] = i;
        }

        return result;
    }

    // =========================================================================
    // Clone and Conversion
    // =========================================================================
    
    /// @brief Deep copy with optional new block strategy
    [[nodiscard]] Sparse clone(
        BlockStrategy strategy = BlockStrategy::adaptive()) const
    {
        if (!valid()) return {};

        const Index pdim = primary_dim();
        std::vector<Index> nnzs(pdim);
        for (Index i = 0; i < pdim; ++i) {
            nnzs[i] = lengths[i];
        }

        Sparse result = create(rows_, cols_, nnzs, strategy);
        if (!result.valid()) return {};

        // Copy data
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] > 0) {
                std::memcpy(result.data_ptrs[i], data_ptrs[i], 
                           lengths[i] * sizeof(T));
                std::memcpy(result.indices_ptrs[i], indices_ptrs[i],
                           lengths[i] * sizeof(Index));
            }
        }

        return result;
    }
    
    /// @brief Convert to contiguous storage (if not already)
    [[nodiscard]] Sparse to_contiguous() const {
        if (!valid()) return {};
        if (is_contiguous()) return clone(BlockStrategy::contiguous());
        return clone(BlockStrategy::contiguous());
    }
    
    /// @brief Convert to transposed format (CSR <-> CSC)
    [[nodiscard]] TransposeType transpose() const {
        if (!valid()) return {};

        const Index new_pdim = secondary_dim();
        const Index new_sdim = primary_dim();

        // Count nnz per new primary dimension
        std::vector<Index> new_nnzs(new_pdim, 0);
        
        for (Index i = 0; i < primary_dim(); ++i) {
            auto indices = primary_indices(i);
            for (Index idx : indices) {
                ++new_nnzs[idx];
            }
        }

        // Create transposed matrix
        TransposeType result = TransposeType::create(
            IsCSR ? cols_ : rows_,
            IsCSR ? rows_ : cols_,
            new_nnzs,
            BlockStrategy::adaptive());
            
        if (!result.valid()) return {};

        // Fill transposed data
        std::vector<Index> insert_pos(new_pdim, 0);
        
        for (Index i = 0; i < primary_dim(); ++i) {
            auto indices = primary_indices(i);
            auto values = primary_values(i);
            
            for (Index k = 0; k < lengths[i]; ++k) {
                Index j = indices[k];
                Index pos = insert_pos[j]++;
                
                static_cast<T*>(result.data_ptrs[j])[pos] = values[k];
                static_cast<Index*>(result.indices_ptrs[j])[pos] = i;
            }
        }

        // Sort each row/col
        result.sort_indices();

        return result;
    }

    // =========================================================================
    // Slicing: View (Shared Memory)
    // =========================================================================
    
    /// @brief Create a row slice that shares memory with the original
    /// @note Modifications to the slice affect the original
    /// @warning Cannot create view of wrapped matrix (created with wrap_traditional)
    ///          Use row_slice_copy instead for wrapped matrices
    [[nodiscard]] Sparse row_slice_view(std::span<const Index> row_indices) const
        requires (IsCSR)
    {
        if (!valid() || row_indices.empty()) return {};
        
        // Wrapped matrices don't have proper refcounting, so views are unsafe
        if (!owns_data_) {
            SCL_ASSERT(false, 
                "Cannot create view of wrapped matrix - use row_slice_copy instead");
            return {};  // Release mode: return empty matrix
        }

        auto& reg = get_registry();
        const Index new_rows = static_cast<Index>(row_indices.size());

        // Allocate new metadata
        auto* new_dp = reg.new_array<Pointer>(new_rows);
        auto* new_ip = reg.new_array<Pointer>(new_rows);
        auto* new_len = reg.new_array<Index>(new_rows);

        if (!new_dp || !new_ip || !new_len) {
            if (new_dp) reg.unregister_ptr(new_dp);
            if (new_ip) reg.unregister_ptr(new_ip);
            if (new_len) reg.unregister_ptr(new_len);
            return {};
        }

        // Copy pointers and increment refcounts
        Index new_nnz = 0;
        std::vector<void*> data_aliases, idx_aliases;
        data_aliases.reserve(new_rows);
        idx_aliases.reserve(new_rows);

        for (Index i = 0; i < new_rows; ++i) {
            Index src = row_indices[i];
            SCL_ASSERT(src >= 0 && src < rows_, "row index out of bounds");

            new_dp[i] = data_ptrs[src];
            new_ip[i] = indices_ptrs[src];
            new_len[i] = lengths[src];
            new_nnz += lengths[src];

            if (new_dp[i]) data_aliases.push_back(new_dp[i]);
            if (new_ip[i]) idx_aliases.push_back(new_ip[i]);
        }

        // Increment reference counts for shared buffers by registering new aliases
        if (!data_aliases.empty()) {
            increment_alias_refcounts_safe(reg, data_aliases);
        }
        if (!idx_aliases.empty()) {
            increment_alias_refcounts_safe(reg, idx_aliases);
        }

        // Create view with is_view_=true (owns_data_=true, is_view_=true)
        return Sparse(new_dp, new_ip, new_len, new_rows, cols_, new_nnz, true, true);
    }
    
    /// @brief Create a contiguous row range view
    [[nodiscard]] Sparse row_range_view(Index start, Index end) const
        requires (IsCSR)
    {
        SCL_CHECK_ARG(start >= 0 && end <= rows_ && start <= end,
                      "invalid row range");
        
        if (start == end) return zeros(0, cols_);
        
        std::vector<Index> indices(end - start);
        std::iota(indices.begin(), indices.end(), start);
        return row_slice_view(indices);
    }
    
    /// @brief Column slice view (CSC version)
    /// @warning Cannot create view of wrapped matrix (created with wrap_traditional)
    ///          Use col_slice_copy instead for wrapped matrices
    [[nodiscard]] Sparse col_slice_view(std::span<const Index> col_indices) const
        requires (!IsCSR)
    {
        if (!valid() || col_indices.empty()) return {};
        
        // Wrapped matrices don't have proper refcounting, so views are unsafe
        if (!owns_data_) {
            SCL_ASSERT(false,
                "Cannot create view of wrapped matrix - use col_slice_copy instead");
            return {};  // Release mode: return empty matrix
        }

        auto& reg = get_registry();
        const Index new_cols = static_cast<Index>(col_indices.size());

        auto* new_dp = reg.new_array<Pointer>(new_cols);
        auto* new_ip = reg.new_array<Pointer>(new_cols);
        auto* new_len = reg.new_array<Index>(new_cols);

        if (!new_dp || !new_ip || !new_len) {
            if (new_dp) reg.unregister_ptr(new_dp);
            if (new_ip) reg.unregister_ptr(new_ip);
            if (new_len) reg.unregister_ptr(new_len);
            return {};
        }

        Index new_nnz = 0;
        std::vector<void*> data_aliases, idx_aliases;
        data_aliases.reserve(new_cols);
        idx_aliases.reserve(new_cols);

        for (Index i = 0; i < new_cols; ++i) {
            Index src = col_indices[i];
            SCL_ASSERT(src >= 0 && src < cols_, "col index out of bounds");

            new_dp[i] = data_ptrs[src];
            new_ip[i] = indices_ptrs[src];
            new_len[i] = lengths[src];
            new_nnz += lengths[src];

            if (new_dp[i]) data_aliases.push_back(new_dp[i]);
            if (new_ip[i]) idx_aliases.push_back(new_ip[i]);
        }

        if (!data_aliases.empty()) {
            increment_alias_refcounts_safe(reg, data_aliases);
        }
        if (!idx_aliases.empty()) {
            increment_alias_refcounts_safe(reg, idx_aliases);
        }

        // Create view with is_view_=true (owns_data_=true, is_view_=true)
        return Sparse(new_dp, new_ip, new_len, rows_, new_cols, new_nnz, true, true);
    }

    // =========================================================================
    // Slicing: Copy (Independent Memory)
    // =========================================================================
    
    /// @brief Create a row slice with copied (independent) data
    /// @param strategy Block strategy for the new matrix
    [[nodiscard]] Sparse row_slice_copy(
        std::span<const Index> row_indices,
        BlockStrategy strategy = BlockStrategy::adaptive()) const
        requires (IsCSR)
    {
        if (!valid() || row_indices.empty()) return {};

        const Index new_rows = static_cast<Index>(row_indices.size());

        // Gather nnz counts
        std::vector<Index> new_nnzs(new_rows);
        for (Index i = 0; i < new_rows; ++i) {
            Index src = row_indices[i];
            SCL_ASSERT(src >= 0 && src < rows_, "row index out of bounds");
            new_nnzs[i] = lengths[src];
        }

        // Create new matrix
        Sparse result = create(new_rows, cols_, new_nnzs, strategy);
        if (!result.valid()) return {};

        // Copy data
        for (Index i = 0; i < new_rows; ++i) {
            Index src = row_indices[i];
            if (lengths[src] > 0) {
                std::memcpy(result.data_ptrs[i], data_ptrs[src],
                           lengths[src] * sizeof(T));
                std::memcpy(result.indices_ptrs[i], indices_ptrs[src],
                           lengths[src] * sizeof(Index));
            }
        }

        return result;
    }
    
    /// @brief Create a contiguous row range copy
    [[nodiscard]] Sparse row_range_copy(
        Index start, Index end,
        BlockStrategy strategy = BlockStrategy::adaptive()) const
        requires (IsCSR)
    {
        SCL_CHECK_ARG(start >= 0 && end <= rows_ && start <= end,
                      "invalid row range");
        
        if (start == end) return zeros(0, cols_);
        
        std::vector<Index> indices(end - start);
        std::iota(indices.begin(), indices.end(), start);
        return row_slice_copy(indices, strategy);
    }
    
    /// @brief Column slice copy (CSC version)
    [[nodiscard]] Sparse col_slice_copy(
        std::span<const Index> col_indices,
        BlockStrategy strategy = BlockStrategy::adaptive()) const
        requires (!IsCSR)
    {
        if (!valid() || col_indices.empty()) return {};

        const Index new_cols = static_cast<Index>(col_indices.size());

        std::vector<Index> new_nnzs(new_cols);
        for (Index i = 0; i < new_cols; ++i) {
            Index src = col_indices[i];
            SCL_ASSERT(src >= 0 && src < cols_, "col index out of bounds");
            new_nnzs[i] = lengths[src];
        }

        Sparse result = create(rows_, new_cols, new_nnzs, strategy);
        if (!result.valid()) return {};

        for (Index i = 0; i < new_cols; ++i) {
            Index src = col_indices[i];
            if (lengths[src] > 0) {
                std::memcpy(result.data_ptrs[i], data_ptrs[src],
                           lengths[src] * sizeof(T));
                std::memcpy(result.indices_ptrs[i], indices_ptrs[src],
                           lengths[src] * sizeof(Index));
            }
        }

        return result;
    }

    // =========================================================================
    // Secondary Dimension Slicing (Requires Reconstruction)
    // =========================================================================
    
    /// @brief Column slice for CSR (requires reconstruction)
    /// @note This is an expensive operation: O(nnz)
    [[nodiscard]] Sparse col_slice(
        std::span<const Index> col_indices,
        BlockStrategy strategy = BlockStrategy::adaptive()) const
        requires (IsCSR)
    {
        if (!valid() || col_indices.empty()) return {};

        const Index new_cols = static_cast<Index>(col_indices.size());

        // Build column index mapping: old_col -> new_col (-1 if not included)
        std::vector<Index> col_map(cols_, -1);
        for (Index j = 0; j < new_cols; ++j) {
            Index src = col_indices[j];
            SCL_ASSERT(src >= 0 && src < cols_, "col index out of bounds");
            col_map[src] = j;
        }

        // Count nnz per row in new matrix
        std::vector<Index> new_nnzs(rows_, 0);
        for (Index i = 0; i < rows_; ++i) {
            auto indices = row_indices(i);
            for (Index col : indices) {
                if (col_map[col] >= 0) {
                    ++new_nnzs[i];
                }
            }
        }

        // Create new matrix
        Sparse result = create(rows_, new_cols, new_nnzs, strategy);
        if (!result.valid()) return {};

        // Fill data
        for (Index i = 0; i < rows_; ++i) {
            auto old_indices = row_indices(i);
            auto old_values = row_values(i);
            Index pos = 0;
            
            for (Index k = 0; k < lengths[i]; ++k) {
                Index new_col = col_map[old_indices[k]];
                if (new_col >= 0) {
                    static_cast<T*>(result.data_ptrs[i])[pos] = old_values[k];
                    static_cast<Index*>(result.indices_ptrs[i])[pos] = new_col;
                    ++pos;
                }
            }
        }

        // Sort indices (column order may have changed)
        result.sort_indices();

        return result;
    }
    
    /// @brief Row slice for CSC (requires reconstruction)
    [[nodiscard]] Sparse row_slice(
        std::span<const Index> row_indices,
        BlockStrategy strategy = BlockStrategy::adaptive()) const
        requires (!IsCSR)
    {
        if (!valid() || row_indices.empty()) return {};

        const Index new_rows = static_cast<Index>(row_indices.size());

        std::vector<Index> row_map(rows_, -1);
        for (Index i = 0; i < new_rows; ++i) {
            Index src = row_indices[i];
            SCL_ASSERT(src >= 0 && src < rows_, "row index out of bounds");
            row_map[src] = i;
        }

        std::vector<Index> new_nnzs(cols_, 0);
        for (Index j = 0; j < cols_; ++j) {
            auto indices = col_indices(j);
            for (Index row : indices) {
                if (row_map[row] >= 0) {
                    ++new_nnzs[j];
                }
            }
        }

        Sparse result = create(new_rows, cols_, new_nnzs, strategy);
        if (!result.valid()) return {};

        for (Index j = 0; j < cols_; ++j) {
            auto old_indices = col_indices(j);
            auto old_values = col_values(j);
            Index pos = 0;
            
            for (Index k = 0; k < lengths[j]; ++k) {
                Index new_row = row_map[old_indices[k]];
                if (new_row >= 0) {
                    static_cast<T*>(result.data_ptrs[j])[pos] = old_values[k];
                    static_cast<Index*>(result.indices_ptrs[j])[pos] = new_row;
                    ++pos;
                }
            }
        }

        result.sort_indices();

        return result;
    }

    // =========================================================================
    // In-place Operations
    // =========================================================================
    
    /// @brief Sort indices within each row/column
    void sort_indices() {
        if (!valid()) return;

        const Index pdim = primary_dim();
        
        // Find maximum row/column length for buffer preallocation
        Index max_len = 0;
        for (Index i = 0; i < pdim; ++i) {
            max_len = std::max(max_len, lengths[i]);
        }
        
        if (max_len <= 1) return;  // Already sorted or empty
        
        // Preallocate buffers (reused for all rows/columns)
        std::vector<Index> perm(max_len);
        std::vector<T> temp_vals(max_len);
        std::vector<Index> temp_idxs(max_len);
        
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] <= 1) continue;

            auto* vals = static_cast<T*>(data_ptrs[i]);
            auto* idxs = static_cast<Index*>(indices_ptrs[i]);
            const Index len = lengths[i];

            // Create permutation array
            std::iota(perm.begin(), perm.begin() + len, 0);
            std::sort(perm.begin(), perm.begin() + len, [&](Index a, Index b) {
                return idxs[a] < idxs[b];
            });

            // Apply permutation (using preallocated temp arrays)
            for (Index k = 0; k < len; ++k) {
                temp_vals[k] = vals[perm[k]];
                temp_idxs[k] = idxs[perm[k]];
            }
            
            std::memcpy(vals, temp_vals.data(), len * sizeof(T));
            std::memcpy(idxs, temp_idxs.data(), len * sizeof(Index));
        }
    }
    
    /// @brief Verify that all indices are sorted (debug/validation)
    /// @note In production, indices are guaranteed sorted per CSR/CSC specification.
    ///       This method is for debugging or validating external data.
    [[nodiscard]] bool verify_sorted() const noexcept {
        if (!valid()) return true;

        const Index pdim = primary_dim();
        
        for (Index i = 0; i < pdim; ++i) {
            if (lengths[i] <= 1) continue;
            
            auto indices = primary_indices(i);
            for (Index k = 1; k < lengths[i]; ++k) {
                if (indices[k] <= indices[k-1]) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /// @brief Alias for verify_sorted() (backward compatibility)
    [[nodiscard]] bool is_sorted() const noexcept {
        return verify_sorted();
    }
    
    /// @brief Scale all values by a constant
    void scale(T factor) {
        if (!valid()) return;

        const Index pdim = primary_dim();
        for (Index i = 0; i < pdim; ++i) {
            auto values = primary_values(i);
            for (Index k = 0; k < lengths[i]; ++k) {
                values[k] *= factor;
            }
        }
    }

    // =========================================================================
    // Export to Traditional Format
    // =========================================================================
    
    struct TraditionalFormat {
        std::vector<T> values;
        std::vector<Index> indices;
        std::vector<Index> offsets;
    };
    
    /// @brief Export to traditional CSR/CSC arrays
    [[nodiscard]] TraditionalFormat to_traditional() const {
        TraditionalFormat result;
        
        if (!valid()) return result;

        const Index pdim = primary_dim();
        
        result.values.reserve(nnz_);
        result.indices.reserve(nnz_);
        result.offsets.resize(pdim + 1);
        
        result.offsets[0] = 0;
        for (Index i = 0; i < pdim; ++i) {
            auto vals = primary_values(i);
            auto idxs = primary_indices(i);
            
            result.values.insert(result.values.end(), vals.begin(), vals.end());
            result.indices.insert(result.indices.end(), idxs.begin(), idxs.end());
            result.offsets[i + 1] = result.offsets[i] + lengths[i];
        }
        
        return result;
    }
    
    /// @brief Export to dense matrix (row-major)
    [[nodiscard]] std::vector<T> to_dense() const {
        std::vector<T> result(rows_ * cols_, T{0});
        
        if (!valid()) return result;

        if constexpr (IsCSR) {
            for (Index i = 0; i < rows_; ++i) {
                auto vals = row_values(i);
                auto cols = row_indices(i);
                for (Index k = 0; k < lengths[i]; ++k) {
                    result[i * cols_ + cols[k]] = vals[k];
                }
            }
        } else {
            for (Index j = 0; j < cols_; ++j) {
                auto vals = col_values(j);
                auto rows = col_indices(j);
                for (Index k = 0; k < lengths[j]; ++k) {
                    result[rows[k] * cols_ + j] = vals[k];
                }
            }
        }
        
        return result;
    }

    // =========================================================================
    // Memory Management
    // =========================================================================
    
    /// @brief Release all registered memory
    void unregister_all() {
        release_resources();
    }
    
private:
    /// @brief Internal method to release resources (used by destructor and move assignment)
    void release_resources() {
        if (!valid()) return;

        const Index pdim = primary_dim();
        auto& reg = get_registry();

        // Only unregister data aliases if we own the data
        if (owns_data_) {
            // Collect all non-null aliases
            std::vector<void*> aliases;
            aliases.reserve(pdim * 2);

            for (Index i = 0; i < pdim; ++i) {
                if (data_ptrs[i]) aliases.push_back(data_ptrs[i]);
                if (indices_ptrs[i]) aliases.push_back(indices_ptrs[i]);
            }

            if (!aliases.empty()) {
                if (is_view_) {
                    // View: only decrement refcounts, don't remove alias mappings
                    // The original matrix owns the alias mappings
                    reg.decrement_buffer_refcounts(aliases);
                } else {
                    // Original matrix: remove alias mappings and decrement refcounts
                    reg.unregister_aliases(aliases);
                }
            }
        }

        // Free metadata arrays (always owned)
        reg.unregister_ptr(data_ptrs);
        reg.unregister_ptr(indices_ptrs);
        reg.unregister_ptr(lengths);

        // Reset state
        data_ptrs = nullptr;
        indices_ptrs = nullptr;
        lengths = nullptr;
        rows_ = cols_ = nnz_ = 0;
        owns_data_ = true;
        is_view_ = false;
    }
    
public:
    
    /// @brief Release memory only if this is the sole owner
    /// @return true if memory was released, false if still shared
    bool try_unregister() {
        if (!valid()) return true;

        // Check if any data is shared
        auto& reg = get_registry();
        const Index pdim = primary_dim();
        
        for (Index i = 0; i < pdim; ++i) {
            if (data_ptrs[i]) {
                BufferID id = reg.get_buffer_id(data_ptrs[i]);
                if (id != 0 && reg.get_refcount(id) > 1) {
                    return false;
                }
            }
        }
        
        unregister_all();
        return true;
    }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================
    
    /// @brief Safely increment reference counts for aliases in a view
    /// @note When creating a view, the new metadata arrays contain pointer values
    ///       that are identical to the original (same memory addresses). These
    ///       pointers are already registered as aliases. We need to increment
    ///       the reference count for each unique buffer that the view references.
    /// @note If an alias is not registered (id == 0), it means the original matrix
    ///       was created with wrap_traditional, and we can't create a view safely.
    ///       In this case, the function silently continues, but the view won't
    ///       prevent the original from being destroyed.
    static void increment_alias_refcounts_safe(
        Registry& reg, std::span<void* const> aliases)
    {
        if (aliases.empty()) return;
        
        // Collect unique buffer IDs and count how many aliases reference each buffer
        std::unordered_map<BufferID, std::uint32_t> increments;
        
        for (void* alias : aliases) {
            if (!alias) continue;
            BufferID id = reg.get_buffer_id(alias);
            if (id != 0) {
                ++increments[id];
            }
            // If id == 0, the alias is not registered (e.g., from wrap_traditional)
            // We silently ignore it - the view will work but won't have proper refcounting
        }
        
        // Increment reference counts for each unique buffer
        for (const auto& [buffer_id, increment] : increments) {
            reg.increment_buffer_refcount(buffer_id, increment);
        }
    }
    
    static bool allocate_blocks_impl(
        Pointer* dp, Pointer* ip, Index* len,
        Index primary_dim, std::span<const Index> nnzs,
        Index block_size, Registry& reg)
    {
        Index row_start = 0;

        while (row_start < primary_dim) {
            // Determine block extent
            Index block_nnz = 0;
            Index row_end = row_start;

            while (row_end < primary_dim) {
                Index row_len = nnzs[row_end];
                // Ensure at least one row per block
                if (block_nnz > 0 && block_nnz + row_len > block_size) {
                    break;
                }
                block_nnz += row_len;
                ++row_end;
            }

            // Handle empty block
            if (block_nnz == 0) {
                for (Index i = row_start; i < row_end; ++i) {
                    dp[i] = nullptr;
                    ip[i] = nullptr;
                }
                row_start = row_end;
                continue;
            }

            // Allocate data block
            T* data_block = new (std::nothrow) T[static_cast<size_t>(block_nnz)]();
            Index* idx_block = new (std::nothrow) Index[static_cast<size_t>(block_nnz)]();

            if (!data_block || !idx_block) {
                delete[] data_block;
                delete[] idx_block;
                return false;
            }

            // Build alias lists
            std::vector<void*> data_aliases;
            std::vector<void*> idx_aliases;
            data_aliases.reserve(row_end - row_start);
            idx_aliases.reserve(row_end - row_start);

            Index offset = 0;
            for (Index i = row_start; i < row_end; ++i) {
                if (len[i] > 0) {
                    dp[i] = data_block + offset;
                    ip[i] = idx_block + offset;
                    data_aliases.push_back(dp[i]);
                    idx_aliases.push_back(ip[i]);
                } else {
                    dp[i] = nullptr;
                    ip[i] = nullptr;
                }
                offset += len[i];
            }

            // Register as shared buffers
            if (!data_aliases.empty()) {
                if (!reg.register_buffer_with_aliases(
                        data_block, block_nnz * sizeof(T),
                        data_aliases, AllocType::ArrayNew)) {
                    delete[] data_block;
                    delete[] idx_block;
                    return false;
                }
            } else {
                delete[] data_block;
            }

            if (!idx_aliases.empty()) {
                if (!reg.register_buffer_with_aliases(
                        idx_block, block_nnz * sizeof(Index),
                        idx_aliases, AllocType::ArrayNew)) {
                    // Data block already registered, will be cleaned up
                    delete[] idx_block;
                    return false;
                }
            } else {
                delete[] idx_block;
            }

            row_start = row_end;
        }

        return true;
    }

    static void cleanup_partial_impl(
        Pointer* dp, Pointer* ip, Index primary_dim, Registry& reg)
    {
        std::vector<void*> aliases;
        aliases.reserve(primary_dim * 2);

        for (Index i = 0; i < primary_dim; ++i) {
            if (dp[i]) aliases.push_back(dp[i]);
            if (ip[i]) aliases.push_back(ip[i]);
        }

        if (!aliases.empty()) {
            reg.unregister_aliases(aliases);
        }
    }
};

// =============================================================================
// Type Aliases
// =============================================================================

template <typename T>
using SparseCSR = Sparse<T, true>;

template <typename T>
using SparseCSC = Sparse<T, false>;

// Default types
using CSR = Sparse<Real, true>;
using CSC = Sparse<Real, false>;
using CSRf = Sparse<float, true>;
using CSCf = Sparse<float, false>;
using CSRd = Sparse<double, true>;
using CSCd = Sparse<double, false>;
using CSRi = Sparse<int, true>;
using CSCi = Sparse<int, false>;

// =============================================================================
// Concept Verification
// =============================================================================

static_assert(CSRLike<CSR>);
static_assert(CSCLike<CSC>);
static_assert(SparseLike<CSR>);
static_assert(SparseLike<CSC>);

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Concatenate matrices vertically (CSR only)
template <typename T>
[[nodiscard]] Sparse<T, true> vstack(
    std::span<const Sparse<T, true>> matrices,
    BlockStrategy strategy = BlockStrategy::adaptive())
{
    if (matrices.empty()) return {};

    // Check first matrix validity
    if (!matrices[0].valid()) return {};

    // Validate dimensions
    Index cols = matrices[0].cols();
    Index total_rows = 0;
    Index total_nnz = 0;

    for (const auto& mat : matrices) {
        if (!mat.valid()) return {};  // All matrices must be valid
        SCL_CHECK_ARG(mat.cols() == cols, "column dimension mismatch");
        total_rows += mat.rows();
        total_nnz += mat.nnz();
    }

    // Gather nnz counts
    std::vector<Index> nnzs;
    nnzs.reserve(total_rows);

    for (const auto& mat : matrices) {
        for (Index i = 0; i < mat.rows(); ++i) {
            nnzs.push_back(mat.lengths[i]);
        }
    }

    // Create result
    Sparse<T, true> result = Sparse<T, true>::create(total_rows, cols, nnzs, strategy);
    if (!result.valid()) return {};

    // Copy data
    Index dst_row = 0;
    for (const auto& mat : matrices) {
        for (Index i = 0; i < mat.rows(); ++i) {
            if (mat.lengths[i] > 0) {
                std::memcpy(result.data_ptrs[dst_row], mat.data_ptrs[i],
                           mat.lengths[i] * sizeof(T));
                std::memcpy(result.indices_ptrs[dst_row], mat.indices_ptrs[i],
                           mat.lengths[i] * sizeof(Index));
            }
            ++dst_row;
        }
    }

    return result;
}

/// @brief Concatenate matrices horizontally (CSC only)
template <typename T>
[[nodiscard]] Sparse<T, false> hstack(
    std::span<const Sparse<T, false>> matrices,
    BlockStrategy strategy = BlockStrategy::adaptive())
{
    if (matrices.empty()) return {};

    // Check first matrix validity
    if (!matrices[0].valid()) return {};

    Index rows = matrices[0].rows();
    Index total_cols = 0;
    Index total_nnz = 0;

    for (const auto& mat : matrices) {
        if (!mat.valid()) return {};  // All matrices must be valid
        SCL_CHECK_ARG(mat.rows() == rows, "row dimension mismatch");
        total_cols += mat.cols();
        total_nnz += mat.nnz();
    }

    std::vector<Index> nnzs;
    nnzs.reserve(total_cols);

    for (const auto& mat : matrices) {
        for (Index j = 0; j < mat.cols(); ++j) {
            nnzs.push_back(mat.lengths[j]);
        }
    }

    Sparse<T, false> result = Sparse<T, false>::create(rows, total_cols, nnzs, strategy);
    if (!result.valid()) return {};

    Index dst_col = 0;
    for (const auto& mat : matrices) {
        for (Index j = 0; j < mat.cols(); ++j) {
            if (mat.lengths[j] > 0) {
                std::memcpy(result.data_ptrs[dst_col], mat.data_ptrs[j],
                           mat.lengths[j] * sizeof(T));
                std::memcpy(result.indices_ptrs[dst_col], mat.indices_ptrs[j],
                           mat.lengths[j] * sizeof(Index));
            }
            ++dst_col;
        }
    }

    return result;
}

} // namespace scl
