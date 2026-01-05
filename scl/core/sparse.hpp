#pragma once

/**
 * @file scl/core/sparse.hpp
 * @brief Sparse Matrix with Span-based Storage.
 *
 * This header provides:
 *   - Sparse<ValueT, IndexT, IsCSR>: CSR/CSC sparse matrix with flexible storage
 *   - CSRMatrix<T>, CSCMatrix<T>: Type aliases for convenience
 *
 * ## Design Philosophy
 *
 * Each row (CSR) or column (CSC) is stored as an independent pair of:
 *   - Span<ValueT>: Non-zero values
 *   - Span<IndexT>: Column (CSR) or row (CSC) indices
 *
 * This design enables:
 *   - Zero-copy slicing via Span's reference counting
 *   - Flexible memory management (owned/shared/view modes)
 *   - Natural row/column independence
 *
 * ## Memory Layout
 *
 * ```
 * Sparse<T, IndexT, IsCSR>:
 *   - values_:  std::vector<Span<T>>      (primary_dim elements)
 *   - indices_: std::vector<Span<IndexT>> (primary_dim elements)
 *   - rows_, cols_: IndexT
 * ```
 *
 * Each Span manages its own lifecycle (24 bytes per span).
 * Total overhead: ~48 bytes per row/column + vector overhead.
 *
 * ## Invariants
 *
 *   - Indices within each row (CSR) or column (CSC) are strictly ascending
 *   - values_[i].size() == indices_[i].size() for all i
 *   - All indices are within bounds [0, secondary_dim)
 *   - values_.size() == indices_.size() == primary_dim
 *
 * ## Thread Safety
 *
 *   - Read operations are thread-safe when Spans are in shared mode
 *   - Write operations require external synchronization
 *   - Copy/move operations are not thread-safe
 *
 * ## Buffer Allocation Strategies
 *
 * Sparse supports flexible buffer allocation strategies to balance memory
 * fragmentation and performance:
 *
 * - **Fragmented**: Each row/column uses separate buffer (best for slicing)
 * - **SingleBuffer**: All rows/columns share one buffer (best locality)
 * - **MinBufferSize**: Group rows/cols by minimum buffer size (balanced)
 * - **BufferCount**: Divide into N buffers (predictable memory)
 * - **Auto**: Platform-optimal automatic selection (default)
 */

#include "scl/core/error.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/span.hpp"
#include "scl/core/threading.hpp"
#include "scl/core/type.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace scl {

// =============================================================================
// Constants
// =============================================================================

namespace detail {

/// @brief Threshold for parallel execution in sparse operations
inline constexpr Size kSparseParallelThreshold = 100;

}  // namespace detail

// =============================================================================
// SECTION 1: Forward Declarations
// =============================================================================

template <typename ValueT, typename IndexT, bool IsCSR>
class Sparse;

template <typename T, typename IndexT = Index>
using CSRMatrix = Sparse<T, IndexT, true>;

template <typename T, typename IndexT = Index>
using CSCMatrix = Sparse<T, IndexT, false>;

// Default types
using CSR = CSRMatrix<Real>;
using CSC = CSCMatrix<Real>;
using CSRf = CSRMatrix<float>;
using CSCf = CSCMatrix<float>;
using CSRd = CSRMatrix<double>;
using CSCd = CSCMatrix<double>;

// =============================================================================
// SECTION 2: Sparse Layout Info
// =============================================================================

/// @brief Information about sparse matrix memory layout
struct SparseLayoutInfo {
  std::size_t values_bytes = 0;   ///< Total bytes for all values
  std::size_t indices_bytes = 0;  ///< Total bytes for all indices
  std::size_t overhead_bytes = 0; ///< Vector and Span overhead
  Size num_spans = 0;             ///< Number of Span objects
  Size num_shared = 0;            ///< Number of Spans in shared mode
  Size num_owned = 0;             ///< Number of Spans in owned mode
  Size num_view = 0;              ///< Number of Spans in view mode

  [[nodiscard]]
  auto total_bytes() const noexcept -> std::size_t {
    return values_bytes + indices_bytes + overhead_bytes;
  }
};

// =============================================================================
// SECTION 3: Buffer Allocation Strategy
// =============================================================================

/// @brief Sparse matrix buffer allocation strategy
struct SparseBufferStrategy {
  /// @brief Strategy type enumeration
  enum class Type : std::uint8_t {
    Fragmented,    ///< Each row/column uses separate buffer
    SingleBuffer,  ///< All rows/columns share one large buffer
    MinBufferSize, ///< Group rows/cols by minimum buffer size (bytes)
    BufferCount,   ///< Group rows/cols by target buffer count
    Auto           ///< Platform-optimal automatic selection
  };

  // Configuration constants
  static constexpr Size kMediumMatrixThreshold = 100;
  static constexpr Size kBufferSizeKB = 64;
  static constexpr Size kLargePrimaryDimThreshold = 10000;
  static constexpr Size kPrimaryDimDivisor = 1000;
  static constexpr Size kMaxBufferSizeBytes = Size{1024} * Size{1024} * Size{1024};
  static constexpr Size kMaxBufferCount = 10000;
  static constexpr Size kMB = Size{1024} * Size{1024};
  static constexpr Size kKB = Size{1024};

  Type type = Type::Auto; ///< Strategy type
  Size value = 0;         ///< Strategy-specific value

  // -------------------------------------------------------------------------
  // Factory Methods
  // -------------------------------------------------------------------------

  /// @brief Create fragmented strategy (each row/col separate buffer)
  [[nodiscard]]
  static
  constexpr
  auto fragmented() noexcept -> SparseBufferStrategy {
    return {Type::Fragmented, 0};
  }

  /// @brief Create single buffer strategy (all rows/cols share one buffer)
  [[nodiscard]]
  static
  constexpr
  auto single_buffer() noexcept -> SparseBufferStrategy {
    return {Type::SingleBuffer, 0};
  }

  /// @brief Create min buffer size strategy
  /// @param[in] bytes Minimum buffer size in bytes
  [[nodiscard]]
  static
  constexpr
  auto min_size(Size bytes) noexcept -> SparseBufferStrategy {
    return {Type::MinBufferSize, bytes};
  }

  /// @brief Create buffer count strategy
  /// @param[in] count Target number of buffers
  [[nodiscard]]
  static
  constexpr
  auto buffer_count(Size count) noexcept -> SparseBufferStrategy {
    return {Type::BufferCount, count};
  }

  /// @brief Create automatic strategy (platform-optimal default)
  [[nodiscard]]
  static
  constexpr
  auto auto_strategy() noexcept -> SparseBufferStrategy {
    return {Type::Auto, 0};
  }

  // -------------------------------------------------------------------------
  // Core Methods
  // -------------------------------------------------------------------------

  /// @brief Compute min_buffer_size for internal allocation logic
  /// @param[in] total_nnz Total number of non-zeros
  /// @param[in] primary_dim Number of rows (CSR) or columns (CSC)
  /// @param[in] value_size Size of value element in bytes
  /// @param[in] index_size Size of index element in bytes
  /// @return Minimum buffer size in bytes (0 = single buffer, SIZE_MAX = fragmented)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto compute_min_buffer_size(Size total_nnz, Size primary_dim,
                               Size value_size, Size index_size) const noexcept
      -> Size {
    switch (type) {
      case Type::Fragmented:
        return static_cast<Size>(-1); // SIZE_MAX

      case Type::SingleBuffer:
        return 0;

      case Type::MinBufferSize:
        return value;

      case Type::BufferCount:
        if (value == 0 || total_nnz == 0) {
          return 0;
        }
        // Divide total size by buffer count
        return (total_nnz + value - 1) / value;

      case Type::Auto:
      default: {
        // Auto strategy based on matrix size
        const Size total_bytes = total_nnz * (value_size + index_size);

        if (total_bytes < kMB) {
          // Small matrix (< 1MB): use fragmented for flexibility
          return static_cast<Size>(-1);
        }
        if (total_bytes < kMediumMatrixThreshold * kMB) {
          // Medium matrix (1-100MB): use 64KB buffers
          return kBufferSizeKB * kKB;
        }
        // Large matrix (> 100MB): adaptive based on sparsity
        const Size num_threads = threading::hardware_concurrency();
        Size target_count = num_threads > 0 ? num_threads * 2 : 8;

        if (primary_dim > kLargePrimaryDimThreshold) {
          target_count = std::max(target_count, primary_dim / kPrimaryDimDivisor);
        }

        return total_bytes / target_count;
      }
    }
  }

  /// @brief Validate strategy parameters
  /// @return true if strategy is valid, false otherwise
  [[nodiscard]]
  SCL_FORCE_INLINE
  constexpr
  auto validate() const noexcept -> bool {
    switch (type) {
      case Type::Fragmented:
      case Type::SingleBuffer:
      case Type::Auto:
        return true;

      case Type::MinBufferSize:
        // Buffer size should be reasonable (< 1GB)
        return value > 0 && value < kMaxBufferSizeBytes;

      case Type::BufferCount:
        // Count should be reasonable (1-10000)
        return value > 0 && value <= kMaxBufferCount;

      default:
        return false;
    }
  }
};

// =============================================================================
// SECTION 4: Sparse Matrix Implementation
// =============================================================================

/// @brief CSR/CSC Sparse Matrix with Span-based storage
///
/// @tparam ValueT Value type
/// @tparam IndexT Index type
/// @tparam IsCSR true for CSR format, false for CSC format
template <typename ValueT = Real, typename IndexT = Index, bool IsCSR = true>
class Sparse {
  // Compile-time type validation
  static_assert(is_supported_value_type_v<ValueT>,
                "ValueT must be one of: Real32, Real64, Int8-64, UInt8-64");
  static_assert(std::is_same_v<IndexT, Index32> || std::is_same_v<IndexT, Index64>,
                "IndexT must be Index32 or Index64");

  // Friend declaration for transpose type
  friend class Sparse<ValueT, IndexT, !IsCSR>;

 public:
  using ValueType = ValueT;
  using IndexType = IndexT;
  using SelfType = Sparse<ValueT, IndexT, IsCSR>;
  using TransposeType = Sparse<ValueT, IndexT, !IsCSR>;

  static constexpr bool is_csr = IsCSR;
  static constexpr bool is_csc = !IsCSR;

  // -------------------------------------------------------------------------
  // Constructors
  // -------------------------------------------------------------------------

  /// @brief Default: empty matrix
  constexpr
  Sparse() noexcept = default;

  /// @brief Construct from dimensions (empty rows/columns)
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  Sparse(IndexT rows, IndexT cols) noexcept : rows_(rows), cols_(cols) {
    const IndexT pdim = primary_dim();
    if (pdim > 0) {
      values_.resize(static_cast<std::size_t>(pdim));
      indices_.resize(static_cast<std::size_t>(pdim));
    }
    set_nnz(0);  // All rows/columns are empty
  }

  // Default copy/move (Span handles reference counting)
  Sparse(const Sparse&) = default;
  Sparse(Sparse&&) noexcept = default;
  auto operator=(const Sparse&) -> Sparse& = default;
  auto operator=(Sparse&&) noexcept -> Sparse& = default;
  ~Sparse() = default;

  // -------------------------------------------------------------------------
  // Basic Queries
  // -------------------------------------------------------------------------

  /// @brief Check if matrix is valid (has proper dimensions and consistent sizes)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto valid() const noexcept -> bool {
    const IndexT pdim = primary_dim();
    
    // Check vector sizes match primary dimension
    if (static_cast<IndexT>(values_.size()) != pdim ||
        static_cast<IndexT>(indices_.size()) != pdim) {
      return false;
    }
    
    // Check each value/index span pair has matching sizes
    for (IndexT i = 0; i < pdim; ++i) {
      if (values_[static_cast<std::size_t>(i)].size() !=
          indices_[static_cast<std::size_t>(i)].size()) {
        return false;
      }
    }
    
    return true;
  }

  /// @brief Check if matrix is valid (bool conversion)
  [[nodiscard]]
  SCL_FORCE_INLINE
  explicit operator bool() const noexcept {
    return valid();
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto rows() const noexcept -> IndexT {
    return rows_;
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto cols() const noexcept -> IndexT {
    return cols_;
  }

  /// @brief Get total number of non-zeros (cached, lazy recomputation)
  /// @return Total number of non-zero elements
  /// @note O(1) if cache is valid, O(n) if dirty
  [[nodiscard]]
  auto nnz() const noexcept -> IndexT {
    if (nnz_dirty_) {
      // Recompute NNZ
      IndexT total = 0;
      for (const auto& v : values_) {
        total += static_cast<IndexT>(v.size());
      }
      nnz_cached_ = total;
      nnz_dirty_ = false;
    }
    return nnz_cached_;
  }

  /// @brief Get primary dimension (rows for CSR, cols for CSC)
  [[nodiscard]]
  SCL_FORCE_INLINE
  constexpr
  auto primary_dim() const noexcept -> IndexT {
    return IsCSR ? rows_ : cols_;
  }

  /// @brief Get secondary dimension (cols for CSR, rows for CSC)
  [[nodiscard]]
  SCL_FORCE_INLINE
  constexpr
  auto secondary_dim() const noexcept -> IndexT {
    return IsCSR ? cols_ : rows_;
  }

  /// @brief Check if matrix is empty
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto empty() const noexcept -> bool {
    return rows_ == 0 || cols_ == 0;
  }

  /// @brief Calculate sparsity (fraction of zeros)
  [[nodiscard]]
  auto sparsity() const noexcept -> double {
    if (rows_ == 0 || cols_ == 0) {
      return 1.0;
    }
    const auto total = static_cast<double>(rows_) * static_cast<double>(cols_);
    return 1.0 - static_cast<double>(nnz()) / total;
  }

  /// @brief Calculate density (fraction of non-zeros)
  [[nodiscard]]
  auto density() const noexcept -> double {
    return 1.0 - sparsity();
  }

  /// @brief Get memory layout information
  [[nodiscard]]
  auto layout_info() const noexcept -> SparseLayoutInfo {
    SparseLayoutInfo info;
    info.num_spans = static_cast<Size>(values_.size());

    for (const auto& v : values_) {
      info.values_bytes += v.size_bytes();
      if (v.is_shared()) {
        ++info.num_shared;
      } else if (v.is_owned()) {
        ++info.num_owned;
      } else if (v.is_view()) {
        ++info.num_view;
      }
    }

    for (const auto& idx : indices_) {
      info.indices_bytes += idx.size_bytes();
    }

    // Overhead: vector capacity + Span objects
    info.overhead_bytes = values_.capacity() * sizeof(Span<ValueT>) +
                          indices_.capacity() * sizeof(Span<IndexT>) +
                          2 * sizeof(std::vector<Span<ValueT>>);

    return info;
  }

  // -------------------------------------------------------------------------
  // Data Access
  // -------------------------------------------------------------------------

  /// @brief Get values vector
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto values() noexcept -> std::vector<Span<ValueT>>& {
    return values_;
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto values() const noexcept -> const std::vector<Span<ValueT>>& {
    return values_;
  }

  /// @brief Get indices vector
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto indices() noexcept -> std::vector<Span<IndexT>>& {
    return indices_;
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto indices() const noexcept -> const std::vector<Span<IndexT>>& {
    return indices_;
  }

  // -------------------------------------------------------------------------
  // Row/Column Access (CSR/CSC specific)
  // -------------------------------------------------------------------------

  /// @brief Get row length (unchecked, CSR only)
  /// @param[in] i Row index
  /// @return Number of non-zeros in row
  /// @warning No bounds checking - caller must ensure i < rows()
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_length_unchecked(IndexT i) const noexcept -> IndexT
      requires(IsCSR) {
    return static_cast<IndexT>(values_[static_cast<std::size_t>(i)].size());
  }

  /// @brief Get row length (CSR only)
  /// @param[in] i Row index
  /// @return Number of non-zeros in row
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_length(IndexT i) const noexcept -> IndexT
      requires(IsCSR) {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < rows_, "row index out of bounds");
#endif
    return row_length_unchecked(i);
  }

  /// @brief Get row values as Span (unchecked, CSR only)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_values_unchecked(IndexT i) const noexcept -> const Span<ValueT>&
      requires(IsCSR) {
    return values_[static_cast<std::size_t>(i)];
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_values_unchecked(IndexT i) noexcept -> Span<ValueT>&
      requires(IsCSR) {
    return values_[static_cast<std::size_t>(i)];
  }

  /// @brief Get row values as Span (CSR only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_values(IndexT i) const noexcept -> const Span<ValueT>&
      requires(IsCSR) {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < rows_, "row index out of bounds");
#endif
    return row_values_unchecked(i);
  }

  /// @brief Get mutable row values (CSR only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_values(IndexT i) noexcept -> Span<ValueT>&
      requires(IsCSR) {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < rows_, "row index out of bounds");
#endif
    return row_values_unchecked(i);
  }

  /// @brief Get row indices as Span (unchecked, CSR only)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_indices_unchecked(IndexT i) const noexcept -> const Span<IndexT>&
      requires(IsCSR) {
    return indices_[static_cast<std::size_t>(i)];
  }

  /// @brief Get row indices as Span (CSR only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto row_indices(IndexT i) const noexcept -> const Span<IndexT>&
      requires(IsCSR) {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < rows_, "row index out of bounds");
#endif
    return row_indices_unchecked(i);
  }

  /// @brief Get column length (unchecked, CSC only)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_length_unchecked(IndexT j) const noexcept -> IndexT
      requires(!IsCSR) {
    return static_cast<IndexT>(values_[static_cast<std::size_t>(j)].size());
  }

  /// @brief Get column length (CSC only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_length(IndexT j) const noexcept -> IndexT
      requires(!IsCSR) {
#ifndef NDEBUG
    error::check_arg(j >= 0 && j < cols_, "column index out of bounds");
#endif
    return col_length_unchecked(j);
  }

  /// @brief Get column values as Span (unchecked, CSC only)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_values_unchecked(IndexT j) const noexcept -> const Span<ValueT>&
      requires(!IsCSR) {
    return values_[static_cast<std::size_t>(j)];
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_values_unchecked(IndexT j) noexcept -> Span<ValueT>&
      requires(!IsCSR) {
    return values_[static_cast<std::size_t>(j)];
  }

  /// @brief Get column values as Span (CSC only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_values(IndexT j) const noexcept -> const Span<ValueT>&
      requires(!IsCSR) {
#ifndef NDEBUG
    error::check_arg(j >= 0 && j < cols_, "column index out of bounds");
#endif
    return col_values_unchecked(j);
  }

  /// @brief Get mutable column values (CSC only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_values(IndexT j) noexcept -> Span<ValueT>&
      requires(!IsCSR) {
#ifndef NDEBUG
    error::check_arg(j >= 0 && j < cols_, "column index out of bounds");
#endif
    return col_values_unchecked(j);
  }

  /// @brief Get column indices as Span (unchecked, CSC only)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_indices_unchecked(IndexT j) const noexcept -> const Span<IndexT>&
      requires(!IsCSR) {
    return indices_[static_cast<std::size_t>(j)];
  }

  /// @brief Get column indices as Span (CSC only)
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto col_indices(IndexT j) const noexcept -> const Span<IndexT>&
      requires(!IsCSR) {
#ifndef NDEBUG
    error::check_arg(j >= 0 && j < cols_, "column index out of bounds");
#endif
    return col_indices_unchecked(j);
  }

  /// @brief Get primary dimension length (unchecked)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_length_unchecked(IndexT i) const noexcept -> IndexT {
    return static_cast<IndexT>(values_[static_cast<std::size_t>(i)].size());
  }

  /// @brief Get primary dimension length
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_length(IndexT i) const noexcept -> IndexT {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < primary_dim(), "primary index out of bounds");
#endif
    return primary_length_unchecked(i);
  }

  /// @brief Get primary dimension values (unchecked)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_values_unchecked(IndexT i) const noexcept -> const Span<ValueT>& {
    return values_[static_cast<std::size_t>(i)];
  }

  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_values_unchecked(IndexT i) noexcept -> Span<ValueT>& {
    return values_[static_cast<std::size_t>(i)];
  }

  /// @brief Get primary dimension values
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_values(IndexT i) const noexcept -> const Span<ValueT>& {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < primary_dim(), "primary index out of bounds");
#endif
    return primary_values_unchecked(i);
  }

  /// @brief Get mutable primary dimension values
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_values(IndexT i) noexcept -> Span<ValueT>& {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < primary_dim(), "primary index out of bounds");
#endif
    return primary_values_unchecked(i);
  }

  /// @brief Get primary dimension indices (unchecked)
  /// @warning No bounds checking
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_indices_unchecked(IndexT i) const noexcept -> const Span<IndexT>& {
    return indices_[static_cast<std::size_t>(i)];
  }

  /// @brief Get primary dimension indices
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto primary_indices(IndexT i) const noexcept -> const Span<IndexT>& {
#ifndef NDEBUG
    error::check_arg(i >= 0 && i < primary_dim(), "primary index out of bounds");
#endif
    return primary_indices_unchecked(i);
  }

  // -------------------------------------------------------------------------
  // Element Access
  // -------------------------------------------------------------------------

  /// @brief Get value at (row, col), returns 0 if not found
  /// @note O(log n) - uses binary search
  [[nodiscard]]
  auto at(IndexT row, IndexT col) const noexcept -> ValueT {
    if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
      return ValueT{0};
    }

    const IndexT primary_idx = IsCSR ? row : col;
    const IndexT secondary_idx = IsCSR ? col : row;

    const auto& idx_span = indices_[static_cast<std::size_t>(primary_idx)];
    if (idx_span.empty()) {
      return ValueT{0};
    }

    // Binary search
    auto it = std::lower_bound(idx_span.begin(), idx_span.end(), secondary_idx);
    if (it != idx_span.end() && *it == secondary_idx) {
      const auto pos = static_cast<Size>(it - idx_span.begin());
      return values_[static_cast<std::size_t>(primary_idx)][pos];
    }
    return ValueT{0};
  }

  /// @brief Check if element exists at (row, col)
  [[nodiscard]]
  auto exists(IndexT row, IndexT col) const noexcept -> bool {
    if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
      return false;
    }

    const IndexT primary_idx = IsCSR ? row : col;
    const IndexT secondary_idx = IsCSR ? col : row;

    const auto& idx_span = indices_[static_cast<std::size_t>(primary_idx)];
    if (idx_span.empty()) {
      return false;
    }

    auto it = std::lower_bound(idx_span.begin(), idx_span.end(), secondary_idx);
    return it != idx_span.end() && *it == secondary_idx;
  }

  // -------------------------------------------------------------------------
  // Factory: Create Empty/Zero Matrix
  // -------------------------------------------------------------------------

  /// @brief Create zero matrix (empty rows/columns)
  [[nodiscard]]
  static
  auto zeros(IndexT rows, IndexT cols) -> Sparse {
    if (rows < 0 || cols < 0) {
      return {};
    }
    return Sparse(rows, cols);
  }

  // -------------------------------------------------------------------------
  // Factory: Create from NNZ Counts
  // -------------------------------------------------------------------------

  /// @brief Create matrix with specified NNZ per row/column
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] primary_nnzs NNZ count for each primary dimension
  /// @param[in] strategy Buffer allocation strategy
  /// @return Sparse matrix with allocated but uninitialized data
  [[nodiscard]]
  static
  auto create(IndexT rows, IndexT cols, std::span<const IndexT> primary_nnzs,
              SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy())
      -> Sparse {
    error::check_arg(rows >= 0 && cols >= 0, "dimensions must be non-negative");

    const IndexT pdim = IsCSR ? rows : cols;
    error::check_arg(static_cast<IndexT>(primary_nnzs.size()) == pdim,
                     "nnz array size mismatch");

    if (rows == 0 || cols == 0) {
      return {};
    }

    // Calculate total NNZ for strategy
    IndexT total_nnz = 0;
    for (const auto nnz_val : primary_nnzs) {
      error::check_arg(nnz_val >= 0, "nnz counts must be non-negative");
      total_nnz += nnz_val;
    }

    Sparse result(rows, cols);

    // Compute min_buffer_size from strategy
    const Size min_buffer_size = strategy.compute_min_buffer_size(
        static_cast<Size>(total_nnz), static_cast<Size>(pdim),
        sizeof(ValueT), sizeof(IndexT));

    // Allocate with strategy
    if (!allocate_with_strategy(result.values_, result.indices_, primary_nnzs,
                                min_buffer_size)) {
      return {};
    }

    return result;
  }

  // -------------------------------------------------------------------------
  // Factory: Create from Traditional Format
  // -------------------------------------------------------------------------

  /// @brief Create from traditional CSR/CSC arrays (copies data)
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] values Non-zero values
  /// @param[in] indices Column (CSR) or row (CSC) indices
  /// @param[in] indptr Row (CSR) or column (CSC) pointers
  /// @param[in] strategy Buffer allocation strategy
  [[nodiscard]]
  static
  auto from_arrays(IndexT rows, IndexT cols, std::span<const ValueT> values,
                   std::span<const IndexT> indices,
                   std::span<const IndexT> indptr,
                   SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy())
      -> Sparse {
    const IndexT pdim = IsCSR ? rows : cols;
    error::check_arg(static_cast<IndexT>(indptr.size()) == pdim + 1,
                     "indptr size must be primary_dim + 1");
    error::check_arg(pdim == 0 || indptr[0] == 0, "indptr[0] must be 0");

    const IndexT nnz_count =
        pdim > 0 ? indptr[static_cast<std::size_t>(pdim)] : 0;
    error::check_arg(static_cast<IndexT>(values.size()) >= nnz_count,
                     "values size mismatch");
    error::check_arg(static_cast<IndexT>(indices.size()) >= nnz_count,
                     "indices size mismatch");

    if (rows == 0 || cols == 0 || nnz_count == 0) {
      return zeros(rows, cols);
    }

    // Extract row/column lengths
    std::vector<IndexT> primary_nnzs(static_cast<std::size_t>(pdim));
    for (IndexT i = 0; i < pdim; ++i) {
      primary_nnzs[static_cast<std::size_t>(i)] =
          indptr[static_cast<std::size_t>(i + 1)] -
          indptr[static_cast<std::size_t>(i)];
    }

    auto result = create(rows, cols, primary_nnzs, strategy);
    if (!result) {
      return {};
    }
    
    // Set NNZ directly (we know the total count)
    result.set_nnz(nnz_count);

    // Parallel copy for large matrices
    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      threading::parallel_for(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          [&result, &values, &indices, &indptr,
           &primary_nnzs](threading::Index i) {
            const IndexT start = indptr[static_cast<std::size_t>(i)];
            const IndexT len = primary_nnzs[static_cast<std::size_t>(i)];

            if (len > 0) {
              auto& val_span = result.values_[static_cast<std::size_t>(i)];
              auto& idx_span = result.indices_[static_cast<std::size_t>(i)];

              // Use optimized memory copy
              memory::copy(values.subspan(static_cast<std::size_t>(start),
                                          static_cast<std::size_t>(len)),
                           val_span.to_std_span());
              memory::copy(indices.subspan(static_cast<std::size_t>(start),
                                           static_cast<std::size_t>(len)),
                           idx_span.to_std_span());
            }
          },
          threading::DEFAULT_GRAIN_SIZE);
    } else {
      // Serial copy for small matrices
      for (IndexT i = 0; i < pdim; ++i) {
        const IndexT start = indptr[static_cast<std::size_t>(i)];
        const IndexT len = primary_nnzs[static_cast<std::size_t>(i)];

        if (len > 0) {
          auto& val_span = result.values_[static_cast<std::size_t>(i)];
          auto& idx_span = result.indices_[static_cast<std::size_t>(i)];

          memory::copy(values.subspan(static_cast<std::size_t>(start),
                                      static_cast<std::size_t>(len)),
                       val_span.to_std_span());
          memory::copy(indices.subspan(static_cast<std::size_t>(start),
                                       static_cast<std::size_t>(len)),
                       idx_span.to_std_span());
        }
      }
    }

    return result;
  }

  /// @brief Wrap existing arrays (zero-copy view)
  /// @warning Caller must ensure arrays outlive the Sparse object
  [[deprecated("Use view_custom_sparse instead")]]
  [[nodiscard]]
  static
  auto view_arrays(IndexT rows, IndexT cols, ValueT* values_ptr,
                   IndexT* indices_ptr, IndexT* indptr_ptr) -> Sparse {
    return view_custom_sparse(rows, cols, values_ptr, indices_ptr, indptr_ptr);
  }
  
  /// @brief Create view of traditional CSR/CSC arrays (zero-copy)
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] values Non-zero values array
  /// @param[in] indices Column (CSR) or row (CSC) indices
  /// @param[in] indptr Row (CSR) or column (CSC) pointers
  /// @return Sparse matrix with View spans
  /// @warning Caller must ensure arrays outlive the Sparse object
  [[nodiscard]]
  static
  auto view_custom_sparse(IndexT rows, IndexT cols,
                          const ValueT* values,
                          const IndexT* indices,
                          const IndexT* indptr) -> Sparse {
    error::check_arg(rows >= 0 && cols >= 0, "dimensions must be non-negative");
    error::check_arg(values != nullptr && indices != nullptr && indptr != nullptr,
                     "pointers must not be null");
    
    const IndexT pdim = IsCSR ? rows : cols;
    Sparse result(rows, cols);

    // Create views for each row/column
    for (IndexT i = 0; i < pdim; ++i) {
      const IndexT start = indptr[i];
      const IndexT len = indptr[i + 1] - start;

      if (len > 0) {
        result.values_[static_cast<std::size_t>(i)] =
            Span<ValueT>::view(values + start, static_cast<Size>(len));
        result.indices_[static_cast<std::size_t>(i)] =
            Span<IndexT>::view(indices + start, static_cast<Size>(len));
      }
    }

    return result;
  }
  
  /// @brief Adopt ownership of traditional CSR/CSC arrays
  /// @tparam Dealloc Custom deallocator type
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] values Non-zero values array (will be owned)
  /// @param[in] indices Column/row indices (will be owned)
  /// @param[in] indptr Row/column pointers (will be freed immediately)
  /// @param[in] dealloc Custom deallocator for values and indices
  /// @return Sparse matrix with Shared spans
  /// @note indptr is used to create spans then deallocated
  template<typename Dealloc>
  [[nodiscard]]
  static
  auto from_custom_sparse(IndexT rows, IndexT cols,
                          ValueT* values,
                          IndexT* indices,
                          IndexT* indptr,
                          Dealloc dealloc) -> Sparse {
    error::check_arg(rows >= 0 && cols >= 0, "dimensions must be non-negative");
    error::check_arg(values != nullptr && indices != nullptr && indptr != nullptr,
                     "pointers must not be null");
    
    const IndexT pdim = IsCSR ? rows : cols;
    const IndexT total_nnz = indptr[pdim];
    
    // Create wrapper for custom deallocator
    auto deallocator = [dealloc](void* ptr, std::size_t /*size*/, void* /*context*/) noexcept {
      dealloc(ptr);
    };
    
    // Create Storage for values and indices with custom deallocator
    auto* val_storage = Storage::from_external(
        values,
        static_cast<std::size_t>(total_nnz) * sizeof(ValueT),
        deallocator,
        nullptr
    );
    
    auto* idx_storage = Storage::from_external(
        indices,
        static_cast<std::size_t>(total_nnz) * sizeof(IndexT),
        deallocator,
        nullptr
    );
    
    if (val_storage == nullptr || idx_storage == nullptr) {
      // Cleanup on failure
      if (val_storage != nullptr) {
        val_storage->decref();
      }
      if (idx_storage != nullptr) {
        idx_storage->decref();
      }
      dealloc(values);
      dealloc(indices);
      dealloc(indptr);
      return {};
    }
    
    Sparse result(rows, cols);
    
    // Create Shared spans for each row/column
    bool first_non_empty_val = true;
    bool first_non_empty_idx = true;
    
    for (IndexT i = 0; i < pdim; ++i) {
      const IndexT start = indptr[i];
      const IndexT len = indptr[i + 1] - start;
      
      if (len > 0) {
        // Values
        if (first_non_empty_val) {
          // First non-empty span adopts storage
          result.values_[static_cast<std::size_t>(i)] = Span<ValueT>(
              val_storage,
              static_cast<std::ptrdiff_t>(start * sizeof(ValueT)),
              static_cast<Size>(len),
              adopt);
          first_non_empty_val = false;
        } else {
          // Subsequent spans share (incref)
          result.values_[static_cast<std::size_t>(i)] = Span<ValueT>(
              val_storage,
              static_cast<std::ptrdiff_t>(start * sizeof(ValueT)),
              static_cast<Size>(len));
        }
        
        // Indices
        if (first_non_empty_idx) {
          // First non-empty span adopts storage
          result.indices_[static_cast<std::size_t>(i)] = Span<IndexT>(
              idx_storage,
              static_cast<std::ptrdiff_t>(start * sizeof(IndexT)),
              static_cast<Size>(len),
              adopt);
          first_non_empty_idx = false;
        } else {
          // Subsequent spans share (incref)
          result.indices_[static_cast<std::size_t>(i)] = Span<IndexT>(
              idx_storage,
              static_cast<std::ptrdiff_t>(start * sizeof(IndexT)),
              static_cast<Size>(len));
        }
      }
    }
    
    // If all rows/columns are empty, manually decref storages
    if (first_non_empty_val) {
      val_storage->decref();
    }
    if (first_non_empty_idx) {
      idx_storage->decref();
    }
    
    // Free indptr (no longer needed)
    dealloc(indptr);
    
    return result;
  }

  // -------------------------------------------------------------------------
  // Factory: Create from COO Format
  // -------------------------------------------------------------------------

  /// @brief Create from COO (Coordinate) format
  /// @note Indices are sorted; duplicates use last value
  [[nodiscard]]
  static
  auto from_coo(IndexT rows, IndexT cols,
                std::span<const IndexT> row_indices,
                std::span<const IndexT> col_indices,
                std::span<const ValueT> values,
                SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy())
      -> Sparse {
    const auto nnz_count = static_cast<IndexT>(values.size());
    error::check_arg(static_cast<IndexT>(row_indices.size()) == nnz_count,
                     "row_indices size mismatch");
    error::check_arg(static_cast<IndexT>(col_indices.size()) == nnz_count,
                     "col_indices size mismatch");

    if (nnz_count == 0) {
      return zeros(rows, cols);
    }

    const IndexT pdim = IsCSR ? rows : cols;
    const IndexT sdim = IsCSR ? cols : rows;

    // Count NNZ per primary dimension
    std::vector<IndexT> nnz_per_primary(static_cast<std::size_t>(pdim), 0);
    for (IndexT i = 0; i < nnz_count; ++i) {
      const IndexT pidx = IsCSR ? row_indices[static_cast<std::size_t>(i)]
                                : col_indices[static_cast<std::size_t>(i)];
      const IndexT sidx = IsCSR ? col_indices[static_cast<std::size_t>(i)]
                                : row_indices[static_cast<std::size_t>(i)];
      error::check_arg(pidx >= 0 && pidx < pdim, "primary index out of bounds");
      error::check_arg(sidx >= 0 && sidx < sdim,
                       "secondary index out of bounds");
      ++nnz_per_primary[static_cast<std::size_t>(pidx)];
    }

    // Create matrix with strategy
    auto result = create(rows, cols, nnz_per_primary, strategy);
    if (!result) {
      return {};
    }

    // Track insertion positions
    std::vector<IndexT> insert_pos(static_cast<std::size_t>(pdim), 0);

    // Fill data (unsorted) with prefetching
    constexpr IndexT PREFETCH_DISTANCE = 16;
    for (IndexT i = 0; i < nnz_count; ++i) {
      // Prefetch future indices
      if (i + PREFETCH_DISTANCE < nnz_count) [[likely]] {
        const IndexT future_pidx =
            IsCSR
                ? row_indices[static_cast<std::size_t>(i + PREFETCH_DISTANCE)]
                : col_indices[static_cast<std::size_t>(i + PREFETCH_DISTANCE)];
        SCL_PREFETCH_WRITE(
            result.values_[static_cast<std::size_t>(future_pidx)].data(), 1);
        SCL_PREFETCH_WRITE(
            result.indices_[static_cast<std::size_t>(future_pidx)].data(), 1);
      }

      const IndexT pidx = IsCSR ? row_indices[static_cast<std::size_t>(i)]
                                : col_indices[static_cast<std::size_t>(i)];
      const IndexT sidx = IsCSR ? col_indices[static_cast<std::size_t>(i)]
                                : row_indices[static_cast<std::size_t>(i)];
      const IndexT pos = insert_pos[static_cast<std::size_t>(pidx)]++;

      result.values_[static_cast<std::size_t>(pidx)][static_cast<Size>(pos)] =
          values[static_cast<std::size_t>(i)];
      result.indices_[static_cast<std::size_t>(pidx)][static_cast<Size>(pos)] =
          sidx;
    }

    // Sort indices within each primary dimension
    result.sort_indices();
    
    // Set NNZ directly (we know the count)
    result.set_nnz(nnz_count);

    return result;
  }

  // -------------------------------------------------------------------------
  // Factory: Create Identity Matrix
  // -------------------------------------------------------------------------

  /// @brief Create identity matrix
  [[nodiscard]]
  static
  auto identity(IndexT n) -> Sparse {
    if (n <= 0) {
      return {};
    }

    std::vector<IndexT> nnzs(static_cast<std::size_t>(n), 1);
    auto result = create(n, n, nnzs);
    if (!result) {
      return {};
    }

    for (IndexT i = 0; i < n; ++i) {
      result.values_[static_cast<std::size_t>(i)][0] = ValueT{1};
      result.indices_[static_cast<std::size_t>(i)][0] = i;
    }

    return result;
  }

  // -------------------------------------------------------------------------
  // Factory: Create from Spans (Zero-Copy)
  // -------------------------------------------------------------------------

  /// @brief Create from existing Span vectors (zero-copy move)
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] values Values spans (moved)
  /// @param[in] indices Indices spans (moved)
  /// @return Sparse matrix wrapping the spans
  /// @note Efficient when you already have allocated Span vectors
  [[nodiscard]]
  static
  auto from_spans(IndexT rows, IndexT cols,
                  std::vector<Span<ValueT>>&& values,
                  std::vector<Span<IndexT>>&& indices) -> Sparse {
    const IndexT pdim = IsCSR ? rows : cols;
    
    error::check_arg(rows >= 0 && cols >= 0, "dimensions must be non-negative");
    error::check_arg(static_cast<IndexT>(values.size()) == pdim,
                     "values size must equal primary dimension");
    error::check_arg(static_cast<IndexT>(indices.size()) == pdim,
                     "indices size must equal primary dimension");
    
    // Validate all spans have matching sizes
    for (IndexT i = 0; i < pdim; ++i) {
      error::check_arg(values[static_cast<std::size_t>(i)].size() ==
                       indices[static_cast<std::size_t>(i)].size(),
                       "value and index span sizes must match");
    }

    Sparse result;
    result.rows_ = rows;
    result.cols_ = cols;
    result.values_ = std::move(values);
    result.indices_ = std::move(indices);
    result.mark_nnz_dirty();  // NNZ unknown from spans
    
    return result;
  }

  // -------------------------------------------------------------------------
  // Factory: Create from Dense Matrix
  // -------------------------------------------------------------------------

  /// @brief Create from dense matrix (row-major)
  /// @param[in] data Dense matrix data (row-major layout)
  /// @param[in] rows Number of rows
  /// @param[in] cols Number of columns
  /// @param[in] threshold Sparsity threshold (values <= threshold treated as zero)
  /// @param[in] strategy Buffer allocation strategy
  /// @return Sparse matrix
  [[nodiscard]]
  static
  auto from_dense(std::span<const ValueT> data, IndexT rows, IndexT cols,
                  ValueT threshold = ValueT{0},
                  SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy())
      -> Sparse {
    error::check_arg(rows >= 0 && cols >= 0, "dimensions must be non-negative");
    error::check_arg(static_cast<IndexT>(data.size()) >= rows * cols,
                     "data size must be at least rows * cols");

    if (rows == 0 || cols == 0) {
      return zeros(rows, cols);
    }

    const IndexT pdim = IsCSR ? rows : cols;
    const IndexT sdim = IsCSR ? cols : rows;

    // Count NNZ per primary dimension
    auto nnz_counts = count_nonzeros_per_dim(data, rows, cols, pdim, sdim, threshold);

    // Create matrix with strategy
    auto result = create(rows, cols, nnz_counts, strategy);
    if (!result) {
      return {};
    }

    // Fill data
    fill_from_dense_data(result, data, cols, pdim, sdim, threshold);
    
    // Mark dirty (could compute during fill, but simpler to recompute on demand)
    result.mark_nnz_dirty();

    return result;
  }

  // -------------------------------------------------------------------------
  // Clone and Conversion
  // -------------------------------------------------------------------------

  /// @brief Deep copy with buffer allocation strategy
  [[nodiscard]]
  auto clone(SparseBufferStrategy strategy = SparseBufferStrategy::auto_strategy())
      const -> Sparse {
    if (!valid()) {
      return {};
    }

    const IndexT pdim = primary_dim();

    // Collect NNZ counts
    std::vector<IndexT> nnzs(static_cast<std::size_t>(pdim));
    for (IndexT i = 0; i < pdim; ++i) {
      nnzs[static_cast<std::size_t>(i)] =
          static_cast<IndexT>(values_[static_cast<std::size_t>(i)].size());
    }

    // Create new matrix with strategy
    auto result = create(rows_, cols_, nnzs, strategy);
    if (!result) {
      return {};
    }

    // Copy NNZ (avoid recomputation)
    result.nnz_cached_ = this->nnz_cached_;
    result.nnz_dirty_ = this->nnz_dirty_;
    
    // Copy data (parallel for large matrices)
    constexpr Size kStreamCopyThreshold = 1024;  // Use stream_copy for large rows
    
    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      threading::parallel_for(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          [this, &result](threading::Index i) {
            const auto& src_vals = values_[static_cast<std::size_t>(i)];
            const auto& src_idxs = indices_[static_cast<std::size_t>(i)];
            auto& dst_vals = result.values_[static_cast<std::size_t>(i)];
            auto& dst_idxs = result.indices_[static_cast<std::size_t>(i)];

            if (src_vals.size() > 0) {
              // Use stream_copy for large rows to bypass cache
              if (src_vals.size() >= kStreamCopyThreshold) {
                memory::stream_copy(src_vals.to_std_span(), dst_vals.to_std_span());
                memory::stream_copy(src_idxs.to_std_span(), dst_idxs.to_std_span());
              } else {
                memory::copy(src_vals.to_std_span(), dst_vals.to_std_span());
                memory::copy(src_idxs.to_std_span(), dst_idxs.to_std_span());
              }
            }
          },
          threading::DEFAULT_GRAIN_SIZE);
    } else {
      // Serial copying for small matrices
      for (IndexT i = 0; i < pdim; ++i) {
        const auto& src_vals = values_[static_cast<std::size_t>(i)];
        const auto& src_idxs = indices_[static_cast<std::size_t>(i)];
        auto& dst_vals = result.values_[static_cast<std::size_t>(i)];
        auto& dst_idxs = result.indices_[static_cast<std::size_t>(i)];

        if (src_vals.size() > 0) {
          memory::copy(src_vals.to_std_span(), dst_vals.to_std_span());
          memory::copy(src_idxs.to_std_span(), dst_idxs.to_std_span());
        }
      }
    }

    return result;
  }

  /// @brief Convert to transposed format (CSR <-> CSC)
  /// @note Cache-optimized with prefetching
  [[nodiscard]]
  auto transpose() const -> TransposeType {
    if (!valid()) {
      return {};
    }

    const IndexT new_rows = cols_;
    const IndexT new_cols = rows_;
    const IndexT new_pdim = !IsCSR ? new_rows : new_cols;
    const IndexT old_pdim = primary_dim();

    // Count NNZ per new primary dimension (with prefetching)
    std::vector<IndexT> new_nnzs(static_cast<std::size_t>(new_pdim), 0);

    constexpr IndexT kPrefetchDistance = 8;
    for (IndexT i = 0; i < old_pdim; ++i) {
      // Prefetch future indices for counting
      if (i + kPrefetchDistance < old_pdim) [[likely]] {
        const auto& future_idx = indices_[static_cast<std::size_t>(
            i + kPrefetchDistance)];
        SCL_PREFETCH_READ(future_idx.data(), 1);
      }
      
      const auto& idx_span = indices_[static_cast<std::size_t>(i)];
      for (const auto idx : idx_span) {
        ++new_nnzs[static_cast<std::size_t>(idx)];
      }
    }

    auto result = TransposeType::create(new_rows, new_cols, new_nnzs);
    if (!result) {
      return {};
    }

    // Fill transposed data with cache-aware access
    std::vector<IndexT> insert_pos(static_cast<std::size_t>(new_pdim), 0);

    for (IndexT i = 0; i < old_pdim; ++i) {
      const auto& val_span = values_[static_cast<std::size_t>(i)];
      const auto& idx_span = indices_[static_cast<std::size_t>(i)];

      // Prefetch next row/column data
      if (i + 1 < old_pdim) [[likely]] {
        const auto& next_vals = values_[static_cast<std::size_t>(i + 1)];
        const auto& next_idxs = indices_[static_cast<std::size_t>(i + 1)];
        if (!next_vals.empty()) {
          SCL_PREFETCH_READ(next_vals.data(), 2);
        }
        if (!next_idxs.empty()) {
          SCL_PREFETCH_READ(next_idxs.data(), 2);
        }
      }

      // Process with prefetching for scattered writes
      const Size len = val_span.size();
      for (Size k = 0; k < len; ++k) {
        const IndexT j = idx_span[k];
        
        // Prefetch future destination for scattered write
        if (k + kPrefetchDistance < len) [[likely]] {
          const IndexT future_j = idx_span[k + kPrefetchDistance];
          SCL_PREFETCH_WRITE(result.values_[static_cast<std::size_t>(future_j)].data(), 1);
          SCL_PREFETCH_WRITE(result.indices_[static_cast<std::size_t>(future_j)].data(), 1);
        }
        
        const IndexT pos = insert_pos[static_cast<std::size_t>(j)]++;
        result.values_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] =
            val_span[k];
        result.indices_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] =
            i;
      }
    }

    result.sort_indices();
    
    // Mark NNZ dirty after transpose
    result.mark_nnz_dirty();
    
    return result;
  }

  // -------------------------------------------------------------------------
  // Row/Column Slicing (Zero-Copy)
  // -------------------------------------------------------------------------

  /// @brief Row range slice (zero-copy, CSR only)
  [[nodiscard]]
  auto row_slice(IndexT start, IndexT end) const -> Sparse
      requires(IsCSR) {
    error::check_arg(start >= 0 && end <= rows_ && start <= end,
                     "invalid row range");

    if (start == end) {
      return zeros(0, cols_);
    }

    const IndexT new_rows = end - start;
    Sparse result(new_rows, cols_);

    // Copy Spans (reference counted, zero-copy)
    for (IndexT i = 0; i < new_rows; ++i) {
      result.values_[static_cast<std::size_t>(i)] =
          values_[static_cast<std::size_t>(start + i)];
      result.indices_[static_cast<std::size_t>(i)] =
          indices_[static_cast<std::size_t>(start + i)];
    }
    
    result.mark_nnz_dirty();  // Slicing changes NNZ

    return result;
  }

  /// @brief Row selection slice (zero-copy, CSR only)
  [[nodiscard]]
  auto row_select(std::span<const IndexT> row_idx) const -> Sparse
      requires(IsCSR) {
    if (row_idx.empty()) {
      return zeros(0, cols_);
    }

    const auto new_rows = static_cast<IndexT>(row_idx.size());
    Sparse result(new_rows, cols_);

    for (IndexT i = 0; i < new_rows; ++i) {
      const IndexT src = row_idx[static_cast<std::size_t>(i)];
      error::check_arg(src >= 0 && src < rows_, "row index out of bounds");

      result.values_[static_cast<std::size_t>(i)] =
          values_[static_cast<std::size_t>(src)];
      result.indices_[static_cast<std::size_t>(i)] =
          indices_[static_cast<std::size_t>(src)];
    }
    
    result.mark_nnz_dirty();  // Selection changes NNZ

    return result;
  }

  /// @brief Column range slice (zero-copy, CSC only)
  [[nodiscard]]
  auto col_slice(IndexT start, IndexT end) const -> Sparse
      requires(!IsCSR) {
    error::check_arg(start >= 0 && end <= cols_ && start <= end,
                     "invalid column range");

    if (start == end) {
      return zeros(rows_, 0);
    }

    const IndexT new_cols = end - start;
    Sparse result(rows_, new_cols);

    for (IndexT j = 0; j < new_cols; ++j) {
      result.values_[static_cast<std::size_t>(j)] =
          values_[static_cast<std::size_t>(start + j)];
      result.indices_[static_cast<std::size_t>(j)] =
          indices_[static_cast<std::size_t>(start + j)];
    }
    
    result.mark_nnz_dirty();  // Slicing changes NNZ

    return result;
  }

  /// @brief Column selection slice (zero-copy, CSC only)
  [[nodiscard]]
  auto col_select(std::span<const IndexT> col_idx) const -> Sparse
      requires(!IsCSR) {
    if (col_idx.empty()) {
      return zeros(rows_, 0);
    }

    const auto new_cols = static_cast<IndexT>(col_idx.size());
    Sparse result(rows_, new_cols);

    for (IndexT j = 0; j < new_cols; ++j) {
      const IndexT src = col_idx[static_cast<std::size_t>(j)];
      error::check_arg(src >= 0 && src < cols_, "column index out of bounds");

      result.values_[static_cast<std::size_t>(j)] =
          values_[static_cast<std::size_t>(src)];
      result.indices_[static_cast<std::size_t>(j)] =
          indices_[static_cast<std::size_t>(src)];
    }
    
    result.mark_nnz_dirty();  // Selection changes NNZ

    return result;
  }

  // -------------------------------------------------------------------------
  // In-place Operations
  // -------------------------------------------------------------------------

  /// @brief Sort indices within each row/column
  auto sort_indices() -> void {
    if (!valid()) {
      return;
    }

    const IndexT pdim = primary_dim();
    
    // Small buffer optimization threshold
    constexpr Size kSmallBufferThreshold = 256;

    // For large matrices, use thread work areas
    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      // Estimate max row/col length for buffer sizing
      Size max_len = 0;
      for (IndexT i = 0; i < pdim; ++i) {
        max_len = std::max(max_len, values_[static_cast<std::size_t>(i)].size());
      }
      
      // Create thread work areas with pre-allocated buffers
      auto perm_work = threading::make_work_buffers<Size>(max_len);
      auto idx_work = threading::make_work_buffers<IndexT>(max_len);
      auto val_work = threading::make_work_buffers<ValueT>(max_len);
      
      // Pre-allocate all thread buffers
      perm_work.preallocate();
      idx_work.preallocate();
      val_work.preallocate();
      
      threading::parallel_for(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          [this, &perm_work, &idx_work, &val_work](threading::Index i) {
            auto& vals = values_[static_cast<std::size_t>(i)];
            auto& idxs = indices_[static_cast<std::size_t>(i)];

            const Size len = vals.size();
            if (len <= 1) {
              return;
            }

            // Get thread work buffers (already allocated)
            auto& perm = perm_work.get();
            auto& sorted_idx = idx_work.get();
            auto& sorted_val = val_work.get();
            
            // Resize to needed size (cheap, keeps capacity)
            perm.resize(len);
            sorted_idx.resize(len);
            sorted_val.resize(len);
            
            // Create permutation
            std::iota(perm.begin(), perm.end(), Size{0});
            std::sort(perm.begin(), perm.end(), [&idxs](Size a, Size b) {
              return idxs[a] < idxs[b];
            });
            
            // Apply permutation
            for (Size k = 0; k < len; ++k) {
              sorted_idx[k] = idxs[perm[k]];
              sorted_val[k] = vals[perm[k]];
            }
            
            // Copy back
            for (Size k = 0; k < len; ++k) {
              idxs[k] = sorted_idx[k];
              vals[k] = sorted_val[k];
            }
          },
          threading::DEFAULT_GRAIN_SIZE);
    } else {
      // Serial sorting - use helper for both cases
      for (IndexT i = 0; i < pdim; ++i) {
        sort_row_indices_serial(i, kSmallBufferThreshold);
      }
    }
  }

  /// @brief Verify indices are sorted
  [[nodiscard]]
  auto is_sorted() const noexcept -> bool {
    if (!valid()) {
      return true;
    }

    const IndexT pdim = primary_dim();

    for (IndexT i = 0; i < pdim; ++i) {
      const auto& idx_span = indices_[static_cast<std::size_t>(i)];
      for (Size k = 1; k < idx_span.size(); ++k) {
        if (idx_span[k] <= idx_span[k - 1]) {
          return false;
        }
      }
    }
    return true;
  }

  /// @brief Scale all values (SIMD-optimized)
  /// @param[in] factor Scale factor
  auto scale(ValueT factor) -> void {
    if (!valid()) {
      return;
    }

    const IndexT pdim = primary_dim();

    // Parallel scaling for large matrices
    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      threading::parallel_for(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          [this, factor](threading::Index i) {
            scale_row_values(values_[static_cast<std::size_t>(i)], factor);
          },
          threading::DEFAULT_GRAIN_SIZE);
    } else {
      for (auto& v : values_) {
        scale_row_values(v, factor);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Sparse Matrix-Vector Multiplication (SpMV)
  // -------------------------------------------------------------------------

  /// @brief Sparse matrix-vector multiplication: y = A * x (CSR)
  /// @param[in] x Input vector
  /// @return Result vector y
  /// @note Parallel for large matrices, SIMD-optimized
  [[nodiscard]]
  auto spmv(std::span<const ValueT> x) const -> std::vector<ValueT>
      requires(IsCSR) {
    error::check_arg(static_cast<IndexT>(x.size()) == cols_,
                     "vector size must match matrix columns");

    std::vector<ValueT> y(static_cast<std::size_t>(rows_), ValueT{0});
    
    if (!valid() || rows_ == 0 || cols_ == 0) {
      return y;
    }

    // Parallel SpMV for large matrices
    if (rows_ > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      threading::parallel_for(
          static_cast<threading::Index>(0), static_cast<threading::Index>(rows_),
          [this, &x, &y](threading::Index i) {
            y[static_cast<std::size_t>(i)] = compute_row_dot_product(i, x);
          },
          threading::DEFAULT_GRAIN_SIZE);
    } else {
      // Serial SpMV for small matrices
      for (IndexT i = 0; i < rows_; ++i) {
        y[static_cast<std::size_t>(i)] = compute_row_dot_product(i, x);
      }
    }

    return y;
  }

  /// @brief Sparse matrix-vector multiplication: y = A^T * x (CSC)
  /// @param[in] x Input vector
  /// @return Result vector y
  [[nodiscard]]
  auto spmv(std::span<const ValueT> x) const -> std::vector<ValueT>
      requires(!IsCSR) {
    error::check_arg(static_cast<IndexT>(x.size()) == rows_,
                     "vector size must match matrix rows");

    std::vector<ValueT> y(static_cast<std::size_t>(cols_), ValueT{0});
    
    if (!valid() || rows_ == 0 || cols_ == 0) {
      return y;
    }

    // For CSC, accumulate into result vector
    for (IndexT j = 0; j < cols_; ++j) {
      const auto& vals = values_[static_cast<std::size_t>(j)];
      const auto& idxs = indices_[static_cast<std::size_t>(j)];
      
      for (Size k = 0; k < vals.size(); ++k) {
        y[static_cast<std::size_t>(j)] +=
            vals[k] * x[static_cast<std::size_t>(idxs[k])];
      }
    }

    return y;
  }

  // -------------------------------------------------------------------------
  // Reduction Operations
  // -------------------------------------------------------------------------

  /// @brief Sum of all non-zero values (parallel)
  /// @return Sum of all values
  [[nodiscard]]
  auto sum() const noexcept -> ValueT {
    if (!valid()) {
      return ValueT{0};
    }

    const IndexT pdim = primary_dim();

    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      return threading::parallel_reduce(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          ValueT{0},
          [this](threading::Index i) -> ValueT {
            const auto& vals = values_[static_cast<std::size_t>(i)];
            return vectorize::sum(vals.to_std_span());
          },
          std::plus<ValueT>{});
    }

    ValueT result = ValueT{0};
    for (const auto& v : values_) {
      result += vectorize::sum(v.to_std_span());
    }
    return result;
  }

  /// @brief Sum of absolute values (L1 norm)
  /// @return L1 norm
  [[nodiscard]]
  auto sum_abs() const noexcept -> ValueT {
    if (!valid()) {
      return ValueT{0};
    }

    const IndexT pdim = primary_dim();

    if (pdim > threading::MIN_PARALLEL_SIZE / detail::kSparseParallelThreshold) {
      return threading::parallel_reduce(
          static_cast<threading::Index>(0), static_cast<threading::Index>(pdim),
          ValueT{0},
          [this](threading::Index i) -> ValueT {
            const auto& vals = values_[static_cast<std::size_t>(i)];
            ValueT local_sum = ValueT{0};
            for (const auto& val : vals) {
              local_sum += (val >= ValueT{0}) ? val : -val;
            }
            return local_sum;
          },
          std::plus<ValueT>{});
    }

    ValueT result = ValueT{0};
    for (const auto& v : values_) {
      for (const auto& val : v) {
        result += (val >= ValueT{0}) ? val : -val;
      }
    }
    return result;
  }

  /// @brief Maximum absolute value
  /// @return Maximum |value|
  [[nodiscard]]
  auto max_abs() const noexcept -> ValueT {
    if (!valid() || nnz() == 0) {
      return ValueT{0};
    }

    ValueT max_val = ValueT{0};
    for (const auto& v : values_) {
      for (const auto& val : v) {
        const auto abs_val = (val >= ValueT{0}) ? val : -val;
        if (abs_val > max_val) {
          max_val = abs_val;
        }
      }
    }
    return max_val;
  }

  // -------------------------------------------------------------------------
  // Row/Column Operations
  // -------------------------------------------------------------------------

  /// @brief Get row sum for each row (CSR)
  /// @return Vector of row sums
  [[nodiscard]]
  auto row_sums() const -> std::vector<ValueT>
      requires(IsCSR) {
    std::vector<ValueT> sums(static_cast<std::size_t>(rows_), ValueT{0});
    
    if (!valid()) {
      return sums;
    }

    for (IndexT i = 0; i < rows_; ++i) {
      const auto& vals = values_[static_cast<std::size_t>(i)];
      sums[static_cast<std::size_t>(i)] = vectorize::sum(vals.to_std_span());
    }

    return sums;
  }

  /// @brief Get column sum for each column (CSC)
  /// @return Vector of column sums
  [[nodiscard]]
  auto col_sums() const -> std::vector<ValueT>
      requires(!IsCSR) {
    std::vector<ValueT> sums(static_cast<std::size_t>(cols_), ValueT{0});
    
    if (!valid()) {
      return sums;
    }

    for (IndexT j = 0; j < cols_; ++j) {
      const auto& vals = values_[static_cast<std::size_t>(j)];
      sums[static_cast<std::size_t>(j)] = vectorize::sum(vals.to_std_span());
    }

    return sums;
  }

  /// @brief Normalize rows to unit sum (CSR)
  /// @note Skips rows with zero sum
  auto normalize_rows() -> void
      requires(IsCSR) {
    if (!valid()) {
      return;
    }

    for (IndexT i = 0; i < rows_; ++i) {
      auto& vals = values_[static_cast<std::size_t>(i)];
      const ValueT row_sum = vectorize::sum(vals.to_std_span());
      
      if (row_sum > ValueT{0} || row_sum < ValueT{0}) {
        for (auto& val : vals) {
          val /= row_sum;
        }
      }
    }
  }

  /// @brief Normalize columns to unit sum (CSC)
  /// @note Skips columns with zero sum
  auto normalize_cols() -> void
      requires(!IsCSR) {
    if (!valid()) {
      return;
    }

    for (IndexT j = 0; j < cols_; ++j) {
      auto& vals = values_[static_cast<std::size_t>(j)];
      const ValueT col_sum = vectorize::sum(vals.to_std_span());
      
      if (col_sum > ValueT{0} || col_sum < ValueT{0}) {
        for (auto& val : vals) {
          val /= col_sum;
        }
      }
    }
  }

  /// @brief Remove elements below threshold (in-place)
  /// @param[in] threshold Absolute value threshold
  /// @note This modifies spans in-place without reallocation
  /// @note Invalidates NNZ cache
  auto remove_below_threshold(ValueT threshold) -> void {
    if (!valid() || threshold < ValueT{0}) {
      return;
    }

    const IndexT pdim = primary_dim();

    for (IndexT i = 0; i < pdim; ++i) {
      auto& vals = values_[static_cast<std::size_t>(i)];
      auto& idxs = indices_[static_cast<std::size_t>(i)];
      
      if (vals.empty()) {
        continue;
      }

      // Find elements above threshold and compact
      Size write_pos = 0;
      for (Size read_pos = 0; read_pos < vals.size(); ++read_pos) {
        const auto val = vals[read_pos];
        const auto abs_val = (val >= ValueT{0}) ? val : -val;
        
        if (abs_val > threshold) {
          if (write_pos != read_pos) {
            vals[write_pos] = vals[read_pos];
            idxs[write_pos] = idxs[read_pos];
          }
          ++write_pos;
        }
      }

      // Shrink spans if elements were removed
      if (write_pos < vals.size()) {
        vals.set_size(write_pos);
        idxs.set_size(write_pos);
      }
    }

    // Mark NNZ dirty
    mark_nnz_dirty();
  }

  // -------------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------------

  /// @brief Export to dense matrix (row-major)
  [[nodiscard]]
  auto to_dense() const -> std::vector<ValueT> {
    const auto size =
        static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
    std::vector<ValueT> result(size, ValueT{0});

    if (!valid()) {
      return result;
    }

    const IndexT pdim = primary_dim();

    if constexpr (IsCSR) {
      for (IndexT i = 0; i < rows_; ++i) {
        const auto& vals = values_[static_cast<std::size_t>(i)];
        const auto& idxs = indices_[static_cast<std::size_t>(i)];
        const auto row_offset =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_);
        for (Size k = 0; k < vals.size(); ++k) {
          result[row_offset + static_cast<std::size_t>(idxs[k])] = vals[k];
        }
      }
    } else {
      for (IndexT j = 0; j < cols_; ++j) {
        const auto& vals = values_[static_cast<std::size_t>(j)];
        const auto& idxs = indices_[static_cast<std::size_t>(j)];
        for (Size k = 0; k < vals.size(); ++k) {
          result[static_cast<std::size_t>(idxs[k]) *
                     static_cast<std::size_t>(cols_) +
                 static_cast<std::size_t>(j)] = vals[k];
        }
      }
    }

    return result;
  }

 private:
  // -------------------------------------------------------------------------
  // NNZ Cache Management
  // -------------------------------------------------------------------------

  /// @brief Mark NNZ as dirty (needs recomputation)
  auto mark_nnz_dirty() const noexcept -> void {
    nnz_dirty_ = true;
  }

  /// @brief Set NNZ directly (for factory methods that know the count)
  /// @param[in] nnz Known NNZ count
  auto set_nnz(IndexT nnz) const noexcept -> void {
    nnz_cached_ = nnz;
    nnz_dirty_ = false;
  }

  // -------------------------------------------------------------------------
  // Scale Helpers
  // -------------------------------------------------------------------------

  /// @brief Scale a single row/column values (SIMD-optimized)
  /// @param[in,out] v Values span to scale
  /// @param[in] factor Scale factor
  static SCL_FORCE_INLINE
  auto scale_row_values(Span<ValueT>& v, ValueT factor) -> void {
    if (v.empty()) {
      return;
    }
    
    if constexpr (std::is_arithmetic_v<ValueT>) {
      // Use SIMD scale for arithmetic types
      vectorize::scale(v.to_std_span(), v.to_std_span(), factor);
    } else {
      for (auto& val : v) {
        val *= factor;
      }
    }
  }

  // -------------------------------------------------------------------------
  // SpMV Helpers
  // -------------------------------------------------------------------------

  /// @brief Compute dot product for a single row (CSR, SIMD-optimized)
  /// @param[in] row Row index
  /// @param[in] x Input vector
  /// @return Dot product of row with x
  [[nodiscard]]
  SCL_FORCE_INLINE
  auto compute_row_dot_product(IndexT row, std::span<const ValueT> x) const noexcept
      -> ValueT 
      requires(IsCSR) {
    const auto& vals = values_[static_cast<std::size_t>(row)];
    const auto& idxs = indices_[static_cast<std::size_t>(row)];
    
    const Size len = vals.size();
    if (len == 0) {
      return ValueT{0};
    }

    // For dense access patterns or small rows, use simple loop
    constexpr Size kGatherThreshold = 32;
    if (len < kGatherThreshold) {
      ValueT sum = ValueT{0};
      for (Size k = 0; k < len; ++k) {
        sum += vals[k] * x[static_cast<std::size_t>(idxs[k])];
      }
      return sum;
    }

    // For larger sparse rows with arithmetic types, use SIMD with gather
    if constexpr (std::is_arithmetic_v<ValueT>) {
      // Gather x values, then use SIMD dot product
      // Note: vectorize::dot expects contiguous data, so we gather first
      ValueT sum = ValueT{0};
      
      // Process with prefetching for irregular access
      constexpr Size kPrefetchDistance = 16;
      for (Size k = 0; k < len; ++k) {
        // Prefetch future x values
        if (k + kPrefetchDistance < len) [[likely]] {
          const auto future_idx = static_cast<std::size_t>(
              idxs[k + kPrefetchDistance]);
          SCL_PREFETCH_READ(&x[future_idx], 0);
        }
        sum += vals[k] * x[static_cast<std::size_t>(idxs[k])];
      }
      return sum;
    }

    // Fallback for non-arithmetic types
    ValueT sum = ValueT{0};
    for (Size k = 0; k < len; ++k) {
      sum += vals[k] * x[static_cast<std::size_t>(idxs[k])];
    }
    return sum;
  }

  // -------------------------------------------------------------------------
  // Sort Helpers
  // -------------------------------------------------------------------------

  /// @brief Sort a single row/column indices (serial)
  /// @param[in] i Primary dimension index
  /// @param[in] small_threshold Stack allocation threshold
  auto sort_row_indices_serial(IndexT i, Size small_threshold) -> void {
    auto& vals = values_[static_cast<std::size_t>(i)];
    auto& idxs = indices_[static_cast<std::size_t>(i)];

    const Size len = vals.size();
    if (len <= 1) {
      return;
    }

    // Small buffer optimization
    if (len <= small_threshold) {
      sort_row_stack_allocated(vals, idxs, len, small_threshold);
    } else {
      sort_row_heap_allocated(vals, idxs, len);
    }
  }

  /// @brief Sort using stack-allocated buffers
  static auto sort_row_stack_allocated(Span<ValueT>& vals, Span<IndexT>& idxs,
                                       Size len, Size threshold) -> void {
    // Use VLA-like pattern with placement new
    auto* perm_buf = static_cast<Size*>(
        alloca(threshold * (sizeof(Size) + sizeof(IndexT) + sizeof(ValueT))));
    auto* idx_buf = reinterpret_cast<IndexT*>(perm_buf + threshold);
    auto* val_buf = reinterpret_cast<ValueT*>(idx_buf + threshold);
    
    for (Size k = 0; k < len; ++k) {
      perm_buf[k] = k;
    }
    
    std::sort(perm_buf, perm_buf + len,
             [&idxs](Size a, Size b) { return idxs[a] < idxs[b]; });
    
    for (Size k = 0; k < len; ++k) {
      idx_buf[k] = idxs[perm_buf[k]];
      val_buf[k] = vals[perm_buf[k]];
    }
    
    for (Size k = 0; k < len; ++k) {
      idxs[k] = idx_buf[k];
      vals[k] = val_buf[k];
    }
  }

  /// @brief Sort using heap-allocated buffers
  static auto sort_row_heap_allocated(Span<ValueT>& vals, Span<IndexT>& idxs,
                                      Size len) -> void {
    std::vector<Size> perm(len);
    std::iota(perm.begin(), perm.end(), Size{0});
    
    std::sort(perm.begin(), perm.end(), [&idxs](Size a, Size b) {
      return idxs[a] < idxs[b];
    });
    
    std::vector<IndexT> sorted_idx(len);
    std::vector<ValueT> sorted_val(len);
    for (Size k = 0; k < len; ++k) {
      sorted_idx[k] = idxs[perm[k]];
      sorted_val[k] = vals[perm[k]];
    }
    
    for (Size k = 0; k < len; ++k) {
      idxs[k] = sorted_idx[k];
      vals[k] = sorted_val[k];
    }
  }

  // -------------------------------------------------------------------------
  // Dense Conversion Helpers
  // -------------------------------------------------------------------------

  /// @brief Count non-zeros per primary dimension
  [[nodiscard]]
  static
  auto count_nonzeros_per_dim(std::span<const ValueT> data,
                              [[maybe_unused]] IndexT rows, IndexT cols,
                              IndexT pdim, IndexT sdim,
                              ValueT threshold) -> std::vector<IndexT> {
    std::vector<IndexT> counts(static_cast<std::size_t>(pdim), 0);
    
    for (IndexT i = 0; i < pdim; ++i) {
      for (IndexT j = 0; j < sdim; ++j) {
        const auto idx = IsCSR 
            ? static_cast<std::size_t>(i * cols + j)
            : static_cast<std::size_t>(j * cols + i);
        
        const auto val = data[idx];
        if (val > threshold || val < -threshold) {
          ++counts[static_cast<std::size_t>(i)];
        }
      }
    }
    
    return counts;
  }

  /// @brief Fill sparse matrix from dense data
  static
  auto fill_from_dense_data(Sparse& result, std::span<const ValueT> data,
                            IndexT cols, IndexT pdim, IndexT sdim,
                            ValueT threshold) -> void {
    std::vector<IndexT> insert_pos(static_cast<std::size_t>(pdim), 0);
    
    for (IndexT i = 0; i < pdim; ++i) {
      for (IndexT j = 0; j < sdim; ++j) {
        const auto idx = IsCSR 
            ? static_cast<std::size_t>(i * cols + j)
            : static_cast<std::size_t>(j * cols + i);
        
        const auto val = data[idx];
        if (val > threshold || val < -threshold) {
          const IndexT pos = insert_pos[static_cast<std::size_t>(i)]++;
          result.values_[static_cast<std::size_t>(i)][static_cast<Size>(pos)] = val;
          result.indices_[static_cast<std::size_t>(i)][static_cast<Size>(pos)] = j;
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Buffer Allocation Helpers
  // -------------------------------------------------------------------------

  /// @brief Allocate buffers using make_batch_spans
  /// @param[out] values Values spans (will be populated)
  /// @param[out] indices Indices spans (will be populated)
  /// @param[in] nnzs NNZ counts per row/column
  /// @param[in] min_buffer_size Strategy hint (0=single, SIZE_MAX=fragmented)
  /// @return true on success, false on allocation failure
  static
  auto allocate_with_strategy(std::vector<Span<ValueT>>& values,
                              std::vector<Span<IndexT>>& indices,
                              std::span<const IndexT> nnzs,
                              Size min_buffer_size) -> bool {
    const auto pdim = static_cast<Size>(nnzs.size());
    if (pdim == 0) {
      return true;
    }

    // Validate and convert to Size vector
    std::vector<Size> sizes;
    sizes.reserve(pdim);
    for (const auto count : nnzs) {
      if (count < 0) {
        return false;
      }
      sizes.push_back(static_cast<Size>(count));
    }

    // Determine allocation strategy
    if (min_buffer_size == static_cast<Size>(-1)) {
      // Fragmented: each row/column separate
      for (Size i = 0; i < pdim; ++i) {
        if (sizes[i] > 0) {
          values[i] = Span<ValueT>::create_aligned(sizes[i]);
          indices[i] = Span<IndexT>::create_aligned(sizes[i]);
          if (!values[i] || !indices[i]) {
            return false;
          }
        }
      }
      return true;
    }

    // Single buffer or grouped: use make_batch_spans
    auto val_batch = make_batch_spans<ValueT>(sizes);
    auto idx_batch = make_batch_spans<IndexT>(sizes);
    
    if (!val_batch || !idx_batch) {
      return false;
    }

    // Transfer ownership
    for (Size i = 0; i < pdim; ++i) {
      values[i] = std::move(val_batch.spans[i]);
      indices[i] = std::move(idx_batch.spans[i]);
    }

    return true;
  }

  std::vector<Span<ValueT>> values_;  ///< Values for each row/column
  std::vector<Span<IndexT>> indices_; ///< Indices for each row/column
  IndexT rows_ = 0;
  IndexT cols_ = 0;
  mutable IndexT nnz_cached_ = 0;     ///< Cached total NNZ count
  mutable bool nnz_dirty_ = true;     ///< NNZ needs recomputation
};

// =============================================================================
// SECTION 5: Static Assertions
// =============================================================================

static_assert(std::is_nothrow_move_constructible_v<CSR>,
              "CSR must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<CSR>,
              "CSR must be nothrow move assignable");

}  // namespace scl

