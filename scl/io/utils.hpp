#pragma once

#include "scl/io/mmatrix.hpp"
#include <string>
#include <cstdio>
#include <cinttypes>
#include <variant>

// =============================================================================
/// @file utils.hpp
/// @brief Utility Functions for Common File Organizations
///
/// Provides optional utility functions for standard file layouts and operations.
/// Users can use these for convenience or define their own organization.
///
/// This module encodes opinions about file structure (unlike mmap.hpp/mmatrix.hpp
/// which are pure data structures).
///
/// Supports both CSR and CSC formats through unified interface.
// =============================================================================

namespace scl::io {

// =============================================================================
// Standard Directory Layout Helpers
// =============================================================================

/// @brief Load sparse matrix from standard directory layout.
///
/// Expected structure:
///   {dir_path}/data.bin    - Values (type T)
///   {dir_path}/indices.bin - Column/Row indices (int64)
///   {dir_path}/indptr.bin  - Row/Col pointers (int64, size: primary_dim+1)
///
/// This is a convenience function that encodes a specific file organization.
/// For custom layouts, construct MappedCustomSparse directly from MappedArrays.
///
/// @tparam T Value type
/// @tparam IsCSR true for CSR, false for CSC
/// @param dir_path Directory containing binary files
/// @param rows Number of rows
/// @param cols Number of columns
/// @return MappedCustomSparse constructed from standard layout
template <typename T, bool IsCSR = true>
inline MappedCustomSparse<T, IsCSR> mount_standard_layout(
    const std::string& dir_path,
    Index rows,
    Index cols
) {
    return MappedCustomSparse<T, IsCSR>(
        MappedArray<T>(dir_path + "/data.bin"),
        MappedArray<Index>(dir_path + "/indices.bin"),
        MappedArray<Index>(dir_path + "/indptr.bin"),
        rows, cols
    );
}

// =============================================================================
// Metadata File Helpers
// =============================================================================

/// @brief Simple metadata format (plain text).
///
/// Format:
///   rows <value>
///   cols <value>
///   format <csr|csc>  (optional, defaults to csr)
///
/// This is ONE possible metadata format. Users can define their own.
struct MatrixMetadata {
    Index rows;
    Index cols;
    bool is_csr;
    
    MatrixMetadata() : rows(0), cols(0), is_csr(true) {}
    MatrixMetadata(Index r, Index c, bool csr = true) 
        : rows(r), cols(c), is_csr(csr) {}
};

inline MatrixMetadata load_simple_metadata(const std::string& meta_path) {
    MatrixMetadata meta;
    
#if SCL_PLATFORM_POSIX
    FILE* file = ::fopen(meta_path.c_str(), "r");
#elif SCL_PLATFORM_WINDOWS
    FILE* file = nullptr;
    ::fopen_s(&file, meta_path.c_str(), "r");
#endif
    
    if (!file) {
        throw IOError("Failed to open metadata: " + meta_path);
    }
    
    char line[256];
    char format_str[32] = "csr";
    
    while (::fgets(line, sizeof(line), file)) {
#if SCL_PLATFORM_POSIX
        if (::sscanf(line, "rows %" SCNd64, &meta.rows) == 1) continue;
        if (::sscanf(line, "cols %" SCNd64, &meta.cols) == 1) continue;
        if (::sscanf(line, "format %31s", format_str) == 1) continue;
#elif SCL_PLATFORM_WINDOWS
        if (::sscanf_s(line, "rows %lld", &meta.rows) == 1) continue;
        if (::sscanf_s(line, "cols %lld", &meta.cols) == 1) continue;
        if (::sscanf_s(line, "format %31s", format_str, 32) == 1) continue;
#endif
    }
    
    ::fclose(file);
    
    if (meta.rows == 0 || meta.cols == 0) {
        throw ValueError("Invalid metadata: " + meta_path);
    }
    
    // Parse format
    std::string fmt(format_str);
    if (fmt == "csc" || fmt == "CSC") {
        meta.is_csr = false;
    } else {
        meta.is_csr = true;  // Default to CSR
    }
    
    return meta;
}

/// @brief Save metadata to simple text format.
inline void save_simple_metadata(
    const std::string& meta_path,
    Index rows,
    Index cols,
    bool is_csr = true
) {
#if SCL_PLATFORM_POSIX
    FILE* file = ::fopen(meta_path.c_str(), "w");
#elif SCL_PLATFORM_WINDOWS
    FILE* file = nullptr;
    ::fopen_s(&file, meta_path.c_str(), "w");
#endif
    
    if (!file) {
        throw IOError("Failed to create metadata: " + meta_path);
    }
    
#if SCL_PLATFORM_POSIX
    ::fprintf(file, "rows %" PRId64 "\n", rows);
    ::fprintf(file, "cols %" PRId64 "\n", cols);
    ::fprintf(file, "format %s\n", is_csr ? "csr" : "csc");
#elif SCL_PLATFORM_WINDOWS
    ::fprintf(file, "rows %lld\n", rows);
    ::fprintf(file, "cols %lld\n", cols);
    ::fprintf(file, "format %s\n", is_csr ? "csr" : "csc");
#endif
    
    ::fclose(file);
}

/// @brief Load sparse matrix with automatic metadata loading.
///
/// Looks for {dir_path}/matrix.meta and loads dimensions automatically.
/// This encodes TWO opinions:
/// 1. Standard directory layout (data.bin, indices.bin, indptr.bin)
/// 2. Metadata file naming (matrix.meta)
///
/// @tparam T Value type
/// @tparam IsCSR true for CSR, false for CSC
/// @param dir_path Directory containing matrix files
/// @return MappedCustomSparse with auto-loaded metadata
template <typename T, bool IsCSR = true>
inline MappedCustomSparse<T, IsCSR> mount_with_metadata(const std::string& dir_path) {
    auto meta = load_simple_metadata(dir_path + "/matrix.meta");
    
    // Verify format matches template parameter
    if (meta.is_csr != IsCSR) {
        throw ValueError("Metadata format mismatch: expected " + 
                        std::string(IsCSR ? "CSR" : "CSC") + 
                        ", got " + std::string(meta.is_csr ? "CSR" : "CSC"));
    }
    
    return mount_standard_layout<T, IsCSR>(dir_path, meta.rows, meta.cols);
}

/// @brief Load sparse matrix with automatic format detection.
///
/// Returns either CSR or CSC based on metadata.
/// This is a runtime-polymorphic version using std::variant.
template <typename T>
inline std::variant<MappedCustomSparse<T, true>, MappedCustomSparse<T, false>> 
mount_with_auto_format(const std::string& dir_path) {
    auto meta = load_simple_metadata(dir_path + "/matrix.meta");
    
    if (meta.is_csr) {
        return mount_standard_layout<T, true>(dir_path, meta.rows, meta.cols);
    } else {
        return mount_standard_layout<T, false>(dir_path, meta.rows, meta.cols);
    }
}

// =============================================================================
// Custom Layout Examples
// =============================================================================

/// @brief Example: Load matrix from non-standard file names.
///
/// Demonstrates flexibility - users can map ANY file layout.
template <typename T, bool IsCSR = true>
inline MappedCustomSparse<T, IsCSR> mount_custom_layout(
    const std::string& values_file,
    const std::string& indices_file,
    const std::string& indptr_file,
    Index rows,
    Index cols
) {
    return MappedCustomSparse<T, IsCSR>(
        MappedArray<T>(values_file),
        MappedArray<Index>(indices_file),
        MappedArray<Index>(indptr_file),
        rows, cols
    );
}

/// @brief Example: Load matrix where data is in a subdirectory.
template <typename T, bool IsCSR = true>
inline MappedCustomSparse<T, IsCSR> mount_nested_layout(
    const std::string& base_dir,
    const std::string& matrix_name,
    Index rows,
    Index cols
) {
    std::string matrix_dir = base_dir + "/matrices/" + matrix_name;
    return mount_standard_layout<T, IsCSR>(matrix_dir, rows, cols);
}

// =============================================================================
// Batch Operations
// =============================================================================

/// @brief Load multiple matrices from a directory tree.
///
/// Example directory structure:
///   {base_dir}/matrix1/data.bin, indices.bin, indptr.bin, matrix.meta
///   {base_dir}/matrix2/data.bin, indices.bin, indptr.bin, matrix.meta
///   ...
///
/// @tparam T Value type
/// @tparam IsCSR true for CSR, false for CSC
/// @param base_dir Base directory
/// @param matrix_names Names of subdirectories
/// @return Vector of MappedCustomSparse objects
template <typename T, bool IsCSR = true>
inline std::vector<MappedCustomSparse<T, IsCSR>> mount_batch(
    const std::string& base_dir,
    const std::vector<std::string>& matrix_names
) {
    std::vector<MappedCustomSparse<T, IsCSR>> matrices;
    matrices.reserve(matrix_names.size());
    
    for (const auto& name : matrix_names) {
        matrices.push_back(mount_with_metadata<T, IsCSR>(base_dir + "/" + name));
    }
    
    return matrices;
}

// =============================================================================
// Materialization Helpers
// =============================================================================

/// @brief Save OwnedSparse to standard directory layout.
///
/// Creates:
///   {dir_path}/data.bin
///   {dir_path}/indices.bin
///   {dir_path}/indptr.bin
///   {dir_path}/matrix.meta
///
/// @tparam T Value type
/// @tparam IsCSR true for CSR, false for CSC
/// @param dir_path Target directory (must exist)
/// @param matrix Matrix to save
template <typename T, bool IsCSR>
inline void save_to_standard_layout(
    const std::string& dir_path,
    const OwnedSparse<T, IsCSR>& matrix
) {
    // Save binary data
    {
        std::string data_path = dir_path + "/data.bin";
#if SCL_PLATFORM_POSIX
        FILE* file = ::fopen(data_path.c_str(), "wb");
#elif SCL_PLATFORM_WINDOWS
        FILE* file = nullptr;
        ::fopen_s(&file, data_path.c_str(), "wb");
#endif
        if (!file) throw IOError("Failed to create: " + data_path);
        ::fwrite(matrix.data.data(), sizeof(T), matrix.data.size(), file);
        ::fclose(file);
    }
    
    // Save indices
    {
        std::string indices_path = dir_path + "/indices.bin";
#if SCL_PLATFORM_POSIX
        FILE* file = ::fopen(indices_path.c_str(), "wb");
#elif SCL_PLATFORM_WINDOWS
        FILE* file = nullptr;
        ::fopen_s(&file, indices_path.c_str(), "wb");
#endif
        if (!file) throw IOError("Failed to create: " + indices_path);
        ::fwrite(matrix.indices.data(), sizeof(Index), matrix.indices.size(), file);
        ::fclose(file);
    }
    
    // Save indptr
    {
        std::string indptr_path = dir_path + "/indptr.bin";
#if SCL_PLATFORM_POSIX
        FILE* file = ::fopen(indptr_path.c_str(), "wb");
#elif SCL_PLATFORM_WINDOWS
        FILE* file = nullptr;
        ::fopen_s(&file, indptr_path.c_str(), "wb");
#endif
        if (!file) throw IOError("Failed to create: " + indptr_path);
        ::fwrite(matrix.indptr.data(), sizeof(Index), matrix.indptr.size(), file);
        ::fclose(file);
    }
    
    // Save metadata
    save_simple_metadata(dir_path + "/matrix.meta", matrix.rows, matrix.cols, IsCSR);
}

/// @brief Save any SparseLike matrix to standard layout.
///
/// Materializes the matrix first if needed.
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
inline void save_sparse_to_layout(
    const std::string& dir_path,
    const MatrixT& matrix
) {
    // Materialize if needed
    if constexpr (requires { matrix.materialize(); }) {
        auto owned = matrix.materialize();
        save_to_standard_layout(dir_path, owned);
    } else {
        // Already OwnedSparse
        save_to_standard_layout(dir_path, matrix);
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

template <typename T>
using MappedCSR = MappedCustomSparse<T, true>;

template <typename T>
using MappedCSC = MappedCustomSparse<T, false>;

using MappedCSRReal = MappedCSR<Real>;
using MappedCSCReal = MappedCSC<Real>;

// =============================================================================
// Backward Compatibility (Deprecated)
// =============================================================================

/// @deprecated Use mount_standard_layout<T, true> instead
template <typename T>
[[deprecated("Use mount_standard_layout<T, true> instead")]]
inline MappedCustomSparse<T, true> mount_standard_layout(
    const std::string& dir_path,
    Index rows,
    Index cols
) {
    return mount_standard_layout<T, true>(dir_path, rows, cols);
}

/// @deprecated Use load_simple_metadata() which returns MatrixMetadata
[[deprecated("Use load_simple_metadata() which returns MatrixMetadata")]]
inline std::tuple<Index, Index, Index> load_simple_metadata_legacy(
    const std::string& meta_path
) {
    auto meta = load_simple_metadata(meta_path);
    // Compute nnz by loading indptr (expensive!)
    // This is why the new API doesn't include nnz in metadata
    return std::make_tuple(meta.rows, meta.cols, Index(0));
}

} // namespace scl::io
