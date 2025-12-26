#pragma once

#include "scl/io/mmatrix.hpp"
#include <string>
#include <cstdio>
#include <cinttypes>

// =============================================================================
/// @file utils.hpp
/// @brief Utility Functions for Common File Organizations
///
/// Provides optional utility functions for standard file layouts and operations.
/// Users can use these for convenience or define their own organization.
///
/// This module encodes opinions about file structure (unlike mmap.hpp/mmatrix.hpp
/// which are pure data structures).
// =============================================================================

namespace scl::io {

// =============================================================================
// Standard Directory Layout Helpers
// =============================================================================

/// @brief Load CSR matrix from standard directory layout.
///
/// Expected structure:
///   {dir_path}/data.bin    - Values (type T)
///   {dir_path}/indices.bin - Column indices (int64)
///   {dir_path}/indptr.bin  - Row pointers (int64, size: rows+1)
///
/// This is a convenience function that encodes a specific file organization.
/// For custom layouts, construct MappedCustomSparse directly from MappedArrays.
///
/// @param dir_path Directory containing binary files
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @return MappedCustomSparse constructed from standard layout
template <typename T>
inline MappedCustomSparse<T> mount_standard_layout(
    const std::string& dir_path,
    Index rows,
    Index cols,
    Index nnz
) {
    return MappedCustomSparse<T>(
        MappedArray<T>(dir_path + "/data.bin"),
        MappedArray<Index>(dir_path + "/indices.bin"),
        MappedArray<Index>(dir_path + "/indptr.bin"),
        rows, cols, nnz
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
///   nnz <value>
///
/// This is ONE possible metadata format. Users can define their own.
inline std::tuple<Index, Index, Index> load_simple_metadata(const std::string& meta_path) {
    Index rows = 0, cols = 0, nnz = 0;
    
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
    while (::fgets(line, sizeof(line), file)) {
#if SCL_PLATFORM_POSIX
        if (::sscanf(line, "rows %" SCNd64, &rows) == 1) continue;
        if (::sscanf(line, "cols %" SCNd64, &cols) == 1) continue;
        if (::sscanf(line, "nnz %" SCNd64, &nnz) == 1) continue;
#elif SCL_PLATFORM_WINDOWS
        if (::sscanf_s(line, "rows %lld", &rows) == 1) continue;
        if (::sscanf_s(line, "cols %lld", &cols) == 1) continue;
        if (::sscanf_s(line, "nnz %lld", &nnz) == 1) continue;
#endif
    }
    
    ::fclose(file);
    
    if (rows == 0 || cols == 0) {
        throw ValueError("Invalid metadata: " + meta_path);
    }
    
    return std::make_tuple(rows, cols, nnz);
}

/// @brief Save metadata to simple text format.
inline void save_simple_metadata(
    const std::string& meta_path,
    Index rows,
    Index cols,
    Index nnz
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
    ::fprintf(file, "nnz %" PRId64 "\n", nnz);
#elif SCL_PLATFORM_WINDOWS
    ::fprintf(file, "rows %lld\n", rows);
    ::fprintf(file, "cols %lld\n", cols);
    ::fprintf(file, "nnz %lld\n", nnz);
#endif
    
    ::fclose(file);
}

/// @brief Load CSR matrix with automatic metadata loading.
///
/// Looks for {dir_path}/matrix.meta and loads dimensions automatically.
/// This encodes TWO opinions:
/// 1. Standard directory layout (data.bin, indices.bin, indptr.bin)
/// 2. Metadata file naming (matrix.meta)
///
/// @param dir_path Directory containing matrix files
/// @return MappedCustomSparse with auto-loaded metadata
template <typename T>
inline MappedCustomSparse<T> mount_with_metadata(const std::string& dir_path) {
    auto [rows, cols, nnz] = load_simple_metadata(dir_path + "/matrix.meta");
    return mount_standard_layout<T>(dir_path, rows, cols, nnz);
}

// =============================================================================
// Custom Layout Examples
// =============================================================================

/// @brief Example: Load matrix from non-standard file names.
///
/// Demonstrates flexibility - users can map ANY file layout.
template <typename T>
inline MappedCustomSparse<T> mount_custom_layout(
    const std::string& values_file,
    const std::string& cols_file,
    const std::string& rows_file,
    Index rows,
    Index cols,
    Index nnz
) {
    return MappedCustomSparse<T>(
        MappedArray<T>(values_file),
        MappedArray<Index>(cols_file),
        MappedArray<Index>(rows_file),
        rows, cols, nnz
    );
}

/// @brief Example: Load matrix where data is in a subdirectory.
template <typename T>
inline MappedCustomSparse<T> mount_nested_layout(
    const std::string& base_dir,
    const std::string& matrix_name,
    Index rows,
    Index cols,
    Index nnz
) {
    std::string matrix_dir = base_dir + "/matrices/" + matrix_name;
    return mount_standard_layout<T>(matrix_dir, rows, cols, nnz);
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
/// @param base_dir Base directory
/// @param matrix_names Names of subdirectories
/// @return Vector of MappedCustomSparse objects
template <typename T>
inline std::vector<MappedCustomSparse<T>> mount_batch(
    const std::string& base_dir,
    const std::vector<std::string>& matrix_names
) {
    std::vector<MappedCustomSparse<T>> matrices;
    matrices.reserve(matrix_names.size());
    
    for (const auto& name : matrix_names) {
        matrices.push_back(mount_with_metadata<T>(base_dir + "/" + name));
    }
    
    return matrices;
}

} // namespace scl::io

