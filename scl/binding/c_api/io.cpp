// =============================================================================
/// @file io.cpp
/// @brief C ABI Wrapper for SCL I/O Module
///
/// Provides C-compatible exports for:
/// - HDF5 sparse matrix loading (h5_tools.hpp)
/// - Memory-mapped array operations (mmap.hpp)
/// - File I/O utilities
///
/// All functions follow the standard error protocol:
/// - Return 0 on success
/// - Return -1 on failure (error message in thread-local buffer)
// =============================================================================

#include "scl/binding/c_api/scl_c_api.h"
#include "scl/binding/c_api/error.hpp"
#include "scl/core/type.hpp"
#include "scl/io/mmap.hpp"

#ifdef SCL_HAS_HDF5
#include "scl/io/hdf5.hpp"
#include "scl/io/h5_tools.hpp"
#endif

#include <string>
#include <vector>
#include <sys/stat.h>
#include <fstream>

extern "C" {

// =============================================================================
// SECTION 1: Memory-Mapped Array Operations
// =============================================================================

/// @brief Create memory-mapped array from file path
int scl_mmap_array_open(
    const char* filepath,
    scl_size_t element_size,
    bool writable,
    void** out_ptr,
    scl_size_t* out_size
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_ptr || !out_size) {
            throw scl::ValueError("Null pointer argument");
        }

        if (element_size == sizeof(scl::Real)) {
            auto arr = new scl::io::MappedArray<scl::Real>(filepath, writable);
            *out_ptr = const_cast<scl::Real*>(arr->data());
            *out_size = arr->size();
        } else if (element_size == sizeof(scl::Index)) {
            auto arr = new scl::io::MappedArray<scl::Index>(filepath, writable);
            *out_ptr = const_cast<scl::Index*>(arr->data());
            *out_size = arr->size();
        } else if (element_size == sizeof(scl::Byte)) {
            auto arr = new scl::io::MappedArray<scl::Byte>(filepath, writable);
            *out_ptr = const_cast<scl::Byte*>(arr->data());
            *out_size = arr->size();
        } else {
            throw scl::ValueError("Unsupported element size");
        }
    );
}

/// @brief Prefetch mapped array into page cache
int scl_mmap_array_prefetch(void* ptr, scl_size_t byte_size) {
    SCL_C_API_WRAPPER(
        if (!ptr) {
            throw scl::ValueError("Null pointer");
        }
        SCL_MMAP_ADVISE_WILLNEED(ptr, byte_size);
    );
}

/// @brief Drop mapped array from page cache
int scl_mmap_array_drop_cache(void* ptr, scl_size_t byte_size) {
    SCL_C_API_WRAPPER(
        if (!ptr) {
            throw scl::ValueError("Null pointer");
        }
        SCL_MMAP_ADVISE_DONTNEED(ptr, byte_size);
    );
}

/// @brief Advise sequential access pattern
int scl_mmap_array_advise_sequential(void* ptr, scl_size_t byte_size) {
    SCL_C_API_WRAPPER(
        if (!ptr) {
            throw scl::ValueError("Null pointer");
        }
        SCL_MMAP_ADVISE_SEQUENTIAL(ptr, byte_size);
    );
}

/// @brief Advise random access pattern
int scl_mmap_array_advise_random(void* ptr, scl_size_t byte_size) {
    SCL_C_API_WRAPPER(
        if (!ptr) {
            throw scl::ValueError("Null pointer");
        }
        SCL_MMAP_ADVISE_RANDOM(ptr, byte_size);
    );
}

// =============================================================================
// SECTION 2: HDF5 Sparse Matrix Loading
// =============================================================================

#ifdef SCL_HAS_HDF5

/// @brief Load sparse matrix from HDF5 file (CSR format)
int scl_h5_load_sparse_csr(
    const char* filepath,
    const char* group_path,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !group_path || !data_out || !indices_out || !indptr_out) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::io::h5::File file(filepath);
        scl::io::h5::Group group(file.id(), group_path);

        // Read dimensions from attributes
        auto shape = group.read_attr<scl::Index, 2>("shape");
        *out_rows = shape[0];
        *out_cols = shape[1];

        // Read datasets
        scl::io::h5::Dataset data_dset(group.id(), "data");
        scl::io::h5::Dataset indices_dset(group.id(), "indices");
        scl::io::h5::Dataset indptr_dset(group.id(), "indptr");

        *out_nnz = static_cast<scl::Index>(data_dset.get_size());

        data_dset.read(data_out);
        indices_dset.read(indices_out);
        indptr_dset.read(indptr_out);
    );
}

/// @brief Load sparse matrix column subset from HDF5 (basic implementation)
int scl_h5_load_sparse_csr_cols(
    const char* filepath,
    const char* group_path,
    const scl_index_t* col_indices,
    scl_index_t num_cols,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_nnz
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !group_path || !col_indices || !data_out || !indices_out || !indptr_out || !out_nnz) {
            throw scl::ValueError("Null pointer argument");
        }

        (void)num_cols;  // Unused for now

        // For now, load full matrix and filter in Python
        // TODO: Implement optimized column filtering using h5_tools query engine
        throw scl::NotImplementedError("Column filtering not yet implemented in C API");
    );
}

/// @brief Estimate NNZ for masked HDF5 load (placeholder)
int scl_h5_estimate_masked_nnz(
    const char* filepath,
    const char* group_path,
    const scl_byte_t* row_mask,
    const scl_byte_t* col_mask,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t* out_nnz
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !group_path || !out_nnz) {
            throw scl::ValueError("Null pointer argument");
        }

        (void)row_mask;  // Unused for now
        (void)col_mask;  // Unused for now
        (void)rows;      // Unused for now
        (void)cols;      // Unused for now

        // Placeholder: return full NNZ
        // TODO: Implement using zone maps
        scl::io::h5::File file(filepath);
        scl::io::h5::Group group(file.id(), group_path);
        scl::io::h5::Dataset data_dset(group.id(), "data");

        *out_nnz = static_cast<scl::Index>(data_dset.get_size());
    );
}

#endif // SCL_HAS_HDF5

// =============================================================================
// SECTION 3: File I/O Utilities
// =============================================================================

/// @brief Check if file exists
int scl_file_exists(const char* filepath, int* out_exists) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_exists) {
            throw scl::ValueError("Null pointer argument");
        }

        struct stat buffer;
        *out_exists = (stat(filepath, &buffer) == 0) ? 1 : 0;
    );
}

/// @brief Get file size in bytes
int scl_file_size(const char* filepath, scl_size_t* out_size) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_size) {
            throw scl::ValueError("Null pointer argument");
        }

        struct stat buffer;
        if (stat(filepath, &buffer) != 0) {
            throw scl::IOError("File not found or inaccessible");
        }

        *out_size = static_cast<scl_size_t>(buffer.st_size);
    );
}

/// @brief Create directory
int scl_create_directory(const char* dirpath) {
    SCL_C_API_WRAPPER(
        if (!dirpath) {
            throw scl::ValueError("Null pointer argument");
        }

        int result = mkdir(dirpath, 0755);

        if (result != 0 && errno != EEXIST) {
            throw scl::IOError("Failed to create directory");
        }
    );
}

/// @brief Write binary array to file
int scl_write_binary_array(
    const char* filepath,
    const void* data,
    scl_size_t element_size,
    scl_size_t num_elements
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !data) {
            throw scl::ValueError("Null pointer argument");
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw scl::IOError("Failed to open file for writing");
        }

        file.write(static_cast<const char*>(data), element_size * num_elements);

        if (!file) {
            throw scl::IOError("Failed to write data");
        }
    );
}

/// @brief Read binary array from file
int scl_read_binary_array(
    const char* filepath,
    void* data,
    scl_size_t element_size,
    scl_size_t num_elements
) {
    SCL_C_API_WRAPPER(
        if (!filepath || !data) {
            throw scl::ValueError("Null pointer argument");
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw scl::IOError("Failed to open file for reading");
        }

        file.read(static_cast<char*>(data), element_size * num_elements);

        if (!file) {
            throw scl::IOError("Failed to read data");
        }
    );
}

// =============================================================================
// SECTION 4: Path Utilities
// =============================================================================

/// @brief Get file extension
int scl_get_file_extension(const char* filepath, char* out_ext) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_ext) {
            throw scl::ValueError("Null pointer argument");
        }

        std::string path(filepath);
        size_t pos = path.find_last_of('.');
        std::string ext = (pos != std::string::npos) ? path.substr(pos + 1) : "";

        std::strncpy(out_ext, ext.c_str(), 63);
        out_ext[63] = '\0';
    );
}

/// @brief Get parent directory path
int scl_get_parent_directory(const char* filepath, char* out_dir) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_dir) {
            throw scl::ValueError("Null pointer argument");
        }

        std::string path(filepath);
        size_t pos = path.find_last_of("/\\");
        std::string dir = (pos != std::string::npos) ? path.substr(0, pos) : ".";

        std::strncpy(out_dir, dir.c_str(), 511);
        out_dir[511] = '\0';
    );
}

/// @brief Get filename without extension
int scl_get_filename_stem(const char* filepath, char* out_name) {
    SCL_C_API_WRAPPER(
        if (!filepath || !out_name) {
            throw scl::ValueError("Null pointer argument");
        }

        std::string path(filepath);

        // Get filename
        size_t pos = path.find_last_of("/\\");
        std::string filename = (pos != std::string::npos) ? path.substr(pos + 1) : path;

        // Remove extension
        pos = filename.find_last_of('.');
        std::string stem = (pos != std::string::npos) ? filename.substr(0, pos) : filename;

        std::strncpy(out_name, stem.c_str(), 255);
        out_name[255] = '\0';
    );
}

} // extern "C"
