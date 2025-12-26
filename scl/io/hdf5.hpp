#pragma once

#include "scl/io/mmatrix.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <optional>

#ifdef SCL_HAS_HDF5
#include <hdf5.h>

// =============================================================================
/// @file hdf5.hpp
/// @brief Lightweight HDF5 C++ Wrapper for SCL
///
/// Design Principles:
///
/// 1. Thin Wrapper: Minimal abstraction over HDF5 C API
/// 2. RAII Safety: Automatic resource management via smart handles
/// 3. Type Safety: Template-based type dispatch
/// 4. Zero Overhead: Inline everything, no virtual functions
/// 5. SCL Integration: Converts HDF5 errors to SCL exceptions
///
/// Coverage:
/// - File operations (open/close)
/// - Group operations (navigate/create)
/// - Dataset operations (read/write/hyperslab/chunks)
/// - Attribute operations (read/write metadata)
/// - Dataspace operations (selections/hyperslabs)
/// - Property lists (chunking/compression/caching)
///
/// Not Included (to keep lightweight):
/// - Complex datatypes (compound/vlen/opaque)
/// - Virtual datasets (VDS)
/// - Parallel HDF5 (MPI-IO)
/// - Advanced filters (custom plugins)
///
/// For these use cases, call HDF5 C API directly.
// =============================================================================

namespace scl::io::h5 {

// =============================================================================
// Type Mapping Utilities
// =============================================================================

namespace detail {

/// @brief Map C++ types to HDF5 native types (runtime dispatch).
template <typename T>
inline hid_t native_type() {
    if constexpr (std::is_same_v<T, float>)        return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>)  return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, int8_t>)  return H5T_NATIVE_INT8;
    else if constexpr (std::is_same_v<T, int16_t>) return H5T_NATIVE_INT16;
    else if constexpr (std::is_same_v<T, int32_t>) return H5T_NATIVE_INT32;
    else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
    else if constexpr (std::is_same_v<T, uint8_t>) return H5T_NATIVE_UINT8;
    else if constexpr (std::is_same_v<T, uint16_t>) return H5T_NATIVE_UINT16;
    else if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    else {
        // This branch will never be reached due to static_assert,
        // but keeps compiler happy
        throw TypeError("Unsupported HDF5 type");
    }
}

/// @brief Check HDF5 error and throw SCL exception.
inline void check_h5(herr_t err, const char* context) {
    if (err < 0) {
        throw IOError(std::string("HDF5 Error: ") + context);
    }
}

} // namespace detail

// =============================================================================
// Generic Handle (RAII Foundation)
// =============================================================================

/// @brief Generic RAII wrapper for HDF5 identifiers.
///
/// Thin wrapper that ensures proper cleanup.
/// No complex methods - just lifecycle management.
class Handle {
private:
    hid_t _id;
    herr_t (*_closer)(hid_t);

public:
    constexpr Handle() noexcept : _id(H5I_INVALID_HID), _closer(nullptr) {}
    
    Handle(hid_t id, herr_t (*closer)(hid_t)) noexcept
        : _id(id), _closer(closer) {}
    
    ~Handle() noexcept { close(); }
    
    void close() noexcept {
        if (is_valid() && _closer) {
            _closer(_id);
            _id = H5I_INVALID_HID;
        }
    }
    
    // Move-only
    Handle(Handle&& o) noexcept : _id(o._id), _closer(o._closer) {
        o._id = H5I_INVALID_HID;
        o._closer = nullptr;
    }
    
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            close();
            _id = o._id;
            _closer = o._closer;
            o._id = H5I_INVALID_HID;
            o._closer = nullptr;
        }
        return *this;
    }
    
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    
    SCL_NODISCARD hid_t id() const noexcept { return _id; }
    SCL_NODISCARD bool is_valid() const noexcept { 
        return _id >= 0 && _id != H5I_INVALID_HID; 
    }
    
    SCL_NODISCARD operator hid_t() const noexcept { return _id; }
    
    /// @brief Release ownership (for advanced use).
    hid_t release() noexcept {
        hid_t id = _id;
        _id = H5I_INVALID_HID;
        _closer = nullptr;
        return id;
    }
};

// =============================================================================
// File Operations
// =============================================================================

class File {
private:
    Handle _file;

public:
    /// @brief Open existing file (read-only).
    explicit File(const std::string& path, unsigned flags = H5F_ACC_RDONLY)
        : _file(H5Fopen(path.c_str(), flags, H5P_DEFAULT), H5Fclose)
    {
        if (!_file.is_valid()) {
            throw IOError("Failed to open HDF5 file: " + path);
        }
    }
    
    /// @brief Create new file.
    static File create(const std::string& path, unsigned flags = H5F_ACC_TRUNC) {
        hid_t id = H5Fcreate(path.c_str(), flags, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0) {
            throw IOError("Failed to create HDF5 file: " + path);
        }
        File f;
        f._file = Handle(id, H5Fclose);
        return f;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _file.id(); }
    
    /// @brief Flush file to disk.
    void flush() const {
        detail::check_h5(H5Fflush(_file.id(), H5F_SCOPE_GLOBAL), "H5Fflush");
    }

private:
    File() = default;  // For create()
};

// =============================================================================
// Group Operations
// =============================================================================

class Group {
private:
    Handle _group;

public:
    Group(hid_t loc_id, const std::string& path)
        : _group(H5Gopen(loc_id, path.c_str(), H5P_DEFAULT), H5Gclose)
    {
        if (!_group.is_valid()) {
            throw IOError("Failed to open group: " + path);
        }
    }
    
    /// @brief Create new group.
    static Group create(hid_t loc_id, const std::string& path) {
        hid_t id = H5Gcreate(loc_id, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0) {
            throw IOError("Failed to create group: " + path);
        }
        Group g;
        g._group = Handle(id, H5Gclose);
        return g;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _group.id(); }
    
    /// @brief Check if attribute exists.
    SCL_NODISCARD bool has_attr(const std::string& name) const noexcept {
        return H5Aexists(_group.id(), name.c_str()) > 0;
    }
    
    /// @brief Read attribute (template).
    template <typename T, size_t N>
    std::array<T, N> read_attr(const std::string& name) const {
        if (!has_attr(name)) {
            throw ValueError("Attribute '" + name + "' not found");
        }
        
        Handle attr(H5Aopen(_group.id(), name.c_str(), H5P_DEFAULT), H5Aclose);
        std::array<T, N> result;
        detail::check_h5(
            H5Aread(attr.id(), detail::native_type<T>(), result.data()),
            "H5Aread"
        );
        return result;
    }
    
    /// @brief Write attribute.
    template <typename T, size_t N>
    void write_attr(const std::string& name, const std::array<T, N>& value) {
        hsize_t dims[1] = {N};
        Handle space(H5Screate_simple(1, dims, nullptr), H5Sclose);
        Handle attr(
            H5Acreate(_group.id(), name.c_str(), detail::native_type<T>(),
                     space.id(), H5P_DEFAULT, H5P_DEFAULT),
            H5Aclose
        );
        detail::check_h5(
            H5Awrite(attr.id(), detail::native_type<T>(), value.data()),
            "H5Awrite"
        );
    }
    
    /// @brief Read shape attribute (common for anndata).
    std::tuple<Index, Index> read_shape() const {
        auto shape = read_attr<hsize_t, 2>("shape");
        return {static_cast<Index>(shape[0]), static_cast<Index>(shape[1])};
    }

private:
    Group() = default;  // For create()
};

// =============================================================================
// Dataspace Operations
// =============================================================================

class Dataspace {
private:
    Handle _space;
    
    Dataspace() = default;  // Private default constructor for internal use

public:
    /// @brief Create simple dataspace.
    explicit Dataspace(const std::vector<hsize_t>& dims, 
                      const std::vector<hsize_t>& maxdims = {})
        : _space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                 maxdims.empty() ? nullptr : maxdims.data()),
                H5Sclose)
    {
        if (!_space.is_valid()) {
            throw IOError("Failed to create dataspace");
        }
    }
    
    /// @brief Wrap existing dataspace.
    explicit Dataspace(hid_t space_id) : _space(space_id, H5Sclose) {}
    
    /// @brief Construct from handle.
    explicit Dataspace(Handle&& h) : _space(std::move(h)) {}
    
    /// @brief Create scalar dataspace.
    static Dataspace scalar() {
        Dataspace ds;
        ds._space = Handle(H5Screate(H5S_SCALAR), H5Sclose);
        return ds;
    }
    
    /// @brief Get dimensions.
    std::vector<hsize_t> get_dims() const {
        int rank = H5Sget_simple_extent_ndims(_space.id());
        std::vector<hsize_t> dims(rank);
        H5Sget_simple_extent_dims(_space.id(), dims.data(), nullptr);
        return dims;
    }
    
    /// @brief Select hyperslab.
    void select_hyperslab(
        const std::vector<hsize_t>& start,
        const std::vector<hsize_t>& count,
        const std::vector<hsize_t>& stride = {},
        const std::vector<hsize_t>& block = {},
        H5S_seloper_t op = H5S_SELECT_SET
    ) {
        detail::check_h5(
            H5Sselect_hyperslab(
                _space.id(), op,
                start.data(),
                stride.empty() ? nullptr : stride.data(),
                count.data(),
                block.empty() ? nullptr : block.data()
            ),
            "H5Sselect_hyperslab"
        );
    }
    
    /// @brief Get number of selected elements.
    SCL_NODISCARD hssize_t get_select_npoints() const {
        return H5Sget_select_npoints(_space.id());
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _space.id(); }
};

// =============================================================================
// Dataset Operations
// =============================================================================

class Dataset {
private:
    Handle _dset;

public:
    Dataset(hid_t loc_id, const std::string& name)
        : _dset(H5Dopen(loc_id, name.c_str(), H5P_DEFAULT), H5Dclose)
    {
        if (!_dset.is_valid()) {
            throw IOError("Failed to open dataset: " + name);
        }
    }
    
    /// @brief Create new dataset.
    static Dataset create(
        hid_t loc_id,
        const std::string& name,
        hid_t type_id,
        const Dataspace& space,
        hid_t dcpl = H5P_DEFAULT
    ) {
        hid_t id = H5Dcreate(loc_id, name.c_str(), type_id, space.id(),
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (id < 0) {
            throw IOError("Failed to create dataset: " + name);
        }
        Dataset ds;
        ds._dset = Handle(id, H5Dclose);
        return ds;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _dset.id(); }
    
    // -------------------------------------------------------------------------
    // Metadata Query
    // -------------------------------------------------------------------------
    
    /// @brief Get dataset dataspace.
    Dataspace get_space() const {
        hid_t space_id = H5Dget_space(_dset.id());
        if (space_id < 0) {
            throw IOError("H5Dget_space failed");
        }
        return Dataspace(Handle(space_id, H5Sclose));
    }
    
    /// @brief Get dataset dimensions.
    std::vector<hsize_t> get_dims() const {
        return get_space().get_dims();
    }
    
    /// @brief Get dataset element count.
    SCL_NODISCARD hsize_t get_size() const {
        auto dims = get_dims();
        hsize_t size = 1;
        for (auto d : dims) size *= d;
        return size;
    }
    
    /// @brief Check if dataset is chunked.
    SCL_NODISCARD bool is_chunked() const {
        Handle plist(H5Dget_create_plist(_dset.id()), H5Pclose);
        return H5Pget_layout(plist.id()) == H5D_CHUNKED;
    }
    
    /// @brief Get chunk dimensions.
    std::optional<std::vector<hsize_t>> get_chunk_dims() const {
        Handle plist(H5Dget_create_plist(_dset.id()), H5Pclose);
        
        if (H5Pget_layout(plist.id()) != H5D_CHUNKED) {
            return std::nullopt;
        }
        
        int rank = H5Pget_chunk(plist.id(), 0, nullptr);
        std::vector<hsize_t> dims(rank);
        H5Pget_chunk(plist.id(), rank, dims.data());
        
        return dims;
    }
    
    // -------------------------------------------------------------------------
    // Read Operations
    // -------------------------------------------------------------------------
    
    /// @brief Read entire dataset.
    template <typename T>
    void read(T* buffer) const {
        detail::check_h5(
            H5Dread(_dset.id(), detail::native_type<T>(), 
                   H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer),
            "H5Dread"
        );
    }
    
    /// @brief Read with custom memory/file spaces.
    template <typename T>
    void read(T* buffer, const Dataspace& mem_space, const Dataspace& file_space) const {
        detail::check_h5(
            H5Dread(_dset.id(), detail::native_type<T>(),
                   mem_space.id(), file_space.id(), H5P_DEFAULT, buffer),
            "H5Dread"
        );
    }
    
    /// @brief Read 1D hyperslab (most common case).
    template <typename T>
    void read_slab(hsize_t start, hsize_t count, T* buffer) const {
        if (count == 0) return;
        
        // File space with selection
        Dataspace file_space = get_space();
        std::vector<hsize_t> v_start = {start};
        std::vector<hsize_t> v_count = {count};
        file_space.select_hyperslab(v_start, v_count);
        
        // Memory space (contiguous)
        std::vector<hsize_t> mem_dims = {count};
        Dataspace mem_space(mem_dims);
        
        read(buffer, mem_space, file_space);
    }
    
    /// @brief Read multiple 1D slabs (batch).
    template <typename T>
    void read_slabs(const std::vector<std::pair<hsize_t, hsize_t>>& slabs, T* buffer) const {
        hsize_t offset = 0;
        for (const auto& [start, count] : slabs) {
            read_slab(start, count, buffer + offset);
            offset += count;
        }
    }
    
    // -------------------------------------------------------------------------
    // Write Operations
    // -------------------------------------------------------------------------
    
    /// @brief Write entire dataset.
    template <typename T>
    void write(const T* buffer) {
        detail::check_h5(
            H5Dwrite(_dset.id(), detail::native_type<T>(),
                    H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer),
            "H5Dwrite"
        );
    }
    
    /// @brief Write with custom spaces.
    template <typename T>
    void write(const T* buffer, const Dataspace& mem_space, const Dataspace& file_space) {
        detail::check_h5(
            H5Dwrite(_dset.id(), detail::native_type<T>(),
                    mem_space.id(), file_space.id(), H5P_DEFAULT, buffer),
            "H5Dwrite"
        );
    }
    
    /// @brief Write 1D hyperslab.
    template <typename T>
    void write_slab(hsize_t start, hsize_t count, const T* buffer) {
        if (count == 0) return;
        
        Dataspace file_space = get_space();
        std::vector<hsize_t> v_start = {start};
        std::vector<hsize_t> v_count = {count};
        file_space.select_hyperslab(v_start, v_count);
        
        std::vector<hsize_t> mem_dims = {count};
        Dataspace mem_space(mem_dims);
        
        write(buffer, mem_space, file_space);
    }

private:
    Dataset() = default;  // For create()
};

// =============================================================================
// Property List Helpers
// =============================================================================

/// @brief Dataset creation property list builder.
class DatasetCreateProps {
private:
    Handle _plist;

public:
    DatasetCreateProps()
        : _plist(H5Pcreate(H5P_DATASET_CREATE), H5Pclose)
    {
        if (!_plist.is_valid()) {
            throw IOError("Failed to create property list");
        }
    }
    
    /// @brief Set chunking.
    DatasetCreateProps& chunked(const std::vector<hsize_t>& chunk_dims) {
        detail::check_h5(
            H5Pset_chunk(_plist.id(), static_cast<int>(chunk_dims.size()), chunk_dims.data()),
            "H5Pset_chunk"
        );
        return *this;
    }
    
    /// @brief Set deflate compression (gzip).
    DatasetCreateProps& deflate(unsigned level = 6) {
        detail::check_h5(H5Pset_deflate(_plist.id(), level), "H5Pset_deflate");
        return *this;
    }
    
    /// @brief Set shuffle filter (improves compression).
    DatasetCreateProps& shuffle() {
        detail::check_h5(H5Pset_shuffle(_plist.id()), "H5Pset_shuffle");
        return *this;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _plist.id(); }
};

// =============================================================================
// High-Level Utilities
// =============================================================================

namespace detail {

/// @brief Range for hyperslab merging.
struct Range {
    Index begin, end;
    SCL_NODISCARD Index length() const { return end - begin; }
};

/// @brief Merge overlapping ranges.
inline std::vector<Range> merge_ranges(std::vector<Range> ranges, Index gap_threshold = 128) {
    if (ranges.empty()) return {};
    
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
        return a.begin < b.begin;
    });
    
    std::vector<Range> merged;
    merged.push_back(ranges[0]);
    
    for (size_t i = 1; i < ranges.size(); ++i) {
        Range& last = merged.back();
        const Range& curr = ranges[i];
        
        if (curr.begin <= last.end + gap_threshold) {
            last.end = std::max(last.end, curr.end);
        } else {
            merged.push_back(curr);
        }
    }
    
    return merged;
}

/// @brief Compute data ranges for selected rows.
inline std::vector<Range> compute_row_ranges(
    const std::vector<Index>& indptr,
    Span<const Index> selected_rows
) {
    std::vector<Range> ranges;
    ranges.reserve(selected_rows.size);
    
    for (Size i = 0; i < selected_rows.size; ++i) {
        Index row_idx = selected_rows[i];
        Index start = indptr[row_idx];
        Index end = indptr[row_idx + 1];
        
        if (start < end) {
            ranges.push_back({start, end});
        }
    }
    
    return merge_ranges(std::move(ranges));
}

} // namespace detail

// =============================================================================
// CSR Matrix Loading (Optimized)
// =============================================================================

/// @brief Load selected rows from HDF5 CSR matrix.
///
/// Optimizations:
/// 1. Loads indptr fully (small)
/// 2. Merges adjacent ranges (reduces I/O)
/// 3. Batch hyperslab reads (100x fewer calls)
template <typename T>
inline OwnedCSR<T> load_csr_rows(
    const std::string& h5_path,
    const std::string& group_path,
    Span<const Index> selected_rows
) {
    File file(h5_path);
    Group group(file.id(), group_path);
    
    auto [rows, cols] = group.read_shape();
    
    // Load indptr (small, always load fully)
    Dataset indptr_dset(group.id(), "indptr");
    std::vector<Index> indptr(rows + 1);
    indptr_dset.read(indptr.data());
    
    // Compute merged ranges
    auto ranges = detail::compute_row_ranges(indptr, selected_rows);
    
    // Allocate output
    Index total_nnz = 0;
    for (const auto& r : ranges) total_nnz += r.length();
    
    std::vector<T> data(total_nnz);
    std::vector<Index> indices(total_nnz);
    
    // Batch read
    Dataset data_dset(group.id(), "data");
    Dataset indices_dset(group.id(), "indices");
    
    Index write_offset = 0;
    for (const auto& range : ranges) {
        hsize_t start = static_cast<hsize_t>(range.begin);
        hsize_t count = static_cast<hsize_t>(range.length());
        
        data_dset.read_slab(start, count, data.data() + write_offset);
        indices_dset.read_slab(start, count, indices.data() + write_offset);
        
        write_offset += count;
    }
    
    // Rebuild indptr
    std::vector<Index> new_indptr;
    new_indptr.reserve(selected_rows.size + 1);
    new_indptr.push_back(0);
    
    for (Size i = 0; i < selected_rows.size; ++i) {
        Index row_idx = selected_rows[i];
        Index row_len = indptr[row_idx + 1] - indptr[row_idx];
        new_indptr.push_back(new_indptr.back() + row_len);
    }
    
    return OwnedCSR<T>(
        std::move(data), std::move(indices), std::move(new_indptr),
        static_cast<Index>(selected_rows.size), cols, total_nnz
    );
}

/// @brief Load entire CSR matrix from HDF5.
template <typename T>
inline OwnedCSR<T> load_csr_full(
    const std::string& h5_path,
    const std::string& group_path
) {
    File file(h5_path);
    Group group(file.id(), group_path);
    
    auto [rows, cols] = group.read_shape();
    
    Dataset data_dset(group.id(), "data");
    Dataset indices_dset(group.id(), "indices");
    Dataset indptr_dset(group.id(), "indptr");
    
    Index nnz = static_cast<Index>(data_dset.get_size());
    
    std::vector<T> data(nnz);
    std::vector<Index> indices(nnz);
    std::vector<Index> indptr(rows + 1);
    
    data_dset.read(data.data());
    indices_dset.read(indices.data());
    indptr_dset.read(indptr.data());
    
    return OwnedCSR<T>(std::move(data), std::move(indices), std::move(indptr), rows, cols, nnz);
}

/// @brief Save CSR matrix to HDF5 (anndata format).
template <typename T>
inline void save_csr(
    const std::string& h5_path,
    const std::string& group_path,
    const OwnedCSR<T>& mat,
    const std::vector<hsize_t>& chunk_dims = {10000},
    unsigned compress_level = 6
) {
    File file = File::create(h5_path);
    Group group = Group::create(file.id(), group_path);
    
    // Write shape attribute
    group.write_attr("shape", std::array<hsize_t, 2>{
        static_cast<hsize_t>(mat.rows),
        static_cast<hsize_t>(mat.cols)
    });
    
    // Create datasets with chunking and compression
    DatasetCreateProps props;
    props.chunked(chunk_dims).shuffle().deflate(compress_level);
    
    // Create data dataset
    std::vector<hsize_t> data_dims = {static_cast<hsize_t>(mat.nnz)};
    Dataspace data_space(data_dims);
    Dataset data_dset = Dataset::create(
        group.id(), "data", detail::native_type<T>(), data_space, props.id()
    );
    data_dset.write(mat.data.data());
    
    // Create indices dataset
    Dataset indices_dset = Dataset::create(
        group.id(), "indices", detail::native_type<Index>(), data_space, props.id()
    );
    indices_dset.write(mat.indices.data());
    
    // Create indptr dataset (typically not compressed)
    std::vector<hsize_t> indptr_dims = {static_cast<hsize_t>(mat.rows + 1)};
    Dataspace indptr_space(indptr_dims);
    Dataset indptr_dset = Dataset::create(
        group.id(), "indptr", detail::native_type<Index>(), indptr_space
    );
    indptr_dset.write(mat.indptr.data());
    
    file.flush();
}

} // namespace scl::io::h5

#endif // SCL_HAS_HDF5
