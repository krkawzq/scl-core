#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"

#include <string>
#include <vector>
#include <array>
#include <optional>

#ifdef SCL_HAS_HDF5
#include <hdf5.h>

// =============================================================================
/// @file hdf5.hpp
/// @brief Lightweight HDF5 C API C++ Adapter
///
/// Pure thin wrapper over HDF5 C API with RAII resource management.
/// NO business logic, NO optimization algorithms - just type-safe API wrapping.
///
/// Design Principles:
///
/// 1. Thin Adapter: 1:1 mapping to HDF5 C API
/// 2. RAII Only: Automatic resource cleanup, nothing more
/// 3. Type Safety: Template type dispatch for read/write
/// 4. Zero Logic: No algorithms, no optimizations (those go in h5_tools.hpp)
/// 5. Exception Safety: Convert HDF5 errors to SCL exceptions
///
/// Coverage:
/// - File: H5F* operations
/// - Group: H5G* operations  
/// - Dataset: H5D* operations
/// - Dataspace: H5S* operations
/// - Attribute: H5A* operations
/// - Property: H5P* operations
///
/// Not Included (call C API directly if needed):
/// - Complex types (compound/vlen/opaque)
/// - Virtual datasets
/// - Parallel HDF5
/// - Advanced filters
// =============================================================================

namespace scl::io::h5 {

namespace detail {

/// @brief Map C++ types to HDF5 native types.
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
        throw TypeError("Unsupported HDF5 type");
    }
}

/// @brief Check HDF5 error and throw SCL exception.
inline void check_h5(herr_t err, const char* context) {
    if (err < 0) {
        throw IOError(std::string("HDF5: ") + context);
    }
}

} // namespace detail

// =============================================================================
// Handle - Generic RAII Wrapper
// =============================================================================

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
    
    hid_t release() noexcept {
        hid_t id = _id;
        _id = H5I_INVALID_HID;
        _closer = nullptr;
        return id;
    }
};

// =============================================================================
// File
// =============================================================================

class File {
private:
    Handle _file;
    File() = default;

public:
    explicit File(const std::string& path, unsigned flags = H5F_ACC_RDONLY)
        : _file(H5Fopen(path.c_str(), flags, H5P_DEFAULT), H5Fclose)
    {
        if (!_file.is_valid()) {
            throw IOError("Failed to open HDF5 file: " + path);
        }
    }
    
    static File create(const std::string& path, unsigned flags = H5F_ACC_TRUNC) {
        File f;
        f._file = Handle(H5Fcreate(path.c_str(), flags, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        if (!f._file.is_valid()) {
            throw IOError("Failed to create HDF5 file: " + path);
        }
        return f;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _file.id(); }
    
    void flush() const {
        detail::check_h5(H5Fflush(_file.id(), H5F_SCOPE_GLOBAL), "H5Fflush");
    }
};

// =============================================================================
// Group
// =============================================================================

class Group {
private:
    Handle _group;
    Group() = default;

public:
    Group(hid_t loc_id, const std::string& path)
        : _group(H5Gopen(loc_id, path.c_str(), H5P_DEFAULT), H5Gclose)
    {
        if (!_group.is_valid()) {
            throw IOError("Failed to open group: " + path);
        }
    }
    
    static Group create(hid_t loc_id, const std::string& path) {
        Group g;
        g._group = Handle(H5Gcreate(loc_id, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
        if (!g._group.is_valid()) {
            throw IOError("Failed to create group: " + path);
        }
        return g;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _group.id(); }
    
    SCL_NODISCARD bool has_attr(const std::string& name) const noexcept {
        return H5Aexists(_group.id(), name.c_str()) > 0;
    }
    
    template <typename T, size_t N>
    std::array<T, N> read_attr(const std::string& name) const {
        if (!has_attr(name)) {
            throw ValueError("Attribute not found: " + name);
        }
        
        Handle attr(H5Aopen(_group.id(), name.c_str(), H5P_DEFAULT), H5Aclose);
        std::array<T, N> result;
        detail::check_h5(
            H5Aread(attr.id(), detail::native_type<T>(), result.data()),
            "H5Aread"
        );
        return result;
    }
    
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
};

// =============================================================================
// Dataspace
// =============================================================================

class Dataspace {
private:
    Handle _space;
    Dataspace() = default;

public:
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
    
    explicit Dataspace(hid_t space_id) : _space(space_id, H5Sclose) {}
    explicit Dataspace(Handle&& h) : _space(std::move(h)) {}
    
    static Dataspace scalar() {
        Dataspace ds;
        ds._space = Handle(H5Screate(H5S_SCALAR), H5Sclose);
        return ds;
    }
    
    std::vector<hsize_t> get_dims() const {
        int rank = H5Sget_simple_extent_ndims(_space.id());
        std::vector<hsize_t> dims(rank);
        H5Sget_simple_extent_dims(_space.id(), dims.data(), nullptr);
        return dims;
    }
    
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
    
    SCL_NODISCARD hssize_t get_select_npoints() const {
        return H5Sget_select_npoints(_space.id());
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _space.id(); }
};

// =============================================================================
// Dataset
// =============================================================================

class Dataset {
private:
    Handle _dset;
    Dataset() = default;

public:
    Dataset(hid_t loc_id, const std::string& name)
        : _dset(H5Dopen(loc_id, name.c_str(), H5P_DEFAULT), H5Dclose)
    {
        if (!_dset.is_valid()) {
            throw IOError("Failed to open dataset: " + name);
        }
    }
    
    static Dataset create(
        hid_t loc_id,
        const std::string& name,
        hid_t type_id,
        const Dataspace& space,
        hid_t dcpl = H5P_DEFAULT
    ) {
        Dataset ds;
        ds._dset = Handle(
            H5Dcreate(loc_id, name.c_str(), type_id, space.id(),
                     H5P_DEFAULT, dcpl, H5P_DEFAULT),
            H5Dclose
        );
        if (!ds._dset.is_valid()) {
            throw IOError("Failed to create dataset: " + name);
        }
        return ds;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _dset.id(); }
    
    Dataspace get_space() const {
        hid_t space_id = H5Dget_space(_dset.id());
        if (space_id < 0) {
            throw IOError("H5Dget_space failed");
        }
        return Dataspace(Handle(space_id, H5Sclose));
    }
    
    std::vector<hsize_t> get_dims() const {
        return get_space().get_dims();
    }
    
    SCL_NODISCARD hsize_t get_size() const {
        auto dims = get_dims();
        hsize_t size = 1;
        for (auto d : dims) size *= d;
        return size;
    }
    
    SCL_NODISCARD bool is_chunked() const {
        Handle plist(H5Dget_create_plist(_dset.id()), H5Pclose);
        return H5Pget_layout(plist.id()) == H5D_CHUNKED;
    }
    
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
    
    template <typename T>
    void read(T* buffer) const {
        detail::check_h5(
            H5Dread(_dset.id(), detail::native_type<T>(), 
                   H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer),
            "H5Dread"
        );
    }
    
    template <typename T>
    void read(T* buffer, const Dataspace& mem_space, const Dataspace& file_space) const {
        detail::check_h5(
            H5Dread(_dset.id(), detail::native_type<T>(),
                   mem_space.id(), file_space.id(), H5P_DEFAULT, buffer),
            "H5Dread"
        );
    }
    
    template <typename T>
    void write(const T* buffer) {
        detail::check_h5(
            H5Dwrite(_dset.id(), detail::native_type<T>(),
                    H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer),
            "H5Dwrite"
        );
    }
    
    template <typename T>
    void write(const T* buffer, const Dataspace& mem_space, const Dataspace& file_space) {
        detail::check_h5(
            H5Dwrite(_dset.id(), detail::native_type<T>(),
                    mem_space.id(), file_space.id(), H5P_DEFAULT, buffer),
            "H5Dwrite"
        );
    }
};

// =============================================================================
// Property Lists
// =============================================================================

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
    
    DatasetCreateProps& chunked(const std::vector<hsize_t>& chunk_dims) {
        detail::check_h5(
            H5Pset_chunk(_plist.id(), static_cast<int>(chunk_dims.size()), chunk_dims.data()),
            "H5Pset_chunk"
        );
        return *this;
    }
    
    DatasetCreateProps& deflate(unsigned level = 6) {
        detail::check_h5(H5Pset_deflate(_plist.id(), level), "H5Pset_deflate");
        return *this;
    }
    
    DatasetCreateProps& shuffle() {
        detail::check_h5(H5Pset_shuffle(_plist.id()), "H5Pset_shuffle");
        return *this;
    }
    
    SCL_NODISCARD hid_t id() const noexcept { return _plist.id(); }
};

} // namespace scl::io::h5

#endif // SCL_HAS_HDF5
