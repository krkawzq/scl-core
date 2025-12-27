#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"

#include <string>
#include <vector>
#include <array>
#include <optional>
#include <functional>
#include <memory>
#include <variant>
#include <exception>

#ifdef SCL_HAS_HDF5
#include <hdf5.h>

// =============================================================================
// FILE: scl/io/hdf5.hpp
// BRIEF: Modern C++ Object-Oriented HDF5 Wrapper
// =============================================================================

namespace scl::io::h5 {

// Forward declarations
class Object;
class Location;
class File;
class Group;
class Dataset;
class Datatype;
class Attribute;
class Dataspace;
class PropertyList;

namespace detail {

// Map C++ types to HDF5 native types
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
    else if constexpr (std::is_same_v<T, bool>) {
        static_assert(sizeof(bool) == sizeof(hbool_t), "bool size mismatch with hbool_t");
        return H5T_NATIVE_HBOOL;
    }
    else {
        throw TypeError("Unsupported HDF5 type");
    }
}

inline void check_h5(herr_t err, const char* context) {
    if (err < 0) {
        std::string msg = std::string("HDF5: ") + context;
        
        // Get HDF5 error stack
        struct ErrorWalker {
            std::string* msg;
        } walker{&msg};
        
        herr_t walk_err = H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD,
            [](unsigned, const H5E_error2_t* err, void* data) -> herr_t {
                auto* w = static_cast<ErrorWalker*>(data);
                if (err->desc) {
                    *w->msg += "\n  " + std::string(err->desc);
                }
                return 0;
            }, &walker);
        
        // If error walk failed, at least we have the context message
        if (walk_err < 0) {
            msg += " (failed to retrieve error details)";
        }
        
        throw IOError(msg);
    }
}

inline void check_id(hid_t id, const char* context) {
    if (id < 0) {
        throw IOError(std::string("HDF5 invalid ID: ") + context);
    }
}

} // namespace detail

// =============================================================================
// Enums and Types
// =============================================================================

enum class ObjectType {
    Unknown,
    File,
    Group,
    Dataset,
    Datatype,
    Attribute,
    Dataspace
};

enum class LinkType {
    Hard,
    Soft,
    External,
    Error
};

struct ObjectInfo {
    ObjectType type;
    size_t ref_count;
    hsize_t num_attrs;
    time_t atime;
    time_t mtime;
    time_t ctime;
    time_t btime;
};

struct LinkInfo {
    LinkType type;
    bool corder_valid;
    int64_t corder;
    H5T_cset_t cset;
    size_t value_size;
};

// =============================================================================
// Object - Base Class for All HDF5 Objects
// =============================================================================

class Object {
protected:
    hid_t _id;
    herr_t (*_closer)(hid_t);

    explicit Object(hid_t id, herr_t (*closer)(hid_t)) noexcept
        : _id(id), _closer(closer) {}

    Object() noexcept : _id(H5I_INVALID_HID), _closer(nullptr) {}

public:
    virtual ~Object() noexcept { close(); }

    void close() noexcept {
        if (is_valid() && _closer) {
            _closer(_id);
            _id = H5I_INVALID_HID;
        }
    }

    Object(Object&& other) noexcept
        : _id(other._id), _closer(other._closer)
    {
        other._id = H5I_INVALID_HID;
        other._closer = nullptr;
    }

    Object& operator=(Object&& other) noexcept {
        if (this != &other) {
            close();
            _id = other._id;
            _closer = other._closer;
            other._id = H5I_INVALID_HID;
            other._closer = nullptr;
        }
        return *this;
    }

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

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

    SCL_NODISCARD int get_ref_count() const {
        if (!is_valid()) return 0;
        return H5Iget_ref(_id);
    }

    void inc_ref() {
        if (is_valid()) {
            detail::check_h5(H5Iinc_ref(_id), "H5Iinc_ref");
        }
    }

    void dec_ref() {
        if (is_valid()) {
            detail::check_h5(H5Idec_ref(_id), "H5Idec_ref");
        }
    }

    SCL_NODISCARD H5I_type_t get_type() const {
        if (!is_valid()) return H5I_BADID;
        return H5Iget_type(_id);
    }
};

// =============================================================================
// Dataspace
// =============================================================================

class Dataspace : public Object {
public:
    explicit Dataspace(const std::vector<hsize_t>& dims,
                      const std::vector<hsize_t>& maxdims = {})
        : Object(H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                 maxdims.empty() ? nullptr : maxdims.data()),
                H5Sclose)
    {
        detail::check_id(_id, "H5Screate_simple");
    }

    explicit Dataspace(hid_t space_id)
        : Object(space_id, H5Sclose) {}

    static Dataspace scalar() {
        hid_t id = H5Screate(H5S_SCALAR);
        detail::check_id(id, "H5Screate(SCALAR)");
        return Dataspace(id);
    }

    static Dataspace null() {
        hid_t id = H5Screate(H5S_NULL);
        detail::check_id(id, "H5Screate(NULL)");
        return Dataspace(id);
    }

    SCL_NODISCARD int get_rank() const {
        return H5Sget_simple_extent_ndims(_id);
    }

    std::vector<hsize_t> get_dims() const {
        int rank = get_rank();
        if (rank < 0) return {};
        std::vector<hsize_t> dims(rank);
        H5Sget_simple_extent_dims(_id, dims.data(), nullptr);
        return dims;
    }

    std::vector<hsize_t> get_max_dims() const {
        int rank = get_rank();
        if (rank < 0) return {};
        std::vector<hsize_t> maxdims(rank);
        H5Sget_simple_extent_dims(_id, nullptr, maxdims.data());
        return maxdims;
    }

    SCL_NODISCARD hssize_t get_num_elements() const {
        return H5Sget_simple_extent_npoints(_id);
    }

    SCL_NODISCARD bool is_simple() const {
        return H5Sis_simple(_id) > 0;
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
                _id, op,
                start.data(),
                stride.empty() ? nullptr : stride.data(),
                count.data(),
                block.empty() ? nullptr : block.data()
            ),
            "H5Sselect_hyperslab"
        );
    }

    void select_all() {
        detail::check_h5(H5Sselect_all(_id), "H5Sselect_all");
    }

    void select_none() {
        detail::check_h5(H5Sselect_none(_id), "H5Sselect_none");
    }

    SCL_NODISCARD hssize_t get_select_npoints() const {
        return H5Sget_select_npoints(_id);
    }

    SCL_NODISCARD bool is_select_valid() const {
        return H5Sselect_valid(_id) > 0;
    }

protected:
    Dataspace() = default;
};

// =============================================================================
// Datatype
// =============================================================================

class Datatype : public Object {
public:
    explicit Datatype(hid_t type_id)
        : Object(H5Tcopy(type_id), H5Tclose)
    {
        detail::check_id(_id, "H5Tcopy");
    }

    template <typename T>
    static Datatype native() {
        return Datatype(detail::native_type<T>());
    }

    static Datatype string_vlen() {
        hid_t id = H5Tcopy(H5T_C_S1);
        detail::check_id(id, "H5Tcopy(H5T_C_S1)");
        H5Tset_size(id, H5T_VARIABLE);
        return from_id(id);
    }

    static Datatype string_fixed(size_t len) {
        hid_t id = H5Tcopy(H5T_C_S1);
        detail::check_id(id, "H5Tcopy(H5T_C_S1)");
        H5Tset_size(id, len);
        return from_id(id);
    }
    
    static Datatype from_id(hid_t type_id) {
        return Datatype(type_id, no_copy_tag{});
    }

    SCL_NODISCARD size_t get_size() const {
        return H5Tget_size(_id);
    }

    SCL_NODISCARD H5T_class_t get_class() const {
        return H5Tget_class(_id);
    }

    SCL_NODISCARD bool equals(const Datatype& other) const {
        return H5Tequal(_id, other._id) > 0;
    }

protected:
    Datatype() = default;
    
    struct no_copy_tag {};
    explicit Datatype(hid_t type_id, no_copy_tag)
        : Object(type_id, H5Tclose) {}
};

// =============================================================================
// Attribute
// =============================================================================

class Attribute : public Object {
public:
    Attribute(hid_t loc_id, const std::string& name)
        : Object(H5Aopen(loc_id, name.c_str(), H5P_DEFAULT), H5Aclose)
    {
        detail::check_id(_id, ("H5Aopen: " + name).c_str());
    }

    static Attribute create(
        hid_t loc_id,
        const std::string& name,
        hid_t type_id,
        const Dataspace& space
    ) {
        hid_t id = H5Acreate(loc_id, name.c_str(), type_id, space.id(),
                            H5P_DEFAULT, H5P_DEFAULT);
        detail::check_id(id, ("H5Acreate: " + name).c_str());
        return Attribute(id, false);
    }

    Dataspace get_space() const {
        hid_t space_id = H5Aget_space(_id);
        detail::check_id(space_id, "H5Aget_space");
        return Dataspace(space_id);
    }

    Datatype get_type() const {
        hid_t type_id = H5Aget_type(_id);
        detail::check_id(type_id, "H5Aget_type");
        return Datatype::from_id(type_id);
    }

    std::string get_name() const {
        ssize_t size = H5Aget_name(_id, 0, nullptr);
        if (size < 0) return "";
        std::string name(size, '\0');
        H5Aget_name(_id, size + 1, name.data());
        return name;
    }

    template <typename T>
    void read(T* buffer) const {
        detail::check_h5(
            H5Aread(_id, detail::native_type<T>(), buffer),
            "H5Aread"
        );
    }

    template <typename T>
    void write(const T* buffer) {
        detail::check_h5(
            H5Awrite(_id, detail::native_type<T>(), buffer),
            "H5Awrite"
        );
    }

    template <typename T>
    T read_scalar() const {
        T value;
        read(&value);
        return value;
    }

    template <typename T>
    void write_scalar(const T& value) {
        write(&value);
    }

    std::string read_string() const {
        Datatype dtype = get_type();
        
        if (H5Tis_variable_str(dtype.id()) > 0) {
            // Variable-length string
            char* str_ptr = nullptr;
            hid_t mem_type = H5Tcopy(H5T_C_S1);
            detail::check_id(mem_type, "H5Tcopy(H5T_C_S1)");
            H5Tset_size(mem_type, H5T_VARIABLE);
            
            try {
                detail::check_h5(H5Aread(_id, mem_type, &str_ptr), "H5Aread vlen string");
                
                std::string result(str_ptr ? str_ptr : "");
                
                // Release HDF5-allocated memory
                if (str_ptr) {
                    Dataspace space = get_space();
                    H5Dvlen_reclaim(mem_type, space.id(), H5P_DEFAULT, &str_ptr);
                }
                H5Tclose(mem_type);
                
                return result;
            } catch (...) {
                // Ensure mem_type is closed even if exception occurs
                H5Tclose(mem_type);
                throw;
            }
        } else {
            // Fixed-length string
            size_t size = dtype.get_size();
            std::string value(size, '\0');
            detail::check_h5(
                H5Aread(_id, dtype.id(), value.data()),
                "H5Aread string"
            );
            
            // Remove trailing null characters
            // Note: Fixed-length strings may contain trailing nulls as padding
            auto null_pos = value.find('\0');
            if (null_pos != std::string::npos) {
                value.resize(null_pos);
            }
            return value;
        }
    }

    void write_string(const std::string& value) {
        Datatype dtype = Datatype::string_fixed(value.size() + 1);  // +1 for null terminator
        detail::check_h5(
            H5Awrite(_id, dtype.id(), value.c_str()),
            "H5Awrite string"
        );
    }

private:
    Attribute(hid_t id, bool) : Object(id, H5Aclose) {}
};

// =============================================================================
// PropertyList - Base Class for Property Lists
// =============================================================================

class PropertyList : public Object {
protected:
    explicit PropertyList(hid_t cls_id)
        : Object(H5Pcreate(cls_id), H5Pclose)
    {
        detail::check_id(_id, "H5Pcreate");
    }

    PropertyList() = default;

    PropertyList(hid_t id, herr_t (*closer)(hid_t))
        : Object(id, closer) {}

public:
    PropertyList copy() const {
        hid_t new_id = H5Pcopy(_id);
        detail::check_id(new_id, "H5Pcopy");
        return PropertyList(new_id, H5Pclose);
    }
};

class FileAccessProps : public PropertyList {
public:
    FileAccessProps() : PropertyList(H5P_FILE_ACCESS) {}

protected:
    FileAccessProps(hid_t id, herr_t (*closer)(hid_t))
        : PropertyList(id, closer) {}
    friend class File;

public:
    FileAccessProps copy() const {
        hid_t new_id = H5Pcopy(_id);
        detail::check_id(new_id, "H5Pcopy");
        return FileAccessProps(new_id, H5Pclose);
    }
    
    FileAccessProps& cache(size_t nslots, size_t nbytes, double preemption) {
        detail::check_h5(
            H5Pset_cache(_id, 0, nslots, nbytes, preemption),
            "H5Pset_cache"
        );
        return *this;
    }

    FileAccessProps& alignment(hsize_t threshold, hsize_t alignment) {
        detail::check_h5(
            H5Pset_alignment(_id, threshold, alignment),
            "H5Pset_alignment"
        );
        return *this;
    }
};

class FileCreateProps : public PropertyList {
public:
    FileCreateProps() : PropertyList(H5P_FILE_CREATE) {}

protected:
    FileCreateProps(hid_t id, herr_t (*closer)(hid_t))
        : PropertyList(id, closer) {}
    friend class File;

public:
    FileCreateProps copy() const {
        hid_t new_id = H5Pcopy(_id);
        detail::check_id(new_id, "H5Pcopy");
        return FileCreateProps(new_id, H5Pclose);
    }
};

class DatasetAccessProps : public PropertyList {
public:
    DatasetAccessProps() : PropertyList(H5P_DATASET_ACCESS) {}

protected:
    DatasetAccessProps(hid_t id, herr_t (*closer)(hid_t))
        : PropertyList(id, closer) {}
    friend class Dataset;

public:
    DatasetAccessProps copy() const {
        hid_t new_id = H5Pcopy(_id);
        detail::check_id(new_id, "H5Pcopy");
        return DatasetAccessProps(new_id, H5Pclose);
    }
    
    DatasetAccessProps& chunk_cache(size_t nslots, size_t nbytes, double preemption) {
        detail::check_h5(
            H5Pset_chunk_cache(_id, nslots, nbytes, preemption),
            "H5Pset_chunk_cache"
        );
        return *this;
    }
};

class DatasetCreateProps : public PropertyList {
public:
    DatasetCreateProps() : PropertyList(H5P_DATASET_CREATE) {}

    DatasetCreateProps& chunked(const std::vector<hsize_t>& chunk_dims) {
        detail::check_h5(
            H5Pset_chunk(_id, static_cast<int>(chunk_dims.size()), chunk_dims.data()),
            "H5Pset_chunk"
        );
        return *this;
    }

    DatasetCreateProps& deflate(unsigned level = 6) {
        detail::check_h5(H5Pset_deflate(_id, level), "H5Pset_deflate");
        return *this;
    }

    DatasetCreateProps& shuffle() {
        detail::check_h5(H5Pset_shuffle(_id), "H5Pset_shuffle");
        return *this;
    }

    template <typename T>
    DatasetCreateProps& fill_value(const T& value) {
        detail::check_h5(
            H5Pset_fill_value(_id, detail::native_type<T>(), &value),
            "H5Pset_fill_value"
        );
        return *this;
    }

    DatasetCreateProps& alloc_time(H5D_alloc_time_t alloc_time) {
        detail::check_h5(
            H5Pset_alloc_time(_id, alloc_time),
            "H5Pset_alloc_time"
        );
        return *this;
    }

protected:
    DatasetCreateProps(hid_t id, herr_t (*closer)(hid_t))
        : PropertyList(id, closer) {}
    friend class Dataset;

public:
    DatasetCreateProps copy() const {
        hid_t new_id = H5Pcopy(_id);
        detail::check_id(new_id, "H5Pcopy");
        return DatasetCreateProps(new_id, H5Pclose);
    }
};

class DatasetTransferProps : public PropertyList {
public:
    DatasetTransferProps() : PropertyList(H5P_DATASET_XFER) {}
};

// =============================================================================
// Location - Base Class for File and Group
// =============================================================================

class Location : public Object {
protected:
    using Object::Object;
    Location() = default;

public:
    // Existence and Type Queries

    SCL_NODISCARD bool exists(const std::string& name) const {
        return H5Lexists(_id, name.c_str(), H5P_DEFAULT) > 0;
    }

    SCL_NODISCARD bool has_attr(const std::string& name) const {
        return H5Aexists(_id, name.c_str()) > 0;
    }

    SCL_NODISCARD ObjectType get_object_type(const std::string& name) const {
        H5O_info_t info;
        if (H5Oget_info_by_name(_id, name.c_str(), &info, H5P_DEFAULT) < 0) {
            return ObjectType::Unknown;
        }

        switch (info.type) {
            case H5O_TYPE_GROUP: return ObjectType::Group;
            case H5O_TYPE_DATASET: return ObjectType::Dataset;
            case H5O_TYPE_NAMED_DATATYPE: return ObjectType::Datatype;
            default: return ObjectType::Unknown;
        }
    }

    SCL_NODISCARD LinkType get_link_type(const std::string& name) const {
        H5L_info_t info;
        if (H5Lget_info(_id, name.c_str(), &info, H5P_DEFAULT) < 0) {
            return LinkType::Error;
        }

        switch (info.type) {
            case H5L_TYPE_HARD: return LinkType::Hard;
            case H5L_TYPE_SOFT: return LinkType::Soft;
            case H5L_TYPE_EXTERNAL: return LinkType::External;
            default: return LinkType::Error;
        }
    }

    // Object Information

    ObjectInfo get_info(const std::string& name = ".") const {
        H5O_info_t h5info;
        detail::check_h5(
            H5Oget_info_by_name(_id, name.c_str(), &h5info, H5P_DEFAULT),
            "H5Oget_info_by_name"
        );

        ObjectInfo info{};
        switch (h5info.type) {
            case H5O_TYPE_GROUP: info.type = ObjectType::Group; break;
            case H5O_TYPE_DATASET: info.type = ObjectType::Dataset; break;
            case H5O_TYPE_NAMED_DATATYPE: info.type = ObjectType::Datatype; break;
            default: info.type = ObjectType::Unknown;
        }

        info.ref_count = h5info.rc;
        info.num_attrs = h5info.num_attrs;
        info.atime = h5info.atime;
        info.mtime = h5info.mtime;
        info.ctime = h5info.ctime;
        info.btime = h5info.btime;

        return info;
    }

    SCL_NODISCARD hsize_t get_num_objs() const {
        H5G_info_t info;
        detail::check_h5(
            H5Gget_info(_id, &info),
            "H5Gget_info"
        );
        return info.nlinks;
    }

    std::string get_objname_by_idx(hsize_t idx) const {
        ssize_t size = H5Lget_name_by_idx(_id, ".", H5_INDEX_NAME, H5_ITER_INC, idx, nullptr, 0, H5P_DEFAULT);
        if (size < 0) return "";
        std::string name(size, '\0');
        H5Lget_name_by_idx(_id, ".", H5_INDEX_NAME, H5_ITER_INC, idx, name.data(), size + 1, H5P_DEFAULT);
        return name;
    }

    std::vector<std::string> list_objects() const {
        hsize_t num = get_num_objs();
        std::vector<std::string> names;
        names.reserve(num);
        for (hsize_t i = 0; i < num; ++i) {
            names.push_back(get_objname_by_idx(i));
        }
        return names;
    }

    // Iteration

    void iterate(const std::function<bool(const std::string&, ObjectType)>& func) const {
        struct Context {
            const std::function<bool(const std::string&, ObjectType)>* func;
            std::exception_ptr exception;
        };

        Context ctx{&func, nullptr};

        auto callback = [](hid_t, const char* name, const H5L_info_t*, void* op_data) -> herr_t {
            auto* context = static_cast<Context*>(op_data);
            try {
                bool cont = (*context->func)(name, ObjectType::Unknown);
                return cont ? 0 : 1;
            } catch (...) {
                context->exception = std::current_exception();
                return -1;  // Stop iteration
            }
        };

        H5Literate(_id, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, callback, &ctx);

        if (ctx.exception) {
            std::rethrow_exception(ctx.exception);
        }
    }

    // Link Operations

    void unlink(const std::string& name) {
        detail::check_h5(
            H5Ldelete(_id, name.c_str(), H5P_DEFAULT),
            "H5Ldelete"
        );
    }

    void link_soft(const std::string& target, const std::string& link_name) {
        detail::check_h5(
            H5Lcreate_soft(target.c_str(), _id, link_name.c_str(), H5P_DEFAULT, H5P_DEFAULT),
            "H5Lcreate_soft"
        );
    }

    void link_hard(const std::string& target, const std::string& link_name) {
        detail::check_h5(
            H5Lcreate_hard(_id, target.c_str(), _id, link_name.c_str(), H5P_DEFAULT, H5P_DEFAULT),
            "H5Lcreate_hard"
        );
    }

    void move_link(const std::string& src, const std::string& dst) {
        detail::check_h5(
            H5Lmove(_id, src.c_str(), _id, dst.c_str(), H5P_DEFAULT, H5P_DEFAULT),
            "H5Lmove"
        );
    }

    void copy_link(const std::string& src, const std::string& dst) {
        detail::check_h5(
            H5Lcopy(_id, src.c_str(), _id, dst.c_str(), H5P_DEFAULT, H5P_DEFAULT),
            "H5Lcopy"
        );
    }

    // Attribute Operations

    SCL_NODISCARD int get_num_attrs() const {
        H5O_info_t info;
        detail::check_h5(
            H5Oget_info(_id, &info),
            "H5Oget_info"
        );
        return static_cast<int>(info.num_attrs);
    }

    Attribute open_attr(const std::string& name) const {
        return Attribute(_id, name);
    }

    template <typename T>
    Attribute create_attr(const std::string& name, const std::vector<hsize_t>& dims = {}) {
        Dataspace space = dims.empty() ? Dataspace::scalar() : Dataspace(dims);
        return Attribute::create(_id, name, detail::native_type<T>(), space);
    }

    void delete_attr(const std::string& name) {
        detail::check_h5(
            H5Adelete(_id, name.c_str()),
            "H5Adelete"
        );
    }

    template <typename T>
    T read_attr(const std::string& name) const {
        Attribute attr = open_attr(name);
        return attr.read_scalar<T>();
    }

    template <typename T>
    void write_attr(const std::string& name, const T& value) {
        if (!has_attr(name)) {
            create_attr<T>(name).write_scalar(value);
        } else {
            open_attr(name).write_scalar(value);
        }
    }

    template <typename T>
    std::vector<T> read_attr_array(const std::string& name) const {
        Attribute attr = open_attr(name);
        Dataspace space = attr.get_space();
        hssize_t npoints = space.get_num_elements();
        std::vector<T> data(npoints);
        attr.read(data.data());
        return data;
    }

    template <typename T>
    void write_attr_array(const std::string& name, const std::vector<T>& data) {
        if (has_attr(name)) {
            delete_attr(name);  // Delete old attribute to avoid dimension mismatch
        }
        
        std::vector<hsize_t> dims = {data.size()};
        Attribute attr = create_attr<T>(name, dims);
        attr.write(data.data());
    }

    std::string read_attr_string(const std::string& name) const {
        return open_attr(name).read_string();
    }

    void write_attr_string(const std::string& name, const std::string& value) {
        if (has_attr(name)) {
            delete_attr(name);  // Delete old attribute to avoid size mismatch
        }
        
        Dataspace space = Dataspace::scalar();
        Attribute attr = Attribute::create(_id, name,
            Datatype::string_fixed(value.size() + 1).id(), space);  // +1 for null terminator
        attr.write_string(value);
    }
};

// =============================================================================
// Dataset
// =============================================================================

class Dataset : public Object {
public:
    Dataset(hid_t loc_id, const std::string& name,
            hid_t dapl = H5P_DEFAULT)
        : Object(H5Dopen(loc_id, name.c_str(), dapl), H5Dclose)
    {
        detail::check_id(_id, ("H5Dopen: " + name).c_str());
    }

    static Dataset create(
        hid_t loc_id,
        const std::string& name,
        hid_t type_id,
        const Dataspace& space,
        hid_t lcpl = H5P_DEFAULT,
        hid_t dcpl = H5P_DEFAULT,
        hid_t dapl = H5P_DEFAULT
    ) {
        hid_t id = H5Dcreate(loc_id, name.c_str(), type_id, space.id(),
                            lcpl, dcpl, dapl);
        detail::check_id(id, ("H5Dcreate: " + name).c_str());
        return Dataset(id, false);
    }

    template <typename T>
    static Dataset create(
        hid_t loc_id,
        const std::string& name,
        const std::vector<hsize_t>& dims,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        Dataspace space(dims);
        return create(loc_id, name, detail::native_type<T>(), space,
                     H5P_DEFAULT, props.id(), H5P_DEFAULT);
    }

    // Metadata Queries

    Dataspace get_space() const {
        hid_t space_id = H5Dget_space(_id);
        detail::check_id(space_id, "H5Dget_space");
        return Dataspace(space_id);
    }

    Datatype get_type() const {
        hid_t type_id = H5Dget_type(_id);
        detail::check_id(type_id, "H5Dget_type");
        return Datatype::from_id(type_id);
    }

    DatasetCreateProps get_create_plist() const {
        hid_t plist_id = H5Dget_create_plist(_id);
        detail::check_id(plist_id, "H5Dget_create_plist");
        return DatasetCreateProps(plist_id, H5Pclose);
    }

    DatasetAccessProps get_access_plist() const {
        hid_t plist_id = H5Dget_access_plist(_id);
        detail::check_id(plist_id, "H5Dget_access_plist");
        return DatasetAccessProps(plist_id, H5Pclose);
    }

    std::vector<hsize_t> get_dims() const {
        return get_space().get_dims();
    }

    SCL_NODISCARD int get_rank() const {
        return get_space().get_rank();
    }

    SCL_NODISCARD hsize_t get_num_elements() const {
        return static_cast<hsize_t>(get_space().get_num_elements());
    }

    SCL_NODISCARD hsize_t get_storage_size() const {
        return H5Dget_storage_size(_id);
    }

    // Layout and Chunking

    SCL_NODISCARD H5D_layout_t get_layout() const {
        DatasetCreateProps props = get_create_plist();
        return H5Pget_layout(props.id());
    }

    SCL_NODISCARD bool is_chunked() const {
        return get_layout() == H5D_CHUNKED;
    }

    std::optional<std::vector<hsize_t>> get_chunk_dims() const {
        if (!is_chunked()) return std::nullopt;

        DatasetCreateProps props = get_create_plist();
        int rank = H5Pget_chunk(props.id(), 0, nullptr);
        if (rank < 0) return std::nullopt;

        std::vector<hsize_t> dims(rank);
        H5Pget_chunk(props.id(), rank, dims.data());
        return dims;
    }

    // I/O Operations

    template <typename T>
    void read(T* buffer, hid_t xfer_plist = H5P_DEFAULT) const {
        detail::check_h5(
            H5Dread(_id, detail::native_type<T>(),
                   H5S_ALL, H5S_ALL, xfer_plist, buffer),
            "H5Dread"
        );
    }

    template <typename T>
    void read(T* buffer,
             const Dataspace& mem_space,
             const Dataspace& file_space,
             hid_t xfer_plist = H5P_DEFAULT) const {
        detail::check_h5(
            H5Dread(_id, detail::native_type<T>(),
                   mem_space.id(), file_space.id(), xfer_plist, buffer),
            "H5Dread"
        );
    }

    template <typename T>
    void write(const T* buffer, hid_t xfer_plist = H5P_DEFAULT) {
        detail::check_h5(
            H5Dwrite(_id, detail::native_type<T>(),
                    H5S_ALL, H5S_ALL, xfer_plist, buffer),
            "H5Dwrite"
        );
    }

    template <typename T>
    void write(const T* buffer,
              const Dataspace& mem_space,
              const Dataspace& file_space,
              hid_t xfer_plist = H5P_DEFAULT) {
        detail::check_h5(
            H5Dwrite(_id, detail::native_type<T>(),
                    mem_space.id(), file_space.id(), xfer_plist, buffer),
            "H5Dwrite"
        );
    }

    template <typename T>
    std::vector<T> read_vector() const {
        hsize_t size = get_num_elements();
        std::vector<T> data(size);
        read(data.data());
        return data;
    }

    template <typename T>
    void write_vector(const std::vector<T>& data) {
        write(data.data());
    }

    // Partial I/O

    template <typename T>
    void read_hyperslab(
        T* buffer,
        const std::vector<hsize_t>& start,
        const std::vector<hsize_t>& count,
        const std::vector<hsize_t>& stride = {},
        const std::vector<hsize_t>& block = {}
    ) const {
        Dataspace file_space = get_space();
        file_space.select_hyperslab(start, count, stride, block);

        Dataspace mem_space(count);
        read(buffer, mem_space, file_space);
    }

    template <typename T>
    void write_hyperslab(
        const T* buffer,
        const std::vector<hsize_t>& start,
        const std::vector<hsize_t>& count,
        const std::vector<hsize_t>& stride = {},
        const std::vector<hsize_t>& block = {}
    ) {
        Dataspace file_space = get_space();
        file_space.select_hyperslab(start, count, stride, block);

        Dataspace mem_space(count);
        write(buffer, mem_space, file_space);
    }

    // Extension

    void extend(const std::vector<hsize_t>& size) {
        detail::check_h5(
            H5Dset_extent(_id, size.data()),
            "H5Dset_extent"
        );
    }

    // Attribute Operations

    SCL_NODISCARD bool has_attr(const std::string& name) const {
        return H5Aexists(_id, name.c_str()) > 0;
    }

    Attribute open_attr(const std::string& name) const {
        return Attribute(_id, name);
    }

private:
    Dataset(hid_t id, bool) : Object(id, H5Dclose) {}
};

// =============================================================================
// Group
// =============================================================================

class Group : public Location {
public:
    Group(hid_t loc_id, const std::string& name,
          hid_t gapl = H5P_DEFAULT)
        : Location()
    {
        _id = H5Gopen(loc_id, name.c_str(), gapl);
        _closer = H5Gclose;
        detail::check_id(_id, ("H5Gopen: " + name).c_str());
    }

    static Group create(
        hid_t loc_id,
        const std::string& name,
        hid_t lcpl = H5P_DEFAULT,
        hid_t gcpl = H5P_DEFAULT,
        hid_t gapl = H5P_DEFAULT
    ) {
        hid_t id = H5Gcreate(loc_id, name.c_str(), lcpl, gcpl, gapl);
        detail::check_id(id, ("H5Gcreate: " + name).c_str());
        return Group(id, false);
    }

    // Object Creation

    Group create_group(const std::string& name) {
        return Group::create(_id, name);
    }

    Group open_group(const std::string& name) {
        return Group(_id, name);
    }

    template <typename T>
    Dataset create_dataset(
        const std::string& name,
        const std::vector<hsize_t>& dims,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        return Dataset::create<T>(_id, name, dims, props);
    }

    Dataset create_dataset(
        const std::string& name,
        hid_t type_id,
        const Dataspace& space,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        return Dataset::create(_id, name, type_id, space,
                              H5P_DEFAULT, props.id(), H5P_DEFAULT);
    }

    Dataset open_dataset(const std::string& name) {
        return Dataset(_id, name);
    }

    // Convenience I/O

    template <typename T>
    void write_dataset(
        const std::string& name,
        const T* data,
        const std::vector<hsize_t>& dims,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        Dataset dset = create_dataset<T>(name, dims, props);
        dset.write(data);
    }

    template <typename T>
    void write_dataset(
        const std::string& name,
        const std::vector<T>& data,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        std::vector<hsize_t> dims = {data.size()};
        write_dataset(name, data.data(), dims, props);
    }

    template <typename T>
    std::vector<T> read_dataset(const std::string& name) {
        Dataset dset = open_dataset(name);
        return dset.read_vector<T>();
    }

protected:
    Group() = default;

private:
    Group(hid_t id, bool) : Location() {
        _id = id;
        _closer = H5Gclose;
    }
};

// =============================================================================
// File
// =============================================================================

class File : public Location {
public:
    explicit File(const std::string& path,
                 unsigned flags = H5F_ACC_RDONLY,
                 hid_t fapl = H5P_DEFAULT)
        : Location()
    {
        _id = H5Fopen(path.c_str(), flags, fapl);
        _closer = H5Fclose;
        detail::check_id(_id, ("H5Fopen: " + path).c_str());
    }

    static File create(
        const std::string& path,
        unsigned flags = H5F_ACC_TRUNC,
        hid_t fcpl = H5P_DEFAULT,
        hid_t fapl = H5P_DEFAULT
    ) {
        hid_t id = H5Fcreate(path.c_str(), flags, fcpl, fapl);
        detail::check_id(id, ("H5Fcreate: " + path).c_str());
        return File(id, false);
    }

    // File Operations

    void flush(H5F_scope_t scope = H5F_SCOPE_GLOBAL) {
        detail::check_h5(H5Fflush(_id, scope), "H5Fflush");
    }

    SCL_NODISCARD hsize_t get_file_size() const {
        hsize_t size = 0;
        detail::check_h5(H5Fget_filesize(_id, &size), "H5Fget_filesize");
        return size;
    }

    std::string get_name() const {
        ssize_t size = H5Fget_name(_id, nullptr, 0);
        if (size < 0) return "";
        std::string name(size, '\0');
        H5Fget_name(_id, name.data(), size + 1);
        return name;
    }

    SCL_NODISCARD hssize_t get_freespace() const {
        return H5Fget_freespace(_id);
    }

    FileCreateProps get_create_plist() const {
        hid_t plist_id = H5Fget_create_plist(_id);
        detail::check_id(plist_id, "H5Fget_create_plist");
        return FileCreateProps(plist_id, H5Pclose);
    }

    FileAccessProps get_access_plist() const {
        hid_t plist_id = H5Fget_access_plist(_id);
        detail::check_id(plist_id, "H5Fget_access_plist");
        return FileAccessProps(plist_id, H5Pclose);
    }

    SCL_NODISCARD ssize_t get_obj_count(unsigned types = H5F_OBJ_ALL) const {
        return H5Fget_obj_count(_id, types);
    }

    // Root Group Access

    Group create_group(const std::string& name) {
        return Group::create(_id, name);
    }

    Group open_group(const std::string& name) {
        return Group(_id, name);
    }

    template <typename T>
    Dataset create_dataset(
        const std::string& name,
        const std::vector<hsize_t>& dims,
        const DatasetCreateProps& props = DatasetCreateProps()
    ) {
        return Dataset::create<T>(_id, name, dims, props);
    }

    Dataset open_dataset(const std::string& name) {
        return Dataset(_id, name);
    }

protected:
    File() = default;

private:
    File(hid_t id, bool) : Location() {
        _id = id;
        _closer = H5Fclose;
    }
};

} // namespace scl::io::h5

#endif // SCL_HAS_HDF5
