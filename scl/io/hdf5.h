// =============================================================================
// FILE: scl/io/hdf5.h
// BRIEF: API reference for Modern C++ HDF5 Wrapper
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <ctime>
#include <string>
#include <vector>
#include <optional>
#include <functional>

#ifdef SCL_HAS_HDF5
#include <hdf5.h>

namespace scl::io::h5 {

// =============================================================================
// MODULE OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: HDF5 C++ Wrapper
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Modern C++ object-oriented wrapper over HDF5 C API.
 *
 * PURPOSE:
 *     Provides type-safe, RAII-managed interface for HDF5 operations with:
 *     - Automatic resource cleanup
 *     - Type safety via templates
 *     - Exception-based error handling
 *     - Rich query and manipulation APIs
 *
 * OBJECT HIERARCHY:
 *     Object (base for all HDF5 objects)
 *       |-- Location (containers)
 *       |     |-- File (top-level)
 *       |     |-- Group (subcontainer)
 *       |-- Dataset (data arrays)
 *       |-- Datatype (type descriptors)
 *       |-- Attribute (metadata)
 *       |-- Dataspace (array layout)
 *       |-- PropertyList (configuration)
 *             |-- FileAccessProps
 *             |-- FileCreateProps
 *             |-- DatasetAccessProps
 *             |-- DatasetCreateProps
 *             |-- DatasetTransferProps
 *
 * DESIGN PRINCIPLES:
 *     1. RAII: Automatic HDF5 ID cleanup via destructors
 *     2. Move-only: No copy construction (unique ownership)
 *     3. Type Safety: Template-based type dispatch
 *     4. Zero Overhead: Inline wrappers compile to direct C calls
 *     5. Exception Safety: All HDF5 errors throw SCL exceptions
 *
 * THREAD SAFETY:
 *     Unsafe - HDF5 C library is not thread-safe by default
 *
 * REQUIREMENTS:
 *     SCL_HAS_HDF5 must be defined, HDF5 C library must be linked
 * -------------------------------------------------------------------------- */

// =============================================================================
// ENUMS AND TYPES
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: ObjectType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 object type enumeration.
 *
 * VALUES:
 *     Unknown    - Unrecognized or invalid object type
 *     File       - HDF5 file object
 *     Group      - Group (container) object
 *     Dataset    - Dataset (array) object
 *     Datatype   - Named datatype object
 *     Attribute  - Attribute (metadata) object
 *     Dataspace  - Dataspace (layout) object
 * -------------------------------------------------------------------------- */
enum class ObjectType;

/* -----------------------------------------------------------------------------
 * ENUM: LinkType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 link type enumeration.
 *
 * VALUES:
 *     Hard     - Hard link (reference counted)
 *     Soft     - Soft link (symbolic, may dangle)
 *     External - External link (points to another file)
 *     Error    - Invalid or unrecognized link type
 * -------------------------------------------------------------------------- */
enum class LinkType;

/* -----------------------------------------------------------------------------
 * STRUCT: ObjectInfo
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 object metadata information.
 *
 * MEMBERS:
 *     type       [ObjectType] - Object type
 *     ref_count  [size_t]     - Reference count
 *     num_attrs  [hsize_t]    - Number of attributes
 *     atime      [time_t]     - Access time
 *     mtime      [time_t]     - Modification time
 *     ctime      [time_t]     - Change time
 *     btime      [time_t]     - Birth (creation) time
 * -------------------------------------------------------------------------- */
struct ObjectInfo;

/* -----------------------------------------------------------------------------
 * STRUCT: LinkInfo
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 link metadata information.
 *
 * MEMBERS:
 *     type          [LinkType]  - Link type
 *     corder_valid  [bool]      - Creation order tracking enabled
 *     corder        [int64_t]   - Creation order value
 *     cset          [H5T_cset_t]- Character set encoding
 *     value_size    [size_t]    - Size of link value
 * -------------------------------------------------------------------------- */
struct LinkInfo;

// =============================================================================
// OBJECT - BASE CLASS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Object
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Base class for all HDF5 objects with RAII management.
 *
 * PURPOSE:
 *     Provides automatic resource management and common functionality:
 *     - Automatic HDF5 ID cleanup via destructor
 *     - Move-only semantics (no copy)
 *     - ID validity checking
 *     - Reference counting operations
 *
 * MUTABILITY:
 *     Move-only, non-copyable
 *
 * THREAD SAFETY:
 *     Unsafe
 * -------------------------------------------------------------------------- */
class Object {
protected:
    hid_t _id;                   // HDF5 ID
    herr_t (*_closer)(hid_t);    // Cleanup function

    explicit Object(
        hid_t id,                // HDF5 ID
        herr_t (*closer)(hid_t)  // Closer function (H5Fclose, H5Dclose, etc)
    ) noexcept;

    Object() noexcept;           // Default constructor (invalid object)

public:
    virtual ~Object() noexcept;  // Calls close()

    /* -------------------------------------------------------------------------
     * FUNCTION: close
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Close HDF5 object and invalidate ID.
     *
     * POSTCONDITIONS:
     *     - If valid: closer function called, ID set to invalid
     *     - If invalid: no-op
     *
     * MUTABILITY:
     *     INPLACE - modifies object state
     *
     * THREAD SAFETY:
     *     Unsafe
     * ---------------------------------------------------------------------- */
    void close() noexcept;

    Object(Object&& other) noexcept;              // Move constructor
    Object& operator=(Object&& other) noexcept;   // Move assignment

    Object(const Object&) = delete;               // No copy
    Object& operator=(const Object&) = delete;

    /* -------------------------------------------------------------------------
     * FUNCTION: id
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get raw HDF5 ID.
     *
     * RETURN VALUE:
     *     HDF5 ID (hid_t), may be invalid
     * ---------------------------------------------------------------------- */
    hid_t id() const noexcept;

    /* -------------------------------------------------------------------------
     * FUNCTION: is_valid
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if object has valid HDF5 ID.
     *
     * RETURN VALUE:
     *     true if ID is valid, false otherwise
     * ---------------------------------------------------------------------- */
    bool is_valid() const noexcept;

    operator hid_t() const noexcept;  // Implicit conversion for HDF5 C API

    /* -------------------------------------------------------------------------
     * FUNCTION: release
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Release ownership of ID without closing.
     *
     * POSTCONDITIONS:
     *     - ID set to invalid
     *     - Caller owns returned ID (must close manually)
     *
     * RETURN VALUE:
     *     Previously owned HDF5 ID
     * ---------------------------------------------------------------------- */
    hid_t release() noexcept;

    int get_ref_count() const;        // Get reference count
    void inc_ref();                   // Increment reference count
    void dec_ref();                   // Decrement reference count
    H5I_type_t get_type() const;      // Get HDF5 type
};

// =============================================================================
// DATASPACE
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Dataspace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 dataspace (defines array dimensionality and selection).
 *
 * PURPOSE:
 *     Describes:
 *     - Array dimensions (rank and extent)
 *     - Element selections (hyperslab, all, none)
 *     - Memory and file space layouts
 *
 * USE CASES:
 *     - Creating datasets with specific dimensions
 *     - Partial I/O with hyperslab selections
 *     - Resizable datasets with maximum dimensions
 * -------------------------------------------------------------------------- */
class Dataspace : public Object {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Dataspace(dims, maxdims)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create simple (rectangular) dataspace.
     *
     * PARAMETERS:
     *     dims    [in] - Current dimensions
     *     maxdims [in] - Maximum dimensions (empty = fixed size)
     *
     * POSTCONDITIONS:
     *     Dataspace created with specified dimensions
     *
     * THROWS:
     *     IOError - if HDF5 creation fails
     * ---------------------------------------------------------------------- */
    explicit Dataspace(
        const std::vector<hsize_t>& dims,         // Current dimensions
        const std::vector<hsize_t>& maxdims = {}  // Max dimensions (unlimited if empty)
    );

    explicit Dataspace(hid_t space_id);  // Wrap existing dataspace ID

    static Dataspace scalar();           // Create scalar (single value) dataspace
    static Dataspace null();             // Create null (no elements) dataspace

    int get_rank() const;                     // Get number of dimensions
    std::vector<hsize_t> get_dims() const;    // Get current dimensions
    std::vector<hsize_t> get_max_dims() const;// Get maximum dimensions
    hssize_t get_num_elements() const;        // Get total element count
    bool is_simple() const;                   // Check if simple (rectangular)

    /* -------------------------------------------------------------------------
     * FUNCTION: select_hyperslab
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Select rectangular region for partial I/O.
     *
     * PARAMETERS:
     *     start  [in] - Starting coordinates
     *     count  [in] - Number of blocks in each dimension
     *     stride [in] - Stride between blocks (default: contiguous)
     *     block  [in] - Size of each block (default: single element)
     *     op     [in] - Selection operator (SET, OR, AND, XOR, NOTB, NOTA)
     *
     * POSTCONDITIONS:
     *     Selection region updated according to operator
     *
     * MUTABILITY:
     *     INPLACE - modifies selection
     *
     * THROWS:
     *     IOError - if selection is invalid
     * ---------------------------------------------------------------------- */
    void select_hyperslab(
        const std::vector<hsize_t>& start,         // Starting coordinates
        const std::vector<hsize_t>& count,         // Number of blocks
        const std::vector<hsize_t>& stride = {},   // Stride (contiguous if empty)
        const std::vector<hsize_t>& block = {},    // Block size (1 if empty)
        H5S_seloper_t op = H5S_SELECT_SET          // Selection operator
    );

    void select_all();                    // Select all elements
    void select_none();                   // Select no elements
    hssize_t get_select_npoints() const;  // Get number of selected points
    bool is_select_valid() const;         // Check if selection is valid

protected:
    Dataspace() = default;
};

// =============================================================================
// DATATYPE
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Datatype
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 datatype descriptor.
 *
 * PURPOSE:
 *     Describes element types for datasets and attributes:
 *     - Native types (int, float, etc)
 *     - String types (fixed/variable length)
 *     - Custom compound types
 *
 * TYPE SAFETY:
 *     Template method native<T>() ensures compile-time type checking
 * -------------------------------------------------------------------------- */
class Datatype : public Object {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Datatype(type_id)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create Datatype by copying HDF5 type ID.
     *
     * PARAMETERS:
     *     type_id [in] - HDF5 type ID (must be valid)
     *
     * POSTCONDITIONS:
     *     - Datatype created with copy of type_id
     *     - Original type_id remains valid (not owned)
     *
     * THROWS:
     *     IOError - if H5Tcopy fails
     * ---------------------------------------------------------------------- */
    explicit Datatype(hid_t type_id);     // Always copies type_id

    /* -------------------------------------------------------------------------
     * FUNCTION: native<T>
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create native type from C++ type.
     *
     * PARAMETERS:
     *     T [template] - C++ type (float, double, int32_t, etc)
     *
     * RETURN VALUE:
     *     Datatype representing T in HDF5
     *
     * THROWS:
     *     TypeError - if T is not supported
     *
     * SUPPORTED TYPES:
     *     float, double, int8_t, int16_t, int32_t, int64_t,
     *     uint8_t, uint16_t, uint32_t, uint64_t, bool
     * ---------------------------------------------------------------------- */
    template <typename T>
    static Datatype native();

    static Datatype string_vlen();        // Variable-length string
    static Datatype string_fixed(size_t len);  // Fixed-length string

    /* -------------------------------------------------------------------------
     * FUNCTION: from_id
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create Datatype from HDF5 type ID without copying (takes ownership).
     *
     * PARAMETERS:
     *     type_id [in] - HDF5 type ID (must be newly created, will be owned)
     *
     * POSTCONDITIONS:
     *     - Datatype owns type_id (will close on destruction)
     *     - Should only be used with newly created type IDs
     *
     * USE CASE:
     *     Internal use for string_vlen() and string_fixed() factory methods
     * ---------------------------------------------------------------------- */
    static Datatype from_id(hid_t type_id);

    size_t get_size() const;              // Get size in bytes
    H5T_class_t get_class() const;        // Get type class
    bool equals(const Datatype& other) const;  // Check equality

protected:
    Datatype() = default;
};

// =============================================================================
// ATTRIBUTE
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Attribute
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 attribute (metadata attached to objects).
 *
 * PURPOSE:
 *     Attributes store small metadata associated with groups, datasets,
 *     or datatypes. Unlike datasets:
 *     - Cannot be chunked or compressed
 *     - No partial I/O
 *     - Intended for small metadata (< 64KB)
 * -------------------------------------------------------------------------- */
class Attribute : public Object {
public:
    Attribute(
        hid_t loc_id,            // Parent object ID
        const std::string& name  // Attribute name
    );

    /* -------------------------------------------------------------------------
     * FUNCTION: create
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create new attribute.
     *
     * PARAMETERS:
     *     loc_id [in] - Parent object ID
     *     name   [in] - Attribute name
     *     type_id[in] - Datatype ID
     *     space  [in] - Dataspace
     *
     * RETURN VALUE:
     *     New Attribute object
     *
     * THROWS:
     *     IOError - if creation fails
     * ---------------------------------------------------------------------- */
    static Attribute create(
        hid_t loc_id,               // Parent object
        const std::string& name,    // Attribute name
        hid_t type_id,              // Datatype
        const Dataspace& space      // Dataspace
    );

    Dataspace get_space() const;         // Get dataspace
    Datatype get_type() const;           // Get datatype
    std::string get_name() const;        // Get attribute name

    template <typename T>
    void read(T* buffer) const;          // Read attribute data

    template <typename T>
    void write(const T* buffer);         // Write attribute data

    template <typename T>
    T read_scalar() const;               // Read scalar attribute

    template <typename T>
    void write_scalar(const T& value);   // Write scalar attribute

    /* -------------------------------------------------------------------------
     * FUNCTION: read_string
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Read string attribute, handling both fixed and variable-length strings.
     *
     * RETURN VALUE:
     *     String value from attribute
     *
     * ALGORITHM:
     *     If variable-length string:
     *         1. Allocate memory type with H5T_VARIABLE
     *         2. Read string pointer from HDF5
     *         3. Copy string content to std::string
     *         4. Release HDF5-allocated memory via H5Dvlen_reclaim
     *     If fixed-length string:
     *         1. Read buffer of fixed size
     *         2. Remove trailing null characters
     *
     * EXCEPTION SAFETY:
     *     Strong guarantee - memory properly released even on exception
     *
     * THROWS:
     *     IOError - if read fails
     * ---------------------------------------------------------------------- */
    std::string read_string() const;     // Read string attribute

    void write_string(const std::string& value);  // Write string attribute

private:
    Attribute(hid_t id, bool);           // Private constructor for create()
};

// =============================================================================
// PROPERTY LISTS
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: PropertyList
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Base class for HDF5 property lists.
 *
 * PURPOSE:
 *     Property lists control HDF5 operation behavior:
 *     - File creation/access settings
 *     - Dataset creation/access/transfer settings
 *     - Link creation settings
 * -------------------------------------------------------------------------- */
class PropertyList : public Object {
protected:
    explicit PropertyList(hid_t cls_id);  // Create from property class
    PropertyList() = default;
    PropertyList(hid_t id, herr_t (*closer)(hid_t));

public:
    PropertyList copy() const;            // Copy property list
};

/* -----------------------------------------------------------------------------
 * CLASS: FileAccessProps
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     File access property list.
 *
 * PURPOSE:
 *     Controls file access behavior:
 *     - Chunk cache parameters
 *     - File driver settings
 *     - Alignment parameters
 * -------------------------------------------------------------------------- */
class FileAccessProps : public PropertyList {
public:
    FileAccessProps();

    /* -------------------------------------------------------------------------
     * FUNCTION: copy
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create copy of property list.
     *
     * RETURN VALUE:
     *     New FileAccessProps object (not PropertyList base class)
     *
     * POSTCONDITIONS:
     *     - New object has independent copy of all properties
     *     - Both objects can be modified independently
     * ---------------------------------------------------------------------- */
    FileAccessProps copy() const;

    FileAccessProps& cache(
        size_t nslots,          // Number of chunk slots
        size_t nbytes,          // Total cache size in bytes
        double preemption       // Preemption policy (0.0-1.0)
    );

    FileAccessProps& alignment(
        hsize_t threshold,      // Threshold for alignment
        hsize_t alignment       // Alignment value
    );
};

class FileCreateProps : public PropertyList {
public:
    FileCreateProps();

    /* -------------------------------------------------------------------------
     * FUNCTION: copy
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create copy of property list.
     *
     * RETURN VALUE:
     *     New FileCreateProps object (not PropertyList base class)
     *
     * POSTCONDITIONS:
     *     - New object has independent copy of all properties
     *     - Both objects can be modified independently
     * ---------------------------------------------------------------------- */
    FileCreateProps copy() const;
};

class DatasetAccessProps : public PropertyList {
public:
    DatasetAccessProps();

    /* -------------------------------------------------------------------------
     * FUNCTION: copy
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create copy of property list.
     *
     * RETURN VALUE:
     *     New DatasetAccessProps object (not PropertyList base class)
     *
     * POSTCONDITIONS:
     *     - New object has independent copy of all properties
     *     - Both objects can be modified independently
     * ---------------------------------------------------------------------- */
    DatasetAccessProps copy() const;

    DatasetAccessProps& chunk_cache(
        size_t nslots,          // Number of chunk slots
        size_t nbytes,          // Total cache size
        double preemption       // Preemption policy
    );
};

/* -----------------------------------------------------------------------------
 * CLASS: DatasetCreateProps
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Dataset creation property list.
 *
 * PURPOSE:
 *     Controls dataset creation behavior:
 *     - Chunking (required for compression)
 *     - Compression (deflate/gzip)
 *     - Filters (shuffle, etc)
 *     - Fill values
 * -------------------------------------------------------------------------- */
class DatasetCreateProps : public PropertyList {
public:
    DatasetCreateProps();

    /* -------------------------------------------------------------------------
     * FUNCTION: copy
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create copy of property list.
     *
     * RETURN VALUE:
     *     New DatasetCreateProps object (not PropertyList base class)
     *
     * POSTCONDITIONS:
     *     - New object has independent copy of all properties
     *     - Both objects can be modified independently
     * ---------------------------------------------------------------------- */
    DatasetCreateProps copy() const;

    DatasetCreateProps& chunked(
        const std::vector<hsize_t>& chunk_dims  // Chunk dimensions
    );

    DatasetCreateProps& deflate(
        unsigned level = 6      // Compression level (0-9)
    );

    DatasetCreateProps& shuffle();  // Enable shuffle filter

    template <typename T>
    DatasetCreateProps& fill_value(
        const T& value          // Fill value
    );

    DatasetCreateProps& alloc_time(
        H5D_alloc_time_t alloc_time  // Allocation time
    );
};

class DatasetTransferProps : public PropertyList {
public:
    DatasetTransferProps();
};

// =============================================================================
// LOCATION - BASE FOR FILE AND GROUP
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Location
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Base class for HDF5 containers (File and Group).
 *
 * PURPOSE:
 *     Provides common functionality for objects that can contain other objects:
 *     - Object existence queries
 *     - Object creation and deletion
 *     - Link management
 *     - Attribute operations
 *     - Iteration over contents
 * -------------------------------------------------------------------------- */
class Location : public Object {
protected:
    using Object::Object;
    Location() = default;

public:
    // Existence and Type Queries

    bool exists(const std::string& name) const;             // Check if object exists
    bool has_attr(const std::string& name) const;           // Check if attribute exists
    ObjectType get_object_type(const std::string& name) const;  // Get object type
    LinkType get_link_type(const std::string& name) const;      // Get link type

    // Object Information

    ObjectInfo get_info(const std::string& name = ".") const;   // Get object info

    /* -------------------------------------------------------------------------
     * FUNCTION: get_num_objs
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of objects (links) in group.
     *
     * RETURN VALUE:
     *     Number of links (hsize_t)
     *
     * ALGORITHM:
     *     Uses H5Gget_info() (modern API, replaces deprecated H5Gget_num_objs)
     *
     * THROWS:
     *     IOError - if query fails
     * ---------------------------------------------------------------------- */
    hsize_t get_num_objs() const;                           // Get number of objects

    /* -------------------------------------------------------------------------
     * FUNCTION: get_objname_by_idx
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get object name by index using creation order.
     *
     * PARAMETERS:
     *     idx [in] - Object index
     *
     * RETURN VALUE:
     *     Object name, empty string if index invalid
     *
     * ALGORITHM:
     *     Uses H5Lget_name_by_idx() (modern API, replaces deprecated H5Gget_objname_by_idx)
     *
     * THROWS:
     *     None - returns empty string on error
     * ---------------------------------------------------------------------- */
    std::string get_objname_by_idx(hsize_t idx) const;      // Get object name by index

    std::vector<std::string> list_objects() const;          // List all object names

    // Iteration

    /* -------------------------------------------------------------------------
     * FUNCTION: iterate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Iterate over objects in location, calling callback for each.
     *
     * PARAMETERS:
     *     func [in] - Callback function returning true to continue, false to stop
     *
     * POSTCONDITIONS:
     *     - Callback called for each object name
     *     - If callback throws, exception is caught and rethrown after iteration stops
     *
     * EXCEPTION SAFETY:
     *     Strong guarantee - exceptions from callback are properly propagated
     *
     * THREAD SAFETY:
     *     Unsafe - HDF5 library is not thread-safe
     *
     * THROWS:
     *     Any exception thrown by callback function
     *     IOError - if iteration fails
     * ---------------------------------------------------------------------- */
    void iterate(
        const std::function<bool(const std::string&, ObjectType)>& func  // Callback
    ) const;

    // Link Operations

    void unlink(const std::string& name);                   // Delete link
    void link_soft(const std::string& target, const std::string& link_name);    // Soft link
    void link_hard(const std::string& target, const std::string& link_name);    // Hard link
    void move_link(const std::string& src, const std::string& dst);  // Move/rename link
    void copy_link(const std::string& src, const std::string& dst);  // Copy link

    // Attribute Operations

    /* -------------------------------------------------------------------------
     * FUNCTION: get_num_attrs
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get number of attributes attached to this object.
     *
     * RETURN VALUE:
     *     Number of attributes (int)
     *
     * ALGORITHM:
     *     Uses H5Oget_info() (modern API, replaces deprecated H5Aget_num_attrs)
     *
     * THROWS:
     *     IOError - if query fails
     * ---------------------------------------------------------------------- */
    int get_num_attrs() const;                              // Get attribute count

    Attribute open_attr(const std::string& name) const;     // Open attribute

    template <typename T>
    Attribute create_attr(
        const std::string& name,                 // Attribute name
        const std::vector<hsize_t>& dims = {}    // Dimensions (scalar if empty)
    );

    void delete_attr(const std::string& name);              // Delete attribute

    template <typename T>
    T read_attr(const std::string& name) const;             // Read scalar attribute

    template <typename T>
    void write_attr(const std::string& name, const T& value);  // Write scalar attribute

    template <typename T>
    std::vector<T> read_attr_array(const std::string& name) const;  // Read array attribute

    /* -------------------------------------------------------------------------
     * FUNCTION: write_attr_array
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Write array attribute, deleting existing attribute if present.
     *
     * PARAMETERS:
     *     name [in] - Attribute name
     *     data [in] - Data array
     *
     * POSTCONDITIONS:
     *     - If attribute exists: deleted and recreated with new dimensions
     *     - Attribute created with dimensions matching data.size()
     *
     * MUTABILITY:
     *     INPLACE - modifies existing attribute if present
     *
     * RATIONALE:
     *     Deletes existing attribute to avoid dimension mismatch errors
     *
     * THROWS:
     *     IOError - if write fails
     * ---------------------------------------------------------------------- */
    template <typename T>
    void write_attr_array(const std::string& name, const std::vector<T>& data);  // Write array

    std::string read_attr_string(const std::string& name) const;    // Read string attribute

    /* -------------------------------------------------------------------------
     * FUNCTION: write_attr_string
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Write string attribute, deleting existing attribute if present.
     *
     * PARAMETERS:
     *     name  [in] - Attribute name
     *     value [in] - String value
     *
     * POSTCONDITIONS:
     *     - If attribute exists: deleted and recreated with new size
     *     - Attribute created with size = value.size() + 1 (includes null terminator)
     *
     * MUTABILITY:
     *     INPLACE - modifies existing attribute if present
     *
     * RATIONALE:
     *     Deletes existing attribute to avoid size mismatch errors when
     *     new string length differs from existing attribute size
     *
     * THROWS:
     *     IOError - if write fails
     * ---------------------------------------------------------------------- */
    void write_attr_string(const std::string& name, const std::string& value);  // Write string
};

// =============================================================================
// DATASET
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Dataset
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 dataset (multidimensional array storage).
 *
 * PURPOSE:
 *     Primary data storage in HDF5:
 *     - Multidimensional arrays
 *     - Chunked or contiguous layout
 *     - Compression and filters
 *     - Partial I/O (hyperslabs)
 *     - Resizable dimensions
 *
 * PERFORMANCE:
 *     - Chunking enables compression but adds overhead
 *     - Contiguous layout fastest for full reads
 *     - Hyperslab I/O efficient for partial access
 * -------------------------------------------------------------------------- */
class Dataset : public Object {
public:
    Dataset(
        hid_t loc_id,            // Parent location
        const std::string& name, // Dataset name
        hid_t dapl = H5P_DEFAULT // Access property list
    );

    /* -------------------------------------------------------------------------
     * FUNCTION: create
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create new dataset.
     *
     * PARAMETERS:
     *     loc_id [in] - Parent location ID
     *     name   [in] - Dataset name
     *     type_id[in] - Datatype ID
     *     space  [in] - Dataspace
     *     lcpl   [in] - Link creation property list
     *     dcpl   [in] - Dataset creation property list
     *     dapl   [in] - Dataset access property list
     *
     * RETURN VALUE:
     *     New Dataset object
     *
     * THROWS:
     *     IOError - if creation fails
     * ---------------------------------------------------------------------- */
    static Dataset create(
        hid_t loc_id,                      // Parent location
        const std::string& name,           // Dataset name
        hid_t type_id,                     // Datatype
        const Dataspace& space,            // Dataspace
        hid_t lcpl = H5P_DEFAULT,          // Link creation props
        hid_t dcpl = H5P_DEFAULT,          // Dataset creation props
        hid_t dapl = H5P_DEFAULT           // Dataset access props
    );

    template <typename T>
    static Dataset create(
        hid_t loc_id,                             // Parent location
        const std::string& name,                  // Dataset name
        const std::vector<hsize_t>& dims,         // Dimensions
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    // Metadata Queries

    Dataspace get_space() const;                 // Get dataspace
    Datatype get_type() const;                   // Get datatype
    DatasetCreateProps get_create_plist() const; // Get creation properties
    DatasetAccessProps get_access_plist() const; // Get access properties
    std::vector<hsize_t> get_dims() const;       // Get dimensions
    int get_rank() const;                        // Get rank
    hsize_t get_num_elements() const;            // Get total elements
    hsize_t get_storage_size() const;            // Get storage size in bytes

    // Layout and Chunking

    H5D_layout_t get_layout() const;             // Get layout type
    bool is_chunked() const;                     // Check if chunked
    std::optional<std::vector<hsize_t>> get_chunk_dims() const;  // Get chunk dimensions

    // I/O Operations

    template <typename T>
    void read(
        T* buffer,                       // Output buffer
        hid_t xfer_plist = H5P_DEFAULT   // Transfer property list
    ) const;

    template <typename T>
    void read(
        T* buffer,                       // Output buffer
        const Dataspace& mem_space,      // Memory dataspace
        const Dataspace& file_space,     // File dataspace
        hid_t xfer_plist = H5P_DEFAULT   // Transfer property list
    ) const;

    template <typename T>
    void write(
        const T* buffer,                 // Input buffer
        hid_t xfer_plist = H5P_DEFAULT   // Transfer property list
    );

    template <typename T>
    void write(
        const T* buffer,                 // Input buffer
        const Dataspace& mem_space,      // Memory dataspace
        const Dataspace& file_space,     // File dataspace
        hid_t xfer_plist = H5P_DEFAULT   // Transfer property list
    );

    template <typename T>
    std::vector<T> read_vector() const;          // Read entire dataset to vector

    template <typename T>
    void write_vector(const std::vector<T>& data);  // Write vector to dataset

    // Partial I/O

    template <typename T>
    void read_hyperslab(
        T* buffer,                                   // Output buffer
        const std::vector<hsize_t>& start,           // Start coordinates
        const std::vector<hsize_t>& count,           // Count
        const std::vector<hsize_t>& stride = {},     // Stride
        const std::vector<hsize_t>& block = {}       // Block size
    ) const;

    template <typename T>
    void write_hyperslab(
        const T* buffer,                             // Input buffer
        const std::vector<hsize_t>& start,           // Start coordinates
        const std::vector<hsize_t>& count,           // Count
        const std::vector<hsize_t>& stride = {},     // Stride
        const std::vector<hsize_t>& block = {}       // Block size
    );

    // Extension

    void extend(const std::vector<hsize_t>& size);   // Extend dataset (chunked only)

    // Attribute Operations

    bool has_attr(const std::string& name) const;    // Check attribute exists
    Attribute open_attr(const std::string& name) const;  // Open attribute

private:
    Dataset(hid_t id, bool);                         // Private constructor
};

// =============================================================================
// GROUP
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Group
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 group (container for datasets and subgroups).
 *
 * PURPOSE:
 *     Organizes HDF5 file structure:
 *     - Contains datasets, groups, and datatypes
 *     - Forms hierarchical directory structure
 *     - Inherits all Location functionality
 *
 * USE CASES:
 *     - Organizing related datasets
 *     - Creating hierarchical data structures
 *     - Namespace management
 * -------------------------------------------------------------------------- */
class Group : public Location {
public:
    Group(
        hid_t loc_id,            // Parent location
        const std::string& name, // Group name
        hid_t gapl = H5P_DEFAULT // Access property list
    );

    static Group create(
        hid_t loc_id,                   // Parent location
        const std::string& name,        // Group name
        hid_t lcpl = H5P_DEFAULT,       // Link creation props
        hid_t gcpl = H5P_DEFAULT,       // Group creation props
        hid_t gapl = H5P_DEFAULT        // Group access props
    );

    // Object Creation

    Group create_group(const std::string& name);      // Create subgroup
    Group open_group(const std::string& name);        // Open subgroup

    template <typename T>
    Dataset create_dataset(
        const std::string& name,                      // Dataset name
        const std::vector<hsize_t>& dims,             // Dimensions
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    Dataset create_dataset(
        const std::string& name,                      // Dataset name
        hid_t type_id,                                // Datatype
        const Dataspace& space,                       // Dataspace
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    Dataset open_dataset(const std::string& name);    // Open dataset

    // Convenience I/O

    template <typename T>
    void write_dataset(
        const std::string& name,                      // Dataset name
        const T* data,                                // Data buffer
        const std::vector<hsize_t>& dims,             // Dimensions
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    template <typename T>
    void write_dataset(
        const std::string& name,                      // Dataset name
        const std::vector<T>& data,                   // Data vector
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    template <typename T>
    std::vector<T> read_dataset(const std::string& name);  // Read dataset to vector

protected:
    Group() = default;

private:
    Group(hid_t id, bool);                            // Private constructor
};

// =============================================================================
// FILE
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: File
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     HDF5 file (top-level container).
 *
 * PURPOSE:
 *     Root of HDF5 object hierarchy:
 *     - Contains groups, datasets, and datatypes
 *     - Manages file I/O and flushing
 *     - Inherits all Location functionality
 *
 * FILE ACCESS MODES:
 *     H5F_ACC_RDONLY - Read-only
 *     H5F_ACC_RDWR   - Read-write (must exist)
 *     H5F_ACC_TRUNC  - Create, truncate if exists
 *     H5F_ACC_EXCL   - Create, fail if exists
 * -------------------------------------------------------------------------- */
class File : public Location {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: File(path, flags, fapl)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Open existing HDF5 file.
     *
     * PARAMETERS:
     *     path  [in] - File path
     *     flags [in] - Access flags (H5F_ACC_RDONLY, H5F_ACC_RDWR)
     *     fapl  [in] - File access property list
     *
     * THROWS:
     *     IOError - if file cannot be opened
     * ---------------------------------------------------------------------- */
    explicit File(
        const std::string& path,            // File path
        unsigned flags = H5F_ACC_RDONLY,    // Access flags
        hid_t fapl = H5P_DEFAULT            // File access props
    );

    /* -------------------------------------------------------------------------
     * FUNCTION: create
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create new HDF5 file.
     *
     * PARAMETERS:
     *     path  [in] - File path
     *     flags [in] - Creation flags (H5F_ACC_TRUNC, H5F_ACC_EXCL)
     *     fcpl  [in] - File creation property list
     *     fapl  [in] - File access property list
     *
     * RETURN VALUE:
     *     New File object
     *
     * THROWS:
     *     IOError - if file cannot be created
     * ---------------------------------------------------------------------- */
    static File create(
        const std::string& path,            // File path
        unsigned flags = H5F_ACC_TRUNC,     // Creation flags
        hid_t fcpl = H5P_DEFAULT,           // File creation props
        hid_t fapl = H5P_DEFAULT            // File access props
    );

    // File Operations

    void flush(H5F_scope_t scope = H5F_SCOPE_GLOBAL);  // Flush to disk
    hsize_t get_file_size() const;                     // Get file size in bytes
    std::string get_name() const;                      // Get file name
    hssize_t get_freespace() const;                    // Get free space
    FileCreateProps get_create_plist() const;          // Get creation props
    FileAccessProps get_access_plist() const;          // Get access props
    ssize_t get_obj_count(unsigned types = H5F_OBJ_ALL) const;  // Get open object count

    // Root Group Access

    Group create_group(const std::string& name);       // Create group at root
    Group open_group(const std::string& name);         // Open group

    template <typename T>
    Dataset create_dataset(
        const std::string& name,                       // Dataset name
        const std::vector<hsize_t>& dims,              // Dimensions
        const DatasetCreateProps& props = DatasetCreateProps()  // Creation props
    );

    Dataset open_dataset(const std::string& name);     // Open dataset

protected:
    File() = default;

private:
    File(hid_t id, bool);                              // Private constructor
};

} // namespace scl::io::h5

#endif // SCL_HAS_HDF5
