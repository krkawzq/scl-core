#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"

#include <string>
#include <future>
#include <cstring>

// =============================================================================
/// @file mmap.hpp
/// @brief Pure Memory-Mapped File Utilities
///
/// Provides generic RAII wrappers for memory-mapped files with no assumptions
/// about data structure or file organization. All platform-specific code is
/// abstracted through macros in scl/core/macros.hpp.
///
/// Design Philosophy:
///
/// 1. Generic: Works with any file type (binary arrays, matrices, custom formats)
/// 2. Platform Agnostic: Zero platform-specific code
/// 3. Handle-Based: Initialize from file handles or paths
/// 4. Composable: Building block for higher-level abstractions
///
/// Core Component:
///
/// - MappedArray<T>: Generic memory-mapped typed array
///
/// Future Extensions:
///
/// - MappedFile: Untyped memory-mapped file (raw bytes)
/// - MappedRegion: Partial file mapping (windows into large files)
/// - MappedBuffer: Writable memory-mapped buffer
// =============================================================================

namespace scl::io {

// =============================================================================
// MappedArray: Generic Memory-Mapped Typed Array
// =============================================================================

/// @brief Generic RAII wrapper for memory-mapped typed arrays.
///
/// This is a pure data structure abstraction with no assumptions about:
/// - File naming conventions
/// - Directory structure
/// - Data organization
///
/// Can be initialized from:
/// 1. File handle directly (most flexible, zero I/O assumptions)
/// 2. File path (convenience wrapper)
///
/// Supports both read-only and writable mappings.
///
/// Use Cases:
/// - Load large arrays without RAM constraints
/// - Zero-copy file access
/// - Write directly to disk-backed memory
/// - Building block for complex data structures (matrices, graphs, etc.)
///
/// Thread Safety:
/// - Read-only mode: Safe for concurrent access
/// - Writable mode: User must synchronize writes
/// - write_from(): Serial by design (no nested parallelism)
///
/// @tparam T Element type (must be trivially copyable)
///
/// Satisfies ArrayLike concept for seamless integration with SCL utilities.
template <typename T>
class MappedArray {
public:
    using value_type = T;
private:
    void* _ptr;
    Size _num_elements;
    Size _byte_size;
    bool _writable;
    
    SCL_FileHandle _file_handle;
    SCL_MapHandle _map_handle;

public:
    // -------------------------------------------------------------------------
    // Construction & Destruction
    // -------------------------------------------------------------------------

    /// @brief Default constructor (empty mapping).
    constexpr MappedArray() noexcept
        : _ptr(nullptr), _num_elements(0), _byte_size(0), _writable(false),
          _file_handle(SCL_INVALID_FILE_HANDLE), _map_handle(SCL_INVALID_MAP_HANDLE)
    {}

    /// @brief Construct from file path (convenience, read-only).
    ///
    /// Opens file, queries size, and creates read-only mapping in one step.
    ///
    /// @param path File path (absolute or relative)
    /// @param writable If true, create writable mapping
    ///
    /// @throws IOError if file cannot be opened or mapped
    /// @throws ValueError if file size is not aligned to element size
    explicit MappedArray(const std::string& path, bool writable = false)
        : MappedArray()
    {
        _writable = writable;
        
        Size file_size = 0;
        SCL_MMAP_OPEN_FILE(path.c_str(), _file_handle, file_size);
        
        if (_file_handle == SCL_INVALID_FILE_HANDLE) {
            throw IOError("Failed to open file: " + path);
        }
        
        if (file_size == 0) {
            return;
        }
        
        if (file_size % sizeof(T) != 0) {
            SCL_MMAP_CLOSE(_ptr, _byte_size, _map_handle, _file_handle);
            throw ValueError("File size not aligned to element size");
        }
        
        if (_writable) {
            SCL_MMAP_CREATE_WRITABLE(_file_handle, file_size, _map_handle, _ptr);
        } else {
            SCL_MMAP_CREATE(_file_handle, file_size, _map_handle, _ptr);
        }
        
        if (!_ptr) {
            SCL_MMAP_CLOSE(_ptr, _byte_size, _map_handle, _file_handle);
            throw IOError("Failed to create mapping for: " + path);
        }
        
        _byte_size = file_size;
        _num_elements = file_size / sizeof(T);
        
        SCL_MMAP_ADVISE_SEQUENTIAL(_ptr, _byte_size);
        SCL_MMAP_ADVISE_HUGEPAGE(_ptr, _byte_size);
    }

    /// @brief Construct from file handle (pure, zero file system assumptions).
    ///
    /// Takes ownership of the file handle. Most flexible initialization method.
    ///
    /// @param file_handle Open file handle (must be valid)
    /// @param file_size Size of file in bytes
    /// @param writable If true, create writable mapping
    ///
    /// @throws ValueError if handle is invalid or size is misaligned
    /// @throws IOError if mapping creation fails
    MappedArray(SCL_FileHandle file_handle, Size file_size, bool writable = false)
        : MappedArray()
    {
        if (file_handle == SCL_INVALID_FILE_HANDLE) {
            throw ValueError("Invalid file handle");
        }
        
        _writable = writable;
        
        if (file_size == 0) {
            _file_handle = file_handle;
            return;
        }
        
        if (file_size % sizeof(T) != 0) {
            throw ValueError("File size not aligned to element size");
        }
        
        _file_handle = file_handle;
        
        if (_writable) {
            SCL_MMAP_CREATE_WRITABLE(_file_handle, file_size, _map_handle, _ptr);
        } else {
            SCL_MMAP_CREATE(_file_handle, file_size, _map_handle, _ptr);
        }
        
        if (!_ptr) {
            throw IOError("Failed to create mapping from handle");
        }
        
        _byte_size = file_size;
        _num_elements = file_size / sizeof(T);
        
        SCL_MMAP_ADVISE_SEQUENTIAL(_ptr, _byte_size);
        SCL_MMAP_ADVISE_HUGEPAGE(_ptr, _byte_size);
    }

    /// @brief Destructor - automatically unmaps memory and closes handles.
    ~MappedArray() noexcept {
        SCL_MMAP_CLOSE(_ptr, _byte_size, _map_handle, _file_handle);
    }

    // -------------------------------------------------------------------------
    // Move Semantics (Transfer Ownership)
    // -------------------------------------------------------------------------

    /// @brief Move constructor.
    MappedArray(MappedArray&& other) noexcept
        : _ptr(other._ptr), _num_elements(other._num_elements), _byte_size(other._byte_size),
          _writable(other._writable), _file_handle(other._file_handle), _map_handle(other._map_handle)
    {
        other._ptr = nullptr;
        other._num_elements = 0;
        other._byte_size = 0;
        other._writable = false;
        other._file_handle = SCL_INVALID_FILE_HANDLE;
        other._map_handle = SCL_INVALID_MAP_HANDLE;
    }

    /// @brief Move assignment.
    MappedArray& operator=(MappedArray&& other) noexcept {
        if (this != &other) {
            SCL_MMAP_CLOSE(_ptr, _byte_size, _map_handle, _file_handle);
            
            _ptr = other._ptr;
            _num_elements = other._num_elements;
            _byte_size = other._byte_size;
            _writable = other._writable;
            _file_handle = other._file_handle;
            _map_handle = other._map_handle;
            
            other._ptr = nullptr;
            other._num_elements = 0;
            other._byte_size = 0;
            other._writable = false;
            other._file_handle = SCL_INVALID_FILE_HANDLE;
            other._map_handle = SCL_INVALID_MAP_HANDLE;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Disable Copy (Non-Copyable Resource)
    // -------------------------------------------------------------------------

    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// @brief Get raw pointer to mapped memory.
    SCL_NODISCARD const T* data() const noexcept {
        return static_cast<const T*>(_ptr);
    }

    /// @brief Get number of elements.
    SCL_NODISCARD Size size() const noexcept {
        return _num_elements;
    }

    /// @brief Get total size in bytes.
    SCL_NODISCARD Size byte_size() const noexcept {
        return _byte_size;
    }

    /// @brief Check if array is empty.
    SCL_NODISCARD bool empty() const noexcept {
        return _num_elements == 0;
    }

    /// @brief Array subscript operator (with debug bounds checking).
    SCL_NODISCARD const T& operator[](Index i) const noexcept {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && static_cast<Size>(i) < _num_elements,
                   "MappedArray: Index out of bounds");
#endif
        return data()[i];
    }

    // -------------------------------------------------------------------------
    // Iterator Interface (STL-Compatible)
    // -------------------------------------------------------------------------

    SCL_NODISCARD const T* begin() const noexcept { return data(); }
    SCL_NODISCARD const T* end() const noexcept { return data() + _num_elements; }

    // -------------------------------------------------------------------------
    // Writability Query
    // -------------------------------------------------------------------------

    /// @brief Check if mapping is writable.
    SCL_NODISCARD bool is_writable() const noexcept {
        return _writable;
    }

    // -------------------------------------------------------------------------
    // Write Operations (Writable Mappings Only)
    // -------------------------------------------------------------------------

    /// @brief Write data from memory into mapped region (SIMD-optimized, serial).
    ///
    /// Copies data from source memory to mapped memory region.
    ///
    /// Design Constraints:
    /// - Serial by design (no nested parallelism to avoid thread contention)
    /// - SIMD-optimized for maximum single-threaded throughput
    /// - Optional async mode (returns immediately, writes in background)
    ///
    /// Performance:
    /// - Sequential write: ~10-20 GB/s (SIMD)
    /// - Async write: Non-blocking return, OS handles flush
    ///
    /// @param source Source memory to copy from
    /// @param count Number of elements to write (0 = all)
    /// @param offset Offset in mapped array to start writing
    /// @param async If true, return immediately without waiting for flush
    ///
    /// @throws ValueError if not writable or bounds check fails
    void write_from(
        const T* source,
        Size count = 0,
        Size offset = 0,
        bool async = false
    ) {
        SCL_CHECK_ARG(_writable, "MappedArray: Cannot write to read-only mapping");
        
        if (count == 0) {
            count = _num_elements;
        }
        
        SCL_CHECK_ARG(offset + count <= _num_elements, "MappedArray: Write bounds exceeded");
        
        if (count == 0) return;
        
        T* dest = static_cast<T*>(_ptr) + offset;
        
        // Use memcpy (compiler auto-vectorizes, often faster than hand-written SIMD)
        // This is serial by design (no parallel_for to avoid nested parallelism)
        std::memcpy(dest, source, count * sizeof(T));
        
        // Sync to disk
        if (async) {
            SCL_MMAP_SYNC_ASYNC(_ptr, _byte_size);
        } else {
            SCL_MMAP_SYNC(_ptr, _byte_size);
        }
    }

    /// @brief Async write with future (non-blocking).
    ///
    /// Launches write operation in separate thread.
    /// Returns immediately with future for synchronization.
    ///
    /// @param source Source memory
    /// @param count Number of elements
    /// @param offset Offset in mapped array
    /// @return Future that completes when write finishes
    std::future<void> write_from_async(
        const T* source,
        Size count = 0,
        Size offset = 0
    ) {
        return std::async(std::launch::async, [this, source, count, offset]() {
            write_from(source, count, offset, true);
        });
    }

    /// @brief Flush changes to disk (force sync).
    ///
    /// Ensures all pending writes are persisted to disk.
    /// Blocks until OS confirms write completion.
    void flush() {
        SCL_CHECK_ARG(_writable, "MappedArray: Cannot flush read-only mapping");
        SCL_MMAP_SYNC(_ptr, _byte_size);
    }

    // -------------------------------------------------------------------------
    // Memory Management Hints
    // -------------------------------------------------------------------------

    /// @brief Prefetch data into page cache (async hint to OS).
    void prefetch() const noexcept {
        SCL_MMAP_ADVISE_WILLNEED(_ptr, _byte_size);
    }

    /// @brief Drop pages from cache to free memory.
    void drop_cache() const noexcept {
        SCL_MMAP_ADVISE_DONTNEED(_ptr, _byte_size);
    }

    /// @brief Advise OS that access will be random.
    void advise_random() const noexcept {
        SCL_MMAP_ADVISE_RANDOM(_ptr, _byte_size);
    }

    /// @brief Advise OS that access will be sequential.
    void advise_sequential() const noexcept {
        SCL_MMAP_ADVISE_SEQUENTIAL(_ptr, _byte_size);
    }
};

// =============================================================================
// Concept Verification (Compile-Time Checks)
// =============================================================================

// Verify that MappedArray satisfies ArrayLike concept
static_assert(ArrayLike<MappedArray<Real>>, "MappedArray must satisfy ArrayLike concept");
static_assert(ArrayLike<MappedArray<Index>>, "MappedArray<Index> must satisfy ArrayLike concept");

// =============================================================================
// Type Aliases
// =============================================================================

/// @brief Memory-mapped float32 array.
using MappedArrayF32 = MappedArray<float>;

/// @brief Memory-mapped float64 array.
using MappedArrayF64 = MappedArray<double>;

/// @brief Memory-mapped int64 array.
using MappedArrayI64 = MappedArray<Index>;

/// @brief Memory-mapped byte array.
using MappedArrayU8 = MappedArray<Byte>;

} // namespace scl::io
