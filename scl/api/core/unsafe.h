/**
 * @file scl/api/core/unsafe.h
 * @brief Unsafe Internal Access for SCL C-API
 *
 * @warning This header exposes internal implementation details.
 *          Use at your own risk. ABI stability is NOT guaranteed.
 *          Direct manipulation of these structures is EXTREMELY DANGEROUS.
 *
 * This header provides:
 *   - scl_span_unsafe_t: Raw span structure (values/indices storage)
 *   - scl_sparse_unsafe_t: Raw sparse matrix handle structure
 *   - Direct memory access functions
 *
 * ## Enabling Unsafe Access
 *
 * Define SCL_UNSAFE_ACCESS before including this header:
 *
 *     #define SCL_UNSAFE_ACCESS
 *     #include "scl/api/core/unsafe.h"
 *
 * ## Memory Layout
 *
 * These structures are manually aligned to match C++ internal layouts.
 * Modifying them incorrectly WILL cause memory corruption.
 */

#ifndef SCL_API_CORE_UNSAFE_H_
#define SCL_API_CORE_UNSAFE_H_

#include "type.h"

#ifndef SCL_UNSAFE_ACCESS
    #error "Define SCL_UNSAFE_ACCESS before including unsafe.h to acknowledge the risks"
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: Span Unsafe Structure
 * ============================================================================ */

/**
 * @brief Ownership mode for spans
 */
typedef enum scl_span_mode_e {
    SCL_SPAN_VIEW   = 0,  /**< Non-owning view (buffer == NULL) */
    SCL_SPAN_OWNED  = 1,  /**< Exclusive ownership (buffer == sentinel) */
    SCL_SPAN_SHARED = 2   /**< Shared via SharedBuffer (buffer == valid ptr) */
} scl_span_mode_t;

/**
 * @brief Raw span structure - DANGEROUS DIRECT ACCESS
 *
 * This structure matches the memory layout of scl::SharedSpan<T>.
 * Size: 24 bytes on 64-bit systems.
 *
 * @warning DO NOT modify this structure unless you fully understand
 *          the SharedSpan ownership model. Incorrect modifications
 *          WILL cause memory leaks, double-frees, or corruption.
 *
 * Ownership determination:
 *   - buffer == NULL: View mode (non-owning, safe to copy)
 *   - buffer == OWNED_SENTINEL: Owned mode (delete[] data on destroy)
 *   - buffer != NULL && != SENTINEL: Shared mode (reference counted)
 *
 * Memory layout (must match SharedSpan<T>):
 *   offset 0:  void* buffer   (8 bytes) - SharedBuffer* or sentinel
 *   offset 8:  void* data     (8 bytes) - Pointer to T*
 *   offset 16: int64_t size   (8 bytes) - Element count
 */
typedef struct scl_span_unsafe_s {
    void*   buffer;   /**< SharedBuffer* or ownership sentinel - DO NOT MODIFY */
    void*   data;     /**< Pointer to actual data (T*) */
    int64_t size;     /**< Number of elements */
} scl_span_unsafe_t;

/** @brief Size of scl_span_unsafe_t in bytes */
#define SCL_SPAN_UNSAFE_SIZE 24

/** @brief Alignment of scl_span_unsafe_t in bytes */
#define SCL_SPAN_UNSAFE_ALIGN 8

/* ============================================================================
 * SECTION 2: Sparse Handle Unsafe Structure
 * ============================================================================ */

/**
 * @brief Sparse handle header - type metadata
 *
 * First 16 bytes of every sparse handle.
 */
typedef struct scl_sparse_header_s {
    int32_t real_type;    /**< scl_real_type_t value */
    int32_t index_type;   /**< scl_index_type_t value */
    int32_t layout;       /**< scl_layout_t value */
    int32_t _reserved;    /**< Padding for alignment */
} scl_sparse_header_t;

/** @brief Size of scl_sparse_header_t in bytes */
#define SCL_SPARSE_HEADER_SIZE 16

/**
 * @brief Raw sparse handle structure - EXTREMELY DANGEROUS DIRECT ACCESS
 *
 * This structure matches the memory layout of the internal sparse handle.
 * The variant storage follows the header and contains one of 8 possible
 * Sparse<V, I, IsCSR> matrix types.
 *
 * @warning DO NOT access or modify this structure directly unless you:
 *   1. Know exactly which variant type is active
 *   2. Understand the std::variant ABI for your compiler
 *   3. Are prepared for undefined behavior on mistakes
 *
 * Memory layout:
 *   offset 0:  scl_sparse_header_t header (16 bytes)
 *   offset 16: variant storage (size varies by platform/compiler)
 *
 * Variant indices:
 *   0: CSR<Real32, Index32>
 *   1: CSR<Real64, Index32>
 *   2: CSR<Real32, Index64>
 *   3: CSR<Real64, Index64>
 *   4: CSC<Real32, Index32>
 *   5: CSC<Real64, Index32>
 *   6: CSC<Real32, Index64>
 *   7: CSC<Real64, Index64>
 */
typedef struct scl_sparse_unsafe_s {
    scl_sparse_header_t header;  /**< Type metadata */
    
    /**
     * @brief Variant storage - DO NOT ACCESS DIRECTLY
     *
     * This is opaque storage for std::variant. The actual size and layout
     * depend on the compiler and platform. Use scl_unsafe_sparse_variant_size()
     * to get the runtime size.
     *
     * For GCC/Clang on x86_64 with standard Sparse types, this is typically
     * around 80-120 bytes, but this is NOT guaranteed.
     */
    char variant_storage[];  /* Flexible array member - actual size varies */
} scl_sparse_unsafe_t;

/** @brief Offset to variant storage within sparse handle */
#define SCL_SPARSE_VARIANT_OFFSET 16

/* ============================================================================
 * SECTION 3: Span Access Functions
 * ============================================================================ */

/**
 * @brief Get ownership mode of a span
 * @param span Pointer to span structure
 * @return Ownership mode
 */
scl_span_mode_t scl_unsafe_span_mode(const scl_span_unsafe_t* span);

/**
 * @brief Get use count for shared span
 * @param span Pointer to span structure
 * @return Use count (0 for view, 1 for owned, >1 for shared)
 */
int32_t scl_unsafe_span_use_count(const scl_span_unsafe_t* span);

/**
 * @brief Get byte offset within SharedBuffer
 * @param span Pointer to span structure
 * @return Byte offset, or 0 if not shared
 */
int64_t scl_unsafe_span_offset_bytes(const scl_span_unsafe_t* span);

/**
 * @brief Manually increment reference count (shared spans only)
 * @param span Pointer to span structure
 * @warning Only valid for shared mode spans. Will crash otherwise.
 */
void scl_unsafe_span_incref(scl_span_unsafe_t* span);

/**
 * @brief Manually decrement reference count (shared spans only)
 * @param span Pointer to span structure
 * @warning May free underlying buffer if count reaches 0
 */
void scl_unsafe_span_decref(scl_span_unsafe_t* span);

/* ============================================================================
 * SECTION 4: Sparse Handle Access Functions
 * ============================================================================ */

/**
 * @brief Cast sparse handle to unsafe structure
 * @param handle Opaque sparse handle
 * @return Pointer to unsafe structure
 */
scl_sparse_unsafe_t* scl_unsafe_sparse_cast(scl_sparse_t handle);

/**
 * @brief Get sparse handle header
 * @param handle Sparse handle
 * @return Pointer to header
 */
scl_sparse_header_t* scl_unsafe_sparse_header(scl_sparse_t handle);

/**
 * @brief Get pointer to variant storage
 * @param handle Sparse handle
 * @return Pointer to variant storage bytes
 */
void* scl_unsafe_sparse_variant_ptr(scl_sparse_t handle);

/**
 * @brief Get primary dimension (rows for CSR, cols for CSC)
 * @param handle Sparse handle
 * @return Primary dimension size
 */
int64_t scl_unsafe_sparse_primary_dim(scl_sparse_t handle);

/**
 * @brief Get secondary dimension (cols for CSR, rows for CSC)
 * @param handle Sparse handle
 * @return Secondary dimension size
 */
int64_t scl_unsafe_sparse_secondary_dim(scl_sparse_t handle);

/* ============================================================================
 * SECTION 5: Direct Data Access Functions
 * ============================================================================ */

/**
 * @brief Get all row/column data in one call
 *
 * @param handle Sparse handle
 * @param values Output array of value pointers (size = primary_dim)
 * @param indices Output array of index pointers (size = primary_dim)
 * @param lengths Output array of lengths (size = primary_dim)
 * @return 0 on success, error code otherwise
 */
int32_t scl_unsafe_sparse_get_all_rows(
    scl_sparse_t handle,
    void** values,
    void** indices,
    int64_t* lengths
);

/**
 * @brief Set row/column data directly as view - EXTREMELY DANGEROUS
 *
 * Replaces the data pointers for a row/column without copying.
 * The span will be in VIEW mode (non-owning).
 *
 * @param handle Sparse handle
 * @param idx Primary dimension index
 * @param values New values pointer
 * @param indices New indices pointer
 * @param length New length
 * @return 0 on success, error code otherwise
 *
 * @warning Caller must ensure data outlives the matrix
 * @warning May break matrix invariants (sorted indices, etc.)
 * @warning Will cause undefined behavior if types don't match
 */
int32_t scl_unsafe_sparse_set_row_view(
    scl_sparse_t handle,
    int64_t idx,
    void* values,
    void* indices,
    int64_t length
);

/* ============================================================================
 * SECTION 6: Memory Layout Query Functions
 * ============================================================================ */

/**
 * @brief Get size of internal sparse handle structure
 * @return Size in bytes
 */
size_t scl_unsafe_sparse_handle_size(void);

/**
 * @brief Get alignment of internal sparse handle
 * @return Alignment in bytes
 */
size_t scl_unsafe_sparse_handle_align(void);

/**
 * @brief Get size of variant storage
 * @return Size in bytes
 */
size_t scl_unsafe_sparse_variant_size(void);

/**
 * @brief Get offset to variant storage within handle
 * @return Offset in bytes
 */
size_t scl_unsafe_sparse_variant_offset(void);

/* ============================================================================
 * SECTION 7: Validation Helpers
 * ============================================================================ */

/**
 * @brief Validate span structure layout
 * @return 1 if layout matches expectations, 0 otherwise
 */
int32_t scl_unsafe_validate_span_layout(void);

/**
 * @brief Validate sparse handle structure layout
 * @return 1 if layout matches expectations, 0 otherwise
 */
int32_t scl_unsafe_validate_sparse_layout(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * SECTION 8: C++ Direct Access (for binding implementations)
 * ============================================================================ */

#ifdef __cplusplus

#include "scl/core/sparse.hpp"
#include "scl/core/span.hpp"

#include <variant>

namespace scl {
namespace unsafe {

/* ----------------------------------------------------------------------------
 * Type Definitions
 * ---------------------------------------------------------------------------- */

/** @brief Variant type for all sparse matrix types */
typedef std::variant<
    Sparse<Real32, Index32, true>,   /* 0: CSR<f32, i32> */
    Sparse<Real64, Index32, true>,   /* 1: CSR<f64, i32> */
    Sparse<Real32, Index64, true>,   /* 2: CSR<f32, i64> */
    Sparse<Real64, Index64, true>,   /* 3: CSR<f64, i64> */
    Sparse<Real32, Index32, false>,  /* 4: CSC<f32, i32> */
    Sparse<Real64, Index32, false>,  /* 5: CSC<f64, i32> */
    Sparse<Real32, Index64, false>,  /* 6: CSC<f32, i64> */
    Sparse<Real64, Index64, false>   /* 7: CSC<f64, i64> */
> SparseVariant;

/** @brief Internal sparse handle structure */
struct SparseHandle {
    scl_real_type_t  real_type;
    scl_index_type_t index_type;
    scl_layout_t     layout;
    SparseVariant    data;
};

/* ----------------------------------------------------------------------------
 * Unsafe Cast Functions
 * ---------------------------------------------------------------------------- */

/** @brief Cast opaque handle to internal structure (mutable) */
inline SparseHandle* cast_mut(scl_sparse_t handle) {
    return reinterpret_cast<SparseHandle*>(handle);
}

/** @brief Cast opaque handle to internal structure (const) */
inline const SparseHandle* cast_const(scl_sparse_t handle) {
    return reinterpret_cast<const SparseHandle*>(handle);
}

/** @brief Cast C span to SharedSpan (mutable) */
template<typename T>
inline SharedSpan<T>* cast_span_mut(scl_span_unsafe_t* span) {
    return reinterpret_cast<SharedSpan<T>*>(span);
}

/** @brief Cast C span to SharedSpan (const) */
template<typename T>
inline const SharedSpan<T>* cast_span_const(const scl_span_unsafe_t* span) {
    return reinterpret_cast<const SharedSpan<T>*>(span);
}

/* ----------------------------------------------------------------------------
 * Typed Access Functions
 * ---------------------------------------------------------------------------- */

/** @brief Get mutable reference to sparse matrix with specific types */
template<typename V, typename I, bool IsCSR>
inline Sparse<V, I, IsCSR>& get_mut(scl_sparse_t handle) {
    return std::get<Sparse<V, I, IsCSR>>(cast_mut(handle)->data);
}

/** @brief Get const reference to sparse matrix with specific types */
template<typename V, typename I, bool IsCSR>
inline const Sparse<V, I, IsCSR>& get_const(scl_sparse_t handle) {
    return std::get<Sparse<V, I, IsCSR>>(cast_const(handle)->data);
}

/** @brief Visit sparse matrix with mutable visitor */
template<typename Visitor>
inline auto visit_mut(scl_sparse_t handle, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), cast_mut(handle)->data);
}

/** @brief Visit sparse matrix with const visitor */
template<typename Visitor>
inline auto visit_const(scl_sparse_t handle, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), cast_const(handle)->data);
}

/* ----------------------------------------------------------------------------
 * Handle Creation/Extraction
 * ---------------------------------------------------------------------------- */

/** @brief Create handle from existing Sparse matrix (takes ownership) */
template<typename V, typename I, bool IsCSR>
inline scl_sparse_t create_handle(Sparse<V, I, IsCSR>&& matrix) {
    SparseHandle* handle = new SparseHandle{
        std::is_same_v<V, Real32> ? SCL_REAL32 : SCL_REAL64,
        std::is_same_v<I, Index32> ? SCL_INDEX32 : SCL_INDEX64,
        IsCSR ? SCL_LAYOUT_CSR : SCL_LAYOUT_CSC,
        std::move(matrix)
    };
    return reinterpret_cast<scl_sparse_t>(handle);
}

/** @brief Extract Sparse matrix from handle (destroys handle) */
template<typename V, typename I, bool IsCSR>
inline Sparse<V, I, IsCSR> extract(scl_sparse_t handle) {
    SparseHandle* h = cast_mut(handle);
    Sparse<V, I, IsCSR> result = std::move(std::get<Sparse<V, I, IsCSR>>(h->data));
    delete h;
    return result;
}

}  /* namespace unsafe */
}  /* namespace scl */

#endif  /* __cplusplus */

#endif  /* SCL_API_CORE_UNSAFE_H_ */
