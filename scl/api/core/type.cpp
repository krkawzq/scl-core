/// @file scl/api/core/type.cpp
/// @brief Implementation of SCL C-API type query functions

#include "type.h"
#include "scl/core/type.hpp"

#include <cstdint>

// =============================================================================
// C-ABI Export Macros
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef SCL_BUILDING_LIBRARY
        #define SCL_API extern "C" __declspec(dllexport)
    #else
        #define SCL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define SCL_API extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// Value Type Query Functions
// =============================================================================

SCL_API
auto scl_value_type_category(scl_value_type_t type) -> std::int32_t {
    // Extract category from bits 4-5
    std::int32_t category_bits = (static_cast<std::int32_t>(type) >> 4) & 0x03;
    
    switch (category_bits) {
        case 0x00: return 0;  // Real
        case 0x01: return 1;  // Int
        case 0x02: return 2;  // Uint
        default:   return -1; // Invalid
    }
}

SCL_API
auto scl_value_type_sizeof(scl_value_type_t type) -> std::int32_t {
    // Extract size from bits 0-3
    std::int32_t size_bits = static_cast<std::int32_t>(type) & 0x0F;
    
    // Size bits directly encode the byte size (1, 2, 4, 8)
    switch (size_bits) {
        case 1:
        case 2:
        case 4:
        case 8:
            return size_bits;
        default:
            return -1;  // Invalid
    }
}

SCL_API
auto scl_value_type_name(scl_value_type_t type) -> const char* {
    switch (type) {
        // Real types
        case SCL_REAL32:  return "Real32";
        case SCL_REAL64:  return "Real64";
        
        // Int types
        case SCL_INT8:    return "Int8";
        case SCL_INT16:   return "Int16";
        case SCL_INT32:   return "Int32";
        case SCL_INT64:   return "Int64";
        
        // Uint types
        case SCL_UINT8:   return "Uint8";
        case SCL_UINT16:  return "Uint16";
        case SCL_UINT32:  return "Uint32";
        case SCL_UINT64:  return "Uint64";
        
        default:          return "Unknown";
    }
}

SCL_API
auto scl_value_type_is_real(scl_value_type_t type) -> std::int32_t {
    return scl_value_type_category(type) == 0 ? 1 : 0;
}

SCL_API
auto scl_value_type_is_int(scl_value_type_t type) -> std::int32_t {
    return scl_value_type_category(type) == 1 ? 1 : 0;
}

SCL_API
auto scl_value_type_is_uint(scl_value_type_t type) -> std::int32_t {
    return scl_value_type_category(type) == 2 ? 1 : 0;
}
