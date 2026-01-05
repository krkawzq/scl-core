/**
 * @file scl/c_api/core/type.cpp
 * @brief Implementation of SCL C-API type query functions
 *
 * This file implements the non-inline type query functions declared in
 * scl/include/core/type.h, including:
 *   - Runtime platform support detection
 *   - Type name string conversion
 *
 * @note Uses scl::type system to query backend capabilities
 */

#include "scl/include/core/type.h"
#include "scl/core/type.hpp"

extern "C" {

// =============================================================================
// Platform Support Detection
// =============================================================================

int scl_is_type_supported(scl_value_type_t type) {
  switch (type) {
    case SCL_REAL16:
      return scl::kHasReal16Support ? 1 : 0;
    case SCL_REAL32:
    case SCL_REAL64:
    case SCL_INT8:
    case SCL_INT16:
    case SCL_INT32:
    case SCL_INT64:
    case SCL_UINT8:
    case SCL_UINT16:
    case SCL_UINT32:
    case SCL_UINT64:
      return 1;  // Always supported
    case SCL_REAL128:
      return scl::kHasReal128Support ? 1 : 0;
    default:
      return 0;
  }
}

// =============================================================================
// Type Name Strings
// =============================================================================

const char* scl_value_type_name(scl_value_type_t type) {
  switch (type) {
    case SCL_REAL16:
      return "Real16";
    case SCL_REAL32:
      return "Real32";
    case SCL_REAL64:
      return "Real64";
    case SCL_REAL128:
      return "Real128";
    case SCL_INT8:
      return "Int8";
    case SCL_INT16:
      return "Int16";
    case SCL_INT32:
      return "Int32";
    case SCL_INT64:
      return "Int64";
    case SCL_UINT8:
      return "UInt8";
    case SCL_UINT16:
      return "UInt16";
    case SCL_UINT32:
      return "UInt32";
    case SCL_UINT64:
      return "UInt64";
    default:
      return "Unknown";
  }
}

const char* scl_index_type_name(scl_index_type_t type) {
  switch (type) {
    case SCL_INDEX32:
      return "Index32";
    case SCL_INDEX64:
      return "Index64";
    default:
      return "Unknown";
  }
}

}  // extern "C"

