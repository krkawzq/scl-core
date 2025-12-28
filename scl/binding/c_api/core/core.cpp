// =============================================================================
// FILE: scl/binding/c_api/core.cpp
// BRIEF: Core C API error handling implementation
// =============================================================================

#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/error.hpp"

#include <cstring>
#include <exception>

namespace scl::binding {

// =============================================================================
// Thread-Local Error State
// =============================================================================

namespace {
    thread_local scl_error_t last_error_code = SCL_OK;
    thread_local char last_error_message[512] = {0};
}

void set_last_error(scl_error_t code, const char* message) {
    last_error_code = code;
    if (message) {
        std::strncpy(last_error_message, message, sizeof(last_error_message) - 1);
        last_error_message[sizeof(last_error_message) - 1] = '\0';
    } else {
        last_error_message[0] = '\0';
    }
}

void clear_last_error() {
    last_error_code = SCL_OK;
    last_error_message[0] = '\0';
}

const char* get_last_error_message() {
    return last_error_message[0] != '\0' ? last_error_message : "No error";
}

// =============================================================================
// Exception to Error Code Conversion
// =============================================================================

scl_error_t handle_exception() {
    try {
        throw;
    } catch (const DimensionError& e) {
        set_last_error(SCL_ERROR_DIMENSION_MISMATCH, e.what());
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const ValueError& e) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, e.what());
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const TypeError& e) {
        set_last_error(SCL_ERROR_TYPE_MISMATCH, e.what());
        return SCL_ERROR_TYPE_MISMATCH;
    } catch (const InternalError& e) {
        set_last_error(SCL_ERROR_INTERNAL, e.what());
        return SCL_ERROR_INTERNAL;
    } catch (const Exception& e) {
        set_last_error(SCL_ERROR_UNKNOWN, e.what());
        return SCL_ERROR_UNKNOWN;
    } catch (const std::bad_alloc&) {
        set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Out of memory");
        return SCL_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_last_error(SCL_ERROR_UNKNOWN, e.what());
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        set_last_error(SCL_ERROR_UNKNOWN, "Unknown exception");
        return SCL_ERROR_UNKNOWN;
    }
}

} // namespace scl::binding

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

const char* scl_get_last_error(void) {
    return scl::binding::get_last_error_message();
}

void scl_clear_error(void) {
    scl::binding::clear_last_error();
}

} // extern "C"
