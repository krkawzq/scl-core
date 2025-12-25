#include "scl/error.hpp"
#include <mutex>
#include <vector>
#include <stdexcept>
#include <memory>

namespace scl {
namespace error {

// =============================================================================
// Error Registry Implementation
// =============================================================================

class ErrorRegistryImpl {
private:
    std::vector<ErrorInstance> errors_;
    std::mutex mutex_;
    static constexpr size_t MAX_ERRORS = 1024;
    
public:
    static ErrorRegistryImpl& instance() {
        static ErrorRegistryImpl registry;
        return registry;
    }
    
    const ErrorInstance* register_error(ErrorCode code, const char* message) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Prevent unbounded growth
        if (errors_.size() >= MAX_ERRORS) {
            errors_.clear();
        }
        
        errors_.emplace_back(code, message);
        return &errors_.back();
    }
    
    const ErrorInstance* get_error(const ErrorInstance* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Validate pointer is within our storage
        for (const auto& err : errors_) {
            if (&err == ptr) {
                return ptr;
            }
        }
        
        return nullptr;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        errors_.clear();
    }
};

// =============================================================================
// Public API Implementation
// =============================================================================

const ErrorInstance* ErrorRegistry::register_error(ErrorCode code, const char* message) {
    return ErrorRegistryImpl::instance().register_error(code, message);
}

const ErrorInstance* ErrorRegistry::get_error(const ErrorInstance* ptr) {
    return ErrorRegistryImpl::instance().get_error(ptr);
}

void ErrorRegistry::clear() {
    ErrorRegistryImpl::instance().clear();
}

const ErrorInstance* exception_to_error(const std::exception& e) {
    const char* msg = e.what();
    
    // Try to identify exception type
    if (dynamic_cast<const std::invalid_argument*>(&e)) {
        return ErrorRegistry::register_error(ErrorCode::INVALID_ARGUMENT, msg);
    } else if (dynamic_cast<const std::bad_alloc*>(&e)) {
        return ErrorRegistry::register_error(ErrorCode::OUT_OF_MEMORY, msg);
    } else if (dynamic_cast<const std::runtime_error*>(&e)) {
        return ErrorRegistry::register_error(ErrorCode::RUNTIME_ERROR, msg);
    } else {
        return ErrorRegistry::register_error(ErrorCode::UNKNOWN_ERROR, msg);
    }
}

const ErrorInstance* unknown_exception_to_error() {
    return ErrorRegistry::register_error(
        ErrorCode::UNKNOWN_ERROR,
        "Unknown C++ exception caught"
    );
}

} // namespace error
} // namespace scl

// =============================================================================
// C-ABI Implementation
// =============================================================================

extern "C" {

scl_error_code_t scl_error_get_code(scl_error_t err) {
    if (err == nullptr) {
        return static_cast<scl_error_code_t>(scl::error::ErrorCode::SUCCESS);
    }
    return static_cast<scl_error_code_t>(err->code);
}

const char* scl_error_get_message(scl_error_t err) {
    if (err == nullptr) {
        return "";
    }
    return err->message;
}

} // extern "C"

