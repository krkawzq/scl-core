# =============================================================================
# SCL C API Tests - Standalone Makefile
# =============================================================================
#
# Independent build system for C API tests.
# Supports parallel development without modifying CMakeLists.txt.
#
# Usage:
#   make test-<name>          # Build and run specific test
#   make build-<name>         # Build specific test only
#   make run-<name>           # Run specific test (must be built first)
#   make build-all            # Build all tests
#   make test-all             # Build and run all tests
#   make clean                # Clean all binaries
#
# Examples:
#   make test-sparse_basic    # Build and run sparse_basic test
#   make build-demo run-demo  # Build then run demo test
#   make test-all             # Run all tests
#   make test-all VERBOSE=1   # Run with verbose output
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Directories
SRC_DIR := src
INCLUDE_DIR := include
UNITS_DIR := units
ROOT_DIR := ../..
LIBS_DIR := $(ROOT_DIR)/python/scl/libs

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic \
            -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare
CPPFLAGS := -I$(ROOT_DIR) -I$(INCLUDE_DIR)
LDFLAGS := -L$(LIBS_DIR) -Wl,-rpath,$(LIBS_DIR)
LDLIBS := -lscl_f64_i64 -ldl

# Optimization
ifdef DEBUG
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O2
endif

# Eigen3 (optional, auto-detected)
EIGEN_INCLUDE := $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I$(ROOT_DIR)/build/_deps/eigen-src")
CPPFLAGS += $(EIGEN_INCLUDE)

# Colors
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[38;5;114m
COLOR_BLUE := \033[38;5;80m
COLOR_YELLOW := \033[38;5;221m
COLOR_RED := \033[38;5;203m

# -----------------------------------------------------------------------------
# Auto-Discovery of Test Files
# -----------------------------------------------------------------------------

# Find all test_*.cpp files
TEST_SOURCES := $(wildcard $(SRC_DIR)/test_*.cpp)
TEST_NAMES := $(patsubst $(SRC_DIR)/test_%.cpp,%,$(TEST_SOURCES))
TEST_BINS := $(patsubst %,$(UNITS_DIR)/test_%,$(TEST_NAMES))

# -----------------------------------------------------------------------------
# Phony Targets
# -----------------------------------------------------------------------------

.PHONY: all build-all test-all clean help list
.PHONY: $(addprefix test-,$(TEST_NAMES))
.PHONY: $(addprefix build-,$(TEST_NAMES))
.PHONY: $(addprefix run-,$(TEST_NAMES))

# -----------------------------------------------------------------------------
# Default Target
# -----------------------------------------------------------------------------

all: build-all

# -----------------------------------------------------------------------------
# Build Rules
# -----------------------------------------------------------------------------

# Create output directory
$(UNITS_DIR):
	@mkdir -p $(UNITS_DIR)

# Generic build rule for any test
$(UNITS_DIR)/test_%: $(SRC_DIR)/test_%.cpp $(INCLUDE_DIR)/*.hpp | $(UNITS_DIR)
	@echo "$(COLOR_BLUE)Building:$(COLOR_RESET) test_$*"
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "$(COLOR_GREEN)  ✓ Built:$(COLOR_RESET) $@"

# -----------------------------------------------------------------------------
# High-Level Targets
# -----------------------------------------------------------------------------

# Build all tests
build-all: $(TEST_BINS)
	@echo ""
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✅ All tests built successfully!$(COLOR_RESET)"
	@echo "   Tests: $(words $(TEST_BINS))"
	@echo "   Output: $(UNITS_DIR)/"
	@echo ""

# Run all tests
test-all: build-all
	@echo ""
	@echo "$(COLOR_BLUE)$(COLOR_BOLD)🧪 Running all tests...$(COLOR_RESET)"
	@echo ""
	@failed=0; \
	passed=0; \
	for test in $(TEST_BINS); do \
		name=$$(basename $$test); \
		if [ -n "$(VERBOSE)" ]; then \
			echo "$(COLOR_BLUE)Running:$(COLOR_RESET) $$name"; \
			$$test --verbose && passed=$$((passed+1)) || failed=$$((failed+1)); \
		else \
			$$test --minimal > /dev/null 2>&1 && { \
				echo "$(COLOR_GREEN)  ✓$(COLOR_RESET) $$name"; \
				passed=$$((passed+1)); \
			} || { \
				echo "$(COLOR_RED)  ✗$(COLOR_RESET) $$name"; \
				failed=$$((failed+1)); \
			}; \
		fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "$(COLOR_GREEN)$(COLOR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOR_RESET)"; \
		echo "$(COLOR_GREEN)$(COLOR_BOLD)  ✅ All $$passed tests passed!$(COLOR_RESET)"; \
		echo "$(COLOR_GREEN)$(COLOR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOR_RESET)"; \
		echo ""; \
	else \
		echo "$(COLOR_RED)$(COLOR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOR_RESET)"; \
		echo "$(COLOR_RED)$(COLOR_BOLD)  ❌ $$failed test(s) failed, $$passed passed$(COLOR_RESET)"; \
		echo "$(COLOR_RED)$(COLOR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOR_RESET)"; \
		echo ""; \
		exit 1; \
	fi

# Clean
clean:
	@echo "$(COLOR_YELLOW)Cleaning test binaries...$(COLOR_RESET)"
	@rm -rf $(UNITS_DIR)
	@echo "$(COLOR_GREEN)  ✓ Clean complete$(COLOR_RESET)"

# List available tests
list:
	@echo ""
	@echo "$(COLOR_BOLD)Available tests:$(COLOR_RESET)"
	@echo ""
	@for name in $(TEST_NAMES); do \
		echo "  $(COLOR_BLUE)$$name$(COLOR_RESET)"; \
	done
	@echo ""
	@echo "$(COLOR_BOLD)Commands:$(COLOR_RESET)"
	@echo "  make test-<name>    # Build and run"
	@echo "  make build-<name>   # Build only"
	@echo "  make run-<name>     # Run only"
	@echo ""

# Help
help:
	@echo ""
	@echo "$(COLOR_BOLD)SCL C API Tests - Makefile$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Build Commands:$(COLOR_RESET)"
	@echo "  make build-all              # Build all tests"
	@echo "  make build-<name>           # Build specific test"
	@echo ""
	@echo "$(COLOR_BOLD)Test Commands:$(COLOR_RESET)"
	@echo "  make test-all               # Build and run all tests"
	@echo "  make test-all VERBOSE=1     # Run with verbose output"
	@echo "  make test-<name>            # Build and run specific test"
	@echo "  make run-<name>             # Run specific test (must be built)"
	@echo ""
	@echo "$(COLOR_BOLD)Utility Commands:$(COLOR_RESET)"
	@echo "  make list                   # List available tests"
	@echo "  make clean                  # Remove all binaries"
	@echo "  make help                   # Show this help"
	@echo ""
	@echo "$(COLOR_BOLD)Available Tests:$(COLOR_RESET)"
	@for name in $(TEST_NAMES); do \
		echo "  - $$name"; \
	done
	@echo ""
	@echo "$(COLOR_BOLD)Examples:$(COLOR_RESET)"
	@echo "  make test-demo"
	@echo "  make test-sparse_basic VERBOSE=1"
	@echo "  make build-guards run-guards"
	@echo "  make test-all"
	@echo ""
	@echo "$(COLOR_BOLD)Environment Variables:$(COLOR_RESET)"
	@echo "  DEBUG=1        # Build with debug symbols"
	@echo "  VERBOSE=1      # Run tests with verbose output"
	@echo ""

# -----------------------------------------------------------------------------
# Per-Test Targets (Auto-Generated)
# -----------------------------------------------------------------------------

# Generate build-<name> targets
$(foreach name,$(TEST_NAMES),$(eval build-$(name): $(UNITS_DIR)/test_$(name)))

# Generate run-<name> targets
define RUN_TEST_TEMPLATE
run-$(1): $(UNITS_DIR)/test_$(1)
	@echo "$(COLOR_BLUE)Running:$(COLOR_RESET) test_$(1)"
	@if [ -n "$$(VERBOSE)" ]; then \
		$$< --verbose; \
	else \
		$$<; \
	fi
endef

$(foreach name,$(TEST_NAMES),$(eval $(call RUN_TEST_TEMPLATE,$(name))))

# Generate test-<name> targets (build + run)
define TEST_TEMPLATE
test-$(1): build-$(1)
	@$$(MAKE) run-$(1) VERBOSE=$$(VERBOSE)
endef

$(foreach name,$(TEST_NAMES),$(eval $(call TEST_TEMPLATE,$(name))))

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

.PHONY: check-libs
check-libs:
	@if [ ! -f "$(LIBS_DIR)/libscl_f64_i64.so" ]; then \
		echo "$(COLOR_RED)Error:$(COLOR_RESET) Library not found: $(LIBS_DIR)/libscl_f64_i64.so"; \
		echo ""; \
		echo "Please run: cd $(ROOT_DIR) && make compile-cpp"; \
		echo ""; \
		exit 1; \
	fi

# Add check-libs as prerequisite for build-all
build-all: check-libs

# -----------------------------------------------------------------------------
# Debug Info
# -----------------------------------------------------------------------------

.PHONY: info
info:
	@echo ""
	@echo "$(COLOR_BOLD)Build Configuration:$(COLOR_RESET)"
	@echo "  CXX:        $(CXX)"
	@echo "  CXXFLAGS:   $(CXXFLAGS)"
	@echo "  CPPFLAGS:   $(CPPFLAGS)"
	@echo "  LDFLAGS:    $(LDFLAGS)"
	@echo "  LDLIBS:     $(LDLIBS)"
	@echo "  Tests:      $(words $(TEST_NAMES))"
	@echo "  Output:     $(UNITS_DIR)/"
	@echo "  Library:    $(LIBS_DIR)/libscl_f64_i64.so"
	@echo ""
	@if [ -f "$(LIBS_DIR)/libscl_f64_i64.so" ]; then \
		echo "$(COLOR_GREEN)  ✓ Library found$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_RED)  ✗ Library not found$(COLOR_RESET)"; \
	fi
	@echo ""

