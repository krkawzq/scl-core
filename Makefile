# =============================================================================
# Makefile for scl-core
# =============================================================================

# Project configuration
PROJECT_NAME := scl-core
BUILD_DIR := build
CMAKE := cmake
CMAKE_GENERATOR := Ninja
CMAKE_FLAGS := -G $(CMAKE_GENERATOR)
NINJA := ninja

# Default target
.DEFAULT_GOAL := all

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Release
endif
ifeq ($(UNAME_S),Darwin)
    CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Release
endif

# =============================================================================
# Phony Targets
# =============================================================================

.PHONY: all clean configure build install help compile_commands cloc tree

# =============================================================================
# Main Targets
# =============================================================================

all: configure build
	@echo "Build complete!"

configure:
	@echo "Configuring CMake..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. $(CMAKE_FLAGS)

build: configure
	@echo "Building with Ninja..."
	@cd $(BUILD_DIR) && $(NINJA)

install: build
	@echo "Installing..."
	@cd $(BUILD_DIR) && $(CMAKE) --install .

clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

# =============================================================================
# Utility Targets
# =============================================================================

compile_commands: configure
	@echo "Generating compile_commands.json..."
	@if [ -f $(BUILD_DIR)/compile_commands.json ]; then \
		cp $(BUILD_DIR)/compile_commands.json . && \
		echo "compile_commands.json copied to project root"; \
	else \
		echo "Warning: compile_commands.json not found in $(BUILD_DIR)"; \
	fi

cloc:
	@echo "Running cloc on project, filtering by .gitignore..."
	@if ! command -v cloc >/dev/null 2>&1; then \
		echo "Error: cloc is not installed."; \
		exit 1; \
	fi
	@if ! command -v git >/dev/null 2>&1; then \
		echo "Error: git is not installed."; \
		exit 1; \
	fi
	@git ls-files | xargs cloc

tree:
	@echo "Running tree on tracked files, displaying git repository tree structure..."
	@git ls-tree -r --name-only HEAD | tree --fromfile .   

help:
	@echo "SCL Core Makefile (Ninja Backend)"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Configure and build (default)"
	@echo "  configure        - Run CMake configuration with Ninja"
	@echo "  build            - Build the project using Ninja"
	@echo "  install          - Install the project"
	@echo "  clean            - Remove build directory"
	@echo "  compile_commands - Generate and copy compile_commands.json"
	@echo "  cloc             - Count lines of code, respects .gitignore"
	@echo "  tree             - Display directory tree, respects .gitignore"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  BUILD_DIR        - Build directory (default: build)"
	@echo "  CMAKE_FLAGS      - Additional CMake flags"
	@echo "  CMAKE_GENERATOR  - CMake generator (default: Ninja)"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build with Ninja (default)"
	@echo "  make CMAKE_FLAGS=-DCMAKE_BUILD_TYPE=Debug  # Debug build"
	@echo "  make compile_commands         # Generate compile_commands.json"
	@echo "  make cloc                     # Count lines of code"
	@echo "  make tree                     # Show directory tree"

# =============================================================================
# Development Targets
# =============================================================================

debug:
	@$(MAKE) CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug" all

release:
	@$(MAKE) CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release" all

# Threading backend selection
backend-serial:
	@$(MAKE) CMAKE_FLAGS="$(CMAKE_FLAGS) -DSCL_THREADING_BACKEND=SERIAL" configure build

backend-bs:
	@$(MAKE) CMAKE_FLAGS="$(CMAKE_FLAGS) -DSCL_THREADING_BACKEND=BS" configure build

backend-openmp:
	@$(MAKE) CMAKE_FLAGS="$(CMAKE_FLAGS) -DSCL_THREADING_BACKEND=OPENMP" configure build

backend-tbb:
	@$(MAKE) CMAKE_FLAGS="$(CMAKE_FLAGS) -DSCL_THREADING_BACKEND=TBB" configure build

