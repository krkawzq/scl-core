# =============================================================================
# Makefile for scl-core
# =============================================================================

# Project configuration
PROJECT_NAME := scl-core
BUILD_DIR := build
CMAKE := cmake
CMAKE_FLAGS :=

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

.PHONY: all clean configure build install help compile_commands

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
	@echo "Building..."
	@cd $(BUILD_DIR) && $(CMAKE) --build . --parallel

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

help:
	@echo "SCL Core Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Configure and build (default)"
	@echo "  configure        - Run CMake configuration"
	@echo "  build            - Build the project"
	@echo "  install          - Install the project"
	@echo "  clean            - Remove build directory"
	@echo "  compile_commands - Generate and copy compile_commands.json"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  BUILD_DIR        - Build directory (default: build)"
	@echo "  CMAKE_FLAGS      - Additional CMake flags"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build with default settings"
	@echo "  make CMAKE_FLAGS=-DCMAKE_BUILD_TYPE=Debug  # Debug build"
	@echo "  make compile_commands         # Generate compile_commands.json"

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

