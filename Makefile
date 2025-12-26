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
.PHONY: python-install python-dev python-test python-clean
.PHONY: test test-python test-cpp
.PHONY: debug release

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
# Python Package Targets (using uv)
# =============================================================================

python-install:
	@echo "Installing Python package (development mode with uv)..."
	@uv pip install -e .
	@echo "Python package installed!"

python-dev:
	@echo "Installing Python package with dev dependencies..."
	@uv pip install -e ".[dev]"
	@echo "Python dev environment ready!"

python-test:
	@echo "Running Python tests..."
	@uv run pytest tests/python/ -v --tb=short
	@echo "Python tests complete!"

python-test-quick:
	@echo "Running Python tests (quick mode)..."
	@uv run pytest tests/python/ -v --tb=short -x
	@echo "Python tests complete!"

python-clean:
	@echo "Cleaning Python build artifacts..."
	@rm -rf src/*.egg-info
	@rm -rf src/__pycache__
	@rm -rf src/*/__pycache__
	@rm -rf src/*/*/__pycache__
	@rm -rf .pytest_cache
	@rm -rf build/lib*
	@rm -rf build/temp*
	@rm -rf dist/
	@echo "Python artifacts cleaned!"

# Alternative: Use system python (fallback)
python-install-system:
	@echo "Installing with system pip..."
	@python3 -m pip install -e . --user
	@echo "Python package installed!"

python-test-system:
	@echo "Running tests with system python..."
	@python3 -m pytest tests/python/ -v --tb=short
	@echo "Python tests complete!"

# =============================================================================
# Testing Targets
# =============================================================================

test: test-python
	@echo "All tests complete!"

test-python: python-test

test-cpp:
	@echo "Running C++ tests..."
	@if [ -f $(BUILD_DIR)/tests/test_runner ]; then \
		$(BUILD_DIR)/tests/test_runner; \
	else \
		echo "No C++ tests found. Build with -DBUILD_TESTING=ON"; \
	fi

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
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  SCL Core Makefile (Ninja Backend + Python)               ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "C++ Build Targets:"
	@echo "  all              - Configure and build (default)"
	@echo "  configure        - Run CMake configuration with Ninja"
	@echo "  build            - Build the C++ library using Ninja"
	@echo "  install          - Install the C++ library"
	@echo "  clean            - Remove build directory"
	@echo "  debug            - Build in Debug mode"
	@echo "  release          - Build in Release mode"
	@echo ""
	@echo "Python Package Targets:"
	@echo "  python-install   - Install Python package (pip install -e .)"
	@echo "  python-dev       - Install with dev dependencies"
	@echo "  python-test      - Run Python test suite"
	@echo "  python-clean     - Clean Python build artifacts"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test             - Run all tests (Python + C++)"
	@echo "  test-python      - Run Python tests only"
	@echo "  test-cpp         - Run C++ tests only"
	@echo ""
	@echo "Utility Targets:"
	@echo "  compile_commands - Generate compile_commands.json"
	@echo "  cloc             - Count lines of code (respects .gitignore)"
	@echo "  tree             - Display directory tree (respects .gitignore)"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Threading Backend Targets:"
	@echo "  backend-serial   - Build with serial backend (no threading)"
	@echo "  backend-bs       - Build with BS threading backend"
	@echo "  backend-openmp   - Build with OpenMP backend (default)"
	@echo "  backend-tbb      - Build with TBB backend"
	@echo ""
	@echo "Environment Variables:"
	@echo "  BUILD_DIR        - Build directory (default: build)"
	@echo "  CMAKE_FLAGS      - Additional CMake flags"
	@echo "  CMAKE_GENERATOR  - CMake generator (default: Ninja)"
	@echo ""
	@echo "Quick Start Examples:"
	@echo "  make                          # Build C++ library"
	@echo "  make python-dev               # Setup Python dev environment"
	@echo "  make test                     # Run all tests"
	@echo "  make debug                    # Debug build"
	@echo "  make compile_commands         # For IDE integration"
	@echo ""
	@echo "Complete Workflow:"
	@echo "  make clean                    # Clean previous builds"
	@echo "  make release                  # Build optimized C++ lib"
	@echo "  make python-install           # Install Python package"
	@echo "  make test                     # Verify everything works"

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

# =============================================================================
# Formatting and Linting
# =============================================================================

format-cpp:
	@echo "Formatting C++ code with clang-format..."
	@find scl -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i
	@echo "C++ code formatted!"

format-python:
	@echo "Formatting Python code with black..."
	@black src/ tests/python/
	@echo "Python code formatted!"

format: format-cpp format-python
	@echo "All code formatted!"

lint-python:
	@echo "Linting Python code..."
	@flake8 src/ tests/python/ --max-line-length=100
	@echo "Python lint complete!"

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Generating documentation..."
	@if [ -d docs ]; then \
		cd docs && make html; \
	else \
		echo "Warning: docs directory not found"; \
	fi

docs-serve:
	@echo "Serving documentation..."
	@python3 -m http.server --directory docs/_build/html 8000

# =============================================================================
# Benchmarking
# =============================================================================

benchmark:
	@echo "Running benchmarks..."
	@if [ -f $(BUILD_DIR)/benchmarks/benchmark_runner ]; then \
		$(BUILD_DIR)/benchmarks/benchmark_runner; \
	else \
		echo "No benchmarks found. Build with -DBUILD_BENCHMARKS=ON"; \
	fi

benchmark-python:
	@echo "Running Python benchmarks..."
	@python3 -m pytest tests/benchmarks/ -v --benchmark-only

# =============================================================================
# Full Clean (Nuclear Option)
# =============================================================================

distclean: clean python-clean
	@echo "Performing deep clean..."
	@rm -rf compile_commands.json
	@rm -rf .cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@echo "Deep clean complete!"

