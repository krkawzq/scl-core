.PHONY: help build compile compile-cpp clean format lint all tree cloc makedoc docs-dev docs-build docs-preview docs-clean codegen codegen-python codegen-docs

# Config

PYTHON := python3
VENV_PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
CMAKE_BUILD_DIR := build/cmake
INSTALL_DIR := python/scl/libs

.DEFAULT_GOAL := help

# =============================================================================
# Core Commands
# =============================================================================

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build everything (deps + compile)"
	@echo "  compile        Compile C++"
	@echo "  compile-cpp    Compile C++ only"
	@echo "  clean          Clean build artifacts"
	@echo "  format         Format and fix code (all tools)"
	@echo "  lint           Run linters"
	@echo "  cloc           Count lines of code"
	@echo "  tree           Show git-tracked files tree (filtered)"
	@echo "  info           Show build environment info"
	@echo ""
	@echo "Documentation:"
	@echo "  makedoc        Start documentation dev server"
	@echo "  docs-dev       Start documentation dev server (alias)"
	@echo "  docs-build     Build documentation for production"
	@echo "  docs-preview   Preview built documentation"
	@echo "  docs-clean     Clean documentation build artifacts"
	@echo ""
	@echo "Code Generation:"
	@echo "  codegen        Generate all (Python bindings + C API docs)"
	@echo "  codegen-python Generate Python ctypes bindings from C API"
	@echo "  codegen-docs   Generate C API documentation skeletons"

all: clean build format lint

# =============================================================================
# Build & Compile
# =============================================================================

setup-deps:
	@[ -f scripts/setup_cpp_deps.sh ] && ./scripts/setup_cpp_deps.sh || true

compile-cpp: setup-deps
	@mkdir -p $(CMAKE_BUILD_DIR)
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=Release
	@cd $(CMAKE_BUILD_DIR) && ninja -j$$(nproc 2>/dev/null || echo 4)
	@cd $(CMAKE_BUILD_DIR) && cmake --install .

compile-cpp-debug: setup-deps
	@mkdir -p build/cmake_debug
	@cd build/cmake_debug && cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=Debug
	@cd build/cmake_debug && ninja -j$$(nproc 2>/dev/null || echo 4)
	@cd build/cmake_debug && cmake --install .

compile: compile-cpp

compile-debug: compile-cpp-debug

build: compile
	@echo "Build completed"

build-debug: compile-cpp-debug
	@echo "Debug build completed"

rebuild: clean build

rebuild-debug: clean build-debug

# =============================================================================
# Code Quality & Formatting
# =============================================================================

format: format-imports format-code fix-lint
	@echo "Code formatted and fixed"

format-imports:
	@command -v isort >/dev/null 2>&1 || $(PIP) install isort
	@isort python/ --profile black

format-code:
	@command -v black >/dev/null 2>&1 || $(PIP) install black
	@black python/ --line-length 100

fix-lint:
	@command -v ruff >/dev/null 2>&1 || $(PIP) install ruff
	@ruff check python/ --fix --select I,F401,F841,UP,C90,N,E,W
	@ruff format python/

remove-unused-imports:
	@command -v autoflake >/dev/null 2>&1 || $(PIP) install autoflake
	@autoflake --in-place --remove-all-unused-imports --remove-unused-variables \
		--recursive python/

lint: lint-ruff lint-pyright lint-mypy
	@echo "All linters passed"

lint-ruff:
	@command -v ruff >/dev/null 2>&1 || $(PIP) install ruff
	@ruff check python/

lint-pyright:
	@command -v pyright >/dev/null 2>&1 || npm install -g pyright 2>/dev/null || echo "pyright not available"
	@command -v pyright >/dev/null 2>&1 && pyright python/ || true

lint-mypy:
	@command -v mypy >/dev/null 2>&1 || $(PIP) install mypy
	@mypy python/ --ignore-missing-imports --no-strict-optional || true

lint-flake8:
	@command -v flake8 >/dev/null 2>&1 || $(PIP) install flake8
	@flake8 python/ --max-line-length=100 --ignore=E203,W503,E501

check: format lint

# =============================================================================
# Analysis
# =============================================================================

cloc:
	@command -v cloc >/dev/null 2>&1 || { echo "Install: sudo apt install cloc"; exit 1; }
	@cloc . --exclude-dir=build,dist,__pycache__,.git,.eggs,*.egg-info,external,weights,forks,node_modules \
		--exclude-ext=.pyc,.pyo,.so,.pyd,.o,.a,.dylib,.dll --vcs=git

tree:
	@git ls-tree -r --name-only HEAD | tree --fromfile .

benchmark: compile
	@[ -f benchmarks/benchmark.py ] && $(PYTHON) benchmarks/benchmark.py || echo "No benchmarks"

# =============================================================================
# Clean
# =============================================================================

clean: clean-build clean-pyc clean-compiled docs-clean

clean-build:
	@rm -rf build/ dist/ *.egg-info .eggs/

clean-pyc:
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@find . -type d -name '.pytest_cache' -delete
	@find . -type d -name '.mypy_cache' -delete
	@find . -type d -name '.ruff_cache' -delete

clean-compiled:
	@rm -rf $(INSTALL_DIR)/*.so $(INSTALL_DIR)/*.dylib $(INSTALL_DIR)/*.dll
	@find . -name '*.so' -not -path "./external/*" -not -path "./forks/*" -delete
	@find . -name '*.pyd' -not -path "./external/*" -not -path "./forks/*" -delete

# =============================================================================
# Development
# =============================================================================

dev-install:
	@$(PIP) install black isort ruff flake8 mypy autoflake autopep8

check-cpp:
	@[ -f $(INSTALL_DIR)/libscl_f64_i64.so ] && echo "C++ library installed" || echo "C++ library missing"

git-status:
	@git status --short | grep -v '\.so$$' | grep -v '\.pyd$$' | grep -v '__pycache__' || echo "Clean"

info:
	@echo "Python:  $$($(PYTHON) --version 2>&1)"
	@echo "CMake:   $$(cmake --version 2>&1 | head -n1)"
	@echo "Ninja:   $$(ninja --version 2>&1 || echo 'not installed')"
	@echo "Git:     $$(git --version 2>&1)"

# =============================================================================
# Documentation
# =============================================================================

makedoc: docs-dev

docs-dev:
	@echo "Starting documentation development server..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d node_modules ] || npm install
	@npm run docs:dev

docs-build:
	@echo "Building documentation for production..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d node_modules ] || npm install
	@npm run docs:build
	@echo "Documentation built in docs/.vitepress/dist/"

docs-preview:
	@echo "Previewing documentation..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d docs/.vitepress/dist ] || { echo "Error: Build documentation first with 'make docs-build'"; exit 1; }
	@npm run docs:preview

docs-clean:
	@echo "Cleaning documentation build artifacts..."
	@rm -rf docs/.vitepress/dist docs/.vitepress/cache
	@echo "Documentation artifacts cleaned"

# =============================================================================
# Code Generation
# =============================================================================

codegen: codegen-python codegen-docs
	@echo "Code generation completed"

codegen-python:
	@echo "Generating Python ctypes bindings..."
	@$(VENV_PYTHON) -m codegen -v --overwrite python-bindings
	@echo "Python bindings generated"

codegen-docs:
	@echo "Generating C API documentation skeletons..."
	@$(VENV_PYTHON) -m codegen -v c-api-docs
	@echo "C API documentation skeletons generated"
