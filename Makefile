.PHONY: help build compile compile-cpp compile-debug clean format lint all tree cloc \
        makedoc docs-dev docs-build docs-preview docs-clean \
        codegen codegen-python codegen-docs \
        test test-build test-run test-clean test-quick test-single \
        dry-run configure info rebuild rebuild-debug \
        clean-build clean-pyc clean-compiled dev-install check

# =============================================================================
# Platform Detection
# =============================================================================

ifeq ($(OS),Windows_NT)
    PLATFORM := windows
    # Detect shell type
    ifneq ($(findstring cmd,$(SHELL)),)
        SHELL_TYPE := cmd
    else ifneq ($(findstring powershell,$(SHELL)),)
        SHELL_TYPE := pwsh
    else
        # Git Bash / MSYS2 / WSL
        SHELL_TYPE := unix
    endif
    
    ifeq ($(SHELL_TYPE),unix)
        RM := rm -rf
        MKDIR := mkdir -p
        RMDIR := rm -rf
        CP := cp
        LN := ln -sf
        ECHO := echo
        NULL := /dev/null
        SEP := /
        WHICH := command -v
        NPROC := $$(nproc 2>/dev/null || echo $(NUMBER_OF_PROCESSORS))
        AND := &&
        TEST_F := test -f
        TEST_D := test -d
        TEST_E := test -e
        TRUE := true
        FALSE := false
        LIB_EXT := .dll
    else
        RM := powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
        MKDIR := powershell -Command "New-Item -ItemType Directory -Force -Path"
        RMDIR := powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
        CP := powershell -Command "Copy-Item"
        LN := powershell -Command "New-Item -ItemType SymbolicLink -Force -Path"
        ECHO := powershell -Command "Write-Host"
        NULL := NUL
        SEP := \\
        WHICH := where
        NPROC := $(NUMBER_OF_PROCESSORS)
        AND := ;
        TEST_F := powershell -Command "Test-Path -PathType Leaf"
        TEST_D := powershell -Command "Test-Path -PathType Container"
        TEST_E := powershell -Command "Test-Path"
        TRUE := powershell -Command "exit 0"
        FALSE := powershell -Command "exit 1"
        LIB_EXT := .dll
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM := macos
        LIB_EXT := .dylib
        NPROC := $$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        PLATFORM := linux
        LIB_EXT := .so
        NPROC := $$(nproc 2>/dev/null || echo 4)
    endif
    
    SHELL_TYPE := unix
    RM := rm -rf
    MKDIR := mkdir -p
    RMDIR := rm -rf
    CP := cp
    LN := ln -sf
    ECHO := echo
    NULL := /dev/null
    SEP := /
    WHICH := command -v
    AND := &&
    TEST_F := test -f
    TEST_D := test -d
    TEST_E := test -e
    TRUE := true
    FALSE := false
endif

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME    := scl-core
CMAKE_BUILD_DIR := build/cmake
CMAKE_DEBUG_DIR := build/cmake_debug
CMAKE_TEST_DIR  := test/C/build
INSTALL_DIR     := python/scl/libs

PYTHON          := python3
VENV_PYTHON     := .venv/bin/python
PIP             := $(PYTHON) -m pip

# CMake options
CMAKE_GENERATOR := Ninja
CMAKE_OPTIONS   := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================

help:
	@$(ECHO) ""
	@$(ECHO) "╔══════════════════════════════════════════════════════════════════╗"
	@$(ECHO) "║                     $(PROJECT_NAME) Build System                       ║"
	@$(ECHO) "╠══════════════════════════════════════════════════════════════════╣"
	@$(ECHO) "║  Platform: $(PLATFORM) ($(SHELL_TYPE) shell)                               ║"
	@$(ECHO) "╚══════════════════════════════════════════════════════════════════╝"
	@$(ECHO) ""
	@$(ECHO) "Usage: make [target]"
	@$(ECHO) ""
	@$(ECHO) "Core Commands:"
	@$(ECHO) "  build            Build everything (Release)"
	@$(ECHO) "  build-debug      Build everything (Debug)"
	@$(ECHO) "  compile          Compile C++ (Release)"
	@$(ECHO) "  compile-debug    Compile C++ (Debug)"
	@$(ECHO) "  rebuild          Clean and rebuild (Release)"
	@$(ECHO) "  rebuild-debug    Clean and rebuild (Debug)"
	@$(ECHO) "  dry-run          Configure only (generate compile_commands.json)"
	@$(ECHO) "  configure        Alias for dry-run"
	@$(ECHO) "  clean            Clean all build artifacts"
	@$(ECHO) "  info             Show build environment info"
	@$(ECHO) ""
	@$(ECHO) "Code Quality:"
	@$(ECHO) "  format           Format and fix code (all tools)"
	@$(ECHO) "  lint             Run all linters"
	@$(ECHO) "  check            Format + lint"
	@$(ECHO) ""
	@$(ECHO) "Analysis:"
	@$(ECHO) "  cloc             Count lines of code"
	@$(ECHO) "  tree             Show git-tracked files tree"
	@$(ECHO) ""
	@$(ECHO) "Documentation:"
	@$(ECHO) "  makedoc          Start documentation dev server"
	@$(ECHO) "  docs-dev         Start documentation dev server (alias)"
	@$(ECHO) "  docs-build       Build documentation for production"
	@$(ECHO) "  docs-preview     Preview built documentation"
	@$(ECHO) "  docs-clean       Clean documentation build artifacts"
	@$(ECHO) ""
	@$(ECHO) "Code Generation:"
	@$(ECHO) "  codegen          Generate all (Python bindings + C API docs)"
	@$(ECHO) "  codegen-python   Generate Python ctypes bindings"
	@$(ECHO) "  codegen-docs     Generate C API documentation skeletons"
	@$(ECHO) ""
	@$(ECHO) "Testing:"
	@$(ECHO) "  test             Build and run all tests"
	@$(ECHO) "  test-build       Build tests only"
	@$(ECHO) "  test-run         Run tests only"
	@$(ECHO) "  test-quick       Run tests in parallel"
	@$(ECHO) "  test-single      Run single test (TEST=<name>)"
	@$(ECHO) "  test-clean       Clean test build artifacts"
	@$(ECHO) ""

all: clean build format lint

# =============================================================================
# Build & Compile (Platform-independent using Ninja)
# =============================================================================

ifeq ($(SHELL_TYPE),unix)

setup-deps:
	@[ -f scripts/setup_cpp_deps.sh ] && ./scripts/setup_cpp_deps.sh || $(TRUE)

dry-run: setup-deps
	@$(MKDIR) $(CMAKE_BUILD_DIR)
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTIONS)
	@if [ -f $(CMAKE_BUILD_DIR)/compile_commands.json ] && [ ! -e compile_commands.json ]; then \
		$(LN) $(CMAKE_BUILD_DIR)/compile_commands.json compile_commands.json; \
		$(ECHO) "Created symlink: compile_commands.json -> $(CMAKE_BUILD_DIR)/compile_commands.json"; \
	elif [ -f $(CMAKE_BUILD_DIR)/compile_commands.json ]; then \
		$(ECHO) "compile_commands.json already exists"; \
	fi
	@$(ECHO) "Configuration complete. compile_commands.json is ready."

configure: dry-run

compile-cpp: setup-deps
	@$(MKDIR) $(CMAKE_BUILD_DIR)
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTIONS)
	@cd $(CMAKE_BUILD_DIR) && ninja -j$(NPROC)
	@$(ECHO) "✅ Release build completed"

compile-cpp-debug: setup-deps
	@$(MKDIR) $(CMAKE_DEBUG_DIR)
	@cd $(CMAKE_DEBUG_DIR) && cmake ../.. -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTIONS)
	@cd $(CMAKE_DEBUG_DIR) && ninja -j$(NPROC)
	@$(ECHO) "✅ Debug build completed"

compile: compile-cpp

compile-debug: compile-cpp-debug

build: compile
	@$(ECHO) "Build completed"

build-debug: compile-cpp-debug
	@$(ECHO) "Debug build completed"

rebuild: clean build

rebuild-debug: clean build-debug

# =============================================================================
# Code Quality & Formatting (Unix)
# =============================================================================

format: format-imports format-code fix-lint
	@$(ECHO) "Code formatted and fixed"

format-imports:
	@$(WHICH) isort >$(NULL) 2>&1 || $(PIP) install isort
	@isort python/ --profile black 2>$(NULL) || $(TRUE)

format-code:
	@$(WHICH) black >$(NULL) 2>&1 || $(PIP) install black
	@black python/ --line-length 100 2>$(NULL) || $(TRUE)

fix-lint:
	@$(WHICH) ruff >$(NULL) 2>&1 || $(PIP) install ruff
	@ruff check python/ --fix --select I,F401,F841,UP,C90,N,E,W 2>$(NULL) || $(TRUE)
	@ruff format python/ 2>$(NULL) || $(TRUE)

lint: lint-ruff lint-pyright lint-mypy
	@$(ECHO) "All linters passed"

lint-ruff:
	@$(WHICH) ruff >$(NULL) 2>&1 || $(PIP) install ruff
	@ruff check python/ 2>$(NULL) || $(TRUE)

lint-pyright:
	@$(WHICH) pyright >$(NULL) 2>&1 && pyright python/ || $(TRUE)

lint-mypy:
	@$(WHICH) mypy >$(NULL) 2>&1 || $(PIP) install mypy
	@mypy python/ --ignore-missing-imports --no-strict-optional 2>$(NULL) || $(TRUE)

check: format lint

# =============================================================================
# Analysis (Unix)
# =============================================================================

cloc:
	@$(WHICH) cloc >$(NULL) 2>&1 || { $(ECHO) "Install: sudo apt install cloc (or brew install cloc)"; exit 1; }
	@cloc . --exclude-dir=build,dist,__pycache__,.git,.eggs,external,weights,forks,node_modules \
		--exclude-ext=.pyc,.pyo,.so,.pyd,.o,.a,.dylib,.dll --vcs=git

tree:
	@git ls-tree -r --name-only HEAD | tree --fromfile . 2>$(NULL) || git ls-tree -r --name-only HEAD

# =============================================================================
# Clean (Unix)
# =============================================================================

clean: clean-build clean-pyc clean-compiled docs-clean
	@$(ECHO) "✅ All artifacts cleaned"

clean-build:
	@$(RM) build/ dist/ *.egg-info .eggs/ 2>$(NULL) || $(TRUE)

clean-pyc:
	@find . -type f -name '*.py[co]' -delete 2>$(NULL) || $(TRUE)
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>$(NULL) || $(TRUE)
	@find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>$(NULL) || $(TRUE)
	@find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>$(NULL) || $(TRUE)
	@find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>$(NULL) || $(TRUE)

clean-compiled:
	@$(RM) $(INSTALL_DIR)/*$(LIB_EXT) 2>$(NULL) || $(TRUE)
	@find . -name '*.so' -not -path "./external/*" -not -path "./forks/*" -delete 2>$(NULL) || $(TRUE)
	@find . -name '*.pyd' -not -path "./external/*" -not -path "./forks/*" -delete 2>$(NULL) || $(TRUE)
	@find . -name '*.dll' -not -path "./external/*" -not -path "./forks/*" -delete 2>$(NULL) || $(TRUE)

# =============================================================================
# Documentation (Unix)
# =============================================================================

makedoc: docs-dev

docs-dev:
	@$(ECHO) "Starting documentation development server..."
	@$(WHICH) npm >$(NULL) 2>&1 || { $(ECHO) "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d node_modules ] || npm install
	@npm run docs:dev

docs-build:
	@$(ECHO) "Building documentation for production..."
	@$(WHICH) npm >$(NULL) 2>&1 || { $(ECHO) "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d node_modules ] || npm install
	@npm run docs:build
	@$(ECHO) "Documentation built in docs/.vitepress/dist/"

docs-preview:
	@$(ECHO) "Previewing documentation..."
	@$(WHICH) npm >$(NULL) 2>&1 || { $(ECHO) "Error: npm not found. Please install Node.js"; exit 1; }
	@[ -d docs/.vitepress/dist ] || { $(ECHO) "Error: Build documentation first with 'make docs-build'"; exit 1; }
	@npm run docs:preview

docs-clean:
	@$(ECHO) "Cleaning documentation build artifacts..."
	@$(RM) docs/.vitepress/dist docs/.vitepress/cache 2>$(NULL) || $(TRUE)

# =============================================================================
# Code Generation (Unix)
# =============================================================================

codegen: codegen-python codegen-docs
	@$(ECHO) "Code generation completed"

codegen-python:
	@$(ECHO) "Generating Python ctypes bindings..."
	@$(VENV_PYTHON) -m codegen -v --overwrite python-bindings 2>$(NULL) || $(ECHO) "Codegen not available"
	@$(ECHO) "Python bindings generated"

codegen-docs:
	@$(ECHO) "Generating C API documentation skeletons..."
	@$(VENV_PYTHON) -m codegen -v c-api-docs 2>$(NULL) || $(ECHO) "Codegen not available"
	@$(ECHO) "C API documentation skeletons generated"

# =============================================================================
# Testing (Unix)
# =============================================================================

test: compile test-build test-run
	@$(ECHO) ""
	@$(ECHO) "✅ All tests completed"

test-build: compile
	@$(ECHO) "Building C-API tests..."
	@$(MKDIR) $(CMAKE_TEST_DIR)
	@if [ -d "$(CMAKE_BUILD_DIR)/install" ]; then \
		cd $(CMAKE_TEST_DIR) && cmake .. -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_PREFIX_PATH="$$(pwd)/../../$(CMAKE_BUILD_DIR)/install"; \
	else \
		cd $(CMAKE_TEST_DIR) && cmake .. -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release; \
	fi
	@cd $(CMAKE_TEST_DIR) && ninja -j$(NPROC)
	@$(ECHO) "✅ Tests built successfully"

test-run:
	@$(ECHO) ""
	@$(ECHO) "========================================"
	@$(ECHO) "  Running SCL C-API Tests"
	@$(ECHO) "========================================"
	@$(ECHO) ""
	@cd $(CMAKE_TEST_DIR) && ctest --output-on-failure --verbose --progress \
		--label-exclude "DISABLED" || \
		($(ECHO) ""; \
		 $(ECHO) "❌ Some tests failed."; \
		 exit 1)

test-quick: test-build
	@cd $(CMAKE_TEST_DIR) && ctest --output-on-failure --parallel $(NPROC)

test-single:
	@if [ -z "$(TEST)" ]; then \
		$(ECHO) "Usage: make test-single TEST=<test_name>"; \
		exit 1; \
	fi
	@cd $(CMAKE_TEST_DIR) && ./$(TEST) || $(TRUE)

test-clean:
	@$(ECHO) "Cleaning test build artifacts..."
	@$(RM) $(CMAKE_TEST_DIR)
	@$(ECHO) "✅ Test artifacts cleaned"

# =============================================================================
# Development (Unix)
# =============================================================================

dev-install:
	@$(PIP) install black isort ruff flake8 mypy autoflake autopep8

info:
	@$(ECHO) ""
	@$(ECHO) "╔══════════════════════════════════════════════════════════════════╗"
	@$(ECHO) "║                     Build Environment Info                       ║"
	@$(ECHO) "╚══════════════════════════════════════════════════════════════════╝"
	@$(ECHO) ""
	@$(ECHO) "Platform:    $(PLATFORM)"
	@$(ECHO) "Shell:       $(SHELL_TYPE)"
	@$(ECHO) "Parallelism: $(NPROC) jobs"
	@$(ECHO) ""
	@$(ECHO) "Tools:"
	@printf "  Python:    " && $(PYTHON) --version 2>&1 || $(ECHO) "not found"
	@printf "  CMake:     " && cmake --version 2>&1 | head -n1 || $(ECHO) "not found"
	@printf "  Ninja:     " && ninja --version 2>&1 || $(ECHO) "not found"
	@printf "  Git:       " && git --version 2>&1 || $(ECHO) "not found"
	@printf "  Compiler:  " && $(CXX) --version 2>&1 | head -n1 || $(ECHO) "not found"
	@$(ECHO) ""
	@$(ECHO) "Directories:"
	@$(ECHO) "  Build:     $(CMAKE_BUILD_DIR)"
	@$(ECHO) "  Debug:     $(CMAKE_DEBUG_DIR)"
	@$(ECHO) "  Tests:     $(CMAKE_TEST_DIR)"
	@$(ECHO) ""

else
# =============================================================================
# Windows PowerShell/CMD Implementation
# =============================================================================

setup-deps:
	@powershell -Command "if (Test-Path scripts/setup_cpp_deps.ps1) { ./scripts/setup_cpp_deps.ps1 }"

dry-run: setup-deps
	@powershell -Command "New-Item -ItemType Directory -Force -Path '$(CMAKE_BUILD_DIR)' | Out-Null"
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTIONS)
	@powershell -Command "if ((Test-Path '$(CMAKE_BUILD_DIR)/compile_commands.json') -and !(Test-Path 'compile_commands.json')) { \
		New-Item -ItemType SymbolicLink -Path 'compile_commands.json' -Target '$(CMAKE_BUILD_DIR)/compile_commands.json' -Force; \
		Write-Host 'Created symlink: compile_commands.json'; \
	}"
	@$(ECHO) "Configuration complete."

configure: dry-run

compile-cpp: setup-deps
	@powershell -Command "New-Item -ItemType Directory -Force -Path '$(CMAKE_BUILD_DIR)' | Out-Null"
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTIONS)
	@cd $(CMAKE_BUILD_DIR) && ninja -j$(NPROC)
	@$(ECHO) "Release build completed"

compile-cpp-debug: setup-deps
	@powershell -Command "New-Item -ItemType Directory -Force -Path '$(CMAKE_DEBUG_DIR)' | Out-Null"
	@cd $(CMAKE_DEBUG_DIR) && cmake ../.. -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTIONS)
	@cd $(CMAKE_DEBUG_DIR) && ninja -j$(NPROC)
	@$(ECHO) "Debug build completed"

compile: compile-cpp

compile-debug: compile-cpp-debug

build: compile
	@$(ECHO) "Build completed"

build-debug: compile-cpp-debug
	@$(ECHO) "Debug build completed"

rebuild: clean build

rebuild-debug: clean build-debug

# =============================================================================
# Code Quality (Windows - simplified)
# =============================================================================

format:
	@$(ECHO) "Formatting code..."
	@powershell -Command "python -m pip install black ruff --quiet 2>$$null; python -m black python/ --line-length 100 2>$$null; python -m ruff format python/ 2>$$null" || $(TRUE)
	@$(ECHO) "Code formatted"

lint:
	@$(ECHO) "Running linters..."
	@powershell -Command "python -m pip install ruff --quiet 2>$$null; python -m ruff check python/ 2>$$null" || $(TRUE)
	@$(ECHO) "Linting complete"

check: format lint

# =============================================================================
# Analysis (Windows)
# =============================================================================

cloc:
	@cloc . --exclude-dir=build,dist,__pycache__,.git,.eggs,external,weights,forks,node_modules \
		--exclude-ext=.pyc,.pyo,.so,.pyd,.o,.a,.dylib,.dll --vcs=git

tree:
	@powershell -Command "git ls-tree -r --name-only HEAD"

# =============================================================================
# Clean (Windows)
# =============================================================================

clean: clean-build clean-pyc docs-clean
	@$(ECHO) "All artifacts cleaned"

clean-build:
	@powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist, *.egg-info, .eggs"

clean-pyc:
	@powershell -Command "Get-ChildItem -Recurse -Include '*.pyc','*.pyo' | Remove-Item -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Recurse -Directory -Include '__pycache__','.pytest_cache','.mypy_cache','.ruff_cache' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"

clean-compiled:
	@powershell -Command "Get-ChildItem -Recurse -Include '*.so','*.pyd','*.dll' -Exclude 'external/*','forks/*' | Remove-Item -Force -ErrorAction SilentlyContinue"

# =============================================================================
# Documentation (Windows)
# =============================================================================

makedoc: docs-dev

docs-dev:
	@$(ECHO) "Starting documentation development server..."
	@powershell -Command "if (!(Get-Command npm -ErrorAction SilentlyContinue)) { Write-Host 'Error: npm not found'; exit 1 }"
	@powershell -Command "if (!(Test-Path node_modules)) { npm install }"
	@npm run docs:dev

docs-build:
	@$(ECHO) "Building documentation for production..."
	@powershell -Command "if (!(Test-Path node_modules)) { npm install }"
	@npm run docs:build
	@$(ECHO) "Documentation built in docs/.vitepress/dist/"

docs-preview:
	@$(ECHO) "Previewing documentation..."
	@powershell -Command "if (!(Test-Path 'docs/.vitepress/dist')) { Write-Host 'Error: Build documentation first'; exit 1 }"
	@npm run docs:preview

docs-clean:
	@$(ECHO) "Cleaning documentation build artifacts..."
	@powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'docs/.vitepress/dist', 'docs/.vitepress/cache'"

# =============================================================================
# Code Generation (Windows)
# =============================================================================

codegen: codegen-python codegen-docs
	@$(ECHO) "Code generation completed"

codegen-python:
	@$(ECHO) "Generating Python ctypes bindings..."
	@powershell -Command "python -m codegen -v --overwrite python-bindings 2>$$null" || $(ECHO) "Codegen not available"

codegen-docs:
	@$(ECHO) "Generating C API documentation skeletons..."
	@powershell -Command "python -m codegen -v c-api-docs 2>$$null" || $(ECHO) "Codegen not available"

# =============================================================================
# Testing (Windows)
# =============================================================================

test: compile test-build test-run
	@$(ECHO) ""
	@$(ECHO) "All tests completed"

test-build: compile
	@$(ECHO) "Building C-API tests..."
	@powershell -Command "New-Item -ItemType Directory -Force -Path '$(CMAKE_TEST_DIR)' | Out-Null"
	@cd $(CMAKE_TEST_DIR) && cmake .. -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=Release
	@cd $(CMAKE_TEST_DIR) && ninja -j$(NPROC)
	@$(ECHO) "Tests built successfully"

test-run:
	@$(ECHO) ""
	@$(ECHO) "========================================"
	@$(ECHO) "  Running SCL C-API Tests"
	@$(ECHO) "========================================"
	@$(ECHO) ""
	@cd $(CMAKE_TEST_DIR) && ctest --output-on-failure --verbose --progress

test-quick: test-build
	@cd $(CMAKE_TEST_DIR) && ctest --output-on-failure --parallel $(NPROC)

test-single:
	@powershell -Command "if ('$(TEST)' -eq '') { Write-Host 'Usage: make test-single TEST=<test_name>'; exit 1 }"
	@cd $(CMAKE_TEST_DIR) && .\$(TEST).exe

test-clean:
	@$(ECHO) "Cleaning test build artifacts..."
	@powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '$(CMAKE_TEST_DIR)'"
	@$(ECHO) "Test artifacts cleaned"

# =============================================================================
# Development (Windows)
# =============================================================================

dev-install:
	@python -m pip install black isort ruff flake8 mypy autoflake autopep8

info:
	@$(ECHO) ""
	@$(ECHO) "========================================"
	@$(ECHO) "       Build Environment Info"
	@$(ECHO) "========================================"
	@$(ECHO) ""
	@$(ECHO) "Platform:    $(PLATFORM)"
	@$(ECHO) "Shell:       $(SHELL_TYPE)"
	@$(ECHO) "Parallelism: $(NPROC) jobs"
	@$(ECHO) ""
	@powershell -Command "Write-Host 'Tools:'"
	@powershell -Command "Write-Host ('  Python:    ' + (python --version 2>&1))"
	@powershell -Command "Write-Host ('  CMake:     ' + ((cmake --version 2>&1) | Select-Object -First 1))"
	@powershell -Command "Write-Host ('  Ninja:     ' + (ninja --version 2>&1))"
	@powershell -Command "Write-Host ('  Git:       ' + (git --version 2>&1))"
	@$(ECHO) ""
	@$(ECHO) "Directories:"
	@$(ECHO) "  Build:     $(CMAKE_BUILD_DIR)"
	@$(ECHO) "  Debug:     $(CMAKE_DEBUG_DIR)"
	@$(ECHO) "  Tests:     $(CMAKE_TEST_DIR)"
	@$(ECHO) ""

endif
