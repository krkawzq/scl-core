#!/usr/bin/env bash
set -euo pipefail

# SCL Core Test Script
# Usage: ./CI/test.sh [options]

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
BUILD_DIR="${BUILD_DIR:-build}"
RUN_CPP_TESTS="${RUN_CPP_TESTS:-1}"
RUN_PYTHON_TESTS="${RUN_PYTHON_TESTS:-1}"
COVERAGE="${COVERAGE:-0}"
VERBOSE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            RUN_PYTHON_TESTS=0
            shift
            ;;
        --python-only)
            RUN_CPP_TESTS=0
            shift
            ;;
        --coverage)
            COVERAGE=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --cpp-only     Run only C++ tests"
            echo "  --python-only  Run only Python tests"
            echo "  --coverage     Enable coverage reporting"
            echo "  --verbose      Enable verbose test output"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  BUILD_DIR          Build directory (default: build)"
            echo "  RUN_CPP_TESTS      Run C++ tests (default: 1)"
            echo "  RUN_PYTHON_TESTS   Run Python tests (default: 1)"
            echo "  COVERAGE           Enable coverage (default: 0)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info "SCL Core Test Suite"
log_info "===================="
log_info "C++ Tests:    $([ $RUN_CPP_TESTS -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "Python Tests: $([ $RUN_PYTHON_TESTS -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "Coverage:     $([ $COVERAGE -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "===================="

TEST_FAILED=0

# Run C++ tests
if [ $RUN_CPP_TESTS -eq 1 ]; then
    log_info "Running C++ tests..."

    # Check if C++ test directory exists and has tests
    if [ -d "tests/c" ] && [ "$(ls -A tests/c/*.cpp 2>/dev/null)" ]; then
        # Check if CTest is available
        if [ -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
            cd "$BUILD_DIR"
            if [ $VERBOSE -eq 1 ]; then
                ctest --verbose --output-on-failure
            else
                ctest --output-on-failure
            fi

            if [ $? -ne 0 ]; then
                log_error "C++ tests failed"
                TEST_FAILED=1
            else
                log_info "C++ tests passed"
            fi
            cd "$PROJECT_ROOT"
        else
            log_warn "No C++ test targets found in build directory"
        fi
    else
        log_warn "No C++ test files found in tests/c/"
    fi
fi

# Run Python tests
if [ $RUN_PYTHON_TESTS -eq 1 ]; then
    log_info "Running Python tests..."

    # Check if Python tests exist
    if [ -d "tests/python" ] && [ "$(ls -A tests/python/*.py 2>/dev/null)" ]; then
        # Prepare pytest arguments
        PYTEST_ARGS=()

        if [ $VERBOSE -eq 1 ]; then
            PYTEST_ARGS+=(-v)
        fi

        if [ $COVERAGE -eq 1 ]; then
            PYTEST_ARGS+=(--cov=scl --cov-report=term --cov-report=html)
        fi

        # Add tests directory
        PYTEST_ARGS+=(tests/python)

        # Run pytest
        if command -v pytest &> /dev/null; then
            pytest "${PYTEST_ARGS[@]}"

            if [ $? -ne 0 ]; then
                log_error "Python tests failed"
                TEST_FAILED=1
            else
                log_info "Python tests passed"
            fi

            if [ $COVERAGE -eq 1 ]; then
                log_info "Coverage report generated in htmlcov/"
            fi
        else
            log_error "pytest not found. Install it with: pip install pytest"
            TEST_FAILED=1
        fi
    else
        log_warn "No Python test files found in tests/python/"
    fi
fi

# Final summary
log_info ""
if [ $TEST_FAILED -eq 0 ]; then
    log_info "All tests passed!"
    exit 0
else
    log_error "Some tests failed"
    exit 1
fi
