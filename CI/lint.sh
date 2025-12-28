#!/usr/bin/env bash
set -euo pipefail

# SCL Core Lint Script
# Usage: ./CI/lint.sh [options]

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
CHECK_CPP="${CHECK_CPP:-1}"
CHECK_PYTHON="${CHECK_PYTHON:-1}"
FIX="${FIX:-0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            CHECK_PYTHON=0
            shift
            ;;
        --python-only)
            CHECK_CPP=0
            shift
            ;;
        --fix)
            FIX=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --cpp-only     Check only C++ code"
            echo "  --python-only  Check only Python code"
            echo "  --fix          Automatically fix issues where possible"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  CHECK_CPP      Check C++ code (default: 1)"
            echo "  CHECK_PYTHON   Check Python code (default: 1)"
            echo "  FIX            Auto-fix issues (default: 0)"
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

log_info "SCL Core Code Quality Checks"
log_info "=============================="
log_info "C++ Checks:    $([ $CHECK_CPP -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "Python Checks: $([ $CHECK_PYTHON -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "Auto-fix:      $([ $FIX -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
log_info "=============================="

LINT_FAILED=0

# Check C++ code
if [ $CHECK_CPP -eq 1 ]; then
    log_info "Checking C++ code..."

    # Find all C++ source and header files
    CPP_FILES=$(find scl -type f \( -name "*.cpp" -o -name "*.hpp" \) 2>/dev/null || true)

    if [ -n "$CPP_FILES" ]; then
        # Check if clang-format is available
        if command -v clang-format &> /dev/null; then
            log_info "Running clang-format..."

            if [ $FIX -eq 1 ]; then
                # Fix formatting in-place
                echo "$CPP_FILES" | xargs clang-format -i
                log_info "C++ code formatting applied"
            else
                # Check formatting without modifying files
                FORMAT_ISSUES=0
                for file in $CPP_FILES; do
                    if ! clang-format --dry-run --Werror "$file" &> /dev/null; then
                        log_error "Formatting issues in: $file"
                        FORMAT_ISSUES=1
                    fi
                done

                if [ $FORMAT_ISSUES -eq 1 ]; then
                    log_error "C++ formatting check failed. Run with --fix to auto-format"
                    LINT_FAILED=1
                else
                    log_info "C++ formatting check passed"
                fi
            fi
        else
            log_warn "clang-format not found, skipping C++ format check"
        fi

        # Check if clang-tidy is available (optional)
        if command -v clang-tidy &> /dev/null; then
            log_info "Running clang-tidy..."

            # Only run clang-tidy if compile_commands.json exists
            if [ -f "compile_commands.json" ]; then
                TIDY_ISSUES=0
                for file in $CPP_FILES; do
                    if ! clang-tidy "$file" -p . --quiet 2>&1 | grep -v "^[0-9]* warning"; then
                        TIDY_ISSUES=1
                    fi
                done

                if [ $TIDY_ISSUES -eq 1 ]; then
                    log_warn "clang-tidy found potential issues (non-fatal)"
                else
                    log_info "clang-tidy check passed"
                fi
            else
                log_warn "compile_commands.json not found, skipping clang-tidy"
            fi
        else
            log_warn "clang-tidy not found, skipping C++ static analysis"
        fi
    else
        log_warn "No C++ files found"
    fi
fi

# Check Python code
if [ $CHECK_PYTHON -eq 1 ]; then
    log_info "Checking Python code..."

    # Find all Python files (excluding virtual environments and build directories)
    PYTHON_FILES=$(find . -type f -name "*.py" \
        -not -path "./.venv/*" \
        -not -path "./venv/*" \
        -not -path "./build/*" \
        -not -path "./_build/*" \
        -not -path "./node_modules/*" \
        -not -path "./forks/*" \
        2>/dev/null || true)

    if [ -n "$PYTHON_FILES" ]; then
        # Check if black is available
        if command -v black &> /dev/null; then
            log_info "Running black..."

            if [ $FIX -eq 1 ]; then
                # Fix formatting in-place
                echo "$PYTHON_FILES" | xargs black
                log_info "Python code formatting applied"
            else
                # Check formatting without modifying files
                if echo "$PYTHON_FILES" | xargs black --check --quiet; then
                    log_info "Python formatting check passed"
                else
                    log_error "Python formatting check failed. Run with --fix to auto-format"
                    LINT_FAILED=1
                fi
            fi
        else
            log_warn "black not found, skipping Python format check"
        fi

        # Check if flake8 is available
        if command -v flake8 &> /dev/null; then
            log_info "Running flake8..."

            if echo "$PYTHON_FILES" | xargs flake8 --max-line-length=100; then
                log_info "Python linting check passed"
            else
                log_error "Python linting check failed"
                LINT_FAILED=1
            fi
        else
            log_warn "flake8 not found, skipping Python linting"
        fi
    else
        log_warn "No Python files found"
    fi
fi

# Final summary
log_info ""
if [ $LINT_FAILED -eq 0 ]; then
    log_info "All code quality checks passed!"
    exit 0
else
    log_error "Some code quality checks failed"
    exit 1
fi
