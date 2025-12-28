#!/usr/bin/env bash
set -euo pipefail

# SCL Core Build Script
# Usage: ./CI/build.sh [options]

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
BUILD_TYPE="${BUILD_TYPE:-Release}"
SCL_THREADING_BACKEND="${SCL_THREADING_BACKEND:-AUTO}"
BUILD_DIR="${BUILD_DIR:-build}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Parse command line arguments
CLEAN_BUILD=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --threading)
            SCL_THREADING_BACKEND="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean              Clean build directory before building"
            echo "  --verbose            Enable verbose build output"
            echo "  --build-type TYPE    Set build type (Release|Debug|RelWithDebInfo)"
            echo "  --threading BACKEND  Set threading backend (AUTO|SERIAL|BS|OPENMP|TBB)"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  BUILD_TYPE              Build type (default: Release)"
            echo "  SCL_THREADING_BACKEND   Threading backend (default: AUTO)"
            echo "  BUILD_DIR               Build directory (default: build)"
            echo "  NUM_JOBS                Number of parallel jobs (default: auto-detect)"
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

log_info "SCL Core Build Configuration"
log_info "=============================="
log_info "Build Type:        $BUILD_TYPE"
log_info "Threading Backend: $SCL_THREADING_BACKEND"
log_info "Build Directory:   $BUILD_DIR"
log_info "Parallel Jobs:     $NUM_JOBS"
log_info "=============================="

# Clean build directory if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
log_info "Configuring CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DSCL_THREADING_BACKEND="$SCL_THREADING_BACKEND"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ $VERBOSE -eq 1 ]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake "${CMAKE_ARGS[@]}" ..

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed"
    exit 1
fi

# Build
log_info "Building project with $NUM_JOBS parallel jobs..."
if [ $VERBOSE -eq 1 ]; then
    cmake --build . --parallel "$NUM_JOBS" --verbose
else
    cmake --build . --parallel "$NUM_JOBS"
fi

if [ $? -ne 0 ]; then
    log_error "Build failed"
    exit 1
fi

# Display build summary
log_info ""
log_info "Build completed successfully!"
log_info "=============================="
log_info "Built libraries:"
ls -lh lib/*.so lib/*.dylib lib/*.dll 2>/dev/null || log_warn "No shared libraries found in lib/"
log_info "=============================="
