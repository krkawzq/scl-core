#!/bin/bash
# =============================================================================
# SCL Core - C++ Dependencies Setup Script
# =============================================================================
#
# This script installs system-level dependencies required for building scl-core.
# It supports Linux (apt/yum) and macOS (Homebrew).
#
# Usage:
#   ./scripts/setup_cpp_dependencies.sh [backend]
#
# Backend options:
#   - openmp  : Install OpenMP (default on Linux/Windows)
#   - tbb     : Install Intel TBB
#   - bs      : No system dependencies (header-only)
#   - serial  : No system dependencies
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default backend
BACKEND="${1:-openmp}"

# =============================================================================
# Helper Functions
# =============================================================================

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Platform Detection
# =============================================================================

detect_platform() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Detect Linux distribution
        if [ -f /etc/debian_version ]; then
            if check_command apt-get; then
                PLATFORM="debian"
                PKG_MGR="apt-get"
            fi
        elif [ -f /etc/redhat-release ]; then
            if check_command yum; then
                PLATFORM="redhat"
                PKG_MGR="yum"
            elif check_command dnf; then
                PLATFORM="redhat"
                PKG_MGR="dnf"
            fi
        else
            PLATFORM="linux"
            PKG_MGR="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macos"
        if check_command brew; then
            PKG_MGR="brew"
        else
            PKG_MGR="unknown"
            print_error "Homebrew not found. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    else
        print_error "Unsupported platform: $OSTYPE"
        exit 1
    fi
    
    print_info "Detected platform: $PLATFORM ($PKG_MGR)"
}

# =============================================================================
# Dependency Installation Functions
# =============================================================================

install_openmp_linux() {
    print_info "Installing OpenMP for Linux..."
    
    if [ "$PKG_MGR" = "apt-get" ]; then
        sudo apt-get update
        sudo apt-get install -y libomp-dev
    elif [ "$PKG_MGR" = "yum" ] || [ "$PKG_MGR" = "dnf" ]; then
        sudo $PKG_MGR install -y libgomp-devel
    else
        print_error "Unknown package manager for Linux"
        return 1
    fi
    
    print_success "OpenMP installed successfully"
}

install_openmp_macos() {
    print_info "Installing OpenMP for macOS (libomp)..."
    
    if ! check_command brew; then
        print_error "Homebrew is required for macOS OpenMP installation"
        return 1
    fi
    
    brew install libomp
    
    print_success "OpenMP (libomp) installed successfully"
    print_warning "Note: You may need to set CMAKE_PREFIX_PATH or link libomp manually"
    print_info "  export CMAKE_PREFIX_PATH=\$(brew --prefix libomp):\$CMAKE_PREFIX_PATH"
}

install_tbb_linux() {
    print_info "Installing Intel TBB for Linux..."
    
    if [ "$PKG_MGR" = "apt-get" ]; then
        sudo apt-get update
        sudo apt-get install -y libtbb-dev
    elif [ "$PKG_MGR" = "yum" ] || [ "$PKG_MGR" = "dnf" ]; then
        sudo $PKG_MGR install -y tbb-devel
    else
        print_error "Unknown package manager for Linux"
        return 1
    fi
    
    print_success "Intel TBB installed successfully"
}

install_tbb_macos() {
    print_info "Installing Intel TBB for macOS..."
    
    if ! check_command brew; then
        print_error "Homebrew is required for macOS TBB installation"
        return 1
    fi
    
    brew install tbb
    
    print_success "Intel TBB installed successfully"
}

install_build_tools() {
    print_info "Installing build tools..."
    
    if [ "$PLATFORM" = "macos" ]; then
        if ! check_command cmake; then
            brew install cmake
        fi
        if ! check_command make; then
            # Make is usually pre-installed on macOS
            print_info "Make should be available via Xcode Command Line Tools"
        fi
    elif [ "$PLATFORM" = "debian" ] || [ "$PLATFORM" = "redhat" ]; then
        if [ "$PKG_MGR" = "apt-get" ]; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git
        elif [ "$PKG_MGR" = "yum" ] || [ "$PKG_MGR" = "dnf" ]; then
            sudo $PKG_MGR install -y gcc gcc-c++ cmake git
        fi
    fi
    
    print_success "Build tools ready"
}

# =============================================================================
# Main Installation Logic
# =============================================================================

main() {
    echo "=========================================="
    echo "SCL Core - C++ Dependencies Setup"
    echo "=========================================="
    echo ""
    
    detect_platform
    
    # Install build tools first
    install_build_tools
    
    # Install dependencies based on backend
    case "$BACKEND" in
        openmp|OPENMP)
            print_info "Setting up dependencies for OpenMP backend..."
            if [ "$PLATFORM" = "macos" ]; then
                install_openmp_macos
            else
                install_openmp_linux
            fi
            ;;
        tbb|TBB)
            print_info "Setting up dependencies for TBB backend..."
            if [ "$PLATFORM" = "macos" ]; then
                install_tbb_macos
            else
                install_tbb_linux
            fi
            ;;
        bs|BS|serial|SERIAL)
            print_info "No system dependencies required for $BACKEND backend"
            print_info "BS::thread_pool is header-only (fetched by CMake)"
            ;;
        *)
            print_error "Unknown backend: $BACKEND"
            echo "Supported backends: openmp, tbb, bs, serial"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    print_success "Dependencies setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run CMake configuration:"
    echo "     cd $PROJECT_ROOT"
    echo "     mkdir -p build && cd build"
    echo "     cmake .."
    echo ""
    echo "  2. Or use the Makefile:"
    echo "     cd $PROJECT_ROOT"
    echo "     make"
    echo ""
    
    if [ "$PLATFORM" = "macos" ] && [ "$BACKEND" = "openmp" ]; then
        echo "  3. macOS OpenMP Note:"
        echo "     If CMake can't find libomp, set:"
        echo "     export CMAKE_PREFIX_PATH=\$(brew --prefix libomp):\$CMAKE_PREFIX_PATH"
        echo ""
    fi
}

# Run main function
main "$@"

