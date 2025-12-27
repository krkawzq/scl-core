#!/bin/bash
# =============================================================================
# SCL Core - macOS Dependencies Setup Script
# =============================================================================
#
# This script installs system-level dependencies for building scl-core on macOS.
# Requires: Homebrew package manager
#
# Usage:
#   ./scripts/setup_cpp_dependencies_macos.sh [backend]
#
# Backend options:
#   - bs      : BS::thread_pool (default for macOS, header-only)
#   - openmp  : Install OpenMP via libomp
#   - tbb     : Install Intel TBB
#   - serial  : No system dependencies
#
# Dependencies installed:
#   - Xcode Command Line Tools
#   - Homebrew (if not present)
#   - CMake 3.15+
#   - Ninja build system
#   - Threading backend (libomp or TBB)
#   - HDF5 (optional, for .h5ad support)
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default backend (BS is preferred on macOS to avoid libomp issues)
BACKEND="${1:-bs}"

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
# macOS Prerequisites
# =============================================================================

check_xcode() {
    print_info "Checking Xcode Command Line Tools..."
    
    if xcode-select -p &> /dev/null; then
        print_success "Xcode Command Line Tools installed"
        return 0
    else
        print_warning "Xcode Command Line Tools not found"
        read -p "Install Xcode Command Line Tools? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            xcode-select --install
            echo "Please complete the installation and re-run this script"
            exit 0
        else
            print_error "Xcode Command Line Tools are required"
            exit 1
        fi
    fi
}

check_homebrew() {
    print_info "Checking Homebrew..."
    
    if check_command brew; then
        print_success "Homebrew installed: $(brew --version | head -n1)"
        return 0
    else
        print_warning "Homebrew not found"
        read -p "Install Homebrew? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            
            # Add Homebrew to PATH for Apple Silicon
            if [[ $(uname -m) == "arm64" ]]; then
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                eval "$(/opt/homebrew/bin/brew shellenv)"
            fi
            
            print_success "Homebrew installed"
        else
            print_error "Homebrew is required for dependency management"
            exit 1
        fi
    fi
}

# =============================================================================
# Dependency Installation
# =============================================================================

install_build_tools() {
    print_info "Installing build tools..."
    
    # Update Homebrew
    brew update
    
    # Install CMake
    if ! check_command cmake; then
        brew install cmake
    else
        CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '\d+\.\d+\.\d+')
        print_info "CMake $CMAKE_VERSION already installed"
    fi
    
    # Install Ninja
    if ! check_command ninja; then
        brew install ninja
    else
        print_info "Ninja already installed"
    fi
    
    # Install Git (usually pre-installed)
    if ! check_command git; then
        brew install git
    fi
    
    print_success "Build tools ready"
}

install_openmp() {
    print_info "Installing OpenMP (libomp)..."
    
    brew install libomp
    
    # Get libomp path
    LIBOMP_PREFIX=$(brew --prefix libomp)
    
    print_success "OpenMP (libomp) installed"
    print_warning "Note: You may need to set environment variables:"
    echo ""
    echo "  export CMAKE_PREFIX_PATH=\"$LIBOMP_PREFIX:\$CMAKE_PREFIX_PATH\""
    echo "  export LDFLAGS=\"-L$LIBOMP_PREFIX/lib\""
    echo "  export CPPFLAGS=\"-I$LIBOMP_PREFIX/include\""
    echo ""
    echo "Or configure CMake with:"
    echo "  cmake .. -DSCL_MAC_USE_OPENMP=ON -DCMAKE_PREFIX_PATH=$LIBOMP_PREFIX"
    echo ""
}

install_tbb() {
    print_info "Installing Intel TBB..."
    
    brew install tbb
    
    print_success "Intel TBB installed"
}

install_hdf5() {
    print_info "Installing HDF5..."
    
    brew install hdf5
    
    HDF5_PREFIX=$(brew --prefix hdf5)
    
    print_success "HDF5 installed"
    print_info "HDF5 location: $HDF5_PREFIX"
}

# =============================================================================
# Architecture Detection
# =============================================================================

detect_architecture() {
    ARCH=$(uname -m)
    
    if [[ "$ARCH" == "arm64" ]]; then
        print_info "Detected Apple Silicon (ARM64)"
        print_warning "Note: Some dependencies may be ARM-native, others via Rosetta"
    elif [[ "$ARCH" == "x86_64" ]]; then
        print_info "Detected Intel x86_64"
    else
        print_warning "Unknown architecture: $ARCH"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "=========================================="
    echo "SCL Core - macOS Dependencies Setup"
    echo "=========================================="
    echo ""
    
    detect_architecture
    check_xcode
    check_homebrew
    
    # Install build tools
    install_build_tools
    
    # Install HDF5
    read -p "Install HDF5 for .h5ad support? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        install_hdf5
    else
        print_warning "Skipping HDF5"
    fi
    
    # Install threading backend
    case "$BACKEND" in
        bs|BS)
            print_info "Using BS::thread_pool backend (header-only, no dependencies)"
            print_success "No system dependencies required for BS backend"
            ;;
        openmp|OPENMP)
            install_openmp
            ;;
        tbb|TBB)
            install_tbb
            ;;
        serial|SERIAL)
            print_info "Using serial backend (no threading dependencies)"
            ;;
        *)
            print_error "Unknown backend: $BACKEND"
            echo "Supported: bs, openmp, tbb, serial"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    print_success "Dependencies installed!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    
    if [ "$BACKEND" = "openmp" ]; then
        echo "  cmake .. -DSCL_THREADING_BACKEND=OPENMP -DSCL_MAC_USE_OPENMP=ON"
    elif [ "$BACKEND" = "tbb" ]; then
        echo "  cmake .. -DSCL_THREADING_BACKEND=TBB"
    else
        echo "  cmake .."
    fi
    
    echo "  make -j\$(sysctl -n hw.ncpu)"
    echo ""
    
    if [ "$BACKEND" = "bs" ]; then
        echo "Note: BS::thread_pool is the recommended backend for macOS"
        echo "      It requires no system dependencies and avoids libomp issues."
        echo ""
    fi
}

main "$@"

