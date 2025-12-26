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
        if ! check_command ninja; then
            brew install ninja
        fi
        if ! check_command make; then
            # Make is usually pre-installed on macOS
            print_info "Make should be available via Xcode Command Line Tools"
        fi
    elif [ "$PLATFORM" = "debian" ] || [ "$PLATFORM" = "redhat" ]; then
        if [ "$PKG_MGR" = "apt-get" ]; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git ninja-build
        elif [ "$PKG_MGR" = "yum" ] || [ "$PKG_MGR" = "dnf" ]; then
            sudo $PKG_MGR install -y gcc gcc-c++ cmake git ninja-build
        fi
    fi
    
    print_success "Build tools ready (CMake + Ninja)"
}

install_hdf5_linux() {
    print_info "Installing HDF5 for Linux..."
    
    if [ "$PKG_MGR" = "apt-get" ]; then
        sudo apt-get update
        sudo apt-get install -y libhdf5-dev hdf5-tools
    elif [ "$PKG_MGR" = "yum" ] || [ "$PKG_MGR" = "dnf" ]; then
        sudo $PKG_MGR install -y hdf5-devel
    else
        print_error "Unknown package manager for Linux"
        return 1
    fi
    
    print_success "HDF5 installed successfully"
}

install_hdf5_macos() {
    print_info "Installing HDF5 for macOS..."
    
    if ! check_command brew; then
        print_error "Homebrew is required for macOS HDF5 installation"
        return 1
    fi
    
    brew install hdf5
    
    print_success "HDF5 installed successfully"
    print_info "HDF5 location: $(brew --prefix hdf5)"
}

install_highway() {
    print_info "Installing Google Highway (SIMD library)..."
    
    local HWY_VERSION="1.0.7"
    local HWY_DIR="$PROJECT_ROOT/libs/highway"
    
    # Check if already installed
    if [ -d "$HWY_DIR" ]; then
        print_warning "Highway directory already exists at libs/highway"
        
        # Check if it's a valid git repository
        if [ -d "$HWY_DIR/.git" ]; then
            cd "$HWY_DIR"
            
            # Get current version/tag
            CURRENT_VERSION=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo "unknown")
            
            print_info "Current version: $CURRENT_VERSION"
            print_info "Target version:  $HWY_VERSION"
            
            # Check if CMakeLists.txt exists (validation)
            if [ -f "$HWY_DIR/CMakeLists.txt" ]; then
                print_success "Highway installation appears valid"
                
                if [[ "$CURRENT_VERSION" == "$HWY_VERSION" ]]; then
                    print_success "Version matches target, installation is up-to-date"
                    cd "$PROJECT_ROOT"
                    return 0
                else
                    print_warning "Version mismatch detected"
                fi
            else
                print_error "Invalid Highway installation (missing CMakeLists.txt)"
            fi
            
            cd "$PROJECT_ROOT"
        else
            print_warning "Directory exists but is not a git repository"
        fi
        
        # Ask user what to do
        echo ""
        echo "Options:"
        echo "  1) Keep existing installation"
        echo "  2) Remove and reinstall with v${HWY_VERSION}"
        read -p "Choose (1/2): " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[2]$ ]]; then
            print_info "Keeping existing Highway installation"
            return 0
        fi
        
        print_info "Removing existing installation..."
        rm -rf "$HWY_DIR"
    fi
    
    # Create libs directory if not exists
    mkdir -p "$PROJECT_ROOT/libs"
    
    # Clone Highway repository
    print_info "Cloning Highway v${HWY_VERSION} from GitHub..."
    cd "$PROJECT_ROOT/libs"
    
    git clone --depth 1 --branch ${HWY_VERSION} \
        https://github.com/google/highway.git highway
    
    if [ $? -eq 0 ]; then
        # Verify installation
        if [ -f "$HWY_DIR/CMakeLists.txt" ]; then
            print_success "Highway v${HWY_VERSION} installed successfully to libs/highway"
            print_info "CMake will use the local copy instead of FetchContent"
        else
            print_error "Installation completed but validation failed"
            cd "$PROJECT_ROOT"
            return 1
        fi
    else
        print_error "Failed to clone Highway repository"
        print_info "CMake will fall back to FetchContent during build"
        cd "$PROJECT_ROOT"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
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
    
    # Install HDF5 (required for h5ad support)
    read -p "Install HDF5 for native .h5ad support? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if [ "$PLATFORM" = "macos" ]; then
            install_hdf5_macos
        else
            install_hdf5_linux
        fi
    else
        print_warning "Skipping HDF5 - native .h5ad I/O will be disabled"
    fi
    
    # Optionally install Highway locally
    read -p "Install Google Highway to libs/highway? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_highway
    else
        print_info "Skipping Highway installation (CMake will use FetchContent)"
    fi
    
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

