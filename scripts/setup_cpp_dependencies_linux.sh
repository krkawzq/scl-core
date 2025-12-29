#!/bin/bash
# =============================================================================
# SCL Core - Linux Dependencies Setup Script
# =============================================================================
#
# This script installs system-level dependencies for building scl-core on Linux.
# Supports: Ubuntu/Debian (apt), RHEL/CentOS/Fedora (yum/dnf), Arch (pacman)
#
# Usage:
#   ./scripts/setup_cpp_dependencies_linux.sh [backend]
#
# Backend options:
#   - openmp  : Install OpenMP (default)
#   - tbb     : Install Intel TBB
#   - bs      : No system dependencies (header-only)
#   - serial  : No system dependencies
#
# Dependencies installed:
#   - CMake 3.15+
#   - C++20 compatible compiler (GCC 10+ or Clang 12+)
#   - Build tools (make, ninja)
#   - Threading backend (OpenMP or TBB)
#   - HDF5 (optional, for .h5ad support)
#   - Eigen3 (for testing reference implementation)
#   - Git
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
# Distribution Detection
# =============================================================================

detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        DISTRO=$DISTRIB_ID
        VERSION=$DISTRIB_RELEASE
    else
        DISTRO="unknown"
        VERSION="unknown"
    fi
    
    # Detect package manager
    if check_command apt-get; then
        PKG_MGR="apt"
        UPDATE_CMD="sudo apt-get update"
        INSTALL_CMD="sudo apt-get install -y"
    elif check_command dnf; then
        PKG_MGR="dnf"
        UPDATE_CMD="sudo dnf check-update || true"
        INSTALL_CMD="sudo dnf install -y"
    elif check_command yum; then
        PKG_MGR="yum"
        UPDATE_CMD="sudo yum check-update || true"
        INSTALL_CMD="sudo yum install -y"
    elif check_command pacman; then
        PKG_MGR="pacman"
        UPDATE_CMD="sudo pacman -Sy"
        INSTALL_CMD="sudo pacman -S --noconfirm"
    else
        print_error "No supported package manager found (apt/dnf/yum/pacman)"
        exit 1
    fi
    
    print_info "Detected: $DISTRO $VERSION ($PKG_MGR)"
}

# =============================================================================
# Dependency Installation
# =============================================================================

install_build_tools() {
    print_info "Installing build tools..."
    
    case "$PKG_MGR" in
        apt)
            $UPDATE_CMD
            $INSTALL_CMD build-essential cmake git ninja-build pkg-config
            ;;
        dnf|yum)
            $UPDATE_CMD
            $INSTALL_CMD gcc gcc-c++ cmake git ninja-build pkgconfig
            ;;
        pacman)
            $UPDATE_CMD
            $INSTALL_CMD base-devel cmake git ninja
            ;;
    esac
    
    # Check CMake version
    if check_command cmake; then
        CMAKE_VERSION=$(cmake --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
        print_success "CMake $CMAKE_VERSION installed"
        
        # Verify CMake >= 3.15
        if [ "$(printf '%s\n' "3.15" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.15" ]; then
            print_warning "CMake version is < 3.15, may need to upgrade"
        fi
    fi
    
    # Check compiler
    if check_command g++; then
        GCC_VERSION=$(g++ --version | head -n1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
        print_success "GCC $GCC_VERSION installed"
    elif check_command clang++; then
        CLANG_VERSION=$(clang++ --version | head -n1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
        print_success "Clang $CLANG_VERSION installed"
    fi
}

install_openmp() {
    print_info "Installing OpenMP..."
    
    case "$PKG_MGR" in
        apt)
            $INSTALL_CMD libomp-dev
            ;;
        dnf|yum)
            $INSTALL_CMD libgomp-devel
            ;;
        pacman)
            # OpenMP is included in gcc
            print_info "OpenMP is included with GCC on Arch Linux"
            ;;
    esac
    
    print_success "OpenMP installed"
}

install_tbb() {
    print_info "Installing Intel TBB..."
    
    case "$PKG_MGR" in
        apt)
            $INSTALL_CMD libtbb-dev
            ;;
        dnf|yum)
            $INSTALL_CMD tbb-devel
            ;;
        pacman)
            $INSTALL_CMD intel-tbb
            ;;
    esac
    
    print_success "Intel TBB installed"
}

install_hdf5() {
    print_info "Installing HDF5..."
    
    case "$PKG_MGR" in
        apt)
            $INSTALL_CMD libhdf5-dev hdf5-tools
            ;;
        dnf|yum)
            $INSTALL_CMD hdf5-devel
            ;;
        pacman)
            $INSTALL_CMD hdf5
            ;;
    esac
    
    print_success "HDF5 installed"
}

install_test_deps() {
    print_info "Installing testing dependencies..."
    
    case "$PKG_MGR" in
        apt)
            $INSTALL_CMD libeigen3-dev
            ;;
        dnf|yum)
            $INSTALL_CMD eigen3-devel
            ;;
        pacman)
            $INSTALL_CMD eigen
            ;;
    esac
    
    print_success "Testing dependencies installed (Eigen3)"
    print_info "Catch2 will be fetched by CMake (header-only)"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "=========================================="
    echo "SCL Core - Linux Dependencies Setup"
    echo "=========================================="
    echo ""
    
    detect_distro
    
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
    
    # Install test dependencies
    read -p "Install testing dependencies (Eigen3)? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        install_test_deps
    else
        print_warning "Skipping test dependencies"
    fi
    
    # Install threading backend
    case "$BACKEND" in
        openmp|OPENMP)
            install_openmp
            ;;
        tbb|TBB)
            install_tbb
            ;;
        bs|BS|serial|SERIAL)
            print_info "No threading dependencies for $BACKEND backend"
            ;;
        *)
            print_error "Unknown backend: $BACKEND"
            echo "Supported: openmp, tbb, bs, serial"
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
    echo "  cmake .."
    echo "  make -j\$(nproc)"
    echo ""
}

main "$@"

