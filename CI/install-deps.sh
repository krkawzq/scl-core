#!/usr/bin/env bash
set -euo pipefail

# SCL Core Dependency Installation Script
# Usage: ./CI/install-deps.sh [options]

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
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_PYTHON_DEPS="${INSTALL_PYTHON_DEPS:-1}"
INSTALL_DEV_TOOLS="${INSTALL_DEV_TOOLS:-0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --system-only)
            INSTALL_PYTHON_DEPS=0
            shift
            ;;
        --python-only)
            INSTALL_SYSTEM_DEPS=0
            shift
            ;;
        --with-dev-tools)
            INSTALL_DEV_TOOLS=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --system-only      Install only system dependencies"
            echo "  --python-only      Install only Python dependencies"
            echo "  --with-dev-tools   Install development tools (clang-format, clang-tidy)"
            echo "  --help             Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  INSTALL_SYSTEM_DEPS   Install system deps (default: 1)"
            echo "  INSTALL_PYTHON_DEPS   Install Python deps (default: 1)"
            echo "  INSTALL_DEV_TOOLS     Install dev tools (default: 0)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            echo "$ID"
        else
            echo "unknown-linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)

log_info "SCL Core Dependency Installation"
log_info "================================="
log_info "Operating System:  $OS"
log_info "System Deps:       $([ $INSTALL_SYSTEM_DEPS -eq 1 ] && echo 'Yes' || echo 'No')"
log_info "Python Deps:       $([ $INSTALL_PYTHON_DEPS -eq 1 ] && echo 'Yes' || echo 'No')"
log_info "Dev Tools:         $([ $INSTALL_DEV_TOOLS -eq 1 ] && echo 'Yes' || echo 'No')"
log_info "================================="

# Install system dependencies
if [ $INSTALL_SYSTEM_DEPS -eq 1 ]; then
    log_info "Installing system dependencies..."

    case $OS in
        ubuntu|debian)
            log_info "Installing dependencies for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                g++ \
                libhdf5-dev \
                libomp-dev \
                pkg-config

            if [ $INSTALL_DEV_TOOLS -eq 1 ]; then
                log_info "Installing development tools..."
                sudo apt-get install -y \
                    clang-format \
                    clang-tidy
            fi
            ;;

        fedora|rhel|centos)
            log_info "Installing dependencies for Fedora/RHEL/CentOS..."
            sudo dnf install -y \
                gcc-c++ \
                cmake \
                hdf5-devel \
                libomp-devel \
                pkgconfig

            if [ $INSTALL_DEV_TOOLS -eq 1 ]; then
                log_info "Installing development tools..."
                sudo dnf install -y \
                    clang-tools-extra
            fi
            ;;

        arch|manjaro)
            log_info "Installing dependencies for Arch/Manjaro..."
            sudo pacman -S --noconfirm \
                base-devel \
                cmake \
                gcc \
                hdf5 \
                openmp

            if [ $INSTALL_DEV_TOOLS -eq 1 ]; then
                log_info "Installing development tools..."
                sudo pacman -S --noconfirm \
                    clang
            fi
            ;;

        macos)
            log_info "Installing dependencies for macOS..."
            if ! command -v brew &> /dev/null; then
                log_error "Homebrew not found. Please install Homebrew first:"
                log_error "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi

            brew install \
                cmake \
                hdf5 \
                libomp

            if [ $INSTALL_DEV_TOOLS -eq 1 ]; then
                log_info "Installing development tools..."
                brew install \
                    clang-format \
                    llvm
            fi
            ;;

        *)
            log_warn "Unknown OS: $OS"
            log_warn "Please install dependencies manually:"
            log_warn "  - CMake 3.15+"
            log_warn "  - C++20 compiler (GCC 10+ or Clang 12+)"
            log_warn "  - HDF5 development libraries (optional)"
            log_warn "  - OpenMP support (optional)"
            ;;
    esac

    log_info "System dependencies installed successfully"
fi

# Install Python dependencies
if [ $INSTALL_PYTHON_DEPS -eq 1 ]; then
    log_info "Installing Python dependencies..."

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $PYTHON_VERSION"

    # Install pip if not available
    if ! command -v pip3 &> /dev/null; then
        log_info "Installing pip..."
        python3 -m ensurepip --upgrade
    fi

    # Upgrade pip
    python3 -m pip install --upgrade pip

    # Install package in editable mode with all dependencies
    log_info "Installing Python package dependencies..."
    python3 -m pip install -e ".[all]"

    log_info "Python dependencies installed successfully"
fi

# Verify installation
log_info ""
log_info "Verifying installation..."
log_info "================================="

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    log_info "CMake:     $CMAKE_VERSION"
else
    log_warn "CMake:     Not found"
fi

# Check C++ compiler
if command -v g++ &> /dev/null; then
    GXX_VERSION=$(g++ --version | head -n1)
    log_info "g++:       $GXX_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    log_info "clang++:   $CLANG_VERSION"
else
    log_warn "C++ compiler: Not found"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_info "Python:    $PYTHON_VERSION"
else
    log_warn "Python:    Not found"
fi

# Check HDF5
if pkg-config --exists hdf5 2>/dev/null; then
    HDF5_VERSION=$(pkg-config --modversion hdf5)
    log_info "HDF5:      $HDF5_VERSION"
else
    log_warn "HDF5:      Not found (optional)"
fi

log_info "================================="
log_info "Installation complete!"
