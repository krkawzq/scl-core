# Getting Started

Welcome to SCL-Core development! This guide will help you set up your development environment and start contributing.

## Prerequisites

### Required

- **C++17 compiler**:
  - GCC >= 9.0
  - Clang >= 10.0
  - MSVC >= 19.14 (Visual Studio 2017 15.7)

- **CMake** >= 3.15

- **Git**

### Optional

- **Python** >= 3.8 (for Python bindings)
- **Google Test** (for testing)
- **Doxygen** (for documentation)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
```

### 2. Build

```bash
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### 3. Run Tests

```bash
ctest --output-on-failure
```

## Development Environment

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake git

# Clone and build
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# Install dependencies
brew install cmake

# Clone and build
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

```powershell
# Install Visual Studio 2019 or later with C++ support
# Install CMake from https://cmake.org/download/

# Clone and build
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Build Configuration

### Precision Selection

```bash
# Float32 (default)
cmake -DSCL_USE_FLOAT32=ON ..

# Float64
cmake -DSCL_USE_FLOAT64=ON ..

# Float16 (requires GCC >= 12 or Clang >= 15)
cmake -DSCL_USE_FLOAT16=ON ..
```

### Index Type Selection

```bash
# Int32 (default)
cmake -DSCL_USE_INT32=ON ..

# Int64
cmake -DSCL_USE_INT64=ON ..

# Int16 (for small matrices)
cmake -DSCL_USE_INT16=ON ..
```

### SIMD Configuration

```bash
# Auto-detect (default)
cmake ..

# Force AVX2
cmake -DSCL_SIMD_TARGET=AVX2 ..

# Force AVX-512
cmake -DSCL_SIMD_TARGET=AVX512 ..

# Disable SIMD
cmake -DSCL_ENABLE_SIMD=OFF ..
```

### Build Type

```bash
# Debug (default)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## Project Structure

```
scl-core/
├── scl/                  # Source code
│   ├── core/             # Core types and utilities
│   ├── threading/        # Parallel processing
│   ├── kernel/           # Computational operators
│   ├── math/             # Statistical functions
│   ├── mmap/             # Memory-mapped arrays
│   └── io/               # I/O utilities
├── tests/                # Unit tests
├── benchmarks/           # Performance benchmarks
├── docs/                 # Documentation
├── examples/             # Example code
├── CMakeLists.txt        # Build configuration
└── README.md             # Project overview
```

## Next Steps

- [Building from Source](/cpp/getting-started/building) - Detailed build instructions
- [Contributing Guide](/cpp/getting-started/contributing) - Code standards and workflow
- [Testing Guide](/cpp/getting-started/testing) - Writing and running tests
- [Architecture Overview](/cpp/architecture/) - Understand the design

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/krkawzq/scl-core/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions)
- **Contributing**: See the [Contributing Guide](/cpp/getting-started/contributing)

---

::: tip First Time?
Start with the [Contributing Guide](/cpp/getting-started/contributing) to understand our code standards and workflow.
:::

