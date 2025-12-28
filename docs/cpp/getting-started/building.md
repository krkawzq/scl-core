# Building from Source

This guide provides detailed instructions for building SCL-Core from source.

## System Requirements

### Compiler Support

| Compiler | Minimum Version | Recommended |
|----------|----------------|-------------|
| GCC | 9.0 | 11.0+ |
| Clang | 10.0 | 14.0+ |
| MSVC | 19.14 (VS 2017 15.7) | 19.29+ (VS 2019) |

### Dependencies

**Required:**
- CMake >= 3.15
- C++17 standard library

**Optional:**
- Python >= 3.8 (for Python bindings)
- Google Test (for testing, auto-downloaded if not found)
- Google Benchmark (for benchmarks, auto-downloaded if not found)

## Basic Build

### Clone Repository

```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
```

### Configure and Build

```bash
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### Run Tests

```bash
ctest --output-on-failure
```

## CMake Options

### Precision Configuration

```bash
# Float32 (default)
cmake -DSCL_USE_FLOAT32=ON ..

# Float64
cmake -DSCL_USE_FLOAT64=ON ..

# Float16 (requires GCC >= 12 or Clang >= 15)
cmake -DSCL_USE_FLOAT16=ON ..
```

### Index Type Configuration

```bash
# Int32 (default)
cmake -DSCL_USE_INT32=ON ..

# Int64 (for very large matrices)
cmake -DSCL_USE_INT64=ON ..

# Int16 (for small matrices, memory-constrained)
cmake -DSCL_USE_INT16=ON ..
```

### SIMD Configuration

```bash
# Auto-detect best available (default)
cmake ..

# Force specific target
cmake -DSCL_SIMD_TARGET=AVX2 ..
cmake -DSCL_SIMD_TARGET=AVX512 ..
cmake -DSCL_SIMD_TARGET=NEON ..

# Disable SIMD
cmake -DSCL_ENABLE_SIMD=OFF ..
```

### Build Type

```bash
# Debug (default, with assertions)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release (optimized, no assertions)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

# Minimum size release
cmake -DCMAKE_BUILD_TYPE=MinSizeRel ..
```

### Optional Components

```bash
# Build tests (default: ON)
cmake -DSCL_BUILD_TESTS=ON ..

# Build benchmarks (default: OFF)
cmake -DSCL_BUILD_BENCHMARKS=ON ..

# Build Python bindings (default: OFF)
cmake -DSCL_BUILD_PYTHON=ON ..

# Build examples (default: OFF)
cmake -DSCL_BUILD_EXAMPLES=ON ..
```

### Compiler Flags

```bash
# Enable all warnings
cmake -DSCL_ENABLE_WARNINGS=ON ..

# Treat warnings as errors
cmake -DSCL_WARNINGS_AS_ERRORS=ON ..

# Enable sanitizers (Debug builds)
cmake -DSCL_ENABLE_ASAN=ON ..      # Address sanitizer
cmake -DSCL_ENABLE_UBSAN=ON ..     # Undefined behavior sanitizer
cmake -DSCL_ENABLE_TSAN=ON ..      # Thread sanitizer
```

## Platform-Specific Instructions

### Linux

**Ubuntu/Debian:**

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake git

# Build
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

**Fedora/RHEL:**

```bash
# Install dependencies
sudo dnf install gcc-c++ cmake git

# Build (same as above)
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake
brew install cmake

# Build
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

**Visual Studio:**

```powershell
# Open Developer Command Prompt for VS

# Clone
git clone https://github.com/krkawzq/scl-core.git
cd scl-core

# Configure
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..

# Build
cmake --build . --config Release

# Install
cmake --install . --config Release
```

**MinGW:**

```bash
# Install MinGW-w64 and CMake

# Build
mkdir build && cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
mingw32-make -j$(nproc)
```

## Advanced Configuration

### Custom Install Prefix

```bash
cmake -DCMAKE_INSTALL_PREFIX=/custom/path ..
cmake --build .
cmake --install .
```

### Cross-Compilation

```bash
# ARM64 cross-compilation
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-arm64.cmake ..
```

### Static vs Shared Library

```bash
# Static library (default)
cmake -DBUILD_SHARED_LIBS=OFF ..

# Shared library
cmake -DBUILD_SHARED_LIBS=ON ..
```

### Link-Time Optimization

```bash
# Enable LTO/IPO
cmake -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON ..
```

### Custom Compiler

```bash
# Use specific compiler
cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
cmake -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14 ..
```

## Building Python Bindings

### Requirements

- Python >= 3.8
- pybind11 (auto-downloaded if not found)
- NumPy

### Build

```bash
# Configure with Python support
cmake -DSCL_BUILD_PYTHON=ON ..

# Build
cmake --build .

# Install (optional)
cmake --install .

# Or install Python package
cd python
pip install -e .
```

### Test Python Bindings

```bash
python -c "import scl; print(scl.__version__)"
```

## Building Documentation

### Requirements

- Doxygen
- Sphinx (for Python docs)

### Build

```bash
# Configure
cmake -DSCL_BUILD_DOCS=ON ..

# Build
cmake --build . --target docs

# Output in build/docs/html/
```

## Troubleshooting

### CMake Can't Find Compiler

```bash
# Specify compiler explicitly
export CC=gcc-11
export CXX=g++-11
cmake ..
```

### SIMD Not Detected

```bash
# Check CPU features
cat /proc/cpuinfo | grep flags  # Linux
sysctl -a | grep cpu.features   # macOS

# Force SIMD target
cmake -DSCL_SIMD_TARGET=AVX2 ..
```

### Link Errors

```bash
# Clean and rebuild
rm -rf build
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Out of Memory During Build

```bash
# Reduce parallel jobs
cmake --build . -j2
```

### Python Bindings Not Found

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in user directory
pip install --user -e python/
```

## Verification

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Run Benchmarks

```bash
cd build
./benchmarks/benchmark_normalize
./benchmarks/benchmark_neighbors
```

### Check Installation

```bash
# Check headers
ls /usr/local/include/scl/

# Check library
ls /usr/local/lib/libscl*

# Test compilation
cat > test.cpp << EOF
#include <scl/core/type.hpp>
#include <iostream>
int main() {
    std::cout << "SCL-Core Real type: " << scl::DTYPE_NAME << std::endl;
    return 0;
}
EOF

g++ -std=c++17 test.cpp -o test -lscl
./test
```

## Performance Optimization

### Compiler Flags

```bash
# GCC/Clang
cmake -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" ..

# MSVC
cmake -DCMAKE_CXX_FLAGS="/O2 /arch:AVX2" ..
```

### Profile-Guided Optimization

```bash
# Step 1: Build with profiling
cmake -DCMAKE_CXX_FLAGS="-fprofile-generate" ..
cmake --build .

# Step 2: Run benchmarks to generate profile
./benchmarks/benchmark_all

# Step 3: Rebuild with profile
cmake -DCMAKE_CXX_FLAGS="-fprofile-use" ..
cmake --build .
```

---

::: tip Build Performance
Use `-j$(nproc)` to parallelize builds. For very large builds, consider using `ccache` to speed up recompilation.
:::

