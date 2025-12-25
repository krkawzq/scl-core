# Third-Party Dependencies

This directory is reserved for third-party library source code or headers.

The SCL project uses CMake FetchContent to automatically download dependencies:
- BS::thread_pool (header-only)
- Google Highway (SIMD library)

Compiled binaries are placed in:
- `build/lib/` - Shared libraries (.so, .dylib, .dll)
- `build/bin/` - Executables

## Adding New Dependencies

If you need to add a local dependency:
1. Place the source code in this directory
2. Update CMakeLists.txt to include the dependency

## Manually Installing Dependencies

### Google Highway (SIMD Library)

To install Highway locally instead of using FetchContent:

```bash
# Option 1: Use the setup script
./scripts/setup_cpp_dependencies.sh

# Option 2: Manual installation
cd libs/
git clone --depth 1 --branch 1.0.7 https://github.com/google/highway.git
cd ..
```

Once installed in `libs/highway/`, CMake will automatically use the local copy.

### Benefits of Local Installation

- **Faster builds**: No re-downloading on clean builds
- **Offline development**: Work without internet connection
- **Version control**: Lock to specific versions
- **Patching**: Apply custom patches if needed
