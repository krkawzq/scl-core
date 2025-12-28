# CI Configuration

This directory contains continuous integration scripts and configurations for the scl-core project.

## Structure

- `build.sh` - Build script for compiling C++ libraries
- `test.sh` - Test runner for C++ and Python tests
- `lint.sh` - Code quality checks (formatting, linting)
- `install-deps.sh` - Dependency installation script

## Usage

### Local Testing

You can run CI scripts locally to verify changes before pushing:

```bash
# Build the project
./CI/build.sh

# Run tests
./CI/test.sh

# Run linters
./CI/lint.sh
```

### Environment Variables

- `BUILD_TYPE` - CMake build type (Release|Debug), default: Release
- `SCL_THREADING_BACKEND` - Threading backend (AUTO|SERIAL|BS|OPENMP|TBB), default: AUTO
- `PYTHON_VERSION` - Python version for testing (3.10|3.11|3.12), default: system python

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Build Stage**: Compile C++ libraries for multiple platforms
2. **Test Stage**: Run C++ and Python tests
3. **Lint Stage**: Check code quality and formatting

## Platform Support

- Linux (Ubuntu 22.04)
- macOS (latest)
- Windows (planned)
