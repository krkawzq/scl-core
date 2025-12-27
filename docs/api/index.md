# API Reference

SCL-Core provides two levels of API documentation to serve different use cases:

## Python API

High-level Python interface for biological data analysis. This is the primary interface for most users.

[Explore Python API →](/api/python/)

**Target Users:**
- Bioinformaticians and computational biologists
- Data scientists working with single-cell data
- Python developers integrating SCL-Core into analysis pipelines

**Features:**
- Pythonic interface with NumPy/SciPy integration
- Automatic memory management
- Type hints and comprehensive docstrings
- Seamless integration with scanpy and anndata

---

## C API

Low-level C-ABI interface for language bindings and direct system integration.

[Explore C API →](/api/c-api/)

**Target Users:**
- Developers creating language bindings (R, Julia, etc.)
- Systems programmers requiring direct C integration
- Performance-critical applications needing fine-grained control

**Features:**
- Stable C-ABI for cross-language compatibility
- Zero-overhead performance
- Manual memory management with clear ownership semantics
- Direct access to all internal kernels

---

## Choosing the Right API

| Use Case | Recommended API |
|----------|----------------|
| Standard biological analysis | **Python API** |
| Building analysis pipelines | **Python API** |
| Creating R/Julia bindings | **C API** |
| Embedding in C/C++ applications | **C API** |
| Performance-critical scenarios | **C API** |

---

## Quick Links

### Python API
- [Getting Started](/api/python/#getting-started)
- [Preprocessing](/api/python/preprocessing/)
- [Neighbors](/api/python/neighbors/)
- [Statistics](/api/python/stats/)

### C API
- [Core Types](/api/c-api/core/types)
- [Error Handling](/api/c-api/core/error)
- [Kernels Overview](/api/c-api/kernels/)
- [Memory Management](/api/c-api/memory)

