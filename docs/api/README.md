# API Documentation Structure

This directory contains the complete API reference for SCL-Core, organized into two main sections:

## Directory Structure

```
api/
├── index.md                 # API landing page (comparison and guide)
│
├── c-api/                   # C-ABI Interface Documentation
│   ├── index.md            # C API overview and conventions
│   ├── core/               # Core types and utilities
│   │   ├── types.md        # Fundamental types (scl_index_t, scl_real_t, etc.)
│   │   ├── error.md        # Error handling (scl_error_t, error messages)
│   │   └── sparse.md       # Sparse matrix types (scl_sparse_matrix_t)
│   ├── kernels/            # Computational kernels
│   │   ├── index.md        # Kernels overview
│   │   └── normalize.md    # Normalization functions
│   └── memory.md           # Memory management (allocation, ownership)
│
└── python/                  # Python API Documentation
    ├── index.md            # Python API overview and quickstart
    ├── preprocessing/      # Data preprocessing (scl.pp)
    │   └── index.md
    ├── neighbors/          # Neighbor graphs (scl.neighbors)
    │   └── index.md
    ├── stats/              # Statistical analysis (scl.stats)
    │   └── index.md
    └── utilities/          # Utility functions (scl.utils)
        └── index.md
```

## Design Philosophy

### Two-Layer Documentation

**C API** - Low-level interface for:
- Language binding developers (R, Julia, etc.)
- Systems integration
- Performance-critical applications
- Direct C/C++ embedding

**Python API** - High-level interface for:
- Bioinformaticians and data scientists
- Standard analysis workflows
- Integration with scanpy/anndata
- Production pipelines

### Documentation Standards

#### C API Pages
- Auto-generated from C header files (future)
- Focus on stability guarantees and ABI compatibility
- Explicit memory management and error handling
- Performance characteristics and thread safety

#### Python API Pages
- Auto-generated from Python docstrings (future)
- Focus on ease of use and integration
- Examples with real biological data
- Best practices and common patterns

### Current Status

All pages are currently **placeholders** with basic structure. Next steps:

1. Implement auto-generation from C headers
2. Implement auto-generation from Python docstrings
3. Add comprehensive examples
4. Add performance benchmarks
5. Add migration guides from scanpy

## Navigation Structure

The VitePress sidebar is organized to show:
1. API Overview (comparison page)
2. Python API section (collapsed by default)
3. C API section (collapsed by default)

This allows users to quickly navigate to their relevant API layer without confusion.

## Maintenance

When adding new functionality:

1. **C API**: Add to appropriate `c-api/*/` directory
2. **Python API**: Add to appropriate `python/*/` directory
3. Update `docs/.vitepress/config.ts` sidebar configuration
4. Ensure index pages link to new content

## Automation Plan

Future automation will:
- Parse C headers in `scl/binding/c/` to generate C API docs
- Parse Python docstrings to generate Python API docs
- Validate documentation completeness in CI
- Check for broken links and outdated examples

