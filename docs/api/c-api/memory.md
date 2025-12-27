# Memory Management

Documentation for C API memory management.

::: tip Status
This page is under construction.
:::

## Overview

Memory management conventions and best practices for the C API.

## Ownership Semantics

### Function Naming Conventions
- `_create` - Caller owns result, must call `_destroy`
- `_destroy` - Release resources
- `_borrow` - Temporary access, no ownership transfer
- `_transfer` - Ownership transferred to callee
- `_copy` - Creates new owned copy

## Allocation Functions

### `scl_alloc()`
Allocate aligned memory.

### `scl_free()`
Free allocated memory.

## Best Practices

1. Always check return values
2. Match every `_create` with a `_destroy`
3. Handle errors before cleanup
4. Use RAII wrappers in C++

## Coming Soon

Full documentation of memory management patterns.

