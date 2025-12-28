# registry.hpp

> scl/core/registry.hpp · Unified high-performance memory registry with reference counting

## Overview

This file provides the Registry class, a thread-safe memory registry system that tracks allocated memory and provides automatic cleanup. It supports both simple pointer tracking (refcount=1) and reference-counted buffers (multiple aliases).

Key features:
- Thread-safe memory tracking
- Reference counting for buffer aliases
- Automatic cleanup on unregister
- Sharded design for reduced lock contention
- RAII guard for exception safety

**Header**: `#include "scl/core/registry.hpp"`

---

## Main APIs

### Registry

Thread-safe memory registry with reference counting.

::: source_code file="scl/core/registry.hpp" symbol="Registry" collapsed
:::

**Algorithm Description**

Registry is a sharded hash table that tracks memory allocations:
- **Sharded design**: Reduces lock contention via hash-based sharding
- **Simple pointers**: Tracked with refcount=1, cleaned up immediately on unregister
- **Reference-counted buffers**: Support multiple aliases (BufferID), cleaned up when refcount reaches 0
- **Atomic operations**: Thread-safe counters and reference counts

The registry uses ConcurrentFlatMap internally with striped locks for high-performance concurrent access.

**Edge Cases**

- **Double registration**: Overwrites previous registration (may leak if not unregistered first)
- **Unregister non-existent**: Returns false, no-op
- **Concurrent access**: All operations are thread-safe
- **Memory exhausted**: Registry itself may fail if hash table rehashing fails

**Data Guarantees (Preconditions)**

- Pointers must be valid (non-null) when registered
- Allocation type must match actual allocation method
- Custom deleter must be valid if AllocType::Custom

**Complexity Analysis**

- **Time**: O(1) average case for register/unregister/is_registered (hash table)
- **Space**: ~32 bytes per simple pointer, ~48 bytes per reference-counted buffer

**Example**

```cpp
#include "scl/core/registry.hpp"

auto& reg = scl::get_registry();

// Register simple pointer
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// Use data...

// Unregister and cleanup
reg.unregister_ptr(data);  // Calls delete[] automatically

// Register reference-counted buffer
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr1 = reg.get_buffer(id);  // Get alias
void* ptr2 = reg.get_buffer(id);  // Get another alias

// Unregister aliases
reg.unregister_buffer(id);  // Decrements refcount
reg.unregister_buffer(id);  // Refcount = 0, cleanup happens here
```

---

### register_ptr

Register a simple pointer for tracking (refcount = 1).

::: source_code file="scl/core/registry.hpp" symbol="register_ptr" collapsed
:::

**Algorithm Description**

Registers a pointer with the registry:
1. Hash pointer address to determine shard
2. Acquire shard lock
3. Insert pointer record into hash table
4. Update statistics (atomic increment)

The pointer will be automatically freed on unregister_ptr() using the appropriate deleter based on AllocType.

**Edge Cases**

- **Already registered**: Overwrites previous registration (may leak)
- **Null pointer**: Allowed but not useful (cleanup is no-op)

**Data Guarantees (Preconditions)**

- ptr must be allocated with method matching AllocType
- custom_deleter must be valid if AllocType::Custom

**Complexity Analysis**

- **Time**: O(1) average case
- **Space**: O(1) - stores metadata in hash table

**Example**

```cpp
auto& reg = scl::get_registry();

// Register array allocated with new[]
Real* arr = new Real[1000];
reg.register_ptr(arr, 1000 * sizeof(Real), AllocType::ArrayNew);

// Register aligned allocation
Real* aligned = scl::memory::aligned_alloc<Real>(1000, 64);
reg.register_ptr(aligned, 1000 * sizeof(Real), AllocType::AlignedAlloc);
```

---

### unregister_ptr

Unregister a pointer and free memory.

::: source_code file="scl/core/registry.hpp" symbol="unregister_ptr" collapsed
:::

**Algorithm Description**

Unregisters a pointer and frees memory:
1. Hash pointer address to find shard
2. Acquire shard lock
3. Find pointer in hash table
4. Free memory using appropriate deleter
5. Remove from hash table
6. Update statistics

**Edge Cases**

- **Not registered**: Returns false, no-op
- **Double unregister**: Returns false on second call
- **Null pointer**: Safe, returns false

**Data Guarantees (Preconditions)**

- ptr must be registered (or nullptr, in which case returns false)

**Complexity Analysis**

- **Time**: O(1) average case
- **Space**: O(1)

**Example**

```cpp
auto& reg = scl::get_registry();
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// Use data...

// Unregister and cleanup
bool success = reg.unregister_ptr(data);  // Returns true, calls delete[]
// data is now invalid
```

---

### new_buffer

Create a new reference-counted buffer.

::: source_code file="scl/core/registry.hpp" symbol="new_buffer" collapsed
:::

**Algorithm Description**

Allocates memory and registers as reference-counted buffer:
1. Allocate memory based on AllocType
2. Create BufferID (unique identifier)
3. Register in hash table with refcount = 1
4. Return BufferID

Multiple aliases can be obtained via get_buffer() using the same BufferID. The buffer is only freed when refcount reaches 0.

**Edge Cases**

- **Allocation failure**: Returns invalid BufferID (0), check with is_valid_buffer_id()
- **Zero size**: May return invalid BufferID

**Data Guarantees (Preconditions)**

- size > 0 (typically)
- AllocType must be valid

**Complexity Analysis**

- **Time**: O(1) average case (hash table insertion) + allocation time
- **Space**: O(size) for allocated memory + O(1) for metadata

**Example**

```cpp
auto& reg = scl::get_registry();

// Create buffer
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
if (!reg.is_valid_buffer_id(id)) {
    // Handle allocation failure
    return;
}

// Get aliases
void* ptr1 = reg.get_buffer(id);
void* ptr2 = reg.get_buffer(id);  // Same memory, refcount = 3
```

---

### get_buffer

Get pointer alias to a reference-counted buffer (increments refcount).

::: source_code file="scl/core/registry.hpp" symbol="get_buffer" collapsed
:::

**Algorithm Description**

Returns a pointer to the buffer and increments reference count:
1. Hash BufferID to find shard
2. Acquire shard lock
3. Find buffer in hash table
4. Increment atomic refcount
5. Return pointer

**Edge Cases**

- **Invalid BufferID**: Returns nullptr
- **Already freed**: Returns nullptr (race condition handled safely)

**Data Guarantees (Preconditions)**

- BufferID must be valid (obtained from new_buffer)

**Complexity Analysis**

- **Time**: O(1) average case
- **Space**: O(1)

**Example**

```cpp
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr = reg.get_buffer(id);  // Refcount = 2 (1 from new_buffer + 1 from get_buffer)
```

---

### unregister_buffer

Unregister a buffer alias (decrements refcount, frees when refcount = 0).

::: source_code file="scl/core/registry.hpp" symbol="unregister_buffer" collapsed
:::

**Algorithm Description**

Decrements reference count and frees buffer when count reaches 0:
1. Hash BufferID to find shard
2. Acquire shard lock
3. Find buffer in hash table
4. Decrement atomic refcount
5. If refcount == 0: free memory and remove from table
6. Otherwise: just decrement counter

**Edge Cases**

- **Invalid BufferID**: Returns false
- **Double unregister**: Safe, refcount cannot go below 0
- **Unregister from multiple threads**: Thread-safe, atomic operations

**Data Guarantees (Preconditions)**

- BufferID must be valid

**Complexity Analysis**

- **Time**: O(1) average case
- **Space**: O(1)

**Example**

```cpp
BufferID id = reg.new_buffer(1000 * sizeof(Real), AllocType::ArrayNew);
void* ptr1 = reg.get_buffer(id);  // Refcount = 2
void* ptr2 = reg.get_buffer(id);  // Refcount = 3

reg.unregister_buffer(id);  // Refcount = 2
reg.unregister_buffer(id);  // Refcount = 1
reg.unregister_buffer(id);  // Refcount = 0, memory freed here
```

---

## Utility Classes

### RegistryGuard

RAII guard for automatic unregistration.

::: source_code file="scl/core/registry.hpp" symbol="RegistryGuard" collapsed
:::

**Algorithm Description**

RAII wrapper that automatically unregisters a pointer on scope exit:
- Constructor: Stores pointer (does not register)
- Destructor: Calls unregister_ptr() if pointer is still held
- release(): Prevents automatic unregistration

Useful for exception-safe code where unregistration must happen even if exceptions occur.

**Example**

```cpp
auto& reg = scl::get_registry();
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

{
    RegistryGuard guard(data);
    // Use data...
    // Automatic cleanup on scope exit (even if exception thrown)
}
```

---

### get_registry

Get reference to global Registry instance.

**Returns**: Reference to singleton Registry instance

**Thread Safety**: Safe - singleton initialization is thread-safe

**Example**

```cpp
auto& reg = scl::get_registry();
reg.register_ptr(ptr, size, AllocType::ArrayNew);
```

---

## Type Aliases

```cpp
using BufferID = std::uint64_t;  // Unique identifier for reference-counted buffers
using HandlerRegistry = Registry;  // Legacy alias
```

## Enum: AllocType

```cpp
enum class AllocType {
    ArrayNew,      // new[] / delete[]
    ScalarNew,     // new / delete
    AlignedAlloc,  // aligned_alloc / aligned_free
    Custom         // Custom deleter
};
```

## Design Notes

### Sharded Architecture

Registry uses hash-based sharding to reduce lock contention:
- Multiple shards (typically 16-64)
- Pointer hash determines shard
- Each shard has its own lock
- Reduces contention by factor of num_shards

### Reference Counting

Reference-counted buffers support multiple aliases:
- Each get_buffer() increments refcount
- Each unregister_buffer() decrements refcount
- Buffer freed when refcount reaches 0
- Thread-safe using atomic operations

### Thread Safety

All public methods are thread-safe:
- Internal synchronization via striped locks
- Atomic reference counts
- Concurrent reads supported
- Write operations are mutually exclusive per shard

## See Also

- [Memory Management](./memory) - Allocation functions that return pointers for registration
- [Sparse Matrix](./sparse) - Uses Registry for metadata array tracking
