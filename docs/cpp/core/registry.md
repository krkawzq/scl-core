# Registry

The `Registry` class provides centralized memory tracking and lifetime management for Python integration and debugging.

## Overview

`Registry` enables:

- **Memory tracking** - Track all allocated buffers
- **Reference counting** - Shared buffers with automatic cleanup
- **Python integration** - Zero-copy memory transfer
- **Leak detection** - Debug-mode warnings for leaked memory
- **Thread safety** - Concurrent access without data races

## Basic Usage

### Allocating Memory

```cpp
#include "scl/core/registry.hpp"

auto& reg = scl::get_registry();

// Allocate array
Real* data = reg.new_array<Real>(1000);

// Use data...

// Cleanup
reg.unregister_ptr(data);
```

### Registering Existing Memory

```cpp
// Allocate with new[]
Real* data = new Real[1000];

// Register
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// Use data...

// Cleanup (calls delete[])
reg.unregister_ptr(data);
```

## Allocation Types

```cpp
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → aligned_free
    Custom         // Custom deleter function
};
```

### ArrayNew

For arrays allocated with `new[]`:

```cpp
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// Cleanup calls delete[]
reg.unregister_ptr(data);
```

### ScalarNew

For single objects allocated with `new`:

```cpp
MyClass* obj = new MyClass();
reg.register_ptr(obj, sizeof(MyClass), AllocType::ScalarNew);

// Cleanup calls delete
reg.unregister_ptr(obj);
```

### AlignedAlloc

For aligned memory:

```cpp
#include "scl/core/memory.hpp"

Real* data = scl::memory::aligned_alloc<Real>(1000, 64);
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::AlignedAlloc);

// Cleanup calls aligned_free
reg.unregister_ptr(data);
```

### Custom

For custom allocation:

```cpp
void my_deleter(void* ptr) {
    // Custom cleanup logic
    my_free(ptr);
}

void* data = my_alloc(1000);
reg.register_ptr(data, 1000, AllocType::Custom, my_deleter);

// Cleanup calls my_deleter
reg.unregister_ptr(data);
```

## Reference Counting

For shared buffers with multiple aliases:

```cpp
// Allocate main buffer
Real* main_ptr = new Real[1000];

// Create aliases (e.g., column views)
std::vector<void*> aliases;
for (size_t i = 0; i < 10; ++i) {
    aliases.push_back(main_ptr + i * 100);
}

// Register with reference counting
BufferID id = reg.register_buffer_with_aliases(
    main_ptr,                    // Real pointer to free
    1000 * sizeof(Real),         // Byte size
    aliases,                     // Alias pointers
    AllocType::ArrayNew          // Allocation type
);

// Refcount = 11 (main + 10 aliases)

// Unregister aliases
for (auto* alias : aliases) {
    reg.unregister_ptr(alias);  // Decrements refcount
}

// Unregister main pointer
reg.unregister_ptr(main_ptr);  // Refcount = 0, frees memory
```

## Python Integration

### Zero-Copy Transfer

Transfer ownership to Python without copying:

```cpp
// C++ side: Allocate and register
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(1000);

// ... fill data ...

// Python binding: Transfer ownership
py::capsule deleter(data, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});

return py::array_t<Real>(
    {1000},           // Shape
    {sizeof(Real)},   // Strides
    data,             // Data pointer
    deleter           // Cleanup callback
);
```

**Flow:**
1. C++ allocates and registers memory
2. Python takes ownership via capsule
3. When Python array is deleted, capsule calls deleter
4. Deleter unregisters from Registry
5. Registry frees memory

### Shared Buffers

For sparse matrices with block allocation:

```cpp
// C++ side: Register with aliases
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, aliases, AllocType::ArrayNew);

// Python side: Each alias gets a separate array
for (auto* alias : aliases) {
    py::capsule deleter(alias, [](void* ptr) {
        scl::get_registry().unregister_ptr(ptr);
    });
    
    // Create Python array for this alias
    py::array_t<Real> arr(
        {alias_size},
        {sizeof(Real)},
        alias,
        deleter
    );
    
    // Deleter decrements refcount when array is deleted
}

// Memory freed when last Python array is deleted
```

## Query Operations

### Check Registration

```cpp
if (reg.is_registered(ptr)) {
    std::cout << "Pointer is registered\n";
}
```

### Get Statistics

```cpp
size_t num_ptrs = reg.get_num_pointers();
size_t num_buffers = reg.get_num_buffers();
size_t total_bytes = reg.get_total_bytes();

std::cout << "Registered: " << num_ptrs << " pointers, "
          << num_buffers << " buffers, "
          << total_bytes << " bytes\n";
```

### Get Pointer Info

```cpp
if (auto record = reg.get_ptr_record(ptr)) {
    std::cout << "Size: " << record->byte_size << " bytes\n";
    std::cout << "Type: " << static_cast<int>(record->type) << "\n";
}
```

### Get Buffer Info

```cpp
if (auto buffer = reg.get_buffer(ptr)) {
    std::cout << "Real ptr: " << buffer->info.real_ptr << "\n";
    std::cout << "Size: " << buffer->info.byte_size << " bytes\n";
    std::cout << "Refcount: " << buffer->refcount.load() << "\n";
}
```

## Architecture

### Sharded Design

Registry uses sharding to reduce lock contention:

```
Registry
├── Shard 0 (hash % num_shards == 0)
│   ├── PtrMap: { ptr → PtrRecord }
│   └── BufferMap: { ptr → RefCountedBuffer }
├── Shard 1
│   ├── PtrMap
│   └── BufferMap
└── ...
```

**Benefits:**
- Parallel access to different shards
- Reduced lock contention
- Better cache locality per shard

**Default shards:** `std::thread::hardware_concurrency()`

### Thread Safety

All public methods are thread-safe:

```cpp
// Safe: Concurrent registration
parallel_for(Size(0), n, [&](size_t i) {
    Real* data = reg.new_array<Real>(100);
    // ...
    reg.unregister_ptr(data);
});
```

**Internal synchronization:**
- Shared mutex for rehashing (readers can proceed concurrently)
- Striped locks for slot-level access (reduces contention)
- Atomic operations for counters and reference counts

## Performance

### Overhead

- **Per-pointer:** ~32 bytes (hash table slot + metadata)
- **Per-buffer:** ~48 bytes (RefCountedBuffer + hash table slot)
- **Lookup:** O(1) average, O(n) worst case (hash collision)

### When to Use Registry

**Use Registry for:**
- Memory transferred to Python
- Shared buffers with multiple aliases
- Long-lived allocations
- Debugging memory leaks

**Don't use Registry for:**
- Stack-allocated buffers
- Short-lived temporaries in hot loops
- Memory managed by external libraries
- Performance-critical hot paths (use pre-allocated workspaces)

## Debugging

### Leak Detection

In debug builds, Registry warns about leaked memory:

```cpp
// At program exit
~Registry() {
    #ifdef SCL_DEBUG
    if (get_num_pointers() > 0 || get_num_buffers() > 0) {
        std::cerr << "WARNING: Memory leak detected!\n";
        std::cerr << "  Pointers: " << get_num_pointers() << "\n";
        std::cerr << "  Buffers: " << get_num_buffers() << "\n";
        std::cerr << "  Bytes: " << get_total_bytes() << "\n";
    }
    #endif
}
```

### Print Statistics

```cpp
void print_registry_stats() {
    auto& reg = scl::get_registry();
    
    std::cout << "Registry Statistics:\n";
    std::cout << "  Pointers:  " << reg.get_num_pointers() << "\n";
    std::cout << "  Buffers:   " << reg.get_num_buffers() << "\n";
    std::cout << "  Bytes:     " << reg.get_total_bytes() << "\n";
    std::cout << "  Avg size:  " 
              << (reg.get_num_pointers() > 0 
                  ? reg.get_total_bytes() / reg.get_num_pointers() 
                  : 0) 
              << " bytes\n";
}
```

## Best Practices

### 1. Use RAII

Wrap Registry operations in RAII guards:

```cpp
class RegistryGuard {
    void* ptr_;
    
public:
    explicit RegistryGuard(void* ptr) : ptr_(ptr) {}
    
    ~RegistryGuard() {
        if (ptr_) {
            scl::get_registry().unregister_ptr(ptr_);
        }
    }
    
    void* release() {
        void* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    
    RegistryGuard(const RegistryGuard&) = delete;
    RegistryGuard& operator=(const RegistryGuard&) = delete;
};

// Usage
auto* data = reg.new_array<Real>(1000);
RegistryGuard guard(data);
// Automatic cleanup on scope exit
```

### 2. Prefer new_array

Use `new_array` for automatic registration:

```cpp
// GOOD: Automatic registration
auto* data = reg.new_array<Real>(1000);

// BAD: Manual registration (error-prone)
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);
```

### 3. Match Allocation Type

Ensure allocation type matches deallocation:

```cpp
// GOOD: Correct type
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ArrayNew);

// BAD: Wrong type (undefined behavior!)
Real* data = new Real[1000];
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::ScalarNew);
```

### 4. Document Ownership

Clearly document who owns memory:

```cpp
// Returns registered pointer - caller must unregister
Real* allocate_buffer(size_t n);

// Takes ownership of registered pointer
void consume_buffer(Real* ptr);

// Returns non-owning view - do not unregister
const Real* get_data_view() const;
```

### 5. Avoid Registry in Hot Loops

```cpp
// BAD: Registry operations in hot loop
for (size_t i = 0; i < n; ++i) {
    Real* temp = reg.new_array<Real>(100);  // Expensive!
    // ...
    reg.unregister_ptr(temp);
}

// GOOD: Pre-allocate workspace
Real* workspace = reg.new_array<Real>(100);
for (size_t i = 0; i < n; ++i) {
    // Reuse workspace
}
reg.unregister_ptr(workspace);
```

## Advanced Usage

### Custom Shard Count

```cpp
// Create registry with custom shard count
Registry reg(32);  // 32 shards
```

### Batch Operations

For better performance, batch registrations:

```cpp
std::vector<Real*> pointers;

// Allocate all
for (size_t i = 0; i < n; ++i) {
    pointers.push_back(reg.new_array<Real>(100));
}

// Use pointers...

// Cleanup all
for (auto* ptr : pointers) {
    reg.unregister_ptr(ptr);
}
```

### Reference Counting Details

```cpp
// Register buffer
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, aliases, AllocType::ArrayNew);

// Query refcount
if (auto buffer = reg.get_buffer(real_ptr)) {
    uint32_t refcount = buffer->refcount.load();
    std::cout << "Refcount: " << refcount << "\n";
}

// Unregister decrements refcount
for (auto* alias : aliases) {
    reg.unregister_ptr(alias);
    
    if (auto buffer = reg.get_buffer(real_ptr)) {
        std::cout << "Refcount now: " << buffer->refcount.load() << "\n";
    }
}
```

## Global Registry

SCL-Core provides a global registry instance:

```cpp
Registry& get_registry();
```

**Usage:**

```cpp
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(1000);
```

**Thread safety:** The global registry is thread-safe.

**Lifetime:** The global registry is destroyed at program exit.

---

::: tip Memory Safety
Always pair allocations with cleanup. Use RAII guards to ensure cleanup even in the presence of exceptions.
:::

