# scl/core/memory.hpp 中文详细笔记

> 本文件是对 `scl/core/memory.hpp` 头文件全部内容的细致中文笔记，包括所有常量、类型、函数及类字段。建议掌握 C++20 的基础，配合代码和文档查阅。

---

## 1. 内存配置常量

- **`DEFAULT_ALIGNMENT`**  
  默认内存对齐大小（64 字节，适合 AVX-512/CPU 缓存行对齐）。

- **`CACHE_LINE_SIZE`**  
  现代处理器缓存行（Cache line）大小，常为 64 字节（x86/ARM 同理）。

- **`STREAM_THRESHOLD`**  
  启用“流式非时序拷贝”时的阈值：大于 256KB 时才采用（避免小块数据带来的过多开销）。

- **`STREAM_ALIGNMENT`**  
  “流式拷贝”操作的对齐要求：64 字节。

- **`DEFAULT_PREFETCH_DISTANCE`**  
  预取内存块时的默认步长（以元素为单位，默认8个）。

- **`DEFAULT_MAX_PREFETCHES`**  
  每次最多预取多少步（默认16）。

- **`PAGE_SIZE`**  
  标准内存页大小（4096 字节）。

- **`HUGE_PAGE_SIZE`**  
  Huge page 的大小（一般为 2MB，支持大页内存的系统）。

- **`LARGE_ALLOC_THRESHOLD`**  
  超过此阈值（1MB）将采用大块内存分配器（如 mmap/VirtualAlloc）。

- **`MMAP_THRESHOLD`**  
  超过64KB时，考虑直接用 mmap 或 VirtualAlloc 分配内存。

---

## 2. 对齐工具函数（align up/down/is aligned）

全部由 `scl::bits` 命名空间提供，常用于指针和 size_t 的对齐计算：

- **`align_up`**  
  向上对齐到指定倍数。
- **`align_down`**  
  向下对齐。
- **`is_aligned`**  
  检查指针/长度是否对齐。

示例：`bits::align_up(addr, 64)` 返回对齐到64字节的地址。

---

## 3. 对齐内存分配器

### `template<typename T> struct AlignedDeleter`
对齐内存的删除器，给 unique_ptr 用。  
**字段：**
- `alignment_`：对齐字节数。

**方法：**
- 构造函数：可指定对齐字节数，默认 64。
- `operator()(T* ptr)`：释放对齐分配的内存。自动调用平台的释放方法。

### `template<typename T> using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>`
管理对齐数组的智能指针。

### `template<typename T> aligned_alloc`
分配对齐的 T 类型数组，返回 AlignedPtr。
- 零初始化（针对算术类型）。
- 平台依赖实现：Windows 用 `_aligned_malloc`，POSIX 用 `posix_memalign`，C++17 算术类型可用 `operator new` 对齐分配。
- 分配失败返回 nullptr。

### `template<typename T> aligned_free`
辅助函数，用于释放对齐分配的内存，等价于手动析构 unique_ptr。

---

## 4. 对齐缓冲区封装类

### `template<typename T> class AlignedBuffer`
RAII 对齐内存分配，封装为 std::span 兼容接口。

**主要字段：**
- `ptr_`：智能指针（内部用类前面介绍的 AlignedPtr）。
- `count_`：元素数量。

**构造/析构/移动：**
- 构造：指定元素数和对齐字节数。
- 禁用拷贝，允许移动。

**接口方法：**
- `span / operator std::span()`：获取 span 视图。
- `data()`：获取原生指针。
- `size(), empty(), bool`：基础检查。
- `operator[](i)`：索引访问。
- `begin/end/cbegin/cend`：STL 风格迭代。

---

## 5. 大块内存分配（虚拟地址分配 VirtualAlloc/mmap）

### 枚举类型 `enum class AllocFlags`
大块内存分配的标志。可位或。
- `None`
- `HugePages`（使用大页分配）
- `Executable`（允许执行权限，执行 JIT 代码时用）
- `ReadOnly`
- `NoReserve`（Linux，不预留交换区）

位操作有：
- `operator|` 与 `operator&` 支持组合与测试；
- `has_flag(flags, test)` 用来快速测试某个标志位是否被设置。

### `void* virtual_alloc`
大块内存的分配器，使用操作系统的虚拟内存分配（Windows/Unix）。
- 参数：字节数、分配标志。
- 返回：指向新分配内存的指针（出错返回 nullptr）。
- 自动对齐到页/大页边界。
- 内存自动归零。

### `void virtual_free`
释放 virtual_alloc 分配的内存，需传递字节数。

### `template<typename T> class VirtualBuffer`
RAII 包装的虚拟内存分配（大块分配）。
- 字段：`T* ptr_`，`Size count_`，`Size byte_size_`
- span 兼容基本接口
- 析构释放内存
- 允许移动，不可拷贝

---

## 6. 填充与清零操作

### `template<typename T> fill`
用指定值填充 span。如果类型是单字节，采用 memset，否则用 std::fill。

### `template<typename T> zero`
零初始化 span。trivially_copyable 类型用 memset，否则 std::fill。

---

## 7. 拷贝函数

### `template<typename T> copy_fast`
高性能拷贝，无重叠安全检查。trivially_copyable 类型用 memcpy，否则 std::copy。

### `template<typename T> copy`
带重叠安全检查的拷贝。会区分向前/向后以避免覆盖（用 memmove），
非平凡类型用 std::copy / copy_backward。

### `template<typename T> stream_copy`
核心高性能函数。大块且对齐的拷贝采用平台 SIMD 的非时序 NT store：
- x86: AVX-512/AVX/SSE2 使用流式存储命令。
- ARM NEON 无真正 NT store，用普通存储+预取。
- 结束后调用 memory fence 保证可见性。
- 小块或未对齐降级为普通 memcpy。

---

## 8. 预取工具

### `prefetch_read`
给定 span，提升 CPU 缓存友好性，可设置预取局部性等级（如 L1/L2/L3对应 3/2/1/0）。

### `prefetch_write`
同上，针对可写目标。

### `prefetch_ahead`
在遍历时预取之后的若干元素。当遍历到 i 时，预取 i+Distance（默认8）。

---

## 9. 内存内容比较

### `equal`
判断两个 span 内容是否字节相等。平凡类型用 memcmp，否则 std::equal。

### `compare`
字节序比较，用于排序判断。已弃用（多字节类型在 memcmp 下会因大小端差异导致错序），建议用 std::lexicographical_compare。

---

## 10. 交换与反转（已废弃，建议用标准库）

### `swap`, `swap_ranges`
标准库 std::swap/std::swap_ranges 包装，本身无特别优化，仅用于兼容。

### `reverse`, `reverse_copy`
用 std::reverse/std::reverse_copy，无优化，仅兼容。

---

## 11. 其他工具函数

### `is_aligned(void*, Size)`
判断指针/地址是否对齐。

### `cache_lines(Size bytes)`
给定字节数，计算覆盖多少个 cache line。

### `pages(Size bytes)`
计算覆盖多少个页面。

---

## 12. 内存访问建议与锁定

### 枚举 `MemoryAdvice`
给内核的内存访问模式建议（普通/顺序/随机/将被访问/不再访问/可释放）。

### `memory_advise(void*, Size, MemoryAdvice)`
调用 madvise（或 Windows 的 MEM_RESET），告诉内核内存访问模式优化，有助于性能。

### `memory_lock`, `memory_unlock`
用 mlock/munlock（或 VirtualLock/Unlock）把内存锁入/释放，不让 OS 换出到 swap。

---

## 13. 内存栅栏操作

### `store_fence`
保证所有写入操作在 fence 之前完成并对其他线程可见。

### `load_fence`
保证所有读取操作在 fence 之前全部完成。

### `memory_fence`
全内存栅栏，保证内存操作顺序。

---

## 14. 系统页面大小查询

### `get_page_size()`
运行时获取当前主机页面大小。

### `get_huge_page_size()`
获取大页（Huge page）大小，若不支持返回 0。

---

## 总结与建议

- 本模块极大增强了 std::memory 不具备的“高性能”和“底层友好”特性，特别适合性能敏感、大数据、高并发场景。
- 避免直接用已废弃接口，优先用标准库替代品。
- 面向高阶用途（如内存分布分析、NUMA、大页管理、流式NT拷贝等）值得反复研读源码和文档，结合具体平台特性调优效果最佳。

如需特殊用途（NUMA 亲和性、更复杂的页面管理、内存绑核等），推荐扩展封装此模块。

---



