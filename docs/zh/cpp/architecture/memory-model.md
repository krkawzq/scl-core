# 内存模型

SCL-Core 的内存管理系统专为三个目标设计：确定性清理、零拷贝 Python 集成和高性能并发访问。本文档解释注册表系统和内存所有权模式。

## 设计原则

### 1. 显式优于隐式

每个内存分配和释放都是显式和可跟踪的。构造函数中没有隐藏分配，没有隐式复制，没有垃圾收集暂停。

### 2. 确定性清理

当所有权被释放时立即释放内存，而不是在某个未来的垃圾收集周期。这种可预测的行为对大型数据集工作负载至关重要。

### 3. 零拷贝集成

内存可以跨语言边界（C++ ↔ Python）传输而无需复制数据。Python 通过胶囊协议直接获取 C++ 分配缓冲区的所有权。

### 4. 线程安全

多个线程可以并发分配和释放内存而不发生数据竞争。注册表使用分片来最小化锁争用。

## 内存所有权模型

### 简单所有权

单一所有者，单一指针，确定性生命周期：

```cpp
auto& reg = scl::get_registry();

// 分配并注册
Real* data = reg.new_array<Real>(1000);

// 使用数据...
process(data, 1000);

// 清理 - 内存立即释放
reg.unregister_ptr(data);
```

**特征：**
- 注册一个指针
- 取消注册立即释放内存
- 无引用计数开销
- 适合临时缓冲区

**何时使用：**
- 短期分配
- 单线程所有权
- 简单数据结构

### 引用计数缓冲区

多个别名，共享所有权，最后一个引用被释放时自动清理：

```cpp
auto& reg = scl::get_registry();

// 分配主缓冲区
Real* main_ptr = new Real[1000];

// 创建别名（例如，矩阵列）
std::vector<void*> aliases;
for (size_t i = 0; i < 10; ++i) {
    aliases.push_back(main_ptr + i * 100);  // 列 i
}

// 使用引用计数注册
BufferID id = reg.register_buffer_with_aliases(
    main_ptr,                    // 要释放的真实指针
    1000 * sizeof(Real),         // 字节大小
    aliases,                     // 别名指针
    AllocType::ArrayNew          // 如何释放: delete[]
);

// 初始引用计数 = 11 (主 + 10 个别名)

// 别名超出作用域时取消注册
for (auto* alias : aliases) {
    reg.unregister_ptr(alias);  // 递减引用计数
}
// 引用计数 = 1 (仅主)

// 取消注册主指针
reg.unregister_ptr(main_ptr);  // 引用计数 = 0，释放内存
```

**特征：**
- 多个指针引用同一底层缓冲区
- 仅在最后一个引用被释放时释放内存
- 线程安全的引用计数（原子操作）
- 正确处理别名

**何时使用：**
- 稀疏矩阵块（多个行指针到同一块）
- 多个视图之间的共享缓冲区
- Python 集成（多个 NumPy 数组查看同一内存）
- 复杂的所有权图

### 非拥有视图

指向外部管理内存的指针（未在注册表中注册）：

```cpp
// 外部缓冲区 - 例如，来自 Python NumPy 数组
Real* external_data = get_numpy_buffer();

// 创建非拥有视图而不注册
scl::Sparse<Real, true> matrix = 
    scl::Sparse<Real, true>::wrap_traditional(
        external_data,
        indices,
        indptr,
        rows, cols, nnz
    );

// 使用矩阵...
process(matrix);

// 矩阵析构不释放 external_data
// 调用者（Python）保留生命周期责任
```

**特征：**
- 不向注册表注册
- 无自动清理
- 调用者保留所有权
- 零开销包装器

**何时使用：**
- Python 拥有的缓冲区
- 内存映射文件
- 栈分配的缓冲区
- 任何外部管理的内存

## 注册表架构

### 分片设计

注册表使用多个独立的分片以减少锁争用：

```
Registry（全局单例）
├── 分片 0 (hash(ptr) % num_shards == 0)
│   ├── 互斥锁
│   ├── PtrMap: unordered_map<void*, PtrRecord>
│   └── BufferMap: unordered_map<void*, RefCountedBuffer>
│
├── 分片 1 (hash(ptr) % num_shards == 1)
│   ├── 互斥锁
│   ├── PtrMap
│   └── BufferMap
│
├── ...
│
└── 分片 N-1
```

**优势：**
- 并行分配到不同分片 - 无争用
- 通过不同分片上的 const 操作实现无锁读取
- 与线程数线性扩展（对于独立分配）
- 典型配置：16 个分片

**分片选择：**

```cpp
size_t shard_index = hash_ptr(ptr) % num_shards;
```

简单的模运算哈希在分片间均匀分布指针。

### 数据结构

#### PtrRecord

简单的所有权跟踪：

```cpp
struct PtrRecord {
    void* ptr;         // 注册的指针
    size_t bytes;      // 分配大小
    AllocType type;    // 如何释放（delete[]、free 等）
};
```

#### RefCountedBuffer

带别名的共享所有权：

```cpp
struct RefCountedBuffer {
    void* real_ptr;                     // 要释放的实际分配
    size_t byte_size;                   // 字节大小
    AllocType type;                     // 释放方法
    std::atomic<size_t> refcount;       // 线程安全的引用计数
    std::unordered_set<void*> aliases;  // 此缓冲区的所有别名
};
```

### 分配类型

```cpp
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → free
    Custom         // 自定义删除器函数
};
```

当引用计数达到零时，注册表根据 AllocType 调用适当的删除器。

### 核心 API

```cpp
class Registry {
public:
    // 简单分配
    template <typename T>
    T* new_array(size_t count);
    
    // 注册
    void register_ptr(void* ptr, size_t bytes, AllocType type);
    void unregister_ptr(void* ptr);
    
    // 引用计数缓冲区
    BufferID register_buffer_with_aliases(
        void* real_ptr,
        size_t byte_size,
        std::span<void*> alias_ptrs,
        AllocType type
    );
    
    // 查询
    bool is_registered(void* ptr) const;
    size_t get_total_bytes() const;
    size_t get_num_pointers() const;
    size_t get_num_buffers() const;
    
    // 调试/分析
    void print_statistics(std::ostream& os) const;
};
```

### 全局访问

```cpp
Registry& get_registry();  // 线程安全单例
```

全局注册表在首次使用时初始化（Meyers 单例）并在程序持续时间内存活。

## 稀疏矩阵内存

### 不连续存储

SCL-Core 的 `Sparse<T, IsCSR>` 使用指针数组而不是传统的连续 CSR：

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    using Pointer = T*;
    
    Pointer* data_ptrs_;      // 数据指针数组 [primary_dim]
    Pointer* indices_ptrs_;   // 索引指针数组 [primary_dim]
    Index* lengths_;          // 长度数组 [primary_dim]
    
    Index rows_, cols_, nnz_;
};
```

**内存布局：**

```
传统 CSR（连续）：
    data:    [v0 v1 v2 | v3 v4 | v5 v6 v7 v8]  （单次分配）
    indices: [c0 c1 c2 | c3 c4 | c5 c6 c7 c8]  （单次分配）
    indptr:  [0, 3, 5, 9]

SCL-Core 不连续：
    行 0: data_ptrs_[0] → [v0 v1 v2]      indices_ptrs_[0] → [c0 c1 c2]
    行 1: data_ptrs_[1] → [v3 v4]         indices_ptrs_[1] → [c3 c4]
    行 2: data_ptrs_[2] → [v5 v6 v7 v8]   indices_ptrs_[2] → [c5 c6 c7 c8]
```

**优势：**
- 块分配：每块多行
- 灵活的所有权：每个块可以独立管理
- 易于切片：行子集只是指针数组的子集
- 引用计数：多个视图共享同一块
- Python 集成：每个块可以是单独的 NumPy 数组

**权衡：**
- 稍微复杂的索引（一个间接引用）
- 指针数组开销（每行 8 字节）
- 与期望连续 CSR 的库不兼容

### 块分配策略

行/列按块分配以提高效率：

```cpp
struct BlockStrategy {
    // 配置
    Index min_block_elements = 4096;      // 最小: 对于 float32 为 16KB
    Index max_block_elements = 262144;    // 最大: 对于 float32 为 1MB
    
    Index compute_block_size(Index total_nnz, Index primary_dim) const {
        // 目标: 每块 4-8 行以获得良好并行性
        Index avg_nnz_per_row = total_nnz / primary_dim;
        Index target_rows_per_block = 8;
        
        // 块大小 = avg_nnz_per_row * target_rows_per_block
        Index block_size = avg_nnz_per_row * target_rows_per_block;
        
        // 限制在 [min, max] 范围内
        block_size = std::max(block_size, min_block_elements);
        block_size = std::min(block_size, max_block_elements);
        
        return block_size;
    }
};
```

**分配过程：**

```cpp
// 为稀疏矩阵分配块
auto& reg = get_registry();
BlockStrategy strategy;
Index block_size = strategy.compute_block_size(nnz, rows);

for (Index start = 0; start < rows; start += rows_per_block) {
    Index end = std::min(start + rows_per_block, rows);
    Index block_nnz = /* 计算 [start, end) 中的非零数 */;
    
    // 分配块
    Real* data_block = reg.new_array<Real>(block_nnz);
    Index* idx_block = reg.new_array<Index>(block_nnz);
    
    // 分配到行
    Index offset = 0;
    for (Index i = start; i < end; ++i) {
        data_ptrs_[i] = data_block + offset;
        indices_ptrs_[i] = idx_block + offset;
        lengths_[i] = row_nnz[i];
        offset += row_nnz[i];
    }
}
```

### 连续转换

为了与期望传统 CSR/CSC 的库互操作：

```cpp
template <typename T, bool IsCSR>
ContiguousArraysT<T> to_contiguous_arrays(const Sparse<T, IsCSR>& matrix) {
    auto& reg = get_registry();
    
    // 分配连续数组
    T* data = reg.new_array<T>(matrix.nnz());
    Index* indices = reg.new_array<Index>(matrix.nnz());
    Index* indptr = reg.new_array<Index>(matrix.primary_dim() + 1);
    
    // 从不连续复制数据到连续
    indptr[0] = 0;
    Index offset = 0;
    for (Index i = 0; i < matrix.primary_dim(); ++i) {
        Index len = matrix.primary_length(i);
        std::copy_n(matrix.primary_values(i).ptr, len, data + offset);
        std::copy_n(matrix.primary_indices(i).ptr, len, indices + offset);
        offset += len;
        indptr[i + 1] = offset;
    }
    
    return {data, indices, indptr, matrix.nnz(), matrix.primary_dim()};
}
```

`ContiguousArraysT` 中的所有指针都已向注册表注册。Python 可以在不复制的情况下获取所有权。

## Python 集成

### 零拷贝传输

将所有权从 C++ 传输到 Python：

```cpp
// C++ 端: 分配并填充
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(1000);
fill_data(data, 1000);

// Python 绑定: 传输所有权
py::capsule deleter(data, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});

return py::array_t<Real>(
    {1000},           // 形状
    {sizeof(Real)},   // 步幅
    data,             // 数据指针
    deleter           // 清理回调
);
```

**所有权流程：**
1. C++ 分配内存并向注册表注册
2. Python 数组通过 PyCapsule 获取所有权
3. Python 引用计数跟踪生命周期
4. 当 Python 数组被删除时，调用胶囊删除器
5. 删除器从注册表取消注册
6. 注册表释放内存

**无拷贝：** 数据指针直接传输，无 memcpy。

### 共享缓冲区的引用计数

将同一缓冲区的多个视图传输到 Python：

```cpp
// C++ 端: 使用别名注册
auto& reg = scl::get_registry();
Real* main_ptr = new Real[10000];

std::vector<void*> aliases;
for (size_t i = 0; i < 100; ++i) {
    aliases.push_back(main_ptr + i * 100);  // 列 i
}

BufferID id = reg.register_buffer_with_aliases(
    main_ptr, 10000 * sizeof(Real), aliases, AllocType::ArrayNew);

// Python 端: 为每个别名创建数组
py::list result;
for (size_t i = 0; i < 100; ++i) {
    Real* col_ptr = static_cast<Real*>(aliases[i]);
    
    py::capsule deleter(col_ptr, [](void* ptr) {
        scl::get_registry().unregister_ptr(ptr);
    });
    
    py::array_t<Real> col(
        {100},            // 形状
        {sizeof(Real)},   // 步幅
        col_ptr,          // 数据指针
        deleter           // 清理回调
    );
    
    result.append(col);
}

return result;  // NumPy 数组列表，所有视图都是同一缓冲区
```

**生命周期管理：**
- 初始引用计数 = 101（主 + 100 个别名）
- 每个 Python 数组持有一个引用
- 当任何数组被删除时，引用计数递减
- 当最后一个数组被删除时，引用计数 → 0，释放内存
- 线程安全：Python 可以从任何线程删除数组

## 对齐分配

SIMD 操作需要正确对齐的内存：

```cpp
namespace scl::memory {
    template <typename T>
    T* aligned_alloc(size_t count, size_t alignment = 64) {
        // alignment 必须是 2 的幂
        assert((alignment & (alignment - 1)) == 0);
        
        void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        
        return static_cast<T*>(ptr);
    }
    
    void aligned_free(void* ptr) {
        std::free(ptr);  // C++17: free 适用于 aligned_alloc
    }
}
```

**与注册表一起使用：**

```cpp
auto& reg = scl::get_registry();

// 分配对齐内存
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// 使用 aligned_alloc 类型注册
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::AlignedAlloc);

// 用于 SIMD 操作
process_simd(data, 1000);

// 清理 - 注册表调用 free()
reg.unregister_ptr(data);
```

**对齐要求：**
- AVX2: 32 字节对齐（8 个浮点数）
- AVX-512: 64 字节对齐（16 个浮点数）
- 缓存行: 64 字节对齐
- 建议: 始终使用 64 字节对齐

## 工作空间池

预分配的线程本地工作空间避免热循环中的分配：

```cpp
template <typename T>
class WorkspacePool {
    std::vector<std::unique_ptr<T[]>> workspaces_;
    size_t workspace_size_;
    
public:
    WorkspacePool(size_t num_threads, size_t workspace_size)
        : workspace_size_(workspace_size) {
        workspaces_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workspaces_.push_back(std::make_unique<T[]>(workspace_size));
        }
    }
    
    T* get(size_t thread_rank) {
        return workspaces_[thread_rank].get();
    }
    
    size_t size() const {
        return workspace_size_;
    }
};
```

**用法：**

```cpp
// 在并行循环前创建池
const size_t num_threads = get_num_threads();
const size_t workspace_size = compute_required_size();
WorkspacePool<Real> pool(num_threads, workspace_size);

// 在并行循环中使用
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);  // 无锁访问
    
    // 使用工作空间进行临时存储
    compute_with_workspace(workspace, workspace_size);
});

// 池超出作用域时自动清理
```

**优势：**
- 热循环中零分配
- 无同步（线程本地）
- 缓存友好（重用的内存）
- 可预测的性能

## 内存调试

### 统计信息

查询当前内存使用：

```cpp
auto& reg = scl::get_registry();

size_t num_ptrs = reg.get_num_pointers();
size_t num_buffers = reg.get_num_buffers();
size_t total_bytes = reg.get_total_bytes();

std::cout << "注册表统计:\n";
std::cout << "  指针: " << num_ptrs << "\n";
std::cout << "  缓冲区: " << num_buffers << "\n";
std::cout << "  内存: " << (total_bytes / 1024 / 1024) << " MB\n";
```

### 泄漏检测

在调试构建中，注册表警告内存泄漏：

```cpp
// 程序退出时
Registry::~Registry() {
    #ifdef SCL_DEBUG
    if (get_num_pointers() > 0 || get_num_buffers() > 0) {
        std::cerr << "警告: 检测到内存泄漏!\n";
        std::cerr << "  指针: " << get_num_pointers() << "\n";
        std::cerr << "  缓冲区: " << get_num_buffers() << "\n";
        std::cerr << "  总计: " << get_total_bytes() << " 字节\n";
        
        // 打印泄漏分配的详细信息
        print_statistics(std::cerr);
    }
    #endif
    
    // 强制清理所有剩余分配
    cleanup_all();
}
```

### 分配跟踪

启用详细跟踪：

```cpp
#define SCL_TRACK_ALLOCATIONS

// 现在每次分配记录:
// - 分配点（文件:行）
// - 堆栈跟踪
// - 时间戳
// - 线程 ID

// 查询分配详细信息
auto info = reg.get_allocation_info(ptr);
std::cout << "分配于 " << info.file << ":" << info.line << "\n";
std::cout << "线程: " << info.thread_id << "\n";
std::cout << "时间: " << info.timestamp << "\n";
```

## 最佳实践

### 1. RAII 包装器

将注册表操作包装在 RAII 守卫中：

```cpp
template <typename T>
class RegistryGuard {
    T* ptr_;
    
public:
    explicit RegistryGuard(size_t count) {
        ptr_ = scl::get_registry().new_array<T>(count);
    }
    
    ~RegistryGuard() {
        if (ptr_) {
            scl::get_registry().unregister_ptr(ptr_);
        }
    }
    
    // 仅移动
    RegistryGuard(RegistryGuard&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    RegistryGuard& operator=(RegistryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) scl::get_registry().unregister_ptr(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    // 不可复制
    RegistryGuard(const RegistryGuard&) = delete;
    RegistryGuard& operator=(const RegistryGuard&) = delete;
    
    T* get() const { return ptr_; }
    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
};

// 用法
void process_data() {
    RegistryGuard<Real> buffer(1000);
    
    // 使用 buffer.get()
    compute(buffer.get(), 1000);
    
    // 即使抛出异常也自动清理
}
```

### 2. 优先栈分配

对于小型临时缓冲区：

```cpp
// 避免: 小缓冲区的堆分配
Real* temp = new Real[100];
process(temp, 100);
delete[] temp;

// 优先: 栈分配
Real temp[100];
process(temp, 100);
// 自动清理
```

栈分配更快更安全（无手动内存管理）。

### 3. 预分配工作空间

避免循环中的重复分配：

```cpp
// 低效: 每次迭代分配
for (size_t i = 0; i < n; ++i) {
    std::vector<Real> temp(1000);  // 每次迭代分配 + 释放!
    compute_with_temp(temp);
}

// 高效: 预分配工作空间
std::vector<Real> workspace(1000);
for (size_t i = 0; i < n; ++i) {
    compute_with_temp(workspace);  // 重用分配
}
```

### 4. 文档化所有权

清楚地记录谁拥有内存：

```cpp
// 返回拥有指针 - 调用者必须通过注册表释放
Real* allocate_buffer(size_t n);

// 返回非拥有视图 - 不要释放
const Real* get_data_view() const;

// 获取 ptr 的所有权 - 将通过注册表释放
void consume_buffer(Real* ptr);

// 借用 ptr - 不获取所有权
void process_buffer(const Real* ptr, size_t n);
```

## 性能考虑

### 注册表开销

**每指针开销：**
- 哈希表槽: ~24 字节
- PtrRecord: 24 字节
- 总计: 每个注册指针约 48 字节

**每缓冲区开销：**
- 哈希表槽: ~24 字节
- RefCountedBuffer: 56 字节 + 每个别名 8 字节
- 原子引用计数: 分片后争用最小
- 总计: ~80 字节 + 别名

**查找性能：**
- 平均: O(1)，常数因子低
- 最坏情况: 分片的 O(n)（使用良好哈希函数罕见）
- 典型: 注册/取消注册 < 50ns
- 争用: 由于分片最小

### 何时使用注册表

**使用注册表用于：**
- 传输到 Python 的内存（必需）
- 带别名的共享缓冲区（引用计数）
- 长期分配（生命周期管理）
- 大型分配（> 1MB）值得跟踪

**不要使用注册表用于：**
- 栈分配的缓冲区（无需）
- 热循环中的短期临时变量（开销主导）
- 小型分配（< 1KB）（开销不值得）
- 外部库管理的内存（不是我们的责任）

**经验法则：** 如果生命周期明显且本地，跳过注册表。如果生命周期跨越函数/模块边界或涉及 Python，使用注册表。

---

::: tip 通过设计实现内存安全
SCL-Core 的内存模型通过设计防止整类 bug：使用 RAII 包装器时释放后使用不可能，在调试构建中自动检测内存泄漏，零拷贝 Python 集成消除缓冲区复制 bug。
:::

