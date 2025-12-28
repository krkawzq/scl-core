# 设计原则

本文档详细介绍了实现 SCL-Core 高性能的优化策略和设计模式。每个模式都经过实战检验并有性能测量支持。

## 1. 零开销抽象

### 原则

抽象必须编译为与手写底层代码无法区分的最优机器码。如果抽象引入任何运行时开销，它就违反了这一原则。

### 实现策略

#### 热路径中无虚函数

虚函数调用引入间接性并阻止内联。使用编译时多态代替。

```cpp
// 避免: 虚函数开销
class Normalizer {
    virtual void normalize(Real* data, size_t n) = 0;
};

class L2Normalizer : public Normalizer {
    void normalize(Real* data, size_t n) override {
        // 实现
    }
};

// 优先: 基于模板的编译时多态
template <NormMode Mode>
SCL_FORCE_INLINE void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L2) {
        // L2 特定实现
        // 编译器生成专门代码
    } else if constexpr (Mode == NormMode::L1) {
        // L1 特定实现
    }
}
```

**为什么有效：**
- 无虚表查找 - 直接函数调用
- 内联机会 - 优化器可以看穿调用
- 死代码消除 - 未使用的分支被移除
- 常量传播 - 模式特定优化

#### 强制内联关键函数

指导编译器内联性能关键函数：

```cpp
#define SCL_FORCE_INLINE [[gnu::always_inline]] inline

SCL_FORCE_INLINE Real dot_product(const Real* a, const Real* b, size_t n) {
    // 热路径函数 - 总是内联
    Real sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

**何时使用：**
- 在紧密循环中调用的函数
- 小函数（< 20 行）
- 具有常量参数以启用优化的函数

#### Constexpr 用于编译时计算

尽可能将计算从运行时移至编译时：

```cpp
constexpr size_t compute_block_size(size_t total, size_t threads) {
    return (total + threads - 1) / threads;
}

// 如果参数是编译时常量，结果也是
constexpr size_t BLOCK_SIZE = compute_block_size(1000000, 8);

// 可用作数组大小
Real workspace[BLOCK_SIZE];
```

**优势：**
- 零运行时成本
- 可用于模板参数和数组大小
- 启用进一步的编译时优化

#### 仅在边界处类型擦除

保持模板内部，仅在 API 边界擦除类型：

```cpp
// 内部: 模板实现零开销
template <typename T, bool IsCSR>
void normalize_impl(Sparse<T, IsCSR>& matrix, NormMode mode);

// API 边界: 为 Python 类型擦除的 C 接口
extern "C" {
    void scl_normalize_f32_csr(SparseMatrixHandle handle, int mode);
    void scl_normalize_f64_csr(SparseMatrixHandle handle, int mode);
}

void scl_normalize_f32_csr(SparseMatrixHandle handle, int mode) {
    auto* matrix = reinterpret_cast<Sparse<float, true>*>(handle);
    normalize_impl(*matrix, static_cast<NormMode>(mode));
}
```

## 2. SIMD 优化

现代 CPU 可以使用 SIMD 指令每周期处理 4-16 个元素。正确向量化的代码比标量代码快 5-10 倍。

### 多累加器模式

最重要的 SIMD 优化。现代 FMA 单元有 4-5 个周期的延迟，但每周期可以接受新操作。使用多个独立累加器隐藏此延迟。

```cpp
// 差: 单累加器创建依赖链
// 每个 Add 必须等待前一个 Add 完成
auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);  // 依赖于前一次迭代!
}

// 最优: 4 路累加器隐藏延迟
// 多个 FMA 单元可以并行执行
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

size_t i = 0;
// 主循环: 4 路展开
for (; i + 4*lanes <= n; i += 4*lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0*lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1*lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2*lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3*lanes));
}

// 清理循环: 处理剩余元素
for (; i + lanes <= n; i += lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i));
}

// 合并累加器
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

**性能影响：**
- 单累加器: 每 4-5 个周期约 1 次操作（受延迟限制）
- 4 路累加器: 每周期约 4 次操作（受吞吐量限制）
- **加速: 4-5 倍** 对于内存受限操作

**为什么是 4 个累加器？**
- 大多数 CPU 有 2 个 FMA 单元
- FMA 延迟是 4-5 个周期
- 4 个累加器充分利用两个单元

### 融合操作

将多个操作组合成单次遍历以减少内存流量：

```cpp
// 低效: 多次遍历数据
Real sum = 0, sumsq = 0;
for (size_t i = 0; i < n; ++i) {
    sum += data[i];
}
for (size_t i = 0; i < n; ++i) {
    sumsq += data[i] * data[i];
}

// 高效: SIMD 融合的单次遍历
auto v_sum = s::Zero(d), v_sumsq = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
    v_sumsq = s::MulAdd(v, v, v_sumsq);  // FMA: v*v + v_sumsq
}
```

**优势：**
- 内存带宽减少 2 倍
- 更好的缓存利用（数据加载一次）
- 单循环开销
- 为两个操作启用多累加器模式

### 水平规约

高效地将 SIMD 向量规约为标量值：

```cpp
// 对向量中的所有通道求和
Real horizontal_sum(s::Tag d, s::Vec v) {
    // Highway 提供高效实现
    return s::GetLane(s::SumOfLanes(d, v));
}

// 与多累加器结合
auto v_sum0 = /* ... */;
auto v_sum1 = /* ... */;
auto v_sum2 = /* ... */;
auto v_sum3 = /* ... */;

// 在规约前合并向量（更少的规约）
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
Real result = horizontal_sum(d, v_sum);
```

### 预取

对于可预测的访问模式，在处理当前数据时预取未来数据：

```cpp
constexpr size_t PREFETCH_DISTANCE = 16;  // 实验调整

for (size_t i = 0; i < n; i += lanes) {
    // 预取将在未来迭代中使用的数据
    if (i + PREFETCH_DISTANCE * lanes < n) {
        SCL_PREFETCH_READ(data + i + PREFETCH_DISTANCE * lanes, 0);
    }
    
    // 处理当前数据
    auto v = s::Load(d, data + i);
    // ... 计算 ...
}
```

**何时预取：**
- 顺序访问模式
- 内存受限操作
- 访问延迟 > 计算时间

**何时不预取：**
- 随机访问模式（浪费缓存行）
- 计算受限操作
- 适合缓存的小数据集

### 掩码操作

在数组边界处理部分向量而不分支：

```cpp
// 处理完整向量
size_t i = 0;
for (; i + lanes <= n; i += lanes) {
    auto v = s::Load(d, data + i);
    process(v);
}

// 使用掩码处理剩余元素
if (i < n) {
    auto mask = s::FirstN(d, n - i);  // 剩余元素的掩码
    auto v = s::MaskedLoad(mask, d, data + i);
    process(v);
    s::MaskedStore(mask, result, d, output + i);
}
```

## 3. 并行处理

有效利用多核 CPU 同时最小化同步开销。

### 自动并行化

让库决定何时并行化有益：

```cpp
// 如果 n 足够大则自动并行化
// 对于小 n 使用串行执行以避免开销
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});

// 阈值基于以下动态计算:
// - 每个元素的操作成本
// - 可用线程数
// - 线程池开销
```

**典型阈值：**
- 轻量级操作: 10,000+ 元素
- 中等操作: 1,000+ 元素
- 重操作: 100+ 元素

### 批处理

对于轻量级操作，批处理元素以减少每任务开销：

```cpp
constexpr Size BATCH_SIZE = 64;  // 调整以平衡开销和并行性
const Size num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

parallel_for(Size(0), num_batches, [&](size_t batch_idx) {
    const Size start = batch_idx * BATCH_SIZE;
    const Size end = std::min(start + BATCH_SIZE, n);
    
    // 在线程内顺序处理批次
    for (Size i = start; i < end; ++i) {
        process(data[i]);  // 轻量级操作
    }
});
```

**权衡：**
- 更大的批次: 更低的开销，更差的负载平衡
- 更小的批次: 更高的开销，更好的负载平衡
- 最优大小取决于操作成本和数组大小

### 每线程工作空间

通过为每个线程提供自己的工作空间避免同步：

```cpp
// 低效: 共享工作空间需要锁
std::vector<Real> workspace;
std::mutex mtx;

parallel_for(Size(0), n, [&](size_t i) {
    std::lock_guard lock(mtx);  // 争用瓶颈!
    workspace.resize(needed_size);
    use_workspace(workspace);
});

// 高效: 每线程工作空间，无锁
WorkspacePool<Real> pool(num_threads, workspace_size);

parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);  // 无锁访问
    use_workspace(workspace, workspace_size);
});
```

**优势：**
- 零同步开销
- 更好的缓存局部性（线程本地数据）
- 可预测的性能（无锁争用）

### 防止伪共享

确保线程本地数据是缓存行对齐的：

```cpp
// 有问题: 相邻计数器共享缓存行
struct Counters {
    size_t thread_counts[8];  // 8 个线程，每个 8 字节 = 64 字节
};
// 所有线程写入同一缓存行 - 伪共享!

// 正确: 缓存行对齐的计数器
struct alignas(64) AlignedCounter {
    size_t count;
    char padding[56];  // 填充到 64 字节
};

struct Counters {
    AlignedCounter thread_counts[8];
};
// 每个线程写入单独的缓存行
```

## 4. 内存管理

高效的内存管理对性能和正确性都至关重要。

### 对齐分配

SIMD 操作需要正确对齐的内存：

```cpp
// 为 SIMD 分配对齐内存
constexpr size_t ALIGNMENT = 64;  // 缓存行大小
Real* data = scl::memory::aligned_alloc<Real>(count, ALIGNMENT);

// 对齐数据的 SIMD 加载更高效
auto v = s::LoadU(d, data);      // 未对齐加载: 较慢
auto v = s::Load(d, data);       // 对齐加载: 较快
```

**对齐要求：**
- AVX2: 建议 32 字节对齐
- AVX-512: 建议 64 字节对齐
- 尽可能对齐到缓存行（64 字节）

### 块分配策略

对于稀疏矩阵，按块分配行/列：

```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;      // 对于 float32 为 16KB
    Index max_block_elements = 262144;    // 对于 float32 为 1MB
    
    Index compute_block_size(Index total_nnz, Index primary_dim) const {
        // 目标: 每块 4-8 行以获得良好并行性
        Index avg_nnz = total_nnz / primary_dim;
        Index target_rows = 8;
        Index block_size = avg_nnz * target_rows;
        
        // 限制在 min/max 范围内
        block_size = std::max(block_size, min_block_elements);
        block_size = std::min(block_size, max_block_elements);
        
        return block_size;
    }
};
```

**权衡：**
- 更大的块: 更好的内存重用，更难部分释放
- 更小的块: 更容易释放，更多分配开销
- 最优大小: 每块 256KB - 1MB

### 注册表模式

跟踪所有分配以实现 Python 集成：

```cpp
auto& reg = get_registry();

// 使用元数据注册分配
Real* data = reg.new_array<Real>(count);

// 共享缓冲区的引用计数
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, alias_ptrs, AllocType::ArrayNew);

// 使用适当的删除器自动清理
reg.unregister_ptr(data);
```

### 内存池

重用分配以避免重复的 malloc/free：

```cpp
template <typename T>
class MemoryPool {
    std::vector<std::unique_ptr<T[]>> available_;
    std::vector<std::unique_ptr<T[]>> in_use_;
    size_t buffer_size_;
    
public:
    T* acquire() {
        if (available_.empty()) {
            return new T[buffer_size_];
        }
        auto buffer = std::move(available_.back());
        available_.pop_back();
        T* ptr = buffer.get();
        in_use_.push_back(std::move(buffer));
        return ptr;
    }
    
    void release(T* ptr) {
        // 从 in_use 移至 available
    }
};
```

## 5. 循环优化

循环结构中的小细节可能对性能产生很大影响。

### 手动展开

展开循环以暴露指令级并行性：

```cpp
// 4 路展开以获得更好的 ILP
size_t i = 0;
for (; i + 4 <= n; i += 4) {
    result[i+0] = process(data[i+0]);
    result[i+1] = process(data[i+1]);
    result[i+2] = process(data[i+2]);
    result[i+3] = process(data[i+3]);
}

// 剩余部分的清理循环
for (; i < n; ++i) {
    result[i] = process(data[i]);
}
```

**何时展开：**
- 简单的循环体（少量操作）
- 独立的迭代
- 展开因子 4-8 效果最佳

### 循环融合

对相同数据合并多个循环：

```cpp
// 低效: 多次遍历
for (size_t i = 0; i < n; ++i) compute_A(i);
for (size_t i = 0; i < n; ++i) compute_B(i);
for (size_t i = 0; i < n; ++i) compute_C(i);

// 高效: 单次遍历，更好的缓存利用
for (size_t i = 0; i < n; ++i) {
    compute_A(i);
    compute_B(i);
    compute_C(i);
}
```

### 强度削减

用更便宜的等效操作替换昂贵的操作：

```cpp
// 昂贵: 循环中的除法
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] / divisor;
}

// 便宜: 乘以倒数
const Real inv_divisor = Real(1) / divisor;
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * inv_divisor;
}
```

**常见替换：**
- 除法 → 乘以倒数
- 2 的幂次模运算 → 按位 AND
- 常量乘法 → 移位 + 加法

### 循环不变代码外提

将常量计算移出循环：

```cpp
// 低效: 每次迭代重新计算常量
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * compute_scale_factor(params);
}

// 高效: 循环前计算一次
const Real scale = compute_scale_factor(params);
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * scale;
}
```

## 6. 算法优化

选择匹配硬件特性的算法。

### 提前退出

在昂贵操作前检查终止条件：

```cpp
void normalize_rows(Sparse<Real, true>& matrix, Real eps) {
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        const Index len = matrix.primary_length(i);
        
        // 空行的提前退出
        if (len == 0) return;
        
        auto* vals = matrix.primary_values(i).ptr;
        Real norm = compute_norm(vals, len);
        
        // 零范数的提前退出
        if (norm <= eps) return;
        
        // 现在做昂贵的归一化
        const Real inv_norm = Real(1) / norm;
        for (Index j = 0; j < len; ++j) {
            vals[j] *= inv_norm;
        }
    });
}
```

### 数值稳定性

使用补偿求和以获得更好的精度：

```cpp
// Kahan 求和: 减少浮点误差累积
Real kahan_sum(const Real* data, size_t n) {
    Real sum = 0, compensation = 0;
    
    for (size_t i = 0; i < n; ++i) {
        Real y = data[i] - compensation;
        Real t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}
```

**何时使用：**
- 大数组（n > 10,000）
- 高精度要求
- 混合量级值的求和

### 缓存友好访问

按顺序访问内存以最大化缓存命中：

```cpp
// 缓存不友好: 行主数据的列主访问
for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
        process(data[i * cols + j]);  // 跨步访问 - 缓存未命中!
    }
}

// 缓存友好: 行主数据的行主访问
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        process(data[i * cols + j]);  // 顺序访问 - 缓存命中!
    }
}
```

**经验法则：**
- 按存储顺序访问多维数组
- 优先选择单位步长访问模式
- 按缓存行大小的块处理数据（64 字节）

## 7. 编译时优化

利用编译器生成最优代码。

### 模板特化

为常见情况优化：

```cpp
// 通用模板
template <typename T>
void process(T* data, size_t n) {
    // 通用实现
}

// 为 float32 特化 - 使用 SIMD
template <>
void process<float>(float* data, size_t n) {
    using Tag = hn::ScalableTag<float>;
    const Tag d;
    // SIMD 优化实现
}
```

### Constexpr If

在编译时分支以消除死代码：

```cpp
template <NormMode Mode>
void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L1) {
        // L1 特定代码 - 其他分支被移除
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::abs(data[i]);
        }
    } else if constexpr (Mode == NormMode::L2) {
        // L2 特定代码
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] * data[i];
        }
    }
    // 无运行时分支!
}
```

### 属性提示

使用属性指导编译器：

```cpp
// 函数是纯的 - 无副作用，结果仅依赖于参数
[[gnu::pure]] Real compute_norm(const Real* data, size_t n);

// 结果必须使用 - 如果丢弃则警告
[[nodiscard]] Real* allocate_buffer(size_t n);

// 用于分支预测的可能/不可能
if (SCL_LIKELY(n > threshold)) {
    // 常见路径
} else {
    // 罕见路径
}
```

## 性能检查清单

在声明函数"已优化"之前，验证：

- [ ] SIMD: 多累加器模式（4 路）
- [ ] SIMD: 融合操作（单次遍历）
- [ ] 内存: 对齐分配（64 字节）
- [ ] 内存: 顺序访问（缓存友好）
- [ ] 并行: 自动并行化
- [ ] 并行: 每线程工作空间（无锁）
- [ ] 循环: 手动展开（4 路）
- [ ] 循环: 强度削减（除法 → 乘法）
- [ ] 算法: 提前退出检查
- [ ] 算法: 数值稳定性（Kahan 求和）
- [ ] 编译: 常量的 constexpr
- [ ] 编译: 模板特化
- [ ] 编译: 热路径的内联提示

## 性能分析和基准测试

始终测量优化前后：

```cpp
#include <chrono>

template <typename Func>
double benchmark(Func&& func, size_t iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / static_cast<double>(iterations);
}

// 使用
double avg_time = benchmark([&]() {
    normalize_rows(matrix, NormMode::L2);
});

std::cout << "平均时间: " << avg_time << " μs\n";
```

**要跟踪的指标：**
- **吞吐量**: 每秒处理的元素
- **带宽**: 内存带宽利用率（GB/s）
- **效率**: 实际带宽 / 理论峰值
- **可扩展性**: 加速与线程数

---

::: tip 优化哲学
优化不是关于聪明的技巧 - 而是关于理解硬件并编写与 CPU 实际工作方式匹配的代码。本文档中的每个模式都存在，因为现代硬件以可测量的性能提升奖励它。
:::

