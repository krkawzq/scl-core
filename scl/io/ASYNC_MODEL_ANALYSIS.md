# 异步流水线模型：IO与解压的重叠优化

## Executive Summary

传统模型假设IO和解压是**串行**的（同步模型），但现代硬件支持**异步流水线**：在读取下一个chunk的同时解压当前chunk。这使得总时间由**瓶颈**决定，而非简单相加。

**核心创新**：
- 引入 $\gamma$ (IO-CPU比率) 描述系统特性
- 动态校准算法：运行时测量实际IO带宽和解压速度
- 环境感知调度器：在NVMe SSD和HDD上自动采用不同策略

---

## 1. 同步 vs 异步模型对比

### 1.1 同步模型（Naive）

**假设**: IO和解压串行执行

```
Timeline:
[----IO----][--Decompress--]
     ↓           ↓
  T_io       T_cpu

Total = T_io + T_cpu
```

**总时间**：

$$
T_{\text{read}}^{\text{sync}} = T_{\text{io}} + T_{\text{cpu}}
$$

### 1.2 异步流水线模型（Optimized）

**假设**: 使用预取或多线程，IO与解压重叠

```
Timeline (Chunk N and N+1):

Chunk N:  [----IO----]
                      [--Decompress--]
Chunk N+1:           [----IO----]
                                 [--Decompress--]

Effective time per chunk = max(T_io, T_cpu)
```

**总时间**（流水线稳定后）：

$$
T_{\text{read}}^{\text{async}} = \max(T_{\text{io}}, T_{\text{cpu}}) + L_{\text{overhead}}
$$

其中 $L_{\text{overhead}}$ 是系统调用的固定延迟（通常 < 1ms）。

### 1.3 关键参数：IO-CPU比率 $\gamma$

定义**瓶颈比率**：

$$
\gamma = \frac{V_{\text{cpu}}}{V_{\text{io}}} = \frac{\text{Decompression Speed}}{\text{IO Bandwidth}}
$$

其中：
- $V_{\text{io}}$: 磁盘IO带宽 (MB/s)
- $V_{\text{cpu}}$: 解压速度 (MB/s)

**系统特性分类**：

| $\gamma$ | 系统类型 | 瓶颈 | 典型硬件 |
|----------|---------|------|----------|
| $\gamma \ll 1$ | CPU Bound | 解压慢 | HPC集群 + 旧CPU |
| $\gamma \approx 1$ | Balanced | 均衡 | 现代台式机 |
| $\gamma \gg 1$ | IO Bound | 磁盘慢 | 笔记本 + HDD |

**实际读取时间**：

$$
T_{\text{read}} = \max\left(\frac{S}{V_{\text{io}}}, \frac{S}{V_{\text{cpu}}}\right) = \frac{S}{V_{\text{io}}} \cdot \max(1, \gamma)
$$

其中 $S$ 是压缩后的chunk大小。

---

## 2. 动态α计算公式

### 2.1 传统静态α的问题

之前我们硬编码：
```cpp
double alpha = 30.0;  // Gzip level 6
```

**问题**：
- ❌ 假设IO和解压串行
- ❌ 忽略硬件差异（NVMe vs HDD）
- ❌ 未考虑数据特性（压缩率）

### 2.2 新公式：基于瓶颈模型

$$
\alpha_{\text{dynamic}} = \frac{T_{\text{read}}}{T_{\text{check}}} = \frac{\max(T_{\text{io}}, T_{\text{cpu}})}{T_{\text{check}}}
$$

**分情况讨论**：

#### Case 1: IO Bound ($\gamma > 1$)

$$
T_{\text{read}} \approx T_{\text{io}} \implies \alpha \approx \frac{T_{\text{io}}}{T_{\text{check}}}
$$

解压很快，瓶颈在磁盘。$\alpha$ 较小，**倾向DirectRead**（因为check也需要IO）。

#### Case 2: CPU Bound ($\gamma < 1$)

$$
T_{\text{read}} \approx T_{\text{cpu}} \implies \alpha \approx \frac{T_{\text{cpu}}}{T_{\text{check}}}
$$

磁盘很快，瓶颈在CPU。$\alpha$ 较大，**极度倾向BoundaryCheck**（省CPU）。

#### Case 3: Cache Hit

如果数据在Page Cache中：

$$
V_{\text{io}}^{\text{cache}} \approx 10 \text{ GB/s} \implies T_{\text{io}} \approx 0
$$

$$
\alpha \approx \frac{T_{\text{cpu}}}{T_{\text{check}}} \approx 1-3 \text{ (very small)}
$$

**结论**: 数据在内存，直接读最快，不要浪费时间check！

---

## 3. 自动校准算法

### 3.1 测量目标

我们需要在运行时测量：
1. $V_{\text{io}}$: IO带宽（读取压缩数据）
2. $V_{\text{cpu}}$: 解压速度
3. $T_{\text{check}}$: 边界检查延迟

### 3.2 采样策略

**挑战**: Page Cache干扰
**解决**: 
- 随机选择文件中部的chunk（避免预读）
- 多次采样取中位数（避免outlier）
- 首次IO强制绕过cache（Linux: O_DIRECT）

**算法流程**：

```
1. 随机选择 N=3 个非空chunk
2. 对每个chunk:
   a) 测量 T_io (仅读取压缩数据，不解压)
   b) 测量 T_cpu (手动解压内存中的数据)
   c) 测量 T_check (读取indices边界)
3. 计算中位数，排除异常值
4. 计算 V_io, V_cpu, γ, α
```

### 3.3 HDF5实现细节

**Phase 1: 测量IO带宽**

使用 `H5Dread_chunk` 读取原始字节流（绕过filter）：

```cpp
// Get compressed chunk size
uint32_t filter_mask = 0;
hsize_t chunk_offset[2] = {chunk_idx * chunk_size, 0};
size_t chunk_bytes = H5Dget_chunk_storage_size(dset_id, chunk_offset);

// Read raw compressed data
std::vector<uint8_t> compressed_buffer(chunk_bytes);
auto start = high_resolution_clock::now();

herr_t err = H5Dread_chunk(
    dset_id,
    H5P_DEFAULT,
    chunk_offset,
    &filter_mask,
    compressed_buffer.data()
);

auto end = high_resolution_clock::now();
double t_io = duration<double>(end - start).count();

V_io = (chunk_bytes / 1024.0 / 1024.0) / t_io;  // MB/s
```

**Phase 2: 测量解压速度**

手动调用HDF5 filter pipeline：

```cpp
// Get filter info
hid_t dcpl = H5Dget_create_plist(dset_id);
size_t cd_nelmts = 0;
unsigned cd_values[8];
unsigned filter_flags;
H5Z_filter_t filter = H5Pget_filter2(dcpl, 0, &filter_flags, 
                                     &cd_nelmts, cd_values, ...);

// Decompress in memory
std::vector<uint8_t> decompressed_buffer(uncompressed_size);
auto start = high_resolution_clock::now();

if (filter == H5Z_FILTER_DEFLATE) {
    // Use zlib directly
    z_stream stream = {};
    stream.next_in = compressed_buffer.data();
    stream.avail_in = chunk_bytes;
    stream.next_out = decompressed_buffer.data();
    stream.avail_out = uncompressed_size;
    
    inflateInit(&stream);
    inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
}

auto end = high_resolution_clock::now();
double t_cpu = duration<double>(end - start).count();

V_cpu = (uncompressed_size / 1024.0 / 1024.0) / t_cpu;  // MB/s
```

**Phase 3: 测量Check延迟**

```cpp
// Open indices dataset
hid_t indices_dset = H5Dopen(group_id, "indices", H5P_DEFAULT);

auto start = high_resolution_clock::now();

// Read first and last element of chunk
Index boundary[2];
hsize_t start_offset = chunk_idx * chunk_size;
H5Sselect_elements(file_space, H5S_SELECT_SET, 2, 
                   {start_offset, start_offset + chunk_size - 1});
H5Dread(indices_dset, H5T_NATIVE_INT64, mem_space, file_space, 
        H5P_DEFAULT, boundary);

auto end = high_resolution_clock::now();
T_check = duration<double>(end - start).count();
```

### 3.4 结果计算

```cpp
struct SystemProfile {
    double V_io;           // IO bandwidth (MB/s)
    double V_cpu;          // Decompression speed (MB/s)
    double T_check;        // Check latency (seconds)
    
    double gamma;          // V_cpu / V_io
    double alpha;          // max(T_io, T_cpu) / T_check
    
    std::string bottleneck;  // "IO", "CPU", or "Balanced"
};

SystemProfile profile;
profile.gamma = profile.V_cpu / profile.V_io;

// Compute effective read time for typical chunk
double S_typical = 10000 * sizeof(float);  // 40KB uncompressed
double S_compressed = S_typical / compression_ratio;  // e.g., 10KB

double T_io = S_compressed / (profile.V_io * 1024 * 1024);
double T_cpu = S_compressed / (profile.V_cpu * 1024 * 1024);
double T_read = std::max(T_io, T_cpu);

profile.alpha = T_read / profile.T_check;

// Classify bottleneck
if (profile.gamma < 0.5) {
    profile.bottleneck = "CPU";  // Decompress is slow
} else if (profile.gamma > 2.0) {
    profile.bottleneck = "IO";   // Disk is slow
} else {
    profile.bottleneck = "Balanced";
}
```

---

## 4. 环境感知示例

### 4.1 场景A：HPC集群

**硬件**:
- 存储: Lustre并行文件系统（$V_{\text{io}} = 2000$ MB/s）
- CPU: 老旧Xeon（$V_{\text{cpu}} = 500$ MB/s，Gzip）
- 压缩: Gzip level 9

**测量结果**:
```
V_io = 2000 MB/s
V_cpu = 500 MB/s
γ = 0.25  (CPU Bound!)

Typical chunk: 10KB compressed
T_io = 10KB / 2000MB/s = 5 μs
T_cpu = 10KB / 500MB/s = 20 μs
T_read = max(5, 20) = 20 μs

T_check = 0.5 ms (IO latency on Lustre)
α = 20 μs / 500 μs = 0.04  ← 很小！
```

**决策**:
- $\ln(\alpha) = \ln(0.04) = -3.22$
- CI必须非常小（< 0.05）才值得check
- **策略**: 几乎总是DirectRead（因为CPU是瓶颈，check的IO开销太大）

### 4.2 场景B：现代笔记本

**硬件**:
- 存储: NVMe SSD（$V_{\text{io}} = 800$ MB/s）
- CPU: i9-13900K（$V_{\text{cpu}} = 3000$ MB/s，Gzip）
- 压缩: Gzip level 6

**测量结果**:
```
V_io = 800 MB/s
V_cpu = 3000 MB/s
γ = 3.75  (IO Bound!)

Typical chunk: 10KB compressed
T_io = 10KB / 800MB/s = 12.5 μs
T_cpu = 10KB / 3000MB/s = 3.3 μs
T_read = max(12.5, 3.3) = 12.5 μs

T_check = 50 μs (Fast NVMe random read)
α = 12.5 μs / 50 μs = 0.25
```

**决策**:
- $\ln(\alpha) = \ln(0.25) = -1.39$
- 非常小的CI（< 0.02）才值得check
- **策略**: 倾向DirectRead（解压几乎无成本）

### 4.3 场景C：机械硬盘

**硬件**:
- 存储: 7200RPM HDD（$V_{\text{io}} = 100$ MB/s）
- CPU: i7（$V_{\text{cpu}} = 1500$ MB/s）
- 压缩: Gzip level 6

**测量结果**:
```
V_io = 100 MB/s
V_cpu = 1500 MB/s
γ = 15  (极度IO Bound!)

Typical chunk: 10KB compressed
T_io = 10KB / 100MB/s = 100 μs
T_cpu = 10KB / 1500MB/s = 6.7 μs
T_read = max(100, 6.7) = 100 μs

T_check = 10 ms (HDD seek time!)
α = 100 μs / 10 ms = 0.01  ← 极小
```

**决策**:
- $\ln(\alpha) = \ln(0.01) = -4.61$
- **策略**: 永远DirectRead（check的seek开销太大）

### 4.4 场景D：Cache Hit

**状态**: 数据在Page Cache中

**测量结果**:
```
V_io = 10000 MB/s  (内存拷贝速度)
V_cpu = 1500 MB/s
γ = 0.15  (CPU Bound，但是假象)

T_io = 10KB / 10000MB/s = 1 μs  (极快)
T_cpu = 10KB / 1500MB/s = 6.7 μs
T_read = max(1, 6.7) = 6.7 μs

T_check = 50 μs (也需要内存访问)
α = 6.7 μs / 50 μs = 0.13
```

**决策**:
- $\ln(\alpha) = \ln(0.13) = -2.04$
- **策略**: DirectRead（数据在内存，读数据和读索引一样快，不如直接读数据）

**关键洞察**: 模型自动适应Cache情况！

---

## 5. 实现建议

### 5.1 Lazy Calibration

```cpp
class AdaptiveScheduler {
    std::optional<SystemProfile> _profile;
    bool _calibration_done = false;
    
    void ensure_calibrated(const h5::Dataset& dataset) {
        if (!_calibration_done) {
            _profile = calibrate_system_async(dataset);
            _calibration_done = true;
            
            // Use measured α instead of hardcoded
            _config.alpha = _profile->alpha;
            _ln_alpha = std::log(_profile->alpha);
        }
    }
};
```

### 5.2 Cache Detection

如果测出 $V_{\text{io}} > 5000$ MB/s，说明命中Cache：

```cpp
if (profile.V_io > 5000) {
    // Cache hit detected
    // Strategy: Always DirectRead (data in memory)
    return make_direct_read_scheduler();
}
```

### 5.3 Adaptive Threshold

根据瓶颈类型调整安全边际：

```cpp
if (profile.bottleneck == "CPU") {
    config.safety_margin = 1.2;  // More aggressive checking
} else if (profile.bottleneck == "IO") {
    config.safety_margin = 0.8;  // More conservative checking
}
```

---

## 6. 理论极限与实际约束

### 6.1 理论最优加速比

在完美流水线下，加速比：

$$
\text{Speedup}_{\text{pipeline}} = \frac{T_{\text{io}} + T_{\text{cpu}}}{\max(T_{\text{io}}, T_{\text{cpu}})} = 1 + \min\left(\frac{1}{\gamma}, \gamma\right)
$$

| $\gamma$ | Speedup |
|----------|---------|
| 0.1 (CPU Bound) | 1.1x |
| 0.5 | 1.5x |
| 1.0 (Balanced) | 2.0x |
| 2.0 | 1.5x |
| 10 (IO Bound) | 1.1x |

**结论**: 平衡系统受益最大（2x），极端不平衡系统收益有限。

### 6.2 实际约束

1. **HDF5线程安全**: 默认non-thread-safe build
   - 解决: 使用mutex保护或编译thread-safe版本

2. **预取复杂度**: 需要维护pending chunk队列
   - 解决: 使用 `std::future` 和 `std::async`

3. **内存压力**: 同时持有多个chunk
   - 解决: 限制pending queue size（如2-4个chunk）

---

## 7. 总结

### 7.1 关键创新

| 传统模型 | 异步模型 |
|---------|---------|
| $\alpha$ 硬编码 | $\alpha$ 动态测量 |
| 假设串行 | 考虑流水线 |
| 硬件无关 | **环境感知** |
| 单一策略 | 自适应策略 |

### 7.2 性能预期

| 场景 | 传统α | 动态α | 策略改进 |
|------|-------|-------|---------|
| HPC (CPU Bound) | 30 | 0.04 | 更激进DirectRead |
| 笔记本 (Balanced) | 30 | 25 | 接近传统 |
| HDD (IO Bound) | 30 | 0.01 | 完全DirectRead |
| Cache Hit | 30 | 0.13 | 自动DirectRead ✅ |

### 7.3 工程价值

**自动化**:
- ✅ 无需用户配置
- ✅ 适应不同硬件
- ✅ 处理Cache情况

**鲁棒性**:
- ✅ 避免worst-case（HDD + BoundaryCheck）
- ✅ 充分利用best-case（Cache + DirectRead）

**可维护性**:
- ✅ 清晰的物理模型
- ✅ 可测量的参数
- ✅ 可验证的预测

---

**文档版本**: 2.0  
**作者**: SCL IO Module Team  
**日期**: 2025-01  
**状态**: Design Complete

