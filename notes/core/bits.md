# scl/core/bits.hpp 笔记

`scl/core/bits.hpp` 是 SCL 核心库中提供位操作和对齐相关工具的头文件。该文件包含以下主要内容：

## 1. 常量定义

- `kBitsPerU32`：32位整型的比特数（32）。
- `kBitsPerU64`：64位整型的比特数（64）。
- 还有与位移相关的常量，如 `kShift16` 和 `kShift32`。

## 2. 前导零计数 clz

**clz** (count leading zeros) 用于计算一个无符号整数从最高有效位（MSB）开始连续"0"的个数。常用于：

- 高效查找最高位1的位置
- 位运算加速
- 浮点格式编码

用法举例：

```cpp
uint32_t v = 0x00000008; // 二进制: 00000000 00000000 00000000 00001000
int zeros = clz(v); // 结果为28
```

实现方式：  
优先调用编译器内建（如`__builtin_clz`），若无则用循环遍历。

## 3. 尾随零计数 ctz

**ctz** (count trailing zeros) 计算从最低有效位（LSB）连续“0”的数量。用途包括：

- 快速获取最低位1的位置
- 位图、哈希、压缩等算法优化

示例：

```cpp
uint32_t v = 0x00000008; // 二进制: 00000000 00000000 00000000 00001000
int zeros = ctz(v); // 结果为3
```

同样优先使用编译器内建。

## 4. 位1个数统计 popcount

**popcount** 计算整数二进制中“1”的个数（Hamming weight），常用于集合统计、hash、布尔数组计数等。

例子：

```cpp
uint32_t v = 0b1101; // 有3个1
int ones = popcount(v); // 结果为3
```

## 5. 2的幂操作

- `is_power_of_2(val)` 判断val是否是2的幂。
- `next_power_of_2(val)` 向上取最近的2次幂，比如15→16，33→64。

这些操作常用于分配内存、数据结构扩容、位对齐等。

## 6. 对齐工具

- `align_up(value, alignment)`：把value向上对齐到alignment的整数倍（alignment须为2的幂）。
- `align_down(value, alignment)`：向下对齐。
- `is_aligned(value, alignment)`：判断value是否已对齐。

举例：

```cpp
align_up(33, 16); // 结果为48
align_down(33, 16); // 结果为32
is_aligned(64, 16); // true
```

## 7. SIMD 对齐分区

用于 SIMD 向量化处理时，将数组分割为头部（未对齐）、主体（对齐块）、尾部（剩余）三部分。

### AlignmentPartition 结构体

```cpp
struct AlignmentPartition {
  std::size_t head;    // 首个对齐块之前的元素数
  std::size_t body;    // 对齐块中的元素数
  std::size_t tail;    // 最后对齐块之后的元素数
  std::size_t blocks;  // 对齐块数量
};
```

**不变量**：`head + body + tail == count`，`body == blocks * lane_count`

### partition_for_alignment

根据指针地址、元素个数、SIMD 通道数计算分区：

```cpp
// 使用原始指针地址
auto part = partition_for_alignment(
    reinterpret_cast<std::uintptr_t>(ptr),
    count,
    lane_count,
    sizeof(T)
);

// 使用类型化指针（自动推断元素大小）
float* data = ...;
auto part = partition_for_alignment(data, 100, 8);
// part.head: 未对齐头部元素数
// part.body: 对齐主体元素数（可用SIMD处理）
// part.tail: 未对齐尾部元素数
// part.blocks: 对齐块数量
```

### is_simd_aligned

快速检查指针是否对齐到指定 SIMD 通道数：

```cpp
float* ptr = ...;
if (is_simd_aligned(ptr, 8)) {
  // ptr 已对齐到 8 * sizeof(float) = 32 字节
}
```

---

**实现细节：**  
该文件实现既利用了编译器内建，亦有纯C++ fallback，适配不同平台和编译器。模板用来保证类型安全。

> 参考：`scl/core/bits.hpp`，用于所有需要底层位运算和内存对齐的代码场景。
