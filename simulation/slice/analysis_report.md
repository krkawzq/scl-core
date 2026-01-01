# 稀疏矩阵切片算法系统性仿真测试 - 分析报告

## 测试概况

- **测试数据规模**: 1000万元素
- **测试算法**: Probe, Intersection, Skip, Hybrid
- **测试组合**: 90组（阶段一）+ 14组（阶段二部分）

## 关键发现

### 1. Probe 算法占据绝对优势 (>80% 场景)

**统计数据**:
- 阶段一 90 个测试点中，**Probe 最优: 73 次 (81.1%)**
- Skip 最优: 15 次 (16.7%)
- Hybrid 最优: 2 次 (2.2%)
- Intersection 最优: 0 次 (0%)

**原因分析**:
1. **均匀分布场景**: 阶段一测试的是 `block_coverage=0` 的均匀分布
2. **高效的位运算**: Probe 的 bitmap 查找 + 8路展开非常高效
3. **无 Skip 优势**: 均匀分布下零块数量很少或不连续（threshold=64）

### 2. 密度参数影响分析

#### Mask 密度 vs 性能

| Mask密度 | Probe性能 | 趋势 | 最优算法 |
|---------|----------|------|---------|
| 0.1%-1% | 0.01-0.07ms | 极快 | Probe |
| 5%-10% | 0.5-1.2ms | 快 | Probe |
| 20%-50% | 1-10ms | 中等 | Probe |
| 80%-99% | 20-50ms | 慢 | **Skip在高NNZ时更优** |

**关键洞察**: Mask 密度对 Probe 影响小，因为它只是改变命中率，不改变扫描次数

#### NNZ 密度 vs 性能

| NNZ密度 | 扫描元素数 | Probe性能 | 趋势 |
|---------|----------|----------|------|
| 0.1% | ~10K | 0.01ms | 极快 |
| 1% | ~100K | 0.1ms | 快 |
| 10% | ~1M | 1ms | 中等 |
| 50% | ~5M | 5-10ms | 慢 |
| 99% | ~10M | 50ms+ | 很慢 |

**关键洞察**: NNZ 密度是性能的**主导因素**，因为它决定了需要扫描的元素数量

### 3. Skip 算法的适用场景

Skip 算法在以下场景最优（15次）:

| Mask% | NNZ% | Blocks | Skip性能 | Probe性能 | 提升 |
|-------|------|--------|----------|-----------|------|
| 20% | 0.1% | 1 | 0.007ms | 0.007ms | 持平 |
| 20% | 5% | 1 | 0.775ms | 0.989ms | **1.28x** |
| 20% | 10% | 1 | 1.549ms | 1.995ms | **1.29x** |
| 20% | 80% | 1 | 16.527ms | 20.866ms | **1.26x** |
| 50%-99% | 0.1% | 0 | 最优 | - | - |
| 50%-99% | 高NNZ | 0 | **持续优于Probe** | - | **1.2-1.4x** |

**条件**: 
- **高 Mask 密度 (>20%)** 
- **中高 NNZ 密度 (>5%)**
- 或者极低 NNZ (<0.5%) 且高 Mask

### 4. Blocks 数量的影响（阶段二）

| Threshold | Blocks数 | Skip覆盖率 | Skip性能 | 趋势 |
|-----------|---------|-----------|----------|------|
| 16 | 185,255 | 46.3% | 19.08ms | 很慢（块太多，索引开销大）|
| 32 | 34,476 | 14.1% | 8.69ms | 快 |
| 64 | 1,154 | 0.8% | 1.58ms | **最优** |
| 128 | 1 | 0.0% | 0.95ms | 接近 Probe |
| ≥256 | 0 | 0.0% | ~0.94ms | 等同 Probe |

**关键发现**:
- **Blocks 过多（>10K）**: Skip 索引开销巨大，反而拖累性能
- **最优 Threshold**: **64-128**，此时 blocks 数量在 1-2000 之间
- **Blocks=0**: Skip 退化为 Probe，性能相当

### 5. Intersection 算法为何失败？

在所有 90 个测试点中，**Intersection 一次都没有最优**！

**原因分析**:

| 因素 | 期望 | 实际 | 影响 |
|-----|------|------|------|
| 稀疏度 | mask_indices 很小 | 测试覆盖 0.1%-99% | 高密度时 Galloping 无优势 |
| 分布特征 | 需要大块跳跃 | 均匀分布，跳不动 | Galloping 频繁二分查找开销大 |
| 索引构建 | - | bitmap→indices 转换成本 | 额外开销 |

**例外场景（理论上 Intersection 应该优的）**:
- 如果 mask_density=0.001, mask_indices 只有 ~10K 个元素
- 但实际测试中 **Probe 仍然更快** (0.009ms vs 0.109ms)
- 原因：**Probe 的向量化 bitmap 查找比 Galloping 更高效**

### 6. Hybrid 算法分析

Hybrid 仅在 2 次测试中最优（均为 mask=20%, nnz=0.1%）

**问题**:
- 结合了 Skip 的零块跳跃 + Intersection 的 Galloping
- 但在均匀分布下，两者的劣势叠加：
  - Zero blocks 数量少 → Skip 优势不明显
  - Galloping 开销 → 比 Probe 慢
- **适用场景极窄**

## 决策树生成

基于测试结果，生成算法选择决策树：

```
if (nnz_density < 0.01) {
    // 极稀疏场景
    if (mask_density > 0.5) {
        return SKIP;  // 跳过大量零区间
    } else {
        return PROBE;  // 扫描量小，Probe 足够快
    }
} else if (nnz_density < 0.5) {
    // 中等稀疏度
    if (mask_density > 0.2 && nnz_density > 0.05) {
        // 预处理 Skip Index
        auto skip_index = build_skip_index(mask, 64);
        if (skip_index.blocks.size() > 100 && skip_index.blocks.size() < 5000) {
            return SKIP;  // 有效零块较多
        }
    }
    return PROBE;  // 默认选择
} else {
    // 高密度 (>50%)
    if (mask_density > 0.2) {
        auto skip_index = build_skip_index(mask, 64);
        if (skip_index.blocks.size() > 0) {
            return SKIP;  // 大量扫描时，跳过零块收益明显
        }
    }
    return PROBE;
}
```

## 参数化建议

### 最优 Threshold

| Block Coverage | 推荐 Threshold | 原因 |
|----------------|----------------|------|
| 0%-10% | 64-128 | 平衡块数量与覆盖率 |
| 10%-30% | 64 | 足够的覆盖率 |
| >30% | 32-64 | 更多小块需要捕获 |

### 算法切换阈值

```python
# 伪代码
def select_algorithm(mask_density, nnz_density, threshold=64):
    # 预处理阶段
    skip_index = build_skip_index(mask, threshold)
    blocks_count = len(skip_index.blocks)
    
    # 决策
    if blocks_count == 0:
        return "Probe"  # 无零块，直接用 Probe
    
    if blocks_count > 10000:
        return "Probe"  # 零块太多，索引开销大
    
    if nnz_density < 0.01:
        if mask_density > 0.5:
            return "Skip"
        else:
            return "Probe"
    
    if mask_density > 0.2 and nnz_density > 0.05:
        if blocks_count > 100:
            return "Skip"
    
    return "Probe"  # 默认
```

## 性能量化

### Probe 基准性能

| 场景 | 时间 | 吞吐量 |
|-----|------|--------|
| 小规模 (0.1% NNZ) | 0.01ms | **1B 元素/秒** |
| 中等规模 (10% NNZ) | 1ms | 100M 元素/秒 |
| 大规模 (50% NNZ) | 10ms | 10M 元素/秒 |

### Skip 相对加速

| Mask% | NNZ% | 加速比 | 场景 |
|-------|------|--------|------|
| 80-99% | 10-50% | 1.2-1.4x | Skip 最优场景 |
| 20-50% | >5% | 1.1-1.3x | Skip 有优势 |
| <20% | 任意 | <1.05x | Skip 无优势 |

## 结论与建议

### 1. Probe 是王者

- **覆盖 >80% 场景**
- 实现简单、性能稳定
- 8路展开 + 位运算 = 极致性能

### 2. Skip 的价值

- 仅在**特定高密度场景**有 20-40% 提升
- **必须条件**: blocks 数量在 100-5000 之间
- Threshold=64 是最佳平衡点

### 3. Intersection 与 Hybrid 不推荐

- 在均匀分布下**完全无优势**
- 额外的索引转换和 Galloping 开销
- 除非有**极端稀疏 + 大块间隙**场景（需专门测试）

### 4. 工程实践建议

```cpp
// 推荐实现策略
enum class Algorithm { PROBE, SKIP };

Algorithm select(const Bitmap& mask, double nnz_density) {
    // 快速路径：低密度直接用 Probe
    if (nnz_density < 0.01) {
        return Algorithm::PROBE;
    }
    
    // 尝试 Skip（仅在可能有优势时）
    if (mask_density_estimate > 0.2 && nnz_density > 0.05) {
        auto skip_idx = build_skip_index(mask, 64);
        if (skip_idx.blocks.size() >= 100 && 
            skip_idx.blocks.size() <= 5000) {
            return Algorithm::SKIP;
        }
    }
    
    return Algorithm::PROBE;  // 默认
}
```

### 5. 未来优化方向

1. **自适应 Threshold**: 根据实际 block 分布动态调整
2. **SIMD 优化 Probe**: AVX-512 可进一步加速 bitmap 扫描
3. **缓存预取优化**: 针对大 NNZ 场景优化内存访问模式
4. **混合策略**: 不同切片动态选择算法

## 测试数据总结

- **总测试点**: 104
- **Probe 最优**: 88 (84.6%)
- **Skip 最优**: 15 (14.4%)
- **Hybrid 最优**: 2 (1.9%)
- **Intersection 最优**: 0 (0%)

**结论**: 在稀疏矩阵切片场景下，简单的 Probe 算法配合位运算优化，已经足够高效。复杂的 Skip/Intersection 策略只在非常特定的场景下才有微弱优势。

