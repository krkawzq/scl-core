#include <vector>
#include <random>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>

// =============================================================================
// 数据结构
// =============================================================================

struct ZeroBlock {
    std::int64_t start;
    std::int64_t end;
};

struct SkipIndex {
    std::vector<ZeroBlock> blocks;
    std::size_t threshold;
    std::size_t total_skipped_range = 0;  // 统计：所有零块覆盖的总范围
};

// 测试参数配置
struct TestConfig {
    double mask_density;       // mask中1的比例 (0-1)
    double nnz_density;        // 数据稀疏度 (0-1) - 每个切片的元素占总空间的比例
    double block_coverage;     // 大块零区间覆盖率 (0-1)
    std::size_t threshold;     // 零块最小长度阈值
    uint32_t seed;             // 随机种子
};

// 测试结果
struct BenchmarkResult {
    double time_ms;
    std::size_t output_size;
    std::int64_t checksum;
    std::size_t elements_scanned = 0;   // 扫描的元素数
    std::size_t elements_skipped = 0;   // 跳过的元素数
};

// =============================================================================
// 数据生成：独立控制四参数
// =============================================================================

// 生成指定密度和块覆盖率的 mask bitmap
std::vector<uint8_t> generate_mask_bitmap(
    std::size_t bit_count,
    double mask_density,
    double block_coverage,
    std::size_t min_block_size,
    uint32_t seed
) {
    std::vector<uint8_t> bitmap((bit_count + 7) / 8, 0);
    std::mt19937 gen(seed);
    
    if (mask_density <= 0.0) return bitmap;
    if (mask_density >= 1.0) {
        std::memset(bitmap.data(), 0xFF, bitmap.size());
        if (bit_count % 8 != 0) {
            bitmap.back() &= (1u << (bit_count % 8)) - 1;
        }
        return bitmap;
    }
    
    // 计算目标1的数量
    std::size_t target_ones = static_cast<std::size_t>(bit_count * mask_density);
    
    if (block_coverage <= 0.0) {
        // 均匀分布
        std::bernoulli_distribution dist(mask_density);
        for (std::size_t i = 0; i < bit_count; ++i) {
            if (dist(gen)) {
                bitmap[i / 8] |= (1u << (i % 8));
            }
        }
        return bitmap;
    }
    
    // 带块状分布
    // 1. 先生成大块零区间
    std::size_t total_zero_range = static_cast<std::size_t>(bit_count * block_coverage);
    std::vector<std::pair<std::size_t, std::size_t>> zero_blocks;
    
    if (total_zero_range >= min_block_size) {
        // 生成多个零块
        std::uniform_int_distribution<std::size_t> block_size_dist(
            min_block_size, 
            std::min(min_block_size * 10, total_zero_range / 2)
        );
        
        std::size_t remaining = total_zero_range;
        while (remaining >= min_block_size) {
            std::size_t block_size = std::min(block_size_dist(gen), remaining);
            zero_blocks.push_back({0, block_size});  // 先记录大小，后面分配位置
            remaining -= block_size;
        }
        
        // 在空间中均匀分配这些零块
        std::size_t available_space = bit_count;
        std::uniform_int_distribution<std::size_t> pos_dist(0, available_space);
        
        for (auto& block : zero_blocks) {
            if (available_space < block.second) break;
            std::size_t start = pos_dist(gen) % (bit_count - block.second + 1);
            block.first = start;
            // 不需要实际写入，因为默认就是0
        }
    }
    
    // 2. 在非零块区域按调整后的密度填充1
    // 计算非零块区域的有效空间
    std::size_t effective_space = bit_count - total_zero_range;
    double adjusted_density = (effective_space > 0) ? 
        static_cast<double>(target_ones) / effective_space : 0.0;
    adjusted_density = std::min(adjusted_density, 1.0);
    
    std::bernoulli_distribution dist(adjusted_density);
    
    for (std::size_t i = 0; i < bit_count; ++i) {
        // 检查是否在零块中
        bool in_zero_block = false;
        for (const auto& block : zero_blocks) {
            if (i >= block.first && i < block.first + block.second) {
                in_zero_block = true;
                break;
            }
        }
        
        if (!in_zero_block && dist(gen)) {
            bitmap[i / 8] |= (1u << (i % 8));
        }
    }
    
    return bitmap;
}

// 生成指定稀疏度的索引数组（模拟稀疏矩阵切片）
std::vector<std::vector<std::int64_t>> generate_sparse_indices(
    std::size_t total_size,
    std::size_t num_slices,
    double nnz_density,
    uint32_t seed
) {
    std::vector<std::vector<std::int64_t>> indices(num_slices);
    std::mt19937 gen(seed);
    
    // 每个切片的元素数量
    std::size_t elements_per_slice = static_cast<std::size_t>(
        total_size * nnz_density / num_slices
    );
    
    if (elements_per_slice == 0) elements_per_slice = 1;
    
    std::uniform_int_distribution<std::int64_t> idx_dist(0, total_size - 1);
    
    for (std::size_t i = 0; i < num_slices; ++i) {
        indices[i].reserve(elements_per_slice);
        
        // 生成随机索引
        for (std::size_t j = 0; j < elements_per_slice; ++j) {
            indices[i].push_back(idx_dist(gen));
        }
        
        // 排序（交集算法需要）
        std::sort(indices[i].begin(), indices[i].end());
        
        // 去重
        indices[i].erase(std::unique(indices[i].begin(), indices[i].end()), indices[i].end());
    }
    
    return indices;
}

// bitmap 转索引
std::vector<std::int64_t> bitmap_to_indices(
    const std::vector<uint8_t>& bitmap, 
    std::size_t bit_count
) {
    std::vector<std::int64_t> indices;
    indices.reserve(bit_count / 10);
    
    for (std::size_t byte_idx = 0; byte_idx < bitmap.size(); ++byte_idx) {
        uint8_t byte = bitmap[byte_idx];
        if (byte == 0) continue;
        
        std::size_t base = byte_idx * 8;
        while (byte) {
            int bit = __builtin_ctz(byte);
            if (base + bit < bit_count) {
                indices.push_back(static_cast<std::int64_t>(base + bit));
            }
            byte &= byte - 1;
        }
    }
    return indices;
}

// 构建 Skip Index
SkipIndex build_skip_index(
    const std::vector<uint8_t>& bitmap, 
    std::size_t bit_count, 
    std::size_t threshold
) {
    SkipIndex index;
    index.threshold = threshold;
    
    std::size_t i = 0;
    while (i < bit_count) {
        // 找零块起始
        while (i < bit_count && (bitmap[i / 8] & (1u << (i % 8)))) {
            ++i;
        }
        if (i >= bit_count) break;
        
        std::size_t block_start = i;
        
        // 找零块结束
        std::size_t byte_idx = i / 8;
        if (i % 8 == 0) {
            while (byte_idx < bitmap.size() && bitmap[byte_idx] == 0) {
                ++byte_idx;
            }
            i = byte_idx * 8;
        }
        
        while (i < bit_count && !(bitmap[i / 8] & (1u << (i % 8)))) {
            ++i;
        }
        
        std::size_t block_size = i - block_start;
        if (block_size >= threshold) {
            index.blocks.push_back({
                static_cast<std::int64_t>(block_start),
                static_cast<std::int64_t>(i)
            });
            index.total_skipped_range += block_size;
        }
    }
    
    return index;
}

// =============================================================================
// 算法实现
// =============================================================================

// 算法1: Probe
__attribute__((hot, flatten))
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> probe_slice_optimized(
    const std::vector<std::vector<std::int64_t>>& values,
    const std::vector<std::vector<std::int64_t>>& indices,
    const std::vector<uint8_t>& mask,
    std::size_t* scanned_out = nullptr
) {
    std::size_t max_total = 0;
    for (const auto& idx_vec : indices) {
        max_total += idx_vec.size();
    }

    std::vector<std::int64_t> result_values;
    std::vector<std::int64_t> result_indices;
    result_values.reserve(max_total);
    result_indices.reserve(max_total);

    const uint8_t* __restrict mask_ptr = mask.data();
    std::size_t scanned = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int64_t* __restrict val_ptr = values[i].data();
        const std::int64_t* __restrict idx_ptr = indices[i].data();
        const std::size_t size = std::min(values[i].size(), indices[i].size());
        
        scanned += size;
        std::size_t j = 0;
        
        for (; j + 7 < size; j += 8) {
            __builtin_prefetch(idx_ptr + j + 16, 0, 3);
            __builtin_prefetch(val_ptr + j + 16, 0, 3);
            
            std::size_t idx0 = static_cast<std::size_t>(idx_ptr[j]);
            std::size_t idx1 = static_cast<std::size_t>(idx_ptr[j+1]);
            std::size_t idx2 = static_cast<std::size_t>(idx_ptr[j+2]);
            std::size_t idx3 = static_cast<std::size_t>(idx_ptr[j+3]);
            std::size_t idx4 = static_cast<std::size_t>(idx_ptr[j+4]);
            std::size_t idx5 = static_cast<std::size_t>(idx_ptr[j+5]);
            std::size_t idx6 = static_cast<std::size_t>(idx_ptr[j+6]);
            std::size_t idx7 = static_cast<std::size_t>(idx_ptr[j+7]);
            
            uint8_t m0 = (mask_ptr[idx0 >> 3] >> (idx0 & 7)) & 1;
            uint8_t m1 = (mask_ptr[idx1 >> 3] >> (idx1 & 7)) & 1;
            uint8_t m2 = (mask_ptr[idx2 >> 3] >> (idx2 & 7)) & 1;
            uint8_t m3 = (mask_ptr[idx3 >> 3] >> (idx3 & 7)) & 1;
            uint8_t m4 = (mask_ptr[idx4 >> 3] >> (idx4 & 7)) & 1;
            uint8_t m5 = (mask_ptr[idx5 >> 3] >> (idx5 & 7)) & 1;
            uint8_t m6 = (mask_ptr[idx6 >> 3] >> (idx6 & 7)) & 1;
            uint8_t m7 = (mask_ptr[idx7 >> 3] >> (idx7 & 7)) & 1;
            
            if (m0) { result_values.push_back(val_ptr[j]);   result_indices.push_back(idx_ptr[j]); }
            if (m1) { result_values.push_back(val_ptr[j+1]); result_indices.push_back(idx_ptr[j+1]); }
            if (m2) { result_values.push_back(val_ptr[j+2]); result_indices.push_back(idx_ptr[j+2]); }
            if (m3) { result_values.push_back(val_ptr[j+3]); result_indices.push_back(idx_ptr[j+3]); }
            if (m4) { result_values.push_back(val_ptr[j+4]); result_indices.push_back(idx_ptr[j+4]); }
            if (m5) { result_values.push_back(val_ptr[j+5]); result_indices.push_back(idx_ptr[j+5]); }
            if (m6) { result_values.push_back(val_ptr[j+6]); result_indices.push_back(idx_ptr[j+6]); }
            if (m7) { result_values.push_back(val_ptr[j+7]); result_indices.push_back(idx_ptr[j+7]); }
        }
        
        for (; j < size; ++j) {
            std::size_t idx = static_cast<std::size_t>(idx_ptr[j]);
            if ((mask_ptr[idx >> 3] >> (idx & 7)) & 1) {
                result_values.push_back(val_ptr[j]);
                result_indices.push_back(idx_ptr[j]);
            }
        }
    }

    if (scanned_out) *scanned_out = scanned;
    return {std::move(result_values), std::move(result_indices)};
}

// 算法2: Intersection
__attribute__((always_inline))
inline std::size_t galloping_search(
    const std::int64_t* __restrict arr,
    std::size_t start,
    std::size_t size,
    std::int64_t target
) {
    if (start >= size || arr[start] >= target) return start;
    
    std::size_t bound = 1;
    while (start + bound < size && arr[start + bound] < target) {
        bound <<= 1;
    }
    
    std::size_t low = start + (bound >> 1);
    std::size_t high = std::min(start + bound, size);
    
    while (low < high) {
        std::size_t mid = low + ((high - low) >> 1);
        if (arr[mid] < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__attribute__((hot, flatten))
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> intersection_optimized(
    const std::vector<std::vector<std::int64_t>>& values,
    const std::vector<std::vector<std::int64_t>>& indices,
    const std::vector<std::int64_t>& mask_indices,
    std::size_t* scanned_out = nullptr
) {
    if (mask_indices.empty()) {
        if (scanned_out) *scanned_out = 0;
        return {{}, {}};
    }
    
    std::size_t max_total = 0;
    for (const auto& idx_vec : indices) {
        max_total += idx_vec.size();
    }
    
    std::vector<std::int64_t> result_values;
    std::vector<std::int64_t> result_indices;
    result_values.reserve(std::min(max_total, mask_indices.size()));
    result_indices.reserve(std::min(max_total, mask_indices.size()));

    const std::int64_t* __restrict mask_ptr = mask_indices.data();
    const std::size_t mask_size = mask_indices.size();
    std::size_t scanned = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int64_t* __restrict val_ptr = values[i].data();
        const std::int64_t* __restrict idx_ptr = indices[i].data();
        const std::size_t idx_size = indices[i].size();
        
        scanned += idx_size;
        std::size_t j = 0;
        std::size_t k = 0;
        
        __builtin_prefetch(idx_ptr, 0, 3);
        __builtin_prefetch(mask_ptr, 0, 3);
        
        while (j < idx_size && k < mask_size) {
            const std::int64_t idx_val = idx_ptr[j];
            const std::int64_t mask_val = mask_ptr[k];
            
            if (idx_val == mask_val) {
                result_values.push_back(val_ptr[j]);
                result_indices.push_back(idx_val);
                ++j;
                ++k;
            } else if (idx_val < mask_val) {
                j = galloping_search(idx_ptr, j + 1, idx_size, mask_val);
            } else {
                k = galloping_search(mask_ptr, k + 1, mask_size, idx_val);
            }
        }
    }

    if (scanned_out) *scanned_out = scanned;
    return {std::move(result_values), std::move(result_indices)};
}

// 算法3: Skip
__attribute__((hot, flatten))
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> skip_slice_optimized(
    const std::vector<std::vector<std::int64_t>>& values,
    const std::vector<std::vector<std::int64_t>>& indices,
    const std::vector<uint8_t>& mask,
    const SkipIndex& skip_index,
    std::size_t* scanned_out = nullptr,
    std::size_t* skipped_out = nullptr
) {
    std::size_t max_total = 0;
    for (const auto& idx_vec : indices) {
        max_total += idx_vec.size();
    }

    std::vector<std::int64_t> result_values;
    std::vector<std::int64_t> result_indices;
    result_values.reserve(max_total);
    result_indices.reserve(max_total);

    const uint8_t* __restrict mask_ptr = mask.data();
    const ZeroBlock* __restrict blocks = skip_index.blocks.data();
    const std::size_t num_blocks = skip_index.blocks.size();
    
    std::size_t scanned = 0;
    std::size_t skipped = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int64_t* __restrict val_ptr = values[i].data();
        const std::int64_t* __restrict idx_ptr = indices[i].data();
        const std::size_t size = std::min(values[i].size(), indices[i].size());
        
        if (size == 0) continue;
        
        std::size_t block_idx = 0;
        std::int64_t block_start = (num_blocks > 0) ? blocks[0].start : INT64_MAX;
        std::int64_t block_end = (num_blocks > 0) ? blocks[0].end : INT64_MAX;
        
        std::size_t j = 0;
        
        while (j < size) {
            const std::int64_t current_idx = idx_ptr[j];
            
            // 在零块中，跳过
            if (current_idx >= block_start && current_idx < block_end) {
                std::size_t low = j;
                std::size_t high = size;
                while (low < high) {
                    std::size_t mid = low + ((high - low) >> 1);
                    if (idx_ptr[mid] < block_end) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                skipped += (low - j);
                j = low;
                
                ++block_idx;
                if (block_idx < num_blocks) {
                    block_start = blocks[block_idx].start;
                    block_end = blocks[block_idx].end;
                } else {
                    block_start = INT64_MAX;
                    block_end = INT64_MAX;
                }
                continue;
            }
            
            // 更新块指针
            while (block_idx < num_blocks && current_idx >= blocks[block_idx].end) {
                ++block_idx;
                if (block_idx < num_blocks) {
                    block_start = blocks[block_idx].start;
                    block_end = blocks[block_idx].end;
                } else {
                    block_start = INT64_MAX;
                    block_end = INT64_MAX;
                }
            }
            
            // 批量处理到下一个零块
            std::size_t batch_end = size;
            if (block_idx < num_blocks) {
                std::size_t low = j;
                std::size_t high = size;
                while (low < high) {
                    std::size_t mid = low + ((high - low) >> 1);
                    if (idx_ptr[mid] < block_start) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                batch_end = low;
            }
            
            // Probe 处理
            for (; j + 3 < batch_end; j += 4) {
                __builtin_prefetch(idx_ptr + j + 16, 0, 3);
                
                std::size_t idx0 = static_cast<std::size_t>(idx_ptr[j]);
                std::size_t idx1 = static_cast<std::size_t>(idx_ptr[j+1]);
                std::size_t idx2 = static_cast<std::size_t>(idx_ptr[j+2]);
                std::size_t idx3 = static_cast<std::size_t>(idx_ptr[j+3]);
                
                uint8_t m0 = (mask_ptr[idx0 >> 3] >> (idx0 & 7)) & 1;
                uint8_t m1 = (mask_ptr[idx1 >> 3] >> (idx1 & 7)) & 1;
                uint8_t m2 = (mask_ptr[idx2 >> 3] >> (idx2 & 7)) & 1;
                uint8_t m3 = (mask_ptr[idx3 >> 3] >> (idx3 & 7)) & 1;
                
                if (m0) { result_values.push_back(val_ptr[j]);   result_indices.push_back(idx_ptr[j]); }
                if (m1) { result_values.push_back(val_ptr[j+1]); result_indices.push_back(idx_ptr[j+1]); }
                if (m2) { result_values.push_back(val_ptr[j+2]); result_indices.push_back(idx_ptr[j+2]); }
                if (m3) { result_values.push_back(val_ptr[j+3]); result_indices.push_back(idx_ptr[j+3]); }
            }
            
            for (; j < batch_end; ++j) {
                std::size_t idx = static_cast<std::size_t>(idx_ptr[j]);
                if ((mask_ptr[idx >> 3] >> (idx & 7)) & 1) {
                    result_values.push_back(val_ptr[j]);
                    result_indices.push_back(idx_ptr[j]);
                }
            }
        }
        
        scanned += size;
    }

    if (scanned_out) *scanned_out = scanned;
    if (skipped_out) *skipped_out = skipped;
    return {std::move(result_values), std::move(result_indices)};
}

// 算法4: Hybrid
__attribute__((hot, flatten))
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> hybrid_optimized(
    const std::vector<std::vector<std::int64_t>>& values,
    const std::vector<std::vector<std::int64_t>>& indices,
    const std::vector<std::int64_t>& mask_indices,
    const SkipIndex& skip_index,
    std::size_t* scanned_out = nullptr,
    std::size_t* skipped_out = nullptr
) {
    if (mask_indices.empty()) {
        if (scanned_out) *scanned_out = 0;
        if (skipped_out) *skipped_out = 0;
        return {{}, {}};
    }
    
    std::size_t max_total = 0;
    for (const auto& idx_vec : indices) {
        max_total += idx_vec.size();
    }
    
    std::vector<std::int64_t> result_values;
    std::vector<std::int64_t> result_indices;
    result_values.reserve(std::min(max_total, mask_indices.size()));
    result_indices.reserve(std::min(max_total, mask_indices.size()));

    const std::int64_t* __restrict mask_ptr = mask_indices.data();
    const std::size_t mask_size = mask_indices.size();
    const ZeroBlock* __restrict blocks = skip_index.blocks.data();
    const std::size_t num_blocks = skip_index.blocks.size();
    
    std::size_t scanned = 0;
    std::size_t skipped = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int64_t* __restrict val_ptr = values[i].data();
        const std::int64_t* __restrict idx_ptr = indices[i].data();
        const std::size_t idx_size = indices[i].size();
        
        if (idx_size == 0) continue;
        
        scanned += idx_size;
        
        std::size_t j = 0;
        std::size_t k = 0;
        std::size_t block_idx = 0;
        
        std::int64_t block_start = (num_blocks > 0) ? blocks[0].start : INT64_MAX;
        std::int64_t block_end = (num_blocks > 0) ? blocks[0].end : INT64_MAX;
        
        while (j < idx_size && k < mask_size) {
            const std::int64_t idx_val = idx_ptr[j];
            const std::int64_t mask_val = mask_ptr[k];
            
            // 检查并跳过零块
            if (idx_val >= block_start && idx_val < block_end) {
                std::size_t old_j = j;
                j = galloping_search(idx_ptr, j, idx_size, block_end);
                skipped += (j - old_j);
                
                ++block_idx;
                if (block_idx < num_blocks) {
                    block_start = blocks[block_idx].start;
                    block_end = blocks[block_idx].end;
                } else {
                    block_start = INT64_MAX;
                    block_end = INT64_MAX;
                }
                continue;
            }
            
            if (mask_val >= block_start && mask_val < block_end) {
                k = galloping_search(mask_ptr, k, mask_size, block_end);
                continue;
            }
            
            // 更新块指针
            while (block_idx < num_blocks && std::min(idx_val, mask_val) >= blocks[block_idx].end) {
                ++block_idx;
                if (block_idx < num_blocks) {
                    block_start = blocks[block_idx].start;
                    block_end = blocks[block_idx].end;
                } else {
                    block_start = INT64_MAX;
                    block_end = INT64_MAX;
                }
            }
            
            // 标准 intersection
            if (idx_val == mask_val) {
                result_values.push_back(val_ptr[j]);
                result_indices.push_back(idx_val);
                ++j;
                ++k;
            } else if (idx_val < mask_val) {
                std::int64_t target = mask_val;
                if (block_start < target) target = block_start;
                j = galloping_search(idx_ptr, j + 1, idx_size, target);
            } else {
                std::int64_t target = idx_val;
                if (block_start < target) target = block_start;
                k = galloping_search(mask_ptr, k + 1, mask_size, target);
            }
        }
    }

    if (scanned_out) *scanned_out = scanned;
    if (skipped_out) *skipped_out = skipped;
    return {std::move(result_values), std::move(result_indices)};
}

// =============================================================================
// 基准测试框架
// =============================================================================

template<typename Func>
BenchmarkResult benchmark(Func&& func, int iterations = 5) {
    // 预热
    auto [vals, idxs] = func();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = func();
        vals = std::move(result.first);
        idxs = std::move(result.second);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    std::int64_t checksum = 0;
    for (auto val : vals) checksum += val;
    
    return {time_ms, vals.size(), checksum};
}

// =============================================================================
// 三阶段测试
// =============================================================================

void run_phase1_uniform_density_test(std::ofstream& csv_out) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "阶段一：mask_density × nnz_density 交叉测试（均匀分布）" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    const std::size_t TOTAL_SIZE = 10000000;
    const std::size_t NUM_SLICES = 10;
    const int ITERATIONS = 3;
    
    std::vector<double> mask_densities = {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99};
    std::vector<double> nnz_densities = {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99};
    
    csv_out << "phase,mask_density,nnz_density,block_coverage,threshold,"
            << "probe_ms,intersect_ms,skip_ms,hybrid_ms,"
            << "output_size,num_blocks,skip_coverage_pct,best_algo\n";
    
    std::cout << std::setw(10) << "Mask%" 
              << std::setw(10) << "NNZ%"
              << std::setw(10) << "Blocks"
              << std::setw(11) << "Probe"
              << std::setw(11) << "Intersect"
              << std::setw(11) << "Skip"
              << std::setw(11) << "Hybrid"
              << std::setw(11) << "Best"
              << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (double mask_d : mask_densities) {
        for (double nnz_d : nnz_densities) {
            TestConfig config{mask_d, nnz_d, 0.0, 64, 42};
            
            auto mask_bitmap = generate_mask_bitmap(TOTAL_SIZE, mask_d, 0.0, 64, config.seed);
            auto mask_indices = bitmap_to_indices(mask_bitmap, TOTAL_SIZE);
            auto skip_index = build_skip_index(mask_bitmap, TOTAL_SIZE, config.threshold);
            auto indices = generate_sparse_indices(TOTAL_SIZE, NUM_SLICES, nnz_d, config.seed + 1);
            
            // 生成values
            std::vector<std::vector<std::int64_t>> values(NUM_SLICES);
            std::mt19937 gen(config.seed + 2);
            std::uniform_int_distribution<std::int64_t> val_dist(0, 1000000);
            for (std::size_t i = 0; i < NUM_SLICES; ++i) {
                values[i].resize(indices[i].size());
                for (auto& v : values[i]) v = val_dist(gen);
            }
            
            auto r1 = benchmark([&]() { return probe_slice_optimized(values, indices, mask_bitmap); }, ITERATIONS);
            auto r2 = benchmark([&]() { return intersection_optimized(values, indices, mask_indices); }, ITERATIONS);
            auto r3 = benchmark([&]() { return skip_slice_optimized(values, indices, mask_bitmap, skip_index); }, ITERATIONS);
            auto r4 = benchmark([&]() { return hybrid_optimized(values, indices, mask_indices, skip_index); }, ITERATIONS);
            
            double min_time = std::min({r1.time_ms, r2.time_ms, r3.time_ms, r4.time_ms});
            const char* best = (min_time == r1.time_ms) ? "Probe" :
                              (min_time == r2.time_ms) ? "Intersect" :
                              (min_time == r3.time_ms) ? "Skip" : "Hybrid";
            
            // 计算 skip coverage（零块覆盖的范围占总长度的比例）
            double skip_coverage_pct = (skip_index.total_skipped_range * 100.0) / TOTAL_SIZE;
            
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(10) << (mask_d * 100)
                      << std::setw(10) << (nnz_d * 100)
                      << std::setw(10) << skip_index.blocks.size()
                      << std::setw(11) << r1.time_ms
                      << std::setw(11) << r2.time_ms
                      << std::setw(11) << r3.time_ms
                      << std::setw(11) << r4.time_ms
                      << std::setw(11) << best
                      << std::endl;
            
            csv_out << "1," << mask_d << "," << nnz_d << ",0.0,64,"
                    << r1.time_ms << "," << r2.time_ms << "," << r3.time_ms << "," << r4.time_ms << ","
                    << r1.output_size << "," << skip_index.blocks.size() << "," 
                    << skip_coverage_pct << "," << best << "\n";
        }
    }
}

void run_phase2_block_threshold_test(std::ofstream& csv_out) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "阶段二：block_coverage × threshold 交叉测试" << std::endl;
    std::cout << "（目标：找出 blocks 数量对算法性能的影响）" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    const std::size_t TOTAL_SIZE = 10000000;
    const std::size_t NUM_SLICES = 10;
    const int ITERATIONS = 3;
    const double MASK_DENSITY = 0.1;
    const double NNZ_DENSITY = 0.1;
    
    std::vector<double> block_coverages = {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95};
    std::vector<std::size_t> thresholds = {16, 32, 64, 128, 256, 512, 1024};
    
    std::cout << std::setw(12) << "Coverage%"
              << std::setw(12) << "Threshold"
              << std::setw(10) << "Blocks"
              << std::setw(10) << "SkipCov%"
              << std::setw(11) << "Probe"
              << std::setw(11) << "Skip"
              << std::setw(11) << "Hybrid"
              << std::setw(11) << "Best"
              << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    
    for (double coverage : block_coverages) {
        for (std::size_t thresh : thresholds) {
            TestConfig config{MASK_DENSITY, NNZ_DENSITY, coverage, thresh, 42};
            
            auto mask_bitmap = generate_mask_bitmap(TOTAL_SIZE, MASK_DENSITY, coverage, thresh, config.seed);
            auto mask_indices = bitmap_to_indices(mask_bitmap, TOTAL_SIZE);
            auto skip_index = build_skip_index(mask_bitmap, TOTAL_SIZE, thresh);
            auto indices = generate_sparse_indices(TOTAL_SIZE, NUM_SLICES, NNZ_DENSITY, config.seed + 1);
            
            std::vector<std::vector<std::int64_t>> values(NUM_SLICES);
            std::mt19937 gen(config.seed + 2);
            std::uniform_int_distribution<std::int64_t> val_dist(0, 1000000);
            for (std::size_t i = 0; i < NUM_SLICES; ++i) {
                values[i].resize(indices[i].size());
                for (auto& v : values[i]) v = val_dist(gen);
            }
            
            auto r1 = benchmark([&]() { return probe_slice_optimized(values, indices, mask_bitmap); }, ITERATIONS);
            auto r3 = benchmark([&]() { return skip_slice_optimized(values, indices, mask_bitmap, skip_index); }, ITERATIONS);
            auto r4 = benchmark([&]() { return hybrid_optimized(values, indices, mask_indices, skip_index); }, ITERATIONS);
            
            double min_time = std::min({r1.time_ms, r3.time_ms, r4.time_ms});
            const char* best = (min_time == r1.time_ms) ? "Probe" :
                              (min_time == r3.time_ms) ? "Skip" : "Hybrid";
            
            double skip_coverage_pct = (skip_index.total_skipped_range * 100.0) / TOTAL_SIZE;
            
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(12) << (coverage * 100)
                      << std::setw(12) << thresh
                      << std::setw(10) << skip_index.blocks.size()
                      << std::setw(10) << skip_coverage_pct
                      << std::setprecision(3)
                      << std::setw(11) << r1.time_ms
                      << std::setw(11) << r3.time_ms
                      << std::setw(11) << r4.time_ms
                      << std::setw(11) << best
                      << std::endl;
            
            csv_out << "2," << MASK_DENSITY << "," << NNZ_DENSITY << "," << coverage << "," << thresh << ","
                    << r1.time_ms << ",0," << r3.time_ms << "," << r4.time_ms << ","
                    << r3.output_size << "," << skip_index.blocks.size() << ","
                    << skip_coverage_pct << "," << best << "\n";
        }
    }
}

void run_phase3_full_grid_search(std::ofstream& csv_out) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "阶段三：全参数网格搜索（关键区间）" << std::endl;
    std::cout << "（结合 mask/nnz 密度 + block coverage）" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    const std::size_t TOTAL_SIZE = 10000000;
    const std::size_t NUM_SLICES = 10;
    const int ITERATIONS = 3;
    
    // 关键区间：重点测试边界情况
    std::vector<double> mask_densities = {0.001, 0.01, 0.05, 0.1, 0.3, 0.7};
    std::vector<double> nnz_densities = {0.001, 0.01, 0.1, 0.5};
    std::vector<double> block_coverages = {0.0, 0.3, 0.6, 0.9};
    std::vector<std::size_t> thresholds = {32, 64, 128};
    
    int total = mask_densities.size() * nnz_densities.size() * block_coverages.size() * thresholds.size();
    int current = 0;
    
    std::cout << "总计 " << total << " 个测试点..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (double mask_d : mask_densities) {
        for (double nnz_d : nnz_densities) {
            for (double coverage : block_coverages) {
                for (std::size_t thresh : thresholds) {
                    ++current;
                    
                    TestConfig config{mask_d, nnz_d, coverage, thresh, 42};
                    
                    auto mask_bitmap = generate_mask_bitmap(TOTAL_SIZE, mask_d, coverage, thresh, config.seed);
                    auto mask_indices = bitmap_to_indices(mask_bitmap, TOTAL_SIZE);
                    auto skip_index = build_skip_index(mask_bitmap, TOTAL_SIZE, thresh);
                    auto indices = generate_sparse_indices(TOTAL_SIZE, NUM_SLICES, nnz_d, config.seed + 1);
                    
                    std::vector<std::vector<std::int64_t>> values(NUM_SLICES);
                    std::mt19937 gen(config.seed + 2);
                    std::uniform_int_distribution<std::int64_t> val_dist(0, 1000000);
                    for (std::size_t i = 0; i < NUM_SLICES; ++i) {
                        values[i].resize(indices[i].size());
                        for (auto& v : values[i]) v = val_dist(gen);
                    }
                    
                    auto r1 = benchmark([&]() { return probe_slice_optimized(values, indices, mask_bitmap); }, ITERATIONS);
                    auto r2 = benchmark([&]() { return intersection_optimized(values, indices, mask_indices); }, ITERATIONS);
                    auto r3 = benchmark([&]() { return skip_slice_optimized(values, indices, mask_bitmap, skip_index); }, ITERATIONS);
                    auto r4 = benchmark([&]() { return hybrid_optimized(values, indices, mask_indices, skip_index); }, ITERATIONS);
                    
                    double min_time = std::min({r1.time_ms, r2.time_ms, r3.time_ms, r4.time_ms});
                    const char* best = (min_time == r1.time_ms) ? "Probe" :
                                      (min_time == r2.time_ms) ? "Intersect" :
                                      (min_time == r3.time_ms) ? "Skip" : "Hybrid";
                    
                    double skip_coverage_pct = (skip_index.total_skipped_range * 100.0) / TOTAL_SIZE;
                    
                    if (current % 10 == 0) {
                        std::cout << "进度: " << current << "/" << total 
                                  << " (" << (100 * current / total) << "%) | "
                                  << "Mask:" << (mask_d*100) << "% NNZ:" << (nnz_d*100) 
                                  << "% Coverage:" << (coverage*100) << "% Blocks:" << skip_index.blocks.size()
                                  << " -> " << best << std::endl;
                    }
                    
                    csv_out << "3," << mask_d << "," << nnz_d << "," << coverage << "," << thresh << ","
                            << r1.time_ms << "," << r2.time_ms << "," << r3.time_ms << "," << r4.time_ms << ","
                            << r1.output_size << "," << skip_index.blocks.size() << ","
                            << skip_coverage_pct << "," << best << "\n";
                }
            }
        }
    }
    
    std::cout << "完成！" << std::endl;
}

// =============================================================================
// 主函数
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "稀疏矩阵切片算法系统性仿真测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "四种算法：Probe, Intersection, Skip, Hybrid" << std::endl;
    std::cout << "四个参数：mask_density, nnz_density, block_coverage, threshold" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 打开CSV文件
    std::ofstream csv_out("simulation_results.csv");
    if (!csv_out) {
        std::cerr << "无法创建CSV文件！" << std::endl;
        return 1;
    }
    
    // 运行三阶段测试
    run_phase1_uniform_density_test(csv_out);
    run_phase2_block_threshold_test(csv_out);
    run_phase3_full_grid_search(csv_out);
    
    csv_out.close();
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "测试完成！结果已保存到 simulation_results.csv" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n推荐使用 Python/R 进行后续分析：" << std::endl;
    std::cout << "- 绘制热力图：mask_density × nnz_density" << std::endl;
    std::cout << "- 分析最优阈值：block_coverage 影响" << std::endl;
    std::cout << "- 生成决策树/查找表" << std::endl;
    
    return 0;
}
