#pragma once

#include "scl/core/type.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/matrix.hpp"
#include "scl/mmap/convert.hpp"

#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_map>
#include <memory>

namespace scl::mmap {

// =============================================================================
// SECTION 1: Heuristic Cache System
// =============================================================================

/// @brief 访问模式类型
enum class AccessPattern {
    Unknown,
    Sequential,     // 顺序扫描
    Strided,        // 跨步访问
    Random,         // 随机访问
    Repeated        // 重复访问同一行
};

/// @brief 启发式缓存管理器
///
/// 特点：
/// - 自动检测访问模式
/// - 根据模式调整预取策略
/// - 支持多行缓存
template <typename T>
class HeuristicCache {
public:
    struct CacheEntry {
        scl::Index row_idx;
        std::vector<T> values;
        std::vector<scl::Index> indices;
        std::size_t access_count;
        std::size_t last_access;
    };

private:
    std::size_t max_entries_;
    std::size_t access_counter_ = 0;
    std::vector<CacheEntry> entries_;
    std::unordered_map<scl::Index, std::size_t> index_map_;

    // 模式检测
    std::vector<scl::Index> recent_accesses_;
    static constexpr std::size_t PATTERN_WINDOW = 16;
    AccessPattern detected_pattern_ = AccessPattern::Unknown;

public:
    explicit HeuristicCache(std::size_t max_entries = 32)
        : max_entries_(max_entries) {
        entries_.reserve(max_entries);
        recent_accesses_.reserve(PATTERN_WINDOW);
    }

    /// @brief 检查是否缓存命中
    bool contains(scl::Index row_idx) const {
        return index_map_.find(row_idx) != index_map_.end();
    }

    /// @brief 获取缓存条目
    CacheEntry* get(scl::Index row_idx) {
        auto it = index_map_.find(row_idx);
        if (it != index_map_.end()) {
            auto& entry = entries_[it->second];
            entry.access_count++;
            entry.last_access = ++access_counter_;
            record_access(row_idx);
            return &entry;
        }
        return nullptr;
    }

    /// @brief 插入缓存
    template <bool IsCSR>
    void insert(scl::Index row_idx, const MappedSparse<T, IsCSR>& mat) {
        record_access(row_idx);

        // 检查是否需要驱逐
        if (entries_.size() >= max_entries_) {
            evict_one();
        }

        CacheEntry entry;
        entry.row_idx = row_idx;
        entry.access_count = 1;
        entry.last_access = ++access_counter_;

        scl::Array<T> vals;
        scl::Array<scl::Index> inds;
        if constexpr (IsCSR) {
            vals = mat.row_values(row_idx);
            inds = mat.row_indices(row_idx);
        } else {
            vals = mat.col_values(row_idx);
            inds = mat.col_indices(row_idx);
        }

        entry.values.assign(vals.begin(), vals.end());
        entry.indices.assign(inds.begin(), inds.end());

        index_map_[row_idx] = entries_.size();
        entries_.push_back(std::move(entry));
    }

    /// @brief 获取检测到的访问模式
    AccessPattern pattern() const { return detected_pattern_; }

    /// @brief 清空缓存
    void clear() {
        entries_.clear();
        index_map_.clear();
        recent_accesses_.clear();
        detected_pattern_ = AccessPattern::Unknown;
    }

    /// @brief 预取建议
    std::vector<scl::Index> prefetch_hint(scl::Index current, scl::Index max_rows) const {
        std::vector<scl::Index> hints;

        switch (detected_pattern_) {
            case AccessPattern::Sequential:
                // 预取后续行
                for (scl::Index i = 1; i <= 4 && current + i < max_rows; ++i) {
                    if (!contains(current + i)) {
                        hints.push_back(current + i);
                    }
                }
                break;

            case AccessPattern::Strided:
                // 检测步长并预取
                if (recent_accesses_.size() >= 2) {
                    scl::Index stride = recent_accesses_.back() -
                                        recent_accesses_[recent_accesses_.size() - 2];
                    if (stride > 0 && current + stride < max_rows) {
                        if (!contains(current + stride)) {
                            hints.push_back(current + stride);
                        }
                    }
                }
                break;

            default:
                break;
        }

        return hints;
    }

private:
    void record_access(scl::Index row_idx) {
        if (recent_accesses_.size() >= PATTERN_WINDOW) {
            recent_accesses_.erase(recent_accesses_.begin());
        }
        recent_accesses_.push_back(row_idx);
        detect_pattern();
    }

    void detect_pattern() {
        if (recent_accesses_.size() < 4) {
            detected_pattern_ = AccessPattern::Unknown;
            return;
        }

        // 检查顺序模式
        bool is_sequential = true;
        for (std::size_t i = 1; i < recent_accesses_.size(); ++i) {
            if (recent_accesses_[i] != recent_accesses_[i-1] + 1) {
                is_sequential = false;
                break;
            }
        }
        if (is_sequential) {
            detected_pattern_ = AccessPattern::Sequential;
            return;
        }

        // 检查跨步模式
        if (recent_accesses_.size() >= 3) {
            scl::Index stride = recent_accesses_[1] - recent_accesses_[0];
            bool is_strided = true;
            for (std::size_t i = 2; i < recent_accesses_.size(); ++i) {
                if (recent_accesses_[i] - recent_accesses_[i-1] != stride) {
                    is_strided = false;
                    break;
                }
            }
            if (is_strided && stride > 0) {
                detected_pattern_ = AccessPattern::Strided;
                return;
            }
        }

        // 检查重复模式
        std::unordered_map<scl::Index, int> freq;
        for (auto idx : recent_accesses_) {
            freq[idx]++;
        }
        if (freq.size() <= recent_accesses_.size() / 2) {
            detected_pattern_ = AccessPattern::Repeated;
            return;
        }

        detected_pattern_ = AccessPattern::Random;
    }

    void evict_one() {
        if (entries_.empty()) return;

        // LRU + 访问频率混合策略
        std::size_t victim = 0;
        double min_score = std::numeric_limits<double>::max();

        for (std::size_t i = 0; i < entries_.size(); ++i) {
            // score = recency * 0.5 + frequency * 0.5
            double recency = static_cast<double>(access_counter_ - entries_[i].last_access);
            double frequency = 1.0 / (entries_[i].access_count + 1);
            double score = recency * 0.5 + frequency * 100.0;

            if (score < min_score) {
                min_score = score;
                victim = i;
            }
        }

        // 移除 victim
        scl::Index victim_row = entries_[victim].row_idx;
        index_map_.erase(victim_row);

        // 移动最后一个到 victim 位置
        if (victim != entries_.size() - 1) {
            entries_[victim] = std::move(entries_.back());
            index_map_[entries_[victim].row_idx] = victim;
        }
        entries_.pop_back();
    }
};

// =============================================================================
// SECTION 2: Row-wise Statistics
// =============================================================================

/// @brief 计算每行的和
template <typename T, bool IsCSR>
void row_sum(const MappedSparse<T, IsCSR>& mat, T* out) {
    scl::Index rows = mat.rows();

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
        } else {
            // CSC: 需要不同处理，这里简化为遍历
            vals = mat.col_values(i);
        }

        T sum = T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            sum += vals[static_cast<scl::Index>(k)];
        }
        out[i] = sum;
    }
}

/// @brief 计算每行的均值
template <typename T, bool IsCSR>
void row_mean(const MappedSparse<T, IsCSR>& mat, T* out, bool count_zeros = true) {
    scl::Index rows = mat.rows();
    scl::Index cols = mat.cols();

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
        } else {
            vals = mat.col_values(i);
        }

        T sum = T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            sum += vals[static_cast<scl::Index>(k)];
        }

        scl::Index divisor = count_zeros ? cols : static_cast<scl::Index>(vals.size());
        out[i] = divisor > 0 ? sum / static_cast<T>(divisor) : T(0);
    }
}

/// @brief 计算每行的方差
template <typename T, bool IsCSR>
void row_var(const MappedSparse<T, IsCSR>& mat, T* out,
             const T* means = nullptr, bool count_zeros = true) {
    scl::Index rows = mat.rows();
    scl::Index cols = mat.cols();

    // 如果没有提供均值，先计算
    std::vector<T> mean_buf;
    if (!means) {
        mean_buf.resize(static_cast<std::size_t>(rows));
        row_mean<T, IsCSR>(mat, mean_buf.data(), count_zeros);
        means = mean_buf.data();
    }

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
        } else {
            vals = mat.col_values(i);
        }

        T mean = means[i];
        T sum_sq = T(0);

        for (scl::Size k = 0; k < vals.size(); ++k) {
            T diff = vals[static_cast<scl::Index>(k)] - mean;
            sum_sq += diff * diff;
        }

        // 加上零值的贡献
        if (count_zeros) {
            scl::Index num_zeros = cols - static_cast<scl::Index>(vals.size());
            sum_sq += static_cast<T>(num_zeros) * mean * mean;
        }

        scl::Index divisor = count_zeros ? cols : static_cast<scl::Index>(vals.size());
        out[i] = divisor > 0 ? sum_sq / static_cast<T>(divisor) : T(0);
    }
}

/// @brief 计算每行的最大值
template <typename T, bool IsCSR>
void row_max(const MappedSparse<T, IsCSR>& mat, T* out, bool consider_zeros = true) {
    scl::Index rows = mat.rows();

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Index len;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            len = mat.row_length(i);
        } else {
            vals = mat.col_values(i);
            len = mat.col_length(i);
        }

        if (vals.size() == 0) {
            out[i] = T(0);
            continue;
        }

        T max_val = vals[0];
        for (scl::Size k = 1; k < vals.size(); ++k) {
            max_val = std::max(max_val, vals[static_cast<scl::Index>(k)]);
        }

        // 稀疏矩阵中未存储的元素为 0
        if (consider_zeros && static_cast<scl::Index>(vals.size()) < mat.cols()) {
            max_val = std::max(max_val, T(0));
        }

        out[i] = max_val;
    }
}

/// @brief 计算每行的最小值
template <typename T, bool IsCSR>
void row_min(const MappedSparse<T, IsCSR>& mat, T* out, bool consider_zeros = true) {
    scl::Index rows = mat.rows();

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
        } else {
            vals = mat.col_values(i);
        }

        if (vals.size() == 0) {
            out[i] = T(0);
            continue;
        }

        T min_val = vals[0];
        for (scl::Size k = 1; k < vals.size(); ++k) {
            min_val = std::min(min_val, vals[static_cast<scl::Index>(k)]);
        }

        if (consider_zeros && static_cast<scl::Index>(vals.size()) < mat.cols()) {
            min_val = std::min(min_val, T(0));
        }

        out[i] = min_val;
    }
}

// =============================================================================
// SECTION 3: Column-wise Statistics (CSR)
// =============================================================================

/// @brief 计算每列的和 (CSR 格式)
template <typename T>
void col_sum_csr(const MappedSparse<T, true>& mat, T* out) {
    scl::Index rows = mat.rows();
    scl::Index cols = mat.cols();

    std::memset(out, 0, static_cast<std::size_t>(cols) * sizeof(T));

    for (scl::Index i = 0; i < rows; ++i) {
        auto vals = mat.row_values(i);
        auto inds = mat.row_indices(i);

        for (scl::Size k = 0; k < vals.size(); ++k) {
            out[inds[static_cast<scl::Index>(k)]] += vals[static_cast<scl::Index>(k)];
        }
    }
}

/// @brief 计算每列的均值 (CSR 格式)
template <typename T>
void col_mean_csr(const MappedSparse<T, true>& mat, T* out, bool count_zeros = true) {
    scl::Index rows = mat.rows();
    scl::Index cols = mat.cols();

    col_sum_csr(mat, out);

    T divisor = count_zeros ? static_cast<T>(rows) : T(1);  // 简化处理

    for (scl::Index j = 0; j < cols; ++j) {
        out[j] /= divisor;
    }
}

/// @brief 计算每列的非零元素数
template <typename T>
void col_nnz_csr(const MappedSparse<T, true>& mat, scl::Index* out) {
    scl::Index rows = mat.rows();
    scl::Index cols = mat.cols();

    std::memset(out, 0, static_cast<std::size_t>(cols) * sizeof(scl::Index));

    for (scl::Index i = 0; i < rows; ++i) {
        auto inds = mat.row_indices(i);
        for (scl::Size k = 0; k < inds.size(); ++k) {
            ++out[inds[static_cast<scl::Index>(k)]];
        }
    }
}

// =============================================================================
// SECTION 4: Global Statistics
// =============================================================================

/// @brief 计算全局和
template <typename T, bool IsCSR>
T global_sum(const MappedSparse<T, IsCSR>& mat) {
    scl::Index n = IsCSR ? mat.rows() : mat.cols();
    T total = T(0);

    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        if constexpr (IsCSR) {
            vals = mat.row_values(i);
        } else {
            vals = mat.col_values(i);
        }

        for (scl::Size k = 0; k < vals.size(); ++k) {
            total += vals[static_cast<scl::Index>(k)];
        }
    }

    return total;
}

/// @brief 计算全局均值
template <typename T, bool IsCSR>
T global_mean(const MappedSparse<T, IsCSR>& mat, bool count_zeros = true) {
    T sum = global_sum<T, IsCSR>(mat);
    std::size_t count = count_zeros
        ? static_cast<std::size_t>(mat.rows()) * static_cast<std::size_t>(mat.cols())
        : static_cast<std::size_t>(mat.nnz());
    return count > 0 ? sum / static_cast<T>(count) : T(0);
}

// =============================================================================
// SECTION 5: Normalization
// =============================================================================

/// @brief L1 行归一化 (输出到新数组)
template <typename T, bool IsCSR>
void normalize_l1_rows(const MappedSparse<T, IsCSR>& mat,
                       T* data_out,
                       scl::Index* indices_out,
                       scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        // 计算 L1 范数
        T norm = T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            norm += std::abs(vals[static_cast<scl::Index>(k)]);
        }

        // 归一化并输出
        T scale = norm > T(0) ? T(1) / norm : T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = vals[static_cast<scl::Index>(k)] * scale;
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

/// @brief L2 行归一化
template <typename T, bool IsCSR>
void normalize_l2_rows(const MappedSparse<T, IsCSR>& mat,
                       T* data_out,
                       scl::Index* indices_out,
                       scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        // 计算 L2 范数
        T sum_sq = T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            T v = vals[static_cast<scl::Index>(k)];
            sum_sq += v * v;
        }
        T norm = std::sqrt(sum_sq);

        // 归一化并输出
        T scale = norm > T(0) ? T(1) / norm : T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = vals[static_cast<scl::Index>(k)] * scale;
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

/// @brief Max 行归一化 (除以行最大值)
template <typename T, bool IsCSR>
void normalize_max_rows(const MappedSparse<T, IsCSR>& mat,
                        T* data_out,
                        scl::Index* indices_out,
                        scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        // 找最大值
        T max_val = T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            max_val = std::max(max_val, std::abs(vals[static_cast<scl::Index>(k)]));
        }

        T scale = max_val > T(0) ? T(1) / max_val : T(0);
        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = vals[static_cast<scl::Index>(k)] * scale;
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

// =============================================================================
// SECTION 6: Element-wise Operations
// =============================================================================

/// @brief log1p 变换
template <typename T, bool IsCSR>
void log1p_transform(const MappedSparse<T, IsCSR>& mat,
                     T* data_out,
                     scl::Index* indices_out,
                     scl::Index* indptr_out) {
    scl::Index n = IsCSR ? mat.rows() : mat.cols();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = std::log1p(vals[static_cast<scl::Index>(k)]);
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

/// @brief 缩放变换 (乘以常数)
template <typename T, bool IsCSR>
void scale_transform(const MappedSparse<T, IsCSR>& mat,
                     T scale_factor,
                     T* data_out,
                     scl::Index* indices_out,
                     scl::Index* indptr_out) {
    scl::Index n = IsCSR ? mat.rows() : mat.cols();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = vals[static_cast<scl::Index>(k)] * scale_factor;
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

/// @brief 按行缩放 (每行乘以不同因子)
template <typename T, bool IsCSR>
void scale_rows(const MappedSparse<T, IsCSR>& mat,
                const T* row_factors,
                T* data_out,
                scl::Index* indices_out,
                scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        T factor = row_factors[i];
        for (scl::Size k = 0; k < vals.size(); ++k) {
            data_out[offset] = vals[static_cast<scl::Index>(k)] * factor;
            indices_out[offset] = inds[static_cast<scl::Index>(k)];
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

/// @brief 按列缩放 (CSR 格式)
template <typename T>
void scale_cols_csr(const MappedSparse<T, true>& mat,
                    const T* col_factors,
                    T* data_out,
                    scl::Index* indices_out,
                    scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        auto vals = mat.row_values(i);
        auto inds = mat.row_indices(i);

        for (scl::Size k = 0; k < vals.size(); ++k) {
            scl::Index j = inds[static_cast<scl::Index>(k)];
            data_out[offset] = vals[static_cast<scl::Index>(k)] * col_factors[j];
            indices_out[offset] = j;
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

// =============================================================================
// SECTION 7: Sparse-Dense Operations
// =============================================================================

/// @brief 稀疏矩阵与稠密向量相乘 (SpMV): y = A * x
template <typename T, bool IsCSR>
void spmv(const MappedSparse<T, IsCSR>& mat,
          const T* x,
          T* y) {
    if constexpr (IsCSR) {
        scl::Index rows = mat.rows();

        for (scl::Index i = 0; i < rows; ++i) {
            auto vals = mat.row_values(i);
            auto inds = mat.row_indices(i);

            T sum = T(0);
            for (scl::Size k = 0; k < vals.size(); ++k) {
                sum += vals[static_cast<scl::Index>(k)] * x[inds[static_cast<scl::Index>(k)]];
            }
            y[i] = sum;
        }
    } else {
        // CSC: y = A * x
        scl::Index rows = mat.rows();
        scl::Index cols = mat.cols();

        std::memset(y, 0, static_cast<std::size_t>(rows) * sizeof(T));

        for (scl::Index j = 0; j < cols; ++j) {
            auto vals = mat.col_values(j);
            auto inds = mat.col_indices(j);

            T xj = x[j];
            for (scl::Size k = 0; k < vals.size(); ++k) {
                y[inds[static_cast<scl::Index>(k)]] += vals[static_cast<scl::Index>(k)] * xj;
            }
        }
    }
}

/// @brief 带缓存的 SpMV
template <typename T, bool IsCSR>
void spmv_cached(const MappedSparse<T, IsCSR>& mat,
                 const T* x,
                 T* y,
                 HeuristicCache<T>& cache) {
    if constexpr (IsCSR) {
        scl::Index rows = mat.rows();

        for (scl::Index i = 0; i < rows; ++i) {
            // 尝试从缓存获取
            auto* entry = cache.get(i);

            T sum = T(0);
            if (entry) {
                // 缓存命中
                for (std::size_t k = 0; k < entry->values.size(); ++k) {
                    sum += entry->values[k] * x[entry->indices[k]];
                }
            } else {
                // 缓存未命中
                auto vals = mat.row_values(i);
                auto inds = mat.row_indices(i);

                for (scl::Size k = 0; k < vals.size(); ++k) {
                    sum += vals[static_cast<scl::Index>(k)] * x[inds[static_cast<scl::Index>(k)]];
                }

                // 根据访问模式决定是否缓存
                if (cache.pattern() == AccessPattern::Repeated ||
                    cache.pattern() == AccessPattern::Random) {
                    cache.insert(i, mat);
                }
            }

            y[i] = sum;
        }
    } else {
        // CSC 简化处理
        spmv<T, IsCSR>(mat, x, y);
    }
}

// =============================================================================
// SECTION 8: Dot Products
// =============================================================================

/// @brief 两行的点积
template <typename T, bool IsCSR>
T row_dot(const MappedSparse<T, IsCSR>& mat, scl::Index i, scl::Index j) {
    scl::Array<T> vals_i, vals_j;
    scl::Array<scl::Index> inds_i, inds_j;

    if constexpr (IsCSR) {
        vals_i = mat.row_values(i);
        inds_i = mat.row_indices(i);
        vals_j = mat.row_values(j);
        inds_j = mat.row_indices(j);
    } else {
        vals_i = mat.col_values(i);
        inds_i = mat.col_indices(i);
        vals_j = mat.col_values(j);
        inds_j = mat.col_indices(j);
    }

    // 归并求点积
    T dot = T(0);
    scl::Size pi = 0, pj = 0;

    while (pi < inds_i.size() && pj < inds_j.size()) {
        scl::Index ci = inds_i[static_cast<scl::Index>(pi)];
        scl::Index cj = inds_j[static_cast<scl::Index>(pj)];

        if (ci == cj) {
            dot += vals_i[static_cast<scl::Index>(pi)] * vals_j[static_cast<scl::Index>(pj)];
            ++pi;
            ++pj;
        } else if (ci < cj) {
            ++pi;
        } else {
            ++pj;
        }
    }

    return dot;
}

/// @brief 计算行的 L2 范数
template <typename T, bool IsCSR>
T row_norm(const MappedSparse<T, IsCSR>& mat, scl::Index i) {
    scl::Array<T> vals;
    if constexpr (IsCSR) {
        vals = mat.row_values(i);
    } else {
        vals = mat.col_values(i);
    }

    T sum_sq = T(0);
    for (scl::Size k = 0; k < vals.size(); ++k) {
        T v = vals[static_cast<scl::Index>(k)];
        sum_sq += v * v;
    }

    return std::sqrt(sum_sq);
}

// =============================================================================
// SECTION 9: Filtering
// =============================================================================

/// @brief 过滤小于阈值的元素
template <typename T, bool IsCSR>
scl::Index filter_threshold(const MappedSparse<T, IsCSR>& mat,
                             T threshold,
                             T* data_out,
                             scl::Index* indices_out,
                             scl::Index* indptr_out) {
    scl::Index n = IsCSR ? mat.rows() : mat.cols();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < n; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        for (scl::Size k = 0; k < vals.size(); ++k) {
            T v = vals[static_cast<scl::Index>(k)];
            if (std::abs(v) >= threshold) {
                data_out[offset] = v;
                indices_out[offset] = inds[static_cast<scl::Index>(k)];
                ++offset;
            }
        }

        indptr_out[i + 1] = offset;
    }

    return offset;  // 返回新 nnz
}

/// @brief 只保留每行 top-k 元素
template <typename T, bool IsCSR>
void top_k_per_row(const MappedSparse<T, IsCSR>& mat,
                   scl::Index k,
                   T* data_out,
                   scl::Index* indices_out,
                   scl::Index* indptr_out) {
    scl::Index rows = mat.rows();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = mat.row_values(i);
            inds = mat.row_indices(i);
        } else {
            vals = mat.col_values(i);
            inds = mat.col_indices(i);
        }

        // 收集 (value, index) 对
        std::vector<std::pair<T, scl::Index>> pairs;
        pairs.reserve(vals.size());
        for (scl::Size j = 0; j < vals.size(); ++j) {
            pairs.emplace_back(vals[static_cast<scl::Index>(j)],
                               inds[static_cast<scl::Index>(j)]);
        }

        // 部分排序取 top-k
        scl::Index take = std::min(k, static_cast<scl::Index>(pairs.size()));
        std::partial_sort(pairs.begin(), pairs.begin() + take, pairs.end(),
                          [](const auto& a, const auto& b) {
                              return std::abs(a.first) > std::abs(b.first);
                          });

        // 按列索引排序输出
        std::sort(pairs.begin(), pairs.begin() + take,
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        for (scl::Index j = 0; j < take; ++j) {
            data_out[offset] = pairs[static_cast<std::size_t>(j)].first;
            indices_out[offset] = pairs[static_cast<std::size_t>(j)].second;
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

} // namespace scl::mmap
