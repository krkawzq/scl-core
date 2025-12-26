#pragma once

#include "scl/core/type.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/array.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#include <functional>

namespace scl::mmap {

// =============================================================================
// Load Function Types
// =============================================================================

/// @brief 数据加载回调 (page_idx, dest_buffer) -> void
using DataLoadFunc = std::function<void(std::size_t, std::byte*)>;

/// @brief 从文件偏移加载数据的工厂函数
template <typename T>
inline DataLoadFunc make_file_loader(int fd, std::size_t base_offset) {
    return [fd, base_offset](std::size_t page_idx, std::byte* dest) {
        std::size_t offset = base_offset + page_idx * SCL_MMAP_PAGE_SIZE;
        // pread(fd, dest, SCL_MMAP_PAGE_SIZE, offset);
        // 实际实现需要包含 <unistd.h> 并调用 pread
        (void)fd; (void)offset; (void)dest;
    };
}

// =============================================================================
// MappedSparse: Memory-Mapped Sparse Matrix (CSR/CSC)
// =============================================================================

/// @brief 基于虚拟内存映射的稀疏矩阵
///
/// 数据存储在磁盘上，按需加载到内存。适用于超大规模数据集。
///
/// 内存布局 (逻辑上):
/// - data[nnz]: 非零值
/// - indices[nnz]: 列索引 (CSR) 或行索引 (CSC)
/// - indptr[primary+1]: 累积偏移
///
/// 与 CustomSparse 的区别:
/// - CustomSparse: 所有数据在 RAM 中
/// - MappedSparse: 数据在磁盘上，通过页面调度按需加载
template <typename T, bool IsCSR>
class MappedSparse {
public:
    using ValueType = T;
    using Tag = scl::TagSparse<IsCSR>;

    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

private:
    // 虚拟数组 (懒加载)
    std::unique_ptr<VirtualArray<T>> data_;
    std::unique_ptr<VirtualArray<scl::Index>> indices_;
    std::unique_ptr<VirtualArray<scl::Index>> indptr_;

    // 维度信息 (小，直接存内存)
    scl::Index rows_;
    scl::Index cols_;
    scl::Index nnz_;

    // 每行/列缓存 (用于返回 Array 视图)
    mutable std::vector<T> value_cache_;
    mutable std::vector<scl::Index> index_cache_;
    mutable scl::Index cached_primary_idx_ = -1;

public:
    // -------------------------------------------------------------------------
    // 构造函数
    // -------------------------------------------------------------------------

    /// @brief 构造映射稀疏矩阵
    /// @param rows 行数
    /// @param cols 列数
    /// @param nnz 非零元素数
    /// @param data_loader data 数组的加载函数
    /// @param indices_loader indices 数组的加载函数
    /// @param indptr_loader indptr 数组的加载函数
    /// @param config 内存池配置
    MappedSparse(scl::Index rows, scl::Index cols, scl::Index nnz,
                 LoadFunc data_loader,
                 LoadFunc indices_loader,
                 LoadFunc indptr_loader,
                 const MmapConfig& config = MmapConfig{})
        : rows_(rows), cols_(cols), nnz_(nnz)
    {
        // 初始化三个虚拟数组
        data_ = std::make_unique<VirtualArray<T>>(
            static_cast<std::size_t>(nnz), std::move(data_loader), config);

        indices_ = std::make_unique<VirtualArray<scl::Index>>(
            static_cast<std::size_t>(nnz), std::move(indices_loader), config);

        scl::Index primary_dim = IsCSR ? rows : cols;
        indptr_ = std::make_unique<VirtualArray<scl::Index>>(
            static_cast<std::size_t>(primary_dim + 1), std::move(indptr_loader), config);
    }

    ~MappedSparse() = default;

    // 禁止拷贝
    MappedSparse(const MappedSparse&) = delete;
    MappedSparse& operator=(const MappedSparse&) = delete;

    // 允许移动
    MappedSparse(MappedSparse&&) noexcept = default;
    MappedSparse& operator=(MappedSparse&&) noexcept = default;

    // -------------------------------------------------------------------------
    // 基本属性
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Index rows() const { return rows_; }
    SCL_NODISCARD scl::Index cols() const { return cols_; }
    SCL_NODISCARD scl::Index nnz() const { return nnz_; }

    // -------------------------------------------------------------------------
    // CSR 接口
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Array<T> row_values(scl::Index i) const requires (IsCSR) {
        ensure_cached(i);
        return scl::Array<T>(value_cache_.data(), value_cache_.size());
    }

    SCL_NODISCARD scl::Array<scl::Index> row_indices(scl::Index i) const requires (IsCSR) {
        ensure_cached(i);
        return scl::Array<scl::Index>(index_cache_.data(), index_cache_.size());
    }

    SCL_NODISCARD scl::Index row_length(scl::Index i) const requires (IsCSR) {
        scl::Index start = (*indptr_)[static_cast<std::size_t>(i)];
        scl::Index end = (*indptr_)[static_cast<std::size_t>(i + 1)];
        return end - start;
    }

    // -------------------------------------------------------------------------
    // CSC 接口
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Array<T> col_values(scl::Index j) const requires (!IsCSR) {
        ensure_cached(j);
        return scl::Array<T>(value_cache_.data(), value_cache_.size());
    }

    SCL_NODISCARD scl::Array<scl::Index> col_indices(scl::Index j) const requires (!IsCSR) {
        ensure_cached(j);
        return scl::Array<scl::Index>(index_cache_.data(), index_cache_.size());
    }

    SCL_NODISCARD scl::Index col_length(scl::Index j) const requires (!IsCSR) {
        scl::Index start = (*indptr_)[static_cast<std::size_t>(j)];
        scl::Index end = (*indptr_)[static_cast<std::size_t>(j + 1)];
        return end - start;
    }

    // -------------------------------------------------------------------------
    // 批量访问 (高性能路径)
    // -------------------------------------------------------------------------

    /// @brief 批量读取多行/列的数据到缓冲区
    void read_primary_range(scl::Index start, scl::Index count,
                            T* values_out, scl::Index* indices_out,
                            scl::Index* lengths_out) const {
        for (scl::Index i = 0; i < count; ++i) {
            scl::Index idx = start + i;
            scl::Index s = (*indptr_)[static_cast<std::size_t>(idx)];
            scl::Index e = (*indptr_)[static_cast<std::size_t>(idx + 1)];
            scl::Index len = e - s;

            lengths_out[i] = len;

            if (len > 0) {
                data_->read_range(static_cast<std::size_t>(s),
                                  static_cast<std::size_t>(len),
                                  values_out);
                indices_->read_range(static_cast<std::size_t>(s),
                                     static_cast<std::size_t>(len),
                                     indices_out);
                values_out += len;
                indices_out += len;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 预取提示
    // -------------------------------------------------------------------------

    void prefetch_primary(scl::Index start, scl::Index count) {
        scl::Index s = (*indptr_)[static_cast<std::size_t>(start)];
        scl::Index e = (*indptr_)[static_cast<std::size_t>(start + count)];

        data_->prefetch(static_cast<std::size_t>(s), static_cast<std::size_t>(e - s));
        indices_->prefetch(static_cast<std::size_t>(s), static_cast<std::size_t>(e - s));
    }

private:
    void ensure_cached(scl::Index primary_idx) const {
        if (cached_primary_idx_ == primary_idx) return;

        scl::Index start = (*indptr_)[static_cast<std::size_t>(primary_idx)];
        scl::Index end = (*indptr_)[static_cast<std::size_t>(primary_idx + 1)];
        scl::Index len = end - start;

        value_cache_.resize(static_cast<std::size_t>(len));
        index_cache_.resize(static_cast<std::size_t>(len));

        if (len > 0) {
            data_->read_range(static_cast<std::size_t>(start),
                              static_cast<std::size_t>(len),
                              value_cache_.data());
            indices_->read_range(static_cast<std::size_t>(start),
                                 static_cast<std::size_t>(len),
                                 index_cache_.data());
        }

        cached_primary_idx_ = primary_idx;
    }
};

// 类型别名
template <typename T>
using MappedCSR = MappedSparse<T, true>;

template <typename T>
using MappedCSC = MappedSparse<T, false>;

using MappedCSRReal = MappedCSR<scl::Real>;
using MappedCSCReal = MappedCSC<scl::Real>;

// =============================================================================
// MappedDense: Memory-Mapped Dense Matrix
// =============================================================================

/// @brief 基于虚拟内存映射的稠密矩阵
///
/// 数据以行优先 (Row-Major) 顺序存储在磁盘上，按需加载。
template <typename T>
class MappedDense {
public:
    using ValueType = T;
    using Tag = scl::TagDense;

private:
    std::unique_ptr<VirtualArray<T>> data_;
    scl::Index rows_;
    scl::Index cols_;

    // 行缓存
    mutable std::vector<T> row_cache_;
    mutable scl::Index cached_row_ = -1;

public:
    // -------------------------------------------------------------------------
    // 构造函数
    // -------------------------------------------------------------------------

    /// @brief 构造映射稠密矩阵
    /// @param rows 行数
    /// @param cols 列数
    /// @param loader 数据加载函数
    /// @param config 内存池配置
    MappedDense(scl::Index rows, scl::Index cols,
                LoadFunc loader,
                const MmapConfig& config = MmapConfig{})
        : rows_(rows), cols_(cols)
    {
        std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        data_ = std::make_unique<VirtualArray<T>>(total, std::move(loader), config);
    }

    ~MappedDense() = default;

    MappedDense(const MappedDense&) = delete;
    MappedDense& operator=(const MappedDense&) = delete;
    MappedDense(MappedDense&&) noexcept = default;
    MappedDense& operator=(MappedDense&&) noexcept = default;

    // -------------------------------------------------------------------------
    // 基本属性
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Index rows() const { return rows_; }
    SCL_NODISCARD scl::Index cols() const { return cols_; }

    // -------------------------------------------------------------------------
    // 元素访问
    // -------------------------------------------------------------------------

    /// @brief 访问单个元素 (可能触发页面加载)
    SCL_NODISCARD T operator()(scl::Index i, scl::Index j) const {
        std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_)
                        + static_cast<std::size_t>(j);
        return (*data_)[idx];
    }

    /// @brief 获取数据指针 (通过缓存行)
    SCL_NODISCARD const T* data() const {
        // 返回第一行的缓存 (用于兼容某些接口)
        ensure_row_cached(0);
        return row_cache_.data();
    }

    // -------------------------------------------------------------------------
    // 行访问 (高性能路径)
    // -------------------------------------------------------------------------

    /// @brief 获取某一行的数据
    SCL_NODISCARD scl::Array<const T> row(scl::Index i) const {
        ensure_row_cached(i);
        return scl::Array<const T>(row_cache_.data(), static_cast<scl::Size>(cols_));
    }

    /// @brief 批量读取多行到缓冲区
    void read_rows(scl::Index start, scl::Index count, T* out) const {
        std::size_t offset = static_cast<std::size_t>(start) * static_cast<std::size_t>(cols_);
        std::size_t num = static_cast<std::size_t>(count) * static_cast<std::size_t>(cols_);
        data_->read_range(offset, num, out);
    }

    // -------------------------------------------------------------------------
    // 预取提示
    // -------------------------------------------------------------------------

    void prefetch_rows(scl::Index start, scl::Index count) {
        std::size_t offset = static_cast<std::size_t>(start) * static_cast<std::size_t>(cols_);
        std::size_t num = static_cast<std::size_t>(count) * static_cast<std::size_t>(cols_);
        data_->prefetch(offset, num);
    }

private:
    void ensure_row_cached(scl::Index row_idx) const {
        if (cached_row_ == row_idx) return;

        row_cache_.resize(static_cast<std::size_t>(cols_));
        std::size_t offset = static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(cols_);
        data_->read_range(offset, static_cast<std::size_t>(cols_), row_cache_.data());

        cached_row_ = row_idx;
    }
};

using MappedDenseReal = MappedDense<scl::Real>;

// =============================================================================
// Concept 验证
// =============================================================================

// 注意: MappedSparse 返回的 Array 是临时缓存的视图，
// 使用时需确保在下次访问前消费完数据

} // namespace scl::mmap
