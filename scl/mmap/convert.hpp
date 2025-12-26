#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/array.hpp"
#include "scl/mmap/matrix.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace scl::mmap {

// =============================================================================
// SECTION 1: Mask-Based Selection
// =============================================================================

/// @brief 从 uint8 掩码计算选中的索引数量
inline scl::Index mask_count(const uint8_t* mask, scl::Index length) {
    scl::Index count = 0;
    for (scl::Index i = 0; i < length; ++i) {
        if (mask[i]) ++count;
    }
    return count;
}

/// @brief 从掩码构建索引映射表
/// @param mask 输入掩码 (0/非0)
/// @param length 掩码长度
/// @param indices_out 输出选中的原始索引
/// @return 选中的数量
inline scl::Index mask_to_indices(const uint8_t* mask, scl::Index length,
                                   scl::Index* indices_out) {
    scl::Index count = 0;
    for (scl::Index i = 0; i < length; ++i) {
        if (mask[i]) {
            indices_out[count++] = i;
        }
    }
    return count;
}

/// @brief 构建反向映射表 (原始索引 -> 新索引，未选中为 -1)
inline void build_remap(const uint8_t* mask, scl::Index length,
                        scl::Index* remap_out) {
    scl::Index new_idx = 0;
    for (scl::Index i = 0; i < length; ++i) {
        if (mask[i]) {
            remap_out[i] = new_idx++;
        } else {
            remap_out[i] = -1;
        }
    }
}

// =============================================================================
// SECTION 2: MappedVirtualSparse - Zero-Cost Masked View
// =============================================================================

/// @brief 零成本掩码视图的稀疏矩阵
///
/// 设计特点：
/// - 使用 uint8 掩码选择行/列
/// - 不复制底层数据
/// - O(1) 创建视图
/// - 支持链式视图
template <typename T, bool IsCSR>
class MappedVirtualSparse {
public:
    using ValueType = T;
    using Tag = scl::TagSparse<IsCSR>;
    using BaseMatrix = MappedSparse<T, IsCSR>;

    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

private:
    // 底层矩阵 (共享所有权)
    std::shared_ptr<BaseMatrix> base_;

    // 索引映射 (视图索引 -> 底层索引)
    std::vector<scl::Index> row_map_;
    std::vector<scl::Index> col_map_;

    // 反向映射 (底层列索引 -> 视图列索引, -1 表示未选中)
    std::vector<scl::Index> col_remap_;

    // 视图维度
    scl::Index rows_;
    scl::Index cols_;

    // 是否需要列过滤
    bool filter_cols_;

    // 缓存
    mutable std::vector<T> value_cache_;
    mutable std::vector<scl::Index> index_cache_;
    mutable scl::Index cached_idx_ = -1;

public:
    // -------------------------------------------------------------------------
    // 构造函数
    // -------------------------------------------------------------------------

    /// @brief 从 MappedSparse 创建全视图
    explicit MappedVirtualSparse(std::shared_ptr<BaseMatrix> base)
        : base_(std::move(base))
        , rows_(base_->rows())
        , cols_(base_->cols())
        , filter_cols_(false)
    {
        row_map_.resize(static_cast<std::size_t>(rows_));
        std::iota(row_map_.begin(), row_map_.end(), 0);
    }

    /// @brief 从掩码创建视图
    /// @param row_mask 行掩码 (nullptr = 全选)
    /// @param col_mask 列掩码 (nullptr = 全选)
    MappedVirtualSparse(std::shared_ptr<BaseMatrix> base,
                        const uint8_t* row_mask,
                        const uint8_t* col_mask)
        : base_(std::move(base))
        , filter_cols_(col_mask != nullptr)
    {
        // 构建行映射
        if (row_mask) {
            rows_ = mask_count(row_mask, base_->rows());
            row_map_.resize(static_cast<std::size_t>(rows_));
            mask_to_indices(row_mask, base_->rows(), row_map_.data());
        } else {
            rows_ = base_->rows();
            row_map_.resize(static_cast<std::size_t>(rows_));
            std::iota(row_map_.begin(), row_map_.end(), 0);
        }

        // 构建列映射和反向映射
        if (col_mask) {
            cols_ = mask_count(col_mask, base_->cols());
            col_map_.resize(static_cast<std::size_t>(cols_));
            mask_to_indices(col_mask, base_->cols(), col_map_.data());

            col_remap_.resize(static_cast<std::size_t>(base_->cols()), -1);
            build_remap(col_mask, base_->cols(), col_remap_.data());
        } else {
            cols_ = base_->cols();
        }
    }

    /// @brief 从索引数组创建视图
    MappedVirtualSparse(std::shared_ptr<BaseMatrix> base,
                        const scl::Index* row_indices, scl::Index num_rows,
                        const scl::Index* col_indices, scl::Index num_cols)
        : base_(std::move(base))
        , rows_(num_rows)
        , cols_(num_cols > 0 ? num_cols : base_->cols())
        , filter_cols_(col_indices != nullptr && num_cols > 0)
    {
        if (row_indices && num_rows > 0) {
            row_map_.assign(row_indices, row_indices + num_rows);
        } else {
            row_map_.resize(static_cast<std::size_t>(base_->rows()));
            std::iota(row_map_.begin(), row_map_.end(), 0);
            rows_ = base_->rows();
        }

        if (filter_cols_) {
            col_map_.assign(col_indices, col_indices + num_cols);
            col_remap_.resize(static_cast<std::size_t>(base_->cols()), -1);
            for (scl::Index i = 0; i < num_cols; ++i) {
                col_remap_[static_cast<std::size_t>(col_indices[i])] = i;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 基本属性
    // -------------------------------------------------------------------------

    SCL_NODISCARD scl::Index rows() const { return rows_; }
    SCL_NODISCARD scl::Index cols() const { return cols_; }

    SCL_NODISCARD scl::Index nnz() const {
        scl::Index total = 0;
        for (scl::Index i = 0; i < rows_; ++i) {
            total += primary_length_impl(i);
        }
        return total;
    }

    /// @brief 获取底层矩阵
    std::shared_ptr<BaseMatrix> base() const { return base_; }

    /// @brief 获取行映射
    const std::vector<scl::Index>& row_map() const { return row_map_; }

    /// @brief 获取列映射
    const std::vector<scl::Index>& col_map() const { return col_map_; }

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
        return primary_length_impl(i);
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
        return primary_length_impl(j);
    }

    // -------------------------------------------------------------------------
    // 链式视图
    // -------------------------------------------------------------------------

    /// @brief 基于掩码创建子视图
    MappedVirtualSparse subview(const uint8_t* row_mask,
                                 const uint8_t* col_mask) const {
        std::vector<scl::Index> new_row_map;
        std::vector<scl::Index> new_col_map;

        // 行子选择
        if (row_mask) {
            scl::Index count = mask_count(row_mask, rows_);
            new_row_map.reserve(static_cast<std::size_t>(count));
            for (scl::Index i = 0; i < rows_; ++i) {
                if (row_mask[i]) {
                    new_row_map.push_back(row_map_[static_cast<std::size_t>(i)]);
                }
            }
        } else {
            new_row_map = row_map_;
        }

        // 列子选择
        scl::Index new_cols = cols_;
        if (col_mask) {
            new_cols = mask_count(col_mask, cols_);
            new_col_map.reserve(static_cast<std::size_t>(new_cols));

            if (filter_cols_) {
                for (scl::Index j = 0; j < cols_; ++j) {
                    if (col_mask[j]) {
                        new_col_map.push_back(col_map_[static_cast<std::size_t>(j)]);
                    }
                }
            } else {
                for (scl::Index j = 0; j < cols_; ++j) {
                    if (col_mask[j]) {
                        new_col_map.push_back(j);
                    }
                }
            }
        } else if (filter_cols_) {
            new_col_map = col_map_;
        }

        return MappedVirtualSparse(
            base_,
            new_row_map.data(), static_cast<scl::Index>(new_row_map.size()),
            new_col_map.empty() ? nullptr : new_col_map.data(),
            static_cast<scl::Index>(new_col_map.size())
        );
    }

private:
    scl::Index primary_length_impl(scl::Index view_idx) const {
        scl::Index base_idx = row_map_[static_cast<std::size_t>(view_idx)];

        if (!filter_cols_) {
            if constexpr (IsCSR) {
                return base_->row_length(base_idx);
            } else {
                return base_->col_length(base_idx);
            }
        }

        // 需要过滤，计算匹配数量
        scl::Array<scl::Index> base_indices;
        if constexpr (IsCSR) {
            base_indices = base_->row_indices(base_idx);
        } else {
            base_indices = base_->col_indices(base_idx);
        }

        scl::Index count = 0;
        for (scl::Size k = 0; k < base_indices.size(); ++k) {
            scl::Index col = base_indices[static_cast<scl::Index>(k)];
            if (col_remap_[static_cast<std::size_t>(col)] >= 0) {
                ++count;
            }
        }
        return count;
    }

    void ensure_cached(scl::Index view_idx) const {
        if (cached_idx_ == view_idx) return;

        scl::Index base_idx = row_map_[static_cast<std::size_t>(view_idx)];

        scl::Array<T> base_vals;
        scl::Array<scl::Index> base_indices;

        if constexpr (IsCSR) {
            base_vals = base_->row_values(base_idx);
            base_indices = base_->row_indices(base_idx);
        } else {
            base_vals = base_->col_values(base_idx);
            base_indices = base_->col_indices(base_idx);
        }

        value_cache_.clear();
        index_cache_.clear();

        if (!filter_cols_) {
            value_cache_.assign(base_vals.begin(), base_vals.end());
            index_cache_.assign(base_indices.begin(), base_indices.end());
        } else {
            for (scl::Size k = 0; k < base_vals.size(); ++k) {
                scl::Index base_col = base_indices[static_cast<scl::Index>(k)];
                scl::Index new_col = col_remap_[static_cast<std::size_t>(base_col)];
                if (new_col >= 0) {
                    value_cache_.push_back(base_vals[static_cast<scl::Index>(k)]);
                    index_cache_.push_back(new_col);
                }
            }
        }

        cached_idx_ = view_idx;
    }
};

template <typename T>
using MappedVirtualCSR = MappedVirtualSparse<T, true>;

template <typename T>
using MappedVirtualCSC = MappedVirtualSparse<T, false>;

// =============================================================================
// SECTION 3: Load Operations
// =============================================================================

/// @brief 加载 MappedSparse 全部数据到连续数组
template <typename T, bool IsCSR>
void load_full(const MappedSparse<T, IsCSR>& src,
               T* data_out,
               scl::Index* indices_out,
               scl::Index* indptr_out) {
    scl::Index primary_dim = IsCSR ? src.rows() : src.cols();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < primary_dim; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = src.row_values(i);
            inds = src.row_indices(i);
        } else {
            vals = src.col_values(i);
            inds = src.col_indices(i);
        }

        std::copy(vals.begin(), vals.end(), data_out + offset);
        std::copy(inds.begin(), inds.end(), indices_out + offset);
        offset += static_cast<scl::Index>(vals.size());
        indptr_out[i + 1] = offset;
    }
}

/// @brief 加载 MappedDense 全部数据
template <typename T>
void load_full(const MappedDense<T>& src, T* data_out) {
    src.read_rows(0, src.rows(), data_out);
}

/// @brief 基于掩码加载稀疏矩阵子集
/// @param row_mask 行掩码 (nullptr = 全选)
/// @param col_mask 列掩码 (nullptr = 全选)
/// @param out_rows 输出行数
/// @param out_cols 输出列数
/// @param out_nnz 输出非零数
template <typename T, bool IsCSR>
void load_masked(const MappedSparse<T, IsCSR>& src,
                 const uint8_t* row_mask,
                 const uint8_t* col_mask,
                 T* data_out,
                 scl::Index* indices_out,
                 scl::Index* indptr_out,
                 scl::Index* out_rows,
                 scl::Index* out_cols,
                 scl::Index* out_nnz) {
    // 构建映射
    std::vector<scl::Index> row_indices;
    std::vector<scl::Index> col_remap;

    scl::Index new_rows, new_cols;

    if (row_mask) {
        new_rows = mask_count(row_mask, src.rows());
        row_indices.resize(static_cast<std::size_t>(new_rows));
        mask_to_indices(row_mask, src.rows(), row_indices.data());
    } else {
        new_rows = src.rows();
        row_indices.resize(static_cast<std::size_t>(new_rows));
        std::iota(row_indices.begin(), row_indices.end(), 0);
    }

    if (col_mask) {
        new_cols = mask_count(col_mask, src.cols());
        col_remap.resize(static_cast<std::size_t>(src.cols()), -1);
        build_remap(col_mask, src.cols(), col_remap.data());
    } else {
        new_cols = src.cols();
    }

    // 提取数据
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < new_rows; ++i) {
        scl::Index base_row = row_indices[static_cast<std::size_t>(i)];

        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = src.row_values(base_row);
            inds = src.row_indices(base_row);
        } else {
            vals = src.col_values(base_row);
            inds = src.col_indices(base_row);
        }

        if (col_mask) {
            // 需要列过滤
            for (scl::Size k = 0; k < vals.size(); ++k) {
                scl::Index col = inds[static_cast<scl::Index>(k)];
                scl::Index new_col = col_remap[static_cast<std::size_t>(col)];
                if (new_col >= 0) {
                    data_out[offset] = vals[static_cast<scl::Index>(k)];
                    indices_out[offset] = new_col;
                    ++offset;
                }
            }
        } else {
            std::copy(vals.begin(), vals.end(), data_out + offset);
            std::copy(inds.begin(), inds.end(), indices_out + offset);
            offset += static_cast<scl::Index>(vals.size());
        }

        indptr_out[i + 1] = offset;
    }

    *out_rows = new_rows;
    *out_cols = new_cols;
    *out_nnz = offset;
}

/// @brief 基于掩码加载稠密矩阵子集
template <typename T>
void load_masked(const MappedDense<T>& src,
                 const uint8_t* row_mask,
                 const uint8_t* col_mask,
                 T* data_out,
                 scl::Index* out_rows,
                 scl::Index* out_cols) {
    scl::Index new_rows = row_mask ? mask_count(row_mask, src.rows()) : src.rows();
    scl::Index new_cols = col_mask ? mask_count(col_mask, src.cols()) : src.cols();

    std::vector<scl::Index> row_indices(static_cast<std::size_t>(new_rows));
    std::vector<scl::Index> col_indices(static_cast<std::size_t>(new_cols));

    if (row_mask) {
        mask_to_indices(row_mask, src.rows(), row_indices.data());
    } else {
        std::iota(row_indices.begin(), row_indices.end(), 0);
    }

    if (col_mask) {
        mask_to_indices(col_mask, src.cols(), col_indices.data());
    } else {
        std::iota(col_indices.begin(), col_indices.end(), 0);
    }

    T* dest = data_out;
    for (scl::Index i = 0; i < new_rows; ++i) {
        auto row_data = src.row(row_indices[static_cast<std::size_t>(i)]);
        if (col_mask) {
            for (scl::Index j = 0; j < new_cols; ++j) {
                *dest++ = row_data[col_indices[static_cast<std::size_t>(j)]];
            }
        } else {
            std::copy(row_data.begin(), row_data.end(), dest);
            dest += new_cols;
        }
    }

    *out_rows = new_rows;
    *out_cols = new_cols;
}

/// @brief 基于索引数组加载 (fancy indexing)
template <typename T, bool IsCSR>
void load_indexed(const MappedSparse<T, IsCSR>& src,
                  const scl::Index* row_indices, scl::Index num_rows,
                  const scl::Index* col_indices, scl::Index num_cols,
                  T* data_out,
                  scl::Index* indices_out,
                  scl::Index* indptr_out,
                  scl::Index* out_nnz) {
    // 构建列反向映射
    std::vector<scl::Index> col_remap;
    bool filter_cols = (col_indices != nullptr && num_cols > 0);

    if (filter_cols) {
        col_remap.resize(static_cast<std::size_t>(src.cols()), -1);
        for (scl::Index i = 0; i < num_cols; ++i) {
            col_remap[static_cast<std::size_t>(col_indices[i])] = i;
        }
    }

    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < num_rows; ++i) {
        scl::Index base_row = row_indices[i];

        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = src.row_values(base_row);
            inds = src.row_indices(base_row);
        } else {
            vals = src.col_values(base_row);
            inds = src.col_indices(base_row);
        }

        if (filter_cols) {
            for (scl::Size k = 0; k < vals.size(); ++k) {
                scl::Index col = inds[static_cast<scl::Index>(k)];
                scl::Index new_col = col_remap[static_cast<std::size_t>(col)];
                if (new_col >= 0) {
                    data_out[offset] = vals[static_cast<scl::Index>(k)];
                    indices_out[offset] = new_col;
                    ++offset;
                }
            }
        } else {
            std::copy(vals.begin(), vals.end(), data_out + offset);
            std::copy(inds.begin(), inds.end(), indices_out + offset);
            offset += static_cast<scl::Index>(vals.size());
        }

        indptr_out[i + 1] = offset;
    }

    *out_nnz = offset;
}

// =============================================================================
// SECTION 4: Reorder Operations
// =============================================================================

/// @brief 按顺序向量重排行
template <typename T, bool IsCSR>
void reorder_rows(const MappedSparse<T, IsCSR>& src,
                  const scl::Index* order, scl::Index count,
                  T* data_out,
                  scl::Index* indices_out,
                  scl::Index* indptr_out) {
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < count; ++i) {
        scl::Index src_row = order[i];

        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = src.row_values(src_row);
            inds = src.row_indices(src_row);
        } else {
            vals = src.col_values(src_row);
            inds = src.col_indices(src_row);
        }

        std::copy(vals.begin(), vals.end(), data_out + offset);
        std::copy(inds.begin(), inds.end(), indices_out + offset);
        offset += static_cast<scl::Index>(vals.size());
        indptr_out[i + 1] = offset;
    }
}

/// @brief 按顺序向量重排列 (需要重新映射索引)
template <typename T, bool IsCSR>
void reorder_cols(const MappedSparse<T, IsCSR>& src,
                  const scl::Index* col_order, scl::Index num_cols,
                  T* data_out,
                  scl::Index* indices_out,
                  scl::Index* indptr_out) {
    // 构建反向映射: old_col -> new_col
    std::vector<scl::Index> col_remap(static_cast<std::size_t>(src.cols()), -1);
    for (scl::Index i = 0; i < num_cols; ++i) {
        col_remap[static_cast<std::size_t>(col_order[i])] = i;
    }

    scl::Index primary_dim = IsCSR ? src.rows() : src.cols();
    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < primary_dim; ++i) {
        scl::Array<T> vals;
        scl::Array<scl::Index> inds;

        if constexpr (IsCSR) {
            vals = src.row_values(i);
            inds = src.row_indices(i);
        } else {
            vals = src.col_values(i);
            inds = src.col_indices(i);
        }

        // 收集并排序
        std::vector<std::pair<scl::Index, T>> pairs;
        pairs.reserve(vals.size());

        for (scl::Size k = 0; k < vals.size(); ++k) {
            scl::Index old_col = inds[static_cast<scl::Index>(k)];
            scl::Index new_col = col_remap[static_cast<std::size_t>(old_col)];
            if (new_col >= 0) {
                pairs.emplace_back(new_col, vals[static_cast<scl::Index>(k)]);
            }
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& [col, val] : pairs) {
            indices_out[offset] = col;
            data_out[offset] = val;
            ++offset;
        }

        indptr_out[i + 1] = offset;
    }
}

// =============================================================================
// SECTION 5: Stack Operations
// =============================================================================

/// @brief 计算 vstack 后的维度信息
template <typename T, bool IsCSR>
struct StackInfo {
    scl::Index total_rows;
    scl::Index total_cols;
    scl::Index total_nnz;
};

/// @brief 垂直堆叠多个矩阵
template <typename T, bool IsCSR>
void vstack(const MappedSparse<T, IsCSR>* const* matrices, scl::Index count,
            T* data_out,
            scl::Index* indices_out,
            scl::Index* indptr_out) {
    indptr_out[0] = 0;
    scl::Index offset = 0;
    scl::Index row_offset = 0;

    for (scl::Index m = 0; m < count; ++m) {
        const auto* mat = matrices[m];
        scl::Index rows = mat->rows();

        for (scl::Index i = 0; i < rows; ++i) {
            scl::Array<T> vals;
            scl::Array<scl::Index> inds;

            if constexpr (IsCSR) {
                vals = mat->row_values(i);
                inds = mat->row_indices(i);
            } else {
                vals = mat->col_values(i);
                inds = mat->col_indices(i);
            }

            std::copy(vals.begin(), vals.end(), data_out + offset);
            std::copy(inds.begin(), inds.end(), indices_out + offset);
            offset += static_cast<scl::Index>(vals.size());
            indptr_out[row_offset + i + 1] = offset;
        }

        row_offset += rows;
    }
}

/// @brief 水平堆叠多个矩阵
template <typename T, bool IsCSR>
void hstack(const MappedSparse<T, IsCSR>* const* matrices, scl::Index count,
            T* data_out,
            scl::Index* indices_out,
            scl::Index* indptr_out) {
    if (count == 0) return;

    scl::Index rows = matrices[0]->rows();

    // 计算列偏移
    std::vector<scl::Index> col_offsets(static_cast<std::size_t>(count) + 1);
    col_offsets[0] = 0;
    for (scl::Index m = 0; m < count; ++m) {
        col_offsets[static_cast<std::size_t>(m) + 1] =
            col_offsets[static_cast<std::size_t>(m)] + matrices[m]->cols();
    }

    indptr_out[0] = 0;
    scl::Index offset = 0;

    for (scl::Index i = 0; i < rows; ++i) {
        for (scl::Index m = 0; m < count; ++m) {
            const auto* mat = matrices[m];
            scl::Index col_base = col_offsets[static_cast<std::size_t>(m)];

            scl::Array<T> vals;
            scl::Array<scl::Index> inds;

            if constexpr (IsCSR) {
                vals = mat->row_values(i);
                inds = mat->row_indices(i);
            } else {
                vals = mat->col_values(i);
                inds = mat->col_indices(i);
            }

            for (scl::Size k = 0; k < vals.size(); ++k) {
                data_out[offset] = vals[static_cast<scl::Index>(k)];
                indices_out[offset] = inds[static_cast<scl::Index>(k)] + col_base;
                ++offset;
            }
        }
        indptr_out[i + 1] = offset;
    }
}

// =============================================================================
// SECTION 6: Format Conversion
// =============================================================================

/// @brief CSR 转 CSC
template <typename T>
void csr_to_csc(const MappedSparse<T, true>& csr,
                T* csc_data,
                scl::Index* csc_indices,
                scl::Index* csc_indptr) {
    scl::Index rows = csr.rows();
    scl::Index cols = csr.cols();

    // 统计每列元素数
    std::vector<scl::Index> col_counts(static_cast<std::size_t>(cols), 0);
    for (scl::Index i = 0; i < rows; ++i) {
        auto inds = csr.row_indices(i);
        for (scl::Size k = 0; k < inds.size(); ++k) {
            ++col_counts[static_cast<std::size_t>(inds[static_cast<scl::Index>(k)])];
        }
    }

    // 构建 indptr
    csc_indptr[0] = 0;
    for (scl::Index j = 0; j < cols; ++j) {
        csc_indptr[j + 1] = csc_indptr[j] + col_counts[static_cast<std::size_t>(j)];
    }

    // 填充数据
    std::vector<scl::Index> col_pos(static_cast<std::size_t>(cols), 0);
    for (scl::Index i = 0; i < rows; ++i) {
        auto vals = csr.row_values(i);
        auto inds = csr.row_indices(i);

        for (scl::Size k = 0; k < vals.size(); ++k) {
            scl::Index col = inds[static_cast<scl::Index>(k)];
            scl::Index pos = csc_indptr[col] + col_pos[static_cast<std::size_t>(col)];

            csc_data[pos] = vals[static_cast<scl::Index>(k)];
            csc_indices[pos] = i;
            ++col_pos[static_cast<std::size_t>(col)];
        }
    }
}

/// @brief CSC 转 CSR
template <typename T>
void csc_to_csr(const MappedSparse<T, false>& csc,
                T* csr_data,
                scl::Index* csr_indices,
                scl::Index* csr_indptr) {
    scl::Index rows = csc.rows();
    scl::Index cols = csc.cols();

    std::vector<scl::Index> row_counts(static_cast<std::size_t>(rows), 0);
    for (scl::Index j = 0; j < cols; ++j) {
        auto inds = csc.col_indices(j);
        for (scl::Size k = 0; k < inds.size(); ++k) {
            ++row_counts[static_cast<std::size_t>(inds[static_cast<scl::Index>(k)])];
        }
    }

    csr_indptr[0] = 0;
    for (scl::Index i = 0; i < rows; ++i) {
        csr_indptr[i + 1] = csr_indptr[i] + row_counts[static_cast<std::size_t>(i)];
    }

    std::vector<scl::Index> row_pos(static_cast<std::size_t>(rows), 0);
    for (scl::Index j = 0; j < cols; ++j) {
        auto vals = csc.col_values(j);
        auto inds = csc.col_indices(j);

        for (scl::Size k = 0; k < vals.size(); ++k) {
            scl::Index row = inds[static_cast<scl::Index>(k)];
            scl::Index pos = csr_indptr[row] + row_pos[static_cast<std::size_t>(row)];

            csr_data[pos] = vals[static_cast<scl::Index>(k)];
            csr_indices[pos] = j;
            ++row_pos[static_cast<std::size_t>(row)];
        }
    }
}

/// @brief 稀疏转稠密 (行优先)
template <typename T, bool IsCSR>
void to_dense(const MappedSparse<T, IsCSR>& sparse, T* dense_out) {
    scl::Index rows = sparse.rows();
    scl::Index cols = sparse.cols();

    std::memset(dense_out, 0, static_cast<std::size_t>(rows * cols) * sizeof(T));

    if constexpr (IsCSR) {
        for (scl::Index i = 0; i < rows; ++i) {
            auto vals = sparse.row_values(i);
            auto inds = sparse.row_indices(i);
            for (scl::Size k = 0; k < vals.size(); ++k) {
                scl::Index j = inds[static_cast<scl::Index>(k)];
                dense_out[i * cols + j] = vals[static_cast<scl::Index>(k)];
            }
        }
    } else {
        for (scl::Index j = 0; j < cols; ++j) {
            auto vals = sparse.col_values(j);
            auto inds = sparse.col_indices(j);
            for (scl::Size k = 0; k < vals.size(); ++k) {
                scl::Index i = inds[static_cast<scl::Index>(k)];
                dense_out[i * cols + j] = vals[static_cast<scl::Index>(k)];
            }
        }
    }
}

// =============================================================================
// SECTION 7: Statistics (Pre-compute for Python)
// =============================================================================

/// @brief 计算每行/列的 nnz
template <typename T, bool IsCSR>
void compute_row_nnz(const MappedSparse<T, IsCSR>& mat, scl::Index* nnz_out) {
    scl::Index n = IsCSR ? mat.rows() : mat.cols();
    for (scl::Index i = 0; i < n; ++i) {
        if constexpr (IsCSR) {
            nnz_out[i] = mat.row_length(i);
        } else {
            nnz_out[i] = mat.col_length(i);
        }
    }
}

/// @brief 计算掩码选择后的总 nnz (用于预分配)
template <typename T, bool IsCSR>
scl::Index compute_masked_nnz(const MappedSparse<T, IsCSR>& src,
                               const uint8_t* row_mask,
                               const uint8_t* col_mask) {
    std::vector<scl::Index> col_remap;
    bool filter_cols = (col_mask != nullptr);

    if (filter_cols) {
        col_remap.resize(static_cast<std::size_t>(src.cols()), -1);
        build_remap(col_mask, src.cols(), col_remap.data());
    }

    scl::Index total = 0;

    for (scl::Index i = 0; i < src.rows(); ++i) {
        if (row_mask && !row_mask[i]) continue;

        if constexpr (IsCSR) {
            if (filter_cols) {
                auto inds = src.row_indices(i);
                for (scl::Size k = 0; k < inds.size(); ++k) {
                    if (col_remap[static_cast<std::size_t>(inds[static_cast<scl::Index>(k)])] >= 0) {
                        ++total;
                    }
                }
            } else {
                total += src.row_length(i);
            }
        } else {
            if (filter_cols) {
                auto inds = src.col_indices(i);
                for (scl::Size k = 0; k < inds.size(); ++k) {
                    if (col_remap[static_cast<std::size_t>(inds[static_cast<scl::Index>(k)])] >= 0) {
                        ++total;
                    }
                }
            } else {
                total += src.col_length(i);
            }
        }
    }

    return total;
}

// =============================================================================
// SECTION 8: File I/O
// =============================================================================

/// @brief 二进制文件头
struct BinaryHeader {
    char magic[4] = {'S', 'C', 'L', 'M'};
    uint32_t version = 1;
    uint32_t dtype_code;
    uint32_t index_code;
    uint32_t matrix_type;  // 0=dense, 1=csr, 2=csc
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int64_t data_offset;
    int64_t indices_offset;
    int64_t indptr_offset;
};

/// @brief 从文件创建加载函数
inline LoadFunc make_file_loader(const std::string& filepath, int64_t base_offset) {
    return [filepath, base_offset](std::size_t page_idx, std::byte* dest) {
        std::ifstream f(filepath, std::ios::binary);
        f.seekg(base_offset + static_cast<std::streamoff>(page_idx * SCL_MMAP_PAGE_SIZE));
        f.read(reinterpret_cast<char*>(dest), SCL_MMAP_PAGE_SIZE);
    };
}

/// @brief 从原始指针创建加载函数
template <typename T>
inline LoadFunc make_ptr_loader(const T* src, std::size_t total_bytes) {
    return [src, total_bytes](std::size_t page_idx, std::byte* dest) {
        std::size_t offset = page_idx * SCL_MMAP_PAGE_SIZE;
        std::size_t copy_size = std::min(
            static_cast<std::size_t>(SCL_MMAP_PAGE_SIZE),
            total_bytes > offset ? total_bytes - offset : 0
        );
        if (copy_size > 0) {
            std::memcpy(dest, reinterpret_cast<const std::byte*>(src) + offset, copy_size);
        }
    };
}

/// @brief 从 SCL 二进制文件创建 MappedSparse
template <typename T, bool IsCSR>
std::unique_ptr<MappedSparse<T, IsCSR>> open_sparse_file(
    const std::string& filepath,
    const MmapConfig& config = MmapConfig{}) {

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    BinaryHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::strncmp(header.magic, "SCLM", 4) != 0) {
        throw std::runtime_error("Invalid SCL matrix file");
    }

    return std::make_unique<MappedSparse<T, IsCSR>>(
        static_cast<scl::Index>(header.rows),
        static_cast<scl::Index>(header.cols),
        static_cast<scl::Index>(header.nnz),
        make_file_loader(filepath, header.data_offset),
        make_file_loader(filepath, header.indices_offset),
        make_file_loader(filepath, header.indptr_offset),
        config
    );
}

/// @brief 从 SCL 二进制文件创建 MappedDense
template <typename T>
std::unique_ptr<MappedDense<T>> open_dense_file(
    const std::string& filepath,
    const MmapConfig& config = MmapConfig{}) {

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    BinaryHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    return std::make_unique<MappedDense<T>>(
        static_cast<scl::Index>(header.rows),
        static_cast<scl::Index>(header.cols),
        make_file_loader(filepath, header.data_offset),
        config
    );
}

// =============================================================================
// SECTION 9: Utility
// =============================================================================

/// @brief 估算内存需求
template <typename T>
inline std::size_t estimate_sparse_memory(scl::Index rows, scl::Index nnz) {
    return static_cast<std::size_t>(nnz) * (sizeof(T) + sizeof(scl::Index))
         + static_cast<std::size_t>(rows + 1) * sizeof(scl::Index);
}

/// @brief 选择后端建议
enum class BackendHint {
    InMemory,
    Mapped,
    Streaming
};

inline BackendHint suggest_backend(std::size_t data_bytes,
                                    std::size_t available_mb = 4096) {
    std::size_t threshold = available_mb * 1024 * 1024;
    if (data_bytes < threshold / 4) return BackendHint::InMemory;
    if (data_bytes < threshold * 2) return BackendHint::Mapped;
    return BackendHint::Streaming;
}

} // namespace scl::mmap
