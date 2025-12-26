#pragma once

// =============================================================================
// SCL MMAP Module - Main Header
// =============================================================================
//
// 内存映射稀疏矩阵模块，支持超大规模数据集的延迟加载。
//
// 核心组件:
// - Page/PageTable/PageHandle: 页面管理
// - PagePool/Scheduler: 调度和缓存
// - VirtualArray: 虚拟数组抽象
// - MappedSparse/MappedDense: 映射矩阵
// - MappedVirtualSparse: 零成本视图
//
// 使用示例:
//
//   #include "scl/mmap/mmap.hpp"
//
//   // 从文件创建映射矩阵
//   auto mat = scl::mmap::open_sparse_file<float, true>("data.sclm");
//
//   // 创建视图 (零成本)
//   uint8_t row_mask[1000] = {...};
//   auto view = MappedVirtualCSR(mat, row_mask, nullptr);
//
//   // 加载到内存
//   std::vector<float> data(view.nnz());
//   std::vector<int32_t> indices(view.nnz());
//   std::vector<int32_t> indptr(view.rows() + 1);
//   load_full(view, data.data(), indices.data(), indptr.data());
//
// =============================================================================

#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/array.hpp"
#include "scl/mmap/matrix.hpp"
#include "scl/mmap/convert.hpp"
#include "scl/mmap/math.hpp"
