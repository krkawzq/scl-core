#pragma once

#include <cstddef>

// =============================================================================
// SCL MMAP Configuration
// =============================================================================

// 页面大小 (默认 1MB，可通过编译参数覆盖)
#ifndef SCL_MMAP_PAGE_SIZE
#define SCL_MMAP_PAGE_SIZE (1024 * 1024)
#endif

// 默认池大小 (页数)
#ifndef SCL_MMAP_DEFAULT_POOL_SIZE
#define SCL_MMAP_DEFAULT_POOL_SIZE 64
#endif

namespace scl::mmap {

/// @brief Mmap 运行时配置
struct MmapConfig {
    /// 最大驻留页数 (决定内存占用)
    std::size_t max_resident_pages = SCL_MMAP_DEFAULT_POOL_SIZE;

    /// 预取深度 (顺序扫描时预加载的页数)
    std::size_t prefetch_depth = 4;

    /// 是否启用写回 (脏页持久化)
    bool enable_writeback = false;
};

} // namespace scl::mmap
