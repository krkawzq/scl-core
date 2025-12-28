// =============================================================================
// FILE: scl/mmap/memory/numa.hpp
// BRIEF: NUMA-aware memory allocation implementation
// =============================================================================
#pragma once

#include "numa.h"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"

#include <atomic>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>

// Optional libnuma support
#if __has_include(<numa.h>)
#include <numa.h>
#include <numaif.h>
#define SCL_HAS_LIBNUMA 1
#else
#define SCL_HAS_LIBNUMA 0
#endif

#endif // __linux__

namespace scl::mmap::memory {

// =============================================================================
// NUMAConfig Implementation
// =============================================================================

constexpr NUMAConfig NUMAConfig::local() noexcept {
    return NUMAConfig{
        .policy = NUMAPolicy::Local,
        .preferred_node = -1,
        .bind_mask = 0,
        .enable_migrate = false,
        .migrate_threshold = 0
    };
}

constexpr NUMAConfig NUMAConfig::preferred(int node) noexcept {
    return NUMAConfig{
        .policy = NUMAPolicy::Preferred,
        .preferred_node = node,
        .bind_mask = 0,
        .enable_migrate = true,
        .migrate_threshold = 16
    };
}

constexpr NUMAConfig NUMAConfig::interleave() noexcept {
    return NUMAConfig{
        .policy = NUMAPolicy::Interleave,
        .preferred_node = -1,
        .bind_mask = 0,
        .enable_migrate = false,
        .migrate_threshold = 0
    };
}

// =============================================================================
// NUMATopology Implementation
// =============================================================================

NUMATopology NUMATopology::single_node() {
    NUMATopology topo;
    topo.num_nodes = 1;
    topo.node_cpus.resize(1);
    topo.node_memory_bytes.resize(1, 0);
    topo.node_free_bytes.resize(1, 0);
    topo.distance_matrix.resize(1, std::vector<int>(1, 10));

    // Get number of CPUs
#ifdef _SC_NPROCESSORS_ONLN
    int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if (num_cpus <= 0) num_cpus = 1;
#else
    int num_cpus = 1;
#endif

    for (int i = 0; i < num_cpus; ++i) {
        topo.node_cpus[0].push_back(i);
    }

    // Get total memory
#ifdef _SC_PHYS_PAGES
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        topo.node_memory_bytes[0] = static_cast<std::size_t>(pages) *
                                    static_cast<std::size_t>(page_size);
        topo.node_free_bytes[0] = topo.node_memory_bytes[0];
    }
#endif

    return topo;
}

NUMATopology NUMATopology::detect() {
#ifdef __linux__
    // Check if NUMA is available
    const char* node_path = "/sys/devices/system/node";
    DIR* dir = opendir(node_path);
    if (!dir) {
        return single_node();
    }

    NUMATopology topo;
    std::vector<int> nodes;

    // Find all NUMA nodes
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, "node", 4) == 0) {
            int node_id = std::atoi(entry->d_name + 4);
            nodes.push_back(node_id);
        }
    }
    closedir(dir);

    if (nodes.empty()) {
        return single_node();
    }

    std::sort(nodes.begin(), nodes.end());
    topo.num_nodes = nodes.size();
    topo.node_cpus.resize(topo.num_nodes);
    topo.node_memory_bytes.resize(topo.num_nodes, 0);
    topo.node_free_bytes.resize(topo.num_nodes, 0);
    topo.distance_matrix.resize(topo.num_nodes,
                                std::vector<int>(topo.num_nodes, 10));

    // Read CPU list for each node
    for (std::size_t i = 0; i < topo.num_nodes; ++i) {
        std::string cpu_path = std::string(node_path) + "/node" +
                               std::to_string(nodes[i]) + "/cpulist";
        std::ifstream cpu_file(cpu_path);
        if (cpu_file) {
            std::string line;
            if (std::getline(cpu_file, line)) {
                // Parse CPU list (e.g., "0-7,16-23")
                std::istringstream iss(line);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    auto dash = token.find('-');
                    if (dash != std::string::npos) {
                        int start = std::stoi(token.substr(0, dash));
                        int end = std::stoi(token.substr(dash + 1));
                        for (int cpu = start; cpu <= end; ++cpu) {
                            topo.node_cpus[i].push_back(cpu);
                        }
                    } else {
                        topo.node_cpus[i].push_back(std::stoi(token));
                    }
                }
            }
        }

        // Read memory info
        std::string meminfo_path = std::string(node_path) + "/node" +
                                   std::to_string(nodes[i]) + "/meminfo";
        std::ifstream mem_file(meminfo_path);
        if (mem_file) {
            std::string line;
            while (std::getline(mem_file, line)) {
                if (line.find("MemTotal:") != std::string::npos) {
                    std::size_t kb = 0;
                    if (sscanf(line.c_str(), "%*s %*s %*s %zu", &kb) == 1) {
                        topo.node_memory_bytes[i] = kb * 1024;
                    }
                } else if (line.find("MemFree:") != std::string::npos) {
                    std::size_t kb = 0;
                    if (sscanf(line.c_str(), "%*s %*s %*s %zu", &kb) == 1) {
                        topo.node_free_bytes[i] = kb * 1024;
                    }
                }
            }
        }
    }

    // Read distance matrix
    for (std::size_t i = 0; i < topo.num_nodes; ++i) {
        std::string dist_path = std::string(node_path) + "/node" +
                                std::to_string(nodes[i]) + "/distance";
        std::ifstream dist_file(dist_path);
        if (dist_file) {
            for (std::size_t j = 0; j < topo.num_nodes && dist_file; ++j) {
                dist_file >> topo.distance_matrix[i][j];
            }
        }
    }

    return topo;
#else
    return single_node();
#endif
}

int NUMATopology::nearest_node(int cpu_id) const noexcept {
    int node = node_for_cpu(cpu_id);
    return (node >= 0) ? node : 0;
}

int NUMATopology::local_node() const noexcept {
#ifdef __linux__
    int cpu = sched_getcpu();
    if (cpu >= 0) {
        return node_for_cpu(cpu);
    }
#endif
    return 0;
}

int NUMATopology::node_for_cpu(int cpu_id) const noexcept {
    for (std::size_t node = 0; node < num_nodes; ++node) {
        for (int cpu : node_cpus[node]) {
            if (cpu == cpu_id) {
                return static_cast<int>(node);
            }
        }
    }
    return -1;
}

int NUMATopology::distance(int from, int to) const noexcept {
    if (from < 0 || to < 0 ||
        static_cast<std::size_t>(from) >= num_nodes ||
        static_cast<std::size_t>(to) >= num_nodes) {
        return 0;
    }
    return distance_matrix[static_cast<std::size_t>(from)]
                          [static_cast<std::size_t>(to)];
}

bool NUMATopology::is_numa_available() const noexcept {
    return num_nodes > 1;
}

std::size_t NUMATopology::total_memory() const noexcept {
    std::size_t total = 0;
    for (auto mem : node_memory_bytes) {
        total += mem;
    }
    return total;
}

// =============================================================================
// NUMAAllocator Implementation
// =============================================================================

NUMAAllocator::NUMAAllocator(NUMAConfig config)
    : topology_(NUMATopology::detect())
    , config_(config)
    , stats_{}
{
    stats_.allocations_per_node.resize(topology_.num_nodes, 0);
    stats_.bytes_per_node.resize(topology_.num_nodes, 0);
    stats_.migrations = 0;
    stats_.migration_failures = 0;
    stats_.remote_accesses = 0;

    // Set default preferred node if not specified
    if (config_.preferred_node < 0) {
        config_.preferred_node = topology_.local_node();
    }
}

NUMAAllocator::~NUMAAllocator() = default;

NUMAAllocator::NUMAAllocator(NUMAAllocator&& other) noexcept
    : topology_(std::move(other.topology_))
    , config_(other.config_)
    , stats_(std::move(other.stats_))
{}

NUMAAllocator& NUMAAllocator::operator=(NUMAAllocator&& other) noexcept {
    if (this != &other) {
        topology_ = std::move(other.topology_);
        config_ = other.config_;
        stats_ = std::move(other.stats_);
    }
    return *this;
}

int NUMAAllocator::select_node(int hint_node) const {
    if (hint_node >= 0 && static_cast<std::size_t>(hint_node) < topology_.num_nodes) {
        return hint_node;
    }

    switch (config_.policy) {
        case NUMAPolicy::Local:
        case NUMAPolicy::Preferred:
            return (config_.preferred_node >= 0) ?
                   config_.preferred_node : topology_.local_node();

        case NUMAPolicy::Interleave: {
            // Round-robin across nodes
            static std::atomic<std::size_t> counter{0};
            return static_cast<int>(counter.fetch_add(1) % topology_.num_nodes);
        }

        case NUMAPolicy::Bind:
            // Return first node in bind mask
            for (std::size_t i = 0; i < topology_.num_nodes; ++i) {
                if (config_.bind_mask & (1ULL << i)) {
                    return static_cast<int>(i);
                }
            }
            return 0;

        default:
            return 0;
    }
}

void* NUMAAllocator::allocate_on_node(std::size_t size_bytes, int node) {
    // Align size to page boundary
    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    size_bytes = (size_bytes + page_size - 1) & ~(page_size - 1);

    void* ptr = nullptr;

#if SCL_HAS_LIBNUMA
    if (topology_.is_numa_available() && numa_available() >= 0) {
        // Use libnuma for NUMA-aware allocation
        ptr = numa_alloc_onnode(size_bytes, node);
        if (ptr) {
            update_stats(node, size_bytes, true);
            return ptr;
        }
    }
#endif

    // Fallback: mmap with NUMA hints
#ifdef __linux__
    ptr = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (ptr != MAP_FAILED) {
#ifdef MPOL_BIND
        // Try to bind to specific node using mbind
        unsigned long nodemask = 1UL << node;
        if (mbind(ptr, size_bytes, MPOL_BIND, &nodemask,
                  topology_.num_nodes + 1, MPOL_MF_MOVE) == 0) {
            update_stats(node, size_bytes, true);
            return ptr;
        }
#endif
        // mbind failed, but allocation succeeded
        update_stats(0, size_bytes, true);
        return ptr;
    }
#else
    // Non-Linux: use aligned_alloc
    ptr = std::aligned_alloc(page_size, size_bytes);
    if (ptr) {
        std::memset(ptr, 0, size_bytes);
        update_stats(0, size_bytes, true);
        return ptr;
    }
#endif

    return nullptr;
}

void* NUMAAllocator::allocate(std::size_t size_bytes, int hint_node) {
    if (size_bytes == 0) return nullptr;

    int target_node = select_node(hint_node);
    void* ptr = allocate_on_node(size_bytes, target_node);

    if (!ptr && config_.policy == NUMAPolicy::Preferred) {
        // Try other nodes
        for (std::size_t i = 0; i < topology_.num_nodes; ++i) {
            if (static_cast<int>(i) != target_node) {
                ptr = allocate_on_node(size_bytes, static_cast<int>(i));
                if (ptr) break;
            }
        }
    }

    return ptr;
}

void NUMAAllocator::deallocate(void* ptr, std::size_t size_bytes) {
    if (!ptr) return;

    int node = get_node(ptr);
    if (node >= 0) {
        update_stats(node, size_bytes, false);
    }

    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    size_bytes = (size_bytes + page_size - 1) & ~(page_size - 1);

#if SCL_HAS_LIBNUMA
    if (topology_.is_numa_available() && numa_available() >= 0) {
        numa_free(ptr, size_bytes);
        return;
    }
#endif

#ifdef __linux__
    munmap(ptr, size_bytes);
#else
    std::free(ptr);
#endif
}

int NUMAAllocator::get_node(const void* ptr) const noexcept {
    if (!ptr) return -1;

#if SCL_HAS_LIBNUMA
    if (topology_.is_numa_available() && numa_available() >= 0) {
        int node = -1;
        if (get_mempolicy(&node, nullptr, 0,
                          const_cast<void*>(ptr), MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            return node;
        }
    }
#endif

#ifdef __linux__
    // Fallback: use move_pages to query
    int node = -1;
    void* pages[1] = {const_cast<void*>(ptr)};
    int status[1] = {-1};

    if (move_pages(0, 1, pages, nullptr, status, 0) == 0) {
        node = status[0];
    }
    return node;
#else
    return 0;
#endif
}

bool NUMAAllocator::migrate(void* ptr, std::size_t size_bytes, int target_node) {
    if (!ptr || target_node < 0 ||
        static_cast<std::size_t>(target_node) >= topology_.num_nodes) {
        return false;
    }

#if SCL_HAS_LIBNUMA
    if (topology_.is_numa_available() && numa_available() >= 0) {
        unsigned long nodemask = 1UL << target_node;
        if (mbind(ptr, size_bytes, MPOL_BIND, &nodemask,
                  topology_.num_nodes + 1, MPOL_MF_MOVE | MPOL_MF_STRICT) == 0) {
            stats_.migrations++;
            return true;
        }
        stats_.migration_failures++;
        return false;
    }
#endif

#ifdef __linux__
    // Fallback: use move_pages
    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    std::size_t num_pages = (size_bytes + page_size - 1) / page_size;

    std::vector<void*> pages(num_pages);
    std::vector<int> nodes(num_pages, target_node);
    std::vector<int> status(num_pages);

    for (std::size_t i = 0; i < num_pages; ++i) {
        pages[i] = static_cast<char*>(ptr) + i * page_size;
    }

    if (move_pages(0, static_cast<unsigned long>(num_pages),
                   pages.data(), nodes.data(), status.data(),
                   MPOL_MF_MOVE) == 0) {
        stats_.migrations++;
        return true;
    }
    stats_.migration_failures++;
#else
    (void)ptr;
    (void)size_bytes;
    (void)target_node;
#endif

    return false;
}

void NUMAAllocator::touch_pages(void* ptr, std::size_t size_bytes) {
    if (!ptr || size_bytes == 0) return;

    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    volatile char* p = static_cast<volatile char*>(ptr);

    for (std::size_t offset = 0; offset < size_bytes; offset += page_size) {
        // Read and write back to force page allocation
        char tmp = p[offset];
        p[offset] = tmp;
    }
}

const NUMATopology& NUMAAllocator::topology() const noexcept {
    return topology_;
}

const NUMAConfig& NUMAAllocator::config() const noexcept {
    return config_;
}

NUMAStats NUMAAllocator::stats() const noexcept {
    return stats_;
}

void NUMAAllocator::reset_stats() noexcept {
    for (auto& count : stats_.allocations_per_node) count = 0;
    for (auto& bytes : stats_.bytes_per_node) bytes = 0;
    stats_.migrations = 0;
    stats_.migration_failures = 0;
    stats_.remote_accesses = 0;
}

void NUMAAllocator::update_stats(int node, std::size_t bytes, bool is_alloc) {
    if (node < 0 || static_cast<std::size_t>(node) >= stats_.allocations_per_node.size()) {
        return;
    }

    if (is_alloc) {
        stats_.allocations_per_node[static_cast<std::size_t>(node)]++;
        stats_.bytes_per_node[static_cast<std::size_t>(node)] += bytes;
    } else {
        if (stats_.allocations_per_node[static_cast<std::size_t>(node)] > 0) {
            stats_.allocations_per_node[static_cast<std::size_t>(node)]--;
        }
        if (stats_.bytes_per_node[static_cast<std::size_t>(node)] >= bytes) {
            stats_.bytes_per_node[static_cast<std::size_t>(node)] -= bytes;
        }
    }
}

NUMAAllocator& NUMAAllocator::instance() {
    static NUMAAllocator allocator{NUMAConfig::preferred()};
    return allocator;
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* numa_policy_name(NUMAPolicy policy) noexcept {
    switch (policy) {
        case NUMAPolicy::Local:      return "Local";
        case NUMAPolicy::Preferred:  return "Preferred";
        case NUMAPolicy::Interleave: return "Interleave";
        case NUMAPolicy::Bind:       return "Bind";
        default:                     return "Unknown";
    }
}

} // namespace scl::mmap::memory
