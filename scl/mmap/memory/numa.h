// =============================================================================
// FILE: scl/mmap/memory/numa.h
// BRIEF: API reference for NUMA-aware memory allocation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>

namespace scl::mmap::memory {

/* =============================================================================
 * STRUCT: NUMATopology
 * =============================================================================
 * SUMMARY:
 *     Describes NUMA topology of the system.
 *
 * DESIGN PURPOSE:
 *     Provides static information about system NUMA configuration:
 *     - Number of NUMA nodes
 *     - CPUs assigned to each node
 *     - Memory capacity per node
 *     - Inter-node distance matrix
 *
 * USAGE:
 *     auto topology = NUMATopology::detect();
 *     int local_node = topology.local_node();
 *     int nearest = topology.nearest_node(cpu_id);
 *
 * THREAD SAFETY:
 *     Immutable after construction. Thread-safe for reads.
 * -------------------------------------------------------------------------- */
struct NUMATopology {
    std::size_t num_nodes;                       // Number of NUMA nodes
    std::vector<std::vector<int>> node_cpus;     // CPUs per node [node][cpu_list]
    std::vector<std::size_t> node_memory_bytes;  // Total memory per node
    std::vector<std::size_t> node_free_bytes;    // Free memory per node (at detection)
    std::vector<std::vector<int>> distance_matrix; // Inter-node distances [from][to]

    /* -------------------------------------------------------------------------
     * STATIC METHOD: detect
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Detect system NUMA topology.
     *
     * ALGORITHM:
     *     Linux: Parse /sys/devices/system/node/
     *     Other: Return single-node topology
     *
     * POSTCONDITIONS:
     *     Returns valid topology (at least 1 node).
     *     On non-NUMA systems, returns single node with all CPUs.
     *
     * THREAD SAFETY:
     *     Thread-safe (reads sysfs).
     * ---------------------------------------------------------------------- */
    static NUMATopology detect();

    /* -------------------------------------------------------------------------
     * STATIC METHOD: single_node
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Create single-node topology for non-NUMA systems.
     *
     * RETURNS:
     *     Topology with 1 node containing all CPUs.
     * ---------------------------------------------------------------------- */
    static NUMATopology single_node();

    /* -------------------------------------------------------------------------
     * METHOD: nearest_node
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Find NUMA node with lowest distance from given CPU.
     *
     * PARAMETERS:
     *     cpu_id [in] - CPU ID to query
     *
     * RETURNS:
     *     Node ID with minimum distance, or 0 if CPU not found.
     * ---------------------------------------------------------------------- */
    int nearest_node(int cpu_id) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: local_node
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get NUMA node for current thread.
     *
     * ALGORITHM:
     *     1. Get current CPU via sched_getcpu()
     *     2. Return node containing that CPU
     *
     * RETURNS:
     *     Local node ID, or 0 on error.
     * ---------------------------------------------------------------------- */
    int local_node() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: node_for_cpu
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get NUMA node containing given CPU.
     *
     * PARAMETERS:
     *     cpu_id [in] - CPU ID to query
     *
     * RETURNS:
     *     Node ID, or -1 if CPU not found.
     * ---------------------------------------------------------------------- */
    int node_for_cpu(int cpu_id) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: distance
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get distance between two NUMA nodes.
     *
     * PARAMETERS:
     *     from [in] - Source node ID
     *     to   [in] - Destination node ID
     *
     * RETURNS:
     *     Distance value (10 = local, higher = remote).
     *     Returns 0 if invalid node IDs.
     * ---------------------------------------------------------------------- */
    int distance(int from, int to) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: is_numa_available
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if system has multiple NUMA nodes.
     * ---------------------------------------------------------------------- */
    bool is_numa_available() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: total_memory
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Total memory across all nodes in bytes.
     * ---------------------------------------------------------------------- */
    std::size_t total_memory() const noexcept;
};

/* =============================================================================
 * ENUM: NUMAPolicy
 * =============================================================================
 * SUMMARY:
 *     Memory allocation policy for NUMA systems.
 *
 * VALUES:
 *     Local      - Allocate on local node only (fail if unavailable)
 *     Preferred  - Prefer local node, fallback to other nodes
 *     Interleave - Round-robin across all nodes (good for shared data)
 *     Bind       - Strict binding to specified node set
 *
 * SELECTION GUIDE:
 *     Local:      Best for thread-local data
 *     Preferred:  Best for most allocations (default)
 *     Interleave: Best for shared read-only data
 *     Bind:       Best for explicit NUMA control
 * -------------------------------------------------------------------------- */
enum class NUMAPolicy : std::uint8_t {
    Local,
    Preferred,
    Interleave,
    Bind
};

/* =============================================================================
 * STRUCT: NUMAConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for NUMA-aware allocator.
 *
 * FIELDS:
 *     policy          - Allocation policy
 *     preferred_node  - Preferred node for Preferred/Bind policy
 *     bind_mask       - Node mask for Bind policy (bit per node)
 *     enable_migrate  - Allow page migration on access
 *     migrate_threshold - Remote accesses before migration
 * -------------------------------------------------------------------------- */
struct NUMAConfig {
    NUMAPolicy policy = NUMAPolicy::Preferred;
    int preferred_node = -1;          // -1 = auto (local node)
    std::uint64_t bind_mask = 0;      // Bitmask of allowed nodes
    bool enable_migrate = true;       // Enable page migration
    std::size_t migrate_threshold = 16; // Remote accesses before migrate

    /* -------------------------------------------------------------------------
     * FACTORY: local
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for local-only allocation.
     * ---------------------------------------------------------------------- */
    static constexpr NUMAConfig local() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: preferred
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for preferred-node allocation.
     * ---------------------------------------------------------------------- */
    static constexpr NUMAConfig preferred(int node = -1) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: interleave
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for interleaved allocation.
     * ---------------------------------------------------------------------- */
    static constexpr NUMAConfig interleave() noexcept;
};

/* =============================================================================
 * STRUCT: NUMAStats
 * =============================================================================
 * SUMMARY:
 *     Statistics for NUMA allocator.
 * -------------------------------------------------------------------------- */
struct NUMAStats {
    std::vector<std::size_t> allocations_per_node;  // Allocations per node
    std::vector<std::size_t> bytes_per_node;        // Bytes allocated per node
    std::size_t migrations;                          // Total page migrations
    std::size_t migration_failures;                  // Failed migrations
    std::size_t remote_accesses;                     // Detected remote accesses
};

/* =============================================================================
 * CLASS: NUMAAllocator
 * =============================================================================
 * SUMMARY:
 *     NUMA-aware page allocator for optimal memory placement.
 *
 * DESIGN PURPOSE:
 *     Provides NUMA-aware memory allocation:
 *     - Allocate pages on specific NUMA nodes
 *     - Query page location (which node)
 *     - Migrate pages between nodes
 *     - Track allocation statistics
 *
 * ARCHITECTURE:
 *     Uses libnuma on Linux for NUMA operations.
 *     Falls back to standard allocation on non-NUMA systems.
 *
 * MEMORY ALIGNMENT:
 *     All allocations are page-aligned (typically 4KB minimum).
 *     Large pages (2MB/1GB) supported via huge pages.
 *
 * THREAD SAFETY:
 *     All methods are thread-safe.
 * -------------------------------------------------------------------------- */
class NUMAAllocator {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: NUMAAllocator
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     config [in] - NUMA configuration
     *
     * POSTCONDITIONS:
     *     - Topology detected
     *     - Allocator ready for use
     * ---------------------------------------------------------------------- */
    explicit NUMAAllocator(NUMAConfig config = {});

    ~NUMAAllocator();

    NUMAAllocator(const NUMAAllocator&) = delete;
    NUMAAllocator& operator=(const NUMAAllocator&) = delete;
    NUMAAllocator(NUMAAllocator&&) noexcept;
    NUMAAllocator& operator=(NUMAAllocator&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: allocate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate memory on appropriate NUMA node.
     *
     * PARAMETERS:
     *     size_bytes [in] - Allocation size (will be page-aligned)
     *     hint_node  [in] - Preferred node (-1 = use policy default)
     *
     * PRECONDITIONS:
     *     size_bytes > 0
     *
     * POSTCONDITIONS:
     *     On success: Returns aligned pointer, memory zeroed.
     *     On failure: Returns nullptr.
     *
     * ALGORITHM:
     *     1. Round size up to page boundary
     *     2. Determine target node based on policy and hint
     *     3. Attempt allocation on target node
     *     4. If Preferred policy and fails, try other nodes
     *     5. Return pointer or nullptr
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void* allocate(std::size_t size_bytes, int hint_node = -1);

    /* -------------------------------------------------------------------------
     * METHOD: deallocate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Free previously allocated memory.
     *
     * PARAMETERS:
     *     ptr        [in] - Pointer from allocate()
     *     size_bytes [in] - Original allocation size
     *
     * PRECONDITIONS:
     *     ptr was returned by allocate() with same size.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void deallocate(void* ptr, std::size_t size_bytes);

    /* -------------------------------------------------------------------------
     * METHOD: get_node
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Query which NUMA node a page resides on.
     *
     * PARAMETERS:
     *     ptr [in] - Pointer to query
     *
     * RETURNS:
     *     Node ID, or -1 if cannot determine.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    int get_node(const void* ptr) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: migrate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Move memory pages to different NUMA node.
     *
     * PARAMETERS:
     *     ptr         [in] - Pointer to migrate
     *     size_bytes  [in] - Size of allocation
     *     target_node [in] - Destination node
     *
     * PRECONDITIONS:
     *     - ptr was returned by allocate()
     *     - target_node is valid node ID
     *
     * POSTCONDITIONS:
     *     On success: Pages moved to target node.
     *     On failure: Pages remain on original node.
     *
     * RETURNS:
     *     True if migration succeeded.
     *
     * NOTE:
     *     Migration may be slow for large allocations.
     *     Pages are migrated lazily on some systems.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    bool migrate(void* ptr, std::size_t size_bytes, int target_node);

    /* -------------------------------------------------------------------------
     * METHOD: touch_pages
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Force page allocation by touching each page.
     *
     * PARAMETERS:
     *     ptr        [in] - Pointer to touch
     *     size_bytes [in] - Size of allocation
     *
     * DESIGN PURPOSE:
     *     Ensures pages are actually allocated on target node.
     *     Without touching, pages may not be allocated until first access.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void touch_pages(void* ptr, std::size_t size_bytes);

    /* -------------------------------------------------------------------------
     * METHOD: topology
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to detected NUMA topology.
     * ---------------------------------------------------------------------- */
    const NUMATopology& topology() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to current configuration.
     * ---------------------------------------------------------------------- */
    const NUMAConfig& config() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: stats
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current allocation statistics.
     * ---------------------------------------------------------------------- */
    NUMAStats stats() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: reset_stats
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset allocation statistics to zero.
     * ---------------------------------------------------------------------- */
    void reset_stats() noexcept;

    /* -------------------------------------------------------------------------
     * STATIC METHOD: instance
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get global singleton allocator.
     *
     * RETURNS:
     *     Reference to default allocator with Preferred policy.
     *
     * THREAD SAFETY:
     *     Thread-safe (static initialization).
     * ---------------------------------------------------------------------- */
    static NUMAAllocator& instance();

private:
    NUMATopology topology_;
    NUMAConfig config_;
    mutable NUMAStats stats_;

    int select_node(int hint_node) const;
    void* allocate_on_node(std::size_t size_bytes, int node);
    void update_stats(int node, std::size_t bytes, bool is_alloc);
};

/* =============================================================================
 * FUNCTION: numa_policy_name
 * =============================================================================
 * SUMMARY:
 *     Convert NUMAPolicy enum to human-readable string.
 *
 * PARAMETERS:
 *     policy [in] - NUMA policy enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* numa_policy_name(NUMAPolicy policy) noexcept;

} // namespace scl::mmap::memory
