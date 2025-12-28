// =============================================================================
// FILE: scl/mmap/cache/eviction.hpp
// BRIEF: Cache eviction policy implementations
// =============================================================================
#pragma once

#include "eviction.h"
#include "../configuration.hpp"
#include "scl/core/macros.hpp"

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace scl::mmap::cache {

// =============================================================================
// EvictionConfig Implementation
// =============================================================================

constexpr EvictionConfig EvictionConfig::lru() noexcept {
    return EvictionConfig{
        .policy = EvictionPolicy::LRU,
        .arc_p_init = 0.5,
        .lfu_decay_period = 1000,
        .cost_weight = 1.0,
        .recency_weight = 1.0,
        .frequency_weight = 0.0,
        .priority_weight = 0.0
    };
}

constexpr EvictionConfig EvictionConfig::lfu() noexcept {
    return EvictionConfig{
        .policy = EvictionPolicy::LFU,
        .arc_p_init = 0.5,
        .lfu_decay_period = 1000,
        .cost_weight = 0.0,
        .recency_weight = 0.0,
        .frequency_weight = 1.0,
        .priority_weight = 0.0
    };
}

constexpr EvictionConfig EvictionConfig::arc() noexcept {
    return EvictionConfig{
        .policy = EvictionPolicy::ARC,
        .arc_p_init = 0.5,
        .lfu_decay_period = 1000,
        .cost_weight = 1.0,
        .recency_weight = 1.0,
        .frequency_weight = 0.5,
        .priority_weight = 2.0
    };
}

constexpr EvictionConfig EvictionConfig::cost_based() noexcept {
    return EvictionConfig{
        .policy = EvictionPolicy::CostBased,
        .arc_p_init = 0.5,
        .lfu_decay_period = 1000,
        .cost_weight = 2.0,
        .recency_weight = 1.0,
        .frequency_weight = 0.5,
        .priority_weight = 1.0
    };
}

// =============================================================================
// Internal Structures
// =============================================================================

struct LRUEntry {
    std::size_t slot_idx;
    PageMetadata metadata;
};

struct LFUEntry {
    std::size_t slot_idx;
    PageMetadata metadata;
    std::size_t decay_generation;
};

struct ARCEntry {
    std::size_t slot_idx;
    PageMetadata metadata;
    bool in_t2;  // True if in T2 (frequently used), false if in T1
};

struct ClockEntry {
    std::size_t slot_idx;
    PageMetadata metadata;
    bool reference_bit;
};

struct CostEntry {
    std::size_t slot_idx;
    PageMetadata metadata;
    double score;

    bool operator>(const CostEntry& other) const {
        return score > other.score;
    }
};

// =============================================================================
// EvictionTracker::Impl
// =============================================================================

struct EvictionTracker::Impl {
    EvictionConfig config;
    std::size_t capacity;
    EvictionStats stats{};

    // LRU structures
    std::list<LRUEntry> lru_list;
    std::unordered_map<std::size_t, std::list<LRUEntry>::iterator> lru_map;

    // LFU structures
    std::unordered_map<std::size_t, LFUEntry> lfu_map;
    std::size_t lfu_generation = 0;
    std::size_t lfu_access_count = 0;

    // ARC structures (Adaptive Replacement Cache)
    std::list<ARCEntry> t1_list;  // Recent, seen once
    std::list<ARCEntry> t2_list;  // Frequent, seen multiple times
    std::list<std::size_t> b1_list;  // Ghost list for T1
    std::list<std::size_t> b2_list;  // Ghost list for T2
    std::unordered_map<std::size_t, std::list<ARCEntry>::iterator> t1_map;
    std::unordered_map<std::size_t, std::list<ARCEntry>::iterator> t2_map;
    // Maps for O(1) ghost list removal (instead of O(n) list.remove())
    std::unordered_map<std::size_t, std::list<std::size_t>::iterator> b1_map;
    std::unordered_map<std::size_t, std::list<std::size_t>::iterator> b2_map;
    std::unordered_set<std::size_t> b1_set;
    std::unordered_set<std::size_t> b2_set;
    double arc_p = 0.5;  // Target size for T1

    // Clock structures
    // Clock buffer and hand are mutable: reference bit clearing and hand movement
    // during select_victim are implementation details that don't affect logical constness
    mutable std::vector<ClockEntry> clock_buffer;
    mutable std::size_t clock_hand = 0;
    std::unordered_map<std::size_t, std::size_t> clock_map;  // slot -> buffer index

    // Cost-based structures
    std::unordered_map<std::size_t, CostEntry> cost_map;

    // Pinned pages
    // Mutable: select_victims temporarily modifies this for batch selection,
    // but restores original state before returning (net effect is zero)
    mutable std::unordered_set<std::size_t> pinned_slots;

    explicit Impl(std::size_t cap, EvictionConfig cfg)
        : config(cfg), capacity(cap), arc_p(cfg.arc_p_init) {
        if (config.policy == EvictionPolicy::Clock) {
            clock_buffer.reserve(cap);
        }
    }

    // LRU implementation
    void lru_access(std::size_t slot_idx, const PageMetadata& metadata, bool is_hit) {
        auto it = lru_map.find(slot_idx);
        if (it != lru_map.end()) {
            // Move to front (most recently used)
            lru_list.splice(lru_list.begin(), lru_list, it->second);
            it->second->metadata = metadata;
        } else {
            // Insert at front
            lru_list.push_front(LRUEntry{slot_idx, metadata});
            lru_map[slot_idx] = lru_list.begin();
        }
    }

    void lru_evict(std::size_t slot_idx) {
        auto it = lru_map.find(slot_idx);
        if (it != lru_map.end()) {
            lru_list.erase(it->second);
            lru_map.erase(it);
        }
    }

    std::size_t lru_select_victim(bool exclude_pinned) const noexcept {
        for (auto it = lru_list.rbegin(); it != lru_list.rend(); ++it) {
            if (!exclude_pinned || pinned_slots.find(it->slot_idx) == pinned_slots.end()) {
                return it->slot_idx;
            }
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // LFU implementation
    void lfu_access(std::size_t slot_idx, const PageMetadata& metadata, bool is_hit) {
        ++lfu_access_count;
        if (lfu_access_count >= config.lfu_decay_period) {
            // Decay all frequencies
            for (auto& [_, entry] : lfu_map) {
                entry.metadata.access_count /= 2;
            }
            lfu_access_count = 0;
            ++lfu_generation;
        }

        auto it = lfu_map.find(slot_idx);
        if (it != lfu_map.end()) {
            it->second.metadata = metadata;
            it->second.metadata.access_count++;
        } else {
            lfu_map[slot_idx] = LFUEntry{slot_idx, metadata, lfu_generation};
            lfu_map[slot_idx].metadata.access_count = 1;
        }
    }

    void lfu_evict(std::size_t slot_idx) {
        lfu_map.erase(slot_idx);
    }

    std::size_t lfu_select_victim(bool exclude_pinned) const noexcept {
        std::size_t victim = std::numeric_limits<std::size_t>::max();
        std::uint32_t min_freq = std::numeric_limits<std::uint32_t>::max();

        for (const auto& [slot, entry] : lfu_map) {
            if (exclude_pinned && pinned_slots.find(slot) != pinned_slots.end()) {
                continue;
            }
            if (entry.metadata.access_count < min_freq) {
                min_freq = entry.metadata.access_count;
                victim = slot;
            }
        }
        return victim;
    }

    // ARC implementation
    void arc_access(std::size_t slot_idx, const PageMetadata& metadata, bool is_hit) {
        // Check ghost lists first
        if (b1_set.find(slot_idx) != b1_set.end()) {
            // Hit in B1 ghost: increase p (favor recency)
            double delta = static_cast<double>(b2_set.size()) /
                          std::max(b1_set.size(), std::size_t(1));
            arc_p = std::min(arc_p + std::max(delta, 1.0), static_cast<double>(capacity));
            // Remove from both set, map, and list (O(1) via map iterator)
            auto it = b1_map.find(slot_idx);
            if (it != b1_map.end()) {
                b1_list.erase(it->second);
                b1_map.erase(it);
            }
            b1_set.erase(slot_idx);
            stats.ghost_hits++;
            stats.adaptive_changes++;
        } else if (b2_set.find(slot_idx) != b2_set.end()) {
            // Hit in B2 ghost: decrease p (favor frequency)
            double delta = static_cast<double>(b1_set.size()) /
                          std::max(b2_set.size(), std::size_t(1));
            arc_p = std::max(arc_p - std::max(delta, 1.0), 0.0);
            // Remove from both set, map, and list (O(1) via map iterator)
            auto it = b2_map.find(slot_idx);
            if (it != b2_map.end()) {
                b2_list.erase(it->second);
                b2_map.erase(it);
            }
            b2_set.erase(slot_idx);
            stats.ghost_hits++;
            stats.adaptive_changes++;
        }

        // Check if in T1
        auto t1_it = t1_map.find(slot_idx);
        if (t1_it != t1_map.end()) {
            // Move from T1 to T2 (promote to frequent)
            auto entry = *t1_it->second;
            t1_list.erase(t1_it->second);
            t1_map.erase(t1_it);

            entry.metadata = metadata;
            entry.in_t2 = true;
            t2_list.push_front(entry);
            t2_map[slot_idx] = t2_list.begin();
            return;
        }

        // Check if in T2
        auto t2_it = t2_map.find(slot_idx);
        if (t2_it != t2_map.end()) {
            // Move to front of T2
            t2_list.splice(t2_list.begin(), t2_list, t2_it->second);
            t2_it->second->metadata = metadata;
            return;
        }

        // New entry: add to T1
        ARCEntry entry{slot_idx, metadata, false};
        t1_list.push_front(entry);
        t1_map[slot_idx] = t1_list.begin();
    }

    void arc_evict(std::size_t slot_idx) {
        auto t1_it = t1_map.find(slot_idx);
        if (t1_it != t1_map.end()) {
            t1_list.erase(t1_it->second);
            t1_map.erase(t1_it);
            // Add to B1 ghost list (front = newest)
            b1_list.push_front(slot_idx);
            b1_map[slot_idx] = b1_list.begin();
            b1_set.insert(slot_idx);
            // Enforce capacity limit (remove oldest from back)
            while (b1_list.size() > capacity) {
                std::size_t oldest = b1_list.back();
                b1_list.pop_back();
                b1_map.erase(oldest);
                b1_set.erase(oldest);
            }
            return;
        }

        auto t2_it = t2_map.find(slot_idx);
        if (t2_it != t2_map.end()) {
            t2_list.erase(t2_it->second);
            t2_map.erase(t2_it);
            // Add to B2 ghost list (front = newest)
            b2_list.push_front(slot_idx);
            b2_map[slot_idx] = b2_list.begin();
            b2_set.insert(slot_idx);
            // Enforce capacity limit (remove oldest from back)
            while (b2_list.size() > capacity) {
                std::size_t oldest = b2_list.back();
                b2_list.pop_back();
                b2_map.erase(oldest);
                b2_set.erase(oldest);
            }
        }
    }

    std::size_t arc_select_victim(bool exclude_pinned) const noexcept {
        // Use p to decide which list to evict from
        std::size_t t1_size = t1_list.size();
        std::size_t t2_size = t2_list.size();

        bool prefer_t1 = (t1_size > 0) &&
                        ((t1_size > static_cast<std::size_t>(arc_p)) ||
                         (t2_size == 0));

        if (prefer_t1) {
            // Evict from T1 (LRU of recent)
            for (auto it = t1_list.rbegin(); it != t1_list.rend(); ++it) {
                if (!exclude_pinned || pinned_slots.find(it->slot_idx) == pinned_slots.end()) {
                    return it->slot_idx;
                }
            }
        }

        // Evict from T2 (LRU of frequent)
        for (auto it = t2_list.rbegin(); it != t2_list.rend(); ++it) {
            if (!exclude_pinned || pinned_slots.find(it->slot_idx) == pinned_slots.end()) {
                return it->slot_idx;
            }
        }

        // Fallback to T1 if T2 has no candidates
        for (auto it = t1_list.rbegin(); it != t1_list.rend(); ++it) {
            if (!exclude_pinned || pinned_slots.find(it->slot_idx) == pinned_slots.end()) {
                return it->slot_idx;
            }
        }

        return std::numeric_limits<std::size_t>::max();
    }

    // Clock implementation
    void clock_access(std::size_t slot_idx, const PageMetadata& metadata, bool is_hit) {
        auto it = clock_map.find(slot_idx);
        if (it != clock_map.end()) {
            // Set reference bit
            clock_buffer[it->second].reference_bit = true;
            clock_buffer[it->second].metadata = metadata;
        } else {
            // Add new entry
            std::size_t idx = clock_buffer.size();
            clock_buffer.push_back(ClockEntry{slot_idx, metadata, true});
            clock_map[slot_idx] = idx;
        }
    }

    void clock_evict(std::size_t slot_idx) {
        auto it = clock_map.find(slot_idx);
        if (it != clock_map.end()) {
            std::size_t idx = it->second;
            // Swap with last element
            if (idx < clock_buffer.size() - 1) {
                std::swap(clock_buffer[idx], clock_buffer.back());
                clock_map[clock_buffer[idx].slot_idx] = idx;
            }
            clock_buffer.pop_back();
            clock_map.erase(it);
            if (clock_hand >= clock_buffer.size()) {
                clock_hand = 0;
            }
        }
    }

    std::size_t clock_select_victim(bool exclude_pinned) const noexcept {
        if (clock_buffer.empty()) {
            return std::numeric_limits<std::size_t>::max();
        }

        std::size_t start = clock_hand;
        do {
            auto& entry = clock_buffer[clock_hand];
            bool is_pinned = pinned_slots.find(entry.slot_idx) != pinned_slots.end();

            if (!entry.reference_bit && (!exclude_pinned || !is_pinned)) {
                return entry.slot_idx;
            }

            // Clear reference bit (second chance)
            if (!is_pinned) {
                entry.reference_bit = false;
            }

            clock_hand = (clock_hand + 1) % clock_buffer.size();
        } while (clock_hand != start);

        // All pages referenced, return any unpinned
        for (const auto& entry : clock_buffer) {
            if (!exclude_pinned || pinned_slots.find(entry.slot_idx) == pinned_slots.end()) {
                return entry.slot_idx;
            }
        }

        return std::numeric_limits<std::size_t>::max();
    }

    // Cost-based implementation
    void cost_access(std::size_t slot_idx, const PageMetadata& metadata, bool is_hit) {
        auto now = std::chrono::steady_clock::now();
        double score = compute_eviction_score(metadata, config, now);
        cost_map[slot_idx] = CostEntry{slot_idx, metadata, score};
    }

    void cost_evict(std::size_t slot_idx) {
        cost_map.erase(slot_idx);
    }

    std::size_t cost_select_victim(bool exclude_pinned) const noexcept {
        auto now = std::chrono::steady_clock::now();
        std::size_t victim = std::numeric_limits<std::size_t>::max();
        double min_score = std::numeric_limits<double>::max();

        for (const auto& [slot, entry] : cost_map) {
            if (exclude_pinned && pinned_slots.find(slot) != pinned_slots.end()) {
                continue;
            }
            double score = compute_eviction_score(entry.metadata, config, now);
            if (score < min_score) {
                min_score = score;
                victim = slot;
            }
        }
        return victim;
    }

    std::size_t current_size() const noexcept {
        switch (config.policy) {
            case EvictionPolicy::LRU:
                return lru_list.size();
            case EvictionPolicy::LFU:
                return lfu_map.size();
            case EvictionPolicy::ARC:
                return t1_list.size() + t2_list.size();
            case EvictionPolicy::Clock:
                return clock_buffer.size();
            case EvictionPolicy::CostBased:
                return cost_map.size();
            default:
                return 0;
        }
    }

    void clear_all() {
        lru_list.clear();
        lru_map.clear();
        lfu_map.clear();
        t1_list.clear();
        t2_list.clear();
        b1_list.clear();
        b2_list.clear();
        b1_set.clear();
        b2_set.clear();
        t1_map.clear();
        t2_map.clear();
        arc_p = config.arc_p_init;
        clock_buffer.clear();
        clock_map.clear();
        clock_hand = 0;
        cost_map.clear();
        pinned_slots.clear();
    }
};

// =============================================================================
// EvictionTracker Implementation
// =============================================================================

EvictionTracker::EvictionTracker(std::size_t capacity, EvictionConfig config)
    : impl_(std::make_unique<Impl>(capacity, config)) {}

EvictionTracker::~EvictionTracker() = default;

EvictionTracker::EvictionTracker(EvictionTracker&& other) noexcept = default;
EvictionTracker& EvictionTracker::operator=(EvictionTracker&& other) noexcept = default;

void EvictionTracker::on_access(
    std::size_t slot_idx,
    const PageMetadata& metadata,
    bool is_hit
) {
    switch (impl_->config.policy) {
        case EvictionPolicy::LRU:
            impl_->lru_access(slot_idx, metadata, is_hit);
            break;
        case EvictionPolicy::LFU:
            impl_->lfu_access(slot_idx, metadata, is_hit);
            break;
        case EvictionPolicy::ARC:
            impl_->arc_access(slot_idx, metadata, is_hit);
            break;
        case EvictionPolicy::Clock:
            impl_->clock_access(slot_idx, metadata, is_hit);
            break;
        case EvictionPolicy::CostBased:
            impl_->cost_access(slot_idx, metadata, is_hit);
            break;
    }
}

void EvictionTracker::on_evict(std::size_t slot_idx) {
    impl_->stats.total_evictions++;

    switch (impl_->config.policy) {
        case EvictionPolicy::LRU:
            impl_->lru_evict(slot_idx);
            break;
        case EvictionPolicy::LFU:
            impl_->lfu_evict(slot_idx);
            break;
        case EvictionPolicy::ARC:
            impl_->arc_evict(slot_idx);
            break;
        case EvictionPolicy::Clock:
            impl_->clock_evict(slot_idx);
            break;
        case EvictionPolicy::CostBased:
            impl_->cost_evict(slot_idx);
            break;
    }
}

std::size_t EvictionTracker::select_victim(bool exclude_pinned) const noexcept {
    switch (impl_->config.policy) {
        case EvictionPolicy::LRU:
            return impl_->lru_select_victim(exclude_pinned);
        case EvictionPolicy::LFU:
            return impl_->lfu_select_victim(exclude_pinned);
        case EvictionPolicy::ARC:
            return impl_->arc_select_victim(exclude_pinned);
        case EvictionPolicy::Clock:
            return impl_->clock_select_victim(exclude_pinned);
        case EvictionPolicy::CostBased:
            return impl_->cost_select_victim(exclude_pinned);
        default:
            return std::numeric_limits<std::size_t>::max();
    }
}

std::size_t EvictionTracker::select_victims(
    std::size_t count,
    bool exclude_pinned,
    std::span<std::size_t> output
) const {
    std::size_t selected = 0;
    const std::size_t max_select = std::min(count, output.size());

    while (selected < max_select) {
        std::size_t victim = select_victim(exclude_pinned);
        if (victim == std::numeric_limits<std::size_t>::max()) {
            break;
        }

        // Check if already selected (use output array itself, avoid extra allocation)
        bool duplicate = false;
        for (std::size_t i = 0; i < selected; ++i) {
            if (output[i] == victim) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            break;  // No more unique victims
        }

        output[selected++] = victim;

        // Temporarily mark as pinned to avoid re-selecting
        impl_->pinned_slots.insert(victim);
    }

    // Restore pinned state (all selected items were temporarily pinned)
    for (std::size_t i = 0; i < selected; ++i) {
        impl_->pinned_slots.erase(output[i]);
    }

    return selected;
}

void EvictionTracker::pin(std::size_t slot_idx) {
    impl_->pinned_slots.insert(slot_idx);
}

void EvictionTracker::unpin(std::size_t slot_idx) {
    impl_->pinned_slots.erase(slot_idx);
}

void EvictionTracker::set_priority(std::size_t slot_idx, std::int8_t priority) {
    if (impl_->config.policy == EvictionPolicy::CostBased) {
        auto it = impl_->cost_map.find(slot_idx);
        if (it != impl_->cost_map.end()) {
            it->second.metadata.priority = priority;
        }
    }
}

std::size_t EvictionTracker::capacity() const noexcept {
    return impl_->capacity;
}

std::size_t EvictionTracker::size() const noexcept {
    return impl_->current_size();
}

const EvictionConfig& EvictionTracker::config() const noexcept {
    return impl_->config;
}

EvictionStats EvictionTracker::stats() const noexcept {
    return impl_->stats;
}

void EvictionTracker::reset_stats() noexcept {
    impl_->stats = EvictionStats{};
}

void EvictionTracker::clear() {
    impl_->clear_all();
}

// =============================================================================
// Free Functions
// =============================================================================

inline const char* eviction_policy_name(EvictionPolicy policy) noexcept {
    switch (policy) {
        case EvictionPolicy::LRU:       return "LRU";
        case EvictionPolicy::LFU:       return "LFU";
        case EvictionPolicy::ARC:       return "ARC";
        case EvictionPolicy::Clock:     return "Clock";
        case EvictionPolicy::CostBased: return "CostBased";
        default:                        return "Unknown";
    }
}

inline double compute_eviction_score(
    const PageMetadata& metadata,
    const EvictionConfig& config,
    std::chrono::steady_clock::time_point now
) noexcept {
    // Compute age in seconds
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - metadata.last_access
    ).count();
    double age_seconds = std::max(age / 1000.0, 0.001);  // Avoid division by zero

    // Compute latency factor (higher latency = more valuable to keep)
    double latency_ms = metadata.load_latency.count() / 1e6;
    double latency_factor = std::max(latency_ms, 0.001);

    // Compute score: higher = more valuable to keep
    double score = 0.0;

    // Cost component: expensive pages are more valuable
    score += config.cost_weight * latency_factor;

    // Recency component: recently used pages are more valuable
    score += config.recency_weight * (1.0 / age_seconds);

    // Frequency component: frequently used pages are more valuable
    score += config.frequency_weight * static_cast<double>(metadata.access_count);

    // Priority component: high priority pages are more valuable
    score += config.priority_weight * (metadata.priority + 128.0) / 256.0;

    // Pinned pages get infinite score
    if (metadata.is_pinned) {
        return std::numeric_limits<double>::max();
    }

    return score;
}

} // namespace scl::mmap::cache
