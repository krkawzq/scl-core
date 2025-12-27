<template>
  <span :class="['complexity-badge', type]">
    <span class="badge-icon">{{ icon }}</span>
    <span class="badge-value">{{ value }}</span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  type: 'time' | 'space'
  value: string
}>()

const icon = computed(() => {
  return props.type === 'time' ? '⏱️' : '💾'
})
</script>

<style scoped>
.complexity-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  font-weight: 500;
  font-family: var(--vp-font-family-mono);
  white-space: nowrap;
}

.complexity-badge.time {
  background: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
  border: 1px solid rgba(59, 130, 246, 0.3);
}

.complexity-badge.space {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.dark .complexity-badge.time {
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
}

.dark .complexity-badge.space {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
}

.badge-icon {
  font-size: 14px;
  line-height: 1;
}

.badge-value {
  line-height: 1;
}
</style>

