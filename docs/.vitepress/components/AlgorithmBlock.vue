<template>
  <div class="algorithm-block">
    <h4 class="algorithm-title">
      <span class="algorithm-icon">🧮</span>
      {{ title || 'Algorithm' }}
    </h4>
    
    <div v-if="summary" class="algorithm-summary">
      {{ summary }}
    </div>
    
    <div class="algorithm-content">
      <slot />
    </div>
    
    <div v-if="complexity" class="algorithm-footer">
      <div class="complexity-info">
        <ComplexityBadge 
          v-if="complexity.time" 
          type="time" 
          :value="complexity.time" 
        />
        <ComplexityBadge 
          v-if="complexity.space" 
          type="space" 
          :value="complexity.space" 
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import ComplexityBadge from './ComplexityBadge.vue'

defineProps<{
  title?: string
  summary?: string
  complexity?: {
    time?: string
    space?: string
  }
}>()
</script>

<style scoped>
.algorithm-block {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid var(--vp-c-brand-1);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.algorithm-title {
  margin: 0 0 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-brand-1);
  font-size: 1.25rem;
}

.algorithm-icon {
  font-size: 1.5rem;
  line-height: 1;
}

.algorithm-summary {
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
  line-height: 1.6;
  font-style: italic;
}

.algorithm-content {
  color: var(--vp-c-text-1);
}

.algorithm-content :deep(ol),
.algorithm-content :deep(ul) {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.algorithm-content :deep(li) {
  margin: 0.5rem 0;
  line-height: 1.6;
}

.algorithm-content :deep(code) {
  font-size: 13px;
  padding: 2px 6px;
  border-radius: 4px;
}

.algorithm-footer {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-divider);
}

.complexity-info {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
</style>

