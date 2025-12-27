<template>
  <div class="api-card">
    <div class="api-header">
      <h3 class="api-name">{{ name }}</h3>
      <div class="api-badges">
        <ComplexityBadge 
          v-if="timeComplexity" 
          type="time" 
          :value="timeComplexity" 
        />
        <ComplexityBadge 
          v-if="spaceComplexity" 
          type="space" 
          :value="spaceComplexity" 
        />
      </div>
    </div>
    
    <p v-if="summary" class="api-summary">{{ summary }}</p>
    
    <div class="api-signature">
      <code>{{ signature }}</code>
    </div>
    
    <div v-if="$slots.default" class="api-content">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import ComplexityBadge from './ComplexityBadge.vue'

defineProps<{
  name: string
  signature: string
  summary?: string
  timeComplexity?: string
  spaceComplexity?: string
}>()
</script>

<style scoped>
.api-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1.5rem 0;
  transition: all 0.3s ease;
}

.api-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 12px rgba(95, 103, 238, 0.15);
}

.api-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.api-name {
  margin: 0;
  font-size: 1.5rem;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
}

.api-badges {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.api-summary {
  color: var(--vp-c-text-2);
  margin: 0 0 1rem 0;
  line-height: 1.6;
}

.api-signature {
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  background: var(--vp-code-block-bg);
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
  margin: 1rem 0;
}

.api-signature code {
  color: var(--vp-c-text-1);
  white-space: pre;
}

.api-content {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-divider);
}

@media (max-width: 640px) {
  .api-header {
    flex-direction: column;
  }
  
  .api-name {
    font-size: 1.25rem;
  }
}
</style>

