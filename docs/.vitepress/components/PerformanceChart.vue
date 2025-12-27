<template>
  <div class="performance-chart">
    <h4 v-if="title" class="chart-title">
      <span class="chart-icon">📊</span>
      {{ title }}
    </h4>
    
    <div class="chart-container">
      <div class="chart-placeholder">
        <p class="placeholder-text">
          Performance chart placeholder
        </p>
        <p class="placeholder-hint">
          Integrate with Chart.js or ECharts for interactive visualizations
        </p>
      </div>
    </div>
    
    <div v-if="description" class="chart-description">
      {{ description }}
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string
  description?: string
  data?: any
}>()

// TODO: Integrate with Chart.js or ECharts for actual charting
// For now, this is a placeholder component
</script>

<style scoped>
.performance-chart {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.chart-title {
  margin: 0 0 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-text-1);
  font-size: 1.25rem;
}

.chart-icon {
  font-size: 1.5rem;
  line-height: 1;
}

.chart-container {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-bg);
  border: 2px dashed var(--vp-c-divider);
  border-radius: 6px;
  margin: 1rem 0;
}

.chart-placeholder {
  text-align: center;
  padding: 2rem;
}

.placeholder-text {
  color: var(--vp-c-text-2);
  font-size: 1.1rem;
  margin: 0 0 0.5rem 0;
  font-weight: 500;
}

.placeholder-hint {
  color: var(--vp-c-text-3);
  font-size: 0.9rem;
  margin: 0;
  font-style: italic;
}

.chart-description {
  color: var(--vp-c-text-2);
  font-size: 14px;
  line-height: 1.6;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-divider);
}
</style>

