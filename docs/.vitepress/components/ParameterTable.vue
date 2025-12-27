<template>
  <div class="parameter-table-wrapper">
    <table class="parameter-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Type</th>
          <th>Direction</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="param in parameters" :key="param.name">
          <td><code class="param-name">{{ param.name }}</code></td>
          <td><code class="param-type">{{ param.type }}</code></td>
          <td>
            <span :class="['param-direction', param.direction]">
              {{ param.direction }}
            </span>
          </td>
          <td class="param-description">{{ param.description }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
interface Parameter {
  name: string
  type: string
  direction: 'in' | 'out' | 'in,out'
  description: string
}

defineProps<{
  parameters: Parameter[]
}>()
</script>

<style scoped>
.parameter-table-wrapper {
  margin: 1.5rem 0;
  overflow-x: auto;
}

.parameter-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.parameter-table th,
.parameter-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--vp-c-divider);
}

.parameter-table th {
  background: var(--vp-c-bg-soft);
  font-weight: 600;
  color: var(--vp-c-text-1);
  white-space: nowrap;
}

.parameter-table tbody tr:hover {
  background: var(--vp-c-bg-soft);
}

.param-name,
.param-type {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--vp-code-bg);
  color: var(--vp-c-text-1);
}

.param-name {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.param-direction {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
}

.param-direction.in {
  background: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
}

.param-direction.out {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.param-direction.in\,out {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.dark .param-direction.in {
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
}

.dark .param-direction.out {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
}

.dark .param-direction.in\,out {
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
}

.param-description {
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

@media (max-width: 768px) {
  .parameter-table {
    font-size: 13px;
  }
  
  .parameter-table th,
  .parameter-table td {
    padding: 8px;
  }
}
</style>

