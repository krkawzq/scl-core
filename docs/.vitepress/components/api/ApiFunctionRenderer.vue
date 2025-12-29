<script setup lang="ts">
/**
 * API Function Renderer Component
 *
 * Reads YAML data and renders the function page.
 * Used directly in Markdown files with minimal setup.
 */
import { computed, ref, onMounted } from 'vue'
import { useData } from 'vitepress'

// Import types
interface CApiParam {
  name: string
  type: string
  dir: 'in' | 'out' | 'inout'
  nullable?: boolean
  default?: string
}

interface CApiError {
  code: string
  condition: string
}

interface CApiFunction {
  id: string
  return_type: string
  source: { file: string; line: number }
  params: CApiParam[]
  errors: CApiError[]
  complexity?: { time: string; space: string }
  version: string
  status: string
}

const props = defineProps<{
  data: CApiFunction
}>()

const statusColor = computed(() => {
  switch (props.data.status) {
    case 'stable': return 'green'
    case 'beta': return 'yellow'
    case 'experimental': return 'red'
    case 'deprecated': return 'red'
    default: return 'default'
  }
})

const statusLabel = computed(() => {
  return props.data.status.charAt(0).toUpperCase() + props.data.status.slice(1)
})
</script>

<template>
  <div class="api-func-renderer">
    <!-- Header -->
    <div class="func-header">
      <span class="scl-badge scl-badge--blue scl-badge--mono">v{{ data.version }}</span>
      <span
        class="scl-badge"
        :class="`scl-badge--${statusColor}`"
      >{{ statusLabel }}</span>
    </div>

    <!-- Signature -->
    <div class="scl-api-signature">
      <pre><code><span class="type">{{ data.return_type }}</span> <span class="fn-name">{{ data.id }}</span>(
<template v-for="(param, idx) in data.params" :key="param.name">  <span v-if="param.type.includes('const ')" class="keyword">const </span><span class="type">{{ param.type.replace('const ', '') }}</span> <span class="param-name">{{ param.name }}</span><span v-if="param.default" class="default-value"> = {{ param.default }}</span><span v-if="idx < data.params.length - 1">,
</span></template>
)</code></pre>
    </div>

    <!-- Parameters -->
    <h3>Parameters</h3>
    <table class="scl-param-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Type</th>
          <th>Dir</th>
          <th>Required</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="param in data.params" :key="param.name">
          <td><code class="scl-param-table__name">{{ param.name }}</code></td>
          <td><code class="scl-param-table__type">{{ param.type }}</code></td>
          <td>
            <span
              class="scl-param-table__dir"
              :class="`scl-param-table__dir--${param.dir}`"
            >{{ param.dir }}</span>
          </td>
          <td>{{ param.nullable ? 'No' : 'Yes' }}</td>
        </tr>
      </tbody>
    </table>

    <!-- Errors -->
    <h3>Possible Errors</h3>
    <table class="scl-param-table">
      <thead>
        <tr>
          <th>Code</th>
          <th>Condition</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="err in data.errors" :key="err.code">
          <td><code style="color: var(--scl-badge-red-text);">{{ err.code }}</code></td>
          <td>{{ err.condition }}</td>
        </tr>
      </tbody>
    </table>

    <!-- Complexity -->
    <template v-if="data.complexity">
      <h3>Complexity</h3>
      <div class="complexity-row">
        <span class="scl-badge scl-badge--purple scl-badge--mono">Time: {{ data.complexity.time }}</span>
        <span class="scl-badge scl-badge--purple scl-badge--mono">Space: {{ data.complexity.space }}</span>
      </div>
    </template>

    <!-- Source -->
    <div class="source-row">
      <a
        class="scl-source-link"
        :href="`https://github.com/krkawzq/scl-core/blob/main/${data.source.file}#L${data.source.line}`"
        target="_blank"
      >
        <svg class="scl-source-link__icon" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        {{ data.source.file }}:{{ data.source.line }}
      </a>
    </div>
  </div>
</template>

<style scoped>
.api-func-renderer {
  margin: 24px 0;
}

.func-header {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.complexity-row {
  display: flex;
  gap: 8px;
  margin: 8px 0 24px;
}

.source-row {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--scl-card-border);
}
</style>
