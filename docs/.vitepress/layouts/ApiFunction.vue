<script setup lang="ts">
/**
 * API Function Layout
 *
 * Renders a C API function page by combining:
 * - Structured data from YAML (params, errors, complexity, etc.)
 * - Content from Markdown sections (brief, description, notes, etc.)
 */
import { computed, onMounted, ref } from 'vue'
import { useData } from 'vitepress'

// Components
import Badge from '../components/base/Badge.vue'
import ApiSignature from '../components/api/ApiSignature.vue'
import ParamTable from '../components/api/ParamTable.vue'
import SourceLink from '../components/meta/SourceLink.vue'

// Data loader - will be available after build
const data = ref<any>(null)

onMounted(async () => {
  try {
    const module = await import('../loaders/c-api.data')
    data.value = module.default
  } catch (e) {
    console.error('Failed to load c-api data:', e)
  }
})

const { frontmatter } = useData()

const func = computed(() => {
  if (!data.value?.modules) return null

  const mod = data.value.modules.find(
    (m: any) => m.name === frontmatter.value.module
  )
  if (!mod) return null

  return mod.functions.find(
    (f: any) => f.id === frontmatter.value.function
  )
})

const statusColor = computed(() => {
  if (!func.value) return 'default'
  switch (func.value.status) {
    case 'stable': return 'green'
    case 'beta': return 'yellow'
    case 'experimental': return 'red'
    case 'deprecated': return 'red'
    default: return 'default'
  }
})

// Helper to render markdown content (basic)
const renderMarkdown = (content: string) => {
  if (!content) return ''
  // Basic markdown rendering - convert lists and code
  return content
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(.+)$/gm, (match) => {
      if (match.startsWith('<')) return match
      return `<p>${match}</p>`
    })
}
</script>

<template>
  <div class="api-function-page" v-if="func">
    <!-- Header with badges -->
    <div class="api-function-header">
      <h1>{{ func.id }}</h1>
      <div class="badges">
        <Badge type="version">{{ func.version }}</Badge>
        <Badge type="status" :color="statusColor">{{ func.status }}</Badge>
      </div>
    </div>

    <!-- Brief description -->
    <p class="brief" v-if="func.content.brief">{{ func.content.brief }}</p>

    <!-- Formula (if present) -->
    <div class="formula" v-if="func.content.formula">
      <div v-html="func.content.formula" />
    </div>

    <!-- Function signature -->
    <ApiSignature
      :return-type="func.return_type"
      :name="func.id"
    >
      <template v-for="(param, idx) in func.params" :key="param.name">
        <span v-if="param.type.includes('const ')" class="keyword">const </span>
        <span class="type">{{ param.type.replace('const ', '') }}</span>
        <span class="param-name"> {{ param.name }}</span>
        <span v-if="param.default" class="default-value"> = {{ param.default }}</span>
        <span v-if="idx < func.params.length - 1">,<br>  </span>
      </template>
    </ApiSignature>

    <!-- Parameters -->
    <h2>Parameters</h2>
    <ParamTable :params="func.params.map((p: any) => ({
      name: p.name,
      type: p.type,
      dir: p.dir,
      description: '',
      required: !p.nullable,
      default: p.default
    }))" />

    <!-- Errors -->
    <h2>Errors</h2>
    <table class="scl-error-table">
      <thead>
        <tr>
          <th>Code</th>
          <th>Condition</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="err in func.errors" :key="err.code">
          <td><code>{{ err.code }}</code></td>
          <td>{{ err.condition }}</td>
        </tr>
      </tbody>
    </table>

    <!-- Description -->
    <h2>Description</h2>
    <div class="content-section" v-html="renderMarkdown(func.content.description)" />

    <!-- FFI Stability -->
    <h2>FFI Stability</h2>
    <div class="content-section" v-html="renderMarkdown(func.content.ffi_stability)" />

    <!-- Data Guarantees -->
    <h2>Data Guarantees</h2>
    <div class="content-section" v-html="renderMarkdown(func.content.data_guarantees)" />

    <!-- Mutability -->
    <h2>Mutability</h2>
    <div class="content-section" v-html="renderMarkdown(func.content.mutability)" />

    <!-- Thread Safety -->
    <h2>Thread Safety</h2>
    <div class="content-section" v-html="renderMarkdown(func.content.thread_safety)" />

    <!-- Complexity -->
    <h2 v-if="func.complexity">Complexity</h2>
    <div class="complexity-badges" v-if="func.complexity">
      <Badge type="complexity">Time: {{ func.complexity.time }}</Badge>
      <Badge type="complexity">Space: {{ func.complexity.space }}</Badge>
    </div>

    <!-- Notes -->
    <template v-if="func.content.notes">
      <h2>Notes</h2>
      <div class="content-section" v-html="renderMarkdown(func.content.notes)" />
    </template>

    <!-- Source link -->
    <div class="source-section">
      <SourceLink :file="func.source.file" :line="func.source.line" />
    </div>
  </div>

  <!-- Loading state -->
  <div v-else class="loading">
    Loading function data...
  </div>
</template>

<style scoped>
.api-function-page {
  max-width: 900px;
}

.api-function-header {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.api-function-header h1 {
  margin: 0;
  font-family: var(--scl-font-mono);
}

.badges {
  display: flex;
  gap: 8px;
}

.brief {
  font-size: 1.1em;
  color: var(--vp-c-text-2);
  margin-bottom: 24px;
}

.formula {
  background: var(--scl-card-bg);
  padding: 16px;
  border-radius: var(--scl-radius-lg);
  margin-bottom: 24px;
  overflow-x: auto;
}

.content-section {
  line-height: 1.7;
  color: var(--vp-c-text-2);
}

.content-section :deep(ul) {
  padding-left: 20px;
  margin: 8px 0;
}

.content-section :deep(li) {
  margin: 4px 0;
}

.content-section :deep(code) {
  background: var(--scl-card-bg);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: var(--scl-font-mono);
  font-size: 0.9em;
}

.complexity-badges {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.source-section {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--scl-card-border);
}

.scl-error-table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
}

.scl-error-table th,
.scl-error-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--scl-card-border);
}

.scl-error-table th {
  background: var(--scl-card-bg);
  font-weight: 600;
}

.scl-error-table code {
  font-family: var(--scl-font-mono);
  color: var(--scl-badge-red-text);
}

.loading {
  padding: 40px;
  text-align: center;
  color: var(--vp-c-text-3);
}
</style>
