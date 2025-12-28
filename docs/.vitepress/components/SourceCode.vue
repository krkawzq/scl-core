<template>
  <div class="source-code-block">
    <!-- Header -->
    <div class="source-code-header">
      <div class="source-code-title">
        <span class="source-code-icon">📄</span>
        <span class="title-text">{{ displayTitle }}</span>
      </div>
      <div class="source-code-meta">
        <span class="file-path" :title="fullFilePath">{{ displayPath }}</span>
        <span class="line-range">L{{ sourceData.startLine }}-{{ sourceData.endLine }}</span>
        <button class="copy-button" @click="copyCode" :class="{ copied }">
          <span v-if="!copied">📋</span>
          <span v-else>✓</span>
        </button>
      </div>
    </div>
    
    <!-- Code Content -->
    <div class="source-code-content">
      <div class="line-numbers" v-if="showLineNumbers">
        <span 
          v-for="n in lineCount" 
          :key="n" 
          class="line-number"
        >{{ sourceData.startLine + n - 1 }}</span>
      </div>
      <div class="code-container">
        <pre><code 
          ref="codeElement"
          :class="['language-' + lang, 'shiki']"
          v-html="highlightedCode"
        ></code></pre>
      </div>
    </div>
    
    <!-- Footer -->
    <div v-if="githubUrl" class="source-code-footer">
      <a :href="githubUrl" target="_blank" rel="noopener" class="github-link">
        <svg class="github-icon" viewBox="0 0 16 16" width="16" height="16">
          <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        View on GitHub
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useData } from 'vitepress'

// Props
const props = withDefaults(defineProps<{
  /** Source file path relative to project root */
  file: string
  /** Symbol name to extract (function, class, struct, etc.) */
  symbol: string
  /** Custom title (defaults to symbol name) */
  title?: string
  /** Language for syntax highlighting */
  lang?: string
  /** Show line numbers */
  lineNumbers?: boolean
  /** GitHub repository URL */
  repo?: string
  /** Git branch name */
  branch?: string
}>(), {
  lang: 'cpp',
  lineNumbers: true,
  branch: 'main'
})

// State
const copied = ref(false)
const highlightedCode = ref('')
const codeElement = ref<HTMLElement | null>(null)
const { isDark } = useData()

// Source data (will be populated at build time via virtual module)
const sourceData = ref({
  code: '',
  file: '',
  startLine: 1,
  endLine: 1,
  symbol: ''
})

// Computed
const displayTitle = computed(() => props.title || props.symbol)

const displayPath = computed(() => {
  const parts = sourceData.value.file.split('/')
  const sclIndex = parts.findIndex(p => p === 'scl')
  if (sclIndex !== -1) {
    return parts.slice(sclIndex).join('/')
  }
  return parts.slice(-3).join('/')
})

const fullFilePath = computed(() => sourceData.value.file)

const lineCount = computed(() => {
  return sourceData.value.endLine - sourceData.value.startLine + 1
})

const githubUrl = computed(() => {
  if (!props.repo) return null
  const relativePath = displayPath.value
  return `${props.repo}/blob/${props.branch}/${relativePath}#L${sourceData.value.startLine}-L${sourceData.value.endLine}`
})

// Methods
async function loadSourceCode() {
  try {
    // Dynamic import of virtual module
    const modulePath = `virtual:scl-source?file=${encodeURIComponent(props.file)}&symbol=${encodeURIComponent(props.symbol)}`
    const data = await import(/* @vite-ignore */ modulePath)
    
    sourceData.value = {
      code: data.code,
      file: data.file,
      startLine: data.startLine,
      endLine: data.endLine,
      symbol: data.symbol
    }
    
    await highlightCode()
  } catch (error) {
    console.error('[SourceCode] Failed to load source code:', error)
    highlightedCode.value = `<span class="error">Error loading source code: ${error}</span>`
  }
}

async function highlightCode() {
  const code = sourceData.value.code
  if (!code) return
  
  // Try to use Shiki (VitePress built-in)
  try {
    // @ts-ignore - Shiki is available globally in VitePress
    if (typeof window !== 'undefined' && window.__VP_HASH_MAP__) {
      // Client-side: use basic highlighting
      highlightedCode.value = escapeHtml(code)
    } else {
      // SSR: code will be highlighted by VitePress
      highlightedCode.value = escapeHtml(code)
    }
  } catch {
    // Fallback: no highlighting
    highlightedCode.value = escapeHtml(code)
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

async function copyCode() {
  try {
    await navigator.clipboard.writeText(sourceData.value.code)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch (error) {
    console.error('Failed to copy code:', error)
  }
}

// Lifecycle
onMounted(() => {
  loadSourceCode()
})

// Watch for prop changes
watch(() => [props.file, props.symbol], () => {
  loadSourceCode()
})
</script>

<style scoped>
.source-code-block {
  background: var(--vp-code-block-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  margin: 1.5rem 0;
  overflow: hidden;
  font-family: var(--vp-font-family-mono);
}

/* Header */
.source-code-header {
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
  padding: 0.75rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.source-code-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  font-size: 0.95rem;
  font-family: var(--vp-font-family-base);
}

.source-code-icon {
  font-size: 1.1rem;
  line-height: 1;
}

.source-code-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.file-path {
  background: var(--vp-c-bg);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.line-range {
  opacity: 0.8;
  font-variant-numeric: tabular-nums;
}

.copy-button {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 32px;
}

.copy-button:hover {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-brand-1);
}

.copy-button.copied {
  background: var(--vp-c-green-soft);
  border-color: var(--vp-c-green-1);
  color: var(--vp-c-green-1);
}

/* Code Content */
.source-code-content {
  display: flex;
  overflow-x: auto;
}

.line-numbers {
  flex-shrink: 0;
  padding: 1rem 0;
  background: var(--vp-code-block-bg);
  border-right: 1px solid var(--vp-c-divider);
  text-align: right;
  user-select: none;
}

.line-number {
  display: block;
  padding: 0 0.75rem;
  font-size: var(--vp-code-font-size);
  line-height: var(--vp-code-line-height);
  color: var(--vp-c-text-3);
  font-variant-numeric: tabular-nums;
}

.code-container {
  flex: 1;
  overflow-x: auto;
}

.code-container pre {
  margin: 0;
  padding: 1rem;
  background: transparent;
  overflow: visible;
}

.code-container code {
  font-size: var(--vp-code-font-size);
  line-height: var(--vp-code-line-height);
  color: var(--vp-c-text-1);
  white-space: pre;
  word-wrap: normal;
}

/* Footer */
.source-code-footer {
  background: var(--vp-c-bg-soft);
  border-top: 1px solid var(--vp-c-divider);
  padding: 0.5rem 1rem;
  display: flex;
  justify-content: flex-end;
}

.github-link {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--vp-c-text-2);
  text-decoration: none;
  font-size: 0.8rem;
  font-family: var(--vp-font-family-base);
  transition: color 0.2s;
}

.github-link:hover {
  color: var(--vp-c-brand-1);
}

.github-icon {
  opacity: 0.8;
}

/* Error State */
.error {
  color: var(--vp-c-danger-1);
  padding: 1rem;
  display: block;
}

/* Responsive */
@media (max-width: 640px) {
  .source-code-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .source-code-meta {
    width: 100%;
    justify-content: space-between;
  }
  
  .file-path {
    max-width: 200px;
  }
}
</style>
