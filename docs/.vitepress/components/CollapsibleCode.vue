<template>
  <div class="collapsible-code">
    <div 
      class="collapsible-header" 
      @click="toggle"
      :class="{ expanded: isExpanded }"
    >
      <div class="header-left">
        <span class="toggle-icon">{{ isExpanded ? '▼' : '▶' }}</span>
        <span class="code-title">{{ displayTitle }}</span>
      </div>
      <div class="header-right">
        <span class="code-meta">{{ displayPath }} · L{{ startLineNum }}-{{ endLineNum }}</span>
      </div>
    </div>
    
    <Transition name="expand">
      <div v-show="isExpanded" class="collapsible-content">
        <div v-html="highlightedCode"></div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useData } from 'vitepress'

const props = withDefaults(defineProps<{
  file: string
  symbol: string
  code: string
  startLine: number | string
  endLine: number | string
  title?: string
  collapsed?: boolean | string
  lang?: string
}>(), {
  collapsed: false
})

const { isDark } = useData()
const isExpanded = ref(!(props.collapsed === true || props.collapsed === 'true'))
const highlightedCode = ref('')

const displayTitle = computed(() => props.title || props.symbol)

const displayPath = computed(() => {
  const parts = props.file.split('/')
  const sclIndex = parts.findIndex(p => p === 'scl')
  if (sclIndex !== -1) {
    return parts.slice(sclIndex).join('/')
  }
  return parts.slice(-3).join('/')
})

const startLineNum = computed(() => Number(props.startLine))
const endLineNum = computed(() => Number(props.endLine))

function toggle() {
  isExpanded.value = !isExpanded.value
}

onMounted(() => {
  // The code HTML is already highlighted by VitePress and passed as a prop
  // We just need to set it
  highlightedCode.value = props.code
})
</script>

<style scoped>
.collapsible-code {
  margin: 1.5rem 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-c-bg-soft);
}

.collapsible-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: var(--vp-c-bg-soft);
  cursor: pointer;
  user-select: none;
  transition: background-color 0.2s;
}

.collapsible-header:hover {
  background: var(--vp-c-bg);
}

.collapsible-header.expanded {
  border-bottom: 1px solid var(--vp-c-divider);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
}

.toggle-icon {
  font-size: 0.875rem;
  color: var(--vp-c-brand-1);
  flex-shrink: 0;
  transition: transform 0.2s;
}

.code-title {
  font-weight: 600;
  color: var(--vp-c-brand-1);
  font-size: 0.95rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header-right {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  margin-left: 1rem;
}

.code-meta {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

.collapsible-content {
  overflow: hidden;
}

.collapsible-content :deep(div[class*="language-"]) {
  margin: 0 !important;
  border-radius: 0 !important;
  border: none !important;
}

.collapsible-content :deep(pre) {
  margin: 0 !important;
  border-radius: 0 !important;
}

/* Expand/collapse animation */
.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  max-height: 0;
}

.expand-enter-to,
.expand-leave-from {
  opacity: 1;
  max-height: 2000px;
}

/* Responsive */
@media (max-width: 640px) {
  .collapsible-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .header-right {
    margin-left: 1.5rem;
  }
  
  .code-meta {
    font-size: 0.75rem;
  }
}
</style>

