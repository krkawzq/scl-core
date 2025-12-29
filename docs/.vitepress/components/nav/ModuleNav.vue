<script setup lang="ts">
/**
 * Module Function Navigation
 *
 * Displays a horizontal list of functions in the current module,
 * highlighting the current function.
 */
import { computed } from 'vue'

interface FunctionLink {
  id: string
  name: string
  href: string
  brief?: string
}

const props = defineProps<{
  module: string
  current: string
  functions: FunctionLink[]
}>()

const isActive = (funcId: string) => funcId === props.current
</script>

<template>
  <nav class="module-nav">
    <div class="module-nav__header">
      <span class="module-nav__label">{{ module }}</span>
      <span class="module-nav__count">{{ functions.length }} functions</span>
    </div>
    <div class="module-nav__list">
      <a
        v-for="func in functions"
        :key="func.id"
        :href="func.href"
        class="module-nav__item"
        :class="{ 'module-nav__item--active': isActive(func.id) }"
        :title="func.brief"
      >
        {{ func.name }}
      </a>
    </div>
  </nav>
</template>

<style scoped>
.module-nav {
  margin-bottom: 24px;
  padding: 16px;
  background: var(--scl-card-bg);
  border: 1px solid var(--scl-card-border);
  border-radius: var(--scl-radius-lg);
}

.module-nav__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--scl-card-border);
}

.module-nav__label {
  font-family: var(--scl-font-mono);
  font-size: var(--scl-text-sm);
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.module-nav__count {
  font-size: var(--scl-text-xs);
  color: var(--vp-c-text-3);
}

.module-nav__list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.module-nav__item {
  display: inline-block;
  padding: 6px 12px;
  font-family: var(--scl-font-mono);
  font-size: var(--scl-text-xs);
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  border: 1px solid var(--scl-card-border);
  border-radius: var(--scl-radius-sm);
  text-decoration: none;
  transition: all 0.2s;
}

.module-nav__item:hover {
  color: var(--scl-primary);
  border-color: var(--scl-primary);
  background: var(--scl-primary-soft);
}

.module-nav__item--active {
  color: white;
  background: var(--scl-primary);
  border-color: var(--scl-primary);
}

.module-nav__item--active:hover {
  color: white;
  background: var(--scl-primary-hover);
}
</style>
