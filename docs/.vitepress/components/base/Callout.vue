<script setup lang="ts">
import { ref, computed } from 'vue'

type CalloutType = 'info' | 'tip' | 'warning' | 'danger' | 'note'

interface Props {
  type?: CalloutType
  title?: string
  collapsible?: boolean
  collapsed?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'info',
  collapsible: false,
  collapsed: false
})

const isCollapsed = ref(props.collapsed)

const toggle = () => {
  if (props.collapsible) {
    isCollapsed.value = !isCollapsed.value
  }
}

const icon = computed(() => {
  switch (props.type) {
    case 'info': return 'ℹ️'
    case 'tip': return '💡'
    case 'warning': return '⚠️'
    case 'danger': return '🚨'
    case 'note': return '📝'
    default: return 'ℹ️'
  }
})

const defaultTitle = computed(() => {
  switch (props.type) {
    case 'info': return 'Info'
    case 'tip': return 'Tip'
    case 'warning': return 'Warning'
    case 'danger': return 'Danger'
    case 'note': return 'Note'
    default: return 'Info'
  }
})
</script>

<template>
  <div
    class="scl-callout"
    :class="[
      `scl-callout--${type}`,
      {
        'scl-callout--collapsible': collapsible,
        'scl-callout--collapsed': isCollapsed
      }
    ]"
  >
    <div class="scl-callout__header" @click="toggle">
      <span class="scl-callout__icon">{{ icon }}</span>
      <span class="scl-callout__title">{{ title || defaultTitle }}</span>
      <span v-if="collapsible" class="scl-callout__toggle">▼</span>
    </div>
    <div class="scl-callout__content">
      <slot></slot>
    </div>
  </div>
</template>
