<script setup lang="ts">
import { computed } from 'vue'

type BadgeType = 'default' | 'version' | 'status' | 'complexity' | 'thread-safety' | 'custom'
type BadgeColor = 'default' | 'green' | 'yellow' | 'red' | 'blue' | 'purple'

interface Props {
  type?: BadgeType
  color?: BadgeColor
  mono?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'default',
  color: 'default',
  mono: false
})

const resolvedColor = computed(() => {
  if (props.color !== 'default') return props.color
  switch (props.type) {
    case 'version': return 'blue'
    case 'status': return 'green'
    case 'complexity': return 'purple'
    case 'thread-safety': return 'yellow'
    default: return 'default'
  }
})

const isMono = computed(() => {
  return props.mono || props.type === 'complexity' || props.type === 'version'
})
</script>

<template>
  <span
    class="scl-badge"
    :class="[
      `scl-badge--${resolvedColor}`,
      { 'scl-badge--mono': isMono }
    ]"
  >
    <slot></slot>
  </span>
</template>
