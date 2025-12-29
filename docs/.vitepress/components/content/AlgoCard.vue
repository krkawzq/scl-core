<script setup lang="ts">
interface Props {
  title: string
  icon?: string
  references?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  icon: '📐'
})
</script>

<template>
  <div class="scl-algo-card">
    <div class="scl-algo-card__header">
      <span class="scl-algo-card__icon">{{ icon }}</span>
      <h4 class="scl-algo-card__title">{{ title }}</h4>
    </div>

    <div v-if="$slots.formula" class="scl-algo-card__formula">
      <slot name="formula"></slot>
    </div>

    <div class="scl-algo-card__description">
      <slot></slot>
    </div>

    <div v-if="$slots.complexity" class="scl-algo-card__complexity">
      <slot name="complexity"></slot>
    </div>

    <div v-if="references && references.length" class="scl-algo-card__references">
      References: {{ references.join(', ') }}
    </div>
  </div>
</template>
