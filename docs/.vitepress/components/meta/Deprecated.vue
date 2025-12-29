<script setup lang="ts">
interface Props {
  since: string
  use?: string
}

defineProps<Props>()
</script>

<template>
  <div class="scl-callout scl-callout--warning">
    <div class="scl-callout__header">
      <span class="scl-callout__icon">⚠️</span>
      <span>Deprecated since v{{ since }}</span>
    </div>
    <div class="scl-callout__content">
      <slot></slot>
      <p v-if="use">
        Use <code>{{ use }}</code> instead.
      </p>
    </div>
  </div>
</template>
