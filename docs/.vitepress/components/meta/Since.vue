<script setup lang="ts">
interface Props {
  version: string
}

defineProps<Props>()
</script>

<template>
  <span class="scl-badge scl-badge--blue scl-badge--mono">
    v{{ version }}
  </span>
</template>
