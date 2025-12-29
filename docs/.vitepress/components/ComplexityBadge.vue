<script setup lang="ts">
interface Props {
  type: 'time' | 'space'
  complexity: string
}

defineProps<Props>()
</script>

<template>
  <span class="complexity-badge" :class="type">{{ complexity }}</span>
</template>
