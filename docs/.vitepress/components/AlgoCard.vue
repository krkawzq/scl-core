<script setup lang="ts">
interface Props {
  title: string
}

defineProps<Props>()
</script>

<template>
  <div class="algo-card">
    <div class="algo-card-title">{{ title }}</div>
    <div class="algo-card-content">
      <slot></slot>
    </div>
  </div>
</template>
