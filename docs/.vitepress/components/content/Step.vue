<script setup lang="ts">
interface Props {
  title?: string
}

defineProps<Props>()
</script>

<template>
  <li class="scl-step">
    <div v-if="title" class="scl-step__title">{{ title }}</div>
    <div class="scl-step__content">
      <slot></slot>
    </div>
  </li>
</template>
