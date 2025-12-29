<script setup lang="ts">
interface Props {
  file: string
  line?: number
  repo?: string
}

const props = withDefaults(defineProps<Props>(), {
  repo: 'https://github.com/krkawzq/scl-core'
})

const href = computed(() => {
  const base = `${props.repo}/blob/main/${props.file}`
  return props.line ? `${base}#L${props.line}` : base
})
</script>

<template>
  <a class="source-link" :href="href" target="_blank" rel="noopener">
    {{ file }}<span v-if="line">:{{ line }}</span>
  </a>
</template>
