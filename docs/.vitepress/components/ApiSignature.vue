<script setup lang="ts">
interface Props {
  lang?: 'cpp' | 'python'
  returnType?: string
  name: string
  templateParams?: string
  deprecated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  lang: 'cpp',
  deprecated: false
})
</script>

<template>
  <div class="api-signature" :class="{ deprecated }">
    <div v-if="deprecated" class="deprecated-badge">DEPRECATED</div>
    <pre><code><span v-if="templateParams" class="template">template &lt;{{ templateParams }}&gt;
</span><span v-if="returnType" class="type">{{ returnType }} </span><span class="fn-name">{{ name }}</span>(<slot></slot>)</code></pre>
  </div>
</template>

<style scoped>
.api-signature {
  position: relative;
}

.api-signature.deprecated {
  opacity: 0.7;
  border-color: #ef4444;
}

.deprecated-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 2px 8px;
  background: #ef4444;
  color: white;
  font-size: 10px;
  font-weight: 700;
  border-radius: 4px;
}

.template {
  color: var(--vp-c-text-3);
}
</style>
