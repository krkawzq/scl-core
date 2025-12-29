<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  lang?: 'cpp' | 'python' | 'c'
  returnType?: string
  name: string
  template?: string[]
  deprecated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  lang: 'cpp',
  deprecated: false
})

const copied = ref(false)

const templateStr = computed(() => {
  if (!props.template || props.template.length === 0) return ''
  return `template <${props.template.join(', ')}>`
})

const copySignature = async () => {
  const sig = buildSignatureText()
  try {
    await navigator.clipboard.writeText(sig)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  } catch (e) {
    console.error('Failed to copy:', e)
  }
}

const buildSignatureText = () => {
  let sig = ''
  if (templateStr.value) sig += templateStr.value + '\n'
  if (props.returnType) sig += props.returnType + ' '
  sig += props.name + '(...)'
  return sig
}
</script>

<template>
  <div class="scl-api-signature" :class="{ 'scl-api-signature--deprecated': deprecated }">
    <button
      v-if="!deprecated"
      class="scl-api-signature__copy"
      @click="copySignature"
    >
      {{ copied ? '✓ Copied' : 'Copy' }}
    </button>
    <span v-if="deprecated" class="scl-api-signature__deprecated">DEPRECATED</span>
    <pre><code><span v-if="templateStr" class="template">{{ templateStr }}
</span><span v-if="returnType" class="type">{{ returnType }}</span> <span class="fn-name">{{ name }}</span>(<slot></slot>)</code></pre>
  </div>
</template>
