<script setup lang="ts">
interface Param {
  name: string
  type: string
  direction?: 'in' | 'out' | 'inout'
  description: string
  default?: string
  required?: boolean
}

interface Props {
  params: Param[]
}

defineProps<Props>()

function getDirClass(dir?: string) {
  if (!dir) return ''
  return `dir-${dir}`
}

function getDirLabel(dir?: string) {
  switch (dir) {
    case 'in': return 'in'
    case 'out': return 'out'
    case 'inout': return 'in/out'
    default: return ''
  }
}
</script>

<template>
  <table class="param-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Type</th>
        <th>Dir</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="param in params" :key="param.name">
        <td>
          <code>{{ param.name }}</code>
          <span v-if="param.required === false" class="optional">(optional)</span>
          <span v-else-if="param.required" class="required">*</span>
        </td>
        <td><code class="type">{{ param.type }}</code></td>
        <td>
          <span v-if="param.direction" class="dir-badge" :class="getDirClass(param.direction)">
            {{ getDirLabel(param.direction) }}
          </span>
        </td>
        <td>
          {{ param.description }}
          <span v-if="param.default" class="default">
            Default: <code>{{ param.default }}</code>
          </span>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.type {
  color: var(--api-type-color);
}

.default {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: var(--vp-c-text-3);
}
</style>
