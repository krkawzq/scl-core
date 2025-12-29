<script setup lang="ts">
interface Param {
  name: string
  type: string
  dir?: 'in' | 'out' | 'inout'
  required?: boolean
  default?: string
  description: string
}

interface Props {
  params: Param[]
}

defineProps<Props>()

const getDirLabel = (dir?: string) => {
  switch (dir) {
    case 'in': return 'in'
    case 'out': return 'out'
    case 'inout': return 'in/out'
    default: return ''
  }
}
</script>

<template>
  <table class="scl-param-table">
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
          <span class="scl-param-table__name">
            {{ param.name }}
            <span v-if="param.required !== false" class="required">*</span>
          </span>
        </td>
        <td>
          <code class="scl-param-table__type">{{ param.type }}</code>
        </td>
        <td>
          <span
            v-if="param.dir"
            class="scl-param-table__dir"
            :class="`scl-param-table__dir--${param.dir}`"
          >
            {{ getDirLabel(param.dir) }}
          </span>
        </td>
        <td>
          {{ param.description }}
          <span v-if="param.default" class="scl-param-table__default">
            Default: <code>{{ param.default }}</code>
          </span>
        </td>
      </tr>
    </tbody>
  </table>
</template>
