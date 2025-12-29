# C API Documentation Template

This document defines the template structure for C API documentation pages.

---

## Template Structure

每个 C API 模块文档页面应包含以下部分：

### 1. 页面头部 (Frontmatter)

```yaml
---
title: Module Name
description: Brief description of the module
---
```

### 2. 模块概览

```markdown
# scl_module

Brief description of what this module provides.

<SupportMatrix :features="[...]" />
```

### 3. 每个函数的文档结构

```markdown
## scl_module_function

<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>

Brief one-line description.

<ApiSignature return-type="scl_error_t" name="scl_module_function">
  <span class="type">scl_sparse_t</span> <span class="param-name">matrix</span>,
  <span class="keyword">const</span> <span class="type">scl_real_t</span>* <span class="param-name">input</span>,
  <span class="type">scl_size_t</span> <span class="param-name">size</span>,
  <span class="type">scl_real_t</span>* <span class="param-name">output</span>
</ApiSignature>

### Parameters

<ParamTable :params="[...]" />

### Returns

### Errors

### Thread Safety

### Complexity (if applicable)

### Notes (if applicable)

<SourceLink file="scl/binding/c_api/module.h" :line="42" />
```

---

## Codegen 数据结构

供 codegen 使用的 JSON 数据结构：

```typescript
interface CApiFunction {
  // 基本信息
  name: string                    // 函数名: "scl_algebra_spmv"
  module: string                  // 模块名: "algebra"
  brief: string                   // 一行描述

  // 版本和状态
  since?: string                  // 版本号: "0.4.0"
  status: 'stable' | 'beta' | 'experimental' | 'deprecated'
  deprecated_use?: string         // 如果已弃用，替代函数名

  // 签名
  return_type: string             // 返回类型: "scl_error_t"
  params: CApiParam[]

  // 错误
  errors: CApiError[]

  // 复杂度 (可选)
  time_complexity?: string        // "O(nnz)"
  space_complexity?: string       // "O(n)"

  // 线程安全
  thread_safety: 'safe' | 'unsafe' | 'conditional'
  thread_safety_note?: string     // 条件说明

  // 源码位置
  source_file: string             // "scl/binding/c_api/algebra.h"
  source_line: number

  // 额外说明
  notes?: string[]
  preconditions?: string[]
  postconditions?: string[]
}

interface CApiParam {
  name: string                    // 参数名
  type: string                    // C 类型
  direction: 'in' | 'out' | 'inout'
  nullable: boolean               // 是否可为 NULL
  description: string
  default?: string                // 默认值 (如果有)
}

interface CApiError {
  code: string                    // "SCL_ERROR_NULL_POINTER"
  condition: string               // "If matrix is NULL"
}
```

---

## Jinja2 模板

供 codegen 使用的 Jinja2 模板：

```text
{# templates/c_api_function.md.j2 #}
## {{ func.name }}

{% if func.since %}<Badge type="version">{{ func.since }}</Badge>{% endif %}
{% if func.status == 'stable' %}<Badge type="status" color="green">Stable</Badge>
{% elif func.status == 'beta' %}<Badge type="status" color="yellow">Beta</Badge>
{% elif func.status == 'experimental' %}<Badge type="status" color="red">Experimental</Badge>
{% elif func.status == 'deprecated' %}<Badge type="status" color="red">Deprecated</Badge>{% endif %}

{{ func.brief }}

{% if func.status == 'deprecated' and func.deprecated_use %}
<Deprecated since="{{ func.since }}" use="{{ func.deprecated_use }}">
This function is deprecated and will be removed in a future version.
</Deprecated>
{% endif %}

<ApiSignature return-type="{{ func.return_type }}" name="{{ func.name }}">
{% for param in func.params %}
  {% if param.type.startswith('const') %}<span class="keyword">const</span> {% endif %}<span class="type">{{ param.type | replace('const ', '') }}</span> <span class="param-name">{{ param.name }}</span>{% if not loop.last %},{% endif %}

{% endfor %}
</ApiSignature>

### Parameters

<ParamTable :params="[
{% for param in func.params %}
  { name: '{{ param.name }}', type: '{{ param.type }}', dir: '{{ param.direction }}', description: '{{ param.description }}'{% if param.nullable %}, required: false{% endif %}{% if param.default %}, default: '{{ param.default }}'{% endif %} }{% if not loop.last %},{% endif %}

{% endfor %}
]" />

### Returns

`{{ func.return_type }}` - `SCL_OK` on success, error code otherwise.

### Errors

| Code | Condition |
|------|-----------|
{% for err in func.errors %}
| `{{ err.code }}` | {{ err.condition }} |
{% endfor %}

### Thread Safety

{% if func.thread_safety == 'safe' %}
<Badge color="green">Thread Safe</Badge> This function is safe to call from multiple threads.
{% elif func.thread_safety == 'unsafe' %}
<Badge color="red">Not Thread Safe</Badge> This function must not be called concurrently on the same data.
{% else %}
<Badge color="yellow">Conditionally Safe</Badge> {{ func.thread_safety_note }}
{% endif %}

{% if func.time_complexity or func.space_complexity %}
### Complexity

{% if func.time_complexity %}<Badge type="complexity">Time: {{ func.time_complexity }}</Badge>{% endif %}
{% if func.space_complexity %}<Badge type="complexity">Space: {{ func.space_complexity }}</Badge>{% endif %}
{% endif %}

{% if func.notes %}
### Notes

{% for note in func.notes %}
- {{ note }}
{% endfor %}
{% endif %}

<SourceLink file="{{ func.source_file }}" :line="{{ func.source_line }}" />

---
```

---

## 模块索引页模板

```text
{# templates/c_api_module.md.j2 #}
---
title: {{ module.name }}
description: {{ module.description }}
---

# {{ module.name }}

{{ module.description }}

## Overview

<SupportMatrix :features="[
{% for func in module.functions %}
  { name: '{{ func.name }}', {% for backend in ['numpy', 'sparse', 'dask', 'gpu'] %}{{ backend }}: {{ func.support[backend] | default('false') }}{% if not loop.last %}, {% endif %}{% endfor %} }{% if not loop.last %},{% endif %}

{% endfor %}
]" />

## Functions

{% for func in module.functions %}
{% include 'c_api_function.md.j2' %}
{% endfor %}

## See Also

<SeeAlso :links="[
{% for related in module.related_modules %}
  { href: '/api/c-api/{{ related }}', text: '{{ related }}' }{% if not loop.last %},{% endif %}

{% endfor %}
]" />
```
