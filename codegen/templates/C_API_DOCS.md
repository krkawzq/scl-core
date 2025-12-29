# C API Documentation Codegen Guide

本文档定义了 C API 文档的自动生成规范，供 codegen agent 使用。

---

## 1. 目标输出

为每个 C API 模块生成一个 Markdown 文件，位于 `docs/api/c-api/<module>.md`。

**示例输出**: `docs/api/c-api/algebra.md`

---

## 2. 数据结构定义

### 2.1 模块结构

```python
@dataclass
class CApiModule:
    name: str                      # 模块名: "algebra"
    description: str               # 模块描述
    header_file: str               # 头文件路径: "scl/binding/c_api/algebra.h"
    functions: list[CApiFunction]  # 函数列表
    related_modules: list[str]     # 相关模块: ["sparse", "normalize"]
```

### 2.2 函数结构

```python
@dataclass
class CApiFunction:
    # 基本信息
    name: str                      # 函数名: "scl_algebra_spmv"
    brief: str                     # 一行描述: "Sparse matrix-vector multiplication"

    # 版本和状态
    since: str | None              # 版本号: "0.4.0" (可选)
    status: Literal["stable", "beta", "experimental", "deprecated"]
    deprecated_use: str | None     # 如已弃用，替代函数名

    # 签名
    return_type: str               # 返回类型: "scl_error_t"
    params: list[CApiParam]        # 参数列表

    # 错误
    errors: list[CApiError]        # 可能的错误

    # 复杂度 (可选)
    time_complexity: str | None    # "O(nnz)"
    space_complexity: str | None   # "O(1)"

    # 线程安全
    thread_safety: Literal["safe", "unsafe", "conditional"]
    thread_safety_note: str | None # 条件说明 (当 thread_safety="conditional" 时)

    # 源码位置
    source_file: str               # "scl/binding/c_api/algebra.h"
    source_line: int               # 42

    # 额外信息 (可选)
    notes: list[str] | None        # 注意事项
    formula: str | None            # LaTeX 公式: "y = \\alpha \\cdot A \\cdot x + \\beta \\cdot y"
```

### 2.3 参数结构

```python
@dataclass
class CApiParam:
    name: str                      # 参数名: "matrix"
    type: str                      # C 类型: "scl_sparse_t" 或 "const scl_real_t*"
    direction: Literal["in", "out", "inout"]
    description: str               # 参数描述
    nullable: bool = False         # 是否可为 NULL
    default: str | None = None     # 默认值描述 (如有)
```

### 2.4 错误结构

```python
@dataclass
class CApiError:
    code: str                      # "SCL_ERROR_NULL_POINTER"
    condition: str                 # "If matrix is NULL"
```

---

## 3. 解析规则

### 3.1 从头文件提取信息

**函数声明格式**:
```c
/// @brief General SpMV: y = alpha * A * x + beta * y
/// @param[in] A Sparse matrix (non-null, CSR or CSC)
/// @param[in] x Input vector [secondary_dim] (non-null)
/// @param[in,out] y Output vector [primary_dim] (non-null)
/// @param[in] alpha Scaling factor for A*x
/// @param[in] beta Scaling factor for y
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_algebra_spmv(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
);
```

**提取规则**:

| 字段 | 来源 |
|------|------|
| `name` | 函数名 |
| `brief` | `@brief` 注释 |
| `params[].direction` | `@param[in]` / `@param[out]` / `@param[in,out]` |
| `params[].description` | `@param` 后的描述文本 |
| `return_type` | 函数返回类型 |
| `source_line` | 函数声明行号 |

### 3.2 模块名提取

从函数名提取: `scl_<module>_<operation>` → module = `algebra`

### 3.3 状态推断

- 默认: `stable`
- 如果在 `deprecated.h` 中或有 `@deprecated` 注释: `deprecated`
- 如果在 `experimental/` 目录: `experimental`

### 3.4 线程安全推断

根据函数特征:
- 只读操作 (无 `[out]` 或 `[inout]` 参数到矩阵): `safe`
- 修改矩阵数据: `unsafe`
- 其他: `conditional`

### 3.5 错误推断

根据参数类型自动添加常见错误:
- 有指针参数 → `SCL_ERROR_NULL_POINTER`
- 有 size 参数 → `SCL_ERROR_DIMENSION_MISMATCH`
- 有矩阵参数 → 可能的格式错误

---

## 4. Jinja2 模板

### 4.1 模块页面模板

```text
{# c_api_module.md.j2 #}
---
title: scl_{{ module.name }}
description: {{ module.description }}
---

# scl_{{ module.name }}

{{ module.description }}

## Overview

<SupportMatrix :features="[
{% for func in module.functions %}
  { name: '{{ func.name }}', numpy: true, sparse: true, dask: false, gpu: false }{% if not loop.last %},{% endif %}

{% endfor %}
]" />

---

{% for func in module.functions %}
{% include 'c_api_function.md.j2' %}

---

{% endfor %}

## See Also

<SeeAlso :links="[
{% for related in module.related_modules %}
  { href: '/api/c-api/{{ related }}', text: 'scl_{{ related }}' }{% if not loop.last %},{% endif %}

{% endfor %}
]" />
```

### 4.2 函数文档模板

```text
{# c_api_function.md.j2 #}
## {{ func.name }}

{% if func.since %}<Badge type="version">{{ func.since }}</Badge> {% endif %}
{% if func.status == 'stable' %}<Badge type="status" color="green">Stable</Badge>
{% elif func.status == 'beta' %}<Badge type="status" color="yellow">Beta</Badge>
{% elif func.status == 'experimental' %}<Badge type="status" color="red">Experimental</Badge>
{% elif func.status == 'deprecated' %}<Badge type="status" color="red">Deprecated</Badge>{% endif %}

{{ func.brief }}
{% if func.formula %}

$${{ func.formula }}$$
{% endif %}

{% if func.status == 'deprecated' and func.deprecated_use %}
<Deprecated since="{{ func.since }}" use="{{ func.deprecated_use }}">
This function is deprecated and will be removed in a future version.
</Deprecated>

{% endif %}
<ApiSignature return-type="{{ func.return_type }}" name="{{ func.name }}">
{% for param in func.params %}
  {% if 'const ' in param.type %}<span class="keyword">const</span> {% endif %}<span class="type">{{ param.type | replace('const ', '') }}</span> <span class="param-name">{{ param.name }}</span>{% if not loop.last %},{% endif %}

{% endfor %}
</ApiSignature>

### Parameters

<ParamTable :params="[
{% for param in func.params %}
  { name: '{{ param.name }}', type: '{{ param.type }}', dir: '{{ param.direction }}', description: '{{ param.description | escape_quotes }}'{% if param.nullable %}, required: false{% endif %}{% if param.default %}, default: '{{ param.default }}'{% endif %} }{% if not loop.last %},{% endif %}

{% endfor %}
]" />

### Returns

`{{ func.return_type }}` — `SCL_OK` on success, error code otherwise.

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
<Badge color="red">Not Thread Safe</Badge> This function modifies data in-place and must not be called concurrently on the same data.
{% else %}
<Badge color="yellow">Conditionally Safe</Badge> {{ func.thread_safety_note }}
{% endif %}

{% if func.time_complexity or func.space_complexity %}
### Complexity

{% if func.time_complexity %}<Badge type="complexity">Time: {{ func.time_complexity }}</Badge> {% endif %}
{% if func.space_complexity %}<Badge type="complexity">Space: {{ func.space_complexity }}</Badge>{% endif %}

{% endif %}
{% if func.notes %}
### Notes

{% for note in func.notes %}
- {{ note }}
{% endfor %}

{% endif %}
<SourceLink file="{{ func.source_file }}" :line="{{ func.source_line }}" />
```

---

## 5. 可用组件

文档中可使用以下 Vue 组件:

### 5.1 Badge (徽章)

```html
<Badge type="version">0.4.0</Badge>
<Badge type="status" color="green">Stable</Badge>
<Badge type="status" color="yellow">Beta</Badge>
<Badge type="status" color="red">Deprecated</Badge>
<Badge type="complexity">O(n)</Badge>
<Badge color="green">Thread Safe</Badge>
<Badge color="red">Not Thread Safe</Badge>
```

**Props**:
- `type`: `"version"` | `"status"` | `"complexity"` | `"custom"`
- `color`: `"default"` | `"green"` | `"yellow"` | `"red"` | `"blue"` | `"purple"`

### 5.2 ApiSignature (函数签名)

```html
<ApiSignature return-type="scl_error_t" name="scl_algebra_spmv">
  <span class="type">scl_sparse_t</span> <span class="param-name">A</span>,
  <span class="keyword">const</span> <span class="type">scl_real_t</span>* <span class="param-name">x</span>
</ApiSignature>
```

**Props**:
- `return-type`: 返回类型字符串
- `name`: 函数名
- `:template`: 模板参数数组 (C++ 用)
- `:deprecated`: 是否已弃用

**Slot 中可用的 CSS 类**:
- `.type` — 类型名 (蓝色)
- `.keyword` — 关键字如 const (粉色)
- `.param-name` — 参数名 (灰色)
- `.default-value` — 默认值 (浅灰色)

### 5.3 ParamTable (参数表)

```html
<ParamTable :params="[
  { name: 'A', type: 'scl_sparse_t', dir: 'in', description: 'Sparse matrix handle' },
  { name: 'x', type: 'const scl_real_t*', dir: 'in', description: 'Input vector' },
  { name: 'y', type: 'scl_real_t*', dir: 'out', description: 'Output vector' },
  { name: 'alpha', type: 'scl_real_t', dir: 'in', description: 'Scale factor', default: '1.0' }
]" />
```

**Param 对象字段**:
- `name`: 参数名 (必须)
- `type`: 类型 (必须)
- `dir`: `"in"` | `"out"` | `"inout"` (必须)
- `description`: 描述 (必须)
- `required`: 是否必须，默认 true
- `default`: 默认值

### 5.4 SupportMatrix (支持矩阵)

```html
<SupportMatrix :features="[
  { name: 'scl_algebra_spmv', numpy: true, sparse: true, dask: false, gpu: false },
  { name: 'scl_algebra_spmm', numpy: true, sparse: true, dask: 'partial', gpu: true }
]" />
```

**Feature 对象字段**:
- `name`: 函数名
- `numpy`: `true` | `false` | `"partial"`
- `sparse`: `true` | `false` | `"partial"`
- `dask`: `true` | `false` | `"partial"`
- `gpu`: `true` | `false` | `"partial"`

渲染为: ✓ (true) / ◐ (partial) / ✗ (false)

### 5.5 Callout (提示框)

```html
<Callout type="info" title="Optional Title">
Content here...
</Callout>

<Callout type="warning">
Warning message...
</Callout>
```

**Props**:
- `type`: `"info"` | `"tip"` | `"warning"` | `"danger"` | `"note"`
- `title`: 可选标题
- `collapsible`: 是否可折叠

### 5.6 Deprecated (弃用警告)

```html
<Deprecated since="0.4.0" use="scl_new_function">
This function has performance issues.
</Deprecated>
```

**Props**:
- `since`: 弃用版本
- `use`: 替代函数名

### 5.7 SeeAlso (相关链接)

```html
<SeeAlso :links="[
  { href: '/api/c-api/sparse', text: 'scl_sparse' },
  { href: '/api/c-api/normalize', text: 'scl_normalize' }
]" />
```

### 5.8 SourceLink (源码链接)

```html
<SourceLink file="scl/binding/c_api/algebra.h" :line="42" />
```

**Props**:
- `file`: 相对于仓库根目录的文件路径
- `line`: 行号 (可选)

---

## 6. 示例数据

### 6.1 输入 JSON 示例

```json
{
  "module": {
    "name": "algebra",
    "description": "High-performance sparse linear algebra kernels",
    "header_file": "scl/binding/c_api/algebra.h",
    "related_modules": ["sparse", "normalize", "scale"]
  },
  "functions": [
    {
      "name": "scl_algebra_spmv",
      "brief": "Sparse matrix-vector multiplication",
      "formula": "y = \\alpha \\cdot A \\cdot x + \\beta \\cdot y",
      "since": "0.4.0",
      "status": "stable",
      "return_type": "scl_error_t",
      "params": [
        {
          "name": "A",
          "type": "scl_sparse_t",
          "direction": "in",
          "description": "Sparse matrix handle (CSR or CSC format)",
          "nullable": false
        },
        {
          "name": "x",
          "type": "const scl_real_t*",
          "direction": "in",
          "description": "Input vector of size [secondary_dim]",
          "nullable": false
        },
        {
          "name": "x_size",
          "type": "scl_size_t",
          "direction": "in",
          "description": "Size of input vector",
          "nullable": false
        },
        {
          "name": "y",
          "type": "scl_real_t*",
          "direction": "inout",
          "description": "Output vector of size [primary_dim]",
          "nullable": false
        },
        {
          "name": "y_size",
          "type": "scl_size_t",
          "direction": "in",
          "description": "Size of output vector",
          "nullable": false
        },
        {
          "name": "alpha",
          "type": "scl_real_t",
          "direction": "in",
          "description": "Scaling factor for A·x",
          "nullable": false,
          "default": "1.0"
        },
        {
          "name": "beta",
          "type": "scl_real_t",
          "direction": "in",
          "description": "Scaling factor for y",
          "nullable": false,
          "default": "0.0"
        }
      ],
      "errors": [
        {
          "code": "SCL_ERROR_NULL_POINTER",
          "condition": "If `A`, `x`, or `y` is NULL"
        },
        {
          "code": "SCL_ERROR_DIMENSION_MISMATCH",
          "condition": "If vector sizes don't match matrix dimensions"
        }
      ],
      "time_complexity": "O(nnz)",
      "space_complexity": "O(1)",
      "thread_safety": "safe",
      "source_file": "scl/binding/c_api/algebra.h",
      "source_line": 42,
      "notes": [
        "For CSR format: primary = rows, secondary = cols",
        "For CSC format: primary = cols, secondary = rows"
      ]
    }
  ]
}
```

---

## 7. 输出文件结构

```
docs/api/c-api/
├── index.md          # 手动维护的索引页
├── algebra.md        # 自动生成
├── annotation.md     # 自动生成
├── bbknn.md          # 自动生成
├── centrality.md     # 自动生成
├── clonotype.md      # 自动生成
├── coexpression.md   # 自动生成
├── communication.md  # 自动生成
├── comparison.md     # 自动生成
├── components.md     # 自动生成
├── core.md           # 自动生成
├── correlation.md    # 自动生成
├── dense.md          # 自动生成
├── diffusion.md      # 自动生成
├── doublet.md        # 自动生成
├── enrichment.md     # 自动生成
├── entropy.md        # 自动生成
├── feature.md        # 自动生成
├── gnn.md            # 自动生成
├── gram.md           # 自动生成
├── grn.md            # 自动生成
├── group.md          # 自动生成
├── hotspot.md        # 自动生成
├── hvg.md            # 自动生成
├── impute.md         # 自动生成
├── kernel.md         # 自动生成
├── leiden.md         # 自动生成
├── lineage.md        # 自动生成
├── log1p.md          # 自动生成
├── louvain.md        # 自动生成
├── markers.md        # 自动生成
├── merge.md          # 自动生成
├── metrics.md        # 自动生成
├── mmd.md            # 自动生成
├── mwu.md            # 自动生成
├── neighbors.md      # 自动生成
├── niche.md          # 自动生成
├── normalize.md      # 自动生成
├── outlier.md        # 自动生成
├── permutation.md    # 自动生成
├── projection.md     # 自动生成
├── propagation.md    # 自动生成
├── pseudotime.md     # 自动生成
├── qc.md             # 自动生成
├── reorder.md        # 自动生成
├── resample.md       # 自动生成
├── sampling.md       # 自动生成
├── scale.md          # 自动生成
├── scoring.md        # 自动生成
├── slice.md          # 自动生成
├── softmax.md        # 自动生成
├── sparse.md         # 自动生成
├── spatial.md        # 自动生成
├── state.md          # 自动生成
├── subpopulation.md  # 自动生成
├── tissue.md         # 自动生成
├── transition.md     # 自动生成
├── ttest.md          # 自动生成
├── velocity.md       # 自动生成
└── stat/             # 统计模块子目录
    ├── auroc.md
    ├── effect_size.md
    ├── group_partition.md
    └── kruskal_wallis.md
```

---

## 8. Codegen 命令

```bash
# 生成所有 C API 文档
python -m codegen c-api-docs

# 生成单个模块
python -m codegen c-api-docs --module algebra

# 验证生成的文档
make docs-build
```

---

## 9. 注意事项

1. **转义**: JSON 中的引号和特殊字符需要转义
2. **LaTeX**: 公式中的反斜杠需要双重转义 (`\\alpha`)
3. **顺序**: 函数按在头文件中出现的顺序排列
4. **一致性**: 所有模块使用相同的模板结构
5. **版本**: 如果无法确定版本，使用当前版本号
6. **复杂度**: 如果无法推断，留空不显示该部分
