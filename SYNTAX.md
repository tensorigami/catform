# Catform Syntax Reference

Ground truth for highlight.js (mdBook) and tree-sitter (editors).
Color is determined by **syntactic position**, not by content.

## Grammar

Every line in a function body has the form:

```
name : type = keyword[specifiers](args)
```

No exceptions. `keyword` is one of 10 keywords (7 ops + 2 composition + literal).
Types are `dtype[shape]`. Specifiers and args vary by keyword.

## The 7 color groups

Every visible token belongs to exactly one group.

### §1. Variable

Bound variables — function parameters, assignment outputs, op arguments.
Default text color (no highlight).

```
x, q_flat, scores, cos, sin, wq, pos
```

| hljs     | tree-sitter |
|----------|-------------|
| (none)   | `@variable` |

### §2. Keyword

The keyword after `=`, and the function name at column 0.
Same color — both name a computation.

**Op keywords (the 7 op types):**
```
view  map  fold  tile  gather  scatter  contract
```

**Composition keywords:**
```
call  loop
```

**Value keyword:**
```
literal
```

Also: arrow `->`, braces `{ }`.

| hljs      | tree-sitter |
|-----------|-------------|
| `keyword` | `@keyword`  |

### §3. Dtype

The type keyword in a type annotation (after `:`).

```
: bf16[...]   : f32[...]   : int32[...]
```

| hljs   | tree-sitter    |
|--------|----------------|
| `type` | `@type.builtin`|

### §4. Shape dimension

Identifiers inside shape brackets `[...]` within a type annotation.

```
[N, param.hidden]   [N, param.rope_dim]   [2, 2]
```

Note: `param.X` inside shapes keeps its own coloring (§5).

| hljs   | tree-sitter |
|--------|-------------|
| `meta` | `@type`     |

### §5. Name prefix

The first segment of any dotted name (`prefix.path`).
All dotted names are opaque to catform — `param.X`, `weights.X`, `foo.bar.baz`
get identical treatment: prefix → §5 color, path segments → secondary.

```
param.hidden   weights.attn.q   weights.layer
^^^^^          ^^^^^^^          ^^^^^^^         ← prefix (§5)
      ^^^^^^          ^^^^  ^         ^^^^^     ← path (secondary)
```

| hljs   | tree-sitter  |
|--------|--------------|
| `attr` | `@attribute` |

### §6. Pattern string

Quoted einops pattern string inside specifier brackets.

```
["a b -> b a"]   ["... n d, f d -> ... n f"]
```

| hljs     | tree-sitter |
|----------|-------------|
| `string` | `@string`   |

### §7. Specifier function

Unquoted identifier inside `[...]` that is not a keyword.
The element function, reduction, or callee name.

**In ops:**
```
[mul]  [add]  [silu]  [mean]  [max]  [softmax]  [ge]  [where]
[f32]  [bf16]                                    ← cast ops
```

**In call/loop:**
```
[rmsnorm]  [rope_rotate]  [attention]  [layer]
```

| hljs       | tree-sitter        |
|------------|--------------------|
| `built_in` | `@function.builtin`|

## Secondary

Highlighted but don't need a distinct color group.

| Element          | hljs      | tree-sitter        | Notes                              |
|------------------|-----------|--------------------|-----------------------------------:|
| Comment          | `comment` | `@comment`         | `// ...`                           |
| Number           | `number`  | `@number`          | `1`, `2`, `0.5`, `-inf`           |
| Name path        | `literal` | `@constant`        | segments after prefix dot          |
| Literal value    | `number`  | `@number`          | arg of `literal()`: `[[1,0],[0,1]]`|
| Spec keyword     | `attr`    | `@property`        | `over=`, `count=`                  |
| Punctuation      | various   | various            | `()[]{},:=.`                       |

## Full keyword × specifier × arg table

| Keyword    | Specifiers                       | Args           |
|------------|----------------------------------|----------------|
| `view`     | `[pattern]`                      | `(input)`      |
| `map`      | `[fn]`                           | `(inputs...)`  |
| `fold`     | `[pattern, reduction]`           | `(input)`      |
| `tile`     | `[pattern]`                      | `(input)`      |
| `gather`   | `[pattern]`                      | `(data, index)`|
| `scatter`  | `[pattern, reduction]`           | `(data, index)`|
| `contract` | `[pattern]`                      | `(inputs...)`  |
| `call`     | `[function]`                     | `(args...)`    |
| `loop`     | `[function, over=path, count=N]` | `(args...)`    |
| `literal`  | (none)                           | `(value)`      |

## Tuple assignment

Functions may return multiple values via tuple assignment:

```
(cos, sin): (bf16[N, param.rope_dim], bf16[N, param.rope_dim]) = call[rope_setup](pos)
```
