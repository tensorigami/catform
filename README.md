# catform

**Categorical form** — inspired by [category theory](https://en.wikipedia.org/wiki/Category_theory), **catform** for short — is a domain-specific language for tensor computation. Six primitive operations and one derived form (contraction) suffice to express the full computation of any modern transformer-based language model — and this set is closed under automatic differentiation: the derivative of any catform program is itself a catform program.

A `.cat` file is a standalone, plain-text artifact that states *what* a tensor program computes, without prescribing *how*. For example, softmax over the last axis:

```catform
softmax(x: f32[N, D]) -> (y: f32[N, D]) {
  max_  : f32[N, 1] = fold["... N D -> ... N 1", max](x)
  max   : f32[N, D] = tile["... N 1 -> ... N D"](max_)
  shift : f32[N, D] = map[sub](x, max)
  e     : f32[N, D] = map[exp](shift)
  sum_  : f32[N, 1] = fold["... N D -> ... N 1", sum](e)
  sum   : f32[N, D] = tile["... N 1 -> ... N D"](sum_)
  y     : f32[N, D] = map[div](e, sum)
}
```

Every line has the form `name : type = op[specifiers](args)`. The types are concrete (`f32`, `bf16`, `int32`) with named dimensions (`N`, `D`). The specifiers — string patterns inspired by [einops](https://einops.rocks/) — make every axis transformation explicit.

## Operations

| Op | What it does | Specifier |
|---|---|---|
| **view** | Rearrange axes (transpose, reshape, squeeze) | pattern: `"a b -> b a"` |
| **map** | Elementwise function lift | function: `mul`, `exp`, `silu` |
| **fold** | Reduce an axis (sum, mean, max) | pattern + reduction: `"... N D -> ... N 1", sum` |
| **tile** | Replicate along an axis | pattern: `"1 b -> a b"` |
| **gather** | Data-dependent read: `out[i] = data[idx[i]]` | pattern: `"v d -> _ d"` |
| **scatter** | Data-dependent write: `out[idx[i]] += data[i]` | pattern: `"_ d -> v d"` |
| **contract** | Bilinear contraction (dot, matmul, einsum) | pattern: `"n d, d k -> n k"` |

Contract is a derived form — it decomposes into tile, map, fold, view — but it represents [tensor contraction](https://en.wikipedia.org/wiki/Tensor_contraction), the fundamental operation of linear algebra, and comprises the vast majority of computation in any language model. We give it native status.

Two composition constructs — `call` (function application) and `loop` (iterated application) — allow modular `.cat` files to be written as multiple functions. Both are eliminated by **flattening**, which inlines everything into a single straight-line `main`. They are structural, not computational: after flattening, every line is one of the seven operations above.

## Toolchain

This crate provides:

- **parse** — `.cat` source → AST
- **resolve** — substitute `param.X` names with config values, infer axis sizes
- **flatten** — inline `call` / `loop` into a single flat function
- **fmt** — AST → canonical `.cat` source (inverse of parse)
- **check** — type-check shape and dtype consistency

The AST serializes to JSON via serde, making it consumable from any language.

## Usage

```rust
use catform::{parse, resolve, flatten, fmt, check, config};

let module = parse::parse_file("model.cat");
let cfg = config::load_config("config.toml");
let resolved = resolve::resolve(&module, &cfg);
let flat = flatten::flatten(&resolved, "main");

// type check
let errors = check::check(&flat);

// format back to .cat
let source = fmt::format_cat(&flat, 100);
```

## Context

catform is introduced in the [Tensors](https://tensorigami.github.io/pianola/02_tensors.html) chapter of the textbook [*Structure and Execution of Language Models*](https://tensorigami.github.io/pianola/), which develops the mathematics of tensor computation in executable catform. The [Model](https://tensorigami.github.io/pianola/03_model.html) chapter walks through every operation in a production language model ([Qwen3](https://qwen.ai/blog?id=qwen3)) expressed as a single `.cat` file.

The companion software [**Pianola**](https://github.com/tensorigami/pianola) is the execution engine — it reads `.cat` files and runs them with PyTorch or JAX.
