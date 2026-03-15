# catform

**Categorical form**---inspired by [category theory](https://en.wikipedia.org/wiki/Category_theory), **catform** for short---is a domain specific language for specifying tensor programs. Its op set---just six primitive operations and one derived form---were chosen to be a minimal set of tensor operations that suffice to express the full computation of modern transformer language models, while also remaining closed under automatic differentiation.

A `.cat` file is a standalone, plain-text description that exclusively states *what* a tensor program computes, without prescribing how. For example, softmax over the last axis of a tensor:

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

## Operations

Every function body line in a `.cat` module has the form `name: type = op[specifiers](args)`. The seven operation types:

| Op | What it does | Specifier |
|---|---|---|
| **view** | Rearrange axes (transpose, reshape, squeeze) | pattern: `"a b -> b a"` |
| **map** | Elementwise function lift | function: `mul`, `exp`, `silu` |
| **fold** | Reduce an axis (sum, mean, max) | pattern + reduction: `"... N -> ... 1", sum` |
| **tile** | Replicate along an axis | pattern: `"1 b -> a b"` |
| **gather** | Data-dependent read: `out[i] = data[idx[i]]` | pattern: `"v d -> _ d"` |
| **scatter** | Data-dependent write: `out[idx[i]] += data[i]` | pattern: `"_ d -> v d"` |
| **contract** | Bilinear contraction (dot, matmul, einsum) | pattern: `"n m, m k -> n k"` |

Contract is a derived form---it decomposes into tile, map, fold, view---but it represents the fundamental linear algebra concept of [tensor contraction](https://en.wikipedia.org/wiki/Tensor_contraction), while also comprising the vast majority of computation in any language model, so we give it native status.

## Toolchain

This crate provides:

- **parse** — `.cat` source → AST
- **resolve** — substitute `param.X` names with config values, infer axis sizes
- **flatten** — inline `call` / `loop` composition constructs into a single flat function
- **fmt** — AST → canonical `.cat` source (inverse of parse)
- **check** — type check shape and dtype consistency

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

The AST serializes to JSON via serde, making it consumable from any language.

## Context

The textbook [***Structure and Execution of Language Models***](https://tensorigami.github.io/pianola) introduces the math and computation of language models. Its companion software [***Pianola***](https://github.com/tensorigami/pianola) runs popular open weights models expressed in catform.
