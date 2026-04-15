# TODO

## Per-function type checking

Currently the checker runs *after* flattening, which means function signatures
are erased before checking. A function declared with `bf16[N, param.hidden]`
will silently accept a 3D input like `bf16[N, heads, head_dim]` because the
ops inside use `...` patterns that allow any leading axes.

Fix: check each function's body against its declared signature *before*
flattening, so signature mismatches at call sites become type errors.

## Loop primitive: pure textual unrolling

`loop[f, n](args)` should be a pure text rewriter:
- arg matches an output name → threaded (SSA renamed `_1`, `_2`, ...)
- arg ends with `.*` → indexed (`.0`, `.1`, ...)
- everything else → static

Currently the flattener requires at least one threaded arg
(`expect("Loop must have a threaded arg matching output name")`). It should
not. A loop with no threading (e.g. `y = loop[f, 3](x)`) is valid: it
unrolls to three independent calls with SSA renaming on `y`.

Tuple outputs in loops should also work via the same SSA renaming rule.
