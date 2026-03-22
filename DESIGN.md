# Design

Catform is a first-order language for specifying tensor computations. A
`.cat` file is a pure declarative artifact --- it describes *what* to
compute without prescribing *how*. An external caller (an engine, a
compiler, a training loop) consumes the `.cat` file and executes it.

This document specifies the abstraction boundary between catform and its
caller.

---

## The boundary

A `.cat` file is a lambda with free variables. The caller provides an
environment that binds them. The boundary is the **constraints struct** ---
an environment signature that catform infers mechanically from the code:

```rust
Constraints {
    params:  BTreeSet<String>,                               // param.X names needed
    weights: BTreeMap<String, BTreeMap<String, TensorType>>,  // fn → { path → type }
}
```

Catform emits it. The caller satisfies it. Neither knows the other's
internals.

```
$ uv run main.py check models/qwen3/model.cat

  params: param.attn_scale, param.ffn_dim, param.head_dim, ...
  layer: weights.attn.q : bf16[param.qd × param.hidden]
  layer: weights.ffn.gate : bf16[param.ffn_dim × param.hidden]
  main: weights.embedding : bf16[param.vocab × param.hidden]
  transformer: weights.norm : bf16[param.hidden]
```

---

## What catform owns

**One artifact:** the `.cat` file.

**Names are opaque strings.** `param.hidden`, `weights.attn.q`, and `N` are
all just strings from catform's perspective. Catform never interprets the
prefixes or assigns meaning to the dotted structure. The caller decides what
names mean.

**Types are written at the point of use.** A function parameter `weights: *`
carries no type information. But when that parameter is accessed and passed
to a typed consumption site --- `attention(... q: bf16[param.qd, param.hidden])`
--- the expected type is fully determined. The checker traces every `*`
access to its leaf consumption site and records the constraint. This is
**deferred typing**: the types exist; they are declared where the value is
used, not where it enters.

**The call graph is static.** `call[f]` and `loop[f]` reference functions
by name, not by variable. The entire call graph is known at parse time. This
is what makes constraint inference possible --- every weight access can be
traced statically. It also enables flattening (any first-order program
without recursion can be expanded into a straight-line program) and autodiff
closure (the derivative of a straight-line catform program is itself a
catform program).

**Catform does:**
- Parse
- Validate (syntax, scoping, SSA)
- Type check (symbolic, against function signatures)
- Emit constraints (environment signature)
- Flatten (given iteration count from caller)
- Format (pretty-print `.cat` files)

---

## What the caller owns

**One artifact:** a configuration file (e.g. `config.toml`).

The caller decides what names mean:
- `param.X` → a value from configuration (a dimension, a scalar, a vector)
- `weights.X` → a path into tensor storage (safetensors, checkpoints, etc.)

The tensor storage files are an implementation detail of the caller. Catform
never sees them.

**The caller does:**
- Build a name dict from configuration + tensor storage
- Provide the iteration count for loop unrolling (derived from weight structure)
- Resolve names in the flat program (substituting the name dict)
- Verify constraints against actual tensor shapes

---

## Flattening

Flattening lowers a structured program (calls + loops) into a straight-line
program --- a flat list of primitive ops with no control flow. This is
needed for execution and for autodiff.

Flattening is a single pass that eliminates both `call` and `loop`. The two
cannot be separated: calls live inside loop bodies, so loops must be
unrolled before calls within them can be inlined. Since unrolling requires
the iteration count (external info from the caller), flattening necessarily
crosses the boundary.

The iteration count comes from the caller, not from the `.cat` file. The
`.cat` file says "loop over layers." The caller says "there are 28 of
them." The architecture does not hardcode the count.

---

## Design principles

- **Seven primitive operations** --- small, orthogonal, named for what they
  do. Closed under differentiation: the derivative of any catform program
  is itself a catform program.
- **Code is the spec** --- the `.cat` file is a standalone mathematical
  artifact, not a comment beside an implementation.
- **First-order** --- load-bearing, not limiting. Enables constraint
  inference, flattening, and autodiff closure.
- **Deferred typing** --- `*` parameters carry no type annotation, but
  every leaf access is fully typed at its consumption site. The checker
  infers the structural type.
- **Environment signature** --- inferred by the checker, not maintained by
  hand. The constraints struct is the formal contract between catform and
  its caller.
- **Explicit over clever** --- every dependency is visible in the function
  signature. No implicit state, no spooky action at a distance.
