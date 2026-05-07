use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::ast::*;

// ── Public API ──────────────────────────────────────────────────────

/// Flatten a module against a cache state.
///
/// `cache_keys` is the set of cache slot names currently populated (e.g.,
/// `{"mask", "rot"}`, or `{"kv_buf.0", ..., "kv_buf.27"}`). At each
/// `cache[var, ...]` site:
///   - if `var ∈ cache_keys`: warm path — for idempotent (2-arg), the
///     cache slot is bound via main's input parameter (no body op
///     emitted); for stateful (3-arg), `var = call[extend](var_prev,
///     args...)` is emitted.
///   - if `var ∉ cache_keys`: cold path — `var = call[init](args...)`
///     is emitted; var becomes a fresh output of main.
///
/// In every case, the cache slot is added to main's params (if warm) and/or
/// returns (if its value is produced this call). The single resulting flat
/// program has only the 7 raw op kinds.
pub fn flatten(m: &Module, entry: &str, cache_keys: &HashSet<String>) -> Module {
    let passthrough = collect_param_names(&m.functions);

    // Phase 1: expand. Cache ops survive as primitives; loop unrolling substitutes
    // `.*` in their var names per iteration.
    let (raw_ops, _) = expand(&m.functions, &passthrough, entry);

    // Phase 2: resolve cache ops against `cache_keys`. Each Cache op becomes
    // either a passthrough (warm idempotent), an extend call (warm stateful),
    // or an init call (cold). Synthesized calls get inlined here.
    let mut ctx = CacheCtx::new(cache_keys);
    let resolved_ops = resolve_caches(&m.functions, &passthrough, raw_ops, &mut ctx);

    let entry_fn = &m.functions[entry];

    let mut params = entry_fn.params.clone();
    for slot in &ctx.input_slots {
        params.push(slot.clone());
    }

    let mut returns = entry_fn.returns.clone();
    for slot in &ctx.output_slots {
        returns.push(slot.clone());
    }

    let flat_fn = Function {
        name: entry_fn.name.clone(),
        comments: entry_fn.comments.clone(),
        params,
        returns,
        ops: resolved_ops,
    };
    let mut functions = IndexMap::new();
    functions.insert(entry.to_string(), flat_fn);
    Module {
        header_comments: m.header_comments.clone(),
        functions,
    }
}

// ── Cache context ───────────────────────────────────────────────────

struct CacheCtx<'a> {
    cache_keys: &'a HashSet<String>,
    input_slots: Vec<Param>,
    output_slots: Vec<Param>,
    seen_input: HashSet<String>,
    seen_output: HashSet<String>,
}

impl<'a> CacheCtx<'a> {
    fn new(cache_keys: &'a HashSet<String>) -> Self {
        Self {
            cache_keys,
            input_slots: Vec::new(),
            output_slots: Vec::new(),
            seen_input: HashSet::new(),
            seen_output: HashSet::new(),
        }
    }

    fn record_input(&mut self, name: &str, ty: &TensorType) {
        if self.seen_input.insert(name.to_string()) {
            self.input_slots.push(Param {
                name: name.to_string(),
                ty: ty.clone(),
            });
        }
    }

    fn record_output(&mut self, name: &str, ty: &TensorType) {
        if self.seen_output.insert(name.to_string()) {
            self.output_slots.push(Param {
                name: name.to_string(),
                ty: ty.clone(),
            });
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn collect_param_names(functions: &IndexMap<String, Function>) -> HashSet<String> {
    let mut names = HashSet::new();
    for f in functions.values() {
        for op in &f.ops {
            for a in &op.args {
                if let Atom::Name(n) = a
                    && n.starts_with("param.")
                {
                    names.insert(n.clone());
                }
            }
        }
    }
    names
}

fn is_primitive(op: &Op) -> bool {
    // Cache survives expand as a primitive; phase 2 resolves it.
    !matches!(op.kind, OpKind::Call { .. } | OpKind::Loop { .. })
}

fn is_dict_param(ty: &TensorType) -> bool {
    ty.dtype == "*"
}

/// Resolve a name through exact renames, then dict (prefix) renames, then scope.
fn resolve_name(
    name: &str,
    rename: &HashMap<String, String>,
    dict_renames: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> String {
    if passthrough.contains(name) {
        return name.to_string();
    }
    if let Some(mapped) = rename.get(name) {
        return mapped.clone();
    }
    // Dict prefix renames: if name starts with "w." and "w" → "X.Y",
    // then "w.foo" → "X.Y.foo"
    for (old_prefix, new_prefix) in dict_renames {
        if name == old_prefix {
            return new_prefix.clone();
        }
        if let Some(suffix) = name.strip_prefix(old_prefix) {
            if suffix.starts_with('.') {
                return format!("{new_prefix}{suffix}");
            }
        }
    }
    format!("{scope}{name}")
}

fn resolve_atom(
    a: &Atom,
    rename: &HashMap<String, String>,
    dict_renames: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> Atom {
    match a {
        Atom::Name(n) => Atom::Name(resolve_name(n, rename, dict_renames, scope, passthrough)),
        _ => a.clone(),
    }
}

fn scope_ops(
    ops: &[Op],
    rename: &HashMap<String, String>,
    dict_renames: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> Vec<Op> {
    ops.iter()
        .map(|op| Op {
            kind: op.kind.clone(),
            outputs: op
                .outputs
                .iter()
                .map(|o| resolve_name(o, rename, dict_renames, scope, passthrough))
                .collect(),
            output_types: op.output_types.clone(),
            args: op
                .args
                .iter()
                .map(|a| resolve_atom(a, rename, dict_renames, scope, passthrough))
                .collect(),
            comments: op.comments.clone(),
        })
        .collect()
}

// ── Expansion ───────────────────────────────────────────────────────

fn rename_op_args(op: &Op, renames: &HashMap<String, String>) -> Op {
    if renames.is_empty() {
        return op.clone();
    }
    Op {
        kind: op.kind.clone(),
        outputs: op.outputs.clone(),
        output_types: op.output_types.clone(),
        args: op
            .args
            .iter()
            .map(|a| match a {
                Atom::Name(n) => Atom::Name(renames.get(n).cloned().unwrap_or_else(|| n.clone())),
                _ => a.clone(),
            })
            .collect(),
        comments: op.comments.clone(),
    }
}

fn expand(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    fn_name: &str,
) -> (Vec<Op>, HashSet<String>) {
    let f = &functions[fn_name];
    let mut ops = Vec::new();
    let mut resolved = HashSet::new();
    let mut renames: HashMap<String, String> = HashMap::new();

    for step in &f.ops {
        let step = rename_op_args(step, &renames);

        if is_primitive(&step) {
            ops.push(step);
        } else if let OpKind::Call { .. } = &step.kind {
            let (call_ops, call_resolved) =
                inline_call(functions, globals, &step, &step.outputs[0]);
            ops.extend(call_ops);
            resolved.extend(call_resolved);
        } else if let OpKind::Loop { .. } = &step.kind {
            let (loop_ops, loop_resolved, loop_renames) =
                inline_loop(functions, globals, &step);
            ops.extend(loop_ops);
            resolved.extend(loop_resolved);
            renames.extend(loop_renames);
        }
    }
    (ops, resolved)
}

fn inline_call(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    call: &Op,
    scope_name: &str,
) -> (Vec<Op>, HashSet<String>) {
    let target = match &call.kind {
        OpKind::Call { target } => target.as_str(),
        _ => unreachable!(),
    };
    let callee = &functions[target];
    let (expanded, resolved) = expand(functions, globals, target);

    // Build exact renames (regular params) and dict renames (* params)
    let mut rename = HashMap::new();
    let mut dict_renames = HashMap::new();

    let n_args = call.args.len();
    let n_params = callee.params.len();

    // When caller passes fewer args than callee has params, the last caller
    // arg is a subtree base: each remaining callee param name gets dotted onto it.
    let n_matched = if n_args < n_params { n_args - 1 } else { n_args };

    for (param, arg) in callee.params[..n_matched].iter().zip(call.args.iter()) {
        if let Atom::Name(n) = arg {
            if is_dict_param(&param.ty) {
                dict_renames.insert(param.name.clone(), n.clone());
            } else {
                rename.insert(param.name.clone(), n.clone());
            }
        }
    }

    if n_args < n_params {
        if let Some(Atom::Name(base)) = call.args.last() {
            for param in &callee.params[n_matched..] {
                rename.insert(param.name.clone(), format!("{}.{}", base, param.name));
            }
        }
    }

    // Map return names to call output names
    let param_set: HashSet<&str> = callee.params.iter().map(|p| p.name.as_str()).collect();
    for (ret, out) in callee.returns.iter().zip(call.outputs.iter()) {
        if !param_set.contains(ret.name.as_str()) {
            rename.insert(ret.name.clone(), out.clone());
        }
    }

    let mut passthrough = globals.clone();
    passthrough.extend(resolved.iter().cloned());

    let scope = format!("{scope_name}.");
    (
        scope_ops(&expanded, &rename, &dict_renames, &scope, &passthrough),
        resolved,
    )
}

fn inline_loop(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    loop_op: &Op,
) -> (Vec<Op>, HashSet<String>, HashMap<String, String>) {
    let (target, count) = match &loop_op.kind {
        OpKind::Loop { target, count } => {
            let n = match count {
                LoopCount::Concrete(n) => *n,
                LoopCount::Named(s) => panic!("Loop count '{s}' not resolved before flattening"),
            };
            (target.as_str(), n)
        }
        _ => unreachable!(),
    };

    let output_name = &loop_op.outputs[0];

    // Classify args purely by syntax:
    //   - matches output name → threaded
    //   - ends with .* → indexed (strip .*, append .0, .1, ...)
    //   - everything else → static
    let threaded_idx = loop_op
        .args
        .iter()
        .position(|a| matches!(a, Atom::Name(n) if n == output_name))
        .expect("Loop must have a threaded arg matching output name");

    let mut ops = Vec::new();
    let mut resolved = HashSet::new();
    let mut threaded_name = output_name.to_string();

    for i in 0..count {
        let next_threaded = format!("{output_name}_{}", i + 1);

        let mut call_args: Vec<Atom> = Vec::new();
        for (j, arg) in loop_op.args.iter().enumerate() {
            if j == threaded_idx {
                call_args.push(Atom::Name(threaded_name.clone()));
            } else if let Atom::Name(n) = arg {
                if let Some(base) = n.strip_suffix(".*") {
                    call_args.push(Atom::Name(format!("{base}.{i}")));
                } else {
                    call_args.push(arg.clone());
                }
            } else {
                call_args.push(arg.clone());
            }
        }

        let synth_call = Op {
            kind: OpKind::Call {
                target: target.to_string(),
            },
            outputs: vec![next_threaded.clone()],
            output_types: loop_op.output_types.clone(),
            args: call_args,
            comments: Vec::new(),
        };

        let loop_scope = format!("loop{i}");
        let (inlined, sub_resolved) = inline_call(functions, globals, &synth_call, &loop_scope);
        // Rewrite cache vars with `.*` suffix to use this iteration's index.
        let inlined = inlined
            .into_iter()
            .map(|op| substitute_cache_var_star(op, i))
            .collect::<Vec<_>>();
        ops.extend(inlined);
        resolved.extend(sub_resolved);

        threaded_name = next_threaded;
    }

    // SSA: final iteration produced output_name_{count}; rename downstream
    let mut loop_renames = HashMap::new();
    loop_renames.insert(output_name.to_string(), format!("{output_name}_{count}"));

    (ops, resolved, loop_renames)
}

/// Inside a loop iteration `i`, rewrite any `cache[var.*, ...]` to
/// `cache[var.{i}, ...]`. Mirrors the arg-side `.*` expansion at line 275.
fn substitute_cache_var_star(op: Op, i: usize) -> Op {
    match &op.kind {
        OpKind::Cache { var, init, extend } if var.ends_with(".*") => {
            let base = var.strip_suffix(".*").unwrap();
            Op {
                kind: OpKind::Cache {
                    var: format!("{base}.{i}"),
                    init: init.clone(),
                    extend: extend.clone(),
                },
                ..op
            }
        }
        _ => op,
    }
}

/// Phase 2: walk the flat ops list and replace each Cache op with its
/// resolved equivalent (init call, extend call, or passthrough).
fn resolve_caches(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    ops: Vec<Op>,
    ctx: &mut CacheCtx,
) -> Vec<Op> {
    let mut out = Vec::new();
    for op in ops {
        if matches!(op.kind, OpKind::Cache { .. }) {
            out.extend(resolve_cache(functions, globals, &op, ctx));
        } else {
            out.push(op);
        }
    }
    out
}

/// Resolve a single `cache[var, init, extend?](args...)` op against the cache state.
fn resolve_cache(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    cache_op: &Op,
    ctx: &mut CacheCtx,
) -> Vec<Op> {
    let (var, init, extend) = match &cache_op.kind {
        OpKind::Cache {
            var,
            init,
            extend,
        } => (var.clone(), init.clone(), extend.clone()),
        _ => unreachable!(),
    };

    // Cache slot type = the declared output type of the cache[] op.
    let var_type = cache_op
        .output_types
        .first()
        .and_then(|t| t.as_ref())
        .cloned()
        .expect("cache slot must have a declared type");

    let warm = ctx.cache_keys.contains(&var);

    if warm {
        // Cache slot is populated; bind via main's input parameter.
        ctx.record_input(&var, &var_type);

        match extend {
            None => {
                // Idempotent passthrough — surfaces as an output of main so
                // Pianola can capture it back unchanged. No body op needed:
                // `var` is already bound by the input parameter.
                ctx.record_output(&var, &var_type);
                Vec::new()
            }
            Some(extend_target) => {
                // Stateful: emit `<var>_new = call[extend](var_prev, args...)`.
                // SSA: input parameter is `var`; output gets a fresh name to
                // avoid collision. Pianola sees the renamed output and stores
                // it back as the new cache slot.
                let new_name = format!("{var}_new");
                let mut extend_args = vec![Atom::Name(var.clone())];
                extend_args.extend(cache_op.args.iter().cloned());
                let synth_call = Op {
                    kind: OpKind::Call {
                        target: extend_target,
                    },
                    outputs: vec![new_name.clone()],
                    output_types: cache_op.output_types.clone(),
                    args: extend_args,
                    comments: Vec::new(),
                };
                ctx.record_output(&new_name, &var_type);
                inline_call(functions, globals, &synth_call, &new_name).0
            }
        }
    } else {
        // Cold path: emit `var = call[init](args...)`. Var becomes an output.
        ctx.record_output(&var, &var_type);
        let synth_call = Op {
            kind: OpKind::Call { target: init },
            outputs: vec![var.clone()],
            output_types: cache_op.output_types.clone(),
            args: cache_op.args.clone(),
            comments: cache_op.comments.clone(),
        };
        inline_call(functions, globals, &synth_call, &var).0
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    /// Idempotent cache: cold call emits init, warm call has the slot as input passthrough.
    const IDEMPOTENT_SRC: &str = r#"
build_mask(pos: f32[S]) -> (mask: bf16[S, S]) {
  mask: bf16[S, S] = map[bf16](pos)
}
main(pos: f32[S]) -> (out: bf16[S, S]) {
  mask: bf16[S, S] = cache[mask, build_mask](pos)
  out : bf16[S, S] = map[add](mask, mask)
}
"#;

    #[test]
    fn flatten_idempotent_cold() {
        let m = parse(IDEMPOTENT_SRC);
        let cold = HashSet::new();
        let flat = flatten(&m, "main", &cold);
        let f = &flat.functions["main"];
        // Cold: build_mask call inlined as a primitive (map[bf16]); then map[add]
        let kinds: Vec<_> = f.ops.iter().map(|o| std::mem::discriminant(&o.kind)).collect();
        assert_eq!(f.ops.len(), 2, "expected 2 ops, got {:?}", f.ops);
        // mask should be in the returns now
        let return_names: Vec<&str> = f.returns.iter().map(|p| p.name.as_str()).collect();
        assert!(return_names.contains(&"out"));
        assert!(return_names.contains(&"mask"));
        // mask should NOT be in params (cold path — no input)
        let param_names: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        assert!(!param_names.contains(&"mask"));
        let _ = kinds;
    }

    #[test]
    fn flatten_idempotent_warm() {
        let m = parse(IDEMPOTENT_SRC);
        let mut warm = HashSet::new();
        warm.insert("mask".to_string());
        let flat = flatten(&m, "main", &warm);
        let f = &flat.functions["main"];
        // Warm: only the map[add] op remains; the cache resolves to a passthrough (no body op)
        assert_eq!(f.ops.len(), 1, "expected 1 op (map[add]), got {:?}", f.ops);
        // mask should be in BOTH params (input passthrough) and returns
        let param_names: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        let return_names: Vec<&str> = f.returns.iter().map(|p| p.name.as_str()).collect();
        assert!(param_names.contains(&"mask"));
        assert!(return_names.contains(&"mask"));
        assert!(return_names.contains(&"out"));
    }

    /// Cache inside a loop: each iteration produces its own cache slot suffix.
    const LOOP_CACHE_SRC: &str = r#"
init_buf(x: bf16[D]) -> (buf: bf16[D]) {
  buf: bf16[D] = map[bf16](x)
}
extend_buf(prev: bf16[D], x: bf16[D]) -> (buf: bf16[D]) {
  buf: bf16[D] = map[add](prev, x)
}
layer(x: bf16[D], w: bf16[D]) -> (x: bf16[D]) {
  kv  : bf16[D] = cache[kv_buf.*, init_buf, extend_buf](w)
  x   : bf16[D] = map[add](x, kv)
}
main(x: bf16[D], weights: *) -> (x: bf16[D]) {
  x: bf16[D] = loop[layer, 3](x, weights.*)
}
"#;

    #[test]
    fn flatten_cache_in_loop_cold() {
        let m = parse(LOOP_CACHE_SRC);
        let cold = HashSet::new();
        let flat = flatten(&m, "main", &cold);
        let f = &flat.functions["main"];
        // 3 iterations × (init call inlined → 1 op, plus 1 add op) = 6 ops total
        assert_eq!(f.ops.len(), 6, "expected 6 ops, got {:?}", f.ops);
        // 3 cache slots in returns: kv_buf.0, kv_buf.1, kv_buf.2
        let return_names: Vec<&str> = f.returns.iter().map(|p| p.name.as_str()).collect();
        assert!(return_names.contains(&"kv_buf.0"));
        assert!(return_names.contains(&"kv_buf.1"));
        assert!(return_names.contains(&"kv_buf.2"));
    }

    #[test]
    fn flatten_cache_in_loop_warm() {
        let m = parse(LOOP_CACHE_SRC);
        let mut warm = HashSet::new();
        warm.insert("kv_buf.0".to_string());
        warm.insert("kv_buf.1".to_string());
        warm.insert("kv_buf.2".to_string());
        let flat = flatten(&m, "main", &warm);
        let f = &flat.functions["main"];
        // 3 iterations × (extend call inlined → 1 op, plus 1 add op) = 6 ops total
        assert_eq!(f.ops.len(), 6, "expected 6 ops, got {:?}", f.ops);
        let param_names: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        assert!(param_names.contains(&"kv_buf.0"));
        assert!(param_names.contains(&"kv_buf.1"));
        assert!(param_names.contains(&"kv_buf.2"));
    }
}

