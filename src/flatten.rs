use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::ast::*;

// ── Public API ──────────────────────────────────────────────────────

pub fn flatten(m: &Module, entry: &str) -> Module {
    let passthrough = collect_param_names(&m.functions);
    let (ops, _) = expand(&m.functions, &passthrough, entry);
    let entry_fn = &m.functions[entry];
    let flat_fn = Function {
        name: entry_fn.name.clone(),
        comments: entry_fn.comments.clone(),
        params: entry_fn.params.clone(),
        returns: entry_fn.returns.clone(),
        ops,
    };
    let mut functions = IndexMap::new();
    functions.insert(entry.to_string(), flat_fn);
    Module {
        header_comments: m.header_comments.clone(),
        functions,
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

    for (param, arg) in callee.params.iter().zip(call.args.iter()) {
        if let Atom::Name(n) = arg {
            if is_dict_param(&param.ty) {
                dict_renames.insert(param.name.clone(), n.clone());
            } else {
                rename.insert(param.name.clone(), n.clone());
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
        ops.extend(inlined);
        resolved.extend(sub_resolved);

        threaded_name = next_threaded;
    }

    // SSA: final iteration produced output_name_{count}; rename downstream
    let mut loop_renames = HashMap::new();
    loop_renames.insert(output_name.to_string(), format!("{output_name}_{count}"));

    (ops, resolved, loop_renames)
}
