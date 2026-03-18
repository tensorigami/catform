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

fn resolve_name(
    name: &str,
    rename: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> String {
    if passthrough.contains(name) {
        name.to_string()
    } else if let Some(mapped) = rename.get(name) {
        mapped.clone()
    } else {
        format!("{scope}{name}")
    }
}

fn resolve_atom(
    a: &Atom,
    rename: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> Atom {
    match a {
        Atom::Name(n) => Atom::Name(resolve_name(n, rename, scope, passthrough)),
        _ => a.clone(),
    }
}

fn scope_ops(
    ops: &[Op],
    rename: &HashMap<String, String>,
    scope: &str,
    passthrough: &HashSet<String>,
) -> Vec<Op> {
    ops.iter()
        .map(|op| Op {
            kind: op.kind.clone(),
            outputs: op
                .outputs
                .iter()
                .map(|o| resolve_name(o, rename, scope, passthrough))
                .collect(),
            output_types: op.output_types.clone(),
            args: op
                .args
                .iter()
                .map(|a| resolve_atom(a, rename, scope, passthrough))
                .collect(),
            comments: op.comments.clone(),
        })
        .collect()
}

/// Check if a name has _* suffix
fn has_star(name: &str) -> bool {
    name.ends_with("_*")
}

/// Extract base name: "act_*" → "act", "act_{*+1}" → "act", "cos" → "cos"
fn star_base(name: &str) -> &str {
    name.strip_suffix("_{*+1}")
        .or_else(|| name.strip_suffix("_*"))
        .unwrap_or(name)
}

// ── Expansion ───────────────────────────────────────────────────────

fn expand(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    fn_name: &str,
) -> (Vec<Op>, HashSet<String>) {
    let f = &functions[fn_name];
    let mut ops = Vec::new();
    let mut resolved = HashSet::new();

    // Scope: names bound in this function (params + previous op outputs)
    let mut scope: HashSet<String> = f.params.iter().map(|p| p.name.clone()).collect();

    // Renames from loop expansion: "act_{*+1}" → "act"
    let mut loop_renames: HashMap<String, String> = HashMap::new();

    for step in &f.ops {
        // Apply loop renames to this op's args
        let step = if loop_renames.is_empty() {
            step.clone()
        } else {
            let new_args = step
                .args
                .iter()
                .map(|a| {
                    if let Atom::Name(n) = a {
                        if let Some(renamed) = loop_renames.get(n) {
                            Atom::Name(renamed.clone())
                        } else {
                            a.clone()
                        }
                    } else {
                        a.clone()
                    }
                })
                .collect();
            Op {
                args: new_args,
                ..step.clone()
            }
        };

        if is_primitive(&step) {
            for o in &step.outputs {
                scope.insert(o.clone());
            }
            ops.push(step);
        } else if let OpKind::Call { .. } = &step.kind {
            let (call_ops, call_resolved) =
                inline_call(functions, globals, &step, &step.outputs[0]);
            for o in &step.outputs {
                scope.insert(o.clone());
            }
            ops.extend(call_ops);
            resolved.extend(call_resolved);
        } else if let OpKind::Loop { .. } = &step.kind {
            let (loop_ops, loop_resolved, renames) =
                inline_loop(functions, globals, &step, &f.params);
            ops.extend(loop_ops);
            resolved.extend(loop_resolved);
            for (_, v) in &renames {
                scope.insert(v.clone());
            }
            loop_renames.extend(renames);
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

    let param_names: Vec<&str> = callee.params.iter().map(|p| p.name.as_str()).collect();
    let mut rename = HashMap::new();
    for (pn, arg) in param_names.iter().zip(call.args.iter()) {
        if let Atom::Name(n) = arg {
            rename.insert(pn.to_string(), n.clone());
        }
    }
    let param_set: HashSet<&str> = param_names.iter().copied().collect();
    for (ret, out) in callee.returns.iter().zip(call.outputs.iter()) {
        if !param_set.contains(ret.name.as_str()) {
            rename.insert(ret.name.clone(), out.clone());
        }
    }

    let mut passthrough = globals.clone();
    passthrough.extend(resolved.iter().cloned());

    let scope = format!("{scope_name}.");
    (
        scope_ops(&expanded, &rename, &scope, &passthrough),
        resolved,
    )
}

fn inline_loop(
    functions: &IndexMap<String, Function>,
    globals: &HashSet<String>,
    loop_op: &Op,
    enclosing_params: &[Param],
) -> (Vec<Op>, HashSet<String>, HashMap<String, String>) {
    let target = match &loop_op.kind {
        OpKind::Loop { target } => target.as_str(),
        _ => unreachable!(),
    };

    // Parse output: "act_{*+1}" → base "act"
    let output_name = &loop_op.outputs[0];
    let output_base = star_base(output_name);

    // Build param lookup: name → count dim (for list params)
    let param_counts: HashMap<&str, &Dim> = enclosing_params
        .iter()
        .filter_map(|p| p.count.as_ref().map(|c| (p.name.as_str(), c)))
        .collect();

    // Classify args and determine count from list param types
    let mut threaded_idx: Option<usize> = None;
    let mut list_indices: Vec<usize> = Vec::new();
    let mut count: Option<usize> = None;

    for (i, arg) in loop_op.args.iter().enumerate() {
        if let Atom::Name(n) = arg {
            if has_star(n) && star_base(n) == output_base {
                threaded_idx = Some(i);
            } else if let Some(dim) = param_counts.get(n.as_str()) {
                list_indices.push(i);
                match dim {
                    Dim::Concrete(c) => {
                        let c = *c as usize;
                        if let Some(prev) = count {
                            assert_eq!(prev, c, "All list args must have the same count");
                        }
                        count = Some(c);
                    }
                    Dim::Named(n) => {
                        panic!("List count '{n}' must be resolved before flattening")
                    }
                }
            }
            // else: scalar (no list type annotation)
        }
    }

    let threaded_idx = threaded_idx.expect("Loop must have a threaded _* arg matching output base");
    let count = count.expect("Loop must have at least one list arg");

    let mut ops = Vec::new();
    let mut resolved = HashSet::new();

    // Track the current threaded value name
    let mut threaded_name = format!("{output_base}_0");

    for i in 0..count {
        let next_threaded = format!("{output_base}_{}", i + 1);

        // Build synthetic call args
        let mut call_args: Vec<Atom> = Vec::new();
        for (j, arg) in loop_op.args.iter().enumerate() {
            if j == threaded_idx {
                // Threaded: use current iteration's name
                call_args.push(Atom::Name(threaded_name.clone()));
            } else if list_indices.contains(&j) {
                // List: index with iteration number
                if let Atom::Name(n) = arg {
                    let indexed = format!("{n}_{i}");
                    resolved.insert(indexed.clone());
                    call_args.push(Atom::Name(indexed));
                } else {
                    call_args.push(arg.clone());
                }
            } else {
                // Scalar: pass through unchanged
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
        let (inlined, _) = inline_call(functions, globals, &synth_call, &loop_scope);
        ops.extend(inlined);

        threaded_name = next_threaded;
    }

    // Final rename: act_{count} → base name "act"
    let final_name = format!("{output_base}_{count}");
    let base_name = output_base.to_string();

    if !ops.is_empty() {
        let mut renames = HashMap::new();
        renames.insert(final_name.clone(), base_name.clone());

        ops = ops
            .into_iter()
            .map(|op| {
                let new_args = op
                    .args
                    .iter()
                    .map(|a| {
                        if let Atom::Name(n) = a {
                            if let Some(mapped) = renames.get(n.as_str()) {
                                Atom::Name(mapped.clone())
                            } else {
                                a.clone()
                            }
                        } else {
                            a.clone()
                        }
                    })
                    .collect();
                let new_outputs = op
                    .outputs
                    .iter()
                    .map(|o| {
                        renames
                            .get(o.as_str())
                            .cloned()
                            .unwrap_or_else(|| o.clone())
                    })
                    .collect();
                Op {
                    kind: op.kind,
                    outputs: new_outputs,
                    output_types: op.output_types,
                    args: new_args,
                    comments: op.comments,
                }
            })
            .collect();
    }

    // Return renames so expand() can substitute in subsequent ops
    let mut output_renames = HashMap::new();
    output_renames.insert(output_name.clone(), base_name);

    (ops, resolved, output_renames)
}
