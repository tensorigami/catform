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

fn resolve_weight(name: &str, scope: &str, prefix: &str, target: &str) -> String {
    let placeholder = format!("{scope}{target}.");
    if name.starts_with(&placeholder) {
        format!("{prefix}.{}", &name[placeholder.len()..])
    } else {
        name.to_string()
    }
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

    for step in &f.ops {
        if is_primitive(step) {
            ops.push(step.clone());
        } else if let OpKind::Call { .. } = &step.kind {
            let (call_ops, call_resolved) = inline_call(functions, globals, step, &step.outputs[0]);
            ops.extend(call_ops);
            resolved.extend(call_resolved);
        } else if let OpKind::Loop { .. } = &step.kind {
            let (loop_ops, loop_resolved) = inline_loop(functions, globals, step);
            ops.extend(loop_ops);
            resolved.extend(loop_resolved);
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
) -> (Vec<Op>, HashSet<String>) {
    let (target, over, count) = match &loop_op.kind {
        OpKind::Loop {
            target,
            over,
            count,
        } => {
            let c = match count {
                Atom::Int(n) => *n as usize,
                Atom::Float(f) => *f as usize,
                _ => panic!("Loop count must be resolved to an integer"),
            };
            (target.as_str(), over.as_str(), c)
        }
        _ => unreachable!(),
    };
    let callee = &functions[target];
    let return_names: Vec<&str> = callee.returns.iter().map(|r| r.name.as_str()).collect();
    let n_iterated = return_names.len();

    let loop_args: Vec<Atom> = loop_op.args.clone();
    let mut iterated: Vec<Atom> = loop_args[..n_iterated].to_vec();
    let static_args: Vec<Atom> = loop_args[n_iterated..].to_vec();

    let mut ops = Vec::new();
    let mut resolved = HashSet::new();

    for i in 0..count {
        let scope = format!("loop{i}.");
        let prefix = format!("{over}.{i}");

        let iter_outputs: Vec<String> = (0..n_iterated)
            .map(|j| format!("{}_{i}", loop_op.outputs[j]))
            .collect();

        let synth_call = Op {
            kind: OpKind::Call {
                target: target.to_string(),
            },
            outputs: iter_outputs.clone(),
            output_types: loop_op.output_types.clone(),
            args: [iterated.clone(), static_args.clone()].concat(),
            comments: Vec::new(),
        };
        let loop_scope = format!("loop{i}");
        let (inlined, _) = inline_call(functions, globals, &synth_call, &loop_scope);

        for op in inlined {
            let new_args: Vec<Atom> = op
                .args
                .iter()
                .map(|a| {
                    if let Atom::Name(n) = a {
                        let rn = resolve_weight(n, &scope, &prefix, target);
                        if rn != *n {
                            resolved.insert(rn.clone());
                        }
                        Atom::Name(rn)
                    } else {
                        a.clone()
                    }
                })
                .collect();
            let new_outputs: Vec<String> = op
                .outputs
                .iter()
                .map(|o| resolve_weight(o, &scope, &prefix, target))
                .collect();
            ops.push(Op {
                kind: op.kind,
                outputs: new_outputs,
                output_types: op.output_types,
                args: new_args,
                comments: op.comments,
            });
        }
        iterated = iter_outputs.into_iter().map(Atom::Name).collect();
    }

    // Final iteration: rename outputs to loop's declared names
    if !ops.is_empty() {
        let mut renames = HashMap::new();
        for j in 0..n_iterated {
            renames.insert(
                format!("{}_{}", loop_op.outputs[j], count - 1),
                loop_op.outputs[j].clone(),
            );
        }
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

    (ops, resolved)
}
