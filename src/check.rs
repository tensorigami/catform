use std::collections::HashMap;

use crate::ast::*;

// ── Public API ──────────────────────────────────────────────────────

pub fn check(m: &Module) -> Vec<String> {
    let mut errors = Vec::new();
    for (fn_name, f) in &m.functions {
        errors.extend(check_function(fn_name, f, &m.functions));
    }
    errors
}

// ── Type representation ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Ty {
    dtype: String,
    shape: Vec<Dim>,
}

fn ty_from_tensor_type(t: &TensorType) -> Ty {
    Ty {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
    }
}

// ── Function checking ───────────────────────────────────────────────

fn check_function(
    fn_name: &str,
    f: &Function,
    functions: &indexmap::IndexMap<String, Function>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut env: HashMap<String, Ty> = HashMap::new();

    for p in &f.params {
        env.insert(p.name.clone(), ty_from_tensor_type(&p.ty));
    }

    for op in &f.ops {
        errors.extend(check_op(fn_name, op, &env, functions));
        for (out_name, out_type) in op.outputs.iter().zip(op.output_types.iter()) {
            if let Some(t) = out_type {
                env.insert(out_name.clone(), ty_from_tensor_type(t));
            }
        }
    }

    for ret in &f.returns {
        let rtype = ty_from_tensor_type(&ret.ty);
        if let Some(actual) = env.get(&ret.name) {
            if let Some(err) = types_match(actual, &rtype) {
                errors.push(format!("{fn_name}: return '{}' {err}", ret.name));
            }
        } else {
            errors.push(format!("{fn_name}: return '{}' not found in env", ret.name));
        }
    }

    errors
}

fn is_external(a: &Atom) -> bool {
    match a {
        Atom::Name(n) => n.contains('.') || n.ends_with("_*"),
        _ => true,
    }
}

fn check_op(
    fn_name: &str,
    op: &Op,
    env: &HashMap<String, Ty>,
    functions: &indexmap::IndexMap<String, Function>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let out_name = &op.outputs[0];
    let declared = op.output_types[0].as_ref().map(ty_from_tensor_type);
    let loc = format!("{fn_name}/{out_name}");

    // For call/loop ops, find list-typed params in callee (skip scope check for those args)
    let list_param_indices: std::collections::HashSet<usize> = match &op.kind {
        OpKind::Call { target } | OpKind::Loop { target } => functions
            .get(target.as_str())
            .map(|callee| {
                callee
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.count.is_some())
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default(),
        _ => std::collections::HashSet::new(),
    };

    let inputs: Vec<Option<Ty>> = op
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| match a {
            Atom::Name(n) if env.contains_key(n) => Some(env[n].clone()),
            _ if is_external(a) => None,
            _ if list_param_indices.contains(&i) => None,
            Atom::Name(n) => {
                errors.push(format!("{loc}: input '{n}' not in scope"));
                None
            }
            _ => None,
        })
        .collect();

    let declared = match declared {
        Some(d) => d,
        None => return errors,
    };

    let expected = op_rule(op, &inputs, &declared, functions, &loc, &mut errors);
    if let Some(expected) = expected
        && let Some(err) = types_match(&expected, &declared)
    {
        errors.push(format!(
            "{loc}: declared {declared:?}, expected {expected:?} — {err}"
        ));
    }

    errors
}

// ── Op rules ────────────────────────────────────────────────────────

const CAST_FUNCTIONS: &[&str] = &["f32", "bf16"];
const COMPARISON_FUNCTIONS: &[&str] = &["ge", "gt", "le", "lt", "eq", "ne"];

fn op_rule(
    op: &Op,
    inputs: &[Option<Ty>],
    declared: &Ty,
    functions: &indexmap::IndexMap<String, Function>,
    loc: &str,
    errors: &mut Vec<String>,
) -> Option<Ty> {
    match &op.kind {
        OpKind::Map { function } => rule_map(inputs, declared, function),
        OpKind::Fold { .. } => rule_fold(inputs, declared),
        OpKind::Tile { .. } => rule_tile(inputs, declared),
        OpKind::View { .. } => rule_view(inputs, declared, loc, errors),
        OpKind::Contract { .. } => rule_contract(inputs, declared),
        OpKind::Gather { .. } => rule_gather(inputs, declared, loc, errors),
        OpKind::Scatter { .. } => rule_scatter(inputs, declared),
        OpKind::Call { target } => rule_call(target, op, inputs, functions, declared, loc, errors),
        OpKind::Loop { target, .. } => {
            rule_call(target, op, inputs, functions, declared, loc, errors)
        }
        OpKind::Literal { .. } | OpKind::Random { .. } => Some(declared.clone()),
    }
}

fn rule_map(inputs: &[Option<Ty>], declared: &Ty, function: &str) -> Option<Ty> {
    let known: Vec<&Ty> = inputs.iter().filter_map(|t| t.as_ref()).collect();
    if known.is_empty() {
        return None;
    }
    let ref_ty = known.iter().max_by_key(|t| t.shape.len()).unwrap();

    if CAST_FUNCTIONS.contains(&function) {
        return Some(Ty {
            dtype: function.to_string(),
            shape: ref_ty.shape.clone(),
        });
    }
    if COMPARISON_FUNCTIONS.contains(&function) {
        return Some(Ty {
            dtype: declared.dtype.clone(),
            shape: ref_ty.shape.clone(),
        });
    }
    if function == "where" {
        let vals: Vec<&Ty> = inputs[1..].iter().filter_map(|t| t.as_ref()).collect();
        let dtype = vals
            .first()
            .map_or(declared.dtype.clone(), |t| t.dtype.clone());
        return Some(Ty {
            dtype,
            shape: ref_ty.shape.clone(),
        });
    }
    Some(Ty {
        dtype: ref_ty.dtype.clone(),
        shape: ref_ty.shape.clone(),
    })
}

fn rule_fold(inputs: &[Option<Ty>], declared: &Ty) -> Option<Ty> {
    let inp = inputs.first()?.as_ref()?;
    Some(Ty {
        dtype: inp.dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_tile(inputs: &[Option<Ty>], declared: &Ty) -> Option<Ty> {
    let inp = inputs.first()?.as_ref()?;
    Some(Ty {
        dtype: inp.dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_view(
    inputs: &[Option<Ty>],
    declared: &Ty,
    loc: &str,
    errors: &mut Vec<String>,
) -> Option<Ty> {
    let inp = inputs.first()?.as_ref()?;
    let in_prod = shape_product(&inp.shape);
    let out_prod = shape_product(&declared.shape);
    if let (Some(ip), Some(op)) = (in_prod, out_prod)
        && ip != op
    {
        errors.push(format!("{loc}: view reshapes {ip} elements to {op}"));
    }
    Some(Ty {
        dtype: inp.dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_contract(inputs: &[Option<Ty>], declared: &Ty) -> Option<Ty> {
    let known: Vec<&Ty> = inputs.iter().filter_map(|t| t.as_ref()).collect();
    if known.is_empty() {
        return None;
    }
    Some(Ty {
        dtype: known[0].dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_gather(
    inputs: &[Option<Ty>],
    declared: &Ty,
    loc: &str,
    errors: &mut Vec<String>,
) -> Option<Ty> {
    let data_type = inputs.first()?.as_ref();
    if let Some(idx_type) = inputs.get(1).and_then(|t| t.as_ref())
        && !["int32", "i32", "int64", "i64", "int"].contains(&idx_type.dtype.as_str())
    {
        errors.push(format!(
            "{loc}: gather index dtype {}, expected integer",
            idx_type.dtype
        ));
    }
    let dt = data_type?;
    Some(Ty {
        dtype: dt.dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_scatter(inputs: &[Option<Ty>], declared: &Ty) -> Option<Ty> {
    let dt = inputs.first()?.as_ref()?;
    Some(Ty {
        dtype: dt.dtype.clone(),
        shape: declared.shape.clone(),
    })
}

fn rule_call(
    target: &str,
    _op: &Op,
    inputs: &[Option<Ty>],
    functions: &indexmap::IndexMap<String, Function>,
    declared: &Ty,
    loc: &str,
    errors: &mut Vec<String>,
) -> Option<Ty> {
    let callee = functions.get(target)?;
    let param_types: Vec<Ty> = callee
        .params
        .iter()
        .map(|p| ty_from_tensor_type(&p.ty))
        .collect();

    let polymorphic = inputs
        .iter()
        .zip(param_types.iter())
        .any(|(inp, pt)| inp.as_ref().is_some_and(|i| i.shape.len() > pt.shape.len()));

    for (i, (inp, pt)) in inputs.iter().zip(param_types.iter()).enumerate() {
        if let Some(inp) = inp {
            let err = if polymorphic {
                types_match_polymorphic(inp, pt)
            } else {
                types_match(inp, pt)
            };
            if let Some(err) = err {
                errors.push(format!(
                    "{loc}: arg {i} '{}' — {err}",
                    callee.params[i].name
                ));
            }
        }
    }

    Some(declared.clone())
}

// ── Type matching ───────────────────────────────────────────────────

fn is_free(name: &str) -> bool {
    !name.contains('.')
}

fn dims_match(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::Concrete(x), Dim::Concrete(y)) => x == y,
        (Dim::Named(x), Dim::Named(y)) => {
            if is_free(x) || is_free(y) {
                true
            } else {
                x == y
            }
        }
        _ => true, // mixed concrete/symbolic — can't decide statically
    }
}

fn types_match(actual: &Ty, expected: &Ty) -> Option<String> {
    if actual.dtype != expected.dtype {
        return Some(format!("dtype: {} vs {}", actual.dtype, expected.dtype));
    }
    if actual.shape.len() != expected.shape.len() {
        return Some(format!(
            "rank: {} vs {}",
            actual.shape.len(),
            expected.shape.len()
        ));
    }
    for (i, (a, e)) in actual.shape.iter().zip(expected.shape.iter()).enumerate() {
        if !dims_match(a, e) {
            return Some(format!("dim {i}: {a:?} vs {e:?}"));
        }
    }
    None
}

fn types_match_polymorphic(actual: &Ty, expected: &Ty) -> Option<String> {
    if actual.dtype != expected.dtype {
        return Some(format!("dtype: {} vs {}", actual.dtype, expected.dtype));
    }
    if actual.shape.len() < expected.shape.len() {
        return Some(format!(
            "rank: {} < {}",
            actual.shape.len(),
            expected.shape.len()
        ));
    }
    let offset = actual.shape.len() - expected.shape.len();
    for (i, (a, e)) in actual.shape[offset..]
        .iter()
        .zip(expected.shape.iter())
        .enumerate()
    {
        if let (Dim::Concrete(x), Dim::Concrete(y)) = (a, e)
            && x != y
        {
            return Some(format!("dim {i}: {x} vs {y}"));
        }
    }
    None
}

fn shape_product(shape: &[Dim]) -> Option<i64> {
    let mut product: i64 = 1;
    for dim in shape {
        match dim {
            Dim::Concrete(n) => product *= n,
            Dim::Named(_) => return None,
        }
    }
    Some(product)
}
