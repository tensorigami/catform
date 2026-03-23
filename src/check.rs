use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::Serialize;

use crate::ast::*;

// ── Constraints ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize)]
pub struct Constraints {
    /// Config param names referenced anywhere in the module (param.X)
    pub params: BTreeSet<String>,
    /// Per-function weight constraints: fn_name → { dict_path → expected type }
    pub weights: BTreeMap<String, BTreeMap<String, TensorType>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CheckResult {
    pub errors: Vec<String>,
    pub constraints: Constraints,
}

// ── Public API ──────────────────────────────────────────────────────

pub fn check(m: &Module) -> CheckResult {
    let mut result = CheckResult::default();
    for (fn_name, f) in &m.functions {
        check_function(fn_name, f, &m.functions, &mut result);
    }
    result
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

// ── Helpers ─────────────────────────────────────────────────────────

/// Is this name an access into a dict (*) parameter? e.g. "weights.ln1" when "weights" is *.
fn is_dict_access(name: &str, dict_params: &HashSet<&str>) -> bool {
    name.split('.')
        .next()
        .is_some_and(|base| dict_params.contains(base))
}

// ── Function checking ───────────────────────────────────────────────

fn check_function(
    fn_name: &str,
    f: &Function,
    functions: &indexmap::IndexMap<String, Function>,
    result: &mut CheckResult,
) {
    let mut env: HashMap<String, Ty> = HashMap::new();

    let dict_params: HashSet<&str> = f
        .params
        .iter()
        .filter(|p| p.ty.dtype == "*")
        .map(|p| p.name.as_str())
        .collect();

    // Collect param.X references from parameter type annotations
    for p in &f.params {
        if p.ty.dtype != "*" {
            env.insert(p.name.clone(), ty_from_tensor_type(&p.ty));
            collect_params_from_type(&p.ty, &mut result.constraints.params);
        }
    }

    // Collect param.X references from return type annotations
    for r in &f.returns {
        collect_params_from_type(&r.ty, &mut result.constraints.params);
    }

    for op in &f.ops {
        check_op(fn_name, op, &env, functions, &dict_params, result);
        for (out_name, out_type) in op.outputs.iter().zip(op.output_types.iter()) {
            if let Some(t) = out_type {
                env.insert(out_name.clone(), ty_from_tensor_type(t));
                collect_params_from_type(t, &mut result.constraints.params);
            }
        }
    }

    for ret in &f.returns {
        let rtype = ty_from_tensor_type(&ret.ty);
        if let Some(actual) = env.get(&ret.name) {
            if let Some(err) = types_match(actual, &rtype) {
                result
                    .errors
                    .push(format!("{fn_name}: return '{}' {err}", ret.name));
            }
        } else {
            result
                .errors
                .push(format!("{fn_name}: return '{}' not found in env", ret.name));
        }
    }
}

/// Collect param.X dimension names from a tensor type.
fn collect_params_from_type(t: &TensorType, params: &mut BTreeSet<String>) {
    for d in &t.shape {
        if let Dim::Named(n) = d
            && n.starts_with("param.")
        {
            params.insert(n.clone());
        }
    }
}

fn check_op(
    fn_name: &str,
    op: &Op,
    env: &HashMap<String, Ty>,
    functions: &indexmap::IndexMap<String, Function>,
    dict_params: &HashSet<&str>,
    result: &mut CheckResult,
) {
    let out_name = &op.outputs[0];
    let declared = op.output_types[0].as_ref().map(ty_from_tensor_type);
    let loc = format!("{fn_name}/{out_name}");

    let mut local_errors = Vec::new();

    let inputs: Vec<Option<Ty>> = op
        .args
        .iter()
        .map(|a| match a {
            Atom::Name(n) if env.contains_key(n) => Some(env[n].clone()),
            Atom::Name(n) if n.starts_with("param.") => {
                result.constraints.params.insert(n.clone());
                None
            }
            Atom::Name(n) if is_dict_access(n, dict_params) => None,
            Atom::Name(n) => {
                local_errors.push(format!("{loc}: input '{n}' not in scope"));
                None
            }
            _ => None, // Int, Float literals
        })
        .collect();

    result.errors.extend(local_errors);

    let declared = match declared {
        Some(d) => d,
        None => return,
    };

    let expected = op_rule(
        op,
        &inputs,
        &declared,
        functions,
        &loc,
        &mut result.errors,
        &mut result.constraints,
        fn_name,
        dict_params,
    );
    if let Some(expected) = expected
        && let Some(err) = types_match(&expected, &declared)
    {
        result.errors.push(format!(
            "{loc}: declared {declared:?}, expected {expected:?} — {err}"
        ));
    }
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
    constraints: &mut Constraints,
    fn_name: &str,
    dict_params: &HashSet<&str>,
) -> Option<Ty> {
    match &op.kind {
        OpKind::Map { function } => rule_map(inputs, declared, function),
        OpKind::Fold { .. } => rule_fold(inputs, declared),
        OpKind::Tile { .. } => rule_tile(inputs, declared),
        OpKind::View { .. } => rule_view(inputs, declared, loc, errors),
        OpKind::Contract { .. } => rule_contract(inputs, declared),
        OpKind::Gather { .. } => rule_gather(inputs, declared, loc, errors),
        OpKind::Scatter { .. } => rule_scatter(inputs, declared),
        OpKind::Call { target } => {
            rule_call(target, op, inputs, functions, declared, loc, errors, constraints, fn_name, dict_params)
        }
        OpKind::Loop { target, count } => {
            if let LoopCount::Named(s) = count {
                if s.starts_with("param.") {
                    constraints.params.insert(s.clone());
                }
            }
            rule_call(target, op, inputs, functions, declared, loc, errors, constraints, fn_name, dict_params)
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
    op: &Op,
    inputs: &[Option<Ty>],
    functions: &indexmap::IndexMap<String, Function>,
    declared: &Ty,
    loc: &str,
    errors: &mut Vec<String>,
    constraints: &mut Constraints,
    fn_name: &str,
    dict_params: &HashSet<&str>,
) -> Option<Ty> {
    let callee = functions.get(target)?;
    let param_types: Vec<Option<Ty>> = callee
        .params
        .iter()
        .map(|p| {
            if p.ty.dtype == "*" {
                None
            } else {
                Some(ty_from_tensor_type(&p.ty))
            }
        })
        .collect();

    let polymorphic = inputs
        .iter()
        .zip(param_types.iter())
        .any(|(inp, pt)| {
            inp.as_ref()
                .zip(pt.as_ref())
                .is_some_and(|(i, p)| i.shape.len() > p.shape.len())
        });

    for (i, (inp, pt)) in inputs.iter().zip(param_types.iter()).enumerate() {
        if let (Some(inp), Some(pt)) = (inp, pt) {
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

    // Collect weight constraints: dict-accessed args passed to typed callee params
    for (i, param) in callee.params.iter().enumerate() {
        if param.ty.dtype == "*" {
            continue;
        }
        if let Some(Atom::Name(arg_name)) = op.args.get(i) {
            if is_dict_access(arg_name, dict_params) {
                let fn_weights = constraints
                    .weights
                    .entry(fn_name.to_string())
                    .or_default();
                if let Some(existing) = fn_weights.get(arg_name) {
                    if existing != &param.ty {
                        errors.push(format!(
                            "{loc}: weight '{}' used with conflicting types: {:?} vs {:?}",
                            arg_name, existing, param.ty
                        ));
                    }
                } else {
                    fn_weights.insert(arg_name.clone(), param.ty.clone());
                }
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
            if x == "..." || y == "..." || is_free(x) || is_free(y) {
                true
            } else {
                x == y
            }
        }
        _ => true, // mixed concrete/symbolic — can't decide statically
    }
}

/// Split a shape into (has_ellipsis, fixed_suffix).
/// `f32[..., N, M]` → (true, [N, M]).  `f32[N, M]` → (false, [N, M]).
fn ellipsis_split(shape: &[Dim]) -> (bool, &[Dim]) {
    match shape.first() {
        Some(Dim::Named(n)) if n == "..." => (true, &shape[1..]),
        _ => (false, shape),
    }
}

fn types_match(actual: &Ty, expected: &Ty) -> Option<String> {
    if actual.dtype != expected.dtype {
        return Some(format!("dtype: {} vs {}", actual.dtype, expected.dtype));
    }

    let (a_ell, a_fixed) = ellipsis_split(&actual.shape);
    let (e_ell, e_fixed) = ellipsis_split(&expected.shape);

    if a_ell || e_ell {
        // Ellipsis: match fixed suffixes only
        let (short, long) = if a_fixed.len() <= e_fixed.len() {
            (a_fixed, e_fixed)
        } else {
            (e_fixed, a_fixed)
        };
        let offset = long.len() - short.len();
        for (i, (s, l)) in short.iter().zip(long[offset..].iter()).enumerate() {
            if !dims_match(s, l) {
                return Some(format!("dim {i}: {s:?} vs {l:?}"));
            }
        }
        return None;
    }

    // No ellipsis: exact rank + dim match
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
