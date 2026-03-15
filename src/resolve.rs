use std::collections::HashMap;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use crate::ast::*;

static PATTERN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\.\.\.|\(|\)|[a-zA-Z_]\w*|\d+").unwrap());

// ── Param substitution ─────────────────────────────────────────────

fn sub_dim(d: &Dim, config: &HashMap<String, ConfigValue>) -> Dim {
    match d {
        Dim::Named(name) if name.starts_with("param.") => {
            let key = &name[6..];
            match config.get(key) {
                Some(ConfigValue::Int(n)) => Dim::Concrete(*n),
                Some(ConfigValue::Float(f)) => Dim::Concrete(*f as i64),
                _ => d.clone(),
            }
        }
        _ => d.clone(),
    }
}

fn sub_type(t: &TensorType, config: &HashMap<String, ConfigValue>) -> TensorType {
    TensorType {
        dtype: t.dtype.clone(),
        shape: t.shape.iter().map(|d| sub_dim(d, config)).collect(),
    }
}

fn sub_atom_in_specifiers(a: &Atom, config: &HashMap<String, ConfigValue>) -> Atom {
    match a {
        Atom::Name(name) if name.starts_with("param.") => {
            let key = &name[6..];
            match config.get(key) {
                Some(ConfigValue::Int(n)) => Atom::Int(*n),
                Some(ConfigValue::Float(f)) => Atom::Float(*f),
                _ => a.clone(),
            }
        }
        _ => a.clone(),
    }
}

fn sub_op(op: &Op, config: &HashMap<String, ConfigValue>) -> Op {
    let kind = match &op.kind {
        OpKind::View { pattern, axes } => OpKind::View {
            pattern: pattern.clone(),
            axes: axes.clone(),
        },
        OpKind::Map { function } => OpKind::Map {
            function: function.clone(),
        },
        OpKind::Fold { pattern, function } => OpKind::Fold {
            pattern: pattern.clone(),
            function: function.clone(),
        },
        OpKind::Tile { pattern, axes } => OpKind::Tile {
            pattern: pattern.clone(),
            axes: axes.clone(),
        },
        OpKind::Gather { pattern } => OpKind::Gather {
            pattern: pattern.clone(),
        },
        OpKind::Scatter { pattern } => OpKind::Scatter {
            pattern: pattern.clone(),
        },
        OpKind::Contract { pattern } => OpKind::Contract {
            pattern: pattern.clone(),
        },
        OpKind::Literal { value } => OpKind::Literal {
            value: value.clone(),
        },
        OpKind::Random { function } => OpKind::Random {
            function: function.clone(),
        },
        OpKind::Call { target } => OpKind::Call {
            target: target.clone(),
        },
        OpKind::Loop {
            target,
            over,
            count,
        } => OpKind::Loop {
            target: target.clone(),
            over: over.clone(),
            count: sub_atom_in_specifiers(count, config),
        },
    };

    Op {
        kind,
        outputs: op.outputs.clone(),
        output_types: op
            .output_types
            .iter()
            .map(|t| t.as_ref().map(|t| sub_type(t, config)))
            .collect(),
        // args are wire names — NOT substituted
        args: op.args.clone(),
        comments: op.comments.clone(),
    }
}

fn substitute_params(m: &Module, config: &HashMap<String, ConfigValue>) -> Module {
    let functions = m
        .functions
        .iter()
        .map(|(name, f)| {
            let params = f
                .params
                .iter()
                .map(|p| Param {
                    name: p.name.clone(),
                    ty: sub_type(&p.ty, config),
                })
                .collect();
            let returns = f
                .returns
                .iter()
                .map(|r| Param {
                    name: r.name.clone(),
                    ty: sub_type(&r.ty, config),
                })
                .collect();
            let ops = f.ops.iter().map(|op| sub_op(op, config)).collect();
            (
                name.clone(),
                Function {
                    name: f.name.clone(),
                    comments: f.comments.clone(),
                    params,
                    returns,
                    ops,
                },
            )
        })
        .collect();

    Module {
        header_comments: m.header_comments.clone(),
        functions,
    }
}

// ── Axis inference ──────────────────────────────────────────────────

#[derive(Debug)]
enum PatternElem {
    Axis(String),
    Group(Vec<String>),
}

fn parse_pattern_side(s: &str) -> (bool, Vec<PatternElem>) {
    let toks: Vec<&str> = PATTERN_RE.find_iter(s.trim()).map(|m| m.as_str()).collect();
    let mut has_ellipsis = false;
    let mut elements = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "..." {
            has_ellipsis = true;
            i += 1;
        } else if toks[i] == "(" {
            let mut group = Vec::new();
            i += 1;
            while toks[i] != ")" {
                group.push(toks[i].to_string());
                i += 1;
            }
            i += 1;
            elements.push(PatternElem::Group(group));
        } else {
            elements.push(PatternElem::Axis(toks[i].to_string()));
            i += 1;
        }
    }
    (has_ellipsis, elements)
}

fn dim_value(d: &Dim) -> Option<i64> {
    match d {
        Dim::Concrete(n) => Some(*n),
        Dim::Named(_) => None,
    }
}

fn match_axes_to_type(
    has_ellipsis: bool,
    elements: &[PatternElem],
    type_dims: &[Dim],
) -> HashMap<String, i64> {
    let skip = if has_ellipsis {
        type_dims.len() - elements.len()
    } else {
        0
    };
    let mut result = HashMap::new();
    let mut dim_idx = skip;
    for elem in elements {
        match elem {
            PatternElem::Group(_) => {
                dim_idx += 1;
            }
            PatternElem::Axis(name) => {
                if dim_idx < type_dims.len()
                    && !name.as_bytes()[0].is_ascii_digit()
                    && let Some(val) = dim_value(&type_dims[dim_idx])
                {
                    result.insert(name.clone(), val);
                }
                dim_idx += 1;
            }
        }
    }
    result
}

fn infer_axes(m: &mut Module) {
    for f in m.functions.values_mut() {
        // Type environment: wire_name → TensorType
        let mut type_env: HashMap<String, TensorType> = HashMap::new();
        for p in &f.params {
            type_env.insert(p.name.clone(), p.ty.clone());
        }

        for op in &mut f.ops {
            // Record output types for downstream ops
            for (out_name, out_type) in op.outputs.iter().zip(op.output_types.iter()) {
                if let Some(t) = out_type {
                    type_env.insert(out_name.clone(), t.clone());
                }
            }

            let output_type = match &op.output_types[0] {
                Some(t) => t.clone(),
                None => continue,
            };

            let pattern = match &op.kind {
                OpKind::View { pattern, axes } if axes.is_empty() => pattern.clone(),
                OpKind::Tile { pattern, axes } if axes.is_empty() => pattern.clone(),
                _ => continue,
            };

            if !pattern.contains("->") {
                continue;
            }

            let (input_str, output_str) = pattern.split_once("->").unwrap();

            // Collect input free axes
            let (in_has_ellipsis, in_elements) = parse_pattern_side(input_str);
            let mut input_free = std::collections::HashSet::new();
            for elem in &in_elements {
                if let PatternElem::Axis(name) = elem {
                    input_free.insert(name.clone());
                }
            }

            // Resolve input axis sizes from input wire types
            let mut input_sizes = HashMap::new();
            if let Some(Atom::Name(first_arg)) = op.args.first()
                && let Some(input_type) = type_env.get(first_arg)
            {
                input_sizes = match_axes_to_type(in_has_ellipsis, &in_elements, &input_type.shape);
            }

            // Parse output side
            let (out_has_ellipsis, out_elements) = parse_pattern_side(output_str);
            let type_dims = &output_type.shape;
            let skip = if out_has_ellipsis {
                type_dims.len() - out_elements.len()
            } else {
                0
            };

            let mut axes = IndexMap::new();
            let mut dim_idx = skip;
            for elem in &out_elements {
                match elem {
                    PatternElem::Group(group) => {
                        if dim_idx < type_dims.len()
                            && let Some(total) = dim_value(&type_dims[dim_idx])
                        {
                            let mut known_product: i64 = 1;
                            let mut unknowns = Vec::new();
                            for axis in group {
                                if let Some(&size) = input_sizes.get(axis) {
                                    known_product *= size;
                                } else {
                                    unknowns.push(axis.clone());
                                }
                            }
                            if unknowns.len() == 1 && known_product > 0 {
                                axes.insert(unknowns[0].clone(), total / known_product);
                            }
                        }
                        dim_idx += 1;
                    }
                    PatternElem::Axis(name) => {
                        if dim_idx < type_dims.len()
                            && !name.as_bytes()[0].is_ascii_digit()
                            && !input_free.contains(name)
                            && let Some(val) = dim_value(&type_dims[dim_idx])
                        {
                            axes.insert(name.clone(), val);
                        }
                        dim_idx += 1;
                    }
                }
            }

            if !axes.is_empty() {
                match &mut op.kind {
                    OpKind::View { axes: a, .. } => *a = axes,
                    OpKind::Tile { axes: a, .. } => *a = axes,
                    _ => {}
                }
            }
        }
    }
}

// ── Config value ────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    Int(i64),
    Float(f64),
    Vec(Vec<f64>),
}

// ── Public API ──────────────────────────────────────────────────────

pub fn resolve(m: &Module, config: &HashMap<String, ConfigValue>) -> Module {
    let mut resolved = substitute_params(m, config);
    infer_axes(&mut resolved);
    resolved
}
