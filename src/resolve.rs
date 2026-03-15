use std::collections::HashMap;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use crate::ast::*;

static PATTERN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\.\.\.|\(|\)|[a-zA-Z_]\w*|\d+").unwrap());

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

// ── Public API ──────────────────────────────────────────────────────

pub fn resolve(m: &Module) -> Module {
    let mut resolved = m.clone();
    infer_axes(&mut resolved);
    resolved
}
