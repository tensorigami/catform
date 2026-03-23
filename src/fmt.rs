use crate::ast::*;

// ── Helpers ─────────────────────────────────────────────────────────

fn needs_quoting(s: &str) -> bool {
    s.is_empty()
        || (!s.as_bytes()[0].is_ascii_alphabetic() && s.as_bytes()[0] != b'_')
        || s.bytes()
            .any(|b| !b.is_ascii_alphanumeric() && b != b'_' && b != b'.')
}

fn fmt_type(t: &TensorType) -> String {
    let dims: Vec<String> = t
        .shape
        .iter()
        .map(|d| match d {
            Dim::Concrete(n) => n.to_string(),
            Dim::Named(s) => s.clone(),
        })
        .collect();
    format!("{}[{}]", t.dtype, dims.join(", "))
}

fn fmt_atom(a: &Atom) -> String {
    match a {
        Atom::Int(n) => n.to_string(),
        Atom::Float(f) => format_float(*f),
        Atom::Name(s) if needs_quoting(s) => format!("\"{s}\""),
        Atom::Name(s) => s.clone(),
    }
}

fn format_float(f: f64) -> String {
    if f == f64::INFINITY {
        return "inf".to_string();
    }
    if f == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    // Match Python's float formatting
    let s = format!("{f}");
    // Ensure there's a decimal point
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{f}.0")
    } else {
        s
    }
}

fn fmt_literal(v: &LiteralValue) -> String {
    match v {
        LiteralValue::Int(n) => n.to_string(),
        LiteralValue::Float(f) => format_float(*f),
        LiteralValue::List(items) => {
            let parts: Vec<String> = items.iter().map(fmt_literal).collect();
            format!("[{}]", parts.join(", "))
        }
    }
}

// ── Op RHS ──────────────────────────────────────────────────────────

fn fmt_op_rhs(op: &Op) -> String {
    match &op.kind {
        OpKind::Literal { value } => {
            format!("literal({})", fmt_literal(value))
        }
        OpKind::View { pattern, axes } => {
            let mut bracket = vec![fmt_atom(&Atom::Name(pattern.clone()))];
            for (k, v) in axes {
                bracket.push(format!("{k}={v}"));
            }
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("view[{}]({})", bracket.join(", "), paren.join(", "))
        }
        OpKind::Map { function } => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("map[{function}]({})", paren.join(", "))
        }
        OpKind::Fold { pattern, function } => {
            let bracket_pat = fmt_atom(&Atom::Name(pattern.clone()));
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("fold[{bracket_pat}, {function}]({})", paren.join(", "))
        }
        OpKind::Tile { pattern, axes } => {
            let mut bracket = vec![fmt_atom(&Atom::Name(pattern.clone()))];
            for (k, v) in axes {
                bracket.push(format!("{k}={v}"));
            }
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("tile[{}]({})", bracket.join(", "), paren.join(", "))
        }
        OpKind::Gather { pattern } => {
            let bracket = fmt_atom(&Atom::Name(pattern.clone()));
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("gather[{bracket}]({})", paren.join(", "))
        }
        OpKind::Scatter { pattern } => {
            let bracket = fmt_atom(&Atom::Name(pattern.clone()));
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("scatter[{bracket}]({})", paren.join(", "))
        }
        OpKind::Contract { pattern } => {
            let bracket = fmt_atom(&Atom::Name(pattern.clone()));
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("contract[{bracket}]({})", paren.join(", "))
        }
        OpKind::Random { function } => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("random[{function}]({})", paren.join(", "))
        }
        OpKind::Call { target } => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("call[{target}]({})", paren.join(", "))
        }
        OpKind::Loop { target, count } => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            let bracket = match count {
                LoopCount::Concrete(n) => format!("{target}, {n}"),
                LoopCount::Named(s) => format!("{target}, {s}"),
            };
            format!("loop[{bracket}]({})", paren.join(", "))
        }
    }
}

// ── Signature ───────────────────────────────────────────────────────

fn fmt_param_type(p: &Param) -> String {
    if p.ty.dtype == "*" {
        "*".to_string()
    } else {
        fmt_type(&p.ty)
    }
}

fn fmt_signature(f: &Function, width: usize) -> Vec<String> {
    let param_strs: Vec<String> = f
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, fmt_param_type(p)))
        .collect();
    let ret_strs: Vec<String> = f
        .returns
        .iter()
        .map(|r| format!("{}: {}", r.name, fmt_type(&r.ty)))
        .collect();
    let ret_part = format!("({})", ret_strs.join(", "));

    let single = format!("{}({}) -> {} {{", f.name, param_strs.join(", "), ret_part);
    if single.len() <= width {
        return vec![single];
    }

    // Multi-line: align ':' across params
    let max_name = f.params.iter().map(|p| p.name.len()).max().unwrap_or(0);
    let mut lines = vec![format!("{}(", f.name)];
    for (i, p) in f.params.iter().enumerate() {
        let pad = " ".repeat(max_name - p.name.len());
        let comma = if i < f.params.len() - 1 { "," } else { "" };
        lines.push(format!("  {}{pad}: {}{comma}", p.name, fmt_param_type(p)));
    }
    lines.push(format!(") -> {ret_part} {{"));
    lines
}

// ── Body ────────────────────────────────────────────────────────────

fn is_multi_output(op: &Op) -> bool {
    op.outputs.len() > 1
}

fn name_str(op: &Op) -> String {
    if is_multi_output(op) {
        format!("({})", op.outputs.join(", "))
    } else {
        op.outputs[0].clone()
    }
}

fn type_str(op: &Op) -> String {
    if is_multi_output(op) {
        let types: Vec<String> = op
            .output_types
            .iter()
            .filter_map(|t| t.as_ref().map(fmt_type))
            .collect();
        if types.is_empty() {
            String::new()
        } else {
            format!("({})", types.join(", "))
        }
    } else {
        op.output_types[0].as_ref().map_or(String::new(), fmt_type)
    }
}

fn fmt_body(f: &Function) -> Vec<String> {
    if f.ops.is_empty() {
        return Vec::new();
    }

    let max_name = f.ops.iter().map(|op| name_str(op).len()).max().unwrap_or(0);
    let max_type = f.ops.iter().map(|op| type_str(op).len()).max().unwrap_or(0);

    let mut lines = Vec::new();
    for (i, op) in f.ops.iter().enumerate() {
        if !op.comments.is_empty() {
            if i > 0 {
                lines.push(String::new());
            }
            for c in &op.comments {
                lines.push(format!("  {c}"));
            }
        }

        let lhs = name_str(op);
        let rhs = fmt_op_rhs(op);
        let name_pad = " ".repeat(max_name - lhs.len());

        let ts = type_str(op);
        if !ts.is_empty() {
            let type_pad = " ".repeat(max_type - ts.len());
            lines.push(format!("  {lhs}{name_pad}: {ts}{type_pad} = {rhs}"));
        } else if max_type > 0 {
            let type_pad = " ".repeat(max_type);
            lines.push(format!("  {lhs}{name_pad}  {type_pad} = {rhs}"));
        } else {
            lines.push(format!("  {lhs}{name_pad} = {rhs}"));
        }
    }
    lines
}

// ── Public API ──────────────────────────────────────────────────────

pub fn format_cat(m: &Module, width: usize) -> String {
    let mut lines: Vec<String> = Vec::new();

    for c in &m.header_comments {
        lines.push(c.clone());
    }
    if !m.header_comments.is_empty() {
        lines.push(String::new());
    }

    let mut first = true;
    for f in m.functions.values() {
        if !first {
            lines.push(String::new());
        }
        first = false;

        lines.extend(fmt_signature(f, width));
        for c in &f.comments {
            lines.push(format!("  {c}"));
        }
        lines.extend(fmt_body(f));
        lines.push("}".to_string());
    }

    lines.join("\n") + "\n"
}

