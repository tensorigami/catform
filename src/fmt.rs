use crate::ast::*;

// ── Helpers ─────────────────────────────────────────────────────────

// A pattern is always a multi-token string ("... n d -> ... n d") — it can
// never be a bare identifier, so it is quoted unconditionally. The op kind
// (View/Fold/Tile/Gather/Scatter/Contract) is what makes it a pattern; we
// quote by position, not by sniffing the bytes.
fn fmt_pattern(p: &str) -> String {
    format!("\"{p}\"")
}

fn fmt_type(t: &TensorType) -> String {
    if t.dtype == "*" {
        return "*".to_string();
    }
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
        OpKind::Iota => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("iota({})", paren.join(", "))
        }
        OpKind::View { pattern, axes } => {
            let mut bracket = vec![fmt_pattern(pattern)];
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
        OpKind::Fold { pattern, reduction } => {
            let bracket_pat = fmt_pattern(pattern);
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("fold[{bracket_pat}, {reduction}]({})", paren.join(", "))
        }
        OpKind::Tile { pattern, axes } => {
            let mut bracket = vec![fmt_pattern(pattern)];
            for (k, v) in axes {
                bracket.push(format!("{k}={v}"));
            }
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("tile[{}]({})", bracket.join(", "), paren.join(", "))
        }
        OpKind::Gather { pattern } => {
            let bracket = fmt_pattern(pattern);
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("gather[{bracket}]({})", paren.join(", "))
        }
        OpKind::Scatter { pattern, reduction } => {
            let bracket_pat = fmt_pattern(pattern);
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("scatter[{bracket_pat}, {reduction}]({})", paren.join(", "))
        }
        OpKind::Contract { pattern } => {
            let bracket = fmt_pattern(pattern);
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            format!("contract[{bracket}]({})", paren.join(", "))
        }
        OpKind::Random { lower, upper } => {
            format!("random[{}, {}]", format_float(*lower), format_float(*upper))
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
        OpKind::Cache { var, init, extend } => {
            let paren: Vec<String> = op.args.iter().map(fmt_atom).collect();
            let bracket = match extend {
                Some(e) => format!("{var}, {init}, {e}"),
                None => format!("{var}, {init}"),
            };
            format!("cache[{bracket}]({})", paren.join(", "))
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

/// A binding is one (name, type) row. A single-output op has one binding; a
/// multi-output op has one per output, laid out on its own line (no-paren),
/// with the `= rhs` attached to the last row.
fn binding_rows(op: &Op) -> Vec<(String, String)> {
    op.outputs
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let ty = op
                .output_types
                .get(i)
                .and_then(|t| t.as_ref())
                .map_or(String::new(), fmt_type);
            (name.clone(), ty)
        })
        .collect()
}

fn fmt_body(f: &Function) -> Vec<String> {
    if f.ops.is_empty() {
        return Vec::new();
    }

    // Align `:` and `=` across every binding row in the function (multi-output
    // rows participate individually, so one long combined type can't blow up
    // the alignment).
    let mut max_name = 0;
    let mut max_type = 0;
    for op in &f.ops {
        for (name, ty) in binding_rows(op) {
            max_name = max_name.max(name.len());
            max_type = max_type.max(ty.len());
        }
    }

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

        let rhs = fmt_op_rhs(op);
        let rows = binding_rows(op);
        let last = rows.len() - 1;
        for (j, (name, ty)) in rows.iter().enumerate() {
            let name_pad = " ".repeat(max_name - name.len());
            if j < last {
                // Intermediate output of a multi-output op: `name: type,`
                lines.push(format!("  {name}{name_pad}: {ty},"));
            } else if !ty.is_empty() {
                let type_pad = " ".repeat(max_type - ty.len());
                lines.push(format!("  {name}{name_pad}: {ty}{type_pad} = {rhs}"));
            } else if max_type > 0 {
                let type_pad = " ".repeat(max_type);
                lines.push(format!("  {name}{name_pad}  {type_pad} = {rhs}"));
            } else {
                lines.push(format!("  {name}{name_pad} = {rhs}"));
            }
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

