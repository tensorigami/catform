use indexmap::IndexMap;

use crate::ast::*;

// ── Tokenizer ───────────────────────────────────────────────────────

fn tokenize(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Skip whitespace
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // String literal
        if b == b'"' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            tokens.push(source[start..i].to_string());
            continue;
        }
        // Comment
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            tokens.push(source[start..i].to_string());
            continue;
        }
        // Arrow ->
        if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            tokens.push("->".to_string());
            i += 2;
            continue;
        }
        // Number (possibly negative)
        if b.is_ascii_digit() || (b == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            if b == b'-' {
                i += 1;
            }
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            tokens.push(source[start..i].to_string());
            continue;
        }
        // Identifier (may contain dots: param.hidden, model.layers)
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            tokens.push(source[start..i].to_string());
            continue;
        }
        // Single-char punctuation
        if b"{}()[],:=+-*/".contains(&b) {
            tokens.push(String::from(b as char));
            i += 1;
            continue;
        }
        // Skip unknown
        i += 1;
    }
    tokens
}

// ── Token stream ────────────────────────────────────────────────────

struct Stream {
    tokens: Vec<String>,
    pos: usize,
    pending_comments: Vec<String>,
}

impl Stream {
    fn new(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            pos: 0,
            pending_comments: Vec::new(),
        }
    }

    fn skip_comments(&mut self) {
        while self.pos < self.tokens.len() && self.tokens[self.pos].starts_with("//") {
            self.pending_comments.push(self.tokens[self.pos].clone());
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<&str> {
        self.skip_comments();
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn advance(&mut self) -> String {
        self.skip_comments();
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &str) {
        let tok = self.advance();
        if tok != expected {
            panic!("Expected {expected:?}, got {tok:?}");
        }
    }

    fn take_comments(&mut self) -> Vec<String> {
        self.skip_comments();
        std::mem::take(&mut self.pending_comments)
    }
}

// ── Atom parsing ────────────────────────────────────────────────────

fn parse_atom(tok: &str) -> Atom {
    if tok == "inf" {
        return Atom::Float(f64::INFINITY);
    }
    // Float: non-alpha start with dot or exponent
    if !tok.as_bytes()[0].is_ascii_alphabetic()
        && (tok.contains('.') || tok.contains('e') || tok.contains('E'))
    {
        return Atom::Float(tok.parse::<f64>().unwrap());
    }
    if let Ok(n) = tok.parse::<i64>() {
        return Atom::Int(n);
    }
    Atom::Name(tok.to_string())
}

// ── Type parsing ────────────────────────────────────────────────────

fn parse_type(s: &mut Stream) -> TensorType {
    let dtype = s.advance();
    s.expect("[");
    let mut shape = Vec::new();
    while s.peek() != Some("]") {
        if s.peek() == Some(",") {
            s.advance();
        }
        let tok = s.advance();
        if tok.chars().next().is_some_and(|c| c.is_ascii_digit())
            || (tok.starts_with('-')
                && tok.len() > 1
                && tok[1..].chars().next().is_some_and(|c| c.is_ascii_digit()))
        {
            shape.push(Dim::Concrete(tok.parse::<i64>().unwrap()));
        } else {
            shape.push(Dim::Named(tok));
        }
    }
    s.expect("]");
    TensorType { dtype, shape }
}

// ── Delimited argument parsing ──────────────────────────────────────

fn parse_delimited(s: &mut Stream, open: &str, close: &str) -> (Vec<Atom>, IndexMap<String, Atom>) {
    s.expect(open);
    let mut args = Vec::new();
    let mut kwargs = IndexMap::new();

    while s.peek() != Some(close) {
        if s.peek() == Some(",") {
            s.advance();
        }
        let tok = s.advance();

        // Keyword arg: identifier followed by '='
        if s.peek() == Some("=") && tok.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            s.advance(); // consume '='
            let val_tok = s.advance();
            if val_tok.starts_with('"') && val_tok.ends_with('"') {
                kwargs.insert(tok, Atom::Name(val_tok[1..val_tok.len() - 1].to_string()));
            } else {
                kwargs.insert(tok, parse_atom(&val_tok));
            }
        } else {
            // Positional arg: string literal or atom
            if tok.starts_with('"') && tok.ends_with('"') {
                args.push(Atom::Name(tok[1..tok.len() - 1].to_string()));
            } else {
                args.push(parse_atom(&tok));
            }
        }
    }
    s.expect(close);
    (args, kwargs)
}

// ── Literal parsing ─────────────────────────────────────────────────

fn parse_literal_scalar(s: &mut Stream) -> LiteralValue {
    let tok = s.advance();
    if tok == "-" {
        let nxt = s.advance();
        if nxt == "inf" {
            return LiteralValue::Float(f64::NEG_INFINITY);
        }
        panic!("Expected 'inf' after '-', got {nxt:?}");
    }
    if tok == "inf" {
        return LiteralValue::Float(f64::INFINITY);
    }
    match parse_atom(&tok) {
        Atom::Int(n) => LiteralValue::Int(n),
        Atom::Float(f) => LiteralValue::Float(f),
        Atom::Name(n) => panic!("Expected literal value, got name {n:?}"),
    }
}

fn parse_list_literal(s: &mut Stream) -> LiteralValue {
    s.expect("[");
    let mut items = Vec::new();
    while s.peek() != Some("]") {
        if s.peek() == Some(",") {
            s.advance();
        }
        if s.peek() == Some("[") {
            items.push(parse_list_literal(s));
        } else {
            items.push(parse_literal_scalar(s));
        }
    }
    s.expect("]");
    LiteralValue::List(items)
}

fn parse_literal_value(s: &mut Stream) -> LiteralValue {
    if s.peek() == Some("[") {
        parse_list_literal(s)
    } else {
        parse_literal_scalar(s)
    }
}

// ── Function parsing ────────────────────────────────────────────────

fn parse_fn(s: &mut Stream) -> Function {
    let name = s.advance();
    s.expect("(");

    let mut params = Vec::new();
    while s.peek() != Some(")") {
        if s.peek() == Some(",") {
            s.advance();
        }
        let pname = s.advance();
        s.expect(":");
        if s.peek() == Some("*") {
            s.advance();
            params.push(Param {
                name: pname,
                ty: TensorType {
                    dtype: "*".to_string(),
                    shape: vec![],
                },
            });
        } else {
            let ptype = parse_type(s);
            params.push(Param { name: pname, ty: ptype });
        }
    }
    s.expect(")");

    s.expect("->");
    s.expect("(");
    let mut returns = Vec::new();
    while s.peek() != Some(")") {
        if s.peek() == Some(",") {
            s.advance();
        }
        let rname = s.advance();
        s.expect(":");
        let rtype = parse_type(s);
        returns.push(Param { name: rname, ty: rtype });
    }
    s.expect(")");

    s.expect("{");

    // Docstring: comments immediately after { belong to the function
    let fn_comments = s.take_comments();

    let mut ops = Vec::new();
    while s.peek() != Some("}") {
        let comments = s.take_comments();

        // Multi-output: (name, name): (type, type) = ...
        let (out_names, output_types) = if s.peek() == Some("(") {
            s.advance();
            let mut names = Vec::new();
            while s.peek() != Some(")") {
                if s.peek() == Some(",") {
                    s.advance();
                }
                names.push(s.advance());
            }
            s.expect(")");
            s.expect(":");
            s.expect("(");
            let mut types = Vec::new();
            while s.peek() != Some(")") {
                if s.peek() == Some(",") {
                    s.advance();
                }
                types.push(Some(parse_type(s)));
            }
            s.expect(")");
            (names, types)
        } else {
            // Single output: name [: type] = ...
            let out_name = s.advance();
            let out_type = if s.peek() == Some(":") {
                s.advance();
                Some(parse_type(s))
            } else {
                None
            };
            (vec![out_name], vec![out_type])
        };

        s.expect("=");
        let op_name = s.advance();

        // literal(value)
        if op_name == "literal" {
            s.expect("(");
            let value = parse_literal_value(s);
            s.expect(")");
            ops.push(Op {
                kind: OpKind::Literal { value },
                outputs: out_names,
                output_types,
                args: Vec::new(),
                comments,
            });
            continue;
        }

        // Optional bracket specifiers: op[...]
        let mut bracket_args = Vec::new();
        let mut bracket_kwargs = IndexMap::new();
        if s.peek() == Some("[") {
            let (ba, bk) = parse_delimited(s, "[", "]");
            bracket_args = ba;
            bracket_kwargs = bk;
        }

        // Build specifiers from bracket args
        let mut pattern: Option<String> = None;
        let mut function: Option<String> = None;
        for arg in &bracket_args {
            if let Atom::Name(n) = arg {
                if n.contains(' ') || n.contains("->") {
                    pattern = Some(n.clone());
                } else {
                    function = Some(n.clone());
                }
            }
        }

        // Optional paren args: op(...)
        let mut paren_args: Vec<Atom> = Vec::new();
        let mut paren_kwargs = IndexMap::new();
        if s.peek() == Some("(") {
            let (pa, pk) = parse_delimited(s, "(", ")");
            paren_args = pa;
            paren_kwargs = pk;
        }

        // Convert paren kwargs to axes (i64 values)
        let mut axes = IndexMap::new();
        for (k, v) in &paren_kwargs {
            match v {
                Atom::Int(n) => {
                    axes.insert(k.clone(), *n);
                }
                Atom::Float(f) => {
                    axes.insert(k.clone(), *f as i64);
                }
                _ => {}
            }
        }

        // Build OpKind from op name + specifiers
        let kind = build_op_kind(&op_name, pattern, function.clone(), &bracket_kwargs, axes);

        ops.push(Op {
            kind,
            outputs: out_names,
            output_types,
            args: paren_args,
            comments,
        });
    }

    s.expect("}");

    Function {
        name,
        comments: fn_comments,
        params,
        returns,
        ops,
    }
}

fn build_op_kind(
    op: &str,
    pattern: Option<String>,
    function: Option<String>,
    _bracket_kwargs: &IndexMap<String, Atom>,
    axes: IndexMap<String, i64>,
) -> OpKind {
    match op {
        "view" => OpKind::View {
            pattern: pattern.unwrap_or_default(),
            axes,
        },
        "map" => OpKind::Map {
            function: function.unwrap_or_default(),
        },
        "fold" => OpKind::Fold {
            pattern: pattern.unwrap_or_default(),
            function: function.unwrap_or_default(),
        },
        "tile" => OpKind::Tile {
            pattern: pattern.unwrap_or_default(),
            axes,
        },
        "gather" => OpKind::Gather {
            pattern: pattern.unwrap_or_default(),
        },
        "scatter" => OpKind::Scatter {
            pattern: pattern.unwrap_or_default(),
        },
        "contract" => OpKind::Contract {
            pattern: pattern.unwrap_or_default(),
        },
        "random" => OpKind::Random {
            function: function.unwrap_or_default(),
        },
        "call" => OpKind::Call {
            target: function.unwrap_or_default(),
        },
        "loop" => OpKind::Loop {
            target: function.unwrap_or_default(),
            count: None,
        },
        _ => panic!("Unknown op: {op:?}"),
    }
}

// ── Public API ──────────────────────────────────────────────────────

pub fn parse(source: &str) -> Module {
    let tokens = tokenize(source);
    let mut s = Stream::new(tokens);

    let header_comments = s.take_comments();
    let mut functions = IndexMap::new();

    while s.peek().is_some() {
        let pre_comments = s.take_comments();
        let mut f = parse_fn(&mut s);
        let mut merged = pre_comments;
        merged.append(&mut f.comments);
        f.comments = merged;
        functions.insert(f.name.clone(), f);
    }

    Module {
        header_comments,
        functions,
    }
}

pub fn parse_file(path: &str) -> Module {
    let source = std::fs::read_to_string(path).unwrap();
    parse(&source)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const RMSNORM: &str = r#"
rmsnorm(x: bf16[N, param.hidden], w: bf16[param.hidden]) -> (out: bf16[N, param.hidden]) {
  x_f: f32[N, param.hidden]  = map[f32](x)
  sq : f32[N, param.hidden]  = map[mul](x_f, x_f)
  var: f32[N, 1]             = fold["... N d -> ... N 1", mean](sq)
  ve : f32[N, 1]             = map[add](var, param.rms_norm_eps)
  rs : f32[N, 1]             = map[rsqrt](ve)
  xn : f32[N, param.hidden]  = map[mul](x_f, rs)
  xb : bf16[N, param.hidden] = map[bf16](xn)
  out: bf16[N, param.hidden] = map[mul](xb, w)
}
"#;

    #[test]
    fn parse_rmsnorm() {
        let m = parse(RMSNORM);
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions["rmsnorm"];
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.returns.len(), 1);
        assert_eq!(f.ops.len(), 8);

        // First op: map[f32](x)
        match &f.ops[0].kind {
            OpKind::Map { function } => assert_eq!(function, "f32"),
            other => panic!("Expected Map, got {other:?}"),
        }

        // Third op: fold["... N d -> ... N 1", mean](sq)
        match &f.ops[2].kind {
            OpKind::Fold { pattern, function } => {
                assert_eq!(pattern, "... N d -> ... N 1");
                assert_eq!(function, "mean");
            }
            other => panic!("Expected Fold, got {other:?}"),
        }
    }

    const ROPE: &str = r#"
rope_rotate(
  x  : bf16[N, H, param.head_dim],
  cos: bf16[N, param.rope_dim],
  sin: bf16[N, param.rope_dim]
) -> (out: bf16[N, H, param.head_dim]) {
  rot_I : f32[2, 2] = literal([[1, 0], [0, 1]])
  rot_J : f32[2, 2] = literal([[0, -1], [1, 0]])
  rI    : bf16[2, 2] = map[bf16](rot_I)
  rJ    : bf16[2, 2] = map[bf16](rot_J)
  cos_t : bf16[N, param.rope_dim, 2, 2] = tile["N d -> N d i j"](cos)
  cos_r : bf16[N, param.rope_dim, 2, 2] = map[mul](cos_t, rI)
  sin_t : bf16[N, param.rope_dim, 2, 2] = tile["N d -> N d i j"](sin)
  sin_r : bf16[N, param.rope_dim, 2, 2] = map[mul](sin_t, rJ)
  rot   : bf16[N, param.rope_dim, 2, 2] = map[add](cos_r, sin_r)
  pairs : bf16[N, H, param.rope_dim, 2] = view["... (two d) -> ... d two"](x)
  result: bf16[N, H, param.rope_dim, 2] = contract["... d i j, ... H d j -> ... H d i"](rot, pairs)
  out   : bf16[N, H, param.head_dim]    = view["... d two -> ... (two d)"](result)
}
"#;

    #[test]
    fn parse_rope_with_literals() {
        let m = parse(ROPE);
        let f = &m.functions["rope_rotate"];
        assert_eq!(f.ops.len(), 12);

        // First op: literal
        match &f.ops[0].kind {
            OpKind::Literal { value } => match value {
                LiteralValue::List(rows) => {
                    assert_eq!(rows.len(), 2);
                    match &rows[0] {
                        LiteralValue::List(cols) => assert_eq!(cols.len(), 2),
                        _ => panic!("Expected nested list"),
                    }
                }
                _ => panic!("Expected list literal"),
            },
            other => panic!("Expected Literal, got {other:?}"),
        }
    }

    const MULTI_OUTPUT: &str = r#"
main(pos: f32[N], tokens: int32[N]) -> (logits: bf16[N, param.vocab]) {
  emb       : bf16[N, param.hidden]                              = call[embed](tokens, model.embed_tokens.weight)
  (cos, sin): (bf16[N, param.rope_dim], bf16[N, param.rope_dim]) = call[rope_setup](pos)
  logits    : bf16[N, param.vocab]                                = call[unembed](emb, model.embed_tokens.weight)
}
"#;

    #[test]
    fn parse_multi_output() {
        let m = parse(MULTI_OUTPUT);
        let f = &m.functions["main"];
        assert_eq!(f.ops.len(), 3);

        // Second op: multi-output call
        assert_eq!(f.ops[1].outputs, vec!["cos", "sin"]);
        assert_eq!(f.ops[1].output_types.len(), 2);
        assert!(f.ops[1].output_types[0].is_some());
        assert!(f.ops[1].output_types[1].is_some());
    }

    const LOOP: &str = r#"
transformer(
  x     : bf16[N, param.hidden],
  cos   : bf16[N, param.rope_dim],
  sin   : bf16[N, param.rope_dim],
  pos   : f32[N],
  w     : *
) -> (out: bf16[N, param.hidden]) {
  x  : bf16[N, param.hidden] = loop[layer](x, cos, sin, pos, w.layer)
  out: bf16[N, param.hidden] = call[rmsnorm](x, w.norm)
}
"#;

    #[test]
    fn parse_loop() {
        let m = parse(LOOP);
        let f = &m.functions["transformer"];
        assert_eq!(f.ops.len(), 2);
        match &f.ops[0].kind {
            OpKind::Loop { target, count } => {
                assert_eq!(target, "layer");
                assert_eq!(*count, None);
            }
            other => panic!("Expected Loop, got {other:?}"),
        }
        // Output and threaded arg share the same name
        assert_eq!(f.ops[0].outputs, vec!["x"]);
        assert_eq!(f.ops[0].args[0], Atom::Name("x".to_string()));
        assert_eq!(f.ops[0].args[1], Atom::Name("cos".to_string()));
        assert_eq!(f.ops[0].args[4], Atom::Name("w.layer".to_string()));

        // Dict param
        let w = &f.params[4];
        assert_eq!(w.name, "w");
        assert_eq!(w.ty.dtype, "*");
        assert!(w.ty.shape.is_empty());

        // Regular params have normal types
        assert_eq!(f.params[0].ty.dtype, "bf16");
    }

    #[test]
    fn parse_qwen3() {
        let source = std::fs::read_to_string("../models/qwen3/model.cat").unwrap();
        let m = parse(&source);
        assert_eq!(m.functions.len(), 10);
        assert!(m.functions.contains_key("main"));
        assert!(m.functions.contains_key("rmsnorm"));
        assert!(m.functions.contains_key("attention"));
    }

    #[test]
    fn parse_qwen3_moe() {
        let source = std::fs::read_to_string("../models/qwen3_moe/model.cat").unwrap();
        let m = parse(&source);
        assert_eq!(m.functions.len(), 11);
        assert!(m.functions.contains_key("moe_ffn"));
        assert!(m.functions.contains_key("kill_max"));
    }

    #[test]
    fn json_roundtrip_qwen3() {
        let source = std::fs::read_to_string("../models/qwen3/model.cat").unwrap();
        let m = parse(&source);
        let json = serde_json::to_string(&m).unwrap();
        let m2: crate::ast::Module = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn json_roundtrip_qwen3_moe() {
        let source = std::fs::read_to_string("../models/qwen3_moe/model.cat").unwrap();
        let m = parse(&source);
        let json = serde_json::to_string(&m).unwrap();
        let m2: crate::ast::Module = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn json_roundtrip_resolved() {
        let source = std::fs::read_to_string("../models/qwen3/model.cat").unwrap();
        let m = parse(&source);
        let resolved = crate::resolve::resolve(&m);
        let json = serde_json::to_string(&resolved).unwrap();
        let m2: crate::ast::Module = serde_json::from_str(&json).unwrap();
        assert_eq!(resolved, m2);
    }

    #[test]
    fn json_roundtrip_flat() {
        let source = std::fs::read_to_string("../models/qwen3/model.cat").unwrap();
        let m = parse(&source);
        let resolved = crate::resolve::resolve(&m);
        // Set loop count before flattening (normally done by bridge from config)
        let mut with_count = resolved.clone();
        for f in with_count.functions.values_mut() {
            for op in &mut f.ops {
                if let OpKind::Loop { count, .. } = &mut op.kind {
                    *count = Some(28); // qwen3/0_6b has 28 layers
                }
            }
        }
        let flat = crate::flatten::flatten(&with_count, "main");
        let json = serde_json::to_string(&flat).unwrap();
        let m2: crate::ast::Module = serde_json::from_str(&json).unwrap();
        assert_eq!(flat, m2);
    }
}
