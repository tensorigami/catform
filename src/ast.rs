use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ── Dimensions ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Concrete(i64),
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorType {
    pub dtype: String,
    pub shape: Vec<Dim>,
}

// ── Values ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Atom {
    Int(i64),
    Float(f64),
    Name(String),
}

/// Literal values: integers, floats (including inf/-inf), or nested lists.
/// Custom serde: JSON has no infinity representation, so inf/-inf serialize as strings.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    List(Vec<LiteralValue>),
}

impl Serialize for LiteralValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            LiteralValue::Int(n) => s.serialize_i64(*n),
            LiteralValue::Float(f) if f.is_infinite() && *f > 0.0 => s.serialize_str("inf"),
            LiteralValue::Float(f) if f.is_infinite() => s.serialize_str("-inf"),
            LiteralValue::Float(f) if f.is_nan() => s.serialize_str("nan"),
            LiteralValue::Float(f) => s.serialize_f64(*f),
            LiteralValue::List(items) => items.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for LiteralValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = LiteralValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("integer, float, \"inf\"/\"-inf\", or array")
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(LiteralValue::Int(v))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(LiteralValue::Int(v as i64))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(LiteralValue::Float(v))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "inf" => Ok(LiteralValue::Float(f64::INFINITY)),
                    "-inf" => Ok(LiteralValue::Float(f64::NEG_INFINITY)),
                    "nan" => Ok(LiteralValue::Float(f64::NAN)),
                    _ => Err(E::custom(format!("unexpected string: {v}"))),
                }
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(LiteralValue::List(items))
            }
        }
        deserializer.deserialize_any(V)
    }
}

// ── Ops — the sum type ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    View {
        pattern: String,
        axes: IndexMap<String, i64>,
    },
    Map {
        function: String,
    },
    Fold {
        pattern: String,
        function: String,
    },
    Tile {
        pattern: String,
        axes: IndexMap<String, i64>,
    },
    Gather {
        pattern: String,
    },
    Scatter {
        pattern: String,
    },
    Contract {
        pattern: String,
    },
    Literal {
        value: LiteralValue,
    },
    Random {
        function: String,
    },
    Call {
        target: String,
    },
    Loop {
        target: String,
        count: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub kind: OpKind,
    pub outputs: Vec<String>,
    pub output_types: Vec<Option<TensorType>>,
    pub args: Vec<Atom>,
    pub comments: Vec<String>,
}

// ── Op: manual serde (avoids #[serde(flatten)] overhead) ────────────

impl Serialize for Op {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;

        match &self.kind {
            OpKind::View { pattern, axes } => {
                map.serialize_entry("kind", "view")?;
                map.serialize_entry("pattern", pattern)?;
                if !axes.is_empty() {
                    map.serialize_entry("axes", axes)?;
                }
            }
            OpKind::Map { function } => {
                map.serialize_entry("kind", "map")?;
                map.serialize_entry("function", function)?;
            }
            OpKind::Fold { pattern, function } => {
                map.serialize_entry("kind", "fold")?;
                map.serialize_entry("pattern", pattern)?;
                map.serialize_entry("function", function)?;
            }
            OpKind::Tile { pattern, axes } => {
                map.serialize_entry("kind", "tile")?;
                map.serialize_entry("pattern", pattern)?;
                if !axes.is_empty() {
                    map.serialize_entry("axes", axes)?;
                }
            }
            OpKind::Gather { pattern } => {
                map.serialize_entry("kind", "gather")?;
                map.serialize_entry("pattern", pattern)?;
            }
            OpKind::Scatter { pattern } => {
                map.serialize_entry("kind", "scatter")?;
                map.serialize_entry("pattern", pattern)?;
            }
            OpKind::Contract { pattern } => {
                map.serialize_entry("kind", "contract")?;
                map.serialize_entry("pattern", pattern)?;
            }
            OpKind::Literal { value } => {
                map.serialize_entry("kind", "literal")?;
                map.serialize_entry("value", value)?;
            }
            OpKind::Random { function } => {
                map.serialize_entry("kind", "random")?;
                map.serialize_entry("function", function)?;
            }
            OpKind::Call { target } => {
                map.serialize_entry("kind", "call")?;
                map.serialize_entry("target", target)?;
            }
            OpKind::Loop { target, count } => {
                map.serialize_entry("kind", "loop")?;
                map.serialize_entry("target", target)?;
                if let Some(c) = count {
                    map.serialize_entry("count", c)?;
                }
            }
        }

        map.serialize_entry("outputs", &self.outputs)?;
        map.serialize_entry("output_types", &self.output_types)?;
        map.serialize_entry("args", &self.args)?;
        if !self.comments.is_empty() {
            map.serialize_entry("comments", &self.comments)?;
        }
        map.end()
    }
}

/// Helper struct for Op deserialization — flat fields, no flatten overhead.
#[derive(Deserialize)]
struct OpRaw {
    kind: String,
    #[serde(default)]
    pattern: String,
    #[serde(default)]
    function: String,
    #[serde(default)]
    axes: IndexMap<String, i64>,
    #[serde(default)]
    value: Option<LiteralValue>,
    #[serde(default)]
    target: String,
    #[serde(default)]
    count: Option<usize>,
    outputs: Vec<String>,
    output_types: Vec<Option<TensorType>>,
    args: Vec<Atom>,
    #[serde(default)]
    comments: Vec<String>,
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let r = OpRaw::deserialize(deserializer)?;
        let kind = match r.kind.as_str() {
            "view" => OpKind::View {
                pattern: r.pattern,
                axes: r.axes,
            },
            "map" => OpKind::Map {
                function: r.function,
            },
            "fold" => OpKind::Fold {
                pattern: r.pattern,
                function: r.function,
            },
            "tile" => OpKind::Tile {
                pattern: r.pattern,
                axes: r.axes,
            },
            "gather" => OpKind::Gather { pattern: r.pattern },
            "scatter" => OpKind::Scatter { pattern: r.pattern },
            "contract" => OpKind::Contract { pattern: r.pattern },
            "literal" => OpKind::Literal {
                value: r.value.unwrap_or(LiteralValue::Int(0)),
            },
            "random" => OpKind::Random {
                function: r.function,
            },
            "call" => OpKind::Call { target: r.target },
            "loop" => OpKind::Loop {
                target: r.target,
                count: r.count,
            },
            other => {
                return Err(serde::de::Error::custom(format!(
                    "unknown op kind: {other}"
                )));
            }
        };
        Ok(Op {
            kind,
            outputs: r.outputs,
            output_types: r.output_types,
            args: r.args,
            comments: r.comments,
        })
    }
}

// ── Functions + Module ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<String>,
    pub params: Vec<Param>,
    pub returns: Vec<Param>,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub header_comments: Vec<String>,
    pub functions: IndexMap<String, Function>,
}
