use std::collections::HashMap;

use crate::resolve::ConfigValue;

pub fn load_config(path: &str) -> HashMap<String, ConfigValue> {
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let table: toml::Table = content
        .parse()
        .unwrap_or_else(|e| panic!("failed to parse {path}: {e}"));

    let mut config = HashMap::new();
    for section in ["shape", "scalar", "vector"] {
        if let Some(toml::Value::Table(t)) = table.get(section) {
            for (k, v) in t {
                let cv = match v {
                    toml::Value::Integer(n) => ConfigValue::Int(*n),
                    toml::Value::Float(f) => ConfigValue::Float(*f),
                    toml::Value::Array(arr) => {
                        let vals: Vec<f64> = arr
                            .iter()
                            .map(|v| match v {
                                toml::Value::Float(f) => *f,
                                toml::Value::Integer(n) => *n as f64,
                                _ => panic!("unexpected value in vector section"),
                            })
                            .collect();
                        ConfigValue::Vec(vals)
                    }
                    _ => continue,
                };
                config.insert(k.clone(), cv);
            }
        }
    }
    config
}
