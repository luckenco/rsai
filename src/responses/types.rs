use serde::{Deserialize, Serialize};

fn default_function_call_type() -> String {
    "function_call".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionToolCall {
    /// Required by the API when serialized as an input item.
    /// The `default` is necessary because `OutputContent`'s `#[serde(tag = "type")]`
    /// consumes this field during deserialization, so serde never sees it on the struct.
    #[serde(rename = "type", default = "default_function_call_type")]
    pub r#type: String,
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}
