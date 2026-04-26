use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::responses::types::FunctionToolCall;

#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    pub input: Vec<InputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    pub text: Format,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_tool_choice")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_tools")]
    pub tools: Option<Box<[Tool]>>,

    /// An integer between 0 and 20
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Alter this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,

    /// Used to boost cache hit rates by better bucketing similar requests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Custom serializer for ToolChoice to match API format
fn serialize_tool_choice<S>(
    tool_choice: &Option<ToolChoice>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    match tool_choice {
        Some(tc) => {
            let serializable = create_tool_choice(tc.clone());
            serializable.serialize(serializer)
        }
        None => serializer.serialize_none(),
    }
}

/// Custom serializer for Tools to match API format
fn serialize_tools<S>(tools: &Option<Box<[Tool]>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    match tools {
        Some(tools) => {
            let serializable_tools: Vec<SerializableTool> =
                tools.iter().map(create_serializable_tool).collect();
            serializable_tools.serialize(serializer)
        }
        None => serializer.serialize_none(),
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum InputItem {
    Message(InputMessage),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolCallOutput),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

impl From<crate::core::ToolChoice> for ToolChoice {
    fn from(value: crate::core::ToolChoice) -> Self {
        match value {
            crate::core::ToolChoice::None => ToolChoice::None,
            crate::core::ToolChoice::Auto => ToolChoice::Auto,
            crate::core::ToolChoice::Required => ToolChoice::Required,
            crate::core::ToolChoice::Function { name } => ToolChoice::Function { name },
        }
    }
}

/// Serialize ToolChoice to match API expectations
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum SerializableToolChoice {
    Mode(ToolMode),
    Definite(FunctionToolChoice),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Serialize)]
struct FunctionToolChoice {
    name: String,
    #[serde(rename = "type")]
    r#type: FunctionType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FunctionType {
    Function,
}

/// Convert core ToolChoice to serializable format for API
fn create_tool_choice(tool_choice: ToolChoice) -> SerializableToolChoice {
    match tool_choice {
        ToolChoice::None => SerializableToolChoice::Mode(ToolMode::None),
        ToolChoice::Auto => SerializableToolChoice::Mode(ToolMode::Auto),
        ToolChoice::Required => SerializableToolChoice::Mode(ToolMode::Required),
        ToolChoice::Function { name } => SerializableToolChoice::Definite(FunctionToolChoice {
            name,
            r#type: FunctionType::Function,
        }),
    }
}

/// Tool struct for internal use
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

/// Serialize Tool to match API expectations with proper envelope
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum SerializableTool {
    Function(FunctionTool),
}

#[derive(Debug, Serialize)]
struct FunctionTool {
    name: String,
    parameters: Value,
    strict: bool,
    #[serde(rename = "type")]
    r#type: FunctionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

/// Convert core Tool to serializable format for API
fn create_serializable_tool(tool: &Tool) -> SerializableTool {
    let strict = tool.strict.unwrap_or(true);
    SerializableTool::Function(FunctionTool {
        name: tool.name.clone(),
        parameters: tool.parameters.clone(),
        strict,
        r#type: FunctionType::Function,
        description: tool.description.clone(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Format {
    pub format: FormatType,
}

#[derive(Debug, Serialize, Clone)]
pub struct InputMessage {
    pub role: InputMessageRole,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionToolCallOutput {
    pub call_id: String,
    pub output: serde_json::Value,
    #[serde(rename = "type")]
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum FormatType {
    Text {
        #[serde(rename = "type")]
        r#type: TextType,
    },
    JsonSchema(JsonSchema),
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum TextType {
    Text,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonSchema {
    pub name: String,

    pub schema: serde_json::Value,

    #[serde(rename = "type")]
    pub r#type: JsonSchemaType,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonSchemaType {
    JsonSchema,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_tool_choice_serialization() {
        // Test ToolChoice::None serializes to "none"
        let tool_choice_none = ToolChoice::None;
        let serialized = serde_json::to_string(&create_tool_choice(tool_choice_none)).unwrap();
        assert_eq!(serialized, "\"none\"");

        // Test ToolChoice::Auto serializes to "auto"
        let tool_choice_auto = ToolChoice::Auto;
        let serialized = serde_json::to_string(&create_tool_choice(tool_choice_auto)).unwrap();
        assert_eq!(serialized, "\"auto\"");

        // Test ToolChoice::Required serializes to "required"
        let tool_choice_required = ToolChoice::Required;
        let serialized = serde_json::to_string(&create_tool_choice(tool_choice_required)).unwrap();
        assert_eq!(serialized, "\"required\"");

        // Test ToolChoice::Function serializes to proper object shape
        let tool_choice_function = ToolChoice::Function {
            name: "test_function".to_string(),
        };
        let serialized = serde_json::to_string(&create_tool_choice(tool_choice_function)).unwrap();
        let expected = r#"{"name":"test_function","type":"function"}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_tool_serialization() {
        let tool = Tool {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "param1": { "type": "string" }
                }
            }),
            strict: Some(true),
        };

        let serialized = serde_json::to_string(&create_serializable_tool(&tool)).unwrap();

        // Should serialize with proper envelope: {"type":"function","function":{...}}
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        // Check that it has the function wrapper
        assert!(parsed.get("type").is_some());
        assert_eq!(parsed["type"], "function");

        // Check the inner function object
        let function_obj = &parsed;
        assert_eq!(function_obj["name"], "test_tool");
        assert_eq!(function_obj["description"], "A test tool");
        assert_eq!(function_obj["strict"], true);
        assert!(function_obj["parameters"].is_object());
    }
}
