use serde::Deserialize;

use crate::responses::types::FunctionToolCall;

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub output: Vec<OutputContent>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContent {
    #[serde(rename = "message")]
    OutputMessage(OutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct OutputMessage {
    pub id: String,
    pub status: Status,
    pub content: Vec<MessageContent>,
    pub role: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum MessageContent {
    #[serde(rename = "output_text")]
    OutputText(OutputText),
    #[serde(rename = "refusal")]
    Refusal(Refusal),
}

#[derive(Debug, Deserialize)]
pub struct OutputText {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct Refusal {
    pub refusal: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tagged_function_call_deserializes() {
        let payload = json!({
            "type": "function_call",
            "id": "tool_1",
            "call_id": "tool_1",
            "name": "lookup_weather",
            "arguments": "{\"city\":\"Lisbon\"}"
        });

        let parsed: OutputContent =
            serde_json::from_value(payload).expect("function_call should deserialize");

        match parsed {
            OutputContent::FunctionCall(call) => {
                assert_eq!(call.name, "lookup_weather");
                assert_eq!(call.call_id, "tool_1");
            }
            _ => panic!("expected function_call output content"),
        }
    }

    /// The `type` field is consumed by the tagged enum during deserialization,
    /// so `FunctionToolCall.r#type` is populated by its serde default.
    /// Verify it still serializes correctly for use as an API input item.
    #[test]
    fn function_call_round_trips_with_type_field() {
        let payload = json!({
            "type": "function_call",
            "id": "tool_1",
            "call_id": "tool_1",
            "name": "lookup_weather",
            "arguments": "{\"city\":\"Lisbon\"}"
        });

        let parsed: OutputContent = serde_json::from_value(payload).expect("should deserialize");

        let OutputContent::FunctionCall(call) = parsed else {
            panic!("expected function_call");
        };

        let serialized = serde_json::to_value(&call).expect("should serialize");
        assert_eq!(serialized["type"], "function_call");
    }
}
