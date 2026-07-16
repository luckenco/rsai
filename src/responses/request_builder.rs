use schemars::schema_for;

use crate::core::{ChatRole, ConversationMessage, LlmError, StructuredRequest, Tool};

use super::{
    client::{ResponsesClient, ResponsesProviderConfig},
    request::{
        Format, FormatType, FunctionToolCallOutput, InputItem, InputMessage, InputMessageRole,
        JsonSchema, JsonSchemaType, Request, TextType, Tool as RequestTool,
    },
    types::FunctionToolCall,
};

impl<P: ResponsesProviderConfig> ResponsesClient<P> {
    /// Build a responses API request from core request and input.
    pub fn build_request_with_format(
        &self,
        request: &StructuredRequest,
        responses_input: &[InputItem],
        format: Format,
    ) -> Result<Request, LlmError> {
        build_request_payload_with_format(request, responses_input, format)
    }
}

// Kept separate from the client method so request construction is easy to test.
pub(crate) fn build_request_payload_with_format(
    request: &StructuredRequest,
    responses_input: &[InputItem],
    format: Format,
) -> Result<Request, LlmError> {
    let mut api_request = Request {
        model: request.model.clone(),
        input: responses_input.to_vec(),
        text: format,
        parallel_tool_calls: None,
        temperature: None,
        tools: None,
        tool_choice: None,
        instructions: None,
        max_output_tokens: None,
        store: None,
        top_logprobs: None,
        top_p: None,
        truncation: None,
        user: None,
    };

    if let Some(tool_config) = &request.tool_config {
        api_request.tools = tool_config.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(create_function_tool)
                .collect::<Box<[RequestTool]>>()
        });
        api_request.tool_choice = tool_config
            .tool_choice
            .as_ref()
            .map(|choice| choice.clone().into());
        api_request.parallel_tool_calls = tool_config.parallel_tool_calls;
    }

    if let Some(generation_config) = &request.generation_config {
        api_request.temperature = generation_config.temperature;
        api_request.max_output_tokens = generation_config.max_tokens;
        api_request.top_p = generation_config.top_p;
    }

    Ok(api_request)
}

pub(crate) fn create_function_tool(tool: &Tool) -> RequestTool {
    let strict = tool.strict.unwrap_or(true);
    let mut parameters = tool.parameters.clone();

    if strict {
        make_strict(&mut parameters);
    }

    RequestTool {
        name: tool.name.clone(),
        parameters,
        strict: Some(strict),
        description: tool.description.clone(),
    }
}

fn make_strict(schema: &mut serde_json::Value) {
    let serde_json::Value::Object(schema) = schema else {
        return;
    };

    schema.remove("$schema");

    let is_number = match schema.get("type") {
        Some(serde_json::Value::String(value)) => value == "integer" || value == "number",
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .any(|value| value == "integer" || value == "number"),
        _ => false,
    };
    if is_number {
        schema.remove("format");
    }

    if let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    {
        let required = properties
            .keys()
            .cloned()
            .map(serde_json::Value::String)
            .collect();
        schema.insert("required".to_string(), serde_json::Value::Array(required));
        schema.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    for key in ["properties", "$defs", "definitions"] {
        if let Some(values) = schema
            .get_mut(key)
            .and_then(serde_json::Value::as_object_mut)
        {
            for value in values.values_mut() {
                make_strict(value);
            }
        }
    }

    if let Some(items) = schema.get_mut("items") {
        make_strict(items);
    }

    if let Some(one_of) = schema.remove("oneOf") {
        schema.insert("anyOf".to_string(), one_of);
    }

    for key in ["anyOf", "allOf", "prefixItems"] {
        if let Some(values) = schema
            .get_mut(key)
            .and_then(serde_json::Value::as_array_mut)
        {
            for value in values {
                make_strict(value);
            }
        }
    }
}

pub(crate) fn convert_messages_to_responses_format(
    messages: Vec<ConversationMessage>,
) -> Result<Vec<InputItem>, LlmError> {
    messages
        .into_iter()
        .map(|message| match message {
            ConversationMessage::Chat(message) => Ok(InputItem::Message(InputMessage {
                role: match message.role {
                    ChatRole::System => InputMessageRole::System,
                    ChatRole::User => InputMessageRole::User,
                    ChatRole::Assistant => InputMessageRole::Assistant,
                },
                content: message.content,
            })),
            ConversationMessage::ToolCall(tool_call) => {
                let arguments = serde_json::to_string(&tool_call.arguments).map_err(|error| {
                    LlmError::Parse {
                        message: "Failed to serialize tool call arguments".to_string(),
                        source: Box::new(error),
                    }
                })?;

                Ok(InputItem::FunctionCall(FunctionToolCall {
                    r#type: "function_call".to_string(),
                    id: tool_call.id,
                    call_id: tool_call.call_id,
                    name: tool_call.name,
                    arguments: serde_json::Value::String(arguments),
                }))
            }
            ConversationMessage::ToolCallResult(result) => {
                Ok(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                    call_id: result.tool_call_id,
                    output: result.content,
                    r#type: "function_call_output".to_string(),
                }))
            }
        })
        .collect()
}

pub(crate) fn create_format_for_type<T>() -> Result<Format, LlmError>
where
    T: schemars::JsonSchema,
{
    let schema = schema_for!(T);
    let schema_value = serde_json::to_value(&schema).map_err(|error| LlmError::Parse {
        message: "Failed to build JSON Schema".to_string(),
        source: Box::new(error),
    })?;
    create_format_from_value(schema_value)
}

pub(crate) fn create_format_from_value(
    mut schema_value: serde_json::Value,
) -> Result<Format, LlmError> {
    let schema = schema_value.as_object().ok_or_else(|| LlmError::Provider {
        message: "Failed to build JSON Schema: root is not an object".to_string(),
        source: None,
    })?;

    let schema_name = schema
        .get("title")
        .ok_or_else(|| LlmError::Provider {
            message: "Failed to build JSON Schema: Missing schema name".to_string(),
            source: None,
        })?
        .as_str()
        .ok_or_else(|| LlmError::Provider {
            message: "Failed to build JSON Schema: title is not a string".to_string(),
            source: None,
        })?
        .to_owned();

    let needs_wrapping = schema_value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|schema_type| schema_type != "object");

    if needs_wrapping {
        schema_value = serde_json::json!({
            "type": "object",
            "properties": {
                "value": schema_value
            },
            "required": ["value"],
            "additionalProperties": false
        });
    }

    Ok(Format {
        format: FormatType::JsonSchema(JsonSchema {
            name: schema_name,
            schema: schema_value,
            r#type: JsonSchemaType::JsonSchema,
        }),
    })
}

pub(crate) fn create_text_format() -> Format {
    Format {
        format: FormatType::Text {
            r#type: TextType::Text,
        },
    }
}
