use crate::{
    core::{FunctionCallData, LanguageModelUsage, LlmError, ProviderResponse, ResponseContent},
    provider::Provider,
};

use super::response::{MessageContent, OutputContent, Response};

/// Convert an OpenAI API response to the provider-agnostic response type.
///
/// Function calls take priority over refusals and text when a response contains
/// more than one kind of output.
pub fn convert_to_provider_response(
    response: Response,
    provider: Provider,
) -> Result<ProviderResponse, LlmError> {
    let mut function_calls = Vec::new();
    let mut text_parts = Vec::new();
    let mut refusal = None;

    for output in &response.output {
        match output {
            OutputContent::OutputMessage(message) => {
                for content in &message.content {
                    match content {
                        MessageContent::OutputText(text) => text_parts.push(text.text.clone()),
                        MessageContent::Refusal(value) => refusal = Some(value.refusal.clone()),
                    }
                }
            }
            OutputContent::FunctionCall(call) => function_calls.push(FunctionCallData {
                id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            }),
        }
    }

    let content = if !function_calls.is_empty() {
        ResponseContent::FunctionCalls(function_calls)
    } else if let Some(refusal) = refusal {
        ResponseContent::Refusal(refusal)
    } else if !text_parts.is_empty() {
        ResponseContent::Text(text_parts.join(""))
    } else {
        return Err(LlmError::Provider {
            message: "No output in response".to_string(),
            source: None,
        });
    };

    Ok(ProviderResponse {
        id: response.id,
        model: response.model,
        provider,
        content,
        usage: LanguageModelUsage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            total_tokens: response.usage.total_tokens,
        },
    })
}
