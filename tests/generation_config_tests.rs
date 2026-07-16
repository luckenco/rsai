use rsai::{
    CompletionTarget, GeminiClient, GenerationConfig, LlmError, LlmProvider, OpenAiClient,
    OpenRouterClient, StructuredRequest, TextResponse,
};

fn invalid_request() -> StructuredRequest {
    StructuredRequest {
        model: "test-model".to_string(),
        messages: Vec::new(),
        tool_config: None,
        generation_config: Some(GenerationConfig {
            max_tokens: Some(0),
            temperature: None,
            top_p: None,
        }),
    }
}

async fn assert_rejected_before_dispatch(provider: &impl LlmProvider) {
    let result = provider
        .generate_completion::<TextResponse, ()>(
            invalid_request(),
            TextResponse::format().expect("text format"),
            None,
        )
        .await;

    assert!(matches!(result, Err(LlmError::Builder(message)) if message.contains("max_tokens")));
}

#[tokio::test]
async fn providers_validate_generation_config_before_dispatch() {
    assert_rejected_before_dispatch(&OpenAiClient::new("test-key".to_string()).expect("client"))
        .await;
    assert_rejected_before_dispatch(
        &OpenRouterClient::new("test-key".to_string()).expect("client"),
    )
    .await;
    assert_rejected_before_dispatch(&GeminiClient::new("test-key".to_string()).expect("client"))
        .await;
}
