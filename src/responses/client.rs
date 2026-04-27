//! Shared client logic for providers that use the OpenAI-style responses API.
//!
//! This module contains reusable functionality for:
//! - Building requests from core types
//! - Handling tool calling loops (parallel and sequential)
//! - Converting between core and API message formats
//! - Serializing tools and generating JSON schemas
//! - Parsing API responses back to core types

use crate::{
    CompletionTarget, Provider,
    core::{
        ChatRole, ConversationMessage, HttpClient, InspectorConfig, LlmError, StructuredRequest,
        Tool, ToolCall, ToolCallingGuard, ToolRegistry, http::validate_provider_base_url,
    },
    responses::{
        Format, FormatType, FunctionToolCall, FunctionToolCallOutput, JsonSchema, JsonSchemaType,
        TextType,
        request::{InputItem, InputMessage, InputMessageRole, Request},
        response::{MessageContent, OutputContent, Response},
    },
};
use futures::{StreamExt, TryStreamExt, stream};
use schemars::schema_for;
use tracing;

// Re-export HttpClientConfig from core for backwards compatibility
pub use crate::core::{BaseUrlSecurity, HttpClientConfig};

/// Configuration trait for providers that use the OpenAI-style responses API
pub trait ResponsesProviderConfig {
    /// Model Provider
    fn provider(&self) -> Provider;

    /// Base URL for the API (e.g., `https://api.openai.com`)
    fn base_url(&self) -> &str;

    /// Security policy for custom provider base URLs.
    fn base_url_security(&self) -> BaseUrlSecurity {
        BaseUrlSecurity::HttpsOnly
    }

    /// API endpoint for responses (e.g., `/v1/responses`)
    fn endpoint(&self) -> &str;

    /// Authentication header as (header_name, header_value) tuple
    fn auth_header(&self) -> (String, String);

    /// Additional headers to include with each request
    fn extra_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Configuration for HTTP client resilience
    fn http_config(&self) -> HttpClientConfig {
        HttpClientConfig::default()
    }

    fn user_agent(&self) -> String {
        format!("rsai/{}", env!("CARGO_PKG_VERSION"))
    }

    /// Get the inspector configuration for request/response logging.
    fn inspector_config(&self) -> Option<&InspectorConfig> {
        None
    }
}

/// Shared client for providers using the OpenAI-style responses API
pub struct ResponsesClient<P: ResponsesProviderConfig> {
    pub config: P,
    http: HttpClient,
}

impl<P: ResponsesProviderConfig> ResponsesClient<P> {
    /// Create a new responses client with the given configuration
    pub fn new(config: P) -> Result<Self, LlmError> {
        validate_provider_base_url(config.base_url(), config.base_url_security())?;

        let http_config = config.http_config();
        let user_agent = config.user_agent();
        let inspector_config = config.inspector_config().cloned();

        let http = HttpClient::new(http_config, Some(&user_agent), inspector_config)?;

        Ok(Self { config, http })
    }

    /// Make an API request to the responses endpoint
    #[tracing::instrument(
        name = "http_request",
        skip(self, request),
        fields(
            base_url = %self.config.base_url(),
            endpoint = %self.config.endpoint()
        ),
        err
    )]
    pub async fn make_api_request(&self, request: Request) -> Result<Response, LlmError> {
        let url = format!("{}{}", self.config.base_url(), self.config.endpoint());

        // Build headers
        let mut headers = vec![self.config.auth_header()];
        headers.extend(self.config.extra_headers());

        self.http.post_json(&url, &headers, &request).await
    }

    /// Handle the complete tool calling loop until a final response is received
    pub async fn handle_tool_calling_loop<T, Ctx>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget,
        Ctx: Send + Sync + 'static,
    {
        let timeout_duration = guard.timeout;

        match tokio::time::timeout(
            timeout_duration,
            self.handle_tool_calling_loop_internal::<T, Ctx>(request, tool_registry, guard, format),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(LlmError::ToolCallTimeout {
                timeout: timeout_duration,
            }),
        }
    }

    /// Internal implementation of the tool calling loop without timeout wrapper
    #[tracing::instrument(
        name = "tool_calling_loop",
        level="debug",
        skip(self, request, tool_registry, guard),
        fields(
            model = %request.model,
            max_iterations = %guard.max_iterations
        ),
        err
    )]
    async fn handle_tool_calling_loop_internal<T, Ctx>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget,
        Ctx: Send + Sync + 'static,
    {
        let mut responses_input = convert_messages_to_responses_format(request.messages.clone())?;
        let is_parallel = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.parallel_tool_calls)
            .unwrap_or(true);

        loop {
            guard.increment_iteration()?;

            let iteration_span =
                tracing::debug_span!("tool_loop_iteration", iteration = guard.current_iteration());
            let _enter = iteration_span.enter();

            let responses_request =
                self.build_request_with_format(&request, &responses_input, format.clone())?;
            let api_response = self.make_api_request(responses_request).await?;

            let function_calls = self.extract_function_calls(&api_response);

            if function_calls.is_empty() {
                tracing::debug!("No more tool calls, returning final response");
                let provider_response =
                    convert_to_provider_response(api_response, self.config.provider())?;
                return T::parse_response(provider_response);
            }

            tracing::info!(
                count = function_calls.len(),
                "Model requested tool execution"
            );

            guard.check_tool_calls_for_turn(function_calls.len())?;

            self.process_function_calls(
                &function_calls,
                &mut responses_input,
                tool_registry,
                is_parallel,
                guard,
            )
            .await?;
        }
    }

    /// Build a responses API request from core request and input
    pub fn build_request_with_format(
        &self,
        request: &StructuredRequest,
        responses_input: &[InputItem],
        format: Format,
    ) -> Result<Request, LlmError> {
        build_request_payload_with_format(request, responses_input, format)
    }

    /// Extract function calls from API response
    pub fn extract_function_calls<'a>(
        &self,
        api_response: &'a Response,
    ) -> Vec<&'a super::types::FunctionToolCall> {
        api_response
            .output
            .iter()
            .filter_map(|output| match output {
                OutputContent::FunctionCall(fc) => Some(fc),
                OutputContent::OutputMessage(_) => None,
            })
            .collect()
    }

    /// Process function calls either in parallel or sequentially
    pub async fn process_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        is_parallel: bool,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        if is_parallel && function_calls.len() > 1 {
            self.process_parallel_function_calls(
                function_calls,
                responses_input,
                tool_registry,
                guard,
            )
            .await
        } else {
            self.process_sequential_function_calls(
                function_calls,
                responses_input,
                tool_registry,
                guard,
            )
            .await
        }
    }

    /// Process function calls in parallel (all calls added first, tools executed concurrently, then all results added).
    ///
    /// On failure, completed tools may have produced side-effects but no results are added to `responses_input`.
    pub async fn process_parallel_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        // Add all function calls to input first
        let mut tool_calls = Vec::with_capacity(function_calls.len());
        for function_call in function_calls {
            responses_input.push(InputItem::FunctionCall((*function_call).clone()));

            let arguments = self.parse_function_arguments(&function_call.arguments)?;
            tool_calls.push(ToolCall {
                id: function_call.id.clone(),
                call_id: function_call.call_id.clone(),
                name: function_call.name.clone(),
                arguments,
            });
        }

        // Execute tools with bounded concurrency, preserving result order.
        let tool_timeout = guard.tool_timeout;
        let max_concurrent_tool_calls = guard.max_concurrent_tool_calls();
        let results: Vec<serde_json::Value> = stream::iter(tool_calls.iter().cloned())
            .map(|tool_call| async move {
                self.execute_tool_with_timeout(tool_registry, &tool_call, tool_timeout)
                    .await
            })
            .buffered(max_concurrent_tool_calls)
            .try_collect()
            .await?;

        // Add all results in order
        for (tool_call, result) in tool_calls.iter().zip(results) {
            responses_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                call_id: tool_call.call_id.clone(),
                output: result,
                r#type: "function_call_output".to_string(),
            }));
        }

        Ok(())
    }

    /// Process function calls sequentially (call and result pairs)
    pub async fn process_sequential_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        for function_call in function_calls {
            responses_input.push(InputItem::FunctionCall((*function_call).clone()));

            let arguments = self.parse_function_arguments(&function_call.arguments)?;
            let tool_call = ToolCall {
                id: function_call.id.clone(),
                call_id: function_call.call_id.clone(),
                name: function_call.name.clone(),
                arguments,
            };

            let result = self
                .execute_tool_with_timeout(tool_registry, &tool_call, guard.tool_timeout)
                .await?;

            responses_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                call_id: function_call.call_id.clone(),
                output: result,
                r#type: "function_call_output".to_string(),
            }));
        }

        Ok(())
    }

    async fn execute_tool_with_timeout<Ctx>(
        &self,
        tool_registry: &ToolRegistry<Ctx>,
        tool_call: &ToolCall,
        timeout: std::time::Duration,
    ) -> Result<serde_json::Value, LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        match tokio::time::timeout(timeout, tool_registry.execute(tool_call)).await {
            Ok(result) => result,
            Err(_) => Err(LlmError::ToolExecutionTimeout {
                tool_name: tool_call.name.clone(),
                timeout,
            }),
        }
    }

    /// Shared generate_completion for responses-API providers (OpenAI, OpenRouter).
    ///
    /// Handles the tool calling loop when tools are present, or makes a single
    /// request otherwise.
    pub async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: crate::responses::request::Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
        mut guard: ToolCallingGuard,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget + Send,
        Ctx: Send + Sync + 'static,
    {
        let has_tools = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.tools.as_ref())
            .is_some();

        if has_tools && let Some(tool_registry) = tool_registry {
            return self
                .handle_tool_calling_loop::<T, Ctx>(request, tool_registry, &mut guard, format)
                .await;
        }

        let responses_input = convert_messages_to_responses_format(request.messages.clone())?;
        let responses_request =
            self.build_request_with_format(&request, &responses_input, format)?;
        let api_response = self.make_api_request(responses_request).await?;
        let provider_response = convert_to_provider_response(api_response, self.config.provider())?;
        T::parse_response(provider_response)
    }

    /// Parse function arguments from JSON value
    pub fn parse_function_arguments(
        &self,
        arguments: &serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        match arguments {
            serde_json::Value::String(s) => serde_json::from_str(s).map_err(|e| LlmError::Parse {
                message: "Failed to parse tool arguments".to_string(),
                source: Box::new(e),
            }),
            other => Ok(other.clone()),
        }
    }
}

// This is a separate method to `build_request_with_format` for testing.
pub(crate) fn build_request_payload_with_format(
    request: &StructuredRequest,
    responses_input: &[InputItem],
    format: Format,
) -> Result<Request, LlmError> {
    let mut req = Request {
        model: request.model.clone(),
        input: responses_input.to_vec(),
        text: format,
        // Default fields
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

    // Apply tool configuration if present
    if let Some(tool_config) = &request.tool_config {
        req.tools = tool_config.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(crate::responses::create_function_tool)
                .collect::<Box<[crate::responses::Tool]>>()
        });
        req.tool_choice = tool_config.tool_choice.as_ref().map(|tc| tc.clone().into());
        req.parallel_tool_calls = tool_config.parallel_tool_calls;
    }

    // Apply generation configuration if present
    if let Some(gen_config) = &request.generation_config {
        req.temperature = gen_config.temperature;
        req.max_output_tokens = gen_config.max_tokens;
        req.top_p = gen_config.top_p;
    }

    Ok(req)
}

/// Convert core Tool to responses API Tool
pub(crate) fn create_function_tool(tool: &Tool) -> crate::responses::Tool {
    let strict = tool.strict.unwrap_or(true);
    let mut parameters = tool.parameters.clone();

    // OpenAI strict mode requires ALL properties to be in the required array
    if strict && let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) {
        let all_property_names: Vec<serde_json::Value> = properties
            .keys()
            .map(|k| serde_json::Value::String(k.clone()))
            .collect();

        if let Some(params_obj) = parameters.as_object_mut() {
            params_obj.insert(
                "required".to_string(),
                serde_json::Value::Array(all_property_names),
            );
        }
    }

    crate::responses::Tool {
        name: tool.name.clone(),
        parameters,
        strict: Some(strict),
        description: tool.description.clone(),
    }
}

/// Convert core messages to responses API format
pub(crate) fn convert_messages_to_responses_format(
    messages: Vec<ConversationMessage>,
) -> Result<Vec<InputItem>, LlmError> {
    messages
        .into_iter()
        .map(|msg| match msg {
            ConversationMessage::Chat(m) => Ok(InputItem::Message(InputMessage {
                role: match m.role {
                    ChatRole::System => InputMessageRole::System,
                    ChatRole::User => InputMessageRole::User,
                    ChatRole::Assistant => InputMessageRole::Assistant,
                },
                content: m.content,
            })),
            ConversationMessage::ToolCall(tc) => Ok(InputItem::FunctionCall(FunctionToolCall {
                r#type: "function_call".to_string(),
                id: tc.id,
                call_id: tc.call_id,
                name: tc.name,
                arguments: serde_json::Value::String(
                    serde_json::to_string(&tc.arguments).map_err(|e| LlmError::Parse {
                        message: "Failed to serialize tool call arguments".to_string(),
                        source: Box::new(e),
                    })?,
                ),
            })),
            ConversationMessage::ToolCallResult(tr) => {
                Ok(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                    call_id: tr.tool_call_id,
                    output: tr.content,
                    r#type: "function_call_output".to_string(),
                }))
            }
        })
        .collect()
}

/// Create JSON schema format for a given type
pub(crate) fn create_format_for_type<T>() -> Result<Format, LlmError>
where
    T: schemars::JsonSchema,
{
    let schema = schema_for!(T);
    let schema_value = serde_json::to_value(&schema).map_err(|e| LlmError::Parse {
        message: "Failed to build JSON Schema".to_string(),
        source: Box::new(e),
    })?;
    create_format_from_value(schema_value)
}

pub(crate) fn create_format_from_value(
    mut schema_value: serde_json::Value,
) -> Result<Format, LlmError> {
    let schema_obj = schema_value.as_object().ok_or_else(|| LlmError::Provider {
        message: "Failed to build JSON Schema: root is not an object".to_string(),
        source: None,
    })?;

    let schema_name = schema_obj
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
        .and_then(|t| t.as_str())
        .map(|t| t != "object")
        .unwrap_or(false);

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

/// Convert OpenAI API response to provider-agnostic ProviderResponse.
///
/// Aggregates all output items: collects function calls across all items,
/// concatenates text from all messages, and surfaces refusals.
/// Function calls take priority over text if both are present.
pub fn convert_to_provider_response(
    res: Response,
    provider: crate::provider::Provider,
) -> Result<crate::core::ProviderResponse, LlmError> {
    use crate::core::{FunctionCallData, LanguageModelUsage, ProviderResponse, ResponseContent};

    let mut function_calls = Vec::new();
    let mut text_parts = Vec::new();
    let mut refusal = None;

    for output in &res.output {
        match output {
            OutputContent::OutputMessage(message) => {
                for content in &message.content {
                    match content {
                        MessageContent::OutputText(text) => text_parts.push(text.text.clone()),
                        MessageContent::Refusal(r) => refusal = Some(r.refusal.clone()),
                    }
                }
            }
            OutputContent::FunctionCall(fc) => {
                function_calls.push(FunctionCallData {
                    id: fc.call_id.clone(),
                    name: fc.name.clone(),
                    arguments: fc.arguments.clone(),
                });
            }
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
        id: res.id,
        model: res.model,
        provider,
        content,
        usage: LanguageModelUsage {
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.total_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion_schema;
    use crate::core::{
        ChatRole, ConversationMessage, Message, StructuredRequest, StructuredResponse, ToolRegistry,
    };
    use std::time::Duration;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    // --- Mock Provider Configuration ---

    struct TestProviderConfig {
        base_url: String,
        max_retries: u32,
    }

    impl TestProviderConfig {
        fn new(base_url: String) -> Self {
            Self {
                base_url,
                max_retries: 3,
            }
        }
    }

    impl ResponsesProviderConfig for TestProviderConfig {
        fn provider(&self) -> Provider {
            Provider::OpenAI // Just using OpenAI as a placeholder
        }

        fn base_url(&self) -> &str {
            &self.base_url
        }

        fn base_url_security(&self) -> BaseUrlSecurity {
            BaseUrlSecurity::AllowInsecureHttp
        }

        fn endpoint(&self) -> &str {
            "/responses"
        }

        fn auth_header(&self) -> (String, String) {
            ("Authorization".to_string(), "Bearer test-token".to_string())
        }

        fn http_config(&self) -> HttpClientConfig {
            HttpClientConfig {
                timeout: Duration::from_secs(5),
                max_retries: self.max_retries,
                initial_retry_delay: Duration::from_millis(10), // Fast retries for tests
                max_retry_delay: Duration::from_millis(100),
            }
        }
    }

    // --- Helpers ---

    async fn create_client(server: &MockServer) -> ResponsesClient<TestProviderConfig> {
        let config = TestProviderConfig::new(server.uri());
        ResponsesClient::new(config).expect("Failed to create client")
    }

    fn create_basic_request() -> Request {
        Request {
            model: "test-model".to_string(),
            input: vec![],
            text: create_format_for_type::<serde_json::Value>().unwrap(),
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
        }
    }

    #[test]
    fn test_parse_function_arguments_error_does_not_include_raw_arguments() {
        let client = ResponsesClient::new(TestProviderConfig::new("http://localhost".to_string()))
            .expect("client");
        let raw_arguments = "{\"secret\":\"do-not-log\"";
        let result =
            client.parse_function_arguments(&serde_json::Value::String(raw_arguments.to_string()));

        match result {
            Err(LlmError::Parse { message, source }) => {
                assert_eq!(message, "Failed to parse tool arguments");
                assert!(!message.contains(raw_arguments));
                assert!(source.is::<serde_json::Error>());
                assert!(!source.to_string().contains(raw_arguments));

                let display = LlmError::Parse { message, source }.to_string();
                assert!(!display.contains(raw_arguments));
            }
            other => panic!("Expected Parse Error, got {other:?}"),
        }
    }

    // --- Tests: HTTP Resilience ---

    #[tokio::test]
    async fn test_retry_logic() {
        let server = MockServer::start().await;
        let client = create_client(&server).await;

        // Fail twice, then succeed
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let valid_response = serde_json::json!({
            "id": "resp_123",
            "model": "test-model",
            "output": [{
                "id": "msg_123",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "{\"value\": \"success\"}"
                }]
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        });

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(valid_response))
            .mount(&server)
            .await;

        let request = create_basic_request();
        let result = client.make_api_request(request).await;

        assert!(result.is_ok(), "Client should succeed after retries");
    }

    #[tokio::test]
    async fn test_fatal_errors_401() {
        let server = MockServer::start().await;
        let client = create_client(&server).await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let request = create_basic_request();
        let result = client.make_api_request(request).await;

        match result {
            Err(LlmError::Api {
                status_code: Some(401),
                ..
            }) => (),
            _ => panic!("Expected 401 Api Error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_fatal_errors_400() {
        let server = MockServer::start().await;
        let client = create_client(&server).await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(400).set_body_string("Bad Request"))
            .mount(&server)
            .await;

        let request = create_basic_request();
        let result = client.make_api_request(request).await;

        match result {
            Err(LlmError::Api {
                status_code: Some(400),
                ..
            }) => (),
            _ => panic!("Expected 400 Api Error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_malformed_body() {
        let server = MockServer::start().await;
        let client = create_client(&server).await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{ invalid json"))
            .mount(&server)
            .await;

        let request = create_basic_request();
        let result = client.make_api_request(request).await;

        match result {
            Err(LlmError::Parse { .. }) => (),
            _ => panic!("Expected Parse Error, got {:?}", result),
        }
    }

    // --- Tests: Response Parsing ---

    #[completion_schema]
    #[derive(Debug, Clone, PartialEq)]
    struct TestResponse {
        value: String,
    }

    async fn run_parsing_test<T>(
        server: &MockServer,
        response_body: serde_json::Value,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        let client = create_client(server).await;

        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
            .mount(server)
            .await;

        let request = StructuredRequest {
            model: "test-model".to_string(),
            messages: vec![ConversationMessage::Chat(Message {
                role: ChatRole::User,
                content: "test".to_string(),
            })],
            tool_config: None,
            generation_config: None,
        };

        let tool_registry = ToolRegistry::new();
        let mut guard = ToolCallingGuard::new();

        let format = create_format_for_type::<T>()?;

        client
            .handle_tool_calling_loop::<T, ()>(request, &tool_registry, &mut guard, format)
            .await
    }

    #[tokio::test]
    async fn test_response_parsing_refusal() {
        let server = MockServer::start().await;

        let refusal_response = serde_json::json!({
            "id": "resp_refusal",
            "model": "test-model",
            "output": [{
                "id": "msg_refusal",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "refusal",
                    "refusal": "I cannot do that."
                }]
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
        });

        let result = run_parsing_test::<TestResponse>(&server, refusal_response).await;

        match result {
            Err(LlmError::Api { message, .. }) if message.contains("Model refused") => (),
            _ => panic!("Expected Refusal Error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_response_parsing_wrapped() {
        let server = MockServer::start().await;

        let wrapped_json = serde_json::json!({
            "value": "wrapped_success"
        });

        let wrapped_response = serde_json::json!({
            "id": "resp_wrapped",
            "model": "test-model",
            "output": [{
                "id": "msg_wrapped",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": wrapped_json.to_string()
                }]
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
        });

        // Use String as the target type, which is a "wrapped" schema type (non-object)
        let result = run_parsing_test::<String>(&server, wrapped_response).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().content, "wrapped_success");
    }

    #[tokio::test]
    async fn test_response_parsing_rejects_unknown_fields() {
        let server = MockServer::start().await;

        let output_json = serde_json::json!({
            "value": "ok",
            "extra": true
        });

        let response = serde_json::json!({
            "id": "resp_extra_field",
            "model": "test-model",
            "output": [{
                "id": "msg_extra_field",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": output_json.to_string()
                }]
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
        });

        let result = run_parsing_test::<TestResponse>(&server, response).await;

        match result {
            Err(LlmError::Parse { .. }) => (),
            _ => panic!("Expected Parse Error, got {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_response_parsing_empty() {
        let server = MockServer::start().await;

        let empty_response = serde_json::json!({
            "id": "resp_empty",
            "model": "test-model",
            "output": [], // Empty output
            "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
        });

        let result = run_parsing_test::<TestResponse>(&server, empty_response).await;

        match result {
            Err(LlmError::Provider { message, .. }) if message.contains("No output") => (),
            _ => panic!("Expected No Output Error, got {:?}", result),
        }
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    use crate::core::{
        ChatRole, ConversationMessage, GenerationConfig, Message, StructuredRequest, Tool,
        ToolCall, ToolCallResult, ToolChoice, ToolConfig,
    };
    use crate::responses::request::InputItem;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    #[derive(JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    struct StandardObject {
        city: String,
        temperature: f32,
        active: bool,
    }

    #[derive(JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    struct StringWrapper(String);

    #[derive(JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    enum SimpleEnum {
        VariantA,
        VariantB,
        VariantC,
    }

    fn sample_tool(strict: Option<bool>) -> Tool {
        Tool {
            name: "weather_lookup".to_string(),
            description: Some("Look up weather details for a city".to_string()),
            strict,
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "units": { "type": "string" }
                },
                "required": ["city"]
            }),
        }
    }

    fn sample_request(
        tool_config: Option<ToolConfig>,
        generation_config: Option<GenerationConfig>,
    ) -> StructuredRequest {
        StructuredRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![ConversationMessage::Chat(Message {
                role: ChatRole::User,
                content: "Weather for Lisbon".to_string(),
            })],
            tool_config,
            generation_config,
        }
    }

    #[test]
    fn test_standard_object_schema_is_passthrough_object() {
        let format = create_format_for_type::<StandardObject>().expect("schema");

        let schema = match format.format {
            crate::responses::FormatType::JsonSchema(schema) => schema,
            _ => panic!("expected JSON schema format"),
        };

        assert_eq!(schema.name, "StandardObject");
        let json = schema.schema.as_object().unwrap();
        assert_eq!(json.get("type").unwrap(), "object");
        assert!(json.get("properties").is_some());
    }

    #[test]
    fn test_string_wrapper_schema_is_wrapped() {
        let format = create_format_for_type::<StringWrapper>().expect("schema");

        let schema = match format.format {
            crate::responses::FormatType::JsonSchema(schema) => schema,
            _ => panic!("expected JSON schema format"),
        };

        let json = schema.schema.as_object().unwrap();
        assert_eq!(json.get("type").unwrap(), "object");
        assert_eq!(json.get("required").unwrap(), &json!(["value"]));

        let properties = json.get("properties").unwrap().as_object().unwrap();
        let value_schema = properties.get("value").unwrap().as_object().unwrap();
        assert_eq!(value_schema.get("type").unwrap(), "string");
    }

    #[test]
    fn test_enum_schema_is_wrapped_preserving_variants() {
        let format = create_format_for_type::<SimpleEnum>().expect("schema");

        let schema = match format.format {
            crate::responses::FormatType::JsonSchema(schema) => schema,
            _ => panic!("expected JSON schema format"),
        };

        let json = schema.schema.as_object().unwrap();
        assert_eq!(json.get("type").unwrap(), "object");
        let value_schema = json["properties"]["value"].as_object().unwrap();
        assert_eq!(value_schema["type"], "string");
        let enum_values = value_schema["enum"].as_array().unwrap();
        let expected = vec![json!("VariantA"), json!("VariantB"), json!("VariantC")];
        assert_eq!(enum_values, &expected);
    }

    #[test]
    fn test_schema_missing_title_errors() {
        let err = create_format_from_value(json!({ "type": "object" }))
            .expect_err("schema without title should error");
        matches_provider_error(err, "Missing schema name");
    }

    #[test]
    fn test_schema_non_object_root_errors() {
        let err =
            create_format_from_value(json!(true)).expect_err("non-object schema should error");
        matches_provider_error(err, "root is not an object");
    }

    #[test]
    fn test_convert_messages_to_responses_format_handles_tool_calls() {
        let tool_call = ToolCall {
            id: "tool_1".into(),
            call_id: "tool_1".into(),
            name: "weather_lookup".into(),
            arguments: json!({ "city": "Lisbon" }),
        };

        let tool_result = ToolCallResult {
            id: "result".into(),
            tool_call_id: "tool_1".into(),
            content: json!({ "temperature": 21 }),
        };

        let messages = vec![
            ConversationMessage::ToolCall(tool_call.clone()),
            ConversationMessage::ToolCallResult(tool_result.clone()),
        ];

        let converted = convert_messages_to_responses_format(messages).expect("conversion");
        assert_eq!(converted.len(), 2);

        match &converted[0] {
            InputItem::FunctionCall(call) => {
                assert_eq!(call.id, "tool_1");
                assert_eq!(call.name, "weather_lookup");
                assert_eq!(call.r#type, "function_call");
                assert_eq!(
                    call.arguments,
                    Value::String(json!({ "city": "Lisbon" }).to_string())
                );
            }
            other => panic!("unexpected first input item: {other:?}"),
        }

        match &converted[1] {
            InputItem::FunctionCallOutput(output) => {
                assert_eq!(output.call_id, "tool_1");
                assert_eq!(output.output, json!({ "temperature": 21 }));
                assert_eq!(output.r#type, "function_call_output");
            }
            other => panic!("unexpected second input item: {other:?}"),
        }
    }

    #[test]
    fn test_build_request_includes_generation_and_tool_config() {
        let tool_config = ToolConfig {
            tools: Some(vec![sample_tool(Some(true))].into_boxed_slice()),
            tool_choice: Some(ToolChoice::Function {
                name: "weather_lookup".into(),
            }),
            parallel_tool_calls: Some(false),
        };
        let generation_config = GenerationConfig {
            max_tokens: Some(256),
            temperature: Some(0.2),
            top_p: Some(0.9),
        };

        let request = sample_request(Some(tool_config), Some(generation_config));
        let responses_input =
            convert_messages_to_responses_format(request.messages.clone()).expect("inputs");
        let format = create_format_for_type::<StandardObject>().expect("schema");
        let api_request =
            build_request_payload_with_format(&request, &responses_input, format).expect("request");

        assert_eq!(api_request.model, "gpt-4o-mini");
        assert_eq!(api_request.parallel_tool_calls, Some(false));
        assert_eq!(api_request.temperature, Some(0.2));
        assert_eq!(api_request.max_output_tokens, Some(256));
        assert_eq!(api_request.top_p, Some(0.9));
        assert!(api_request.tools.is_some());
        assert!(api_request.tool_choice.is_some());

        let serialized = serde_json::to_value(&api_request).expect("serialized request");
        assert_eq!(serialized["tool_choice"]["name"], "weather_lookup");
        assert_eq!(serialized["tool_choice"]["type"], "function");

        let tool_entry = serialized["tools"]
            .as_array()
            .expect("tools array")
            .first()
            .expect("tool");
        assert_eq!(tool_entry["name"], "weather_lookup");
        assert_eq!(tool_entry["type"], "function");
        assert_eq!(tool_entry["strict"], true);
        assert_eq!(
            tool_entry["parameters"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(
            tool_entry["parameters"]["required"],
            json!(["city", "units"])
        );
    }

    #[test]
    fn test_build_request_without_optional_configs_leaves_fields_empty() {
        let request = sample_request(None, None);
        let responses_input =
            convert_messages_to_responses_format(request.messages.clone()).expect("inputs");

        let format = create_format_for_type::<StandardObject>().expect("schema");
        let api_request =
            build_request_payload_with_format(&request, &responses_input, format).expect("request");

        assert!(api_request.parallel_tool_calls.is_none());
        assert!(api_request.tools.is_none());
        assert!(api_request.tool_choice.is_none());
        assert!(api_request.temperature.is_none());
        assert!(api_request.max_output_tokens.is_none());
        assert!(api_request.top_p.is_none());
    }

    #[test]
    fn test_create_function_tool_enforces_required_when_strict() {
        let tool = sample_tool(Some(true));
        let responses_tool = create_function_tool(&tool);
        assert_eq!(responses_tool.name, "weather_lookup");
        assert_eq!(
            responses_tool.parameters["required"],
            json!(["city", "units"])
        );

        let non_strict_tool = sample_tool(Some(false));
        let responses_tool = create_function_tool(&non_strict_tool);
        assert_eq!(responses_tool.parameters["required"], json!(["city"]));
    }

    fn matches_provider_error(err: LlmError, expected: &str) {
        match err {
            LlmError::Provider { message, .. } => {
                assert!(
                    message.contains(expected),
                    "expected message to contain '{expected}', got '{message}'"
                );
            }
            other => panic!("expected provider error, got {other:?}"),
        }
    }
}
