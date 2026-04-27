//! Generic completion client for providers that don't use OpenAI's responses API.
//!
//! This module provides reusable infrastructure for completion-style APIs.

use futures::{StreamExt, TryStreamExt, stream};
use serde::{Serialize, de::DeserializeOwned};

pub use crate::core::BaseUrlSecurity;
use crate::{
    core::{
        FunctionCallData, HttpClient, HttpClientConfig, InspectorConfig, LlmError,
        ProviderResponse, StructuredRequest, ToolCall, ToolCallingGuard, ToolRegistry,
        http::validate_provider_base_url,
    },
    responses::Format,
};

/// Trait for building provider-specific requests and parsing responses.
///
/// Each provider (e.g., Gemini) implements this trait to handle the conversion
/// between core types and their native API format.
pub trait CompletionRequestBuilder: Send + Sync {
    /// The provider-specific request type
    type Request: Serialize + Send;
    /// The provider-specific response type
    type Response: DeserializeOwned + Send;

    /// Build a provider-specific request from a core structured request.
    fn build_request(
        &self,
        request: &StructuredRequest,
        format: &Format,
        conversation: &[ConversationItem],
    ) -> Result<Self::Request, LlmError>;

    /// Parse a provider-specific response into a unified ProviderResponse.
    fn parse_response(&self, response: Self::Response) -> Result<ProviderResponse, LlmError>;

    /// Get the API endpoint for a given model.
    fn endpoint(&self, model: &str) -> String;

    /// Extract function calls from the response for tool calling loop.
    /// Returns None if no function calls are present.
    fn extract_function_calls(&self, response: &Self::Response) -> Option<Vec<FunctionCallData>>;
}

/// An item in the conversation history for the tool calling loop.
#[derive(Debug, Clone)]
pub enum ConversationItem {
    /// A regular message (system, user, or assistant)
    Message { role: String, content: String },
    /// A function call made by the model
    FunctionCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// The result of a function call
    FunctionResult {
        call_id: String,
        result: serde_json::Value,
    },
}

/// Configuration trait for completion-style providers.
pub trait CompletionProviderConfig {
    /// Get the base URL for the API
    fn base_url(&self) -> &str;

    /// Security policy for custom provider base URLs.
    fn base_url_security(&self) -> BaseUrlSecurity {
        BaseUrlSecurity::HttpsOnly
    }

    /// Get the authentication header as (name, value) tuple
    fn auth_header(&self) -> (String, String);

    /// Get additional headers to include with each request
    fn extra_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Get the HTTP client configuration
    fn http_config(&self) -> HttpClientConfig {
        HttpClientConfig::default()
    }

    /// Get the user agent string
    fn user_agent(&self) -> String {
        format!("rsai/{}", env!("CARGO_PKG_VERSION"))
    }

    /// Get the inspector configuration for request/response logging.
    fn inspector_config(&self) -> Option<&InspectorConfig> {
        None
    }
}

/// Generic client for completion-style providers.
pub struct CompletionClient<P: CompletionProviderConfig> {
    pub config: P,
    http: HttpClient,
}

impl<P: CompletionProviderConfig> CompletionClient<P> {
    /// Create a new completion client with the given configuration.
    pub fn new(config: P) -> Result<Self, LlmError> {
        validate_provider_base_url(config.base_url(), config.base_url_security())?;

        let http_config = config.http_config();
        let user_agent = config.user_agent();
        let inspector_config = config.inspector_config().cloned();

        let http = HttpClient::new(http_config, Some(&user_agent), inspector_config)?;

        Ok(Self { config, http })
    }

    /// Make an API request using the given request builder.
    pub async fn make_api_request<B: CompletionRequestBuilder>(
        &self,
        builder: &B,
        request: B::Request,
        model: &str,
    ) -> Result<B::Response, LlmError> {
        let url = format!("{}{}", self.config.base_url(), builder.endpoint(model));

        let mut headers = vec![self.config.auth_header()];
        headers.extend(self.config.extra_headers());

        self.http.post_json(&url, &headers, &request).await
    }

    /// Handle the complete tool calling loop until a final response is received.
    pub async fn handle_tool_calling_loop<B: CompletionRequestBuilder, Ctx>(
        &self,
        builder: &B,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<ProviderResponse, LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        let timeout_duration = guard.timeout;

        match tokio::time::timeout(
            timeout_duration,
            self.handle_tool_calling_loop_internal::<B, Ctx>(
                builder,
                request,
                tool_registry,
                guard,
                format,
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(LlmError::ToolCallTimeout {
                timeout: timeout_duration,
            }),
        }
    }

    /// Internal implementation of the tool calling loop.
    async fn handle_tool_calling_loop_internal<B: CompletionRequestBuilder, Ctx>(
        &self,
        builder: &B,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<ProviderResponse, LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        // Convert initial messages to conversation items
        let mut conversation = convert_messages_to_conversation(&request.messages)?;
        let is_parallel = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.parallel_tool_calls)
            .unwrap_or(true);

        loop {
            guard.increment_iteration()?;

            let api_request = builder.build_request(&request, &format, &conversation)?;
            let api_response = self
                .make_api_request(builder, api_request, &request.model)
                .await?;

            // Check for function calls
            let function_calls = builder.extract_function_calls(&api_response);

            if let Some(calls) = function_calls.filter(|c| !c.is_empty()) {
                tracing::info!(count = calls.len(), "Model requested tool execution");
                guard.check_tool_calls_for_turn(calls.len())?;

                let mut tool_calls = Vec::with_capacity(calls.len());
                for call in &calls {
                    conversation.push(ConversationItem::FunctionCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    });

                    tool_calls.push(ToolCall {
                        id: call.id.clone(),
                        call_id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    });
                }

                let max_concurrent_tool_calls = if is_parallel {
                    guard.max_concurrent_tool_calls()
                } else {
                    1
                };
                let tool_timeout = guard.tool_timeout;
                let results: Vec<serde_json::Value> = stream::iter(tool_calls.iter().cloned())
                    .map(|tool_call| async move {
                        self.execute_tool_with_timeout(tool_registry, &tool_call, tool_timeout)
                            .await
                    })
                    .buffered(max_concurrent_tool_calls)
                    .try_collect()
                    .await?;

                for (tool_call, result) in tool_calls.iter().zip(results) {
                    conversation.push(ConversationItem::FunctionResult {
                        call_id: tool_call.call_id.clone(),
                        result,
                    });
                }
            } else {
                tracing::debug!("No more tool calls, returning final response");
                return builder.parse_response(api_response);
            }
        }
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
}

/// Convert core messages to conversation items.
pub(crate) fn convert_messages_to_conversation(
    messages: &[crate::core::ConversationMessage],
) -> Result<Vec<ConversationItem>, LlmError> {
    messages
        .iter()
        .map(|msg| match msg {
            crate::core::ConversationMessage::Chat(m) => {
                let role = match m.role {
                    crate::core::ChatRole::System => "system",
                    crate::core::ChatRole::User => "user",
                    crate::core::ChatRole::Assistant => "assistant",
                };
                Ok(ConversationItem::Message {
                    role: role.to_string(),
                    content: m.content.clone(),
                })
            }
            crate::core::ConversationMessage::ToolCall(tc) => Ok(ConversationItem::FunctionCall {
                id: tc.call_id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            }),
            crate::core::ConversationMessage::ToolCallResult(tr) => {
                Ok(ConversationItem::FunctionResult {
                    call_id: tr.tool_call_id.clone(),
                    result: tr.content.clone(),
                })
            }
        })
        .collect()
}
