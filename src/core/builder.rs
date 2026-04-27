use std::{env, marker::PhantomData, sync::Arc};

use tracing::{debug, instrument};

/// Type alias for inspection callbacks that receive raw JSON payloads.
pub type Inspector = Arc<dyn Fn(&serde_json::Value) + Send + Sync>;

/// Configuration for request/response inspection hooks.
#[derive(Clone, Default)]
pub struct InspectorConfig {
    /// Called with the raw JSON request body before each HTTP request.
    pub request_inspector: Option<Inspector>,
    /// Called with the raw JSON response body after each HTTP response.
    pub response_inspector: Option<Inspector>,
}

use crate::{
    provider::{Provider, gemini, openai, openrouter},
    responses::HttpClientConfig,
};

use super::{
    error::LlmError,
    tool_guard::ToolCallingConfig,
    traits::LlmProvider,
    types::{
        ConversationMessage, GenerationConfig, Message, StructuredRequest, ToolChoice, ToolConfig,
        ToolRegistry,
    },
};

mod private {
    pub struct ProviderSet;
    pub struct ApiKeySet;
    pub struct Configuring;
    pub struct MessagesSet;
    pub struct ToolsSet;

    /// Marker trait for states that can call complete()
    pub trait Completable {}
    impl Completable for MessagesSet {}
    impl Completable for ToolsSet {}
}

/// Builder fields that are shared across all states.
/// Generic over context type `Ctx` for tool execution.
struct BuilderFields<Ctx = ()> {
    // Core configuration
    provider: Option<Provider>,
    api_key: Option<String>,
    model: Option<String>,
    http_client_config: Option<HttpClientConfig>,

    // Request content
    messages: Option<Vec<Message>>,

    // Tool configuration
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    tool_calling_config: Option<ToolCallingConfig>,
    tool_registry: Option<ToolRegistry<Ctx>>,

    // Generation parameters
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,

    // Inspection hooks
    inspector_config: Option<InspectorConfig>,
}

impl BuilderFields<()> {
    fn new() -> Self {
        Self {
            provider: None,
            api_key: None,
            model: None,
            messages: None,
            tool_choice: None,
            parallel_tool_calls: None,
            tool_calling_config: None,
            tool_registry: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            http_client_config: None,
            inspector_config: None,
        }
    }
}

impl<Ctx> BuilderFields<Ctx> {
    /// Create a new BuilderFields with a different context type
    fn with_context_type<NewCtx>(
        self,
        tool_registry: Option<ToolRegistry<NewCtx>>,
    ) -> BuilderFields<NewCtx> {
        BuilderFields {
            provider: self.provider,
            api_key: self.api_key,
            model: self.model,
            http_client_config: self.http_client_config,
            messages: self.messages,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            tool_calling_config: self.tool_calling_config,
            tool_registry,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            inspector_config: self.inspector_config,
        }
    }

    /// Validate that all required fields are present
    fn validate(&self) -> Result<(&Vec<Message>, Provider, &str), LlmError> {
        self.api_key.as_ref().ok_or(LlmError::Builder(
            "Missing API key. Make sure to specify an API key.".into(),
        ))?;

        let messages = self
            .messages
            .as_ref()
            .filter(|messages| !messages.is_empty())
            .ok_or(LlmError::Builder(
                "Missing messages. Make sure to add at least one message.".to_string(),
            ))?;

        let provider = self.provider.ok_or(LlmError::Builder(
            "Missing provider. Make sure to specify a provider.".into(),
        ))?;

        let model = self.model.as_ref().ok_or(LlmError::Builder(
            "Missing model. Make sure to specify a model.".into(),
        ))?;

        Ok((messages, provider, model))
    }
}

/// A type-safe builder for constructing LLM requests using the builder pattern.
/// The builder enforces correct construction order through phantom types.
///
/// Type parameters:
/// - `State`: The current builder state (ProviderSet, ApiKeySet, MessagesSet, ToolsSet)
/// - `Ctx`: The context type for tool execution (defaults to `()` for no context)
pub struct LlmBuilder<State, Ctx = ()> {
    fields: BuilderFields<Ctx>,
    _state: PhantomData<State>,
}

impl<State, Ctx> LlmBuilder<State, Ctx> {
    /// Transition to a new builder state while preserving all field values
    fn transition_state<NewState>(self) -> LlmBuilder<NewState, Ctx> {
        LlmBuilder {
            fields: self.fields,
            _state: PhantomData,
        }
    }

    pub(crate) fn get_api_key(&self) -> Option<&str> {
        self.fields.api_key.as_deref()
    }

    pub(crate) fn get_http_config(&self) -> Option<&HttpClientConfig> {
        self.fields.http_client_config.as_ref()
    }

    pub(crate) fn get_inspector_config(&self) -> Option<&InspectorConfig> {
        self.fields.inspector_config.as_ref()
    }

    pub(crate) fn get_tool_calling_config(&self) -> Option<&ToolCallingConfig> {
        self.fields.tool_calling_config.as_ref()
    }
}

/// Configuration for API key source
pub enum ApiKey {
    /// Use the default environment variable for the provider
    Default,
    /// Use a custom API key string
    Custom(String),
}

impl LlmBuilder<private::ProviderSet, ()> {
    /// Set the API key for the provider.
    /// Use `ApiKey::Default` to load from environment variables or `ApiKey::Custom` for a custom key.
    pub fn api_key(
        mut self,
        api_key: ApiKey,
    ) -> Result<LlmBuilder<private::ApiKeySet, ()>, LlmError> {
        let key = match api_key {
            ApiKey::Default => {
                let provider = self.fields.provider.ok_or(LlmError::Builder(
                    "Provider must be set before API key".into(),
                ))?;

                env::var(provider.default_api_key_env_var()).map_err(|_| {
                    LlmError::Builder(format!(
                        "Missing {} environment variable",
                        provider.default_api_key_env_var()
                    ))
                })?
            }
            ApiKey::Custom(custom_key) => custom_key,
        };

        self.fields.api_key = Some(key);
        Ok(self.transition_state())
    }
}

impl LlmBuilder<private::ApiKeySet, ()> {
    /// Set the model to use for the LLM request.
    pub fn model(mut self, model_id: &str) -> LlmBuilder<private::Configuring, ()> {
        self.fields.model = Some(model_id.to_string());
        self.transition_state()
    }
}

impl LlmBuilder<private::Configuring, ()> {
    /// Set the messages for the conversation.
    pub fn messages(mut self, messages: Vec<Message>) -> LlmBuilder<private::MessagesSet, ()> {
        self.fields.messages = Some(messages);
        self.transition_state()
    }
}

impl<State: private::Completable, Ctx: Send + Sync + 'static> LlmBuilder<State, Ctx> {
    /// Set a custom timeout for the HTTP request.
    /// This is a convenience method that modifies the HttpClientConfig.
    pub fn timeout(mut self, duration: std::time::Duration) -> Self {
        let mut config = self.fields.http_client_config.unwrap_or_default();
        config.timeout = duration;
        self.fields.http_client_config = Some(config);
        self
    }

    /// Set the full HTTP client configuration (retries, backoff, etc).
    pub fn http_client_config(mut self, config: HttpClientConfig) -> Self {
        self.fields.http_client_config = Some(config);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.fields.max_tokens = Some(max_tokens);
        self
    }

    /// Set the temperature for generation (0.0 to 2.0).
    /// Lower values make output more focused and deterministic.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.fields.temperature = Some(temperature);
        self
    }

    /// Set the top_p value for nucleus sampling (0.0 to 1.0).
    /// An alternative to temperature; use one or the other, not both.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.fields.top_p = Some(top_p);
        self
    }

    /// Set a callback to inspect raw JSON requests before they are sent.
    ///
    /// The callback receives a reference to the serialized request body as JSON.
    /// This fires on ALL requests, including each iteration of tool-calling loops.
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{llm, ApiKey, Provider, Message, ChatRole, TextResponse};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let response = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Hello".to_string(),
    ///     }])
    ///     .inspect_request(|req| {
    ///         println!("Request: {}", serde_json::to_string_pretty(req).unwrap());
    ///     })
    ///     .complete::<TextResponse>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn inspect_request<F>(mut self, inspector: F) -> Self
    where
        F: Fn(&serde_json::Value) + Send + Sync + 'static,
    {
        let mut config = self.fields.inspector_config.take().unwrap_or_default();
        config.request_inspector = Some(Arc::new(inspector));
        self.fields.inspector_config = Some(config);
        self
    }

    /// Set a callback to inspect raw JSON responses after they are received.
    ///
    /// The callback receives a reference to the parsed response body as JSON.
    /// This fires on ALL responses, including each iteration of tool-calling loops
    /// and both success and error responses.
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{llm, ApiKey, Provider, Message, ChatRole, TextResponse};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let response = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Hello".to_string(),
    ///     }])
    ///     .inspect_response(|res| {
    ///         println!("Response: {}", serde_json::to_string_pretty(res).unwrap());
    ///     })
    ///     .complete::<TextResponse>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn inspect_response<F>(mut self, inspector: F) -> Self
    where
        F: Fn(&serde_json::Value) + Send + Sync + 'static,
    {
        let mut config = self.fields.inspector_config.take().unwrap_or_default();
        config.response_inspector = Some(Arc::new(inspector));
        self.fields.inspector_config = Some(config);
        self
    }

    /// Execute the LLM request and return an output defined by `T`.
    ///
    /// The target type `T` must implement [`CompletionTarget`]. Structured schemas can be created
    /// with the `#[completion_schema]` macro, while plain text responses can use [`TextResponse`].
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{completion_schema, llm, Message, ChatRole, ApiKey, Provider, TextResponse};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// #[completion_schema]
    /// struct Analysis {
    ///     sentiment: String,
    ///     confidence: f32,
    /// }
    ///
    /// let analysis = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Analyze: 'This library is amazing!'".to_string(),
    ///     }])
    ///     .complete::<Analysis>()
    ///     .await?;
    ///
    /// let text = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Say hello".to_string(),
    ///     }])
    ///     .complete::<TextResponse>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(
        name = "generate_completion",
        skip(self),
        fields(
            model = ?self.fields.model,
            provider = ?self.fields.provider,
            max_tokens = ?self.fields.max_tokens,
        ),
        err
    )]
    pub async fn complete<T>(self) -> Result<T::Output, LlmError>
    where
        T: super::traits::CompletionTarget + Send,
    {
        debug!("Starting generation request");
        let (messages, provider, model) = self.fields.validate()?;
        let model_string = model.to_string();
        let messages = messages.to_vec();
        let format = T::format()?;

        if !T::supports_tools() && self.fields.tool_registry.is_some() {
            return Err(LlmError::Builder(
                "Tools are only supported with structured completion targets".to_string(),
            ));
        }

        // Deferred error handling for tool registry errors in case of a poisoned lock.
        let tool_schemas = if T::supports_tools() {
            self.fields
                .tool_registry
                .as_ref()
                .map(|registry| registry.get_schemas())
                .transpose()?
                .map(|tools| tools.into_boxed_slice())
        } else {
            None
        };

        let conversation_messages: Vec<ConversationMessage> = messages
            .into_iter()
            .map(ConversationMessage::Chat)
            .collect();

        let req = StructuredRequest {
            model: model_string,
            messages: conversation_messages,
            tool_config: tool_schemas.map(|tools| ToolConfig {
                tools: Some(tools),
                tool_choice: self.fields.tool_choice.clone(),
                parallel_tool_calls: self.fields.parallel_tool_calls,
            }),
            generation_config: Some(GenerationConfig {
                max_tokens: self.fields.max_tokens,
                temperature: self.fields.temperature,
                top_p: self.fields.top_p,
            }),
        };

        let tool_registry = self.fields.tool_registry.as_ref();
        match provider {
            Provider::OpenAI => {
                let client = openai::create_openai_client_from_builder(&self)?;
                client
                    .generate_completion::<T, Ctx>(req, format, tool_registry)
                    .await
            }
            Provider::OpenRouter => {
                let client = openrouter::create_openrouter_client_from_builder(&self)?;
                client
                    .generate_completion::<T, Ctx>(req, format, tool_registry)
                    .await
            }
            Provider::Gemini => {
                let client = gemini::create_gemini_client_from_builder(&self)?;
                client
                    .generate_completion::<T, Ctx>(req, format, tool_registry)
                    .await
            }
        }
    }
}

impl LlmBuilder<private::MessagesSet, ()> {
    /// Set the tools for the LLM request with automatic execution support.
    /// This transitions to the ToolsSet state where tool behavior can be configured.
    ///
    /// By default parallel tool calling is enabled. This can be changed by calling `parallel_tool_calls` with `false`.
    pub fn tools<NewCtx: Send + Sync + 'static>(
        mut self,
        toolset: super::types::ToolSet<NewCtx>,
    ) -> LlmBuilder<private::ToolsSet, NewCtx> {
        self.fields.parallel_tool_calls = Some(true);
        let new_fields = self.fields.with_context_type(Some(toolset.registry));
        LlmBuilder {
            fields: new_fields,
            _state: PhantomData,
        }
    }
}

impl<Ctx: Send + Sync + 'static> LlmBuilder<private::ToolsSet, Ctx> {
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.fields.tool_choice = Some(choice);
        self
    }

    /// Set whether to enable parallel tool calls.
    /// When true, the model can call multiple tools simultaneously.
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.fields.parallel_tool_calls = Some(enabled);
        self
    }

    /// Set local safety limits for automatic tool execution.
    pub fn tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.fields.tool_calling_config = Some(config);
        self
    }
}

/// Module containing the main entry point for building LLM requests
pub mod llm {
    use super::*;

    /// Create a new LLM builder with the specified provider.
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{llm, ApiKey, Provider};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let builder = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with(provider: Provider) -> LlmBuilder<private::ProviderSet, ()> {
        let mut fields = BuilderFields::new();
        fields.provider = Some(provider);

        LlmBuilder {
            fields,
            _state: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_inspect_request_is_chainable() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let builder = llm::with(Provider::OpenAI)
            .api_key(ApiKey::Custom("test".into()))
            .unwrap()
            .model("gpt-4o-mini")
            .messages(vec![Message {
                role: super::super::types::ChatRole::User,
                content: "test".to_string(),
            }])
            .inspect_request(move |_| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .max_tokens(100);

        assert!(builder.fields.inspector_config.is_some());
        assert!(
            builder
                .fields
                .inspector_config
                .as_ref()
                .unwrap()
                .request_inspector
                .is_some()
        );
    }

    #[test]
    fn test_inspect_response_is_chainable() {
        let builder = llm::with(Provider::OpenAI)
            .api_key(ApiKey::Custom("test".into()))
            .unwrap()
            .model("gpt-4o-mini")
            .messages(vec![Message {
                role: super::super::types::ChatRole::User,
                content: "test".to_string(),
            }])
            .inspect_response(|_| {})
            .temperature(0.5);

        assert!(builder.fields.inspector_config.is_some());
        assert!(
            builder
                .fields
                .inspector_config
                .as_ref()
                .unwrap()
                .response_inspector
                .is_some()
        );
    }

    #[test]
    fn test_both_inspectors_can_be_set() {
        let builder = llm::with(Provider::OpenAI)
            .api_key(ApiKey::Custom("test".into()))
            .unwrap()
            .model("gpt-4o-mini")
            .messages(vec![Message {
                role: super::super::types::ChatRole::User,
                content: "test".to_string(),
            }])
            .inspect_request(|_| {})
            .inspect_response(|_| {});

        let config = builder.fields.inspector_config.as_ref().unwrap();
        assert!(config.request_inspector.is_some());
        assert!(config.response_inspector.is_some());
    }

    #[test]
    fn test_tool_calling_config_is_chainable_after_tools() {
        let toolset = super::super::types::ToolSet {
            registry: ToolRegistry::new(),
        };
        let config = ToolCallingConfig::new(2, Duration::from_secs(10))
            .with_max_tool_calls_per_turn(3)
            .with_max_concurrent_tool_calls(2)
            .with_tool_timeout(Duration::from_secs(1));

        let builder = llm::with(Provider::OpenAI)
            .api_key(ApiKey::Custom("test".into()))
            .unwrap()
            .model("gpt-4o-mini")
            .messages(vec![Message {
                role: super::super::types::ChatRole::User,
                content: "test".to_string(),
            }])
            .tools(toolset)
            .tool_calling_config(config.clone());

        let stored = builder
            .fields
            .tool_calling_config
            .as_ref()
            .expect("tool calling config");
        assert_eq!(stored.max_iterations, config.max_iterations);
        assert_eq!(stored.timeout, config.timeout);
        assert_eq!(
            stored.max_tool_calls_per_turn,
            config.max_tool_calls_per_turn
        );
        assert_eq!(
            stored.max_concurrent_tool_calls,
            config.max_concurrent_tool_calls
        );
        assert_eq!(stored.tool_timeout, config.tool_timeout);
    }

    #[test]
    fn test_inspector_config_is_cloneable() {
        let config = InspectorConfig {
            request_inspector: Some(Arc::new(|_| {})),
            response_inspector: Some(Arc::new(|_| {})),
        };

        let cloned = config.clone();
        assert!(cloned.request_inspector.is_some());
        assert!(cloned.response_inspector.is_some());
    }
}
