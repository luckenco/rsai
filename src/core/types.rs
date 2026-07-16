use crate::core::{LlmError, traits::CompletionTarget, traits::ToolFunction};
use crate::provider::Provider;
use crate::responses::{self, request::Format};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use tracing::warn;

/// Marker type for context/dependency injection in tools.
///
/// Use this type wrapper in tool function parameters to inject dependencies from the context.
/// The macro will recognize `Ctx<&T>` parameters and extract them from the tool registry's context
/// using `AsRef<T>`.
///
/// # Example
///
/// ```rust,ignore
/// use rsai::{tool, Ctx};
///
/// struct DatabasePool { /* ... */ }
///
/// #[tool]
/// /// Search documents in the database
/// /// query: The search query
/// fn search_docs(db: Ctx<&DatabasePool>, query: String) -> Vec<String> {
///     db.search(&query)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Ctx<T>(pub T);

impl<T> Deref for Ctx<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> From<T> for Ctx<T> {
    fn from(value: T) -> Self {
        Ctx(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Role assigned to a chat message.
pub enum ChatRole {
    /// Instruction that guides the model for the conversation.
    System,
    /// Input supplied by the user.
    User,
    /// Prior output supplied by the assistant.
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
/// One text message in a conversation.
pub struct Message {
    /// Sender role for the message.
    pub role: ChatRole,
    /// Text content of the message.
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
/// A function call requested by a model.
pub struct ToolCall {
    /// Provider response item identifier.
    pub id: String,
    /// Identifier used to pair this call with its result.
    pub call_id: String,
    /// Registered tool name.
    pub name: String,
    /// JSON object containing the tool arguments.
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
/// Result returned for a previous tool call.
pub struct ToolCallResult {
    /// Provider response item identifier.
    pub id: String,
    /// Identifier of the tool call this result answers.
    pub tool_call_id: String,
    /// JSON value returned by the tool.
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
/// Provider-independent conversation history item.
pub enum ConversationMessage {
    /// A regular text message.
    Chat(Message),
    /// A tool call requested by the model.
    ToolCall(ToolCall),
    /// A result supplied for a prior tool call.
    ToolCallResult(ToolCallResult),
}

#[derive(Debug, Clone, PartialEq)]
/// JSON Schema declaration for a callable tool.
pub struct Tool {
    /// Unique function name exposed to the model.
    pub name: String,
    /// Optional description of the function's behavior.
    pub description: Option<String>,
    /// JSON Schema object describing accepted arguments.
    pub parameters: Value,
    /// Whether providers should enforce the parameter schema strictly.
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
/// Strategy used by a provider to select tools.
pub enum ToolChoice {
    /// Do not call a tool.
    None,
    /// Let the model decide whether to call a tool.
    Auto,
    /// Require the model to call at least one tool.
    Required,
    /// Require one named tool.
    Function {
        /// Tool name that the provider must call.
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
/// Provider-independent request passed to an [`LlmProvider`](crate::LlmProvider).
pub struct StructuredRequest {
    /// Provider model identifier.
    pub model: String,
    /// Conversation sent to the model.
    pub messages: Vec<ConversationMessage>,
    /// Optional tool declarations and selection settings.
    pub tool_config: Option<ToolConfig>,
    /// Optional generation settings.
    pub generation_config: Option<GenerationConfig>,
}

impl StructuredRequest {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if let Some(config) = &self.generation_config {
            config.validate()?;
        }
        Ok(())
    }
}

/// Configuration for tool calling behavior
#[derive(Debug, Clone, PartialEq)]
pub struct ToolConfig {
    /// Available tools for the model to call
    pub tools: Option<Box<[Tool]>>,
    /// Strategy for choosing which tools to call
    pub tool_choice: Option<ToolChoice>,
    /// Whether to allow parallel tool calls (default: true)
    pub parallel_tool_calls: Option<bool>,
}

/// Configuration for text generation parameters
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate. Must be greater than zero.
    pub max_tokens: Option<u32>,

    /// Sampling temperature from 0 through 2.
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter from 0 through 1.
    pub top_p: Option<f32>,
}

impl GenerationConfig {
    fn validate(&self) -> Result<(), LlmError> {
        if self.max_tokens == Some(0) {
            return Err(LlmError::Builder(
                "max_tokens must be greater than zero".to_string(),
            ));
        }

        match self.temperature {
            Some(temperature) if !(0.0..=2.0).contains(&temperature) => {
                return Err(LlmError::Builder(
                    "temperature must be between 0 and 2".to_string(),
                ));
            }
            _ => {}
        }

        match self.top_p {
            Some(top_p) if !(0.0..=1.0).contains(&top_p) => {
                return Err(LlmError::Builder(
                    "top_p must be between 0 and 1".to_string(),
                ));
            }
            _ => {}
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Typed structured output together with usage and provider metadata.
pub struct StructuredResponse<T> {
    /// Parsed response content.
    pub content: T,
    /// Token usage reported by the provider.
    pub usage: LanguageModelUsage,
    /// Provider and response identifiers.
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text output together with usage and provider metadata.
pub struct TextResponse {
    /// Generated text.
    pub text: String,
    /// Token usage reported by the provider.
    pub usage: LanguageModelUsage,
    /// Provider and response identifiers.
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq)]
/// Token counts reported for one model request.
pub struct LanguageModelUsage {
    /// Tokens consumed by the input.
    pub prompt_tokens: i32,
    /// Tokens generated in the output.
    pub completion_tokens: i32,
    /// Total input and output tokens.
    pub total_tokens: i32,
}

#[derive(Debug, Clone, PartialEq)]
/// Identifiers attached to a model response.
pub struct ResponseMetadata {
    /// Provider that produced the response.
    pub provider: Provider,
    /// Provider model identifier.
    pub model: String,
    /// Provider response identifier, or an empty string when unavailable.
    pub id: String,
}

/// Provider-agnostic response type that all providers convert to.
/// This is the unified response format used by `CompletionTarget::parse_response`.
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub id: String,
    pub model: String,
    pub provider: Provider,
    pub content: ResponseContent,
    pub usage: LanguageModelUsage,
}

/// The content of a provider response - either text, function calls, or a refusal.
#[derive(Debug, Clone)]
pub enum ResponseContent {
    /// Plain text or structured JSON text response
    Text(String),
    /// One or more function calls requested by the model
    FunctionCalls(Vec<FunctionCallData>),
    /// Model refused to respond
    Refusal(String),
}

/// Data for a function call requested by the model.
#[derive(Debug, Clone)]
pub struct FunctionCallData {
    /// Unique identifier for this function call
    pub id: String,
    /// The name of the function to call
    pub name: String,
    /// The arguments to pass to the function (as JSON)
    pub arguments: Value,
}

type ToolMap<Ctx> = Arc<RwLock<HashMap<String, Arc<dyn ToolFunction<Ctx>>>>>;

/// Thread-safe collection of callable tools sharing one context value.
pub struct ToolRegistry<Ctx = ()> {
    tools: ToolMap<Ctx>,
    context: Arc<Ctx>,
}

impl ToolRegistry<()> {
    /// Create a new tool registry without context (for backward compatibility)
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            context: Arc::new(()),
        }
    }
}

impl<Ctx: Send + Sync + 'static> ToolRegistry<Ctx> {
    /// Create a new tool registry with the given context
    pub fn with_context(context: Ctx) -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            context: Arc::new(context),
        }
    }

    /// Registers a new tool in the registry.
    ///
    /// # Arguments
    /// * `tool` - The tool to register, wrapped in an Arc
    ///
    /// # Returns
    /// * `Ok(())` if the tool was successfully registered
    /// * `Err(LlmError::ToolRegistration)` if a tool with the same name already exists
    ///
    /// # Errors
    /// This function will return an error if:
    /// - A tool with the same name is already registered
    /// - The registry's write lock is poisoned (indicates a panic in another thread)
    ///
    /// # Thread Safety
    /// This method is thread-safe. Multiple threads can register tools concurrently,
    /// but attempting to register the same tool name from multiple threads will
    /// result in only one success and the rest will return errors.
    pub fn register(&self, tool: Arc<dyn ToolFunction<Ctx>>) -> Result<(), LlmError> {
        let schema = tool.schema();
        let schema_name = schema.name.clone();

        let mut w_tools = self.tools.write().map_err(|_| LlmError::ToolRegistration {
            tool_name: schema_name.clone(),
            message: "Failed to acquire write lock on tool registry".to_string(),
        })?;

        if w_tools.contains_key(&schema.name) {
            return Err(LlmError::ToolRegistration {
                tool_name: schema.name.clone(),
                message: format!("Tool {} already registered", schema.name),
            });
        }

        w_tools.insert(schema.name, tool);
        Ok(())
    }

    /// Insert a tool, replacing a tool with the same name if one exists.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::ToolRegistration`] if the registry lock is poisoned.
    pub fn overwrite(&self, tool: Arc<dyn ToolFunction<Ctx>>) -> Result<(), LlmError> {
        let schema = tool.schema();
        let schema_name = schema.name.clone();

        let mut w_tools = self.tools.write().map_err(|_| LlmError::ToolRegistration {
            tool_name: schema_name.clone(),
            message: "Failed to acquire write lock on tool registry".to_string(),
        })?;

        let overwritten_tool = w_tools.insert(schema.name, tool);

        if overwritten_tool.is_some() {
            warn!(schema_name, "Tool was overwritten")
        }

        Ok(())
    }

    /// Return schemas for all registered tools.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::ToolRegistryAccess`] if the registry lock is poisoned.
    pub fn get_schemas(&self) -> Result<Vec<Tool>, LlmError> {
        let r_tools = self
            .tools
            .read()
            .map_err(|_| LlmError::ToolRegistryAccess {
                message: "Failed to acquire read lock (lock poisoned)".to_string(),
            })?;
        let schema = r_tools.values().map(|tool| tool.schema()).collect();
        Ok(schema)
    }

    #[tracing::instrument(
        name = "execute_tool",
        skip(self, tool_call),
        fields(
            tool_name = %tool_call.name,
            call_id = %tool_call.call_id
        ),
        err
    )]
    /// Execute a registered tool with this registry's shared context.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::ToolNotFound`] for an unknown name, a registry access error if the
    /// lock is poisoned, or the error returned by the tool.
    pub async fn execute(&self, tool_call: &ToolCall) -> Result<serde_json::Value, LlmError> {
        tracing::trace!("Executing tool");

        let tool = {
            let r_tools = self
                .tools
                .read()
                .map_err(|_| LlmError::ToolRegistryAccess {
                    message: "Failed to acquire read lock (lock poisoned)".to_string(),
                })?;
            r_tools.get(&tool_call.name).cloned()
        };

        let result = if let Some(tool) = tool {
            tool.execute_owned(self.context.clone(), tool_call.arguments.clone())
                .await
        } else {
            Err(LlmError::ToolNotFound(tool_call.name.clone()))
        };

        if result.is_ok() {
            tracing::debug!("Tool execution completed successfully");
        }

        result
    }
}

impl Default for ToolRegistry<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Boxed, sendable future used by generated tool implementations.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A registry ready to attach to an [`LlmBuilder`](crate::LlmBuilder).
pub struct ToolSet<Ctx = ()> {
    /// Registry containing the tool implementations and shared context.
    pub registry: ToolRegistry<Ctx>,
}

impl<Ctx: Send + Sync + 'static> ToolSet<Ctx> {
    /// Return schemas for all tools in this set.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::ToolRegistryAccess`] if the registry lock is poisoned.
    pub fn tools(&self) -> Result<Vec<Tool>, LlmError> {
        self.registry.get_schemas()
    }
}

/// Builder for creating a ToolSet with context.
/// Created by the `toolset!` macro when a context type is specified.
pub struct ToolSetBuilder<Ctx> {
    tools: Vec<Arc<dyn ToolFunction<Ctx>>>,
}

impl<Ctx: Send + Sync + 'static> ToolSetBuilder<Ctx> {
    /// Create an empty context-aware toolset builder.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool implementation to the builder.
    pub fn add_tool(mut self, tool: Arc<dyn ToolFunction<Ctx>>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Finalize the toolset with the given context.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::ToolRegistration`] when two tools have the same name.
    pub fn with_context(self, context: Ctx) -> Result<ToolSet<Ctx>, LlmError> {
        let registry = ToolRegistry::with_context(context);
        for tool in self.tools {
            registry.register(tool)?;
        }
        Ok(ToolSet { registry })
    }
}

impl<Ctx: Send + Sync + 'static> Default for ToolSetBuilder<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for non-object JSON Schema types (enums, strings, numbers, etc.)
/// OpenAI's structured output API requires the root schema to be an object,
/// so we wrap non-object types in an object with a "value" property.
#[derive(serde::Deserialize)]
struct ValueWrapper<T> {
    value: T,
}

impl<T> CompletionTarget for T
where
    T: DeserializeOwned + JsonSchema + Send,
{
    type Output = StructuredResponse<T>;

    fn format() -> Result<Format, LlmError> {
        responses::create_format_for_type::<T>()
    }

    fn parse_response(res: ProviderResponse) -> Result<Self::Output, LlmError> {
        match res.content {
            ResponseContent::Text(text) => {
                // Try to parse as wrapped value first, then fall back to direct parsing
                let parsed_content: T =
                    if let Ok(wrapped) = serde_json::from_str::<ValueWrapper<T>>(&text) {
                        wrapped.value
                    } else {
                        serde_json::from_str(&text).map_err(|e| LlmError::Parse {
                            message: "Failed to parse structured output".to_string(),
                            source: Box::new(e),
                        })?
                    };

                Ok(StructuredResponse {
                    content: parsed_content,
                    usage: res.usage,
                    metadata: ResponseMetadata {
                        provider: res.provider,
                        model: res.model,
                        id: res.id,
                    },
                })
            }
            ResponseContent::FunctionCalls(_) => Err(LlmError::Provider {
                message: "Function call response received when expecting structured output"
                    .to_string(),
                source: None,
            }),
            ResponseContent::Refusal(refusal) => Err(LlmError::Api {
                message: format!("Model refused: {}", refusal),
                status_code: None,
                source: None,
            }),
        }
    }
}

impl CompletionTarget for TextResponse {
    type Output = TextResponse;

    fn format() -> Result<Format, LlmError> {
        Ok(responses::create_text_format())
    }

    fn parse_response(res: ProviderResponse) -> Result<Self::Output, LlmError> {
        match res.content {
            ResponseContent::Text(text) => Ok(TextResponse {
                text,
                usage: res.usage,
                metadata: ResponseMetadata {
                    provider: res.provider,
                    model: res.model,
                    id: res.id,
                },
            }),
            ResponseContent::FunctionCalls(_) => Err(LlmError::Provider {
                message: "Function call response received when expecting text output".to_string(),
                source: None,
            }),
            ResponseContent::Refusal(refusal) => Err(LlmError::Api {
                message: format!("Model refused: {}", refusal),
                status_code: None,
                source: None,
            }),
        }
    }

    fn supports_tools() -> bool {
        // TextResponse supports tools - the tool calling loop processes function calls
        // and the model eventually returns text after tools are executed.
        // Provider-specific constraints (e.g., Gemini can't combine tools with structured
        // JSON output) are validated at the provider level.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::writer::MakeWriter;

    #[test]
    fn generation_config_validation_accepts_valid_values() {
        for config in [
            GenerationConfig {
                max_tokens: Some(1),
                temperature: Some(0.0),
                top_p: None,
            },
            GenerationConfig {
                max_tokens: None,
                temperature: Some(2.0),
                top_p: None,
            },
            GenerationConfig {
                max_tokens: None,
                temperature: None,
                top_p: Some(1.0),
            },
            GenerationConfig {
                max_tokens: None,
                temperature: Some(1.0),
                top_p: Some(0.5),
            },
        ] {
            config.validate().expect("valid generation config");
        }
    }

    #[test]
    fn generation_config_validation_rejects_invalid_values() {
        let invalid_configs = [
            (
                GenerationConfig {
                    max_tokens: Some(0),
                    temperature: None,
                    top_p: None,
                },
                "max_tokens",
            ),
            (
                GenerationConfig {
                    max_tokens: None,
                    temperature: Some(f32::NAN),
                    top_p: None,
                },
                "temperature",
            ),
            (
                GenerationConfig {
                    max_tokens: None,
                    temperature: Some(2.1),
                    top_p: None,
                },
                "temperature",
            ),
            (
                GenerationConfig {
                    max_tokens: None,
                    temperature: None,
                    top_p: Some(1.1),
                },
                "top_p",
            ),
        ];

        for (config, expected_message) in invalid_configs {
            let error = config.validate().expect_err("invalid generation config");
            assert!(error.to_string().contains(expected_message));
        }
    }

    #[derive(Clone)]
    struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

    impl<'a> MakeWriter<'a> for CapturedLogs {
        type Writer = CapturedLogWriter;

        fn make_writer(&'a self) -> Self::Writer {
            CapturedLogWriter(self.0.clone())
        }
    }

    struct CapturedLogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedLogWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("captured log lock poisoned")
                .extend(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct ObjectTool;

    impl ToolFunction<()> for ObjectTool {
        fn schema(&self) -> Tool {
            Tool {
                name: "object_tool".to_string(),
                description: Some("Returns an object".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                strict: Some(true),
            }
        }

        fn execute<'a>(
            &'a self,
            _ctx: &'a (),
            _params: serde_json::Value,
        ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>> {
            Box::pin(async move {
                Ok(serde_json::json!({
                    "name": "test",
                    "value":42,
                    "active": true
                }))
            })
        }
    }

    #[tokio::test]
    async fn test_tool_registry_preservers_object_types() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(ObjectTool))
            .expect("Failed to register object_tool");

        let tool_call = ToolCall {
            id: "test_Id".to_string(),
            call_id: "call_123".to_string(),
            name: "object_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let result = registry.execute(&tool_call).await.unwrap();

        // Verify the result is still structured data and not just a string
        assert!(result.is_object());
        assert_eq!(result["name"], "test");
        assert_eq!(result["value"], 42);
        assert_eq!(result["active"], true);
    }

    #[tokio::test]
    async fn tool_execution_traces_do_not_include_arguments() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(CapturedLogs(logs.clone()))
            .with_ansi(false)
            .without_time()
            .finish();
        let dispatch = tracing::Dispatch::new(subscriber);
        let _guard = tracing::dispatcher::set_default(&dispatch);

        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(ObjectTool))
            .expect("Failed to register object_tool");

        let secret = "sk-test-secret-tool-argument";
        let tool_call = ToolCall {
            id: "test_Id".to_string(),
            call_id: "call_123".to_string(),
            name: "object_tool".to_string(),
            arguments: serde_json::json!({ "api_key": secret }),
        };

        registry.execute(&tool_call).await.unwrap();
        drop(_guard);

        let logs = String::from_utf8(logs.lock().expect("captured log lock poisoned").clone())
            .expect("captured logs should be valid UTF-8");

        assert!(logs.contains("Executing tool"));
        assert!(!logs.contains(secret));
        assert!(!logs.contains("api_key"));
    }
}
