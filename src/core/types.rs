use crate::core::{LlmError, traits::CompletionTarget, traits::ToolFunction};
use crate::provider::Provider;
use crate::responses::{self, request::Format};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
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
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallResult {
    pub id: String,
    pub tool_call_id: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationMessage {
    Chat(Message),
    ToolCall(ToolCall),
    ToolCallResult(ToolCallResult),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredRequest {
    pub model: String,
    pub messages: Vec<ConversationMessage>,
    pub tool_config: Option<ToolConfig>,
    pub generation_config: Option<GenerationConfig>,
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
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter (0.0 to 1.0)
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredResponse<T> {
    pub content: T,
    pub usage: LanguageModelUsage,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextResponse {
    pub text: String,
    pub usage: LanguageModelUsage,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageModelUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseMetadata {
    pub provider: Provider,
    pub model: String,
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

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct ToolSet<Ctx = ()> {
    pub registry: ToolRegistry<Ctx>,
}

impl<Ctx: Send + Sync + 'static> ToolSet<Ctx> {
    pub fn tools(&self) -> Result<Vec<Tool>, LlmError> {
        self.registry.get_schemas()
    }
}

/// Builder for creating a ToolSet with context.
/// Created by the `toolset!` macro when a context type is specified.
pub struct ToolSetBuilder<Ctx> {
    tools: Vec<Arc<dyn ToolFunction<Ctx>>>,
    _marker: PhantomData<Ctx>,
}

impl<Ctx: Send + Sync + 'static> ToolSetBuilder<Ctx> {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn add_tool(mut self, tool: Arc<dyn ToolFunction<Ctx>>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Finalize the toolset with the given context.
    pub fn with_context(self, context: Ctx) -> ToolSet<Ctx> {
        let registry = ToolRegistry::with_context(context);
        for tool in self.tools {
            registry
                .register(tool)
                .expect("Failed to register tool in ToolSetBuilder");
        }
        ToolSet { registry }
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
