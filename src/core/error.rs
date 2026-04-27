use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM-Builder error: {0}")]
    Builder(String),

    #[error("Provider configuration error: {0}")]
    ProviderConfiguration(String),

    #[error("Provider error: {message}")]
    Provider {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Network error: {message}")]
    Network {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("API error: {message}")]
    Api {
        message: String,
        status_code: Option<u16>,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Parse error: {message}")]
    Parse {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Tool execution error: {message}")]
    ToolExecution {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool registry access failed: {message}")]
    ToolRegistryAccess { message: String },

    #[error("Tool call iteration limit exceeded: {limit} iterations")]
    ToolCallIterationLimit { limit: u32 },

    #[error("Tool call limit exceeded: requested {requested} calls, limit is {limit}")]
    ToolCallLimit { requested: usize, limit: usize },

    #[error("Tool call processing timeout exceeded: {timeout:?}")]
    ToolCallTimeout { timeout: std::time::Duration },

    #[error("Tool execution timeout exceeded for {tool_name}: {timeout:?}")]
    ToolExecutionTimeout {
        tool_name: String,
        timeout: std::time::Duration,
    },

    #[error("Tool registration failed for {tool_name}: {message}")]
    ToolRegistration { tool_name: String, message: String },
}
