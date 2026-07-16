use thiserror::Error;

#[derive(Debug, Error)]
/// Errors returned by request building, providers, parsing, and tool execution.
pub enum LlmError {
    /// The high-level request builder contains invalid or missing values.
    #[error("LLM-Builder error: {0}")]
    Builder(String),

    /// A provider client is configured incorrectly.
    #[error("Provider configuration error: {0}")]
    ProviderConfiguration(String),

    /// A provider returned an error outside a normal API response.
    #[error("Provider error: {message}")]
    Provider {
        /// Human-readable error description.
        message: String,
        /// Optional underlying error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// An HTTP request could not be completed.
    #[error("Network error: {message}")]
    Network {
        /// Human-readable error description.
        message: String,
        /// Underlying transport error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A provider returned an unsuccessful API response.
    #[error("API error: {message}")]
    Api {
        /// Human-readable provider error description.
        message: String,
        /// HTTP status code when one was available.
        status_code: Option<u16>,
        /// Optional underlying response or decoding error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Provider output could not be parsed into the requested type.
    #[error("Parse error: {message}")]
    Parse {
        /// Human-readable parsing error description.
        message: String,
        /// Underlying parsing error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A tool returned an error or invalid result.
    #[error("Tool execution error: {message}")]
    ToolExecution {
        /// Human-readable execution error description.
        message: String,
        /// Optional underlying tool error.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// The requested tool name is not registered.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// The tool registry could not be read or written.
    #[error("Tool registry access failed: {message}")]
    ToolRegistryAccess {
        /// Human-readable registry error description.
        message: String,
    },

    /// Automatic tool calling exceeded its iteration limit.
    #[error("Tool call iteration limit exceeded: {limit} iterations")]
    ToolCallIterationLimit {
        /// Configured maximum number of iterations.
        limit: u32,
    },

    /// A response requested more tool calls than permitted in one turn.
    #[error("Tool call limit exceeded: requested {requested} calls, limit is {limit}")]
    ToolCallLimit {
        /// Number of tool calls requested by the provider.
        requested: usize,
        /// Configured maximum number of calls per turn.
        limit: usize,
    },

    /// Processing a batch of tool calls exceeded its timeout.
    #[error("Tool call processing timeout exceeded: {timeout:?}")]
    ToolCallTimeout {
        /// Configured batch timeout.
        timeout: std::time::Duration,
    },

    /// One tool execution exceeded its timeout.
    #[error("Tool execution timeout exceeded for {tool_name}: {timeout:?}")]
    ToolExecutionTimeout {
        /// Name of the timed-out tool.
        tool_name: String,
        /// Configured per-tool timeout.
        timeout: std::time::Duration,
    },

    /// A tool could not be added to a registry.
    #[error("Tool registration failed for {tool_name}: {message}")]
    ToolRegistration {
        /// Name of the tool that could not be registered.
        tool_name: String,
        /// Human-readable registration error description.
        message: String,
    },
}
