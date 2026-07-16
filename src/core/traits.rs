use async_trait::async_trait;
use std::sync::Arc;

use crate::responses::request::Format;

use super::{
    error::LlmError,
    types::{BoxFuture, ProviderResponse, StructuredRequest, Tool, ToolRegistry},
};

#[async_trait]
/// Provider contract for executing provider-independent completion requests.
pub trait LlmProvider {
    /// Generate and parse one completion, executing registered tools when requested.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when request validation, provider I/O, tool execution, or response
    /// parsing fails.
    async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget + Send,
        Ctx: Send + Sync + 'static;
}

/// Callable tool implementation with an optional shared context type.
pub trait ToolFunction<Ctx = ()>: Send + Sync {
    /// Return the tool name, description, and parameter schema exposed to the model.
    fn schema(&self) -> Tool;

    /// Execute the tool with shared context and JSON arguments.
    fn execute<'a>(
        &'a self,
        ctx: &'a Ctx,
        params: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>>;

    #[doc(hidden)]
    fn execute_owned(
        self: Arc<Self>,
        ctx: Arc<Ctx>,
        params: serde_json::Value,
    ) -> BoxFuture<'static, Result<serde_json::Value, LlmError>>
    where
        Self: 'static,
        Ctx: Send + Sync + 'static,
    {
        Box::pin(async move { self.execute(ctx.as_ref(), params).await })
    }
}

/// Target type that defines provider formatting and response parsing.
pub trait CompletionTarget: Sized + Send {
    /// Value returned to the caller after parsing.
    type Output;

    /// Build the provider-independent response format for this target.
    fn format() -> Result<Format, LlmError>;

    /// Parse a provider-agnostic response into the target output type.
    fn parse_response(res: ProviderResponse) -> Result<Self::Output, LlmError>;

    /// Whether this target supports automatic tool calling.
    fn supports_tools() -> bool {
        true
    }
}
