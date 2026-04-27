use async_trait::async_trait;
use std::sync::Arc;

use crate::responses::request::Format;

use super::{
    error::LlmError,
    types::{BoxFuture, ProviderResponse, StructuredRequest, Tool, ToolRegistry},
};

#[async_trait]
pub trait LlmProvider {
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

pub trait ToolFunction<Ctx = ()>: Send + Sync {
    fn schema(&self) -> Tool;

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

pub trait CompletionTarget: Sized + Send {
    type Output;

    fn format() -> Result<Format, LlmError>;

    /// Parse a provider-agnostic response into the target output type.
    fn parse_response(res: ProviderResponse) -> Result<Self::Output, LlmError>;

    fn supports_tools() -> bool {
        true
    }
}
