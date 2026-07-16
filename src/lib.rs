//! # rsai
//!
//! Predictable development for unpredictable models. Let the compiler handle the chaos.
//!
//! ## ⚠️ WARNING
//!
//! This is a pre-release version with an unstable API. Breaking changes may occur between versions.
//! Use with caution and pin to specific versions in production applications.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rsai::{llm, Message, ChatRole, ApiKey, Provider, TextResponse, completion_schema};
//!
//! #[completion_schema]
//! struct Analysis {
//!     sentiment: String,
//!     confidence: f32,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let analysis = llm::with(Provider::OpenAI)
//!     .api_key(ApiKey::Default)?
//!     .model("gpt-4o-mini")
//!     .messages(vec![Message {
//!         role: ChatRole::User,
//!         content: "Analyze: 'This library is amazing!'".to_string(),
//!     }])
//!     .complete::<Analysis>()
//!     .await?;
//!
//! let reply = llm::with(Provider::OpenAI)
//!     .api_key(ApiKey::Default)?
//!     .model("gpt-4o-mini")
//!     .messages(vec![
//!         Message {
//!             role: ChatRole::System,
//!             content: "You are friendly and concise.".to_string(),
//!         },
//!         Message {
//!             role: ChatRole::User,
//!             content: "Share a fun fact about Rust.".to_string(),
//!         },
//!     ])
//!     .complete::<TextResponse>()
//!     .await?;
//!
//! println!("{}", reply.text);
//! Ok(())
//! }
//! ```
//!
#![warn(missing_docs)]

mod completions;
mod core;
mod provider;
mod responses;

// Core types
pub use core::{ChatRole, ConversationMessage, Ctx, Message};
pub use core::{Tool, ToolCall, ToolCallResult, ToolRegistry, ToolSet, ToolSetBuilder};
pub use core::{ToolCallingConfig, ToolCallingGuard};

// Configuration types
pub use core::{
    ApiKey, BaseUrlSecurity, GenerationConfig, Inspector, InspectorConfig, LlmBuilder, ToolChoice,
    ToolConfig,
};
pub use responses::{Format, HttpClientConfig};

// Response types
pub use core::{
    LanguageModelUsage, ResponseMetadata, StructuredRequest, StructuredResponse, TextResponse,
};

// Async helpers
pub use core::BoxFuture;

// Error handling
pub use core::LlmError;
/// Result type used by rsai APIs.
pub type Result<T> = std::result::Result<T, LlmError>;

#[doc(hidden)]
pub mod __private {
    pub use schemars;
    pub use serde;
    pub use serde_json;

    pub async fn spawn_blocking_tool<F>(f: F) -> crate::Result<serde_json::Value>
    where
        F: FnOnce() -> crate::Result<serde_json::Value> + Send + 'static,
    {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return f();
        };

        match handle.spawn_blocking(f).await {
            Ok(result) => result,
            Err(err) => {
                if err.is_panic() {
                    std::panic::resume_unwind(err.into_panic());
                }

                Err(crate::LlmError::ToolExecution {
                    message: "Blocking tool task failed".to_string(),
                    source: Some(Box::new(err)),
                })
            }
        }
    }
}

// Gen AI request builders
pub use core::llm;

// Gen AI providers
pub use provider::{
    GeminiClient, GeminiConfig, OpenAiClient, OpenAiConfig, OpenRouterClient, OpenRouterConfig,
    Provider,
};

// Traits
pub use core::{CompletionTarget, LlmProvider, ToolFunction};

// Macros from `rsai-macros`
pub use rsai_macros::{completion_schema, tool, toolset};
