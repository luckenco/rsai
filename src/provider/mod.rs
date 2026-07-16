mod constants;
pub(crate) mod gemini;
pub(crate) mod openai;
pub(crate) mod openrouter;

pub use gemini::{GeminiClient, GeminiConfig};
pub use openai::{OpenAiClient, OpenAiConfig};
pub use openrouter::{OpenRouterClient, OpenRouterConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Model provider used for a completion request.
pub enum Provider {
    /// OpenAI Responses API.
    OpenAI,
    /// OpenRouter Responses API.
    OpenRouter,
    /// Google Gemini generate-content API.
    Gemini,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
            Provider::OpenRouter => write!(f, "OpenRouter"),
            Provider::Gemini => write!(f, "Gemini"),
        }
    }
}

impl Provider {
    /// Get the default environment variable name for this provider's API key
    pub fn default_api_key_env_var(&self) -> &'static str {
        match self {
            Provider::OpenAI => constants::openai::API_KEY_ENV_VAR,
            Provider::OpenRouter => constants::openrouter::API_KEY_ENV_VAR,
            Provider::Gemini => constants::gemini::API_KEY_ENV_VAR,
        }
    }
}
