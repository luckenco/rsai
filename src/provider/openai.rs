//! OpenAI provider implementation.
//!
//! # API Compatibility
//!
//! This module preserves all fields from the OpenAI API responses, even those not currently used.
//! Fields marked with `#[allow(dead_code)]` are retained for:
//! - API contract completeness
//! - Future compatibility without breaking changes
//! - Debugging and logging purposes
//!
//! When adding new API structs, include all fields from the OpenAI documentation and mark
//! unused ones with `#[allow(dead_code)]` rather than omitting them.

use crate::provider::constants::openai;

use crate::core::{
    InspectorConfig, LlmBuilder, LlmError, LlmProvider, StructuredRequest, ToolCallingConfig,
    ToolCallingGuard, ToolRegistry,
};
use crate::responses::{HttpClientConfig, ResponsesClient, ResponsesProviderConfig};
use async_trait::async_trait;

/// OpenAI-specific configuration for the responses client
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
    /// Configuration for tool calling limits
    pub tool_calling_config: Option<ToolCallingConfig>,
    pub http_config: HttpClientConfig,
    /// Configuration for request/response inspection
    pub inspector_config: Option<InspectorConfig>,
}

impl OpenAiConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openai::API_BASE.to_string(),
            tool_calling_config: Some(ToolCallingConfig::default()),
            http_config: HttpClientConfig::default(),
            inspector_config: None,
        }
    }

    pub fn with_inspector_config(mut self, config: InspectorConfig) -> Self {
        self.inspector_config = Some(config);
        self
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.tool_calling_config = Some(config);
        self
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Self {
        self.http_config = config;
        self
    }

    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        match self.tool_calling_config {
            Some(ref config) => ToolCallingGuard::from_config(config),
            None => ToolCallingGuard::default(),
        }
    }
}

impl ResponsesProviderConfig for OpenAiConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self) -> &str {
        openai::RESPONSES_ENDPOINT
    }

    fn auth_header(&self) -> (String, String) {
        (
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        )
    }

    fn provider(&self) -> super::Provider {
        self.provider()
    }

    fn http_config(&self) -> HttpClientConfig {
        self.http_config.clone()
    }

    fn inspector_config(&self) -> Option<&InspectorConfig> {
        self.inspector_config.as_ref()
    }
}

impl OpenAiConfig {
    /// Get the provider type for this configuration
    pub fn provider(&self) -> crate::provider::Provider {
        crate::provider::Provider::OpenAI
    }
}

pub struct OpenAiClient {
    responses_client: ResponsesClient<OpenAiConfig>,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = OpenAiConfig::new(api_key);
        Ok(Self {
            responses_client: ResponsesClient::new(config)?,
        })
    }

    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        // Create a new config with the updated base_url using the current API key
        let current_api_key = &self.responses_client.config.api_key;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url,
            tool_calling_config: self.responses_client.config.tool_calling_config.clone(),
            http_config: self.responses_client.config.http_config.clone(),
            inspector_config: self.responses_client.config.inspector_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Result<Self, LlmError> {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: Some(config),
            http_config: self.responses_client.config.http_config.clone(),
            inspector_config: self.responses_client.config.inspector_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Result<Self, LlmError> {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let tool_config = &self.responses_client.config.tool_calling_config;

        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: tool_config.clone(),
            http_config: config,
            inspector_config: self.responses_client.config.inspector_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: crate::responses::Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
    ) -> Result<T::Output, LlmError>
    where
        T: crate::CompletionTarget + Send,
        Ctx: Send + Sync + 'static,
    {
        let guard = self.responses_client.config.get_tool_calling_guard();
        self.responses_client
            .generate_completion::<T, Ctx>(request, format, tool_registry, guard)
            .await
    }
}

pub fn create_openai_client_from_builder<State, Ctx>(
    builder: &LlmBuilder<State, Ctx>,
) -> Result<OpenAiClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENAI_API_KEY not set.".to_string()))?
        .to_string();

    let mut config = OpenAiConfig::new(api_key);

    if let Some(http_config) = builder.get_http_config() {
        config = config.with_http_config(http_config.clone());
    }

    if let Some(inspector_config) = builder.get_inspector_config() {
        config = config.with_inspector_config(inspector_config.clone());
    }

    let client = ResponsesClient::new(config)?;
    Ok(OpenAiClient {
        responses_client: client,
    })
}
