//! OpenRouter provider implementation.
//!
//! # API Compatibility
//!
//! This module preserves all fields from the OpenRouter API responses, even those not currently used.
//! Fields marked with `#[allow(dead_code)]` are retained for:
//! - API contract completeness
//! - Future compatibility without breaking changes
//! - Debugging and logging purposes
//!
//! When adding new API structs, include all fields from the OpenRouter documentation and mark
//! unused ones with `#[allow(dead_code)]` rather than omitting them.

use crate::provider::constants::openrouter;
use crate::responses::{HttpClientConfig, ResponsesClient, ResponsesProviderConfig};

use crate::core::{
    InspectorConfig, LlmBuilder, LlmError, LlmProvider, StructuredRequest, ToolCallingConfig,
    ToolCallingGuard, ToolRegistry,
};
use async_trait::async_trait;

/// OpenRouter-specific configuration for the responses client
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub http_referer: Option<String>,
    pub x_title: Option<String>,
    pub http_config: HttpClientConfig,
    /// Configuration for tool calling limits
    pub tool_calling_config: Option<ToolCallingConfig>,
    /// Configuration for request/response inspection
    pub inspector_config: Option<InspectorConfig>,
}

impl OpenRouterConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openrouter::API_BASE.to_string(),
            http_referer: None,
            x_title: None,
            tool_calling_config: Some(ToolCallingConfig::default()),
            http_config: HttpClientConfig::default(),
            inspector_config: None,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Self {
        self.http_config = config;
        self
    }

    pub fn with_inspector_config(mut self, config: InspectorConfig) -> Self {
        self.inspector_config = Some(config);
        self
    }

    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.http_referer = Some(http_referer);
        self
    }

    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.x_title = Some(x_title);
        self
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.tool_calling_config = Some(config);
        self
    }

    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::with_limits(config.max_iterations, config.timeout)
        } else {
            ToolCallingGuard::new()
        }
    }
}

impl ResponsesProviderConfig for OpenRouterConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self) -> &str {
        openrouter::RESPONSES_ENDPOINT
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

    fn extra_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        if let Some(referer) = &self.http_referer {
            headers.push(("HTTP-Referer".to_string(), referer.clone()));
        }

        if let Some(title) = &self.x_title {
            headers.push(("X-Title".to_string(), title.clone()));
        }

        headers
    }

    fn http_config(&self) -> HttpClientConfig {
        self.http_config.clone()
    }

    fn inspector_config(&self) -> Option<&InspectorConfig> {
        self.inspector_config.as_ref()
    }
}

impl OpenRouterConfig {
    /// Get the provider type for this configuration
    pub fn provider(&self) -> crate::provider::Provider {
        crate::provider::Provider::OpenRouter
    }
}

pub struct OpenRouterClient {
    responses_client: ResponsesClient<OpenRouterConfig>,
}

impl OpenRouterClient {
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = OpenRouterConfig::new(api_key);
        Ok(Self {
            responses_client: ResponsesClient::new(config)?,
        })
    }

    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        // Create a new config with the updated base_url using the current API key
        let current_api_key = &self.responses_client.config.api_key;
        let http_referer = self.responses_client.config.http_referer.clone();
        let x_title = self.responses_client.config.x_title.clone();
        let http_config = self.responses_client.config.http_config.clone();
        let inspector_config = self.responses_client.config.inspector_config.clone();

        let new_config = OpenRouterConfig {
            api_key: current_api_key.clone(),
            base_url,
            http_referer,
            x_title,
            tool_calling_config: self.responses_client.config.tool_calling_config.clone(),
            http_config,
            inspector_config,
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.responses_client.config.http_referer = Some(http_referer);
        self
    }

    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.responses_client.config.x_title = Some(x_title);
        self
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Result<Self, LlmError> {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let http_referer = self.responses_client.config.http_referer.clone();
        let x_title = self.responses_client.config.x_title.clone();
        let http_config = self.responses_client.config.http_config.clone();
        let inspector_config = self.responses_client.config.inspector_config.clone();

        let new_config = OpenRouterConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            http_referer,
            x_title,
            tool_calling_config: Some(config),
            http_config,
            inspector_config,
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Result<Self, LlmError> {
        let current_config = &self.responses_client.config;
        let new_config = OpenRouterConfig {
            api_key: current_config.api_key.clone(),
            base_url: current_config.base_url.clone(),
            http_referer: current_config.http_referer.clone(),
            x_title: current_config.x_title.clone(),
            tool_calling_config: current_config.tool_calling_config.clone(),
            http_config: config,
            inspector_config: current_config.inspector_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }
}

#[async_trait]
impl LlmProvider for OpenRouterClient {
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

pub fn create_openrouter_client_from_builder<State, Ctx>(
    builder: &LlmBuilder<State, Ctx>,
) -> Result<OpenRouterClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENROUTER_API_KEY not set.".to_string()))?
        .to_string();

    let mut config = OpenRouterConfig::new(api_key);

    if let Some(http_config) = builder.get_http_config() {
        config = config.with_http_config(http_config.clone());
    }

    if let Some(inspector_config) = builder.get_inspector_config() {
        config = config.with_inspector_config(inspector_config.clone());
    }

    let client = ResponsesClient::new(config)?;

    Ok(OpenRouterClient {
        responses_client: client,
    })
}
