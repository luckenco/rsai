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
    BaseUrlSecurity, InspectorConfig, LlmBuilder, LlmError, LlmProvider, StructuredRequest,
    ToolCallingConfig, ToolCallingGuard, ToolRegistry,
};
use async_trait::async_trait;

/// OpenRouter-specific configuration for the responses client
pub struct OpenRouterConfig {
    /// API key sent in the authorization header.
    pub api_key: String,
    /// Base URL for Responses API requests.
    pub base_url: String,
    /// Security policy applied to the base URL.
    pub base_url_security: BaseUrlSecurity,
    /// Optional HTTP-Referer header used for OpenRouter attribution.
    pub http_referer: Option<String>,
    /// Optional X-Title header used for OpenRouter attribution.
    pub x_title: Option<String>,
    /// HTTP timeout and retry settings.
    pub http_config: HttpClientConfig,
    /// Configuration for tool calling limits
    pub tool_calling_config: Option<ToolCallingConfig>,
    /// Configuration for request/response inspection
    pub inspector_config: Option<InspectorConfig>,
}

impl OpenRouterConfig {
    /// Create a configuration using OpenRouter's default base URL.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openrouter::API_BASE.to_string(),
            base_url_security: BaseUrlSecurity::HttpsOnly,
            http_referer: None,
            x_title: None,
            tool_calling_config: Some(ToolCallingConfig::default()),
            http_config: HttpClientConfig::default(),
            inspector_config: None,
        }
    }

    /// Set a custom OpenRouter-compatible base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the OpenRouter API key in the `Authorization` header. This
    /// method requires HTTPS; use [`Self::with_insecure_base_url`] only for trusted local or proxy
    /// endpoints that intentionally use HTTP.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self.base_url_security = BaseUrlSecurity::HttpsOnly;
        self
    }

    /// Set a custom OpenRouter-compatible HTTP base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the OpenRouter API key in the `Authorization` header. Use this
    /// only for trusted local or proxy endpoints because the API key may be sent over plaintext
    /// HTTP.
    pub fn with_insecure_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self.base_url_security = BaseUrlSecurity::AllowInsecureHttp;
        self
    }

    /// Set HTTP timeout and retry behavior.
    pub fn with_http_config(mut self, config: HttpClientConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Set raw request and response inspection callbacks.
    pub fn with_inspector_config(mut self, config: InspectorConfig) -> Self {
        self.inspector_config = Some(config);
        self
    }

    /// Set the HTTP-Referer attribution header.
    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.http_referer = Some(http_referer);
        self
    }

    /// Set the X-Title attribution header.
    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.x_title = Some(x_title);
        self
    }

    /// Set limits and timeouts for automatic tool calling.
    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.tool_calling_config = Some(config);
        self
    }

    /// Create a fresh guard from the configured tool-calling limits.
    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::from_config(config)
        } else {
            ToolCallingGuard::new()
        }
    }
}

impl ResponsesProviderConfig for OpenRouterConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn base_url_security(&self) -> BaseUrlSecurity {
        self.base_url_security
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

/// Client for OpenRouter's Responses API.
pub struct OpenRouterClient {
    responses_client: ResponsesClient<OpenRouterConfig>,
}

impl OpenRouterClient {
    /// Create a client using OpenRouter's default base URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = OpenRouterConfig::new(api_key);
        Ok(Self {
            responses_client: ResponsesClient::new(config)?,
        })
    }

    /// Set a custom OpenRouter-compatible base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the OpenRouter API key in the `Authorization` header. This
    /// method requires HTTPS; use [`Self::with_insecure_base_url`] only for trusted local or proxy
    /// endpoints that intentionally use HTTP.
    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        let config = self.responses_client.into_config().with_base_url(base_url);
        self.responses_client = ResponsesClient::new(config)?;
        Ok(self)
    }

    /// Set a custom OpenRouter-compatible HTTP base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the OpenRouter API key in the `Authorization` header. Use this
    /// only for trusted local or proxy endpoints because the API key may be sent over plaintext
    /// HTTP.
    pub fn with_insecure_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        let config = self
            .responses_client
            .into_config()
            .with_insecure_base_url(base_url);
        self.responses_client = ResponsesClient::new(config)?;
        Ok(self)
    }

    /// Set the HTTP-Referer attribution header.
    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.responses_client.config.http_referer = Some(http_referer);
        self
    }

    /// Set the X-Title attribution header.
    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.responses_client.config.x_title = Some(x_title);
        self
    }

    /// Set limits and timeouts for automatic tool calling.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated HTTP client cannot be constructed.
    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Result<Self, LlmError> {
        let config = self
            .responses_client
            .into_config()
            .with_tool_calling_config(config);
        self.responses_client = ResponsesClient::new(config)?;
        Ok(self)
    }

    /// Set HTTP timeout and retry behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated HTTP client cannot be constructed.
    pub fn with_http_config(mut self, config: HttpClientConfig) -> Result<Self, LlmError> {
        let config = self.responses_client.into_config().with_http_config(config);
        self.responses_client = ResponsesClient::new(config)?;
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
        request.validate()?;
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

    if let Some(tool_calling_config) = builder.get_tool_calling_config() {
        config = config.with_tool_calling_config(tool_calling_config.clone());
    }

    if let Some(inspector_config) = builder.get_inspector_config() {
        config = config.with_inspector_config(inspector_config.clone());
    }

    let client = ResponsesClient::new(config)?;

    Ok(OpenRouterClient {
        responses_client: client,
    })
}
