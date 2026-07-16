//! Google Gemini provider implementation.
//!
//! This module implements the Gemini API using the completions abstraction layer.
//! It supports text generation, structured output, and function calling.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completions::{
    CompletionClient, CompletionProviderConfig, CompletionRequestBuilder, ConversationItem,
};
use crate::core::{
    BaseUrlSecurity, FunctionCallData, HttpClientConfig, InspectorConfig, LanguageModelUsage,
    LlmBuilder, LlmError, LlmProvider, ProviderResponse, ResponseContent, StructuredRequest,
    ToolCallingConfig, ToolCallingGuard, ToolRegistry,
};
use crate::provider::constants::gemini;
use crate::responses::{Format, request::FormatType};

// ============================================================================
// Gemini API Request Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GeminiToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    Text(TextPart),
    FunctionCall(FunctionCallPart),
    FunctionResponse(FunctionResponsePart),
}

impl Part {
    pub fn text(s: String) -> Self {
        Self::Text(TextPart { text: s })
    }

    pub fn function_call(name: String, args: Value) -> Self {
        Self::FunctionCall(FunctionCallPart {
            function_call: FunctionCall { name, args },
        })
    }

    pub fn function_response(name: String, response: Value) -> Self {
        Self::FunctionResponse(FunctionResponsePart {
            function_response: FunctionResponse { name, response },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallPart {
    pub function_call: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponsePart {
    pub function_response: FunctionResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "parametersJsonSchema")]
    pub parameters_json_schema: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

// ============================================================================
// Gemini API Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Option<Vec<Candidate>>,
    pub usage_metadata: Option<UsageMetadata>,
    #[allow(dead_code)]
    pub model_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<Content>,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
    #[allow(dead_code)]
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: Option<i32>,
    pub candidates_token_count: Option<i32>,
    pub total_token_count: Option<i32>,
}

// ============================================================================
// Gemini Configuration
// ============================================================================

/// Gemini-specific configuration for generate-content requests.
pub struct GeminiConfig {
    /// API key sent in the x-goog-api-key header.
    pub api_key: String,
    /// Base URL for generate-content requests.
    pub base_url: String,
    /// Security policy applied to the base URL.
    pub base_url_security: BaseUrlSecurity,
    /// Limits and timeouts for automatic tool calling.
    pub tool_calling_config: Option<ToolCallingConfig>,
    /// HTTP timeout and retry settings.
    pub http_config: HttpClientConfig,
    /// Configuration for request/response inspection
    pub inspector_config: Option<InspectorConfig>,
}

impl GeminiConfig {
    /// Create a configuration using Gemini's default base URL.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: gemini::API_BASE.to_string(),
            base_url_security: BaseUrlSecurity::HttpsOnly,
            tool_calling_config: Some(ToolCallingConfig::default()),
            http_config: HttpClientConfig::default(),
            inspector_config: None,
        }
    }

    /// Set a custom Gemini-compatible base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the Gemini API key in the `x-goog-api-key` header. This method
    /// requires HTTPS; use [`Self::with_insecure_base_url`] only for trusted local or proxy
    /// endpoints that intentionally use HTTP.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self.base_url_security = BaseUrlSecurity::HttpsOnly;
        self
    }

    /// Set a custom Gemini-compatible HTTP base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the Gemini API key in the `x-goog-api-key` header. Use this
    /// only for trusted local or proxy endpoints because the API key may be sent over plaintext
    /// HTTP.
    pub fn with_insecure_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self.base_url_security = BaseUrlSecurity::AllowInsecureHttp;
        self
    }

    /// Set limits and timeouts for automatic tool calling.
    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.tool_calling_config = Some(config);
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

    /// Create a fresh guard from the configured tool-calling limits.
    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::from_config(config)
        } else {
            ToolCallingGuard::new()
        }
    }
}

impl CompletionProviderConfig for GeminiConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn base_url_security(&self) -> BaseUrlSecurity {
        self.base_url_security
    }

    fn auth_header(&self) -> (String, String) {
        ("x-goog-api-key".to_string(), self.api_key.clone())
    }

    fn http_config(&self) -> HttpClientConfig {
        self.http_config.clone()
    }

    fn inspector_config(&self) -> Option<&InspectorConfig> {
        self.inspector_config.as_ref()
    }
}

// ============================================================================
// Request Builder Implementation
// ============================================================================

pub struct GeminiRequestBuilder;

impl CompletionRequestBuilder for GeminiRequestBuilder {
    type Request = GeminiRequest;
    type Response = GeminiResponse;

    fn build_request(
        &self,
        request: &StructuredRequest,
        format: &Format,
        conversation: &[ConversationItem],
    ) -> Result<Self::Request, LlmError> {
        let (system_instruction, contents) = build_contents_from_conversation(conversation)?;

        let generation_config = build_generation_config(request, format);
        let (tools, tool_config) = build_tools_config(request);

        // Gemini doesn't support combining function calling with structured JSON output
        if tools.is_some() && matches!(format.format, FormatType::JsonSchema(_)) {
            return Err(LlmError::ProviderConfiguration(
                "Gemini does not support combining function calling with structured JSON output. \
                 Use TextResponse with tools, or structured output without tools."
                    .to_string(),
            ));
        }

        Ok(GeminiRequest {
            contents,
            generation_config,
            system_instruction,
            tools,
            tool_config,
        })
    }

    fn parse_response(&self, response: Self::Response) -> Result<ProviderResponse, LlmError> {
        let candidate = response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .ok_or_else(|| LlmError::Provider {
                message: "No candidates in Gemini response".to_string(),
                source: None,
            })?;

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| LlmError::Provider {
                message: "No content in Gemini candidate".to_string(),
                source: None,
            })?;

        let response_content = parse_parts_to_content(&content.parts)?;

        let usage = response
            .usage_metadata
            .map(|u| LanguageModelUsage {
                prompt_tokens: u.prompt_token_count.unwrap_or(0),
                completion_tokens: u.candidates_token_count.unwrap_or(0),
                total_tokens: u.total_token_count.unwrap_or(0),
            })
            .unwrap_or(LanguageModelUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });

        Ok(ProviderResponse {
            id: String::new(), // Gemini doesn't return an ID
            model: response.model_version.unwrap_or_default(),
            provider: super::Provider::Gemini,
            content: response_content,
            usage,
        })
    }

    fn endpoint(&self, model: &str) -> String {
        format!("/models/{}:generateContent", model)
    }

    fn extract_function_calls(&self, response: &Self::Response) -> Option<Vec<FunctionCallData>> {
        let candidate = response.candidates.as_ref()?.first()?;
        let content = candidate.content.as_ref()?;

        let calls: Vec<FunctionCallData> = extract_function_calls_from_parts(&content.parts);

        if calls.is_empty() { None } else { Some(calls) }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_contents_from_conversation(
    conversation: &[ConversationItem],
) -> Result<(Option<Content>, Vec<Content>), LlmError> {
    let mut system_instruction: Option<Content> = None;
    let mut contents: Vec<Content> = Vec::new();
    let mut pending_role: Option<String> = None;
    let mut pending_parts: Vec<Part> = Vec::new();

    fn flush_pending_function_parts(
        contents: &mut Vec<Content>,
        pending_role: &mut Option<String>,
        pending_parts: &mut Vec<Part>,
    ) {
        let Some(role) = pending_role.take() else {
            return;
        };
        if pending_parts.is_empty() {
            return;
        }

        contents.push(Content {
            role: Some(role),
            parts: std::mem::take(pending_parts),
        });
    }

    fn push_grouped_function_part(
        contents: &mut Vec<Content>,
        pending_role: &mut Option<String>,
        pending_parts: &mut Vec<Part>,
        role: &str,
        part: Part,
    ) {
        if pending_role.as_deref() != Some(role) {
            flush_pending_function_parts(contents, pending_role, pending_parts);
            *pending_role = Some(role.to_string());
        }
        pending_parts.push(part);
    }

    for item in conversation {
        match item {
            ConversationItem::Message { role, content } => {
                flush_pending_function_parts(&mut contents, &mut pending_role, &mut pending_parts);

                if role == "system" {
                    let instruction = system_instruction.get_or_insert_with(|| Content {
                        role: None,
                        parts: Vec::new(),
                    });
                    instruction.parts.push(Part::text(content.clone()));
                } else {
                    let gemini_role = match role.as_str() {
                        "user" => "user",
                        "assistant" => "model",
                        _ => "user",
                    };
                    contents.push(Content {
                        role: Some(gemini_role.to_string()),
                        parts: vec![Part::text(content.clone())],
                    });
                }
            }
            ConversationItem::FunctionCall {
                name, arguments, ..
            } => {
                push_grouped_function_part(
                    &mut contents,
                    &mut pending_role,
                    &mut pending_parts,
                    "model",
                    Part::function_call(name.clone(), arguments.clone()),
                );
            }
            ConversationItem::FunctionResult { call_id, result } => {
                // For Gemini, we need to find the function name from previous calls
                let name = find_function_name_by_call_id(conversation, call_id)
                    .unwrap_or_else(|| call_id.clone());

                // Gemini requires function_response.response to be a Struct (object).
                // Wrap non-object values in a result wrapper.
                let response_value = match result {
                    Value::Object(_) => result.clone(),
                    other => serde_json::json!({ "result": other }),
                };

                push_grouped_function_part(
                    &mut contents,
                    &mut pending_role,
                    &mut pending_parts,
                    "user",
                    Part::function_response(name, response_value),
                );
            }
        }
    }

    flush_pending_function_parts(&mut contents, &mut pending_role, &mut pending_parts);

    Ok((system_instruction, contents))
}

fn find_function_name_by_call_id(
    conversation: &[ConversationItem],
    call_id: &str,
) -> Option<String> {
    for item in conversation {
        if let ConversationItem::FunctionCall { id, name, .. } = item
            && id == call_id
        {
            return Some(name.clone());
        }
    }
    None
}

fn build_generation_config(
    request: &StructuredRequest,
    format: &Format,
) -> Option<GeminiGenerationConfig> {
    let gen_config = request.generation_config.as_ref();

    let (response_mime_type, response_json_schema) = match &format.format {
        FormatType::JsonSchema(json_schema) => (
            Some("application/json".to_string()),
            Some(json_schema.schema.clone()),
        ),
        FormatType::Text { .. } => (None, None),
    };

    // Only create config if there's something to configure
    if gen_config.is_none() && response_mime_type.is_none() {
        return None;
    }

    Some(GeminiGenerationConfig {
        temperature: gen_config.and_then(|c| c.temperature),
        top_p: gen_config.and_then(|c| c.top_p),
        top_k: None,
        max_output_tokens: gen_config.and_then(|c| c.max_tokens),
        response_mime_type,
        response_json_schema,
    })
}

fn build_tools_config(
    request: &StructuredRequest,
) -> (Option<Vec<GeminiTool>>, Option<GeminiToolConfig>) {
    let Some(tool_config) = request.tool_config.as_ref() else {
        return (None, None);
    };
    let Some(tools) = tool_config.tools.as_ref() else {
        return (None, None);
    };

    if tools.is_empty() {
        return (None, None);
    }

    let function_declarations: Vec<GeminiFunctionDeclaration> = tools
        .iter()
        .map(|t| GeminiFunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters_json_schema: t.parameters.clone(),
        })
        .collect();

    let gemini_tools = vec![GeminiTool {
        function_declarations,
    }];

    // Gemini exposes no native parallel_tool_calls request flag; the shared
    // completion loop enforces local sequential or bounded-concurrent execution.
    let mode = match &tool_config.tool_choice {
        Some(crate::core::ToolChoice::None) => "NONE",
        Some(crate::core::ToolChoice::Auto) => "AUTO",
        Some(crate::core::ToolChoice::Required) => "ANY",
        Some(crate::core::ToolChoice::Function { name }) => {
            // For specific function, we use ANY with allowed_function_names
            return (
                Some(gemini_tools),
                Some(GeminiToolConfig {
                    function_calling_config: FunctionCallingConfig {
                        mode: "ANY".to_string(),
                        allowed_function_names: Some(vec![name.clone()]),
                    },
                }),
            );
        }
        None => "AUTO",
    };

    (
        Some(gemini_tools),
        Some(GeminiToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: mode.to_string(),
                allowed_function_names: None,
            },
        }),
    )
}

/// Extract function calls from Gemini response parts with unique IDs.
fn extract_function_calls_from_parts(parts: &[Part]) -> Vec<FunctionCallData> {
    parts
        .iter()
        .enumerate()
        .filter_map(|(idx, part)| {
            if let Part::FunctionCall(FunctionCallPart { function_call }) = part {
                Some(FunctionCallData {
                    id: format!("call_{}_{:08x}", idx, rand::random::<u32>()),
                    name: function_call.name.clone(),
                    arguments: function_call.args.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn parse_parts_to_content(parts: &[Part]) -> Result<ResponseContent, LlmError> {
    let function_calls = extract_function_calls_from_parts(parts);
    if !function_calls.is_empty() {
        return Ok(ResponseContent::FunctionCalls(function_calls));
    }

    let text: String = parts
        .iter()
        .filter_map(|p| match p {
            Part::Text(TextPart { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

    if !text.is_empty() {
        Ok(ResponseContent::Text(text))
    } else {
        Err(LlmError::Provider {
            message: "Empty response from Gemini".to_string(),
            source: None,
        })
    }
}

// ============================================================================
// Gemini Client
// ============================================================================

/// Client for Google's Gemini generate-content API.
pub struct GeminiClient {
    completion_client: CompletionClient<GeminiConfig>,
}

impl GeminiClient {
    /// Create a client using Gemini's default base URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = GeminiConfig::new(api_key);
        Ok(Self {
            completion_client: CompletionClient::new(config)?,
        })
    }

    /// Set a custom Gemini-compatible base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the Gemini API key in the `x-goog-api-key` header. This method
    /// requires HTTPS; use [`Self::with_insecure_base_url`] only for trusted local or proxy
    /// endpoints that intentionally use HTTP.
    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        let config = self.completion_client.into_config().with_base_url(base_url);
        self.completion_client = CompletionClient::new(config)?;
        Ok(self)
    }

    /// Set a custom Gemini-compatible HTTP base URL.
    ///
    /// # Security
    ///
    /// Requests to this URL include the Gemini API key in the `x-goog-api-key` header. Use this
    /// only for trusted local or proxy endpoints because the API key may be sent over plaintext
    /// HTTP.
    pub fn with_insecure_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        let config = self
            .completion_client
            .into_config()
            .with_insecure_base_url(base_url);
        self.completion_client = CompletionClient::new(config)?;
        Ok(self)
    }

    /// Set limits and timeouts for automatic tool calling.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated HTTP client cannot be constructed.
    pub fn with_tool_calling_config(
        mut self,
        tool_config: ToolCallingConfig,
    ) -> Result<Self, LlmError> {
        let config = self
            .completion_client
            .into_config()
            .with_tool_calling_config(tool_config);
        self.completion_client = CompletionClient::new(config)?;
        Ok(self)
    }

    /// Set HTTP timeout and retry behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated HTTP client cannot be constructed.
    pub fn with_http_config(mut self, http_config: HttpClientConfig) -> Result<Self, LlmError> {
        let config = self
            .completion_client
            .into_config()
            .with_http_config(http_config);
        self.completion_client = CompletionClient::new(config)?;
        Ok(self)
    }

    /// Set raw request and response inspection callbacks.
    ///
    /// # Errors
    ///
    /// Returns an error if the updated HTTP client cannot be constructed.
    pub fn with_inspector_config(
        mut self,
        inspector_config: InspectorConfig,
    ) -> Result<Self, LlmError> {
        let config = self
            .completion_client
            .into_config()
            .with_inspector_config(inspector_config);
        self.completion_client = CompletionClient::new(config)?;
        Ok(self)
    }
}

#[async_trait]
impl LlmProvider for GeminiClient {
    async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
    ) -> Result<T::Output, LlmError>
    where
        T: crate::CompletionTarget + Send,
        Ctx: Send + Sync + 'static,
    {
        request.validate()?;
        let builder = GeminiRequestBuilder;

        // If tools are present and we have a registry, handle automatic tool calling
        let has_tools = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.tools.as_ref())
            .is_some();

        if has_tools && let Some(tool_registry) = tool_registry {
            let mut guard = self.completion_client.config.get_tool_calling_guard();
            let provider_response = self
                .completion_client
                .handle_tool_calling_loop::<_, Ctx>(
                    &builder,
                    request,
                    tool_registry,
                    &mut guard,
                    format,
                )
                .await?;
            return T::parse_response(provider_response);
        }

        // Single request without tool calling loop
        let conversation =
            crate::completions::client::convert_messages_to_conversation(&request.messages)?;
        let api_request = builder.build_request(&request, &format, &conversation)?;
        let api_response = self
            .completion_client
            .make_api_request(&builder, api_request, &request.model)
            .await?;
        let provider_response = builder.parse_response(api_response)?;
        T::parse_response(provider_response)
    }
}

// ============================================================================
// Builder Integration
// ============================================================================

pub fn create_gemini_client_from_builder<State, Ctx>(
    builder: &LlmBuilder<State, Ctx>,
) -> Result<GeminiClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("GEMINI_API_KEY not set.".to_string()))?
        .to_string();

    let mut config = GeminiConfig::new(api_key);

    if let Some(http_config) = builder.get_http_config() {
        config = config.with_http_config(http_config.clone());
    }

    if let Some(tool_calling_config) = builder.get_tool_calling_config() {
        config = config.with_tool_calling_config(tool_calling_config.clone());
    }

    if let Some(inspector_config) = builder.get_inspector_config() {
        config = config.with_inspector_config(inspector_config.clone());
    }

    Ok(GeminiClient {
        completion_client: CompletionClient::new(config)?,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::core::{Tool as CoreTool, ToolConfig};
    use crate::responses::{JsonSchema, JsonSchemaType};

    fn request(tool_config: Option<ToolConfig>) -> StructuredRequest {
        StructuredRequest {
            model: "test-model".to_string(),
            messages: Vec::new(),
            tool_config,
            generation_config: None,
        }
    }

    fn schema() -> Value {
        json!({
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": { "city": { "type": "string", "minLength": 1 } },
                    "required": ["city"]
                }
            },
            "type": "object",
            "properties": {
                "address": { "$ref": "#/$defs/Address" },
                "label": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["address"]
        })
    }

    #[test]
    fn structured_response_uses_json_schema_without_conversion() {
        let expected_schema = schema();
        let format = Format {
            format: FormatType::JsonSchema(JsonSchema {
                name: "test".to_string(),
                schema: expected_schema.clone(),
                r#type: JsonSchemaType::JsonSchema,
            }),
        };

        let config = build_generation_config(&request(None), &format).expect("generation config");
        let serialized = serde_json::to_value(config).expect("serialize generation config");

        assert_eq!(serialized["responseJsonSchema"], expected_schema);
        assert!(serialized.get("responseSchema").is_none());
    }

    #[test]
    fn tool_parameters_use_json_schema_without_conversion() {
        let expected_schema = schema();
        let tool_config = ToolConfig {
            tools: Some(
                vec![CoreTool {
                    name: "lookup".to_string(),
                    description: Some("Look up an address".to_string()),
                    parameters: expected_schema.clone(),
                    strict: Some(true),
                }]
                .into_boxed_slice(),
            ),
            tool_choice: None,
            parallel_tool_calls: None,
        };

        let (tools, _) = build_tools_config(&request(Some(tool_config)));
        let serialized =
            serde_json::to_value(tools.expect("Gemini tools")).expect("serialize Gemini tools");

        assert_eq!(
            serialized[0]["functionDeclarations"][0]["parametersJsonSchema"],
            expected_schema
        );
        assert!(
            serialized[0]["functionDeclarations"][0]
                .get("parameters")
                .is_none()
        );
    }

    #[test]
    fn system_messages_are_preserved_in_order() {
        let conversation = vec![
            ConversationItem::Message {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ConversationItem::Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ConversationItem::Message {
                role: "system".to_string(),
                content: "Use metric units.".to_string(),
            },
        ];

        let (system_instruction, contents) =
            build_contents_from_conversation(&conversation).expect("Gemini content");
        let serialized = serde_json::to_value(system_instruction.expect("system instruction"))
            .expect("serialize system instruction");

        assert_eq!(
            serialized["parts"],
            json!([
                { "text": "Be concise." },
                { "text": "Use metric units." }
            ])
        );
        assert_eq!(contents.len(), 1);
    }
}
