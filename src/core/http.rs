//! Shared HTTP client with retry logic for all providers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use serde::{Serialize, de::DeserializeOwned};
use tracing::{debug, warn};

use super::builder::InspectorConfig;
use super::error::LlmError;

/// Security policy for provider base URLs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BaseUrlSecurity {
    /// Require custom provider base URLs to use HTTPS.
    #[default]
    HttpsOnly,
    /// Allow HTTP base URLs for trusted local or proxy endpoints.
    AllowInsecureHttp,
}

/// Configuration for HTTP client resilience
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Timeout applied to each HTTP request.
    pub timeout: Duration,
    /// Maximum number of retries after the initial request.
    pub max_retries: u32,
    /// Base duration for exponential backoff
    pub initial_retry_delay: Duration,
    /// Cap on the backoff duration
    pub max_retry_delay: Duration,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            max_retries: 3,
            initial_retry_delay: Duration::from_millis(500),
            max_retry_delay: Duration::from_secs(10),
        }
    }
}

pub(crate) fn validate_provider_base_url(
    base_url: &str,
    security: BaseUrlSecurity,
) -> Result<(), LlmError> {
    let url = reqwest::Url::parse(base_url).map_err(|e| {
        LlmError::ProviderConfiguration(format!("Invalid provider base URL '{base_url}': {e}"))
    })?;

    match url.scheme() {
        "https" => {}
        "http" if security == BaseUrlSecurity::AllowInsecureHttp => {}
        "http" => {
            return Err(LlmError::ProviderConfiguration(
                "Insecure provider base URL rejected: custom base URLs receive provider API keys. \
                 Use HTTPS or call with_insecure_base_url for a trusted local/proxy endpoint."
                    .to_string(),
            ));
        }
        scheme => {
            return Err(LlmError::ProviderConfiguration(format!(
                "Invalid provider base URL scheme '{scheme}': expected https"
            )));
        }
    }

    if url.host_str().is_none() {
        return Err(LlmError::ProviderConfiguration(
            "Invalid provider base URL: expected an absolute URL with a host".to_string(),
        ));
    }

    Ok(())
}

/// Shared HTTP client with retry logic and exponential backoff.
pub struct HttpClient {
    client: Arc<reqwest::Client>,
    config: HttpClientConfig,
    inspector_config: Option<InspectorConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ClientPoolKey {
    timeout: Duration,
    user_agent: String,
}

const MAX_CLIENT_POOL_ENTRIES: usize = 32;

static CLIENT_POOL: OnceLock<Mutex<HashMap<ClientPoolKey, Arc<reqwest::Client>>>> = OnceLock::new();

fn client_pool() -> &'static Mutex<HashMap<ClientPoolKey, Arc<reqwest::Client>>> {
    CLIENT_POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pooled_reqwest_client(
    timeout: Duration,
    user_agent: String,
) -> Result<Arc<reqwest::Client>, LlmError> {
    let key = ClientPoolKey {
        timeout,
        user_agent,
    };

    let mut clients = client_pool().lock().map_err(|_| {
        LlmError::ProviderConfiguration("Failed to access HTTP client pool".to_string())
    })?;

    if let Some(client) = clients.get(&key) {
        return Ok(Arc::clone(client));
    }

    let client = build_reqwest_client(key.timeout, &key.user_agent)?;

    if clients.len() < MAX_CLIENT_POOL_ENTRIES {
        clients.insert(key, Arc::clone(&client));
    }

    Ok(client)
}

fn build_reqwest_client(
    timeout: Duration,
    user_agent: &str,
) -> Result<Arc<reqwest::Client>, LlmError> {
    Ok(Arc::new(
        reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(user_agent)
            .build()
            .map_err(|e| {
                LlmError::ProviderConfiguration(format!("Failed to build reqwest client: {e}"))
            })?,
    ))
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration.
    pub fn new(
        config: HttpClientConfig,
        user_agent: Option<&str>,
        inspector_config: Option<InspectorConfig>,
    ) -> Result<Self, LlmError> {
        let user_agent = user_agent.map_or_else(
            || format!("rsai/{}", env!("CARGO_PKG_VERSION")),
            str::to_owned,
        );
        let client = pooled_reqwest_client(config.timeout, user_agent)?;

        Ok(Self {
            client,
            config,
            inspector_config,
        })
    }

    /// Make a POST request with JSON body and retry logic.
    ///
    /// Retries on 429 (rate limit) and 5xx errors with exponential backoff.
    /// Fails immediately on 4xx errors (except 429).
    #[tracing::instrument(
        name = "http_post_json",
        skip(self, headers, body),
        fields(url = %url),
        err
    )]
    pub async fn post_json<Req, Res>(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Req,
    ) -> Result<Res, LlmError>
    where
        Req: Serialize,
        Res: DeserializeOwned,
    {
        // Only serialize to Value if we need to inspect the request
        if let Some(ref config) = self.inspector_config
            && let Some(ref inspector) = config.request_inspector
        {
            let body_value = serde_json::to_value(body).map_err(|e| LlmError::Parse {
                message: "Failed to serialize request for inspection".to_string(),
                source: Box::new(e),
            })?;
            inspector(&body_value);
        }

        let mut last_error: Option<LlmError> = None;

        for attempt in 0..=self.config.max_retries {
            // Build request (must be rebuilt each attempt since .send() consumes it)
            let mut req_builder = self.client.post(url).json(body);

            // Add headers
            for (name, value) in headers {
                req_builder = req_builder.header(name, value);
            }

            match req_builder.send().await {
                Err(e) => {
                    warn!(attempt, error = %e, "HTTP request failed, retrying");
                    last_error = Some(LlmError::Network {
                        message: format!(
                            "Request failed (attempt {}/{})",
                            attempt + 1,
                            self.config.max_retries + 1
                        ),
                        source: Box::new(e),
                    });
                }
                Ok(res) => {
                    let status = res.status();

                    // Success
                    if status.is_success() {
                        debug!(status = %status, "HTTP request successful");

                        let response_text = res.text().await.map_err(|e| LlmError::Parse {
                            message: "Failed to read response body".to_string(),
                            source: Box::new(e),
                        })?;

                        // Only go through intermediate Value if we need to inspect
                        if let Some(ref config) = self.inspector_config
                            && let Some(ref inspector) = config.response_inspector
                        {
                            let response_value: serde_json::Value =
                                serde_json::from_str(&response_text).map_err(|e| {
                                    LlmError::Parse {
                                        message: "Failed to parse response as JSON".to_string(),
                                        source: Box::new(e),
                                    }
                                })?;
                            inspector(&response_value);
                            return serde_json::from_value(response_value).map_err(|e| {
                                LlmError::Parse {
                                    message: "Failed to parse API response".to_string(),
                                    source: Box::new(e),
                                }
                            });
                        }

                        return serde_json::from_str(&response_text).map_err(|e| LlmError::Parse {
                            message: "Failed to parse API response".to_string(),
                            source: Box::new(e),
                        });
                    }

                    warn!(attempt, status = %status, "API returned error status");

                    let is_retryable = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error();
                    let error_text = res
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());

                    // Call response inspector for error responses
                    if let Some(ref config) = self.inspector_config
                        && let Some(ref inspector) = config.response_inspector
                    {
                        // Try to parse error as JSON, otherwise wrap in object
                        let error_value = serde_json::from_str(&error_text).unwrap_or_else(|_| {
                            serde_json::json!({
                                "error": error_text,
                                "status_code": status.as_u16()
                            })
                        });
                        inspector(&error_value);
                    }

                    if !is_retryable {
                        // Fatal errors - don't retry
                        return Err(LlmError::Api {
                            message: format!("Fatal API Error: {error_text}"),
                            status_code: Some(status.as_u16()),
                            source: None,
                        });
                    }

                    // Retryable error - capture and continue
                    last_error = Some(LlmError::Api {
                        message: format!("Transient API error ({}): {}", status, error_text),
                        status_code: Some(status.as_u16()),
                        source: None,
                    });
                }
            }

            // Exponential backoff with jitter
            if attempt < self.config.max_retries {
                let base_delay =
                    self.config.initial_retry_delay.as_millis() as f64 * 2_f64.powi(attempt as i32);

                // +/- 10% jitter (0.9 to 1.1)
                let jitter_factor = rand::random::<f64>() * 0.2 + 0.9;
                let delay_ms = (base_delay * jitter_factor) as u64;

                // Cap delay at max
                let delay =
                    std::time::Duration::from_millis(delay_ms).min(self.config.max_retry_delay);

                tokio::time::sleep(delay).await;
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Api {
            message: format!(
                "Request failed after max retries ({}) with unknown error",
                self.config.max_retries
            ),
            status_code: None,
            source: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static CLIENT_POOL_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_client_pool_for_test() -> std::sync::MutexGuard<'static, ()> {
        CLIENT_POOL_TEST_LOCK.lock().expect("client pool test lock")
    }

    fn clear_client_pool() {
        client_pool().lock().expect("client pool lock").clear();
    }

    fn client_pool_len() -> usize {
        client_pool().lock().expect("client pool lock").len()
    }

    fn config_with_timeout(timeout: Duration) -> HttpClientConfig {
        HttpClientConfig {
            timeout,
            ..HttpClientConfig::default()
        }
    }

    #[test]
    fn clients_with_same_timeout_and_user_agent_share_pool_entry() {
        let _guard = lock_client_pool_for_test();
        clear_client_pool();
        let config = config_with_timeout(Duration::from_secs(7));

        let first = HttpClient::new(config.clone(), Some("rsai-test-shared"), None)
            .expect("first client should build");
        let second = HttpClient::new(config, Some("rsai-test-shared"), None)
            .expect("second client should build");

        assert!(Arc::ptr_eq(&first.client, &second.client));
    }

    #[test]
    fn clients_with_different_timeouts_use_distinct_pool_entries() {
        let _guard = lock_client_pool_for_test();
        clear_client_pool();
        let first = HttpClient::new(
            config_with_timeout(Duration::from_secs(11)),
            Some("rsai-test-timeout"),
            None,
        )
        .expect("first client should build");
        let second = HttpClient::new(
            config_with_timeout(Duration::from_secs(12)),
            Some("rsai-test-timeout"),
            None,
        )
        .expect("second client should build");

        assert!(!Arc::ptr_eq(&first.client, &second.client));
    }

    #[test]
    fn clients_with_different_user_agents_use_distinct_pool_entries() {
        let _guard = lock_client_pool_for_test();
        clear_client_pool();
        let config = config_with_timeout(Duration::from_secs(13));

        let first = HttpClient::new(config.clone(), Some("rsai-test-agent-a"), None)
            .expect("first client should build");
        let second = HttpClient::new(config, Some("rsai-test-agent-b"), None)
            .expect("second client should build");

        assert!(!Arc::ptr_eq(&first.client, &second.client));
    }

    #[test]
    fn client_pool_does_not_grow_past_cap() {
        let _guard = lock_client_pool_for_test();
        clear_client_pool();

        for index in 0..MAX_CLIENT_POOL_ENTRIES {
            HttpClient::new(
                config_with_timeout(Duration::from_secs(index as u64 + 1)),
                Some("rsai-test-capped"),
                None,
            )
            .expect("pooled client should build");
        }

        assert_eq!(client_pool_len(), MAX_CLIENT_POOL_ENTRIES);

        let overflow = HttpClient::new(
            config_with_timeout(Duration::from_secs(10_000)),
            Some("rsai-test-capped-overflow"),
            None,
        )
        .expect("overflow client should build");

        assert_eq!(client_pool_len(), MAX_CLIENT_POOL_ENTRIES);

        let second_overflow = HttpClient::new(
            config_with_timeout(Duration::from_secs(10_000)),
            Some("rsai-test-capped-overflow"),
            None,
        )
        .expect("second overflow client should build");

        assert!(!Arc::ptr_eq(&overflow.client, &second_overflow.client));
        assert_eq!(client_pool_len(), MAX_CLIENT_POOL_ENTRIES);
    }

    #[test]
    fn provider_base_url_validation_allows_https() {
        validate_provider_base_url("https://api.openai.com/v1", BaseUrlSecurity::HttpsOnly)
            .expect("https base URL should be accepted");
    }

    #[test]
    fn provider_base_url_validation_rejects_invalid_url() {
        assert!(validate_provider_base_url("not a url", BaseUrlSecurity::HttpsOnly).is_err());
    }

    #[test]
    fn provider_base_url_validation_rejects_unsupported_scheme() {
        assert!(
            validate_provider_base_url("ftp://example.com/v1", BaseUrlSecurity::HttpsOnly).is_err()
        );
    }

    #[test]
    fn provider_base_url_validation_rejects_http_by_default() {
        assert!(
            validate_provider_base_url("http://localhost:8080/v1", BaseUrlSecurity::HttpsOnly)
                .is_err()
        );
    }

    #[test]
    fn provider_base_url_validation_allows_http_with_explicit_opt_out() {
        validate_provider_base_url(
            "http://localhost:8080/v1",
            BaseUrlSecurity::AllowInsecureHttp,
        )
        .expect("explicit insecure opt-out should accept http");
    }
}
