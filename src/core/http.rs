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

/// Configuration for HTTP client resilience
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    pub timeout: Duration,
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

    let client = Arc::new(
        reqwest::Client::builder()
            .timeout(key.timeout)
            .user_agent(key.user_agent.as_str())
            .build()
            .map_err(|e| {
                LlmError::ProviderConfiguration(format!("Failed to build reqwest client: {e}"))
            })?,
    );

    clients.insert(key, Arc::clone(&client));
    Ok(client)
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

    fn config_with_timeout(timeout: Duration) -> HttpClientConfig {
        HttpClientConfig {
            timeout,
            ..HttpClientConfig::default()
        }
    }

    #[test]
    fn clients_with_same_timeout_and_user_agent_share_pool_entry() {
        let config = config_with_timeout(Duration::from_secs(7));

        let first = HttpClient::new(config.clone(), Some("rsai-test-shared"), None)
            .expect("first client should build");
        let second = HttpClient::new(config, Some("rsai-test-shared"), None)
            .expect("second client should build");

        assert!(Arc::ptr_eq(&first.client, &second.client));
    }

    #[test]
    fn clients_with_different_timeouts_use_distinct_pool_entries() {
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
        let config = config_with_timeout(Duration::from_secs(13));

        let first = HttpClient::new(config.clone(), Some("rsai-test-agent-a"), None)
            .expect("first client should build");
        let second = HttpClient::new(config, Some("rsai-test-agent-b"), None)
            .expect("second client should build");

        assert!(!Arc::ptr_eq(&first.client, &second.client));
    }
}
