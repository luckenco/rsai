use dotenvy::dotenv;
use rsai::{ApiKey, ChatRole, HttpClientConfig, Message, Provider, completion_schema, llm};
use std::time::Duration;

#[completion_schema]
struct Fact {
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let messages = vec![Message {
        role: ChatRole::User,
        content: "Tell me a random interesting fact about space.".to_string(),
    }];

    // This sets a strict total request timeout.
    // If the model takes > 5 seconds, this will error with LlmError::Network.
    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(messages.clone())
        // Simple convenience method
        .timeout(Duration::from_secs(5))
        .complete::<Fact>()
        .await;

    match response {
        Ok(res) => println!("Success (Simple): {}", res.content.content),
        Err(e) => println!("Error (Simple): {}", e),
    }

    // Define a custom policy for flaky networks or rate-limited environments
    let resilient_config = HttpClientConfig {
        // Total time for a single attempt
        timeout: Duration::from_secs(10),
        // How many times to retry on 429 (Rate Limit) or 5xx (Server Error)
        max_retries: 5,
        // Start waiting 2s, then 4s, then 8s...
        initial_retry_delay: Duration::from_secs(2),
        // ...but don't wait longer than 15s between retries
        max_retry_delay: Duration::from_secs(15),
    };

    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(messages)
        // Inject the custom configuration
        .http_client_config(resilient_config)
        .complete::<Fact>()
        .await;

    match response {
        Ok(res) => println!("Success (Advanced): {}", res.content.content),
        Err(e) => println!("Error (Advanced): {}", e),
    }

    Ok(())
}
