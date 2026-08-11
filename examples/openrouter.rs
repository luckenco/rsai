//! Example demonstrating OpenRouter provider usage with the responses API.
//!
//! This example shows how to:
//! - Use OpenRouter with the responses API
//! - Set up API key from environment variables
//! - Use OpenRouter-specific headers (HTTP-Referer, X-Title)
//! - Generate structured output

use dotenvy::dotenv;

use rsai::{ApiKey, ChatRole, Message, Provider, completion_schema, llm};

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
    key_points: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let analysis = llm::with(Provider::OpenRouter)
        .api_key(ApiKey::Default)?
        .model("openai/gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content:
                "Analyze this text: 'The new AI library is incredibly powerful and easy to use!'"
                    .to_string(),
        }])
        .complete::<Analysis>()
        .await?;

    println!("Sentiment: {}", analysis.content.sentiment);
    println!("Confidence: {}", analysis.content.confidence);
    println!("Key points: {:?}", analysis.content.key_points);

    Ok(())
}
