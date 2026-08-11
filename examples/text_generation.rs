use dotenvy::dotenv;
use rsai::{ApiKey, ChatRole, Message, Provider, TextResponse, llm};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![
            Message {
                role: ChatRole::System,
                content: "You are a concise, upbeat assistant.".to_string(),
            },
            Message {
                role: ChatRole::User,
                content: "Share a fun fact about Rust programming.".to_string(),
            },
        ])
        .complete::<TextResponse>()
        .await?;

    println!("Assistant:\n{}", response.text);

    Ok(())
}
