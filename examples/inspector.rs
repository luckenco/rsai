/// This example demonstrates the request/response inspection hooks for debugging.
///
/// The inspector callbacks receive raw JSON (`serde_json::Value`) for both
/// requests sent to the API and responses received back. This is useful for:
///  • Debugging generated schemas and prompts
///  • Logging API interactions
///  • Monitoring tool-calling loop iterations
///
/// Run with: cargo run --example inspector
use dotenvy::dotenv;
use rsai::{ApiKey, ChatRole, Message, Provider, TextResponse, llm, tool, toolset};

#[tool]
/// Get current weather for a city
/// city: The city to get weather for
fn get_weather(city: String) -> String {
    format!("Weather in {}: 22°C, sunny", city)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let tools = toolset![get_weather];

    let messages = vec![
        Message {
            role: ChatRole::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: ChatRole::User,
            content: "What's the weather in Paris?".to_string(),
        },
    ];

    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(messages)
        .tools(tools)
        .inspect_request(|req| {
            println!("━━━ REQUEST ━━━");
            println!("{}", serde_json::to_string_pretty(req).unwrap());
            println!();
        })
        .inspect_response(|res| {
            println!("━━━ RESPONSE ━━━");
            println!("{}", serde_json::to_string_pretty(res).unwrap());
            println!();
        })
        .complete::<TextResponse>()
        .await?;

    println!("━━━ FINAL RESULT ━━━");
    println!("{}", response.text);

    Ok(())
}
