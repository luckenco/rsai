//! Example demonstrating Google Gemini provider with function calling.

use dotenvy::dotenv;

use rsai::{ApiKey, ChatRole, Message, Provider, TextResponse, llm, tool, toolset};

#[tool]
/// Calculate the tip for a restaurant bill
/// amount: The bill amount in dollars
/// service_quality: Rating from 1-5 stars
fn calculate_tip(amount: f64, service_quality: u32) -> String {
    let percentage = match service_quality {
        1 => 10,
        2 => 15,
        3 => 18,
        4 => 20,
        5 => 25,
        _ => 18,
    };
    let tip = amount * (percentage as f64 / 100.0);
    format!("{}% tip on ${:.2} = ${:.2}", percentage, amount, tip)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let tools = toolset![calculate_tip];

    let result = llm::with(Provider::Gemini)
        .api_key(ApiKey::Default)?
        .model("gemini-2.5-flash")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "My dinner bill is $85 and the service was excellent. How much should I tip?"
                .to_string(),
        }])
        .tools(tools)
        .complete::<TextResponse>()
        .await?;

    println!("{}", result.text);

    Ok(())
}
