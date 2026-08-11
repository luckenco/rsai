use dotenvy::dotenv;
use rsai::{ApiKey, ChatRole, Message, Provider, completion_schema, llm};

#[completion_schema]
struct CreativeStory {
    title: String,
    genre: String,
    plot_summary: String,
    main_character: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    println!("Generating stories with different generation configs...\n");

    // High temperature for creative writing
    let creative_story = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Write a creative short story concept about a robot discovering emotions"
                .to_string(),
        }])
        .temperature(1.5) // High temperature for creativity
        .max_tokens(500) // Limit the response length
        .complete::<CreativeStory>()
        .await?;

    println!("=== Creative Story (High Temperature: 1.5) ===");
    println!("Title: {}", creative_story.content.title);
    println!("Genre: {}", creative_story.content.genre);
    println!("Main Character: {}", creative_story.content.main_character);
    println!("Plot: {}\n", creative_story.content.plot_summary);

    // Low temperature for factual/deterministic responses
    let factual_story = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Write a story concept about a robot discovering emotions".to_string(),
        }])
        .temperature(0.2) // Low temperature for consistency
        .max_tokens(300) // Shorter response
        .complete::<CreativeStory>()
        .await?;

    println!("=== Factual Story (Low Temperature: 0.2) ===");
    println!("Title: {}", factual_story.content.title);
    println!("Genre: {}", factual_story.content.genre);
    println!("Main Character: {}", factual_story.content.main_character);
    println!("Plot: {}\n", factual_story.content.plot_summary);

    // Using top_p (nucleus sampling) instead of temperature
    let nucleus_story = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Write an experimental story concept about a robot discovering emotions"
                .to_string(),
        }])
        .top_p(0.9) // Nucleus sampling
        .max_tokens(400)
        .complete::<CreativeStory>()
        .await?;

    println!("=== Experimental Story (top_p: 0.9) ===");
    println!("Title: {}", nucleus_story.content.title);
    println!("Genre: {}", nucleus_story.content.genre);
    println!("Main Character: {}", nucleus_story.content.main_character);
    println!("Plot: {}", nucleus_story.content.plot_summary);

    Ok(())
}
