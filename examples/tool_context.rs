//! Dependency injection for tools via context.
//!
//! Tools can receive external resources (DB connections, HTTP clients, config)
//! through a shared context instead of using global state.
//!
//! Run with: `cargo run --example tool-context`

use rsai::{ApiKey, ChatRole, Ctx, Message, Provider, TextResponse, llm, tool, toolset};

// A mock database that tools can access
struct Database {
    data: Vec<(&'static str, &'static str)>,
}

impl Database {
    fn new() -> Self {
        Self {
            data: vec![
                ("rust", "A systems programming language focused on safety."),
                ("tokio", "An async runtime for Rust."),
                ("serde", "A serialization framework for Rust."),
            ],
        }
    }

    fn search(&self, query: &str) -> Vec<String> {
        self.data
            .iter()
            .filter(|(k, _)| k.contains(&query.to_lowercase()))
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect()
    }
}

// Context struct that holds all dependencies
struct AppContext {
    db: Database,
}

// Tools access specific dependencies via AsRef
impl AsRef<Database> for AppContext {
    fn as_ref(&self) -> &Database {
        &self.db
    }
}

// Tool with context injection - receives Database via Ctx<&Database>
#[tool]
/// Search the knowledge base for information.
/// query: The search term.
fn search_docs(db: Ctx<&Database>, query: String) -> Vec<String> {
    db.search(&query)
}

// Tool without context - works alongside context-aware tools
#[tool]
/// Add two numbers together.
/// a: First number.
/// b: Second number.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Create context with dependencies
    let ctx = AppContext {
        db: Database::new(),
    };

    // Create toolset with context type: `ContextType => tools...`
    let tools = toolset![AppContext => search_docs, add].with_context(ctx)?;

    let response = llm::with(Provider::Gemini)
        .api_key(ApiKey::Default)?
        .model("gemini-2.5-flash")
        .messages(vec![
            Message {
                role: ChatRole::System,
                content:
                    "You have access to a knowledge base. Use search_docs to find information."
                        .into(),
            },
            Message {
                role: ChatRole::User,
                content: "What can you tell me about Rust?".into(),
            },
        ])
        .tools(tools)
        .complete::<TextResponse>()
        .await?;

    println!("{}", response.text);
    Ok(())
}
