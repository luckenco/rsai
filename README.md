# rsai

Predictable development for unpredictable models. Let the compiler handle the chaos.

> **Warning:** rsai is pre-release software with an unstable API. Breaking changes may occur between versions. Pin versions in production applications.

rsai is an opinionated Rust client for typed LLM generation and automatic tool calling. It favors complete validation and type safety over streaming and time to first token: callers receive a valid value or an error, never a partial response.

## Supported Providers

| Provider | API | Default API key variable | Notes |
|----------|-----|--------------------------|-------|
| **OpenAI** | Responses API | `OPENAI_API_KEY` | Structured output and automatic tool calling. |
| **OpenRouter** | Responses API | `OPENROUTER_API_KEY` | OpenAI-compatible access to supported OpenRouter models. |
| **Google Gemini** | `generateContent` API | `GEMINI_API_KEY` | Structured output or automatic tool calling; Gemini does not support combining them in one request. |

## Requirements

rsai requires Rust 1.97 or newer.

```toml
[dependencies]
rsai = "0.4"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

# Required when using #[completion_schema]
serde = { version = "1", features = ["derive"] }
schemars = "1"
```

Set the environment variable for your provider, or pass a key with `ApiKey::Custom`.

## Quick Start

```rust
use rsai::{ApiKey, ChatRole, Message, Provider, completion_schema, llm};

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let analysis = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Analyze: 'This library is amazing!'".to_string(),
        }])
        .complete::<Analysis>()
        .await?;

    println!("{} ({})", analysis.content.sentiment, analysis.content.confidence);
    Ok(())
}
```

## Structured Generation

`#[completion_schema]` derives `Deserialize` and `JsonSchema`, rejects unknown response fields, and enables typed completion parsing. Structs and enums are supported. Non-object schemas are wrapped automatically for providers that require an object root.

```rust
use rsai::completion_schema;

#[completion_schema]
enum TaskStatus {
    NotStarted,
    InProgress { percentage: u8 },
    Completed { date: String },
    Blocked { reason: String },
}
```

## Text Generation

Use `TextResponse` when a typed JSON response is unnecessary.

```rust
use rsai::{ApiKey, ChatRole, Message, Provider, TextResponse, llm};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Share a concise Rust fact.".to_string(),
        }])
        .complete::<TextResponse>()
        .await?;

    println!("{}", response.text);
    Ok(())
}
```

## Function Calling

Annotate functions with `#[tool]`, document every non-context parameter, collect them with `toolset!`, and attach the resulting toolset to the request. Tool arguments are schema-validated before execution.

```rust
use rsai::{ApiKey, ChatRole, Message, Provider, TextResponse, llm, tool, toolset};

#[tool]
/// Add two integers.
/// a: First integer.
/// b: Second integer.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tools = toolset![add];
    let response = llm::with(Provider::OpenAI)
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "What is 20 + 22?".to_string(),
        }])
        .tools(tools)
        .complete::<TextResponse>()
        .await?;

    println!("{}", response.text);
    Ok(())
}
```

### Tools With Shared Context

Context-aware tools receive dependencies through `Ctx<&T>`. The containing context must implement `AsRef<T>`. Finalizing a context-aware toolset is fallible because duplicate tool names are rejected.

```rust
use rsai::{Ctx, tool, toolset};

struct Database;
struct AppContext {
    database: Database,
}

impl AsRef<Database> for AppContext {
    fn as_ref(&self) -> &Database {
        &self.database
    }
}

#[tool]
/// Search stored records.
/// query: Search query.
fn search(_database: Ctx<&Database>, query: String) -> String {
    format!("Result for {query}")
}

fn main() -> Result<(), rsai::LlmError> {
    let context = AppContext { database: Database };
    let tools = toolset![AppContext => search].with_context(context)?;

    println!("Registered {} tool", tools.tools()?.len());
    Ok(())
}
```

## Generation and HTTP Configuration

The builder supports:

- `.max_tokens(...)`
- `.temperature(...)` and `.top_p(...)`
- `.timeout(...)` or `.http_client_config(...)`
- `.tool_calling_config(...)` for tool-loop limits, concurrency, and timeouts
- `.inspect_request(...)` and `.inspect_response(...)` for raw JSON inspection

See [`examples/`](examples/) for complete provider, structured generation, function calling, shared context, tracing, inspection, and configuration examples.

```bash
cargo run --example text-generation
cargo run --example function-calling
cargo run --example gemini
cargo run --example tool-context
```

## Design Philosophy

This library deliberately does not support streaming. It prioritizes type safety and complete validation over time to first token. For a general-purpose client with streaming, disparate providers, and broader conversational features, consider [Rig](https://github.com/0xPlaygrounds/rig).

## License

Licensed under the [MIT License](LICENSE).
