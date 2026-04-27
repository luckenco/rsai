> **⚠️ WARNING**: This is a pre-release version with an unstable API. Breaking changes may occur between versions. Use with caution and pin to specific versions in production applications.

## Design Philosophy

This library offers an **opinionated feature set**, rather than trying to be a general-purpose LLM client.

- **Type Safety vs. TTFT**: Streaming is not supported. We explicitly prioritize type safety and validation completeness over Time To First Token (TTFT). You get a valid struct or an error, never a partial state.
- **Alternatives**: For a more general-purpose library that supports streaming, disparate providers, and conversational features, consider [Rig](https://github.com/0xPlaygrounds/rig).

## Available Providers

| Provider | API Type | Notes |
|----------|----------|-------|
| **OpenAI** | Responses API | Uses the `/responses` endpoint for structured interactions. |
| **OpenRouter** | Responses API | Uses the `/responses` endpoint, supporting a wide range of models. |

## Quick Start

```rust
use rsai::{llm, Message, ChatRole, ApiKey, Provider, completion_schema};

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
}

let analysis = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![Message {
        role: ChatRole::User,
        content: "Analyze: 'This library is amazing!'".to_string(),
    }])
    .complete::<Analysis>()
    .await?;
```

## Structured Generation

The `#[completion_schema]` macro automatically adds the necessary derives (`Deserialize`, `JsonSchema`) and attributes for structured output. It supports:

- **Structs**
- **Enums** (Unit, Tuple, and Struct variants)

```rust
#[completion_schema]
enum TaskStatus {
    NotStarted,
    InProgress { percentage: u8 },
    Completed { date: String },
    Blocked { reason: String },
}

let status = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![/* ... */])
    .complete::<TaskStatus>()
    .await?;
```

> **Note**: The library automatically handles provider-specific requirements (e.g., wrapping non-object types for OpenAI).

## Text Generation

For plain text, use `TextResponse`.

```rust
use rsai::{llm, TextResponse, /* ... */};

let response = llm::with(Provider::OpenAI)
    // ... configuration ...
    .complete::<TextResponse>()
    .await?;

println!("{}", response.text);
```

See `examples/` for more runnable examples.

## Known Issues

- **TODO**: Evaluate whether `ToolCallingConfig` should be required rather than optional. Currently providers default to a guard with sensible limits when no config is set, but allowing `None` suggests users can opt out of loop protection entirely. We may want to be opinionated here and always require a `ToolCallingConfig`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
