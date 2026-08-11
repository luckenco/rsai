# rsai-macros

Procedural macros that provide structured AI generation capabilities for the rsai crate. This crate enables type-safe tool calling and automatic JSON schema generation for AI interactions.

## 🚀 Features

- **`#[completion_schema]`** - Automatic JSON schema generation for AI response types
- **`#[tool]`** - Transform Rust functions into AI-callable tools with automatic schema generation
- **`toolset!`** - Create collections of tools for AI agents
- **Type Safety** - Compile-time validation of tool parameters and descriptions
- **Async Support** - First-class support for async tool functions
- **Error Handling** - Comprehensive error mapping and validation

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rsai-macros = "0.5.0"
```

## 🛠️ Macros

### `#[completion_schema]`

Automatically adds the necessary derives and attributes for types used with the `complete::<T>()` method.

#### What it does
- Adds `#[derive(serde::Deserialize, schemars::JsonSchema)]`
- Adds `#[schemars(deny_unknown_fields)]` for strict validation
- Ensures AI responses match your expected structure

#### Example

```rust
use rsai_macros::completion_schema;

#[completion_schema]
struct WeatherResponse {
    city: String,
    temperature: f64,
    conditions: String,
    humidity: Option<f64>,
}

// The macro expands to:
#[derive(serde::Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
struct WeatherResponse {
    city: String,
    temperature: f64,
    conditions: String,
    humidity: Option<f64>,
}
```

### `#[tool]`

Transforms Rust functions into AI-callable tools with automatic JSON schema generation.

#### Features
- **Docstring Parsing**: Extracts function descriptions and parameter documentation
- **Parameter Validation**: Ensures all docstring parameters exist in the function signature
- **Optional Parameters**: Automatically detects `Option<T>` types and marks them as non-required
- **Type Mapping**: Converts Rust types to JSON schema types
- **Async Support**: Handles both sync and async functions; generated sync tools are offloaded to Tokio's blocking pool when run through `ToolRegistry`
- **Error Handling**: Maps errors to `LlmError` with proper context

#### Syntax

```rust
#[tool]
/// Function description (required)
/// param_name: Parameter description (required for each parameter)
/// optional_param: Description for optional parameters (also required)
fn function_name(param: Type, optional_param: Option<Type>) -> ReturnType {
    // implementation
}
```

#### Examples

**Basic Tool:**
```rust
use rsai_macros::tool;

#[tool]
/// Get current weather for a city
/// city: The city to get weather for
/// unit: Temperature unit (celsius or fahrenheit)
fn get_weather(city: String, unit: Option<String>) -> String {
    match unit.as_deref() {
        Some("fahrenheit") => format!("Weather for {}: 72°F", city),
        _ => format!("Weather for {}: 22°C", city),
    }
}
```

**Async Tool:**
```rust
use rsai_macros::tool;
use async_trait::async_trait;

#[tool]
/// Send an email to a recipient
/// to: Email address of the recipient
/// subject: Email subject line
/// body: Email body content
async fn send_email(to: String, subject: String, body: String) -> Result<String, String> {
    // Simulate sending email
    Ok(format!("Email sent to {}", to))
}
```

**Complex Types:**
```rust
use rsai_macros::tool;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Location {
    latitude: f64,
    longitude: f64,
}

#[tool]
/// Calculate distance between two locations
/// from: Starting location with coordinates
/// to: Destination location with coordinates
/// unit: Distance unit (km or miles)
fn calculate_distance(from: Location, to: Location, unit: Option<String>) -> f64 {
    let distance = ((to.latitude - from.latitude).powi(2) +
                   (to.longitude - from.longitude).powi(2)).sqrt();

    match unit.as_deref() {
        Some("miles") => distance * 0.621371,
        _ => distance,
    }
}
```

### `toolset!`

Creates a collection of tools from multiple `#[tool]`-annotated functions.

#### Syntax
```rust
let tools = toolset![function_name1, function_name2, ...];
```

#### Example
```rust
use rsai_macros::{tool, toolset};

#[tool]
/// Get current weather for a city
/// city: The city to get weather for
fn get_weather(city: String) -> String {
    format!("Weather for {}: 22°C", city)
}

#[tool]
/// Calculate distance between two locations
/// from: Starting location
/// to: Destination location
fn calculate_distance(from: String, to: String) -> f64 {
    42.5
}

// Create a toolset containing both tools
let tools = toolset![get_weather, calculate_distance];
assert_eq!(tools.tools().len(), 2);
```

## 🔄 Type Mapping

The macros automatically convert Rust types to JSON schema types:

| Rust Type | JSON Schema Type |
|-----------|------------------|
| `String`, `&str` | `string` |
| `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64` | `number` |
| `bool` | `boolean` |
| `Vec<T>` | `array` |
| `Option<T>` | `T` (optional) |
| `struct` | `object` |
| `enum` | `enum` |

## ⚠️ Error Handling

The macros provide comprehensive compile-time validation:

### Missing Parameter Descriptions
```rust
#[tool]
/// This will cause a compile error
/// city: The city to get weather for
fn get_weather(city: String, country: String) -> String {
    // Error: Missing description for parameter 'country'
}
```

### Mismatched Parameters
```rust
#[tool]
/// This will cause a compile error
/// city: The city to get weather for
/// temperature: Temperature (not in function signature)
fn get_weather(city: String) -> String {
    // Error: Parameter 'temperature' described but not found in function signature
}
```

### Helpful Error Messages
The macros provide clear, actionable error messages to help you fix documentation issues quickly.

## 🧪 Testing

This crate includes comprehensive test coverage:

```bash
# Run all tests
cargo test

# Run specific test modules
cargo test tool_macro_test
cargo test tools_macro_test

# Run UI tests (compilation failure tests)
cargo test --test compile_fail
```

## 📚 Advanced Usage

### Custom Error Types
Your tool functions can return any type that implements `Into<LlmError>`:

```rust
use rsai_macros::tool;
use rsai::LlmError;

#[derive(Debug)]
enum CustomError {
    Network(String),
    Validation(String),
}

impl From<CustomError> for LlmError {
    fn from(err: CustomError) -> Self {
        match err {
            CustomError::Network(msg) => LlmError::ToolError(msg),
            CustomError::Validation(msg) => LlmError::ValidationError(msg),
        }
    }
}

#[tool]
/// Call external API
/// endpoint: API endpoint to call
async fn call_api(endpoint: String) -> Result<String, CustomError> {
    // Your implementation here
    Ok("Success".to_string())
}
```

## 🤝 Contributing

This crate is part of the rsai project. When contributing:

1. Ensure all macros have comprehensive doc comments
2. Add tests for new functionality
3. Include UI tests for compile-time error cases
4. Follow the existing code style and patterns

## 📄 License

This crate is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## 🔗 Related Crates

- [rsai](../) - Main crate for AI interactions
- [schemars](https://docs.rs/schemars/) - JSON schema generation
- [serde](https://docs.rs/serde/) - Serialization framework
- [syn](https://docs.rs/syn/) - Token parsing and manipulation
- [quote](https://docs.rs/quote/) - Quasi-quoting for procedural macros
