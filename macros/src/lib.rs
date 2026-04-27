//! Procedural macros for the rsai crate providing structured generation capabilities.
//!
//! This crate provides three main macros:
//!
//! - [`completion_schema`] - Automatically generates JSON schema for response types
//! - [`tool`](fn@tool) - Transforms Rust functions into callable tools with automatic schema generation
//! - [`toolset`] - Creates collections of tools
//!
//! # Quick Start
//!
//! ```rust
//! use rsai_macros::{completion_schema, tool, toolset};
//! use serde::{Deserialize, Serialize};
//!
//! #[completion_schema]
//! struct WeatherResponse {
//!     city: String,
//!     temperature: f64,
//!     conditions: String,
//! }
//!
//! #[tool]
//! /// Get current weather for a city
//! /// city: The city to get weather for
//! /// unit: Temperature unit (celsius or fahrenheit)
//! fn get_weather(city: String, unit: Option<String>) -> String {
//!     match unit.as_deref() {
//!         Some("fahrenheit") => format!("Weather in {}: 72°F", city),
//!         _ => format!("Weather in {}: 22°C", city),
//!     }
//! }
//!
//! let tools = toolset![get_weather];
//! // Use with rsai::llm builder...
//! ```

use proc_macro::TokenStream;
use quote::quote;

mod common;
mod tool;
mod tools;

/// Attribute macro for types used with the `rsai::llm()::complete::<T>()` method.
///
/// This macro automatically adds the necessary derives and attributes to make a struct
/// suitable for structured responses. It ensures strict validation of
/// responses by rejecting unknown fields.
///
/// # What it does
///
/// - Adds `#[derive(serde::Deserialize, schemars::JsonSchema)]`
/// - Adds `#[serde(deny_unknown_fields)]` for strict local deserialization
/// - Adds `#[schemars(deny_unknown_fields)]` for strict schema generation
/// - Enables automatic JSON schema generation for LLM providers
///
/// # Example
///
/// ```rust
/// use rsai_macros::completion_schema;
/// use serde::Deserialize;
///
/// #[completion_schema]
/// struct WeatherResponse {
///     /// The city name
///     city: String,
///     /// Temperature in Celsius
///     temperature: f64,
///     /// Weather conditions
///     conditions: String,
///     /// Optional humidity percentage
///     humidity: Option<f64>,
/// }
/// ```
///
/// # Field Validation
///
/// The `deny_unknown_fields` attributes ensure that responses must exactly match
/// your struct definition. Any extra fields will cause a local deserialization
/// error and are also rejected in the generated JSON schema.
///
/// # Supported Types
///
/// All types that implement [`serde::Deserialize`] and [`schemars::JsonSchema`] are supported,
/// including:
/// - Primitive types ([`String`], [`i32`], [`f64`], [`bool`], etc.)
/// - Optionals ([`Option<T>`])
/// - Vectors ([`Vec<T>`])
/// - Nested structs and enums
/// - Custom types with appropriate trait implementations
#[proc_macro_attribute]
pub fn completion_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_tokens: proc_macro2::TokenStream = item.into();

    let expanded = quote! {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #[serde(deny_unknown_fields)]
        #[schemars(deny_unknown_fields)]
        #item_tokens
    };

    TokenStream::from(expanded)
}

/// Attribute macro for marking functions as tools that can be called by LLMs.
///
/// This macro generates the necessary boilerplate to make a function callable as a tool,
/// including automatic JSON schema generation, parameter parsing, and error handling.
/// It enables AI models to call your Rust functions safely and predictably.
///
/// # Syntax
///
/// The macro expects a specific docstring format:
///
/// ```rust
/// use rsai_macros::tool;
///
/// #[tool]
/// /// Function description (required - first line of docstring)
/// /// param: Parameter description (required for each parameter)
/// /// optional_param: Description for optional parameters (also required)
/// fn function_name(param: String, optional_param: Option<i32>) -> String {
///     // implementation
///     format!("param: {}, optional: {:?}", param, optional_param)
/// }
/// ```
///
/// # Examples
///
/// ## Basic Tool
///
/// ```rust
/// use rsai_macros::tool;
///
/// #[tool]
/// /// Get current weather for a city
/// /// city: The city to get weather for
/// /// unit: Temperature unit (celsius or fahrenheit)
/// fn get_weather(city: String, unit: Option<String>) -> String {
///     match unit.as_deref() {
///         Some("fahrenheit") => format!("Weather for {}: 72°F", city),
///         _ => format!("Weather for {}: 22°C", city),
///     }
/// }
/// ```
///
/// ## Async Tool
///
/// ```rust
/// use rsai_macros::tool;
///
/// #[tool]
/// /// Send an email to a recipient
/// /// to: Email address of the recipient
/// /// subject: Email subject line
/// /// body: Email body content
/// async fn send_email(to: String, subject: String, body: String) -> Result<String, String> {
///     // Simulate async email sending
///     tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
///     Ok(format!("Email sent to {}", to))
/// }
/// ```
///
/// # Parameter Validation
///
/// The macro performs comprehensive compile-time validation:
///
/// ✅ **Valid**: All parameters documented
/// ```rust
/// use rsai_macros::tool;
///
/// #[tool]
/// /// Get weather info
/// /// city: The city name
/// /// country: The country code
/// fn get_weather(city: String, country: String) -> String {
///     format!("Weather for {}, {}", city, country)
/// }
/// ```
///
/// ❌ **Invalid**: Missing parameter description
/// ```rust,compile_fail
/// #[tool]
/// /// Get weather info
/// /// city: The city name
/// fn get_weather(city: String, country: String) -> String {
///     // Compile error: Missing description for parameter 'country'
/// }
/// ```
///
/// # Type Mapping
///
/// Rust types are automatically mapped to JSON schema types:
///
/// | Rust Type | JSON Schema Type |
/// |-----------|------------------|
/// | `String`, `&str` | `string` |
/// | `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64` | `number` |
/// | `bool` | `boolean` |
/// | `Vec<T>` | `array` |
/// | `Option<T>` | `T` (optional) |
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    match tool::tool_impl(attr.into(), item.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Macro for creating a collection of tools from annotated [`tool`] functions.
///
/// This macro takes a comma-separated list of function names that have been
/// annotated with `#[tool]` and creates a `rsai::ToolSet` containing all of them.
/// It provides a convenient way to group related tools for AI agents.
///
/// # Syntax
///
/// ```rust,ignore
/// use rsai_macros::{tool, toolset};
///
/// // Assume these functions are defined with #[tool] attribute:
/// // #[tool] fn function_name1(param: String) -> String { ... }
/// // #[tool] fn function_name2(param: i32) -> i32 { ... }
/// // #[tool] fn function_name3(param: bool) -> bool { ... }
///
/// let tools = toolset![function_name1, function_name2, function_name3];
/// ```
///
/// # Requirements
///
/// All function names must correspond to functions that have the `#[tool]` attribute
/// applied. The macro will fail to compile if any function is not properly annotated.
///
/// # Examples
///
/// ## Single Tool
///
/// ```rust
/// use rsai_macros::{tool, toolset};
///
/// #[tool]
/// /// Get current weather for a city
/// /// city: The city to get weather for
/// fn get_weather(city: String) -> String {
///     format!("Weather for {}: 22°C", city)
/// }
///
/// let single_tool = toolset![get_weather];
/// assert_eq!(
///     single_tool
///         .tools()
///         .expect("toolset should contain the get_weather tool")
///         .len(),
///     1
/// );
/// ```
///
/// ## Many Tools
///
/// ```rust
/// use rsai_macros::{tool, toolset};
///
/// #[tool]
/// /// Send an email
/// /// to: Recipient address
/// fn send_email(to: String) -> String {
///     format!("Email sent to {}", to)
/// }
///
/// #[tool]
/// /// Get user information
/// /// user_id: User identifier
/// fn get_user(user_id: String) -> String {
///     format!("User: {}", user_id)
/// }
///
/// #[tool]
/// /// Create a file
/// /// path: File path
/// /// content: File content
/// fn create_file(path: String, content: String) -> String {
///     format!("File created at {}", path)
/// }
///
/// let tools = toolset![send_email, get_user, create_file];
/// assert_eq!(
///     tools
///         .tools()
///         .expect("toolset should contain all registered tools")
///         .len(),
///     3
/// );
/// ```
///
/// # Generated Code
///
/// The macro generates code that:
/// 1. Creates a new `rsai::ToolRegistry`
/// 2. Registers each tool function
/// 3. Builds the final `rsai::ToolSet`
///
/// ```rust,no_run
/// use rsai::ToolRegistry;
/// use std::sync::Arc;
///
/// {
///     let registry = ToolRegistry::new();
///     // Note: FunctionName1Tool and FunctionName2Tool would be created by the #[tool] macro
///     // registry.register(FunctionName1Tool);
///     // registry.register(FunctionName2Tool);
///     // The macro then creates the ToolSet from the registry
/// }
/// ```
///
/// # Error Handling
///
/// If any function name doesn't correspond to a `#[tool]`-annotated function,
/// compilation will fail with a clear error message indicating which function
/// is missing the tool annotation.
#[proc_macro]
pub fn toolset(input: TokenStream) -> TokenStream {
    match tools::tools_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
