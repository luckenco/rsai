# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-08-11

### Added

- Generate complete `#[tool]` parameter schemas with Schemars, including array items, nested structs, enums, nullability, and field descriptions ([#68]).
- Validate generation limits before dispatch for OpenAI, OpenRouter, and Gemini.
- Reject duplicate entries in `toolset!` at compile time and duplicate context-aware tools when finalizing a toolset.
- Add request-contract tests for Gemini JSON Schema fields, multiple system messages, and provider-independent generation validation.
- Document all public APIs and expand the README with current providers, tools, shared context, and configuration examples.

### Changed

- `ToolSetBuilder::with_context` now returns `Result<ToolSet<_>, LlmError>` instead of panicking when tool registration fails.
- Tool parameters must implement both `serde::Deserialize` and `schemars::JsonSchema`.
- Tool execution now deserializes the complete argument object and rejects unknown fields.
- OpenAI and OpenRouter recursively normalize strict tool schemas, while Gemini receives provider-neutral JSON Schema through `parametersJsonSchema` and `responseJsonSchema`.
- Preserve all Gemini system messages in their original order.
- Split Responses API request building, response parsing, and tool-loop execution into focused modules.
- Raise the minimum supported Rust version from 1.85 to 1.97, update dependencies, and replace the unmaintained `dotenv` example dependency with `dotenvy`.

### Fixed

- Preserve optional tool arguments expressed as `Option<T>` or `std::option::Option<T>`, including missing and explicit `null` values ([#68]).
- Preserve nested schema details instead of reducing arrays and custom tool parameters to shallow schemas ([#68]).
- Normalize nested OpenAI strict objects recursively and remove unsupported numeric schema formats.

### Breaking Changes

- Rust 1.97 or newer is required.
- Context-aware toolsets require handling the `Result` returned by `.with_context(...)`.
- Custom tool parameter types must derive or implement `Deserialize` and `JsonSchema`.
- Unknown tool arguments that were previously ignored now return `LlmError::ToolExecution`.

## [0.4.0] - 2026-04-27

### Added

- Add tool-loop limits, concurrent execution limits, per-tool timeouts, HTTP client pooling, dependency audit CI, and a committed lockfile.

### Changed

- Run generated synchronous tools on Tokio's blocking pool.
- Declare Rust 1.85 as the minimum supported Rust version.

### Fixed

- Reject insecure provider base URLs unless explicitly enabled.
- Harden tagged response parsing, unknown-field rejection, output aggregation, function-call round trips, and sensitive tracing output.

[Unreleased]: https://github.com/luckenco/rsai/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/luckenco/rsai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/luckenco/rsai/releases/tag/v0.4.0
[#68]: https://github.com/luckenco/rsai/issues/68
