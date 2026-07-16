- Keep things modular.
- Define clear, testable contract boundaries. Document public contracts, and ensure Rustdoc examples compile and run as doctests.
- Use assertions for invariants when encoding within types is not feasible.
- Use clippy and fix code smells reported.
- Prefer iterators, monadic logic, and functional concepts. Where not possible for performance, document why.
- Make invalid state unrepresentable.
- Prefer importing modules over long lists of types. Using `module::Type` at call sites is fine, keeps names unambiguous, and reduces namespace-collision churn.
- Be wary of designs that will not scale. Report the limiting assumption, expected failure mode, and likely impact to the user.

# Repository Guidelines

## Project Structure & Module Organization

This is a Rust workspace with the main `rsai` crate at the repository root and a procedural macro crate in `macros/`. Core library code lives in `src/`: `core/` contains builder, trait, error, HTTP, and type abstractions; `provider/` contains OpenAI, OpenRouter, Gemini, and provider constants; `responses/` and `completions/` contain request/response client flows. Integration tests are in `tests/`; macro tests, including `trybuild` UI cases, are in `macros/tests/`. Runnable usage examples are in `examples/`.

## Build, Test, and Development Commands

- `cargo build` builds the full workspace.
- `cargo test` runs unit, integration, and macro crate tests.
- `cargo test test_name` runs a focused test by name.
- `cargo run --example function-calling` runs an example; replace the name with any example listed in `Cargo.toml`.
- `cargo fmt` formats Rust code with the toolchain-pinned `rustfmt`.
- `cargo clippy --all --all-features -- -Dwarnings` runs the stricter pre-commit lint pass.

The repository pins stable Rust with `rust-toolchain.toml` and declares MSRV `1.85`.

## Coding Style & Naming Conventions

Use Rust 2024 idioms and standard `rustfmt` formatting. Prefer strong typed APIs, enums, `Option<T>`, and the existing builder and trait patterns over ad hoc configuration. Use `snake_case` for functions, modules, variables, and test names; use `CamelCase` for types, traits, and enum variants. Public APIs should have doc comments. Inline comments should explain only non-obvious behavior. Errors should use the existing `LlmError` style with `thiserror`.

## Testing Guidelines

Add or update tests near the affected behavior. Use root-level `tests/*.rs` for public API and provider behavior, and `macros/tests/` for procedural macro behavior. UI compile-fail cases belong in `macros/tests/ui/` with matching `.stderr` files. Before submitting, run `cargo test`, then `cargo clippy --all --all-features -- -Dwarnings`.

## Commit & Pull Request Guidelines

Recent history uses concise imperative subjects such as `Track Rust stable locally and declare MSRV 1.85`, with occasional prefixes like `fix:`, `feature:`, and `chore:`. Keep commits focused and describe the user-visible change or maintenance task. Pull requests should include a short summary, tests run, linked issue when applicable, and notes for API or behavior changes.

## Security & Configuration Tips

Do not commit API keys, `.env` files, or broad local agent permission files. Keep Claude or Codex local settings narrow; avoid blanket command approvals such as unrestricted `cargo:*`, because builds, tests, examples, build scripts, and proc macros can execute code.
