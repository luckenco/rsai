use rsai::{tool, toolset};

#[tool]
/// Echo a value.
/// value: The value to echo.
fn echo(value: String) -> String {
    value
}

fn main() {
    let _tools = toolset![echo, echo];
}
