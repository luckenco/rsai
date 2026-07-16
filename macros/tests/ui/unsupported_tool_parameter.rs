use rsai::tool;

#[derive(serde::Deserialize)]
struct Unsupported;

#[tool]
/// Use an unsupported parameter.
/// value: Unsupported value.
fn unsupported(value: Unsupported) {
    let _ = value;
}

fn main() {}
