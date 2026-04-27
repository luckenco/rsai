use rsai_macros::{completion_schema, tool};

#[completion_schema]
#[derive(Debug, PartialEq)]
struct StrictResponse {
    value: String,
}

#[test]
fn test_completion_schema_rejects_unknown_fields_locally() {
    let parsed: Result<StrictResponse, _> = serde_json::from_str(r#"{"value":"ok","extra":true}"#);

    assert!(parsed.is_err());
}

#[tool]
/// Get current temperature for a given location.
/// _location: City and country e.g. Bogotá, Colombia
fn get_weather(_location: String) -> f64 {
    22.5
}

#[test]
fn test_get_weather_tool_schema() {
    let tool_instance = GetWeatherTool;
    let tool = tool_instance.schema();

    assert_eq!(tool.name, "get_weather");
    assert_eq!(
        tool.description,
        Some("Get current temperature for a given location.".to_string())
    );
    assert_eq!(tool.strict, Some(true));

    let params = &tool.parameters;
    assert_eq!(params["type"], "object");

    let properties = &params["properties"];
    let location_prop = &properties["_location"];
    assert_eq!(location_prop["type"], "string");
    assert_eq!(
        location_prop["description"],
        "City and country e.g. Bogotá, Colombia"
    );

    let required = params["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "_location");

    assert_eq!(params["additionalProperties"], false);

    // Print the full schema for debugging
    println!(
        "Generated schema: {}",
        serde_json::to_string_pretty(&tool.parameters).unwrap()
    );
}

#[tool]
/// This function does multiple things with various parameters.
/// It's a complex function for testing.
/// param1: First parameter description
/// param2: Second parameter that is optional
/// param3: Third parameter with special chars: symbols, numbers, etc.
fn complex_function(param1: String, param2: Option<i32>, param3: bool) -> String {
    format!("{} {} {}", param1, param2.unwrap_or(0), param3)
}

#[test]
fn test_complex_function_docstring_parsing() {
    let tool_instance = ComplexFunctionTool;
    let tool = tool_instance.schema();

    // Check function description (should exclude parameter lines)
    assert_eq!(
        tool.description,
        Some("This function does multiple things with various parameters. It's a complex function for testing.".to_string())
    );

    let params = &tool.parameters;
    let properties = &params["properties"];

    // Check param1
    let param1 = &properties["param1"];
    assert_eq!(param1["type"], "string");
    assert_eq!(param1["description"], "First parameter description");

    // Check param2 (optional)
    let param2 = &properties["param2"];
    assert_eq!(param2["type"], "integer");
    assert_eq!(param2["description"], "Second parameter that is optional");

    // Check param3
    let param3 = &properties["param3"];
    assert_eq!(param3["type"], "boolean");
    assert_eq!(
        param3["description"],
        "Third parameter with special chars: symbols, numbers, etc."
    );

    // Check required array (should only have param1 and param3, not param2 since it's Option<T>)
    let required = params["required"].as_array().unwrap();
    assert_eq!(required.len(), 2);
    assert!(required.contains(&serde_json::Value::String("param1".to_string())));
    assert!(required.contains(&serde_json::Value::String("param3".to_string())));
    assert!(!required.contains(&serde_json::Value::String("param2".to_string())));
}
