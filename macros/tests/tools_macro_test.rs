use rsai_macros::{tool, toolset};

/// Get the current weather for a city
/// _city: The city to get weather for
/// _unit: Temperature unit (celsius or fahrenheit)
#[tool]
async fn get_weather(_city: String, _unit: Option<String>) -> f64 {
    22.0
}

/// Calculate distance between two locations
/// _from: Starting location
/// _to: Destination location
#[tool]
fn calculate_distance(_from: String, _to: String) -> f64 {
    42.5
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct SearchFilter {
    exact: bool,
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
enum MatchMode {
    Exact,
    Prefix,
}

/// Search records by tags.
/// tags: Tags that records must contain.
/// filter: Additional search options.
#[tool]
fn search_records(tags: Vec<String>, filter: SearchFilter) -> usize {
    tags.len() + usize::from(filter.exact)
}

/// Join borrowed text inputs.
/// prefix: Required text prefix.
/// suffix: Optional text suffix.
#[tool]
fn join_borrowed(prefix: &str, suffix: Option<&str>) -> String {
    format!("{prefix}{}", suffix.unwrap_or_default())
}

/// Check whether a qualified optional value is absent.
/// value: Optional value.
#[tool]
fn qualified_optional(value: std::option::Option<String>) -> bool {
    value.is_none()
}

/// Select a matching mode.
/// mode: Matching behavior.
#[tool]
fn select_mode(mode: MatchMode) -> bool {
    matches!(mode, MatchMode::Exact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsai::ToolCall;
    use serde_json::json;

    #[test]
    fn test_single_tool_macro() {
        // Test that the macro compiles and creates a toolset
        let toolset = toolset![get_weather];

        // Basic assertions to verify the structure works
        let tools = toolset
            .tools()
            .expect("toolset macro should expose generated tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert!(tools[0].description.is_some());
    }

    #[test]
    fn test_multiple_tools_macro() {
        // Test with multiple tools
        let toolset = toolset![get_weather, calculate_distance];

        let tools = toolset
            .tools()
            .expect("toolset macro should expose generated tools");
        assert_eq!(tools.len(), 2);

        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        assert!(tool_names.contains(&"get_weather".to_string()));
        assert!(tool_names.contains(&"calculate_distance".to_string()));
    }

    #[test]
    fn test_registry_execution_functionality() {
        let toolset = toolset![get_weather, calculate_distance];

        // Test that registry has correct tools
        let schemas = toolset
            .registry
            .get_schemas()
            .expect("tool registry should return schemas");
        assert_eq!(schemas.len(), 2);

        // Verify schema names match
        let names: Vec<String> = schemas.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"get_weather".to_string()));
        assert!(names.contains(&"calculate_distance".to_string()));
    }

    #[test]
    fn test_tool_parameter_schemas_in_registry() {
        let toolset = toolset![get_weather];
        let schemas = toolset
            .registry
            .get_schemas()
            .expect("tool registry should return schemas");

        assert_eq!(schemas.len(), 1);
        let weather_schema = &schemas[0];

        // Verify parameter schema structure
        assert_eq!(weather_schema.name, "get_weather");
        assert!(weather_schema.description.is_some());
        assert_eq!(
            weather_schema.description.as_ref().unwrap(),
            "Get the current weather for a city"
        );

        // Check parameter structure
        let params = &weather_schema.parameters;
        assert_eq!(params["type"], "object");

        let properties = &params["properties"];
        assert!(properties["_city"].is_object());
        assert!(properties["_unit"].is_object());

        // Verify required parameters
        let required = &params["required"];
        assert!(required.is_array());
        let required_array = required.as_array().unwrap();
        assert!(required_array.contains(&json!("_city")));
        assert!(!required_array.contains(&json!("_unit"))); // Optional parameter
    }

    #[test]
    fn test_tool_schema_preserves_arrays_and_custom_types() {
        let schema = SearchRecordsTool.schema();
        let properties = &schema.parameters["properties"];

        assert_eq!(properties["tags"]["type"], "array");
        assert_eq!(properties["tags"]["items"]["type"], "string");
        assert_eq!(properties["filter"]["$ref"], "#/$defs/SearchFilter");
        assert_eq!(
            schema.parameters["$defs"]["SearchFilter"]["properties"]["exact"]["type"],
            "boolean"
        );
    }

    #[test]
    fn test_tool_schema_preserves_custom_enums() {
        let schema = SelectModeTool.schema();
        let mode = &schema.parameters["$defs"]["MatchMode"];

        assert_eq!(mode["type"], "string");
        assert_eq!(mode["enum"], json!(["Exact", "Prefix"]));
    }

    #[tokio::test]
    async fn test_qualified_option_accepts_missing_and_null() {
        let toolset = toolset![qualified_optional];
        let schema = toolset.tools().expect("tool schema").remove(0);
        let required = schema
            .parameters
            .get("required")
            .and_then(|value| value.as_array());
        assert!(required.is_none_or(|required| !required.contains(&json!("value"))));

        for arguments in [json!({}), json!({ "value": null })] {
            let result = toolset
                .registry
                .execute(&ToolCall {
                    id: "call_optional".to_string(),
                    call_id: "call_optional".to_string(),
                    name: "qualified_optional".to_string(),
                    arguments,
                })
                .await
                .expect("tool execution");

            assert_eq!(result, json!(true));
        }
    }

    #[tokio::test]
    async fn test_tool_supports_borrowed_str_parameters() {
        let toolset = toolset![join_borrowed];
        let schema = toolset.tools().expect("tool schema").remove(0);

        assert_eq!(schema.parameters["properties"]["prefix"]["type"], "string");
        assert_eq!(
            schema.parameters["properties"]["suffix"]["type"][0],
            "string"
        );

        let result = toolset
            .registry
            .execute(&ToolCall {
                id: "call_1".to_string(),
                call_id: "call_1".to_string(),
                name: "join_borrowed".to_string(),
                arguments: json!({ "prefix": "hello", "suffix": " world" }),
            })
            .await
            .expect("tool execution");

        assert_eq!(result, json!("hello world"));
    }

    #[tokio::test]
    async fn test_tool_execution_rejects_unknown_parameters() {
        let toolset = toolset![search_records];
        let call = ToolCall {
            id: "call_search".to_string(),
            call_id: "call_search".to_string(),
            name: "search_records".to_string(),
            arguments: json!({
                "tags": ["rust"],
                "filter": { "exact": true },
                "unexpected": true
            }),
        };

        let error = toolset.registry.execute(&call).await.unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[tokio::test]
    async fn test_registry_execute_method() {
        let toolset = toolset![get_weather, calculate_distance];

        // Test successful execution of get_weather
        let weather_call = ToolCall {
            id: "call_1".to_string(),
            call_id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "New York",
                "_unit": "celsius"
            }),
        };

        let result = toolset.registry.execute(&weather_call).await;
        let result_val = result.unwrap();
        assert!(result_val.is_number());
        assert_eq!(result_val.as_f64(), Some(22.0));

        // Test successful execution of calculate_distance
        let distance_call = ToolCall {
            id: "call_2".to_string(),
            call_id: "call_2".to_string(),
            name: "calculate_distance".to_string(),
            arguments: json!({
                "_from": "New York",
                "_to": "Boston"
            }),
        };

        let result = toolset.registry.execute(&distance_call).await;
        let result_val = result.unwrap();
        assert!(result_val.is_number());
        assert_eq!(result_val.as_f64(), Some(42.5));
    }

    #[tokio::test]
    async fn test_registry_execute_method_error_cases() {
        let toolset = toolset![get_weather];

        // Test execution with non-existent tool
        let invalid_call = ToolCall {
            id: "call_invalid".to_string(),
            call_id: "call_invalid".to_string(),
            name: "non_existent_tool".to_string(),
            arguments: json!({}),
        };

        let result = toolset.registry.execute(&invalid_call).await;
        assert!(result.is_err());

        // Verify error message contains tool name
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("non_existent_tool"));
    }

    #[test]
    fn test_choice_enum_generation() {
        let toolset = toolset![get_weather, calculate_distance];

        // The macro should generate a Choice enum, but we need to access it properly
        // This tests that the macro compiles with choice enum syntax
        // We can't directly test the enum without exposing it, but we can verify
        // the toolset structure that depends on it works correctly

        let tools = toolset
            .tools()
            .expect("toolset macro should expose generated tools");
        assert_eq!(tools.len(), 2);
        let schemas = toolset
            .registry
            .get_schemas()
            .expect("tool registry should return schemas");
        assert_eq!(schemas.len(), 2);
    }

    #[tokio::test]
    async fn test_tool_invocation_through_registry() {
        let toolset = toolset![get_weather, calculate_distance];

        // Test complete workflow: schema -> call -> execution
        let schemas = toolset
            .registry
            .get_schemas()
            .expect("tool registry should return schemas");
        assert_eq!(schemas.len(), 2);

        // Find weather tool schema
        let weather_schema = schemas
            .iter()
            .find(|s| s.name == "get_weather")
            .expect("Should find get_weather schema");

        // Create tool call based on schema
        let tool_call = ToolCall {
            id: "test_call".to_string(),
            call_id: "test_call".to_string(),
            name: weather_schema.name.clone(),
            arguments: json!({
                "_city": "San Francisco",
                "_unit": "fahrenheit"
            }),
        };

        // Execute through registry
        let result = toolset.registry.execute(&tool_call).await;
        let result_val = result.unwrap();
        assert!(result_val.is_number());
        assert_eq!(result_val.as_f64(), Some(22.0));

        // Test synchronous tool (calculate_distance)
        let distance_schema = schemas
            .iter()
            .find(|s| s.name == "calculate_distance")
            .expect("Should find calculate_distance schema");

        let distance_call = ToolCall {
            id: "distance_call".to_string(),
            call_id: "distance_call".to_string(),
            name: distance_schema.name.clone(),
            arguments: json!({
                "_from": "Los Angeles",
                "_to": "San Diego"
            }),
        };

        let result = toolset.registry.execute(&distance_call).await;
        let result_val = result.unwrap();
        assert!(result_val.is_number());
        assert_eq!(result_val.as_f64(), Some(42.5));
    }

    #[tokio::test]
    async fn test_registry_with_optional_parameters() {
        let toolset = toolset![get_weather];

        // Test with optional parameter provided
        let call_with_unit = ToolCall {
            id: "call_1".to_string(),
            call_id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "Tokyo",
                "_unit": "celsius"
            }),
        };

        let result = toolset.registry.execute(&call_with_unit).await;
        assert!(result.is_ok());

        // Test without optional parameter (unit should default to None)
        let call_without_unit = ToolCall {
            id: "call_2".to_string(),
            call_id: "call_2".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "Tokyo"
            }),
        };

        let result = toolset.registry.execute(&call_without_unit).await;
        let result_val = result.unwrap();
        assert!(result_val.is_number());
        assert_eq!(result_val.as_f64(), Some(22.0))
    }
}
