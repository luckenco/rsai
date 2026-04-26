use rsai::{
    GeminiClient, LlmError, OpenAiClient, OpenAiConfig, OpenRouterClient, OpenRouterConfig,
    ToolCall, ToolCallingConfig, ToolChoice, ToolConfig, ToolRegistry, completion_schema, tool,
    toolset,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

#[completion_schema]
#[derive(Debug, Clone, serde::Serialize)]
struct WeatherSummary {
    city: String,
    temperature: f64,
    unit: String,
    conditions: String,
}

#[tool]
/// Get weather summary for a city.
/// city: City to look up (e.g. Lisbon).
/// unit: Temperature unit (celsius or fahrenheit).
fn get_weather(city: String, unit: Option<String>) -> WeatherSummary {
    let base_temp_c = match city.as_str() {
        "Lisbon" => 19.0,
        "Tokyo" => 22.0,
        "London" => 15.0,
        _ => 20.0,
    };

    let (temperature, resolved_unit) = match unit.as_deref() {
        Some("fahrenheit") => (base_temp_c * 9.0 / 5.0 + 32.0, "fahrenheit"),
        _ => (base_temp_c, "celsius"),
    };

    WeatherSummary {
        city,
        temperature,
        unit: resolved_unit.to_string(),
        conditions: "clear skies".to_string(),
    }
}

#[tool]
/// Perform a simple calculation on two numbers.
/// operation: Operation to run (add, subtract, multiply).
/// a: First number.
/// b: Second number.
async fn calculate(operation: String, a: f64, b: f64) -> f64 {
    match operation.as_str() {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        _ => f64::NAN,
    }
}

#[tool]
/// Create a light-weight trip itinerary.
/// destination: City to visit.
/// days: Number of days the trip lasts.
fn generate_itinerary(destination: String, days: u8) -> Vec<String> {
    vec![
        format!("Book flights to {destination}"),
        format!("Plan {days} day stay"),
        format!("Reserve food tour in {destination}"),
    ]
}

fn weather_tool() -> Arc<GetWeatherTool> {
    Arc::new(GetWeatherTool)
}

fn calculator_tool() -> Arc<CalculateTool> {
    Arc::new(CalculateTool)
}

#[test]
fn toolset_exposes_macro_schemas() {
    let toolset = toolset![get_weather, calculate, generate_itinerary];
    let schemas = toolset.tools().expect("tool schemas");

    assert_eq!(schemas.len(), 3);

    let weather_schema = schemas
        .iter()
        .find(|schema| schema.name == "get_weather")
        .expect("weather schema");
    assert_eq!(
        weather_schema.description.as_deref(),
        Some("Get weather summary for a city.")
    );
    assert_eq!(weather_schema.strict, Some(true));

    let weather_params = &weather_schema.parameters;
    assert_eq!(weather_params["type"], "object");
    assert_eq!(weather_params["additionalProperties"], false);

    let required: Vec<String> = weather_params["required"]
        .as_array()
        .expect("required array")
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect();
    assert_eq!(required, vec!["city".to_string()]);

    let calculator_schema = schemas
        .iter()
        .find(|schema| schema.name == "calculate")
        .expect("calculator schema");
    let calc_props = &calculator_schema.parameters["properties"];
    assert_eq!(calc_props["operation"]["type"], "string");
    assert_eq!(calc_props["a"]["type"], "number");
    assert_eq!(calc_props["b"]["type"], "number");
}

#[tokio::test]
async fn toolset_executes_macro_tools_end_to_end() {
    let toolset = toolset![get_weather, calculate, generate_itinerary];

    let weather_call = ToolCall {
        id: "call_weather".to_string(),
        call_id: "call_weather".to_string(),
        name: "get_weather".to_string(),
        arguments: json!({ "city": "Lisbon" }),
    };
    let weather_result = toolset.registry.execute(&weather_call).await.unwrap();
    assert_eq!(weather_result["city"], "Lisbon");
    assert_eq!(weather_result["unit"], "celsius");

    let calculator_call = ToolCall {
        id: "call_calc".to_string(),
        call_id: "call_calc".to_string(),
        name: "calculate".to_string(),
        arguments: json!({
            "operation": "multiply",
            "a": 6.0,
            "b": 7.0
        }),
    };
    let calculator_result = toolset.registry.execute(&calculator_call).await.unwrap();
    assert_eq!(calculator_result.as_f64(), Some(42.0));

    let itinerary_call = ToolCall {
        id: "call_itinerary".to_string(),
        call_id: "call_itinerary".to_string(),
        name: "generate_itinerary".to_string(),
        arguments: json!({
            "destination": "Tokyo",
            "days": 3
        }),
    };
    let itinerary_result = toolset.registry.execute(&itinerary_call).await.unwrap();
    assert_eq!(
        itinerary_result,
        json!([
            "Book flights to Tokyo",
            "Plan 3 day stay",
            "Reserve food tour in Tokyo"
        ])
    );
}

#[test]
fn registry_rejects_duplicate_registration() {
    let registry = ToolRegistry::new();
    registry
        .register(weather_tool())
        .expect("initial registration");

    let error = registry.register(weather_tool()).unwrap_err();
    match error {
        LlmError::ToolRegistration { tool_name, message } => {
            assert_eq!(tool_name, "get_weather");
            assert!(message.contains("already registered"));
        }
        _ => panic!("Expected ToolRegistration error"),
    }

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 1);
}

#[test]
fn registry_overwrite_preserves_latest_tools() {
    let registry = ToolRegistry::new();

    registry.register(weather_tool()).unwrap();
    registry.overwrite(calculator_tool()).unwrap();

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 2);

    let names: Vec<String> = schemas.into_iter().map(|schema| schema.name).collect();
    assert!(names.contains(&"get_weather".to_string()));
    assert!(names.contains(&"calculate".to_string()));
}

#[tokio::test]
async fn registry_is_thread_safe_under_concurrent_registration() {
    let registry = Arc::new(ToolRegistry::new());
    let mut handles = Vec::new();

    for _ in 0..6 {
        let registry_clone = Arc::clone(&registry);
        handles.push(tokio::spawn(async move {
            registry_clone.register(weather_tool())
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    assert!(
        results.iter().any(|result| result.is_ok()),
        "at least one registration should succeed"
    );
    assert!(
        results.iter().any(|result| result.is_err()),
        "duplicate registrations should fail gracefully"
    );

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 1);
    assert_eq!(schemas[0].name, "get_weather");
}

#[tokio::test]
async fn execution_errors_surface_parameter_validation() {
    let toolset = toolset![calculate];

    let invalid_call = ToolCall {
        id: "bad_call".to_string(),
        call_id: "bad_call".to_string(),
        name: "calculate".to_string(),
        arguments: json!({
            "operation": "add",
            "a": "two",
            "b": 2.0
        }),
    };

    let error = toolset.registry.execute(&invalid_call).await.unwrap_err();
    match error {
        LlmError::ToolExecution { message, .. } => {
            assert!(message.contains("Invalid parameter 'a'"));
        }
        _ => panic!("Expected ToolExecution error for invalid parameter"),
    }
}

#[tokio::test]
async fn execution_fails_for_missing_tool() {
    let toolset = toolset![get_weather];
    let missing_call = ToolCall {
        id: "missing".to_string(),
        call_id: "missing".to_string(),
        name: "nonexistent_tool".to_string(),
        arguments: json!({}),
    };

    let error = toolset.registry.execute(&missing_call).await.unwrap_err();
    match error {
        LlmError::ToolNotFound(name) => assert_eq!(name, "nonexistent_tool"),
        _ => panic!("Expected ToolNotFound error"),
    }
}

#[test]
fn tool_config_round_trip_from_toolset() {
    let toolset = toolset![get_weather, calculate];
    let schemas = toolset.tools().unwrap().into_boxed_slice();

    let tool_config = ToolConfig {
        tools: Some(schemas),
        tool_choice: Some(ToolChoice::Auto),
        parallel_tool_calls: Some(true),
    };

    assert!(tool_config.tools.is_some());
    assert_eq!(tool_config.tools.as_ref().unwrap().len(), 2);
    assert_eq!(tool_config.parallel_tool_calls, Some(true));
}

#[test]
fn provider_configs_share_default_tool_calling_limits() {
    let openai = OpenAiConfig::new("openai-key".to_string());
    let openrouter = OpenRouterConfig::new("router-key".to_string());

    let openai_guard = openai.get_tool_calling_guard();
    let openrouter_guard = openrouter.get_tool_calling_guard();
    assert_eq!(openai_guard.max_iterations, 50);
    assert_eq!(openai_guard.timeout, Duration::from_secs(300));
    assert_eq!(openai_guard.max_tool_calls_per_turn, 8);
    assert_eq!(openai_guard.max_concurrent_tool_calls, 4);
    assert_eq!(openai_guard.tool_timeout, Duration::from_secs(30));
    assert_eq!(openai_guard.max_iterations, openrouter_guard.max_iterations);
    assert_eq!(openai_guard.timeout, openrouter_guard.timeout);
    assert_eq!(
        openai_guard.max_tool_calls_per_turn,
        openrouter_guard.max_tool_calls_per_turn
    );
    assert_eq!(
        openai_guard.max_concurrent_tool_calls,
        openrouter_guard.max_concurrent_tool_calls
    );
    assert_eq!(openai_guard.tool_timeout, openrouter_guard.tool_timeout);

    let custom_config = ToolCallingConfig::new(10, Duration::from_secs(60))
        .with_max_tool_calls_per_turn(5)
        .with_max_concurrent_tool_calls(2)
        .with_tool_timeout(Duration::from_secs(20));
    let custom_guard = openai
        .with_tool_calling_config(custom_config.clone())
        .get_tool_calling_guard();
    assert_eq!(custom_guard.max_iterations, 10);
    assert_eq!(custom_guard.timeout, Duration::from_secs(60));
    assert_eq!(custom_guard.max_tool_calls_per_turn, 5);
    assert_eq!(custom_guard.max_concurrent_tool_calls, 2);
    assert_eq!(custom_guard.tool_timeout, Duration::from_secs(20));

    let custom_router_guard = openrouter
        .with_tool_calling_config(custom_config)
        .get_tool_calling_guard();
    assert_eq!(custom_router_guard.max_iterations, 10);
    assert_eq!(custom_router_guard.timeout, Duration::from_secs(60));
    assert_eq!(custom_router_guard.max_tool_calls_per_turn, 5);
    assert_eq!(custom_router_guard.max_concurrent_tool_calls, 2);
    assert_eq!(custom_router_guard.tool_timeout, Duration::from_secs(20));
}

#[test]
fn provider_clients_reject_http_base_urls_by_default() {
    assert_insecure_base_url_error(
        OpenAiClient::new("test-key".to_string())
            .expect("default OpenAI client")
            .with_base_url("http://localhost:8080/v1".to_string()),
    );
    assert_insecure_base_url_error(
        OpenRouterClient::new("test-key".to_string())
            .expect("default OpenRouter client")
            .with_base_url("http://localhost:8080/api/v1".to_string()),
    );
    assert_insecure_base_url_error(
        GeminiClient::new("test-key".to_string())
            .expect("default Gemini client")
            .with_base_url("http://localhost:8080/v1beta".to_string()),
    );
}

#[test]
fn provider_clients_allow_http_base_urls_with_explicit_opt_out() {
    OpenAiClient::new("test-key".to_string())
        .expect("default OpenAI client")
        .with_insecure_base_url("http://localhost:8080/v1".to_string())
        .expect("explicit OpenAI HTTP opt-out");
    OpenRouterClient::new("test-key".to_string())
        .expect("default OpenRouter client")
        .with_insecure_base_url("http://localhost:8080/api/v1".to_string())
        .expect("explicit OpenRouter HTTP opt-out");
    GeminiClient::new("test-key".to_string())
        .expect("default Gemini client")
        .with_insecure_base_url("http://localhost:8080/v1beta".to_string())
        .expect("explicit Gemini HTTP opt-out");
}

fn assert_insecure_base_url_error<T>(result: Result<T, LlmError>) {
    match result {
        Err(LlmError::ProviderConfiguration(message)) => {
            assert!(message.contains("Insecure provider base URL rejected"));
        }
        Err(other) => panic!("expected insecure base URL configuration error, got {other:?}"),
        Ok(_) => panic!("expected insecure base URL configuration error"),
    }
}
