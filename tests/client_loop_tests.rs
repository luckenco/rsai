use std::time::Duration;

use rsai::{
    ChatRole, CompletionTarget, ConversationMessage, LlmError, LlmProvider, Message, OpenAiClient,
    StructuredRequest, ToolCallingConfig, ToolChoice, ToolConfig, ToolSet, completion_schema, tool,
    toolset,
};
use serde_json::{Value, json};
use wiremock::{
    Match, Mock, MockServer, Request as WiremockRequest, ResponseTemplate,
    matchers::{method, path},
};

#[completion_schema]
#[derive(Debug, serde::Serialize)]
struct SumResponse {
    sum: i64,
}

#[derive(Debug, serde::Serialize)]
struct MultiplyResponse {
    product: i64,
}

#[tool]
/// Add two integers and return the sum.
/// a: First addend.
/// b: Second addend.
fn calculate_sum(a: i64, b: i64) -> SumResponse {
    SumResponse { sum: a + b }
}

#[tool]
/// Multiply two integers.
/// a: First factor.
/// b: Second factor.
fn multiply_values(a: i64, b: i64) -> MultiplyResponse {
    MultiplyResponse { product: a * b }
}

#[derive(Clone)]
struct BodyContains(&'static str);

impl Match for BodyContains {
    fn matches(&self, request: &WiremockRequest) -> bool {
        std::str::from_utf8(&request.body)
            .map(|body| body.contains(self.0))
            .unwrap_or(false)
    }
}

#[derive(Clone)]
struct BodyNotContains(&'static str);

impl Match for BodyNotContains {
    fn matches(&self, request: &WiremockRequest) -> bool {
        !BodyContains(self.0).matches(request)
    }
}

#[tokio::test]
async fn sequential_tool_call_flow_appends_history() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(BodyNotContains("function_call_output"))
        .respond_with(tool_call_response(vec![function_call(
            "call_sum",
            "calculate_sum",
            json!({ "a": 1, "b": 2 }),
        )]))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(BodyContains("function_call_output"))
        .respond_with(final_response(json!({ "sum": 3 })))
        .mount(&server)
        .await;

    let toolset = sum_toolset();
    let tool_config = tool_config_for(&toolset, Some(false));
    let request = build_request("Add 1 and 2", tool_config);

    let client = client_for(&server, None);
    let response = client
        .generate_completion::<SumResponse, ()>(
            request,
            <SumResponse as CompletionTarget>::format().expect("format"),
            Some(&toolset.registry),
        )
        .await
        .expect("structured response");
    assert_eq!(response.content.sum, 3);

    let requests = server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 2);

    let first_input = parse_inputs(&requests[0]);
    assert_eq!(first_input.len(), 1);
    assert_eq!(first_input[0]["role"], "user");
    assert_eq!(first_input[0]["content"], "Add 1 and 2");

    let second_input = parse_inputs(&requests[1]);
    assert_eq!(second_input.len(), 3);
    assert_eq!(second_input[1]["type"], "function_call");
    assert_eq!(second_input[1]["name"], "calculate_sum");
    assert_eq!(second_input[2]["type"], "function_call_output");
    assert_eq!(second_input[2]["output"]["sum"], 3);
}

#[tokio::test]
async fn parallel_tool_calls_submit_all_results_together() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(BodyNotContains("function_call_output"))
        .respond_with(tool_call_response(vec![
            function_call("call_sum", "calculate_sum", json!({ "a": 4, "b": 5 })),
            function_call("call_product", "multiply_values", json!({ "a": 2, "b": 3 })),
        ]))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(BodyContains("function_call_output"))
        .respond_with(final_response(json!({ "sum": 15 })))
        .mount(&server)
        .await;

    let toolset = sum_and_product_toolset();
    let tool_config = tool_config_for(&toolset, Some(true));
    let request = build_request("Combine results", tool_config);

    let client = client_for(&server, None);
    let response = client
        .generate_completion::<SumResponse, ()>(
            request,
            <SumResponse as CompletionTarget>::format().expect("format"),
            Some(&toolset.registry),
        )
        .await
        .expect("structured response");
    assert_eq!(response.content.sum, 15);

    let requests = server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 2);

    let second_input = parse_inputs(&requests[1]);
    assert_eq!(
        second_input
            .iter()
            .filter(|item| item["type"] == "function_call")
            .count(),
        2
    );
    assert_eq!(
        second_input
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .count(),
        2
    );

    // Ensure outputs were appended only after both calls were queued
    let first_output_index = second_input
        .iter()
        .position(|item| item["type"] == "function_call_output")
        .expect("output index");
    assert_eq!(first_output_index, 3);

    assert_eq!(second_input[1]["name"], "calculate_sum");
    assert_eq!(second_input[3]["output"]["sum"], 9);
    assert_eq!(second_input[2]["name"], "multiply_values");
    assert_eq!(second_input[4]["output"]["product"], 6);
}

#[tokio::test]
async fn guard_stops_iteration_after_max_limit() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(tool_call_response(vec![function_call(
            "call_sum",
            "calculate_sum",
            json!({ "a": 1, "b": 2 }),
        )]))
        .mount(&server)
        .await;

    let toolset = sum_toolset();
    let tool_config = tool_config_for(&toolset, Some(false));
    let request = build_request("Need repeated calls", tool_config);

    let timeout = Duration::from_secs(5);
    let guard_config = ToolCallingConfig::new(1, timeout);
    let client = client_for(&server, Some(guard_config));
    let err = client
        .generate_completion::<SumResponse, ()>(
            request,
            <SumResponse as CompletionTarget>::format().expect("format"),
            Some(&toolset.registry),
        )
        .await
        .expect_err("iteration guard should trip");

    match err {
        LlmError::ToolCallIterationLimit { limit } => assert_eq!(limit, 1),
        other => panic!("expected ToolCallIterationLimit, got {other:?}"),
    }

    let requests = server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn tool_call_timeout_triggers_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(final_response(json!({ "sum": 42 })).set_delay(Duration::from_millis(200)))
        .mount(&server)
        .await;

    let toolset = sum_toolset();
    let tool_config = tool_config_for(&toolset, Some(false));
    let request = build_request("slow request", tool_config);

    let guard_config = ToolCallingConfig::new(3, Duration::from_millis(50));
    let client = client_for(&server, Some(guard_config.clone()));
    let err = client
        .generate_completion::<SumResponse, ()>(
            request,
            <SumResponse as CompletionTarget>::format().expect("format"),
            Some(&toolset.registry),
        )
        .await
        .expect_err("timeout should trigger");

    match err {
        LlmError::ToolCallTimeout { timeout } => assert_eq!(timeout, guard_config.timeout),
        other => panic!("expected ToolCallTimeout, got {other:?}"),
    }
}

fn client_for(server: &MockServer, config: Option<ToolCallingConfig>) -> OpenAiClient {
    let base_url = format!("{}/v1", server.uri());
    let client = OpenAiClient::new("test-key".to_string())
        .unwrap()
        .with_insecure_base_url(base_url)
        .unwrap();

    if let Some(cfg) = config {
        client.with_tool_calling_config(cfg).unwrap()
    } else {
        client
    }
}

fn tool_config_for(toolset: &ToolSet, parallel: Option<bool>) -> ToolConfig {
    ToolConfig {
        tools: Some(toolset.tools().expect("schemas").into_boxed_slice()),
        tool_choice: Some(ToolChoice::Auto),
        parallel_tool_calls: parallel,
    }
}

fn sum_toolset() -> ToolSet {
    toolset![calculate_sum]
}

fn sum_and_product_toolset() -> ToolSet {
    toolset![calculate_sum, multiply_values]
}

fn build_request(prompt: &str, tool_config: ToolConfig) -> StructuredRequest {
    StructuredRequest {
        model: "mock-model".to_string(),
        messages: vec![ConversationMessage::Chat(Message {
            role: ChatRole::User,
            content: prompt.to_string(),
        })],
        tool_config: Some(tool_config),
        generation_config: None,
    }
}

fn tool_call_response(function_calls: Vec<Value>) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "mock-response",
        "model": "mock-model",
        "output": function_calls,
        "usage": usage_payload(),
    }))
}

fn function_call(call_id: &str, name: &str, arguments: Value) -> Value {
    json!({
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments.to_string(),
    })
}

fn final_response(body: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "mock-final",
        "model": "mock-model",
        "output": [{
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": body.to_string()
            }]
        }],
        "usage": usage_payload()
    }))
}

fn usage_payload() -> Value {
    json!({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15
    })
}

fn parse_inputs(request: &WiremockRequest) -> Vec<Value> {
    let body: Value =
        serde_json::from_slice(&request.body).expect("request body should be valid json");
    body["input"].as_array().expect("input array").clone()
}
