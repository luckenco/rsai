use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use rsai::{
    BoxFuture, ChatRole, CompletionTarget, ConversationMessage, GeminiClient, LlmError,
    LlmProvider, Message, StructuredRequest, TextResponse, Tool, ToolCallingConfig, ToolChoice,
    ToolConfig, ToolFunction, ToolRegistry,
};
use serde_json::{Value, json};
use wiremock::{
    Match, Mock, MockServer, Request as WiremockRequest, ResponseTemplate,
    matchers::{method, path},
};

struct TrackedTool {
    executions: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
    delay: Duration,
}

impl ToolFunction for TrackedTool {
    fn schema(&self) -> Tool {
        Tool {
            name: "tracked_tool".to_string(),
            description: Some("Track execution limits".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "integer" }
                },
                "required": ["value"]
            }),
            strict: Some(true),
        }
    }

    fn execute<'a>(
        &'a self,
        _ctx: &'a (),
        params: Value,
    ) -> BoxFuture<'a, Result<Value, LlmError>> {
        Box::pin(async move {
            self.executions.fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(json!({ "ok": true, "value": params["value"].clone() }))
        })
    }
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
async fn gemini_parallel_tool_calls_respect_concurrency_limit() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .and(BodyNotContains("functionResponse"))
        .respond_with(gemini_tool_call_response(vec![
            gemini_function_call(1),
            gemini_function_call(2),
            gemini_function_call(3),
            gemini_function_call(4),
        ]))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .and(BodyContains("functionResponse"))
        .respond_with(gemini_final_response("done"))
        .mount(&server)
        .await;

    let (registry, executions, max_active) = tracked_tool_registry(Duration::from_millis(50));
    let request = build_request(tool_config_for_registry(&registry, Some(true)));
    let guard_config = ToolCallingConfig::new(3, Duration::from_secs(5))
        .with_max_concurrent_tool_calls(2)
        .with_tool_timeout(Duration::from_secs(1));
    let client = client_for(&server, Some(guard_config));

    let response = client
        .generate_completion::<TextResponse, ()>(
            request,
            TextResponse::format().expect("format"),
            Some(&registry),
        )
        .await
        .expect("text response");

    assert_eq!(response.text, "done");
    assert_eq!(executions.load(Ordering::SeqCst), 4);
    assert_eq!(max_active.load(Ordering::SeqCst), 2);

    let requests = server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 2);

    let second_body = parse_body(&requests[1]);
    let contents = second_body["contents"].as_array().expect("contents");
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(
        contents[1]["parts"].as_array().expect("model parts").len(),
        4
    );
    assert_eq!(contents[2]["role"], "user");
    assert_eq!(
        contents[2]["parts"].as_array().expect("user parts").len(),
        4
    );
}

#[tokio::test]
async fn gemini_parallel_tool_calls_false_executes_batch_sequentially() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .and(BodyNotContains("functionResponse"))
        .respond_with(gemini_tool_call_response(vec![
            gemini_function_call(1),
            gemini_function_call(2),
            gemini_function_call(3),
        ]))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .and(BodyContains("functionResponse"))
        .respond_with(gemini_final_response("done"))
        .mount(&server)
        .await;

    let (registry, executions, max_active) = tracked_tool_registry(Duration::from_millis(50));
    let request = build_request(tool_config_for_registry(&registry, Some(false)));
    let guard_config = ToolCallingConfig::new(3, Duration::from_secs(5))
        .with_max_concurrent_tool_calls(3)
        .with_tool_timeout(Duration::from_secs(1));
    let client = client_for(&server, Some(guard_config));

    let response = client
        .generate_completion::<TextResponse, ()>(
            request,
            TextResponse::format().expect("format"),
            Some(&registry),
        )
        .await
        .expect("text response");

    assert_eq!(response.text, "done");
    assert_eq!(executions.load(Ordering::SeqCst), 3);
    assert_eq!(max_active.load(Ordering::SeqCst), 1);

    let requests = server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 2);

    let first_body = String::from_utf8(requests[0].body.clone()).expect("utf8 body");
    assert!(first_body.contains("functionCallingConfig"));
    assert!(!first_body.contains("parallelToolCalls"));
    assert!(!first_body.contains("parallel_tool_calls"));

    let second_body = parse_body(&requests[1]);
    let contents = second_body["contents"].as_array().expect("contents");
    assert_eq!(
        contents[1]["parts"].as_array().expect("model parts").len(),
        3
    );
    assert_eq!(
        contents[2]["parts"].as_array().expect("user parts").len(),
        3
    );
}

#[tokio::test]
async fn gemini_tool_execution_timeout_triggers_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .respond_with(gemini_tool_call_response(vec![gemini_function_call(1)]))
        .mount(&server)
        .await;

    let (registry, executions, _) = tracked_tool_registry(Duration::from_millis(200));
    let request = build_request(tool_config_for_registry(&registry, Some(false)));
    let guard_config = ToolCallingConfig::new(3, Duration::from_secs(5))
        .with_tool_timeout(Duration::from_millis(50));
    let client = client_for(&server, Some(guard_config.clone()));

    let err = client
        .generate_completion::<TextResponse, ()>(
            request,
            TextResponse::format().expect("format"),
            Some(&registry),
        )
        .await
        .expect_err("tool timeout should trip");

    match err {
        LlmError::ToolExecutionTimeout { tool_name, timeout } => {
            assert_eq!(tool_name, "tracked_tool");
            assert_eq!(timeout, guard_config.tool_timeout);
        }
        other => panic!("expected ToolExecutionTimeout, got {other:?}"),
    }

    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn gemini_guard_rejects_too_many_tool_calls_before_execution() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/mock-model:generateContent"))
        .respond_with(gemini_tool_call_response(vec![
            gemini_function_call(1),
            gemini_function_call(2),
        ]))
        .mount(&server)
        .await;

    let (registry, executions, _) = tracked_tool_registry(Duration::ZERO);
    let request = build_request(tool_config_for_registry(&registry, Some(true)));
    let guard_config =
        ToolCallingConfig::new(3, Duration::from_secs(5)).with_max_tool_calls_per_turn(1);
    let client = client_for(&server, Some(guard_config));

    let err = client
        .generate_completion::<TextResponse, ()>(
            request,
            TextResponse::format().expect("format"),
            Some(&registry),
        )
        .await
        .expect_err("tool call limit should trip");

    match err {
        LlmError::ToolCallLimit { requested, limit } => {
            assert_eq!(requested, 2);
            assert_eq!(limit, 1);
        }
        other => panic!("expected ToolCallLimit, got {other:?}"),
    }

    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

fn client_for(server: &MockServer, config: Option<ToolCallingConfig>) -> GeminiClient {
    let client = GeminiClient::new("test-key".to_string())
        .unwrap()
        .with_insecure_base_url(server.uri())
        .unwrap();

    if let Some(cfg) = config {
        client.with_tool_calling_config(cfg).unwrap()
    } else {
        client
    }
}

fn tool_config_for_registry(registry: &ToolRegistry, parallel: Option<bool>) -> ToolConfig {
    ToolConfig {
        tools: Some(registry.get_schemas().expect("schemas").into_boxed_slice()),
        tool_choice: Some(ToolChoice::Auto),
        parallel_tool_calls: parallel,
    }
}

fn tracked_tool_registry(delay: Duration) -> (ToolRegistry, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let executions = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let registry = ToolRegistry::new();

    registry
        .register(Arc::new(TrackedTool {
            executions: executions.clone(),
            active,
            max_active: max_active.clone(),
            delay,
        }))
        .expect("tracked tool registration");

    (registry, executions, max_active)
}

fn build_request(tool_config: ToolConfig) -> StructuredRequest {
    StructuredRequest {
        model: "mock-model".to_string(),
        messages: vec![ConversationMessage::Chat(Message {
            role: ChatRole::User,
            content: "call tools".to_string(),
        })],
        tool_config: Some(tool_config),
        generation_config: None,
    }
}

fn gemini_tool_call_response(parts: Vec<Value>) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts,
            },
        }],
        "usageMetadata": usage_metadata(),
        "modelVersion": "mock-model",
    }))
}

fn gemini_function_call(value: i64) -> Value {
    json!({
        "functionCall": {
            "name": "tracked_tool",
            "args": { "value": value },
        }
    })
}

fn gemini_final_response(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{ "text": text }],
            },
        }],
        "usageMetadata": usage_metadata(),
        "modelVersion": "mock-model",
    }))
}

fn usage_metadata() -> Value {
    json!({
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    })
}

fn parse_body(request: &WiremockRequest) -> Value {
    serde_json::from_slice(&request.body).expect("json body")
}
