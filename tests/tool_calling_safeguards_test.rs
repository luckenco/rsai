use rsai::{LlmError, OpenAiConfig, OpenRouterConfig, ToolCallingConfig, ToolCallingGuard};
use std::time::Duration;

#[tokio::test]
async fn test_iteration_limit() {
    // Create a guard with very low iteration limit
    let mut guard = ToolCallingGuard::with_limits(3, Duration::from_secs(10));

    // Simulate multiple iterations
    assert!(guard.increment_iteration().is_ok());
    assert_eq!(guard.current_iteration(), 1);

    assert!(guard.increment_iteration().is_ok());
    assert_eq!(guard.current_iteration(), 2);

    assert!(guard.increment_iteration().is_ok());
    assert_eq!(guard.current_iteration(), 3);

    // Fourth iteration should fail
    let result = guard.increment_iteration();
    assert!(result.is_err());

    if let Err(LlmError::ToolCallIterationLimit { limit }) = result {
        assert_eq!(limit, 3);
    } else {
        panic!("Expected ToolCallIterationLimit error");
    }
}

#[tokio::test]
async fn test_default_guard_values() {
    let guard = ToolCallingGuard::new();

    assert_eq!(guard.max_iterations, 50);
    assert_eq!(guard.timeout, Duration::from_secs(300));
    assert_eq!(guard.current_iteration(), 0);
}

#[tokio::test]
async fn test_custom_guard_values() {
    let guard = ToolCallingGuard::with_limits(100, Duration::from_secs(600));

    assert_eq!(guard.max_iterations, 100);
    assert_eq!(guard.timeout, Duration::from_secs(600));
}

#[tokio::test]
async fn test_timeout_error() {
    use tokio::time::{Duration as TokioDuration, sleep};

    // Create a guard with very short timeout
    let guard = ToolCallingGuard::with_limits(1000, Duration::from_millis(100));

    // Simulate a timeout scenario
    let start = std::time::Instant::now();
    let timeout_result = tokio::time::timeout(guard.timeout, async {
        // Simulate long-running operation
        sleep(TokioDuration::from_millis(200)).await;
        Ok::<(), LlmError>(())
    })
    .await;

    let elapsed = start.elapsed();

    // Should timeout
    assert!(timeout_result.is_err());
    assert!(elapsed < Duration::from_millis(150)); // Should timeout quickly
}

// Test that demonstrates the guard prevents infinite loops
#[tokio::test]
async fn test_guard_prevents_infinite_loop() {
    let mut guard = ToolCallingGuard::with_limits(5, Duration::from_secs(1));

    // Simulate what would be an infinite loop
    let mut iterations = 0;
    let result = async {
        loop {
            guard.increment_iteration()?;
            iterations += 1;

            // Simulate some work
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        #[allow(unreachable_code)]
        Ok::<(), LlmError>(())
    }
    .await;

    // Should fail with iteration limit
    assert!(result.is_err());
    assert_eq!(iterations, 5); // Should stop at the limit

    if let Err(LlmError::ToolCallIterationLimit { limit }) = result {
        assert_eq!(limit, 5);
    } else {
        panic!("Expected ToolCallIterationLimit error");
    }
}

#[tokio::test]
async fn test_openai_config_tool_calling() {
    let config = OpenAiConfig::new("test-key".to_string());
    let guard = config.get_tool_calling_guard();

    // Should have default values
    assert_eq!(guard.max_iterations, 50);
    assert_eq!(guard.timeout, Duration::from_secs(300));
    assert_eq!(guard.max_tool_calls_per_turn, 8);
    assert_eq!(guard.max_concurrent_tool_calls, 4);
    assert_eq!(guard.tool_timeout, Duration::from_secs(30));

    // Test custom config
    let custom_config = ToolCallingConfig::new(75, Duration::from_secs(600))
        .with_max_tool_calls_per_turn(6)
        .with_max_concurrent_tool_calls(2)
        .with_tool_timeout(Duration::from_secs(15));

    let config_with_custom =
        OpenAiConfig::new("test-key".to_string()).with_tool_calling_config(custom_config);
    let custom_guard = config_with_custom.get_tool_calling_guard();

    assert_eq!(custom_guard.max_iterations, 75);
    assert_eq!(custom_guard.timeout, Duration::from_secs(600));
    assert_eq!(custom_guard.max_tool_calls_per_turn, 6);
    assert_eq!(custom_guard.max_concurrent_tool_calls, 2);
    assert_eq!(custom_guard.tool_timeout, Duration::from_secs(15));
}

#[tokio::test]
async fn test_openrouter_config_tool_calling() {
    let config = OpenRouterConfig::new("test-key".to_string());
    let guard = config.get_tool_calling_guard();

    // Should have default values
    assert_eq!(guard.max_iterations, 50);
    assert_eq!(guard.timeout, Duration::from_secs(300));
    assert_eq!(guard.max_tool_calls_per_turn, 8);
    assert_eq!(guard.max_concurrent_tool_calls, 4);
    assert_eq!(guard.tool_timeout, Duration::from_secs(30));

    // Test custom config
    let custom_config = ToolCallingConfig::new(100, Duration::from_secs(900))
        .with_max_tool_calls_per_turn(7)
        .with_max_concurrent_tool_calls(3)
        .with_tool_timeout(Duration::from_secs(20));

    let config_with_custom =
        OpenRouterConfig::new("test-key".to_string()).with_tool_calling_config(custom_config);
    let custom_guard = config_with_custom.get_tool_calling_guard();

    assert_eq!(custom_guard.max_iterations, 100);
    assert_eq!(custom_guard.timeout, Duration::from_secs(900));
    assert_eq!(custom_guard.max_tool_calls_per_turn, 7);
    assert_eq!(custom_guard.max_concurrent_tool_calls, 3);
    assert_eq!(custom_guard.tool_timeout, Duration::from_secs(20));
}
