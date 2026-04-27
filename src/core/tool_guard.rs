use crate::core::LlmError;
use std::time::Duration;

/// Configuration for tool calling behavior and limits
#[derive(Debug, Clone)]
pub struct ToolCallingConfig {
    /// Maximum number of iterations in tool calling loop (default: 50)
    pub max_iterations: u32,
    /// Timeout for tool calling loop (default: 5 minutes)
    pub timeout: Duration,
    /// Maximum number of tool calls accepted from one model response (default: 8)
    pub max_tool_calls_per_turn: usize,
    /// Maximum number of tools executed at the same time (default: 4)
    pub max_concurrent_tool_calls: usize,
    /// Timeout for each individual tool execution (default: 30 seconds).
    ///
    /// Generated synchronous `#[tool]` functions are run on Tokio's blocking pool so this timeout
    /// can fire while they are blocked. Async tools and manual `ToolFunction` implementations must
    /// remain cooperative and avoid blocking Tokio worker threads.
    pub tool_timeout: Duration,
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            timeout: Duration::from_secs(300),
            max_tool_calls_per_turn: 8,
            max_concurrent_tool_calls: 4,
            tool_timeout: Duration::from_secs(30),
        }
    }
}

impl ToolCallingConfig {
    /// Create a new config with custom limits
    pub fn new(max_iterations: u32, timeout: Duration) -> Self {
        Self {
            max_iterations,
            timeout,
            ..Self::default()
        }
    }

    /// Set the maximum number of tool calls accepted from one model response.
    pub fn with_max_tool_calls_per_turn(mut self, max_tool_calls: usize) -> Self {
        self.max_tool_calls_per_turn = max_tool_calls;
        self
    }

    /// Set the maximum number of tools executed at the same time.
    ///
    /// A value of `0` is treated as `1` when executing tools.
    pub fn with_max_concurrent_tool_calls(mut self, max_concurrent_tool_calls: usize) -> Self {
        self.max_concurrent_tool_calls = max_concurrent_tool_calls;
        self
    }

    /// Set the timeout for each individual tool execution.
    ///
    /// This timeout can stop waiting for generated synchronous `#[tool]` functions while their
    /// blocking work finishes on Tokio's blocking pool. It does not preempt blocking work inside
    /// async tools or manual `ToolFunction` implementations.
    pub fn with_tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }
}

/// Guard for tracking tool call processing limits and preventing infinite loops
#[derive(Debug, Clone)]
pub struct ToolCallingGuard {
    /// Maximum number of iterations allowed in the tool calling loop
    pub max_iterations: u32,
    /// Timeout duration for the entire tool calling loop
    pub timeout: Duration,
    /// Maximum number of tool calls accepted from one model response
    pub max_tool_calls_per_turn: usize,
    /// Maximum number of tools executed at the same time
    pub max_concurrent_tool_calls: usize,
    /// Timeout duration for each individual tool execution
    pub tool_timeout: Duration,
    /// Current iteration count
    current_iteration: u32,
}

impl ToolCallingGuard {
    /// Create a new ToolCallingGuard with default limits
    pub fn new() -> Self {
        Self::from_config(&ToolCallingConfig::default())
    }

    /// Create a new ToolCallingGuard with custom limits
    pub fn with_limits(max_iterations: u32, timeout: Duration) -> Self {
        Self::from_config(&ToolCallingConfig::new(max_iterations, timeout))
    }

    /// Create a new ToolCallingGuard from a config
    pub fn from_config(config: &ToolCallingConfig) -> Self {
        Self {
            max_iterations: config.max_iterations,
            timeout: config.timeout,
            max_tool_calls_per_turn: config.max_tool_calls_per_turn,
            max_concurrent_tool_calls: config.max_concurrent_tool_calls,
            tool_timeout: config.tool_timeout,
            current_iteration: 0,
        }
    }

    /// Increment iteration count and check if limit is exceeded
    pub fn increment_iteration(&mut self) -> Result<(), LlmError> {
        self.current_iteration = self.current_iteration.saturating_add(1);
        if self.current_iteration > self.max_iterations {
            return Err(LlmError::ToolCallIterationLimit {
                limit: self.max_iterations,
            });
        }
        Ok(())
    }

    /// Check if the current model turn requested too many tool calls.
    pub fn check_tool_calls_for_turn(&self, requested: usize) -> Result<(), LlmError> {
        if requested > self.max_tool_calls_per_turn {
            return Err(LlmError::ToolCallLimit {
                requested,
                limit: self.max_tool_calls_per_turn,
            });
        }
        Ok(())
    }

    /// Get the effective concurrency limit for tool execution.
    pub fn max_concurrent_tool_calls(&self) -> usize {
        self.max_concurrent_tool_calls.max(1)
    }

    /// Get current iteration count
    pub fn current_iteration(&self) -> u32 {
        self.current_iteration
    }
}

impl Default for ToolCallingGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_calling_guard_default() {
        let guard = ToolCallingGuard::default();
        assert_eq!(guard.max_iterations, 50);
        assert_eq!(guard.timeout, Duration::from_secs(300));
        assert_eq!(guard.max_tool_calls_per_turn, 8);
        assert_eq!(guard.max_concurrent_tool_calls, 4);
        assert_eq!(guard.tool_timeout, Duration::from_secs(30));
        assert_eq!(guard.current_iteration(), 0);
    }

    #[test]
    fn test_tool_calling_guard_custom_limits() {
        let guard = ToolCallingGuard::with_limits(100, Duration::from_secs(600));
        assert_eq!(guard.max_iterations, 100);
        assert_eq!(guard.timeout, Duration::from_secs(600));
        assert_eq!(guard.max_tool_calls_per_turn, 8);
        assert_eq!(guard.max_concurrent_tool_calls, 4);
        assert_eq!(guard.tool_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_tool_calling_guard_increment() {
        let mut guard = ToolCallingGuard::with_limits(3, Duration::from_secs(300));

        assert!(guard.increment_iteration().is_ok());
        assert_eq!(guard.current_iteration(), 1);

        assert!(guard.increment_iteration().is_ok());
        assert_eq!(guard.current_iteration(), 2);

        assert!(guard.increment_iteration().is_ok());
        assert_eq!(guard.current_iteration(), 3);

        // Fourth increment should fail
        assert!(guard.increment_iteration().is_err());
    }

    #[test]
    fn test_tool_calling_config_default() {
        let config = ToolCallingConfig::default();
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.max_tool_calls_per_turn, 8);
        assert_eq!(config.max_concurrent_tool_calls, 4);
        assert_eq!(config.tool_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_tool_calling_guard_from_config() {
        let config = ToolCallingConfig::new(75, Duration::from_secs(450))
            .with_max_tool_calls_per_turn(3)
            .with_max_concurrent_tool_calls(2)
            .with_tool_timeout(Duration::from_secs(10));
        let guard = ToolCallingGuard::from_config(&config);

        assert_eq!(guard.max_iterations, 75);
        assert_eq!(guard.timeout, Duration::from_secs(450));
        assert_eq!(guard.max_tool_calls_per_turn, 3);
        assert_eq!(guard.max_concurrent_tool_calls, 2);
        assert_eq!(guard.tool_timeout, Duration::from_secs(10));
        assert_eq!(guard.current_iteration(), 0);
    }

    #[test]
    fn test_tool_calling_guard_checks_calls_per_turn() {
        let guard = ToolCallingGuard::from_config(
            &ToolCallingConfig::default().with_max_tool_calls_per_turn(2),
        );

        assert!(guard.check_tool_calls_for_turn(2).is_ok());
        let err = guard
            .check_tool_calls_for_turn(3)
            .expect_err("too many calls should fail");

        match err {
            LlmError::ToolCallLimit { requested, limit } => {
                assert_eq!(requested, 3);
                assert_eq!(limit, 2);
            }
            other => panic!("expected ToolCallLimit, got {other:?}"),
        }
    }

    #[test]
    fn test_zero_concurrency_limit_runs_as_one() {
        let guard = ToolCallingGuard::from_config(
            &ToolCallingConfig::default().with_max_concurrent_tool_calls(0),
        );

        assert_eq!(guard.max_concurrent_tool_calls(), 1);
    }
}
