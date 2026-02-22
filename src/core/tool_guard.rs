use crate::core::LlmError;
use std::time::Duration;

/// Configuration for tool calling behavior and limits
#[derive(Debug, Clone)]
pub struct ToolCallingConfig {
    /// Maximum number of iterations in tool calling loop (default: 50)
    pub max_iterations: u32,
    /// Timeout for tool calling loop (default: 5 minutes)
    pub timeout: Duration,
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            timeout: Duration::from_secs(300),
        }
    }
}

impl ToolCallingConfig {
    /// Create a new config with custom limits
    pub fn new(max_iterations: u32, timeout: Duration) -> Self {
        Self {
            max_iterations,
            timeout,
        }
    }
}

/// Guard for tracking tool call processing limits and preventing infinite loops
#[derive(Debug, Clone)]
pub struct ToolCallingGuard {
    /// Maximum number of iterations allowed in the tool calling loop
    pub max_iterations: u32,
    /// Timeout duration for the entire tool calling loop
    pub timeout: Duration,
    /// Current iteration count
    current_iteration: u32,
}

impl ToolCallingGuard {
    /// Create a new ToolCallingGuard with default limits
    pub fn new() -> Self {
        Self {
            max_iterations: 50,
            timeout: Duration::from_secs(300), // 5 minutes default
            current_iteration: 0,
        }
    }

    /// Create a new ToolCallingGuard with custom limits
    pub fn with_limits(max_iterations: u32, timeout: Duration) -> Self {
        Self {
            max_iterations,
            timeout,
            current_iteration: 0,
        }
    }

    /// Create a new ToolCallingGuard from a config
    pub fn from_config(config: &ToolCallingConfig) -> Self {
        Self::with_limits(config.max_iterations, config.timeout)
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
        assert_eq!(guard.current_iteration(), 0);
    }

    #[test]
    fn test_tool_calling_guard_custom_limits() {
        let guard = ToolCallingGuard::with_limits(100, Duration::from_secs(600));
        assert_eq!(guard.max_iterations, 100);
        assert_eq!(guard.timeout, Duration::from_secs(600));
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
    }

    #[test]
    fn test_tool_calling_guard_from_config() {
        let config = ToolCallingConfig::new(75, Duration::from_secs(450));
        let guard = ToolCallingGuard::from_config(&config);

        assert_eq!(guard.max_iterations, 75);
        assert_eq!(guard.timeout, Duration::from_secs(450));
        assert_eq!(guard.current_iteration(), 0);
    }
}
