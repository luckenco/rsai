use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    CompletionTarget,
    core::{LlmError, StructuredRequest, ToolCall, ToolCallingGuard, ToolRegistry},
};

use super::{
    client::{ResponsesClient, ResponsesProviderConfig},
    request::{Format, FunctionToolCallOutput, InputItem},
    request_builder::convert_messages_to_responses_format,
    response::{OutputContent, Response},
    response_parser::convert_to_provider_response,
    types::FunctionToolCall,
};

impl<P: ResponsesProviderConfig> ResponsesClient<P> {
    /// Handle the complete tool calling loop until a final response is received.
    pub async fn handle_tool_calling_loop<T, Ctx>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget,
        Ctx: Send + Sync + 'static,
    {
        let timeout = guard.timeout;
        let result =
            self.handle_tool_calling_loop_internal::<T, Ctx>(request, tool_registry, guard, format);

        tokio::time::timeout(timeout, result)
            .await
            .map_err(|_| LlmError::ToolCallTimeout { timeout })?
    }

    #[tracing::instrument(
        name = "tool_calling_loop",
        level = "debug",
        skip(self, request, tool_registry, guard),
        fields(
            model = %request.model,
            max_iterations = %guard.max_iterations
        ),
        err
    )]
    async fn handle_tool_calling_loop_internal<T, Ctx>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &mut ToolCallingGuard,
        format: Format,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget,
        Ctx: Send + Sync + 'static,
    {
        let mut input = convert_messages_to_responses_format(request.messages.clone())?;
        let parallel = request
            .tool_config
            .as_ref()
            .and_then(|config| config.parallel_tool_calls)
            .unwrap_or(true);

        loop {
            guard.increment_iteration()?;

            let span =
                tracing::debug_span!("tool_loop_iteration", iteration = guard.current_iteration());
            let _entered = span.enter();

            let api_request = self.build_request_with_format(&request, &input, format.clone())?;
            let response = self.make_api_request(api_request).await?;
            let function_calls = self.extract_function_calls(&response);

            if function_calls.is_empty() {
                tracing::debug!("No more tool calls, returning final response");
                let response = convert_to_provider_response(response, self.config.provider())?;
                return T::parse_response(response);
            }

            tracing::info!(
                count = function_calls.len(),
                "Model requested tool execution"
            );
            guard.check_tool_calls_for_turn(function_calls.len())?;
            self.process_function_calls(
                &function_calls,
                &mut input,
                tool_registry,
                parallel,
                guard,
            )
            .await?;
        }
    }

    /// Extract function calls from an API response.
    pub fn extract_function_calls<'a>(&self, response: &'a Response) -> Vec<&'a FunctionToolCall> {
        response
            .output
            .iter()
            .filter_map(|output| match output {
                OutputContent::FunctionCall(call) => Some(call),
                OutputContent::OutputMessage(_) => None,
            })
            .collect()
    }

    /// Process function calls either in parallel or sequentially.
    pub async fn process_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        parallel: bool,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        if parallel && function_calls.len() > 1 {
            self.process_parallel_function_calls(function_calls, input, tool_registry, guard)
                .await
        } else {
            self.process_sequential_function_calls(function_calls, input, tool_registry, guard)
                .await
        }
    }

    /// Add all calls, execute them concurrently, then add their results in call order.
    ///
    /// Completed tools may have side effects even when another tool fails.
    pub async fn process_parallel_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        let mut tool_calls = Vec::with_capacity(function_calls.len());
        for function_call in function_calls {
            input.push(InputItem::FunctionCall((*function_call).clone()));
            tool_calls.push(ToolCall {
                id: function_call.id.clone(),
                call_id: function_call.call_id.clone(),
                name: function_call.name.clone(),
                arguments: self.parse_function_arguments(&function_call.arguments)?,
            });
        }

        let timeout = guard.tool_timeout;
        let results: Vec<serde_json::Value> = stream::iter(tool_calls.iter().cloned())
            .map(|tool_call| async move {
                self.execute_tool_with_timeout(tool_registry, &tool_call, timeout)
                    .await
            })
            .buffered(guard.max_concurrent_tool_calls())
            .try_collect()
            .await?;

        for (tool_call, result) in tool_calls.iter().zip(results) {
            input.push(function_call_output(tool_call.call_id.clone(), result));
        }

        Ok(())
    }

    /// Process function calls and append each result before starting the next call.
    pub async fn process_sequential_function_calls<Ctx>(
        &self,
        function_calls: &[&FunctionToolCall],
        input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry<Ctx>,
        guard: &ToolCallingGuard,
    ) -> Result<(), LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        for function_call in function_calls {
            input.push(InputItem::FunctionCall((*function_call).clone()));

            let tool_call = ToolCall {
                id: function_call.id.clone(),
                call_id: function_call.call_id.clone(),
                name: function_call.name.clone(),
                arguments: self.parse_function_arguments(&function_call.arguments)?,
            };
            let result = self
                .execute_tool_with_timeout(tool_registry, &tool_call, guard.tool_timeout)
                .await?;
            input.push(function_call_output(function_call.call_id.clone(), result));
        }

        Ok(())
    }

    async fn execute_tool_with_timeout<Ctx>(
        &self,
        tool_registry: &ToolRegistry<Ctx>,
        tool_call: &ToolCall,
        timeout: std::time::Duration,
    ) -> Result<serde_json::Value, LlmError>
    where
        Ctx: Send + Sync + 'static,
    {
        tokio::time::timeout(timeout, tool_registry.execute(tool_call))
            .await
            .map_err(|_| LlmError::ToolExecutionTimeout {
                tool_name: tool_call.name.clone(),
                timeout,
            })?
    }

    /// Generate a completion, including automatic tool execution when configured.
    pub async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
        mut guard: ToolCallingGuard,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget + Send,
        Ctx: Send + Sync + 'static,
    {
        let has_tools = request
            .tool_config
            .as_ref()
            .and_then(|config| config.tools.as_ref())
            .is_some();

        if has_tools && let Some(tool_registry) = tool_registry {
            return self
                .handle_tool_calling_loop::<T, Ctx>(request, tool_registry, &mut guard, format)
                .await;
        }

        let input = convert_messages_to_responses_format(request.messages.clone())?;
        let api_request = self.build_request_with_format(&request, &input, format)?;
        let response = self.make_api_request(api_request).await?;
        let response = convert_to_provider_response(response, self.config.provider())?;
        T::parse_response(response)
    }

    /// Parse function arguments from the API representation.
    pub fn parse_function_arguments(
        &self,
        arguments: &serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        match arguments {
            serde_json::Value::String(arguments) => {
                serde_json::from_str(arguments).map_err(|error| LlmError::Parse {
                    message: "Failed to parse tool arguments".to_string(),
                    source: Box::new(error),
                })
            }
            arguments => Ok(arguments.clone()),
        }
    }
}

fn function_call_output(call_id: String, output: serde_json::Value) -> InputItem {
    InputItem::FunctionCallOutput(FunctionToolCallOutput {
        call_id,
        output,
        r#type: "function_call_output".to_string(),
    })
}
