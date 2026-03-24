#![deny(unsafe_code)]
#![deny(clippy::all)]
#![deny(unreachable_pub)]
#![warn(missing_docs)]

//! Native OpenAI LLM provider capsule.
//!
//! Talks directly to OpenAI's Chat Completions API with support for
//! OpenAI-specific features not available through generic compat layers:
//!
//! - **Structured outputs** — `response_format: { type: "json_schema", ... }`
//!   with `strict: true` for guaranteed schema adherence
//! - **Strict function calling** — `strict: true` on tool definitions ensures
//!   the model always follows the declared parameter schema
//! - **Reasoning effort** — `reasoning_effort` for o-series models (low/medium/high)
//! - **Service tier** — `auto`/`default`/`flex`/`priority` routing
//! - **Parallel tool calls** — explicit `parallel_tool_calls` control
//! - **Predicted output** — `prediction` for faster generation on known-structure responses
//! - **`max_completion_tokens`** — OpenAI's preferred field (distinct from generic `max_tokens`)
//!
//! For generic OpenAI-compatible providers (Groq, Together, Mistral, etc.),
//! use `astrid-capsule-openai-compat` instead.

mod schemas;

use astrid_sdk::prelude::*;
use astrid_sdk::types::{IpcPayload, Message, MessageContent, MessageRole, StreamEvent};
use schemas::ChatCompletionChunk;
use serde_json::Value;
use uuid::Uuid;

const STREAM_TOPIC: &str = "llm.v1.stream.openai";
const BASE_URL: &str = "https://api.openai.com";
/// Maximum SSE line buffer size (1 MB).
const MAX_LINE_BUFFER_SIZE: usize = 1024 * 1024;

/// Native OpenAI LLM provider capsule.
#[derive(Default)]
pub struct OpenAIProvider;

#[capsule]
impl OpenAIProvider {
    /// Handles incoming LLM generation requests.
    #[astrid::interceptor("handle_llm_request")]
    pub fn handle_llm_request(&self, req: IpcPayload) -> Result<(), SysError> {
        if let IpcPayload::LlmRequest {
            request_id,
            model,
            messages,
            tools,
            system,
            ..
        } = req
            && let Err(e) = Self::execute_request(request_id, &model, &messages, &tools, &system)
        {
            let _ = log::error(format!("OpenAI request failed: {e}"));
            let _ = ipc::publish_json(
                STREAM_TOPIC,
                &IpcPayload::LlmStreamEvent {
                    request_id,
                    event: StreamEvent::Error(e.to_string()),
                },
            );
        }
        Ok(())
    }

    /// Returns provider metadata for IPC-based provider discovery.
    #[astrid::interceptor("llm_describe")]
    pub fn llm_describe(&self, _payload: serde_json::Value) -> Result<serde_json::Value, SysError> {
        let model = env::var("model").unwrap_or_else(|_| "gpt-4.1".into());
        let context_window = env::var("context_window")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(128_000);
        let max_output = env::var("max_output_tokens")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(8_192);
        Ok(serde_json::json!({
            "providers": [{
                "id": "openai",
                "description": format!("OpenAI (default model: {model})"),
                "capabilities": ["text", "vision", "tools", "structured_output", "reasoning"],
                "request_topic": "llm.v1.request.generate.openai",
                "stream_topic": STREAM_TOPIC,
                "context_window": context_window,
                "max_output_tokens": max_output,
            }]
        }))
    }
}

impl OpenAIProvider {
    /// Build and send the HTTP request, then parse the SSE response.
    fn execute_request(
        request_id: Uuid,
        model: &str,
        messages: &[Message],
        tools: &[astrid_sdk::types::LlmToolDefinition],
        system: &str,
    ) -> Result<(), SysError> {
        let url = format!("{BASE_URL}/v1/chat/completions");

        let resolved_model = if model.is_empty() {
            env::var("model").unwrap_or_else(|_| "gpt-4.1".into())
        } else {
            model.to_string()
        };

        let mut api_messages: Vec<Value> = Vec::new();

        if !system.is_empty() {
            api_messages.push(serde_json::json!({
                "role": "system",
                "content": system,
            }));
        }

        for msg in messages {
            if msg.role != MessageRole::System {
                api_messages.push(Self::convert_message(msg));
            }
        }

        let mut request_body = serde_json::json!({
            "model": resolved_model,
            "messages": api_messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        // max_completion_tokens — OpenAI's preferred field.
        if let Ok(max_tokens) = env::var("max_output_tokens")
            && let Ok(n) = max_tokens.parse::<u64>()
            && n > 0
        {
            request_body["max_completion_tokens"] = serde_json::json!(n);
        }

        // Temperature.
        if let Ok(temp) = env::var("temperature")
            && let Ok(t) = temp.parse::<f64>()
        {
            request_body["temperature"] = serde_json::json!(t);
        }

        // Reasoning effort for o-series models.
        if let Ok(effort) = env::var("reasoning_effort")
            && !effort.is_empty()
        {
            request_body["reasoning_effort"] = serde_json::json!(effort);
        }

        // Service tier.
        if let Ok(tier) = env::var("service_tier")
            && !tier.is_empty()
        {
            request_body["service_tier"] = serde_json::json!(tier);
        }

        // Tools with strict mode.
        if !tools.is_empty() {
            let api_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                            "strict": true,
                        }
                    })
                })
                .collect();
            request_body["tools"] = Value::Array(api_tools);
            request_body["parallel_tool_calls"] = serde_json::json!(true);
        }

        let api_key = env::var("api_key").unwrap_or_default();
        if api_key.is_empty() {
            return Err(SysError::ApiError(
                "OpenAI api_key not configured".into(),
            ));
        }

        let req = http::Request::post(&url)
            .header("authorization", format!("Bearer {api_key}"))
            .json(&request_body)?;

        let resp = http::stream_start(&req)?;

        if resp.status != 200 {
            let mut error_body = String::new();
            while let Some(chunk) = http::stream_read(&resp.handle)? {
                error_body.push_str(&String::from_utf8_lossy(&chunk));
                if error_body.len() > 4096 {
                    error_body.truncate(4096);
                    break;
                }
            }
            let _ = http::stream_close(&resp.handle);
            return Err(SysError::ApiError(format!(
                "OpenAI API error ({}): {error_body}",
                resp.status
            )));
        }

        let result = Self::parse_sse_stream(request_id, &resp.handle);
        let _ = http::stream_close(&resp.handle);
        result
    }

    /// Stream SSE chunks, publishing IPC events as they arrive.
    fn parse_sse_stream(
        request_id: Uuid,
        stream: &http::HttpStreamHandle,
    ) -> Result<(), SysError> {
        let mut active_tools: Vec<(String, String)> = Vec::new();
        let mut line_buffer = String::new();

        while let Some(chunk) = http::stream_read(stream)? {
            let chunk_str = String::from_utf8_lossy(&chunk);
            line_buffer.push_str(&chunk_str);

            if line_buffer.len() > MAX_LINE_BUFFER_SIZE {
                return Err(SysError::ApiError(
                    "SSE line buffer exceeded maximum size".into(),
                ));
            }

            while let Some(newline_pos) = line_buffer.find('\n') {
                let line = line_buffer[..newline_pos]
                    .trim_end_matches('\r')
                    .to_string();
                line_buffer = line_buffer[(newline_pos + 1)..].to_string();

                if line.is_empty() {
                    continue;
                }

                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };

                if data == "[DONE]" {
                    Self::publish_stream(request_id, StreamEvent::Done)?;
                    return Ok(());
                }

                let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) else {
                    continue;
                };

                Self::process_chunk(request_id, &chunk, &mut active_tools)?;
            }
        }

        Ok(())
    }

    /// Process a single SSE chunk.
    fn process_chunk(
        request_id: Uuid,
        chunk: &ChatCompletionChunk,
        active_tools: &mut Vec<(String, String)>,
    ) -> Result<(), SysError> {
        if let Some(usage) = &chunk.usage {
            Self::publish_stream(
                request_id,
                StreamEvent::Usage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                },
            )?;
        }

        let Some(choice) = chunk.choices.first() else {
            return Ok(());
        };

        if let Some(ref text) = choice.delta.content
            && !text.is_empty()
        {
            Self::publish_stream(request_id, StreamEvent::TextDelta(text.clone()))?;
        }

        if let Some(ref tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                while active_tools.len() <= tc.index {
                    active_tools.push((String::new(), String::new()));
                }

                if let Some(ref id) = tc.id {
                    active_tools[tc.index].0 = id.clone();
                }

                if let Some(ref func) = tc.function {
                    if let Some(ref name) = func.name {
                        active_tools[tc.index].1 = name.clone();
                        Self::publish_stream(
                            request_id,
                            StreamEvent::ToolCallStart {
                                id: active_tools[tc.index].0.clone(),
                                name: name.clone(),
                            },
                        )?;
                    }

                    if let Some(ref args) = func.arguments
                        && !args.is_empty()
                    {
                        Self::publish_stream(
                            request_id,
                            StreamEvent::ToolCallDelta {
                                id: active_tools[tc.index].0.clone(),
                                args_delta: args.clone(),
                            },
                        )?;
                    }
                }
            }
        }

        if let Some(ref reason) = choice.finish_reason
            && reason == "tool_calls"
        {
            for (id, _name) in active_tools.iter() {
                if !id.is_empty() {
                    Self::publish_stream(request_id, StreamEvent::ToolCallEnd { id: id.clone() })?;
                }
            }
            active_tools.clear();
        }

        Ok(())
    }

    /// Publish a stream event to the event bus.
    fn publish_stream(request_id: Uuid, event: StreamEvent) -> Result<(), SysError> {
        ipc::publish_json(
            STREAM_TOPIC,
            &IpcPayload::LlmStreamEvent { request_id, event },
        )
    }

    /// Convert an Astrid `Message` to OpenAI Chat Completions JSON format.
    fn convert_message(message: &Message) -> Value {
        match &message.content {
            MessageContent::Text(text) => {
                serde_json::json!({
                    "role": Self::role_str(message.role),
                    "content": text,
                })
            }
            MessageContent::ToolCalls(calls) => {
                let tool_calls: Vec<Value> = calls
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": if c.arguments.is_string() {
                                    c.arguments.clone()
                                } else {
                                    Value::String(c.arguments.to_string())
                                },
                            }
                        })
                    })
                    .collect();

                serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls,
                })
            }
            MessageContent::ToolResult(result) => {
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                })
            }
            MessageContent::MultiPart(parts) => {
                let content: Vec<Value> = parts
                    .iter()
                    .map(|p| match p {
                        astrid_sdk::types::ContentPart::Text { text } => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        astrid_sdk::types::ContentPart::Image { media_type, data } => {
                            serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{media_type};base64,{data}"),
                                }
                            })
                        }
                    })
                    .collect();

                serde_json::json!({
                    "role": Self::role_str(message.role),
                    "content": content,
                })
            }
        }
    }

    /// Map Astrid `MessageRole` to OpenAI role string.
    fn role_str(role: MessageRole) -> &'static str {
        match role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }
}
