//! OpenAI Responses API streaming event types.
//!
//! These types map to the Responses API (`POST /v1/responses`) streaming
//! format, which uses named SSE events with typed JSON payloads.

use serde::Deserialize;

/// Text delta event — `response.output_text.delta`
#[derive(Deserialize, Debug)]
pub(crate) struct TextDelta {
    /// The item ID this delta belongs to.
    pub item_id: String,
    /// Index in the output array.
    pub output_index: usize,
    /// Index within the content parts.
    pub content_index: usize,
    /// The text chunk.
    pub delta: String,
}

/// Function call arguments delta — `response.function_call_arguments.delta`
#[derive(Deserialize, Debug)]
pub(crate) struct FunctionCallArgsDelta {
    /// The item ID (tool call ID).
    pub item_id: String,
    /// Index in the output array.
    pub output_index: usize,
    /// Partial JSON arguments.
    pub delta: String,
}

/// Function call arguments done — `response.function_call_arguments.done`
#[derive(Deserialize, Debug)]
pub(crate) struct FunctionCallArgsDone {
    /// The item ID (tool call ID).
    pub item_id: String,
    /// Index in the output array.
    pub output_index: usize,
    /// The function name.
    pub name: String,
    /// Complete JSON arguments.
    pub arguments: String,
}

/// Output item added — `response.output_item.added`
#[derive(Deserialize, Debug)]
pub(crate) struct OutputItemAdded {
    /// The output item.
    pub item: OutputItem,
    /// Index in the output array.
    pub output_index: usize,
}

/// An item in the response output array.
#[derive(Deserialize, Debug)]
pub(crate) struct OutputItem {
    /// Item ID.
    pub id: String,
    /// Item type: "message", "function_call", etc.
    #[serde(rename = "type")]
    pub item_type: String,
    /// Function name (only for function_call items).
    #[serde(default)]
    pub name: Option<String>,
    /// Call ID for function calls.
    #[serde(default)]
    pub call_id: Option<String>,
}

/// Response completed — `response.completed`
#[derive(Deserialize, Debug)]
pub(crate) struct ResponseCompleted {
    /// The full response object.
    pub response: ResponseObject,
}

/// The response object within a completed event.
#[derive(Deserialize, Debug)]
pub(crate) struct ResponseObject {
    /// Response status.
    pub status: String,
    /// Token usage.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Token usage statistics.
#[derive(Deserialize, Debug)]
pub(crate) struct Usage {
    /// Input tokens consumed.
    pub input_tokens: usize,
    /// Output tokens generated.
    pub output_tokens: usize,
}
