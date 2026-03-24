//! OpenAI model registry — static lookup table for model capabilities.
//!
//! Users select a model ID and the capsule resolves context window,
//! max output tokens, and feature flags automatically. Env vars
//! override these defaults when set.
//!
//! Last updated: 2026-03-25. Sources:
//! - https://developers.openai.com/api/docs/models
//! - https://developers.openai.com/api/docs/models/gpt-5.4
//! - https://developers.openai.com/api/docs/models/gpt-5.2

/// Known OpenAI model capabilities.
#[derive(Debug, Clone, Copy)]
#[expect(dead_code, reason = "registry fields used as capabilities expand")]
pub(crate) struct ModelInfo {
    /// Model identifier as used in the API.
    pub id: &'static str,
    /// Human-readable display name.
    pub name: &'static str,
    /// Maximum context window in tokens.
    pub context_window: u64,
    /// Default max output tokens.
    pub max_output_tokens: u64,
    /// Supports vision (image inputs).
    pub supports_vision: bool,
    /// Supports tool/function calling.
    pub supports_tools: bool,
    /// Supports structured outputs (response_format json_schema).
    pub supports_structured_output: bool,
    /// Supports reasoning effort levels (none/low/medium/high/xhigh).
    pub is_reasoning: bool,
}

/// Static registry of known OpenAI models.
///
/// Update this table when OpenAI releases new models.
/// Unknown models fall back to conservative defaults via [`lookup`].
pub(crate) static MODELS: &[ModelInfo] = &[
    // ── GPT-5.4 series (March 2026, current frontier) ────────────
    ModelInfo {
        id: "gpt-5.4",
        name: "GPT-5.4",
        context_window: 1_050_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true, // supports effort: none/low/medium/high/xhigh
    },
    ModelInfo {
        id: "gpt-5.4-mini",
        name: "GPT-5.4 Mini",
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    ModelInfo {
        id: "gpt-5.4-nano",
        name: "GPT-5.4 Nano",
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    // ── GPT-5.3 series ───────────────────────────────────────────
    ModelInfo {
        id: "gpt-5.3",
        name: "GPT-5.3 Instant",
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    ModelInfo {
        id: "gpt-5.3-codex",
        name: "GPT-5.3 Codex",
        context_window: 1_000_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    ModelInfo {
        id: "gpt-5.3-codex-spark",
        name: "GPT-5.3 Codex Spark",
        context_window: 128_000,
        max_output_tokens: 128_000,
        supports_vision: false,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    // ── GPT-5.2 series (December 2025) ───────────────────────────
    ModelInfo {
        id: "gpt-5.2",
        name: "GPT-5.2",
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    ModelInfo {
        id: "gpt-5.2-codex",
        name: "GPT-5.2 Codex",
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    // ── GPT-4.1 series (April 2025, still available) ─────────────
    ModelInfo {
        id: "gpt-4.1",
        name: "GPT-4.1",
        context_window: 1_048_576,
        max_output_tokens: 32_768,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    ModelInfo {
        id: "gpt-4.1-mini",
        name: "GPT-4.1 Mini",
        context_window: 1_048_576,
        max_output_tokens: 32_768,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    ModelInfo {
        id: "gpt-4.1-nano",
        name: "GPT-4.1 Nano",
        context_window: 1_048_576,
        max_output_tokens: 32_768,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    // ── o-series reasoning models ────────────────────────────────
    ModelInfo {
        id: "o3",
        name: "o3",
        context_window: 200_000,
        max_output_tokens: 100_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    ModelInfo {
        id: "o3-mini",
        name: "o3 Mini",
        context_window: 200_000,
        max_output_tokens: 100_000,
        supports_vision: false,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    ModelInfo {
        id: "o4-mini",
        name: "o4 Mini",
        context_window: 200_000,
        max_output_tokens: 100_000,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: true,
    },
    // ── GPT-4o series (legacy, still available) ──────────────────
    ModelInfo {
        id: "gpt-4o",
        name: "GPT-4o",
        context_window: 128_000,
        max_output_tokens: 16_384,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
    ModelInfo {
        id: "gpt-4o-mini",
        name: "GPT-4o Mini",
        context_window: 128_000,
        max_output_tokens: 16_384,
        supports_vision: true,
        supports_tools: true,
        supports_structured_output: true,
        is_reasoning: false,
    },
];

/// Conservative defaults for unknown models.
const UNKNOWN_DEFAULTS: ModelInfo = ModelInfo {
    id: "unknown",
    name: "Unknown Model",
    context_window: 128_000,
    max_output_tokens: 16_384,
    supports_vision: false,
    supports_tools: true,
    supports_structured_output: false,
    is_reasoning: false,
};

/// Look up a model by ID. Returns conservative defaults for unknown models.
///
/// Matches exact IDs first, then tries prefix matching for dated snapshots
/// (e.g., `gpt-5.4-2026-03-05` matches `gpt-5.4`).
pub(crate) fn lookup(model_id: &str) -> &'static ModelInfo {
    // Exact match.
    if let Some(info) = MODELS.iter().find(|m| m.id == model_id) {
        return info;
    }

    // Prefix match for dated snapshots (e.g., gpt-5.4-2026-03-05 → gpt-5.4).
    if let Some(info) = MODELS.iter().find(|m| model_id.starts_with(m.id)) {
        return info;
    }

    &UNKNOWN_DEFAULTS
}

/// List all known model IDs for display purposes.
pub(crate) fn list_model_ids() -> Vec<&'static str> {
    MODELS.iter().map(|m| m.id).collect()
}
