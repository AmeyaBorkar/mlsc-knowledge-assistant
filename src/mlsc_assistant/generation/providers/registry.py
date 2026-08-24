"""Provider registry.

The one place that maps a provider name to an adapter. Adapters are imported lazily so
that selecting Gemini never imports the Anthropic SDK, and so a missing optional extra
produces a message naming the install command rather than an ImportError traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mlsc_assistant.core.errors import ConfigurationError

if TYPE_CHECKING:
    from mlsc_assistant.config import LLMConfig
    from mlsc_assistant.core.ports import LLMProvider

# provider -> the extra that installs it, for the error message
_EXTRAS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "groq": "groq",
}


def build_provider(config: LLMConfig) -> LLMProvider:
    """Construct the configured provider."""
    name = config.provider
    builders: dict[str, Callable[[], LLMProvider]] = {
        "gemini": lambda: _gemini(config),
        "anthropic": lambda: _lazy("anthropic", "AnthropicProvider", config),
        "openai": lambda: _lazy("openai", "OpenAIProvider", config),
        "groq": lambda: _lazy("groq", "GroqProvider", config),
        "ollama": lambda: _lazy("ollama", "OllamaProvider", config),
    }
    try:
        builder = builders[name]
    except KeyError:
        raise ConfigurationError(
            f"Unknown LLM provider {name!r}. Choose one of: {', '.join(sorted(builders))}."
        ) from None
    return builder()


def _gemini(config: LLMConfig) -> LLMProvider:
    from mlsc_assistant.generation.providers.gemini import GeminiProvider

    return GeminiProvider(
        config.resolved_model(),
        api_key=config.api_key(),
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
        thinking_budget=config.thinking_budget,
        requests_per_minute=config.requests_per_minute,
    )


def _lazy(module: str, class_name: str, config: LLMConfig) -> LLMProvider:
    """Import an adapter that ships behind an optional extra.

    These are declared in the registry ahead of being written so the composition root is
    complete and the failure mode is a clear message rather than a KeyError. Adding one
    is a new module plus this import — no change to any caller.
    """
    try:
        imported = __import__(
            f"mlsc_assistant.generation.providers.{module}", fromlist=[class_name]
        )
        provider_class = getattr(imported, class_name)
    except (ImportError, AttributeError) as exc:
        extra = _EXTRAS.get(module)
        hint = (
            f'Run `pip install -e ".[{extra}]"`.'
            if extra
            else "Check that the provider module exists."
        )
        raise ConfigurationError(
            f"The {module!r} provider is not available in this install. {hint} "
            "Gemini is the default and needs no extra."
        ) from exc

    return provider_class(  # type: ignore[no-any-return]
        config.resolved_model(),
        api_key=config.api_key(),
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )
