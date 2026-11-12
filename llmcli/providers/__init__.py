"""Provider registry and factory."""

from llmcli.config import Config
from llmcli.providers.ollama import OllamaProvider
from llmcli.providers.openai_compat import OpenAICompatProvider


def create_provider(config: Config):
    """Factory function to instantiate the correct provider."""
    if config.provider == "ollama":
        return OllamaProvider(config)
    if config.provider == "openai_compat":
        return OpenAICompatProvider(config)
    raise ValueError(f"Unknown provider: {config.provider}. Supported: ollama, openai_compat")
