from typing import Optional, Union, List, Dict, Any
from config import ConfigManager
from langchain_ollama import ChatOllama


class LLMClient:
    """Base client for interacting with language models."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the LLM client.

        Args:
            config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
        """
        # Utilisation de la nouvelle méthode générique pour accéder à la config
        model_settings = config.get("model") if config else ConfigManager().get("model")
        self._model = model_settings.get("name")
        self._temperature = model_settings.get("temperature")
        self._top_k = model_settings.get("top_k")
        self._top_p = model_settings.get("top_p")
        self._context_length = model_settings.get("context_length")
        self._chat_instance = None

    def infer(
        self,
        custom_model: str = None,
        custom_temperature: float = None,
        custom_token_count: bool = False,
    ):
        """Lazy-loaded chat model instance."""
        self._chat_instance = (
            self._chat_instance
            or ChatOllama(
                model=custom_model or self._model,
                temperature=custom_temperature or self._temperature,
            )
            if not custom_token_count
            else CustomTokenCountLLM(model=self._model, temperature=self._temperature)
        )
        return self._chat_instance


class CustomTokenCountLLM(ChatOllama):
    """Custom LLM class that overrides token counting methods.

    This class is used to count tokens in a naive custom way, as the default."""

    def get_num_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        # Simple word-based tokenization
        return len(text.split())

    def get_num_tokens_from_messages(self, messages: List[Union[Dict, Any]]) -> int:
        """Count tokens in a list of messages."""
        count = 0
        for message in messages:
            # Extract message content from different possible formats
            if hasattr(message, "content"):
                content = message.content
            elif isinstance(message, dict) and "content" in message:
                content = message["content"]
            else:
                content = str(message)

            count += self.get_num_tokens(content)
        return count
