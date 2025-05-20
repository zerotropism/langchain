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
        self._params = config.get_model_settings or ConfigManager().get_model_settings
        self._model = self._params["model"]
        self._temperature = self._params["temperature"]
        self._top_k = self._params["top_k"]
        self._top_p = self._params["top_p"]
        self._context_length = self._params["context_length"]
        self._chat_instance = None

    @property
    def get_model(self) -> str:
        """Get the model name."""
        return self._model

    @get_model.setter
    def set_model(self, value: str):
        """Set the model name."""
        self._model = value

    @property
    def get_temperature(self) -> float:
        """Get the temperature setting."""
        return self._temperature

    @get_temperature.setter
    def set_temperature(self, value: float):
        """Set the temperature setting."""
        self._temperature = value

    @property
    def top_k(self) -> int:
        """Get the top_k setting."""
        return self._top_k

    @top_k.setter
    def set_top_k(self, value: int):
        """Set the top_k setting."""
        self._top_k = value

    @property
    def top_p(self) -> float:
        """Get the top_p setting."""
        return self._top_p

    @top_p.setter
    def set_top_p(self, value: float):
        """Set the top_p setting."""
        self._top_p = value

    @property
    def context_length(self) -> int:
        """Get the context length setting."""
        return self._context_length

    @context_length.setter
    def set_context_length(self, value: int):
        """Set the context length setting."""
        self._context_length = value

    @property
    def get_params(self) -> ConfigManager:
        """Get the model parameters."""
        return self._params or {}

    def infer(
        self,
        custom_model: str = None,
        custom_temperature: str = None,
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
