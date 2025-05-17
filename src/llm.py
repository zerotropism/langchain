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
        self._params = config
        self._model = self._params.get_model_params.get("model")
        self._temperature = self._params.get_model_params.get("temperature")
        self._chat_instance = None

    @property
    def what_model(self) -> str:
        """Get the model name."""
        return self._model

    @what_model.setter
    def set_model(self, value: str):
        """Set the model name."""
        self._model = value

    @property
    def what_temperature(self) -> float:
        """Get the temperature setting."""
        return self._temperature

    @what_temperature.setter
    def set_temperature(self, value: float):
        """Set the temperature setting."""
        self._temperature = value

    @property
    def what_params(self) -> ConfigManager:
        """Get the model parameters."""
        return self._params.get_model_params or {}

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
    """Custom LLM class that overrides token counting methods."""

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
