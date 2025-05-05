from typing import Optional
from config import ConfigManager
from langchain_ollama import ChatOllama


class LLMClient:
    """Base client for interacting with language models."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the LLM client.
        Args:
            config (ConfigManager, optional): Pre-loaded settings from the configuration file
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

    @property
    def run(self):
        """Lazy-loaded chat model instance."""
        self._chat_instance = self._chat_instance or ChatOllama(
            model=self._model, temperature=self._temperature
        )
        return self._chat_instance
