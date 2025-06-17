from typing import Optional, Union, List, Dict, Any
from interfaces import IConfigManager, ILLMClient
from langchain_ollama import ChatOllama


class LLMClient(ILLMClient):
    """Base client for interacting with language models."""

    def __init__(self, config: Optional[IConfigManager] = None):
        """
        Initialize the LLM client.

        Args:
            config (`IConfigManager`, optional): Configuration manager with pre-loaded
                settings from `./config.yml` file
        """
        from config import ConfigManager

        config_manager = config or ConfigManager()
        model_settings = config_manager.get("model")

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
    ) -> ChatOllama:
        """Lazy-loaded chat model instance.

        This method initializes the ChatOllama instance with the specified model and temperature.

        Args:
            custom_model (`str`, optional): Custom model name to use instead of the default.
            custom_temperature (`float`, optional): Custom temperature for the model.
            custom_token_count (`bool`, optional): If True, uses a custom token counting method.

        Returns:
            ChatOllama: An instance of the ChatOllama class configured with the specified parameters.
        """
        if not self._chat_instance:
            if custom_token_count:
                self._chat_instance = CustomTokenCountLLM(
                    model=custom_model or self._model,
                    temperature=custom_temperature or self._temperature,
                )
            else:
                self._chat_instance = ChatOllama(
                    model=custom_model or self._model,
                    temperature=custom_temperature or self._temperature,
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
