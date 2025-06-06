from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Optional
from config import ConfigManager
from decorators import handle_exception
from typing import Optional
from llm import LLMClient


class MessageHistoryMemoryManager:
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.llm_client = LLMClient(self.config)
        self.session_store = {}

        # Mapping memory types to their respective classes
        memory_classes = {
            "buffer": ConversationBufferMemory,
            "window": ConversationBufferWindowMemory,
            "token": ConversationTokenBufferMemory,
            "summary": ConversationSummaryBufferMemory,
        }
        memory_type = self.config.get("memory", {}).get("type", "buffer")
        memory_class = memory_classes.get(memory_type, ConversationBufferMemory)

        # Get the appropriate parameters for the memory class
        memory_kwargs = {}
        if memory_type == "window":
            memory_kwargs["k"] = self.config.get("memory", {}).get("window_size", 3)
        if memory_type in ("token", "summary"):
            memory_kwargs["max_token_limit"] = self.config.get("memory", {}).get(
                "max_token_limit", 100
            )

        class MemoryWrapper:
            """Wrapper class to access chat memory messages."""

            def __init__(self, memory):
                self.memory = memory

            @property
            def messages(self):
                return self.memory.chat_memory.messages

            def __getattr__(self, attr):
                return getattr(self.memory, attr)

        def get_history(session_id: str):
            if session_id not in self.session_store:
                self.session_store[session_id] = memory_class(**memory_kwargs)
            return MemoryWrapper(self.session_store[session_id])

        self.runnable = RunnableWithMessageHistory(
            self.llm_client.infer(),
            lambda session_id: get_history(session_id),
        )


class MemoryFactory:
    """Factory class to create appropriate memory managers."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the memory factory.

        Args:
            config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
        """
        self.memory_settings = (
            config.get("memory") if config else ConfigManager().get("memory")
        )
        self.memory_type = self.memory_settings.get("type", "buffer").lower()
        self.window_size = self.memory_settings.get("window_size", 3)
        self.max_token_limit = self.memory_settings.get("max_token_limit", 100)
        self.verbose = self.memory_settings.get("verbose", False)
        self.instance = self.build(self.memory_type)

    @handle_exception
    def build(self, memory_type: Optional[str] = "buffer", **kwargs):
        """
        Build the appropriate memory manager based on the specified type.

        Args:
            memory_type (`str`, optional): The type of memory to create, defaults to "buffer"
            **kwargs: Additional keyword arguments for the memory class

        Returns:
            An instance of the specified memory class.
        """
        # Mapping memory types to their respective classes
        memory_classes = {
            "buffer": ConversationBufferMemory,
            "window": ConversationBufferWindowMemory,
            "token": ConversationTokenBufferMemory,
            "summary": ConversationSummaryBufferMemory,
        }
        memory_class = memory_classes.get(memory_type, ConversationBufferMemory)

        # Get the appropriate parameters for the memory class
        memory_kwargs = {"verbose": kwargs.get("verbose", self.verbose)}
        if memory_type == "window":
            memory_kwargs["k"] = kwargs.get("window_size", self.window_size)
        if memory_type in ("token", "summary"):
            memory_kwargs["max_token_limit"] = kwargs.get(
                "max_token_limit", self.max_token_limit
            )

        return memory_class(**memory_kwargs)
