from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from config import ConfigManager
from llm import LLMClient
from decorators import handle_exception

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
)
from langchain.chains import ConversationChain
from langchain.llms.base import BaseLLM


class BaseMemoryManager(ABC):
    """Abstract base class for memory managers."""

    def __init__(self, llm: BaseLLM, verbose: bool = False):
        """
        Initialize the memory manager.

        Args:
            llm (`BaseLLM`): The language model to use
            verbose (`bool`): Whether to print verbose output
        """
        self.llm = llm
        self.verbose = verbose
        self.memory = self._create_memory()
        self.conversation = self._create_conversation()

    @abstractmethod
    def _create_memory(self) -> Any:
        """Create and return the memory object."""
        pass

    def _create_conversation(self) -> ConversationChain:
        """Create a conversation chain with the memory."""
        return ConversationChain(llm=self.llm, memory=self.memory, verbose=self.verbose)

    def predict(self, input_text: str) -> str:
        """Run prediction with the conversation chain."""
        return self.conversation.predict(input=input_text)

    def get_memory_content(self) -> Dict[str, Any]:
        """Get the current memory variables."""
        return self.memory.load_memory_variables({})

    def get_memory_buffer(self) -> str:
        """Get the raw memory buffer if available."""
        if hasattr(self.memory, "buffer"):
            return self.memory.buffer
        return "Buffer not available for this memory type"

    def add_to_memory(self, user_input: str, ai_output: str) -> None:
        """Manually add a conversation exchange to memory."""
        self.memory.save_context({"input": user_input}, {"output": ai_output})


class BufferMemoryManager(BaseMemoryManager):
    """Manager for `ConversationBufferMemory`.

    Simply stores the conversation history without any additional processing.
    Might be unfit  when the conversation history is bigger than the  model context
    window."""

    def _create_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory()

    def clear_memory(self) -> None:
        """Clear the memory buffer."""
        self.memory.clear()


class WindowMemoryManager(BaseMemoryManager):
    """Manager for `ConversationBufferWindowMemory` with custom number of conversation
    exchanges between the AI and the user."""

    def __init__(self, llm: BaseLLM, window_size: int = 1, verbose: bool = False):
        """
        Initialize with a specific window size.

        Args:
            llm (`BaseLLM`): The language model to use
            window_size (`int`): Number of conversation exchanges to remember
            verbose (`bool`): Whether to print verbose output
        """
        self.window_size = window_size
        super().__init__(llm, verbose)

    def _create_memory(self) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(k=self.window_size)

    def adjust_window_size(self, new_size: int) -> None:
        """Change the window size and recreate the memory."""
        self.window_size = new_size
        self.memory = self._create_memory()
        self.conversation = self._create_conversation()


class TokenMemoryManager(BaseMemoryManager):
    """Manager for `ConversationTokenBufferMemory` with custom sliding number of tokens
    to be remembered by the model."""

    def __init__(self, llm: BaseLLM, max_token_limit: int = 100, verbose: bool = False):
        """
        Initialize with a token limit.

        Args:
            llm (`BaseLLM`): The language model to use with token counting capability
            max_token_limit (`int`): Maximum number of tokens to keep in memory
            verbose (`bool`): Whether to print verbose output
        """
        self.max_token_limit = max_token_limit
        super().__init__(llm, verbose)

    def _create_memory(self) -> ConversationTokenBufferMemory:
        return ConversationTokenBufferMemory(
            llm=self.llm, max_token_limit=self.max_token_limit
        )


class SummaryMemoryManager(BaseMemoryManager):
    """Manager for `ConversationSummaryBufferMemory` providing:
    - a summary of the conversation
    - the last few exchanges
    under the constraint of a default max 2000 tokens."""

    def __init__(self, llm: BaseLLM, max_token_limit: int = 100, verbose: bool = False):
        """
        Initialize with a token limit for summarization.

        Args:
            llm (`BaseLLM`)): The language model to use for summarization,
                              not customizable for now
            max_token_limit (`int`): Maximum number of tokens before summarizing
            verbose (`bool`): Whether to print verbose output
        """
        self.max_token_limit = max_token_limit
        super().__init__(llm, verbose)

    def _create_memory(self) -> ConversationSummaryBufferMemory:
        return ConversationSummaryBufferMemory(
            llm=self.llm, max_token_limit=self.max_token_limit
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

    @handle_exception
    def build(
        self,
        llm: LLMClient,
        custom_memory: Optional[str] = None,
        **kwargs,
    ) -> BaseMemoryManager:
        """
        Create a memory manager of the specified type.

        Args:
            llm (`LLMClient`): The language model client to use
            custom_memory (`str`, optional): Type of memory to create
                (available: `buffer`, `window`, `token`, `summary`), defaults to `buffer`
            **kwargs: Additional arguments for specific memory types
                (available: `window_size`, `max_token_limit`, `verbose`)

        Returns:
            An appropriate memory manager instance
        """
        # Check for custom settings, defaults on config value
        memory = custom_memory or self.memory_type
        window_size = kwargs.get("window_size", self.window_size)
        max_token_limit = kwargs.get("max_token_limit", self.max_token_limit)
        verbose = kwargs.get("verbose", self.verbose)

        # Create the appropriate LLM
        llm = llm.infer(
            custom_token_count=(True if memory in ["token", "summary"] else False),
        )

        # Create the appropriate memory manager
        if memory == "buffer":
            return BufferMemoryManager(llm, **kwargs)
        elif memory == "window":
            return WindowMemoryManager(llm, window_size, verbose)
        elif memory == "token":
            return TokenMemoryManager(llm, max_token_limit, verbose)
        elif memory == "summary":
            return SummaryMemoryManager(llm, max_token_limit, verbose)
        else:
            raise ValueError(f"Unknown memory type: {memory}")
