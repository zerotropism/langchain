from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from llm import LLMClient

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
            llm: The language model to use
            verbose: Whether to print verbose output
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
    """Manager for ConversationBufferMemory."""

    def _create_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory()

    def clear_memory(self) -> None:
        """Clear the memory buffer."""
        self.memory.clear()


class WindowMemoryManager(BaseMemoryManager):
    """Manager for ConversationBufferWindowMemory."""

    def __init__(self, llm: BaseLLM, window_size: int = 1, verbose: bool = False):
        """
        Initialize with a specific window size.
        Args:
            llm: The language model to use
            window_size: Number of conversation exchanges to remember
            verbose: Whether to print verbose output
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
    """Manager for ConversationTokenBufferMemory with custom tokenization."""

    def __init__(self, llm: BaseLLM, max_token_limit: int = 100, verbose: bool = False):
        """
        Initialize with a token limit.
        Args:
            llm: The language model to use with token counting capability
            max_token_limit: Maximum number of tokens to keep in memory
            verbose: Whether to print verbose output
        """
        self.max_token_limit = max_token_limit
        super().__init__(llm, verbose)

    def _create_memory(self) -> ConversationTokenBufferMemory:
        return ConversationTokenBufferMemory(
            llm=self.llm, max_token_limit=self.max_token_limit
        )


class SummaryMemoryManager(BaseMemoryManager):
    """Manager for ConversationSummaryBufferMemory."""

    def __init__(self, llm: BaseLLM, max_token_limit: int = 100, verbose: bool = False):
        """
        Initialize with a token limit for summarization.
        Args:
            llm: The language model to use for summarization
            max_token_limit: Maximum number of tokens before summarizing
            verbose: Whether to print verbose output
        """
        self.max_token_limit = max_token_limit
        super().__init__(llm, verbose)

    def _create_memory(self) -> ConversationSummaryBufferMemory:
        return ConversationSummaryBufferMemory(
            llm=self.llm, max_token_limit=self.max_token_limit
        )


class MemoryFactory:
    """Factory class to create appropriate memory managers."""

    @staticmethod
    def create_memory_manager(
        llm: LLMClient,
        memory_type: Optional[str] = "buffer",
        **kwargs,
    ) -> BaseMemoryManager:
        """
        Create a memory manager of the specified type.
        Args:
            memory_type: Type of memory to create (`buffer`, `window`, `token`, `summary`)
            **kwargs: Additional arguments for specific memory types (`window_size`, `max_token_limit`, `verbose`)
        Returns:
            An appropriate memory manager instance
        Raises:
            ValueError: If memory_type is not recognized
        """
        # Create the appropriate LLM
        llm = llm.infer(
            custom_token_count=True if memory_type in ["token", "summary"] else False,
        )

        # Create the appropriate memory manager
        if memory_type == "buffer":
            return BufferMemoryManager(llm, **kwargs)
        elif memory_type == "window":
            window_size = kwargs.get("window_size", 1)
            return WindowMemoryManager(llm, window_size, **kwargs.get("verbose", False))
        elif memory_type == "token":
            max_token_limit = kwargs.get("max_token_limit", 100)
            return TokenMemoryManager(
                llm, max_token_limit, **kwargs.get("verbose", False)
            )
        elif memory_type == "summary":
            max_token_limit = kwargs.get("max_token_limit", 100)
            return SummaryMemoryManager(
                llm, max_token_limit, **kwargs.get("verbose", False)
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
