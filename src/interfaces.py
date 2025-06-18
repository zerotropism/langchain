from abc import ABC, abstractmethod
from typing import Optional


class IConfigManager(ABC):
    @abstractmethod
    def get(self, section: str, key: Optional[str] = None):
        """Get configuration values."""
        pass

    @abstractmethod
    def get_prompt(self, name: str = "default"):
        """Get a prompt template by name."""
        pass

    @abstractmethod
    def get_example(self, task: str, name: Optional[str] = None):
        """Get an example by task and name."""
        pass

    @abstractmethod
    def get_schema(self, name: str):
        """Get a schema by name."""
        pass


class ILLMClient(ABC):
    @abstractmethod
    def infer(self):
        """Returns a language model instance."""
        pass


class IPromptManager(ABC):
    @abstractmethod
    def formatter(self, prompt, **kwargs):
        """Format a prompt template with provided variables."""
        pass

    @abstractmethod
    def get_example(self, type_name, name):
        """Get an example from the configuration."""
        pass

    @abstractmethod
    def get_template(self, name):
        """Get a template by name."""
        pass

    @abstractmethod
    def create_template(self, template_str):
        """Create a new template from a string."""
        pass

    @abstractmethod
    def build_chat_messages(self, system_prompt=None, user_prompt=None):
        """Build a list of chat messages."""
        pass


class IOutputParser(ABC):
    @abstractmethod
    def get_parser(self, schema_name):
        """Get an output parser for the given schema."""
        pass

    @abstractmethod
    def get_format_instructions(self, parser):
        """Get format instructions for the parser."""
        pass

    @abstractmethod
    def parse_output(self, parser, response):
        """Parse the model response with the given parser."""
        pass


class IMemoryFactory(ABC):
    @abstractmethod
    def build(self, llm, custom_memory, **kwargs):
        """Build a memory manager of the specified type."""
        pass


class IHistoryManager(ABC):
    @abstractmethod
    def add_message(self, session_id, role, content):
        """Add a message to the conversation history."""
        pass

    @abstractmethod
    def get_messages(self, session_id):
        """Get all messages for a session."""
        pass

    @abstractmethod
    def clear_session(self, session_id):
        """Clear all messages for a session."""
        pass


class ITextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt, **kwargs):
        """Generate text based on a prompt."""
        pass


class ITranslator(ABC):
    @abstractmethod
    def translate(self, usecase=None, text=None, style=None):
        """Translate text to a different style."""
        pass


class IExtractor(ABC):
    @abstractmethod
    def extract(self, text=None, schema_name=None):
        """Extract structured information from text."""
        pass


class IChatManager(ABC):
    @abstractmethod
    def chat(self, user_input, session_id="default"):
        """Process a chat interaction and return a response."""
        pass
