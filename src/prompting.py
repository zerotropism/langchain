from typing import List, Optional, Any
from config import ConfigManager
from decorators import handle_exception
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


class PromptManager:
    """Manager for creating and formatting prompt templates."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the prompt manager.
        Args:
            config (ConfigManager, optional): Pre-loaded settings from the configuration file
        """
        self._params = config
        self._templates = {}

        # Preload default template if available
        if self._params._template:
            self._templates["default"] = self.create_template(self._params._template)

    @property
    def what_params(self) -> ConfigManager:
        """Get the model parameters."""
        return self._params.get_model_params or {}

    @property
    def what_templates(self) -> dict:
        """Get the available templates."""
        return self._templates or self._params._template

    @handle_exception
    def get_template(self, name: str = "default") -> Optional[ChatPromptTemplate]:
        """
        Get a pre-loaded template by name.
        Args:
            name (str): Name of the template
        Returns:
            ChatPromptTemplate (optional): The template or None if not found
        """
        if name not in self._templates:
            if self._templates:
                print(
                    f"Template '{name}' not found. Returning the list of available templates:"
                )
                print(list(self._templates.keys()))
                return None
            print("No templates available.")
            return None
        return self._templates.get(name)

    @handle_exception
    def create_template(self, template_string: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template from a string.
        Args:
            template_string (str): The template string with variables in {curly_braces}
        Returns:
            ChatPromptTemplate: A ChatPromptTemplate object
        """
        return ChatPromptTemplate.from_template(template_string)

    @handle_exception
    def format_simple_text(self, prompt: str) -> List[HumanMessage]:
        """
        Format a prompt  as HumanMessage object.
        Args:
            prompt (str): The prompt string
        Returns:
            list: Formatted prompt ready to send to the model
        """
        return [HumanMessage(content=prompt)]

    @handle_exception
    def format_list_of_texts(self, prompts: List[str]) -> List[HumanMessage]:
        """
        Format a list of prompts as HumanMessage object.
        Args:
            prompts (list): The list of prompt strings with variables in {curly_braces}
        Returns:
            list: Formatted prompts ready to send to the model
        """
        return [
            prompt if isinstance(prompt, HumanMessage) else HumanMessage(content=prompt)
            for prompt in prompts
        ]

    @handle_exception
    def format_template_text(
        self, template: ChatPromptTemplate, **kwargs
    ) -> List[HumanMessage]:
        """
        Format a template text  with expected values.
        Args:
            template (ChatPromptTemplate): The prompt template
            **kwargs: The values to fill into the template
        Returns:
            list: Formatted messages ready to send to the model
        """
        return template.format_messages(**kwargs)

    @handle_exception
    def formatter(self, prompt: Optional[Any], **kwargs) -> str:
        """
        Format the prompt based on its type.
        Args:
            prompt (str | list | ChatPromptTemplate): The prompt to format
        Returns:
            str: The formatted prompt
        """
        if not prompt:
            return self.format_simple_text(self._params._prompt)
        elif isinstance(prompt, str):
            return self.format_simple_text(prompt)
        elif isinstance(prompt, list):
            return self.format_list_of_texts(prompt)
        elif isinstance(prompt, ChatPromptTemplate):
            return self.format_template_text(prompt, **kwargs)
        else:
            raise ValueError(
                "Unsupported prompt type. Must be str, list, or ChatPromptTemplate object."
            )
