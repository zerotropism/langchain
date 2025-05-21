from typing import List, Dict, Optional, Any
from config import ConfigManager
from decorators import handle_exception
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


class PromptManager:
    """Manager for creating and formatting prompt templates."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.templates = config.get("prompts")
        self.examples = config.get("examples")
        self.schemas = config.get("schemas")

    @handle_exception
    def create_template(self, template_string: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template from a string
        and return it as a ChatPromptTemplate object.
        Args:
            template_string (`str`): The template string with variables in {curly_braces}
        """
        return ChatPromptTemplate.from_template(template_string)

    @handle_exception
    def get_template(self, name: str = "default") -> ChatPromptTemplate:
        """
        Retrieve a prompt template by name
        and return it as a ChatPromptTemplate object.
        Args:
            name (`str`): Name of the template
        """
        if name not in self.templates:
            if self.templates:
                print(
                    f"Template '{name}' not found. Returning the list of available templates:"
                )
                print(list(self.templates.keys()))
                return None
            print("No templates available.")
            return None
        return self.create_template(self.templates.get(name))

    @handle_exception
    def format_simple_text(self, prompt: str) -> List[HumanMessage]:
        """
        Format a prompt to an LLM-ready message.

        Take a string prompt and return it as HumanMessage object.
        Args:
            prompt (`str`): The prompt string with variables in {curly_braces}
        """
        return [HumanMessage(content=prompt)]

    @handle_exception
    def format_list_of_texts(self, prompts: List[str]) -> List[HumanMessage]:
        """
        Format a list of prompts to LLM-ready messages.

        Take a list of string prompts and return a list of HumanMessage objects.
        Args:
            prompts (`list`): The list of prompt strings with variables in {curly_braces}
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
        Format a template to an LLM-ready template content.

        Take a template text as ChatPromptTemplate object
        and format it with the provided keyword arguments.
        Args:
            template (`ChatPromptTemplate`): The prompt template
            **kwargs: The values to fill into the template
        """
        return template.format_messages(**kwargs)

    @handle_exception
    def formatter(self, prompt: Optional[Any], **kwargs) -> str:
        """
        Call for the right prompt formatting methods based on the prompt type.

        Take a prompt and call the approriate formatting to an LLM-ready message.
        Args:
            prompt (`str` or `list` or `ChatPromptTemplate`): The prompt to format
        """
        if not prompt:
            return self.format_simple_text(self.default_prompt)
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

    # class PromptManager:
    #     """Manager for creating and formatting prompt templates."""

    #     def __init__(self, config: Optional[ConfigManager] = None):
    #         """
    #         Initialize the prompt manager.

    #         Args:
    #             config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
    #         """
    #         self.prompt_settings = (
    #             config.get_prompt_settings or ConfigManager().get_prompt_settings
    #         )
    #         self.system_prompt = self.prompt_settings["system"]
    #         self.default_prompt = self.prompt_settings["one_shot"]
    #         self.templates = self.prompt_settings["templates"]

    #         self.resources = config.get_examples or ConfigManager().get_examples

    #     @property
    #     def get_system_prompt(self) -> str:
    #         """Retrieve the system prompt from the configuration."""
    #         return self.system_prompt

    #     @get_system_prompt.setter
    #     def set_system_prompt(self, value: str):
    #         """Set the system prompt in the configuration."""
    #         self.system_prompt = value

    #     @property
    #     def get_default_prompt(self) -> str:
    #         """Retrieve the default prompt from the configuration."""
    #         return self.default_prompt

    #     @property
    #     def get_templates(self) -> Dict[str, str]:
    #         """Retrieve the templates from the configuration."""
    #         return self.templates

    # @handle_exception
    # def get_resource(self, *args) -> Dict[str, str]:
    #     """Retrieve specific resource from the examples loaded in the configuration."""
    #     resources = self.resources
    #     for arg in args:
    #         resources = resources[arg] if arg in resources else None
    #         if resources is None:
    #             print(f"Resource '{arg}' not found.")
    #             return None
    #     return resources

    # def get_resource2(self, task: str):
    #     # Direct access to the examples
    #     return self.params.config_data["examples"][task]
