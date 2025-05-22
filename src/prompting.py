from typing import List, Dict, Optional, Any
from config import ConfigManager
from decorators import handle_exception
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


class PromptManager:
    """Manager for creating and formatting prompt templates."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.prompt_templates = config.get("prompts")
        self.usecase_examples = config.get("examples")
        self.schema_templates = config.get("schemas")

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
        and return it as a LLM-ready ChatPromptTemplate object.
        Args:
            name (`str`): Name of the template
        """
        if name not in self.prompt_templates:
            if self.prompt_templates:
                print(
                    f"Template '{name}' not found. Returning the list of available templates:"
                )
                print(list(self.prompt_templates.keys()))
                return None
            print("No templates available.")
            return None
        return self.create_template(self.prompt_templates.get(name))

    @handle_exception
    def get_example(self, task: str, name: str):
        """
        Retrieve an example by task and name.
        Args:
            task (`str`): The task name
            name (`str`): The example name
        """
        for example in self.usecase_examples.get(task, []):
            if example.get("name") == name:
                return example
        return None

    @handle_exception
    def get_schema(self, name: str):
        """
        Retrieve a schema by name.
        Args:
            name (`str`): The schema name
        """
        if name not in self.schema_templates:
            if self.schema_templates:
                print(
                    f"Schema '{name}' not found. Returning the list of available schemas:"
                )
                print(list(self.schema_templates.keys()))
                return None
            print("No schemas available.")
            return None
        return self.schema_templates.get(name)

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
            return self.format_simple_text(self.prompt_templates.get("default"))
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
