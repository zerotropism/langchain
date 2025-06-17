from typing import List, Union, Optional, Any
from interfaces import IConfigManager, IPromptManager
from decorators import handle_exception
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage


class PromptManager(IPromptManager):
    """Manager for creating and formatting prompt templates."""

    def __init__(self, config: IConfigManager):
        """
        Initialize the prompt manager.

        Args:
            config (IConfigManager): Configuration manager
        """
        self.config = config
        self.prompt_templates = config.get("prompts")
        self.usecase_examples = config.get("examples")
        self.schema_templates = config.get("schemas")

    @handle_exception
    def create_template(self, template_string: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template from a string and return it as a ChatPromptTemplate object.

        Args:
            template_string (str): The template string with variables in {curly_braces}

        Returns:
            ChatPromptTemplate: A ChatPromptTemplate object ready for use with LLMs.
        """
        return ChatPromptTemplate.from_template(template_string)

    @handle_exception
    def get_template(self, name: str = "default") -> ChatPromptTemplate:
        """
        Retrieve a prompt template by name and return it as an LLM-ready ChatPromptTemplate object.

        Args:
            name (str): Name of the template

        Returns:
            ChatPromptTemplate: A ChatPromptTemplate object ready for use with LLMs.
        """
        if name not in self.prompt_templates:
            if self.prompt_templates:
                print(
                    f"Template '{name}' not found. Available templates: {list(self.prompt_templates.keys())}"
                )
            else:
                print("No templates available.")
            return None
        return self.create_template(self.prompt_templates.get(name))

    @handle_exception
    def get_example(self, task: str, name: str):
        """
        Retrieve an example by task and name.

        Args:
            task (str): The task name
            name (str): The example name

        Returns:
            dict: The example dictionary if found, otherwise None.
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
            name (str): The schema name

        Returns:
            dict: The schema dictionary if found, otherwise None.
        """
        if name not in self.schema_templates:
            if self.schema_templates:
                print(
                    f"Schema '{name}' not found. Available schemas: {list(self.schema_templates.keys())}"
                )
            else:
                print("No schemas available.")
            return None
        return self.schema_templates.get(name)

    @handle_exception
    def formatter(self, prompt: Optional[Any], **kwargs) -> List:
        """
        Format a prompt for use with an LLM.

        Args:
            prompt (str or list or ChatPromptTemplate): The prompt to format
            **kwargs: Variables to use in template formatting

        Returns:
            list: A list of HumanMessage objects ready for use with LLMs.
        """
        if not prompt:
            return [HumanMessage(self.prompt_templates.get("default", ""))]
        elif isinstance(prompt, str):
            return [HumanMessage(content=prompt)]
        elif isinstance(prompt, list):
            return [
                p if isinstance(p, HumanMessage) else HumanMessage(content=p)
                for p in prompt
            ]
        elif isinstance(prompt, ChatPromptTemplate):
            return prompt.format_messages(**kwargs)
        else:
            raise ValueError(
                "Unsupported prompt type. Must be str, list, or ChatPromptTemplate object."
            )

    @handle_exception
    def build_chat_messages(
        self, system_prompt: str = None, user_prompt: str = None
    ) -> List[Union[SystemMessage, HumanMessage]]:
        """
        Builds a list of chat messages from string inputs.

        Args:
            system_prompt (str, optional): The system prompt to include first
            user_prompt (str, optional): The user message to include next

        Returns:
            list: List of SystemMessage and/or HumanMessage depending on the inputs.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if user_prompt:
            messages.append(HumanMessage(content=user_prompt))
        return messages
