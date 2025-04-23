"""
LangChain LLM Integration Toolkit

This module provides a comprehensive, object-oriented implementation for working with LLMs
through LangChain. It includes structured classes for model interaction, prompt management,
output parsing, and high-level text processing workflows.

The module features a clean separation of concerns with specialized components for:
- LLM client communication (LLMClient)
- Prompt template management (PromptManager)
- Structured output parsing (OutputParser)
- Text processing operations (TextProcessor)
"""

from typing import Dict, List, Optional, Union, Any
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from decorators import handle_exception


class Config:
    """Configuration class for managing global settings."""

    def __init__(self, config_data: Optional[Dict] = None):
        # Default LLM configuration
        self.model = "gemma3:12b"
        self.temperature = 0.0

        # Default Prompt configuration
        self.prompt = "What is 1+1?"
        self.template = "Format the following message: {source} into the style: {style}"
        self.source = ""
        self.style = ""

        # Default Schema configuration
        self.schema_name = ""
        self.schema_template = ""

        # Custom default settings if provided
        if config_data and "defaults" in config_data:
            defaults = config_data["defaults"]
            self.model = defaults.get("model", self.model)
            self.temperature = defaults.get("temperature", self.temperature)
            self.prompt = defaults.get("prompt", self.prompt)
            self.template = defaults.get("template", self.template)
            self.source = defaults.get("source", self.source)
            self.style = defaults.get("style", self.style)
            self.schema_name = defaults.get("schema_name", self.schema_name)
            self.schema_template = defaults.get("schema_template", self.schema_template)

            # Store the complete config data for other components to access
            self._config_data = config_data or {}

    def get_schemas(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Retrieve schema definitions from the configuration.

        Returns:
            dict: A dictionary where keys are schema names and values are lists of schema definitions
        """
        return self._config_data.get("schemas", {})


class LLMClient:
    """Base client for interacting with language models."""

    def __init__(self, config: Optional[Union[Config | Dict]] = None):
        """
        Initialize the LLM client.

        Args:
            config (Config or dict, optional): Configuration object or dictionary
        """
        self.params = Config(config) if isinstance(config, dict) else config or Config()
        self.model = self.params.model
        self.temperature = self.params.temperature
        self._chat_model = None

    @property
    def chat_model(self):
        """Lazy-loaded chat model instance."""
        self._chat_model = self._chat_model or ChatOllama(
            model=self.model, temperature=self.temperature
        )
        return self._chat_model

    @handle_exception
    def get_completion(self, prompt: Optional[str] = None) -> str:
        """
        Get completion from a simple prompt.

        Args:
            prompt (str, optional): The text prompt to send to the model, falls back to simplistic default prompt

        Returns:
            str: The model's response as a string
        """
        messages = [HumanMessage(content=prompt or self.params.prompt)]
        response = self.chat_model.invoke(messages)
        return response.content

    @handle_exception
    def chat(self, messages: List[HumanMessage]) -> str:
        """
        Send a list of messages to the chat model.

        Args:
            messages (list[HumanMessage]): List of formatted messages

        Returns:
            str: The model's response as a string
        """
        response = self.chat_model.invoke(messages)
        return response.content


class PromptManager:
    """Manager for creating and formatting prompt templates."""

    def __init__(self, config: Optional[Union[Config | Dict]] = None):
        """
        Initialize the prompt manager.

        Args:
            config (Config or dict, optional): Configuration object or dictionary
        """
        self.params = Config(config) if isinstance(config, dict) else config or Config()
        self._templates = {}

        # Preload default template if available
        if self.params.template:
            self._templates["default"] = self.create_template(self.params.template)

    @staticmethod
    def create_template(template_string: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template from a string.

        Args:
            template_string (str): The template string with variables in {curly_braces}

        Returns:
            ChatPromptTemplate: A ChatPromptTemplate object
        """
        return ChatPromptTemplate.from_template(template_string)

    @staticmethod
    def format_messages(template: ChatPromptTemplate, **kwargs) -> List[HumanMessage]:
        """
        Format a template with values.

        Args:
            template (ChatPromptTemplate): The prompt template
            **kwargs: The values to fill into the template

        Returns:
            list: Formatted messages ready to send to the model
        """
        return template.format_messages(**kwargs)

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


class OutputParser:
    """Parser for structured outputs from language models."""

    def __init__(self, config: Optional[Union[Config | Dict]] = None):
        """
        Initialize the output parser.

        Args:
            config (Config or dict, optional): Configuration object or dictionary
        """
        self.params = Config(config) if isinstance(config, dict) else config or Config()
        self._parsers = {}

        # Preload parsers from config schemas
        for schema_name, schema_def in self.params.get_schemas().items():
            self._parsers[schema_name] = self.create_json_parser(schema_def)

    @staticmethod
    def create_json_parser(
        schema_definitions: List[Dict[str, str]],
    ) -> StructuredOutputParser:
        """
        Create a parser for JSON-formatted outputs.

        Args:
            schema_definitions (list): List of dictionaries containing schema definitions
            Each dict should have 'name' and 'description' keys

        Returns:
            StructuredOutputParser: A configured StructuredOutputParser
        """
        schemas = [
            ResponseSchema(name=schema["name"], description=schema["description"])
            for schema in schema_definitions
        ]
        return StructuredOutputParser.from_response_schemas(schemas)

    @staticmethod
    def get_format_instructions(parser: StructuredOutputParser) -> str:
        """
        Get formatting instructions for a given parser.

        Args:
            parser (StructuredOutputParser): The parser to get instructions for

        Returns:
            str: Formatting instructions as a string
        """
        return parser.get_format_instructions()

    @staticmethod
    def parse_output(parser: StructuredOutputParser, output: str) -> Dict[str, Any]:
        """
        Parse structured output from a model response.

        Args:
            parser (StructuredOutputParser): The parser to use
            output (str): The string output from the model

        Returns:
            dict: A dictionary containing the parsed data
        """
        return parser.parse(output)

    @handle_exception
    def get_parser(self, name: str) -> Optional[StructuredOutputParser]:
        """
        Get a preloaded parser by name.

        Args:
            name (str): Name of the parser/schema

        Returns:
            StructuredOutputParser (optional): The parser or None if not found
        """
        return self._parsers.get(name)


class TextProcessor:
    """High-level interface for common text processing tasks."""

    def __init__(self, config: Optional[Union[Config | Dict]] = None):
        """
        Initialize the text processor.

        Args:
            config (Config or dict, optional): Configuration object or dictionary
        """
        self.params = Config(config) if isinstance(config, dict) else config or Config()
        self.llm_client = LLMClient(self.params)
        self.prompt_manager = PromptManager(self.params)
        self.output_parser = OutputParser(self.params)

    @handle_exception
    def translate_text(
        self, text: Optional[str] = None, style: Optional[str] = None
    ) -> str:
        """
        Translate text to a different style.

        Args:
            text (str, optional): The text to translate, defaults to config.source if None
            style (str, optional): The target style description, defaults to config.style if None

        Returns:
            str: Translated text
        """
        text = text or self.params.source
        target_style = style or self.params.style

        template = self.prompt_manager.get_template() or self.prompt_manager.create_template(
            """Translate the text that is delimited by triple backticks 
            into a style that is {style}.
            text: ```{text}```
            """
        )
        messages = self.prompt_manager.format_messages(
            template, style=target_style, text=text
        )
        return self.llm_client.chat(messages)

    @handle_exception
    def extract_structured_info(
        self, text: str, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from text using configured schemas.

        Args:
            text (str): The text to analyze
            schema_name (str, optional): Name of the schema to use, defaults to config.schema_name if None

        Returns:
            Dictionary with extracted information
        """
        schema_to_use = schema_name or self.params.schema_name
        parser = self.output_parser.get_parser(schema_to_use)

        if not parser:
            raise ValueError(f"Schema '{schema_to_use}' not found in configuration")

        format_instructions = self.output_parser.get_format_instructions(parser)

        template_str = (
            self.params.schema_template
            or """For the following text, extract the following information:
            {format_instructions}
            
            text: {text}
            """
        )

        template = self.prompt_manager.create_template(template_str)
        messages = self.prompt_manager.format_messages(
            template, text=text, format_instructions=format_instructions
        )

        response = self.llm_client.chat(messages)
        return self.output_parser.parse_output(parser, response)

    # For backward compatibility
    @handle_exception
    def extract_review_info(self, review_text: str) -> Dict[str, Any]:
        """
        Extract information from a product review (legacy method).

        Args:
            review_text (str): The review text to analyze

        Returns:
            Dictionary with extracted information
        """
        return self.extract_structured_info(review_text, "product_review")
