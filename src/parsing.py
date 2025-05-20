"""
LangChain LLM Structured output parsing (OutputParser) integration
"""

from typing import Dict, List, Optional, Any
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from config import ConfigManager
from decorators import handle_exception


class OutputParser:
    """Parser for structured outputs from language models."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the output parser.

        Args:
            config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
        """
        self.parse_settings = config.get_examples or ConfigManager().get_examples
        self.parsers = {}

        # Preload parsers from example schemas
        for name, description in self.parse_settings["examples"]["extract"][
            "schemas"
        ].items():
            self.parsers[name] = self.create_json_parser(description)

    @staticmethod
    @handle_exception
    def create_json_parser(
        schema_definitions: List[Dict[str, str]],
    ) -> StructuredOutputParser:
        """
        Create a parser for JSON-formatted outputs.

        Args:
            schema_definitions (`list`): List of dictionaries containing schema definitions
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
    @handle_exception
    def get_format_instructions(parser: StructuredOutputParser) -> str:
        """
        Get formatting instructions for a given parser.

        Args:
            parser (`StructuredOutputParser`): The parser to get instructions for

        Returns:
            str: Formatting instructions as a string
        """
        return parser.get_format_instructions()

    @staticmethod
    @handle_exception
    def parse_output(parser: StructuredOutputParser, output: str) -> Dict[str, Any]:
        """
        Parse structured output from a model response.

        Args:
            parser (`StructuredOutputParser`): The parser to use
            output (`str`): The string output from the model

        Returns:
            dict: A dictionary containing the parsed data
        """
        return parser.parse(output)

    @handle_exception
    def get_parser(self, name: str) -> Optional[StructuredOutputParser]:
        """
        Get a preloaded parser by name.

        Args:
            name (`str`): Name of the parser/schema

        Returns:
            StructuredOutputParser (optional): The parser or None if not found
        """
        return self.parsers.get(name)
