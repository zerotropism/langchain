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
        self.schemas = (
            config.get("schemas") if config else ConfigManager().get("schemas")
        )
        self.parsers = {}
        for schema_name, schema_definitions in self.schemas.items():
            # Each schema definition should be a list of dictionaries
            self.parsers[schema_name] = self.create_json_parser(schema_definitions)

    @staticmethod
    @handle_exception
    def create_json_parser(
        schema_definitions: List[Dict[str, str]],
    ) -> StructuredOutputParser:
        """
        Create a parser for JSON-formatted outputs.

        Take a list of schema definitions and return a StructuredOutputParser object.
        Args:
            schema_definitions (`list`): List of dictionaries containing schema definitions
            Each dict should have 'name' and 'description' keys
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

        Take a parser and return its formatting instructions as a string.
        Args:
            parser (`StructuredOutputParser`): The parser to get instructions from
        """
        return parser.get_format_instructions()

    @staticmethod
    @handle_exception
    def parse_output(parser: StructuredOutputParser, output: str) -> Dict[str, Any]:
        """
        Parse structured output from a model response.

        Take a parser and a string output from the model, and return a dictionary
        Args:
            parser (`StructuredOutputParser`): The parser to use
            output (`str`): The string output from the model
        """
        return parser.parse(output)

    @handle_exception
    def get_parser(self, name: str) -> Optional[StructuredOutputParser]:
        """
        Get a preloaded parser by name.

        Take a name and return the corresponding parser if it exists.
        Args:
            name (`str`): Name of the parser/schema
        """
        return self.parsers.get(name)
