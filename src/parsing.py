from typing import Dict, List, Optional, Any
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from interfaces import IConfigManager, IOutputParser
from decorators import handle_exception


class OutputParser(IOutputParser):
    """Parser for structured outputs from language models."""

    def __init__(self, config: Optional[IConfigManager] = None):
        """
        Initialize the output parser.

        Args:
            config (ConfigManager, optional): Pre-loaded settings from `./config.yml` file
        """
        from config import ConfigManager

        config_manager = config or ConfigManager()
        self.schemas = config_manager.get("schemas")
        self.parsers = {}

        # Preload parsers for each schema defined in the configuration
        if self.schemas:
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

        Args:
            schema_definitions (list): List of dictionaries containing schema definitions
                Each dict should have 'name' and 'description' keys

        Returns:
            StructuredOutputParser: A parser configured with the provided schema definitions
        """
        schemas = [
            ResponseSchema(name=schema["name"], description=schema["description"])
            for schema in schema_definitions
        ]
        return StructuredOutputParser.from_response_schemas(schemas)

    @handle_exception
    def get_parser(self, schema_name: str) -> Optional[StructuredOutputParser]:
        """
        Get a preloaded parser by name.

        Take a name and return the corresponding parser if it exists.

        Args:
            name (str): Name of the parser/schema

        Returns:
            Optional[StructuredOutputParser]: The parser if found, otherwise None.
        """
        if schema_name not in self.parsers:
            if self.schemas and schema_name in self.schemas:
                # Parser not loaded yet but schema exists, let's create it
                self.parsers[schema_name] = self.create_json_parser(
                    self.schemas[schema_name]
                )
            else:
                print(
                    f"Schema '{schema_name}' not found. Available schemas: {list(self.schemas.keys() if self.schemas else [])}"
                )
                return None
        return self.parsers.get(schema_name)

    @handle_exception
    def get_format_instructions(self, parser: StructuredOutputParser) -> str:
        """
        Get formatting instructions for a given parser.

        Take a parser and return its formatting instructions as a string.

        Args:
            parser (StructuredOutputParser): The parser to get instructions from

        Returns:
            str: Formatting instructions for the parser
        """
        if not parser:
            return ""
        return parser.get_format_instructions()

    @handle_exception
    def parse_output(
        self, parser: StructuredOutputParser, response: str
    ) -> Dict[str, Any]:
        """
        Parse structured output from a model response.

        Take a parser and a string output from the model, and return a dictionary

        Args:
            parser (StructuredOutputParser): The parser to use
            output (str): The string output from the model

        Returns:
            dict: Parsed output as a dictionary
        """
        if not parser:
            return {}
        try:
            return parser.parse(response)
        except Exception as e:
            print(f"Error parsing output: {e}")
            return {"error": str(e), "raw_response": response}
