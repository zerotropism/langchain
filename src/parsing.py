from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from decorators import handle_exception
from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class ParamsConfig:
    """Configuration parameters for the Parser class."""

    # Default model configuration
    model: str = "gemma3:12b"
    temperature: float = 0.0

    # Default prompts
    prompt: str = "What is 1+1?"
    template: str = "Format the following message: {source} into the style: {style}"
    source: str = ""
    style: str = ""

    # Default schemas
    schema_name: str = ""
    schema_template: str = (
        "For the following text, extract information according to "
        "these instructions:\n{format_instructions}\n\nText: {text}"
    )


class Parser:
    """A class for handling LLM prompts, completions, and output parsing.

    This class provides methods for interacting with language models,
    formatting prompts, and parsing structured responses.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        """Initializes the Parser class.
        Args:
            config (dict, optional) : Pre-loaded configuration dictionary
        """
        # Load default parameters configuration
        self._params = ParamsConfig()
        self._model_name = self._params.model
        self._model_temperature = self._params.temperature
        self._template = self._params.template
        self._schemas = {}
        self._examples = {}

        # Try and load configuration from the provided dictionary
        if config:
            self.load_params_from_config(config)
        else:
            print("Default configuration loaded")

    @property
    def params(self) -> ParamsConfig:
        """The current parameters configuration."""
        return self._params

    def update_params(self, **kwargs) -> None:
        """Update parameter values in the configuration.
        Args:
            **kwargs: Key-value pairs of parameters to update
        Example:
            parser.update_params(prompt_one_shot="new_prompt")
        """
        for key, value in kwargs.items():
            if hasattr(self._params, key) and value:
                setattr(self._params, key, value)

                # If updating the default template, also update the template object
                if key == "template" and value:
                    self._template = value

    @property
    def model_name(self) -> str:
        """The model name being used."""
        return self._model_name

    @model_name.setter
    def model_name(self, model: str) -> None:
        """Sets the model name to use."""
        self._model_name = model

    @property
    def model_temperature(self) -> float:
        """The current temperature setting for the model."""
        return self._model_temperature

    @model_temperature.setter
    def model_temperature(self, temperature: float) -> None:
        """Sets the temperature setting for the model to apply."""
        self._model_temperature = temperature

    @property
    def template(self) -> ChatPromptTemplate:
        """Returns the current template."""
        if not self._template:
            template_str = (
                self._params.template
                or "Format the following message: {source} into the style: {style}"
            )
            self._template = ChatPromptTemplate.from_template(template_str)
        return self._template

    @template.setter
    def template(self, template_str: str) -> None:
        """Set a new prompt template."""
        if template_str:
            self._template = ChatPromptTemplate.from_template(template_str)

    @property
    def examples(self) -> Dict[str, Dict[str, str]]:
        """Examples for prompting."""
        return self._examples

    @property
    def schemas(self) -> Dict[str, List[Dict[str, str]]]:
        """Schema definitions for structured parsing."""
        return self._schemas

    def _get_llm(
        self, model: Optional[str] = None, temperature: Optional[float] = None
    ) -> ChatOllama:
        """Get a configured LLM instance.
        Args:
            model (str, optional): Override the default model name
            temperature (float, optional): Override the default temperature
        Returns:
            Configured ChatOllama instance
        """
        return ChatOllama(
            model=model or self._model_name,
            temperature=temperature or self._model_temperature,
        )

    @handle_exception
    def load_params_from_config(self, config: dict = None) -> None:
        """Load configuration from a pre-loaded config dictionary.
        Args:
            config (dict): Pre-loaded and pre-processed configuration
        """
        # Update params from defaults section
        if "defaults" in config:
            params_update = {
                "model": config["defaults"].get("model"),
                "temperature": config["defaults"].get("temperature"),
                "prompt": config["defaults"].get("prompt"),
                "template": config["defaults"].get("template"),
                "source": config["defaults"].get("source"),
                "style": config["defaults"].get("style"),
                "schema_name": config["defaults"].get("schema_name"),
                "schema_template": config["defaults"].get("schema_template"),
            }
            # Filter out None values before updating
            params_update = {k: v for k, v in params_update.items() if v is not None}
            self.update_params(**params_update)

        # Load examples if present
        if "examples" in config:
            self._examples = config["examples"]

        # Load schemas if present
        if "schemas" in config:
            self._schemas = config["schemas"]

        # Set template if default template was loaded
        if self._params.template:
            self._template = self._params.template

    @handle_exception
    def get_completion(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Get a simple completion from the LLM based on the provided prompt.
        Args:
            prompt (str, optional): The prompt text (falls back to `prompt` from params)
            model (str, optional): Override the default model
            temperature (float, optional): Override the default temperature
        Returns:
            str: The completion response from the LLM.
        """
        # Create a ChatOllama instance with the model and temperature
        llm = self._get_llm(
            model=model or self._model_name,
            temperature=temperature or self._model_temperature,
        )

        # Create a list of messages the simplest way ChatOllama class expects
        messages = [HumanMessage(content=prompt if prompt else self._params.prompt)]

        # Get the response from the LLM
        return llm(messages).content

    @handle_exception
    def format_prompt(
        self,
        template: Optional[str] = None,
        source: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str:
        """Format a prompt with the provided template, source and style.
        Args:
            template_prompt (str): The template to use for formatting
            source_prompt (str): The source text to format
            target_style (str): The target style description
        Returns:
            str: The formatted prompt as the source prompt formatted to the desired style according to the template.
        """
        # Create a ChatPromptTemplate instance with the provided template or the default one
        template = template or self._template or self._params.template
        return self._template.format_messages(
            style=style or self._params.style,
            source=source or self._params.source,
        )

    @handle_exception
    def get_formatted_completion(
        self,
        source: Optional[str] = None,
        style: Optional[str] = None,
        custom_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Gets a formatted completion from the LLM based on the provided template, source prompt and target styles.
        Args:
            source (str, optional): The source text to format
            style (str, optional): The target style description
            custom_template (str, optional): Use a one-time custom template
            model (str, optional): Override the default model
            temperature (str, optional): Override the default temperature
        Returns:
            str: The formatted completion response from the LLM.
        """
        # Create a ChatOllama instance with the model and temperature
        llm = self._get_llm(
            model=model or self._model_name,
            temperature=temperature or self._model_temperature,
        )

        # Save current template if using a custom one
        original_template = None
        if custom_template:
            original_template = self._template
            self._template = custom_template

        # Set prompt to passed one or format it with the template
        try:
            formatted_messages = self.format_prompt(source, style)
            result = llm(formatted_messages).content
        finally:
            # Restore original template if we changed it
            if original_template:
                self._template = original_template

        return result

    @handle_exception
    def create_schema_parser(
        self, schema_name: Optional[str] = None
    ) -> StructuredOutputParser:
        """Create a structured output parser with the given schema.
        Args:
            schema_name (str, optional): Name of the schema to use
        Returns:
            StructuredOutputParser: Configured parser for structured output
        """
        # Use provided schema name or default
        schema_name = schema_name or self._params.schema_name

        if not schema_name or schema_name not in self._schemas:
            raise ValueError(f"Schema '{schema_name}' not found in loaded schemas")

        schema_fields = self._schemas[schema_name]
        response_schemas = []

        for field in schema_fields:
            response_schemas.append(
                ResponseSchema(name=field["name"], description=field["description"])
            )

        return StructuredOutputParser.from_response_schemas(response_schemas)

    @handle_exception
    def get_structured_completion(
        self,
        text: str,
        schema_name: Optional[str] = None,
        custom_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get a structured completion from the LLM using a schema parser.
        Args:
            text (str): The text to analyze
            schema_name (str, optional): Name of the schema to use
            custom_template (str, optional): Custom template for this request
            model (str, optional): Override the default model
            temperature (float, optional): Override the default temperature
        Returns:
            Dict[str, Any]: Structured data parsed from the LLM response
        """
        parser = self.create_schema_parser(schema_name)
        format_instructions = parser.get_format_instructions()

        # Use provided template, default schema template, or fallback
        template = (
            custom_template
            or self._params.schema_template
            or (
                "For the following text, extract information according to these instructions:\n"
                "{format_instructions}\n\n"
                "Text: {text}"
            )
        )

        prompt = template.format(format_instructions=format_instructions, text=text)

        llm = self._get_llm(model, temperature)
        messages = [HumanMessage(content=prompt)]
        response = llm(messages).content

        return parser.parse(response)

    @handle_exception
    def get_example(self, example_name: str) -> Dict[str, str]:
        """Get a specific example by name.
        Args:
            example_name (str): Name of the example to retrieve
        Returns:
            Dict[str, str]: The example data
        """
        if example_name not in self._examples:
            raise ValueError(f"Example '{example_name}' not found")
        return self._examples[example_name]

    @handle_exception
    def get_example_completion(
        self,
        example_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Get a formatted completion using a named example.
        Args:
            example_name (str): Name of the example to use
            model (str, optional): Override the default model
            temperature (float, optional): Override the default temperature
        Returns:
            str: The formatted completion
        """
        example = self.get_example(example_name)
        return self.get_formatted_completion(
            source=example.get("source", ""),
            style=example.get("style", ""),
            model=model,
            temperature=temperature,
        )
