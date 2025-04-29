from typing import List, Dict, Optional
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

        # Default Example configuration
        self.examples = {}

        # Custom settings if provided
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

    @property
    def get_model_params(self) -> Dict[str, str]:
        """
        Retrieve the model name and temperature from the configuration.
        Returns:
            dict: The model name and temperature settings
        """
        return {
            "model": self.model or self._config_data.get("model"),
            "temperature": self.temperature or self._config_data.get("temperature"),
        } or {}

    @property
    def get_prompt_params(self) -> Dict[str, str]:
        """
        Retrieve the prompt and template from the configuration.
        Returns:
            dict: The last  prompt and template loaded settings
        """
        return {
            "prompt": self.prompt or self._config_data.get("prompt"),
            "template": self.template or self._config_data.get("template"),
            "source": self.source or self._config_data.get("source"),
            "style": self.style or self._config_data.get("style"),
        } or {}

    @property
    def get_schemas(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Retrieve schema definitions from the configuration.
        Returns:
            dict: A dictionary where keys are schema names and values are lists of schema definitions
        """
        return self._config_data.get("schemas", {})

    @property
    def get_examples(self) -> Dict[str, Dict[str, str]]:
        """
        Retrieve example data from the configuration.
        Returns:
            dict: A dictionary where keys are example names and values are dictionaries of example data
        """
        return self._config_data.get("examples", {})
