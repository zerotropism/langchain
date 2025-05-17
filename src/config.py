from typing import List, Dict, Optional


class ConfigManager:
    """Configuration class for managing global settings."""

    def __init__(self, config_data: Optional[Dict] = None):
        # Default LLM configuration
        self._model = "gemma3:12b"
        self._temperature = 0.0

        # Default Memory configuration
        self._memory = "buffer"

        # Default Prompt configuration
        self._prompt = "What is 1+1?"
        self._template = (
            "Format the following message: {source} into the style: {style}"
        )
        self._source = ""
        self._style = ""

        # Default Schema configuration
        self._schema_name = ""
        self._schema_template = ""

        # Default Example configuration
        self._examples = {}

        # Custom settings if provided
        if config_data and "defaults" in config_data:
            defaults = config_data["defaults"]
            self._model = defaults.get("model", self._model)
            self._temperature = defaults.get("temperature", self._temperature)
            self._memory = defaults.get("memory", self._memory)
            self._prompt = defaults.get("prompt", self._prompt)
            self._template = defaults.get("template", self._template)
            self._source = defaults.get("source", self._source)
            self._style = defaults.get("style", self._style)
            self._schema_name = defaults.get("schema_name", self._schema_name)
            self._schema_template = defaults.get(
                "schema_template", self._schema_template
            )

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
            "model": self._model or self._config_data.get("model"),
            "temperature": self._temperature or self._config_data.get("temperature"),
        } or {}

    @property
    def get_prompt_params(self) -> Dict[str, str]:
        """
        Retrieve the prompt and template from the configuration.

        Returns:
            dict: The last  prompt and template loaded settings
        """
        return {
            "prompt": self._prompt or self._config_data.get("prompt"),
            "template": self._template or self._config_data.get("template"),
            "source": self._source or self._config_data.get("source"),
            "style": self._style or self._config_data.get("style"),
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

    @property
    def get_memory_params(self) -> Dict[str, str]:
        """
        Retrieve the memory settings from the configuration.

        Returns:
            str: The memory type
        """
        return self._config_data.get("memory", self.memory)

    @property
    def get_config_data(self) -> Dict:
        """
        Retrieve the complete configuration data.

        Returns:
            dict: The complete configuration data
        """
        return self._config_data or {}
