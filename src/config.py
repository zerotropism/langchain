from typing import Dict, Optional, Any
from interfaces import IConfigManager


class ConfigManager(IConfigManager):
    """Configuration class for managing settings."""

    def __init__(self, config_data: Optional[Dict] = None):
        """Initialize the configuration manager.

        Args:
            config_data (dict): Configuration dictionary, empty dict if None
        """
        self._config = config_data or {}

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Generic method to retrieve configuration values from the config file.

        Args:
            section (str): The section of the configuration to retrieve from.
            key (str, optional): The specific key within the section.
                If None, return the entire section.

        Returns:
            Any: The value associated with the key in the section, or the entire section
                if key is None. Returns "default" if the key does not exist.
        """
        section_data = self._config.get(section, {})
        if key:
            return section_data.get(key, "default")
        return section_data

    def get_prompt(self, name: str = "default") -> str:
        """Retrieve a prompt template by name.

        Args:
            name (str): Name of the prompt template

        Returns:
            str: The prompt template string
        """
        return self.get("prompts", name)

    def get_example(self, task: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve an example by task and name.

        Args:
            task (str): Type of task (e.g., "translate", "extract")
            name (str, optional): Name of the specific example

        Returns:
            dict: The example data or list of examples
        """
        examples = self.get("examples", task, [])
        if name:
            for ex in examples:
                if ex.get("name") == name:
                    return ex
            return None
        return examples

    def get_schema(self, name: str) -> Dict[str, Any]:
        """Retrieve a schema by name.

        Args:
            name (str): Name of the schema

        Returns:
            dict: The schema definition
        """
        return self.get("schemas", name)
