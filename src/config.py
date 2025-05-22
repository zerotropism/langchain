from typing import Dict, Optional


class ConfigManager:
    """Configuration class for managing settings."""

    def __init__(self, config_data: Optional[Dict] = None):
        """Initialize the configuration manager."""
        self._config = config_data or {}

    def get(self, section: str, key: Optional[str] = None):
        """Generic method to retrieve configuration values from the config file.
        Args:
            section (str): The section of the configuration to retrieve from.
            key (str, optional): The specific key within the section.
                If None, return the entire section.
        """
        section_data = self._config.get(section, {})
        if key:
            return section_data.get(key, "default")
        return section_data

    def get_prompt(self, name: str = "default"):
        """Retrieve a prompt template by name."""
        return self.get("prompts", name)

    def get_example(self, task: str, name: Optional[str] = None):
        """Retrieve an example by task and name."""
        examples = self.get("examples", task, [])
        if name:
            for ex in examples:
                if ex.get("name") == name:
                    return ex
            return None
        return examples

    def get_schema(self, name: str):
        """Retrieve a schema by name."""
        return self.get("schemas", name)
