import os
import yaml
from typing import Dict
from config import ConfigManager
from processing import TextProcessor


def mode_selector():
    print(
        """
What do you want to do? (i.e. prompt, chat, rag, agent, evaluate)
    - prompt: Generate text based on a template you can build.
    - chat: Chat with an LLM.
    - rag: Query a document-based retrieval-augmented generator you will setup.
    - agent: Setup & converse with a specialized Agent (available: basic math solver, wikipedia searcher, python coder or custom).
    - evaluate: Setup an automated LLM-based evaluator for a simplistic RAG output.
Input your choice:
        """
    )
    mode = input()
    return mode


def load_configurations(path: str = "src/config.yml") -> Dict:
    """Load configuration from a YAML file.

    Take a filepath string and return a dictionary with the configuration settings.
    Args:
        path (`str`, optional): path to the YAML file, defaults to "config.yml"
    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def prompt(conf: ConfigManager):
    """Prompt mode, simplest way to interact with the LLM.

    Generate text based on a template you can build.
    Args:
        conf (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
    """
    # The processor is a high-level interface for common text processing tasks including one-shot prompting
    processor = TextProcessor(conf)

    # Get a simple completion
    simple_completion = processor.generate()
    print("\nSimple completion:\n", simple_completion)

    # Get a completion using a template
    templated_completion = processor.translate()
    print("\nTemplated completion (translation):\n", templated_completion)

    # Extract information from a text using a schema
    extracted_info = processor.extract()
    print("\nExtracted information from product review:")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")

    return


def chat(conf: ConfigManager):
    """Chat mode, continuous conversation with the LLM.

    Generate memomry-based contextualized responses to user queries.
    Args:
        conf (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
    """
    # The processor is a high-level interface for common text processing tasks including one-shot prompting
    processor = TextProcessor(conf)
    processor.chat()

    return


def rag(conf: ConfigManager):
    """RAG mode."""
    # Implement the logic for RAG mode here
    pass


def agent(conf: ConfigManager):
    """Agent mode."""
    # Implement the logic for agent mode here
    pass


def evaluate(conf: ConfigManager):
    """Evaluate mode."""
    # Implement the logic for evaluate mode here
    pass


def run_mode(mode: str, conf: ConfigManager):
    """Run the specified mode.

    Args:
        mode (`str`): mode as a function name to run
        conf (`ConfigManager`): configuration settings from `./config.yml` file
    """
    # Get the function corresponding to the mode
    function = globals().get(mode)
    if callable(function):
        # Call the function with the configurations
        function(conf)
    else:
        print(f"'{mode}' has no implemented function.")


def main():
    mode = mode_selector().strip().lower()

    # Raise an error if mode is not valid
    if not mode:
        raise ValueError(
            "Mode must be specified. Available modes: 'prompt', 'chat', 'rag', 'agent' & 'evaluate'"
        )

    elif mode not in ["prompt", "chat", "rag", "agent", "evaluate"]:
        raise ValueError(
            "Invalid mode. Available modes: 'prompt', 'chat', 'rag', 'agent' & 'evaluate'"
        )

    else:
        # Load configuration file
        conf_data = load_configurations(path="./config.yml")

        # Instantiate the Config class with loaded settings
        conf = ConfigManager(conf_data)

        # Run the appropriate mode
        run_mode(mode, conf)


if __name__ == "__main__":

    # Create local directories for logs
    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    main()
