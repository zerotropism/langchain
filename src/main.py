import os
import yaml
from typing import Dict
from config import ConfigManager
from processing import TextProcessor
from rag import RAGSystem


def mode_selector():
    modes = [
        ("prompt", "Generate text based on a template you can build."),
        ("chat_memory", "Chat with an LLM (legacy: simple memory)."),
        ("chat_history", "Chat with an LLM (legacy: conversation history)."),
        (
            "simple_rag",
            "Query a document-based retrieval-augmented generator you will setup.",
        ),
        ("chat_rag", "Chat with RAG (only) capabilities model."),
        ("chat_rag_memory", "Chat with RAG & memory capabilities model."),
        (
            "agent",
            "Setup & converse with a specialized Agent. (available: basic math solver, wikipedia searcher, python coder or custom)",
        ),
        (
            "evaluate",
            "Setup an automated LLM-based evaluator for a simplistic RAG output.",
        ),
    ]
    print("What do you want to do?")
    for idx, (mode, desc) in enumerate(modes, 1):
        print(f"({idx}) - {mode}: {desc}")
    choice = input("Enter the number of your choice: ").strip()
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(modes):
            return modes[choice_num - 1][0]
        else:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid number. Please select a valid number.")


def load_configurations(path: str = "src/config.yml") -> Dict:
    """Load configuration from a YAML file.

    Take a filepath string and return a dictionary with the configuration settings.

    Args:
        path (`str`, optional): path to the YAML file, defaults to "config.yml"

    Returns:
        Dict: A dictionary with the configuration settings
    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def prompt(conf: ConfigManager):
    """Prompt mode, simplest way to interact with the LLM."""
    # The processor is a high-level interface for common text processing tasks
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


def chat_memory(conf: ConfigManager):
    """Chat mode, legacy memory capable."""
    processor = TextProcessor(conf)
    processor.chat_legacy_memory()
    return


def chat_history(conf: ConfigManager):
    """Chat mode, legacy memory capable."""
    processor = TextProcessor(conf)
    processor.chat_legacy_history()
    return


def simple_rag(conf: ConfigManager):
    """RAG mode."""
    # Perform a standalone manual search and generate an answer using the LLM
    rag = RAGSystem(config=conf)
    query = "Please list all your shirts with sun protection and summarize each one"
    print(f"\nDemo search on example query: '{query}'\n")
    rag.direct_search_and_answer(query).pretty_print()


def chat_rag(conf: ConfigManager):
    """Chat mode with RAG capabilities."""
    # Converse with an LLM that has RAG capabilities
    rag = RAGSystem(config=conf)
    rag.chat_with_rag()
    return


def chat_rag_memory(conf: ConfigManager):
    """Chat mode with RAG capabilities."""
    # Converse with an LLM that has RAG capabilities
    rag = RAGSystem(config=conf)
    rag.chat_with_rag_hybrid()
    return


def agent(conf: ConfigManager):
    """Agent mode."""
    # Implement the logic for agent mode here
    pass


def evaluate(conf: ConfigManager):
    """Evaluate mode."""
    # Implement the logic for evaluate mode here
    pass


def run_mode(mode: str, conf: ConfigManager):
    function = globals().get(mode)
    if callable(function):
        function(conf)
    else:
        print(f"'{mode}' has no implemented function.")


def main():
    valid_modes = [
        "prompt",
        "chat_memory",
        "chat_history",
        "simple_rag",
        "chat_rag",
        "chat_rag_memory",
        "agent",
        "evaluate",
    ]
    mode = mode_selector().strip().lower()

    # Raise an error if mode is not valid
    if not mode:
        raise ValueError(
            f"Mode must be specified. Available modes: {', '.join(valid_modes)}"
        )

    elif mode not in valid_modes:
        raise ValueError(f"Invalid mode. Available modes: {', '.join(valid_modes)}")

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
