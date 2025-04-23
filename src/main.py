import yaml
from typing import Dict
from parsing import LLMClient, TextProcessor


def load_configurations(path: str = "src/config.yml") -> dict:
    """Load configuration from a YAML file.

    Args:
        path (str, optional): path to the YAML file, defaults to "config.yml"

    Returns:
        Dict: configurations as a dictionary
    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def prompt(conf: Dict):
    """Prompt mode."""
    # Get a simple completion
    llm = LLMClient(conf)
    simple_completion = llm.get_completion()
    print("\nSimple completion:\n", simple_completion)

    # Get a completion with a template
    processor = TextProcessor(conf)
    templated_completion = processor.translate_text()
    print("\nTemplated completion (translation):\n", templated_completion)

    # Extract information from a text
    product_review = (
        conf.get("examples", {}).get("product_review", {}).get("source", "")
    )
    if not product_review:
        print("Error: 'product_review' source not found in configuration.")
        return
    extracted_info = processor.extract_review_info(product_review)
    print("\nExtracted information from product review:")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")

    return


def chat(conf: Dict):
    """Chat mode."""
    # Implement the logic for chat mode here
    pass


def rag(conf: Dict):
    """RAG mode."""
    # Implement the logic for RAG mode here
    pass


def agent(conf: Dict):
    """Agent mode."""
    # Implement the logic for agent mode here
    pass


def evaluate(conf: Dict):
    """Evaluate mode."""
    # Implement the logic for evaluate mode here
    pass


def run_mode(mode: str, conf: Dict):
    """Run the specified mode.
    Args:
        mode (str): mode to run
        conf (dict): configurations
    """
    try:
        # Get the function corresponding to the mode
        function = globals().get(mode)
        if callable(function):
            # Call the function with the configurations
            function(conf)
    except KeyError:
        print(f"'{mode}' has no implemented function.")


def main(mode: str = ""):
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
        conf = load_configurations(path="./config.yml")

        # Run the appropriate mode
        run_mode(mode, conf)


if __name__ == "__main__":
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
    main(mode)
