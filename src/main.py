import yaml
from typing import Dict
from parsing import Parser
from decorators import handle_exception


def load_configurations(path: str = "src/config.yaml") -> Dict:
    """Load configurations from a YAML file.
    Args:
        path (str, optional): path to the YAML file. Defaults to "config.yaml".
    Returns:
        Dict: configurations as a dictionary.
    """
    with open(path, "r") as file:
        conf = yaml.safe_load(file)
    return conf


def prompt(conf: Dict):
    """Prompt mode."""
    # Implement the logic for prompt mode here
    pass


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
        # Initialize parser
        parser = Parser(conf)
        # Check if params are set correctly
        print(parser.params())
        # Test parser on simple completion
        print(parser.get_completion(input()))


if __name__ == "__main__":
    print(
        """
        What do you want to do? (i.e. prompt, chat, rag, agent, evaluate)
            prompt: Generate text based on a template you can build.
            chat: Chat with an LLM.
            rag: Query a document-based retrieval-augmented generator you will setup.
            agent: Setup & converse with a specialized Agent (available: basic math solver, wikipedia searcher, python coder or custom).
            evaluate: Setup an automated LLM-based evaluator for a simplistic RAG output.
        Input your choice:
        """
    )
    mode = input()
    main(mode)
