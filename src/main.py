import os
import yaml
from typing import Dict
from config import Config
from parsing import LLMClient, TextProcessor
from memory import MemoryFactory


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


def prompt(conf: Config):
    """Prompt mode.
    Args:
        conf (dict): configurations to pass to the function
    """
    # instantiate the LLM client
    llm = LLMClient(conf)

    # Get a simple completion
    simple_completion = llm.get_completion()
    print("\nSimple completion:\n", simple_completion)

    # Instantiate the template-based text processor
    processor = TextProcessor(conf)

    # Get a completion with a template
    templated_completion = processor.translate_text()
    print("\nTemplated completion (translation):\n", templated_completion)

    # Extract information from a text from same processor
    product_review = conf.get_examples.get("product_review", {}).get("source", "")
    if not product_review:
        print("Error: 'product_review' source not found in configuration.")
        return
    extracted_info = processor.extract_review_info(product_review)
    print("\nExtracted information from product review:")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")

    return


def chat(conf: Config):
    """Chat mode.
    Args:
        conf (dict): configurations to pass to the function
    """
    # Create a buffer memory manager
    buffer_memory = MemoryFactory.create_memory_manager("buffer", verbose=True)

    # Demonstration of buffer memory
    buffer_memory.predict("Hi, my name is Philantenne!")
    buffer_memory.predict("What is 1+1?")
    response = buffer_memory.predict("What is my name?")
    print(f"Response: {response}")
    print(f"Memory: {buffer_memory.get_memory_content()}")

    # Create a window memory manager with window size 2
    window_memory = MemoryFactory.create_memory_manager("window", window_size=2)

    # Demonstration of window memory
    window_memory.predict("Hi, my name is Philantenne!")
    window_memory.predict("What is 1+1?")
    window_memory.predict("What is my name?")
    print(f"Window Memory: {window_memory.get_memory_content()}")

    # Create a token memory manager
    token_memory = MemoryFactory.create_memory_manager("token", max_token_limit=50)

    # Manually add context to token memory
    token_memory.add_to_memory("AI is what?!", "Amazing!")
    token_memory.add_to_memory("Backpropagation is what?", "Beautiful!")
    token_memory.add_to_memory("Chatbots are what?", "Charming!")
    print(f"Token Memory: {token_memory.get_memory_content()}")

    # Create a summary memory manager
    summary_memory = MemoryFactory.create_memory_manager("summary", max_token_limit=100)

    # Add a long schedule to summary memory
    schedule = """There is a meeting at 8am with your product team. 
    You will need your powerpoint presentation prepared. 
    9am-12pm have time to work on your LangChain project which will go quickly 
    because Langchain is such a powerful tool. At Noon, lunch at the italian 
    restaurant with a customer who is driving from over an hour away to meet 
    you to understand the latest in AI. Be sure to bring your laptop to show 
    the latest LLM demo."""

    summary_memory.add_to_memory("What is on the schedule today?", schedule)
    print(f"Summary Memory: {summary_memory.get_memory_content()}")

    response = summary_memory.predict("What would be a good demo to show?")
    print(f"Response about demo: {response}")
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
        mode (str): mode as function name to run
        conf (dict): configurations to pass to the function
    """
    try:
        # Get the function corresponding to the mode
        function = globals().get(mode)
        if callable(function):
            # Call the function with the configurations
            function(conf)
    except KeyError:
        print(f"'{mode}' has no implemented function.")


def main():
    mode = mode_selector()

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
        conf = Config(conf_data)

        # Run the appropriate mode
        run_mode(mode, conf)


if __name__ == "__main__":
    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    main()
