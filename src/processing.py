from typing import Optional, Dict, Any

# Specific modules and local imports
from llm import LLMClient
from memory import MemoryFactory
from config import ConfigManager
from parsing import OutputParser
from prompting import PromptManager
from history import MessageHistoryMemoryManager
from decorators import handle_exception, timing_decorator


class TextProcessor:
    """High-level interface for common text processing tasks."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the text processor.

        Args:
            config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
        """
        self.config = config or ConfigManager()
        self.llm_client = LLMClient(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.output_parser = OutputParser(self.config)
        self.memory_manager = MemoryFactory(self.config)
        self.history_manager = MessageHistoryMemoryManager(self.config)

    @handle_exception
    @timing_decorator
    def generate(self, prompt: Optional[Any] = None, **kwargs) -> str:
        """
        Generate text based on a prompt. LLMClient class to send a prompt to, and return
        a response from, the model.

        Args:
            prompt: The text prompt to send to the model, defaults to one_shot example
                from config file  if None
            **kwargs: Additional keyword arguments to pass to the prompt manager

        Returns:
            str: The model's response as a string
        """
        formatted_prompt = self.prompt_manager.formatter(
            prompt=prompt or self.config.get("prompts", "one_shot"), **kwargs
        )
        llm = self.llm_client.infer()
        response = llm.invoke(formatted_prompt)
        return response.content

    @handle_exception
    @timing_decorator
    def translate(
        self,
        usecase: Optional[str] = None,
        text: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str:
        """
        Translate text to a different style.

        Take a usecase name from config file or a custom text and style to translate.
        Args:
            usecase (`str`, optional): The use case to translate, defaults to
                translate example from config file if None
            text (`str`, optional): The text to translate, defaults to
                config.source if None
            style (`str`, optional): The style to use for translation,
                defaults to config.style if None
        """
        # Get prompt settings ie. source text, style to use and assignment template
        example = self.prompt_manager.get_example("translate", usecase or "pirate")
        text = example["source"] or text
        style = example["style"] or style

        # Get template if raw string provided in config file or create a default one on the fly
        template = self.prompt_manager.get_template(
            "translate"
        ) or self.prompt_manager.get_template("default")

        # Format messages with gathering prompt settings
        messages = self.prompt_manager.formatter(template, style=style, text=text)
        return self.generate(messages)

    @handle_exception
    @timing_decorator
    def extract(
        self, text: Optional[str] = None, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information by following schematic instructions.

        Take a target text and a schema name, and return a dictionary with the extracted
        information.
        Args:
            text (`str`, optional): The text to analyze,
                defaults to config.source if None
            schema_name (`str`, optional): Name of the schema to use,
                defaults to config.schema_name if None
        """
        # Get prompt settings ie. source text, schema to use and assignment template
        example = self.prompt_manager.get_example(
            "extract", schema_name or "product_review"
        )
        text = text or example["source"]
        schema_name = schema_name or example["schema"]
        parser = self.output_parser.get_parser(schema_name)

        if not parser:
            raise ValueError(f"Schema '{schema_name}' not found in configuration")

        format_instructions = self.output_parser.get_format_instructions(parser)

        # Get template if raw string provided in config file or create a default one on the fly
        template = self.prompt_manager.get_template(
            "extract"
        ) or self.prompt_manager.create_template(
            """For the following text, extract the following information:
            {format_instructions}
            
            text: {text}
            """
        )

        # Format prompt with target text and instructions
        messages = self.prompt_manager.formatter(
            template, text=text, format_instructions=format_instructions
        )

        # Get response from LLM
        response = self.generate(messages)
        return self.output_parser.parse_output(parser, response)

    @handle_exception
    @timing_decorator
    def chat_legacy_memory(
        self,
        custom_llm: Optional[LLMClient] = None,
        custom_memory: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Start a legacy-mode, memory-capable chat instance.

        Args:
            custom_llm (`LLMClient`, optional): Custom LLM client to use,
                defaults to the one from config
            custom_memory (`str`, optional): Type of memory manager to use,
                defaults to "buffer"
            custom_system_prompt (`str`, optional): Custom system prompt use,
                defaults to the one from config
            verbose (`bool`, optional): Whether to print detailed information,
                defaults to False
        """
        # Get proper llm, memory & system prompt parameters, defaulting to config values
        llm = custom_llm or self.llm_client
        memory = (
            custom_memory.lower()
            if custom_memory and type(custom_memory) == str
            else self.memory_manager.memory_type
        )
        system_prompt = (
            custom_system_prompt or self.prompt_manager.prompt_templates.get("system")
        )

        # Create the appropriate memory manager
        chatbot = self.memory_manager.build(llm, memory, verbose=verbose)

        # Add system prompt to memory
        chatbot.add_to_memory(
            user_input=system_prompt
            or self.prompt_manager.prompt_templates.get("system"),
            ai_output="I'll keep it professional from now on. What can I assist you with today?",
        )

        print(
            f"You can now start chatting with the model '{self.config.get("model", "name")}'.\
            Type 'exit' to quit.\n"
        )

        # Start the chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = chatbot.predict(user_input)
            print(f"AI: {response}")
            chatbot.add_to_memory(user_input, response)

    def chat_legacy_history(
        self,
        custom_llm: Optional[LLMClient] = None,
        custom_memory: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Start a legacy-mode, history-capable chat instance.

        Args:
            custom_llm (`LLMClient`, optional): Custom LLM client to use,
                defaults to the one from config
            custom_memory (`str`, optional): Type of memory manager to use,
                defaults to "buffer"
            custom_system_prompt (`str`, optional): Custom system prompt use,
                defaults to the one from config
            verbose (`bool`, optional): Whether to print detailed information,
                defaults to False
        """
        # Set parameters for the chat instance
        session_id = "default"
        system_prompt = (
            custom_system_prompt or self.prompt_manager.prompt_templates.get("system")
        )

        print(
            f"Vous pouvez maintenant discuter avec le modèle '{self.config.get('model', 'name')}'.\nTapez 'exit' pour quitter.\n"
        )

        # Start the chat loop
        first_turn = True
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            if first_turn:
                messages = self.prompt_manager.build_chat_messages(
                    system_prompt=system_prompt, user_prompt=user_input
                )
                first_turn = False
            else:
                messages = self.prompt_manager.build_chat_messages(
                    user_prompt=user_input
                )

            response = self.history_manager.runnable.invoke(
                messages,
                config={"configurable": {"session_id": session_id}},
            )
            print(f"AI: {response.content}")
