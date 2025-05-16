from typing import Optional, Dict, Any
from config import ConfigManager
from llm import LLMClient
from prompting import PromptManager
from parsing import OutputParser
from memory import MemoryFactory
from decorators import handle_exception, timing_decorator


class TextProcessor:
    """High-level interface for common text processing tasks."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the text processor.
        Args:
            config (ConfigManager, optional): Pre-loaded settings from the configuration file
        """
        self._params = config
        self.llm_client = LLMClient(self._params)
        self.prompt_manager = PromptManager(self._params)
        self.output_parser = OutputParser(self._params)

    @handle_exception
    @timing_decorator
    def generate(self, prompt: Optional[Any] = None, **kwargs) -> str:
        """
        Generate text based on a prompt.
        This method uses the LLMClient to send a prompt to the model and receive a response.
        Args:
            prompt: The text prompt.s to send to the model, defaults to config.prompt if None
            **kwargs: Additional keyword arguments to pass to the prompt manager
        Returns:
            str: The model's response as a string
        """
        formatted_prompt = self.prompt_manager.formatter(prompt=prompt, **kwargs)
        llm = self.llm_client.infer()
        response = llm.invoke(formatted_prompt)
        return response.content

    @handle_exception
    @timing_decorator
    def chat(self, memory_type: Optional[str] = "buffer"):
        """
        Start a memory-capable chat instance.
        """
        # Check if memory type is passed, otherwise use the default from configuration
        memory_type = (
            memory_type.lower()
            if memory_type
            else self._params.get("memory", {}).get("memory_type", "buffer")
        )

        # Create the appropriate memory manager
        llm_with_memory = MemoryFactory.create_memory_manager(
            self.llm_client, memory_type, verbose=True
        )
        print(
            f"You can now start chatting with the model {self._params._model}. Type 'exit' to quit.\n"
        )

        # Start the chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = llm_with_memory.predict(user_input)
            print(f"AI: {response}")
            llm_with_memory.add_to_memory(user_input, response)

    @handle_exception
    @timing_decorator
    def translate(self, text: Optional[str] = None, style: Optional[str] = None) -> str:
        """
        Translate text to a different style.
        Args:
            text (str, optional): The text to translate, defaults to config.source if None
            style (str, optional): The target style description, defaults to config.style if None
        Returns:
            str: Translated text
        """
        text = text or self._params._source
        style = style or self._params._style

        template = self.prompt_manager.get_template() or self.prompt_manager.create_template(
            """Translate the text that is delimited by triple backticks 
            into a style that is {style}.
            text: ```{text}```
            """
        )
        messages = self.prompt_manager.formatter(template, style=style, text=text)
        return self.generate(messages)

    @handle_exception
    @timing_decorator
    def extract(
        self, text: Optional[str] = None, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from text using configured schemas.
        Args:
            text (str, optional): The text to analyze, defaults to config.source if None
            schema_name (str, optional): Name of the schema to use, defaults to config.schema_name if None
        Returns:
            dict: Dictionary with extracted information
        """
        # Check if text is passed, otherwise use the default example from configuration
        if not text:
            print(
                "You did not provide any text to extract from, defaulting to example text."
            )
            text = self._params.get_examples.get("product_review", {}).get("source", "")
            if not text:
                print(
                    "Error: default example 'product_review' text not found, please check configuration."
                )
                return

        # Check if a schema has been passed, otherwise use the default schema from configuration
        schema_to_use = schema_name or self._params._schema_name
        parser = self.output_parser.get_parser(schema_to_use)

        if not parser:
            raise ValueError(f"Schema '{schema_to_use}' not found in configuration")

        format_instructions = self.output_parser.get_format_instructions(parser)

        # Check if template is passed, otherwise use the default template from configuration
        template_str = (
            self._params._schema_template
            or """For the following text, extract the following information:
            {format_instructions}
            
            text: {text}
            """
        )
        template = self.prompt_manager.create_template(template_str)

        # Format prompt with text to extract from and the instructions from the schema and template
        messages = self.prompt_manager.formatter(
            template, text=text, format_instructions=format_instructions
        )

        # Generate response using the LLM
        response = self.generate(messages)
        return self.output_parser.parse_output(parser, response)
