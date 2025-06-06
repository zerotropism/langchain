from datetime import date
from typing import List, Optional, Any, Dict, Union

from langchain.agents import load_tools, initialize_agent, AgentType, tool
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.schema import AgentAction, AgentFinish
from config import ConfigManager
from llm import LLMClient
from decorators import handle_exception, timing_decorator


class Tools:
    """Collection of custom tools that can be used with agents."""

    @staticmethod
    @tool
    def time(text: str) -> str:
        """Returns today's date, use this for any questions related to knowing today's date.
        The input should always be an empty string, and this function will always return today's
        date - any date mathematics should occur outside this function.
        """
        return str(date.today())


class AgentRunner:
    """Class to manage running agents with better error handling and logging."""

    def __init__(self, agent: Any, debug: bool = False):
        """Initialize with a configured agent.

        Args:
            agent: A LangChain agent instance
            debug: Whether to enable debug mode for LangChain
        """
        self.agent = agent
        self.debug = debug

    @handle_exception
    @timing_decorator
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent on a query with proper error handling.

        Args:
            query: The user query to process

        Returns:
            The agent's response
        """
        import langchain

        prev_debug = langchain.debug
        if self.debug:
            langchain.debug = True

        try:
            result = self.agent(query)
            return result
        except Exception as e:
            print(f"Error during agent execution: {str(e)}")
            return {"output": f"Exception occurred: {str(e)}"}
        finally:
            langchain.debug = prev_debug


class AgentFactory:
    """Factory class for creating different types of LangChain agents."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the factory with the language model configuration.

        Args:
            config (`ConfigManager`, optional): Pre-loaded settings from `./config.yml` file
        """
        self.config = config or ConfigManager()
        self.model_settings = self.config.get("model", {})
        self.model_name = self.model_settings.get("name", "gemma3:12b")
        self.temperature = self.model_settings.get("temperature", 0.0)
        self.llm_client = LLMClient(self.config)
        self.llm = self.llm_client.infer()

    @handle_exception
    def create_qa_agent(self, verbose: bool = True) -> Any:
        """Create an agent for general QA with math and Wikipedia capabilities.

        Args:
            verbose: Whether to print detailed agent reasoning

        Returns:
            A configured LangChain agent
        """
        tools = load_tools(["llm-math", "wikipedia"], llm=self.llm)
        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=verbose,
        )

    @handle_exception
    def create_python_agent(self, verbose: bool = True) -> Any:
        """Create an agent that can execute Python code.

        Args:
            verbose: Whether to print detailed agent reasoning

        Returns:
            A configured Python-capable LangChain agent
        """
        return create_python_agent(self.llm, tool=PythonREPLTool(), verbose=verbose)

    @handle_exception
    def create_custom_agent(
        self, additional_tools: List = None, verbose: bool = True
    ) -> Any:
        """Create an agent with custom tools in addition to standard ones.

        Args:
            additional_tools: List of additional tools to add to the agent
            verbose: Whether to print detailed agent reasoning

        Returns:
            A configured LangChain agent with custom tools
        """
        if additional_tools is None:
            additional_tools = []

        base_tools = load_tools(["llm-math", "wikipedia"], llm=self.llm)
        all_tools = base_tools + additional_tools

        return initialize_agent(
            all_tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=verbose,
        )

    @handle_exception
    def create_time_agent(self, verbose: bool = True) -> Any:
        """Create an agent with time functionality.

        Args:
            verbose: Whether to print detailed agent reasoning

        Returns:
            A configured LangChain agent with time tool
        """
        return self.create_custom_agent([Tools.time], verbose=verbose)

    @handle_exception
    @timing_decorator
    def run_agent_query(
        self, agent_type: str, query: str, debug: bool = False
    ) -> Dict[str, Any]:
        """Run a query using a specified agent type.

        Args:
            agent_type: Type of agent to use ('qa', 'python', 'custom', 'time')
            query: The query to process
            debug: Whether to enable debug output

        Returns:
            The agent's response
        """
        # Create the appropriate agent
        if agent_type == "qa":
            agent = self.create_qa_agent()
        elif agent_type == "python":
            agent = self.create_python_agent()
        elif agent_type == "time":
            agent = self.create_time_agent()
        elif agent_type == "custom":
            agent = self.create_custom_agent([Tools.time])
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Run the agent
        runner = AgentRunner(agent, debug=debug)
        return runner.run(query)
