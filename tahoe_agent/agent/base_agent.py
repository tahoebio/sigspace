"""Base agent implementation for Tahoe Agent."""

import json
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from tahoe_agent.llm import get_llm, SourceType
from tahoe_agent.model.retriever import Retriever
from tahoe_agent.tool.base_tool import BaseTool, PythonTool, WebSearchTool
from tahoe_agent.tool.tool_registry import ToolRegistry
from tahoe_agent.utils import pretty_print


class AgentState(TypedDict):
    """State of the agent."""

    messages: List[BaseMessage]
    next_step: Optional[str]


class BaseAgent:
    """Base agent implementation inspired by Biomni."""

    def __init__(
        self,
        llm: Union[str, BaseChatModel] = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        use_retriever: bool = False,
        timeout_seconds: int = 300,
        source: Optional[SourceType] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
    ):
        """Initialize the base agent.

        Args:
            llm: LLM model name or instance
            temperature: Temperature for LLM
            use_retriever: Whether to use tool retrieval
            timeout_seconds: Timeout for tool execution
            source: LLM source provider (OpenAI, Anthropic, Lambda, etc.)
            base_url: Base URL for custom model serving
            api_key: API key for custom LLM
        """
        # Initialize LLM
        if isinstance(llm, str):
            self.llm = get_llm(
                model=llm,
                temperature=temperature,
                source=source,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self.llm = llm

        self.temperature = temperature
        self.use_retriever = use_retriever
        self.timeout_seconds = timeout_seconds

        # Store source information to handle provider-specific configurations
        self.source = source
        self.base_url = base_url

        # Initialize tool registry and retriever
        self.tool_registry = ToolRegistry()
        if use_retriever:
            self.retriever = Retriever()

        # Initialize with default tools
        self._setup_default_tools()

        # Initialize workflow
        self.app = None
        self.checkpointer = MemorySaver()
        self.system_prompt = ""

        # Configuration state
        self.configured = False

        # Logging
        self.log: List[tuple[str, str]] = []

        # Configure the agent
        self.configure()

    def _setup_default_tools(self):
        """Set up default tools for the agent."""
        # Add default tools
        python_tool = PythonTool()
        web_search_tool = WebSearchTool()

        self.tool_registry.register_tool(python_tool)
        self.tool_registry.register_tool(web_search_tool)

    def add_tool(self, tool: Union[BaseTool, Dict[str, Any], Callable]) -> int:
        """Add a tool to the agent.

        Args:
            tool: Tool to add (BaseTool instance, dict, or callable)

        Returns:
            Index of the registered tool
        """
        if callable(tool) and not isinstance(tool, BaseTool):
            # Convert callable to BaseTool
            import inspect

            sig = inspect.signature(tool)
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param_name, param in sig.parameters.items():
                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation is int:
                        param_type = "integer"
                    elif param.annotation is float:
                        param_type = "number"
                    elif param.annotation is bool:
                        param_type = "boolean"
                    elif param.annotation is list:
                        param_type = "array"
                    elif param.annotation is dict:
                        param_type = "object"

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}",
                }

                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            class CallableTool(BaseTool):
                def __init__(self, func):
                    super().__init__(
                        name=func.__name__,
                        description=func.__doc__ or f"Tool {func.__name__}",
                        parameters=parameters,
                        required_parameters=parameters["required"],
                    )
                    self.func = func

                def execute(self, **kwargs):
                    return self.func(**kwargs)

            tool = CallableTool(tool)

        index = self.tool_registry.register_tool(tool)

        # Reconfigure if already configured
        if self.configured:
            self.configure()

        return index

    def get_tools(self) -> List[Union[BaseTool, Dict[str, Any]]]:
        """Get all registered tools.

        Returns:
            List of registered tools
        """
        return self.tool_registry.tools

    def search_tools(self, query: str) -> List[Union[BaseTool, Dict[str, Any]]]:
        """Search for tools by query.

        Args:
            query: Search query

        Returns:
            List of matching tools
        """
        return self.tool_registry.search_tools(query)

    def _generate_system_prompt(
        self, tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None
    ) -> str:
        """Generate system prompt for the agent.

        Args:
            tools: List of tools to include in prompt (None for all tools)

        Returns:
            Generated system prompt
        """
        if tools is None:
            tools = self.get_tools()

        tool_descriptions = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                name = tool.name
                desc = tool.description
                params = tool.parameters
            elif isinstance(tool, dict):
                name = tool.get("name", "Unknown")
                desc = tool.get("description", "No description")
                params = tool.get("parameters", {})
            else:
                name = str(tool)
                desc = "Tool"
                params = {}

            tool_desc = f"**{name}**: {desc}"
            if params and isinstance(params, dict) and "properties" in params:
                props = params["properties"]
                required = params.get("required", [])

                param_list = []
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    req_str = " (required)" if is_required else " (optional)"
                    param_list.append(
                        f"  - {param_name} ({param_type}){req_str}: {param_desc}"
                    )

                if param_list:
                    tool_desc += "\n" + "\n".join(param_list)

            tool_descriptions.append(tool_desc)

        tools_section = (
            "\n\n".join(tool_descriptions)
            if tool_descriptions
            else "No tools available"
        )

        system_prompt = f"""You are a helpful AI assistant. You have access to various tools to help you complete tasks.

AVAILABLE TOOLS:
{tools_section}

INSTRUCTIONS:
1. Analyze the user's request carefully
2. Choose the appropriate tools to accomplish the task
3. Execute tools step by step as needed
4. Provide clear explanations of what you're doing
5. Present results in a helpful format

When using tools:
- Call tools using the proper function calling format
- Handle errors gracefully
- Explain your reasoning for tool choices
- Break down complex tasks into smaller steps

Always be helpful, accurate, and clear in your responses."""

        return system_prompt

    def _is_lambda_model(self) -> bool:
        """Check if the current model is using Lambda Labs API.

        Returns:
            True if using Lambda Labs, False otherwise
        """
        # Check if source was explicitly set to Lambda
        if self.source == "Lambda":
            return True

        # Check if base_url contains lambda (for auto-detection cases)
        if self.base_url and "lambda" in self.base_url.lower():
            return True

        return False

    def configure(self):
        """Configure the agent workflow."""
        # Generate system prompt
        self.system_prompt = self._generate_system_prompt()

        # Convert tools to LangChain format for tool calling
        langchain_tools = []
        for tool in self.get_tools():
            if isinstance(tool, BaseTool):
                # Convert BaseTool to LangChain tool
                from langchain_core.tools import tool as lc_tool

                @lc_tool(tool.name, return_direct=False)
                def tool_func(**kwargs):
                    """Tool function wrapper."""
                    return tool.execute(**kwargs)

                tool_func.name = tool.name
                tool_func.description = tool.description
                langchain_tools.append(tool_func)

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Define nodes
        def generate_response(state: AgentState) -> AgentState:
            """Generate response from LLM."""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

            # Bind tools to LLM if available
            if langchain_tools:
                # Handle Lambda Labs API limitation with auto tool choice
                if self._is_lambda_model():
                    # Lambda doesn't support "auto" tool choice without specific server flags
                    # Use "none" tool choice which allows the model to decide whether to use tools
                    llm_with_tools = self.llm.bind_tools(
                        langchain_tools, tool_choice="none"
                    )
                else:
                    # Use default "auto" tool choice for other providers
                    llm_with_tools = self.llm.bind_tools(langchain_tools)
            else:
                llm_with_tools = self.llm

            response = llm_with_tools.invoke(messages)

            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                state["next_step"] = "execute_tools"
            else:
                state["next_step"] = "end"

            state["messages"].append(response)
            return state

        def execute_tools(state: AgentState) -> AgentState:
            """Execute tool calls."""
            last_message = state["messages"][-1]
            tool_messages = []

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Find and execute the tool
                    tool = self.tool_registry.get_tool_by_name(tool_name)
                    if tool:
                        try:
                            if isinstance(tool, BaseTool):
                                result = tool.execute(**tool_args)
                            else:
                                # Handle dict-based tools
                                result = (
                                    f"Tool {tool_name} executed with args {tool_args}"
                                )

                            tool_message = ToolMessage(
                                content=json.dumps(result)
                                if not isinstance(result, str)
                                else result,
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                        except Exception as e:
                            tool_message = ToolMessage(
                                content=f"Error executing tool {tool_name}: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                    else:
                        tool_message = ToolMessage(
                            content=f"Tool {tool_name} not found",
                            name=tool_name,
                            tool_call_id=tool_id,
                        )

                    tool_messages.append(tool_message)

            state["messages"].extend(tool_messages)
            state["next_step"] = "generate"
            return state

        def route_next(state: AgentState) -> str:
            """Route to next step."""
            next_step = state.get("next_step", "end")
            if next_step == "execute_tools":
                return "execute_tools"
            elif next_step == "generate":
                return "generate"
            else:
                return "end"

        # Add nodes
        workflow.add_node("generate", generate_response)
        workflow.add_node("execute_tools", execute_tools)

        # Add edges
        workflow.add_conditional_edges(
            "generate",
            route_next,
            {
                "execute_tools": "execute_tools",
                "generate": "generate",
                "end": END,
            },
        )
        workflow.add_edge("execute_tools", "generate")
        workflow.add_edge(START, "generate")

        # Compile workflow
        self.app = workflow.compile()
        self.app.checkpointer = self.checkpointer

        self.configured = True

    def run(self, prompt: str, **kwargs) -> tuple[List[tuple[str, str]], str]:
        """Run the agent with a given prompt.

        Args:
            prompt: User prompt
            **kwargs: Additional arguments

        Returns:
            Tuple of (log, final_response)
        """
        if not self.configured:
            self.configure()

        # Use retrieval if enabled
        if self.use_retriever and hasattr(self, "retriever"):
            # Get all tools for retrieval
            resources = {"tools": self.get_tools()}

            # Use retrieval to select relevant tools
            selected_resources = self.retriever.prompt_based_retrieval(
                prompt, resources, self.llm
            )
            selected_tools = selected_resources.get("tools", [])

            # Update system prompt with selected tools
            self.system_prompt = self._generate_system_prompt(selected_tools)

        # Initialize state
        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 50, "configurable": {"thread_id": 42}}

        self.log = []
        final_response = ""

        # Stream the workflow
        try:
            for step in self.app.stream(inputs, stream_mode="values", config=config):
                if "messages" in step and step["messages"]:
                    message = step["messages"][-1]
                    role, content = pretty_print(message)
                    self.log.append((role, content))
                    if isinstance(message, AIMessage):
                        # Extract content from AI messages, regardless of tool calls
                        # If there are no tool calls or tool calls are empty, use this as final response
                        if not hasattr(message, "tool_calls") or not message.tool_calls:
                            final_response = content
                        elif message.content:  # Has tool calls but also has content
                            final_response = message.content
        except Exception as e:
            error_msg = f"Error during execution: {str(e)}"
            self.log.append(("error", error_msg))
            final_response = error_msg

        return self.log, final_response

    def chat(self, prompt: str) -> str:
        """Simple chat interface.

        Args:
            prompt: User prompt

        Returns:
            Agent response
        """
        _, response = self.run(prompt)
        return response

    def reset(self):
        """Reset the agent state."""
        self.log = []
        self.checkpointer = MemorySaver()
        if self.app:
            self.app.checkpointer = self.checkpointer

    def get_conversation_history(self) -> List[tuple[str, str]]:
        """Get the conversation history.

        Returns:
            List of (role, content) tuples
        """
        return self.log.copy()

    def save_conversation(self, filename: str):
        """Save conversation to file.

        Args:
            filename: Path to save file
        """
        with open(filename, "w") as f:
            json.dump(self.log, f, indent=2)

    def load_conversation(self, filename: str):
        """Load conversation from file.

        Args:
            filename: Path to load file
        """
        with open(filename, "r") as f:
            self.log = json.load(f)
