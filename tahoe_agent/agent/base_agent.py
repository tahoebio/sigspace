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
        self.tool_registry = ToolRegistry()  # type: ignore
        if use_retriever:
            self.retriever = Retriever()  # type: ignore

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
        self.configure()  # type: ignore

    def _setup_default_tools(self) -> None:
        """Set up default tools for the agent."""
        # Add default tools
        python_tool = PythonTool()  # type: ignore
        web_search_tool = WebSearchTool()  # type: ignore

        self.tool_registry.register_tool(python_tool)
        self.tool_registry.register_tool(web_search_tool)

    def add_tool(
        self, tool: Union[BaseTool, Dict[str, Any], Callable[..., Any]]
    ) -> int:
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
                def __init__(self, func: Callable[..., Any]) -> None:
                    super().__init__(
                        name=func.__name__,
                        description=func.__doc__ or f"Tool {func.__name__}",
                        parameters=parameters,
                        required_parameters=parameters["required"],
                    )
                    self.func = func

                def execute(self, **kwargs: Any) -> Any:
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

        system_prompt = f"""You are a helpful AI assistant with access to tools. You MUST use the available tools to complete tasks - do not just describe what you would do, actually call the tools.

AVAILABLE TOOLS:
{tools_section}

CRITICAL INSTRUCTIONS:
1. When a user asks you to do something that requires a tool, YOU MUST CALL THE TOOL IMMEDIATELY
2. DO NOT just describe what you plan to do - actually execute the tool function call
3. Use the proper function calling format to invoke tools
4. After getting tool results, provide a helpful interpretation of the results

TOOL USAGE EXAMPLES:
- If asked to analyze vision scores: IMMEDIATELY call analyze_vision_scores()
- If asked to calculate something: IMMEDIATELY call python_executor()
- If asked to search: IMMEDIATELY call web_search()

Do not say "I will now call the tool" or "Let me use the tool" - just call it directly.

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

    def _parse_manual_tool_calls(self, content: str) -> list:
        """Parse function calls manually from text content for Lambda Labs."""
        import re
        import ast
        import uuid
        from langchain_core.messages import ToolMessage

        tool_messages = []

        # Pattern to match function calls like: analyze_vision_scores(arg1=value1, arg2=value2)
        pattern = r"(\w+)\((.*?)\)"
        matches = re.findall(pattern, content)

        for func_name, args_str in matches:
            print(f"[DEBUG] Found potential function call: {func_name}({args_str})")

            # Check if this is one of our available tools
            tool = self.tool_registry.get_tool_by_name(func_name)
            if not tool:
                print(f"[DEBUG] Function {func_name} is not a registered tool")
                continue

            try:
                # Parse the arguments
                # Handle both keyword and positional arguments
                if args_str.strip():
                    # Try to parse as keyword arguments first
                    try:
                        # Create a temporary function call string and parse it
                        temp_call = f"dummy({args_str})"
                        # Parse the AST to extract arguments
                        parsed = ast.parse(temp_call, mode="eval")
                        call_node = parsed.body

                        args = {}
                        for keyword in call_node.keywords:
                            if isinstance(keyword.value, ast.Constant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(
                                keyword.value, ast.Str
                            ):  # For older Python versions
                                args[keyword.arg] = keyword.value.s
                            elif isinstance(keyword.value, ast.NameConstant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, ast.Name):
                                # Handle boolean values like True/False
                                if keyword.value.id in ["True", "False", "None"]:
                                    args[keyword.arg] = eval(keyword.value.id)
                                else:
                                    args[keyword.arg] = keyword.value.id

                        print(f"[DEBUG] Parsed arguments: {args}")

                    except Exception as e:
                        print(f"[DEBUG] Failed to parse arguments: {e}")
                        continue
                else:
                    args = {}

                # Execute the tool
                print(
                    f"[DEBUG] Executing manually parsed tool: {func_name} with args: {args}"
                )
                if isinstance(tool, BaseTool):
                    result = tool.execute(**args)
                    print(f"[DEBUG] Manual tool result type: {type(result)}")
                else:
                    result = f"Tool {func_name} executed with args {args}"

                # Create a tool message
                tool_message = ToolMessage(
                    content=json.dumps(result)
                    if not isinstance(result, str)
                    else result,
                    name=func_name,
                    tool_call_id=str(uuid.uuid4()),
                )
                tool_messages.append(tool_message)
                print(
                    f"[DEBUG] Created manual tool message with content length: {len(tool_message.content)}"
                )

            except Exception as e:
                print(f"[DEBUG] Error executing manually parsed tool {func_name}: {e}")
                tool_message = ToolMessage(
                    content=f"Error executing tool {func_name}: {str(e)}",
                    name=func_name,
                    tool_call_id=str(uuid.uuid4()),
                )
                tool_messages.append(tool_message)

        return tool_messages

    def configure(self) -> None:
        """Configure the agent workflow."""
        # Generate system prompt
        self.system_prompt = self._generate_system_prompt()

        # Convert tools to LangChain format for tool calling
        langchain_tools = []
        print(f"[DEBUG] Converting {len(self.get_tools())} tools to LangChain format")
        for i, tool in enumerate(self.get_tools()):
            print(f"[DEBUG] Tool {i}: {tool} (type: {type(tool)})")
            if isinstance(tool, BaseTool):
                print(f"[DEBUG] Converting BaseTool: {tool.name}")
                # Convert BaseTool to LangChain tool
                from langchain_core.tools import tool as lc_tool

                # Fix closure issue by capturing tool in a default parameter
                def create_tool_func(captured_tool: BaseTool) -> Callable[..., Any]:
                    @lc_tool(captured_tool.name, return_direct=False)
                    def tool_func(**kwargs) -> Any:
                        """Tool function wrapper."""
                        print(
                            f"[DEBUG] LangChain tool '{captured_tool.name}' called with kwargs: {kwargs}"
                        )
                        result = captured_tool.execute(**kwargs)
                        print(
                            f"[DEBUG] LangChain tool '{captured_tool.name}' returned: {type(result)}"
                        )
                        return result

                    tool_func.name = captured_tool.name
                    tool_func.description = captured_tool.description
                    return tool_func

                lc_tool_instance = create_tool_func(tool)
                langchain_tools.append(lc_tool_instance)
                print(f"[DEBUG] Created LangChain tool: {lc_tool_instance.name}")
            else:
                print(f"[DEBUG] Skipping non-BaseTool: {tool}")

        print(f"[DEBUG] Final LangChain tools count: {len(langchain_tools)}")
        for tool in langchain_tools:
            print(f"[DEBUG] LangChain tool: {tool.name} - {tool.description}")

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Define nodes
        def generate_response(state: AgentState) -> AgentState:
            """Generate response from LLM."""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

            # Bind tools to LLM if available
            if langchain_tools:
                print(f"[DEBUG] Binding {len(langchain_tools)} tools to LLM")
                # Handle Lambda Labs API limitation with tool choice
                if self._is_lambda_model():
                    # Lambda Labs doesn't support "auto" tool choice, use "none"
                    llm_with_tools = self.llm.bind_tools(
                        langchain_tools, tool_choice="none"
                    )
                    print("[DEBUG] Using Lambda Labs with 'none' tool choice")
                else:
                    # Use default "auto" tool choice for other providers
                    llm_with_tools = self.llm.bind_tools(langchain_tools)
                    print("[DEBUG] Using standard provider with 'auto' tool choice")
            else:
                llm_with_tools = self.llm
                print("[DEBUG] No tools available, using LLM without tools")

            print(f"[DEBUG] Invoking LLM with {len(messages)} messages")
            response = llm_with_tools.invoke(messages)
            print(f"[DEBUG] LLM response type: {type(response)}")
            print(f"[DEBUG] Response content: {response.content[:200]}...")
            print(
                f"[DEBUG] Response has tool_calls attribute: {hasattr(response, 'tool_calls')}"
            )

            if hasattr(response, "tool_calls"):
                print(f"[DEBUG] tool_calls value: {response.tool_calls}")
                print(f"[DEBUG] tool_calls type: {type(response.tool_calls)}")
                print(
                    f"[DEBUG] tool_calls length: {len(response.tool_calls) if response.tool_calls else 'None'}"
                )

            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"[DEBUG] Found {len(response.tool_calls)} tool calls")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"[DEBUG] Tool call {i}: {tool_call}")
                    if isinstance(tool_call, dict):
                        print(
                            f"[DEBUG] Tool call {i}: {tool_call.get('name', 'unknown')} with args {tool_call.get('args', {})}"
                        )
                    else:
                        print(f"[DEBUG] Tool call {i} type: {type(tool_call)}")
                state["next_step"] = "execute_tools"
            else:
                print("[DEBUG] No tool calls found")
                print(
                    f"[DEBUG] Available tool names: {[tool.name for tool in langchain_tools]}"
                )

                # For Lambda Labs, check if there are function calls in the content before routing to execute_tools
                if self._is_lambda_model() and hasattr(response, "content"):
                    # Quick check if there are function calls in the content
                    import re

                    pattern = r"(\w+)\((.*?)\)"
                    matches = re.findall(pattern, response.content)

                    # Filter to only our actual tool names
                    tool_names = {tool.name for tool in langchain_tools}
                    valid_matches = [
                        match for match in matches if match[0] in tool_names
                    ]

                    if valid_matches:
                        print(
                            f"[DEBUG] Lambda Labs model - found {len(valid_matches)} potential function calls in content"
                        )
                        state["next_step"] = "execute_tools"
                    else:
                        print(
                            "[DEBUG] Lambda Labs model - no function calls found in content, ending workflow"
                        )
                        state["next_step"] = "end"
                else:
                    print("[DEBUG] Non-Lambda model - ending workflow")
                    state["next_step"] = "end"

            state["messages"].append(response)
            return state

        def execute_tools(state: AgentState) -> AgentState:
            """Execute tool calls."""
            print("[DEBUG] execute_tools called")
            last_message = state["messages"][-1]
            tool_messages = []

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"[DEBUG] Processing {len(last_message.tool_calls)} tool calls")
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    print(f"[DEBUG] Executing tool: {tool_name} with args: {tool_args}")

                    # Find and execute the tool
                    tool = self.tool_registry.get_tool_by_name(tool_name)
                    if tool:
                        print(f"[DEBUG] Found tool: {tool}")
                        try:
                            if isinstance(tool, BaseTool):
                                print(f"[DEBUG] Executing BaseTool: {tool.name}")
                                result = tool.execute(**tool_args)
                                print(f"[DEBUG] Tool result type: {type(result)}")
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
                            print(
                                f"[DEBUG] Created tool message with content length: {len(tool_message.content)}"
                            )
                        except Exception as e:
                            print(f"[DEBUG] Tool execution error: {e}")
                            tool_message = ToolMessage(
                                content=f"Error executing tool {tool_name}: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                    else:
                        print(f"[DEBUG] Tool {tool_name} not found in registry")
                        tool_message = ToolMessage(
                            content=f"Tool {tool_name} not found",
                            name=tool_name,
                            tool_call_id=tool_id,
                        )

                    tool_messages.append(tool_message)
            else:
                print("[DEBUG] No tool calls found in last message")

                # Fallback: manually parse function calls from text for Lambda Labs
                if self._is_lambda_model() and hasattr(last_message, "content"):
                    print(
                        "[DEBUG] Attempting manual function call parsing for Lambda Labs"
                    )
                    tool_messages.extend(
                        self._parse_manual_tool_calls(last_message.content)
                    )

            print(f"[DEBUG] Adding {len(tool_messages)} tool messages to state")
            state["messages"].extend(tool_messages)
            state["next_step"] = "generate"
            return state

        def route_next(state: AgentState) -> str:
            """Route to next step."""
            next_step = state.get("next_step", "end")
            print(f"[DEBUG] route_next: next_step = {next_step}")
            if next_step == "execute_tools":
                print("[DEBUG] Routing to execute_tools")
                return "execute_tools"
            elif next_step == "generate":
                print("[DEBUG] Routing to generate")
                return "generate"
            else:
                print("[DEBUG] Routing to end")
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

    def run(self, prompt: str, **kwargs: Any) -> tuple[List[tuple[str, str]], str]:
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
        config = {"recursion_limit": 10, "configurable": {"thread_id": 42}}

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

    def reset(self) -> None:
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

    def save_conversation(self, filename: str) -> None:
        """Save conversation to file.

        Args:
            filename: Path to save file
        """
        with open(filename, "w") as f:
            json.dump(self.log, f, indent=2)

    def load_conversation(self, filename: str) -> None:
        """Load conversation from file.

        Args:
            filename: Path to load file
        """
        with open(filename, "r") as f:
            self.log = json.load(f)
