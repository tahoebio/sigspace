"""Base agent implementation for Tahoe Agent."""

import json
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from tahoe_agent.agent._prompts import SYSTEM_PROMPT
from tahoe_agent.llm import get_llm, SourceType
from tahoe_agent.model.retriever import Retriever

# from tahoe_agent.tool.base_tool import python_executor, web_search
from tahoe_agent.tool.vision_scores import analyze_vision_scores
from tahoe_agent.tool.drug_ranking import rank_drugs_by_moa
from tahoe_agent.utils import pretty_print
from functools import partial


class AgentState(TypedDict):
    """State of the agent."""

    messages: List[BaseMessage]
    summary: Optional[str]  # Store the summary from retrieval step
    drug_rankings: Optional[str]  # Store the final drug ranking results


class BaseAgent:
    """Base agent implementation with native LangChain tools."""

    def __init__(
        self,
        llm: Union[str, BaseChatModel] = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        use_retriever: bool = False,
        timeout_seconds: int = 300,
        source: Optional[SourceType] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        tool_config: Optional[Dict[str, Any]] = None,
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
            tool_config: Hidden configuration passed to tools but not visible in prompts
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
        self.tool_config = tool_config or {}  # Store hidden tool configuration

        # Store source information to handle provider-specific configurations
        self.source = source
        self.base_url = base_url

        # Initialize retriever if needed
        if use_retriever:
            self.retriever = Retriever()  # type: ignore

        # Initialize with native LangChain tools
        self._setup_default_tools()

        # Initialize workflow
        self.app: Optional[Any] = None
        self.checkpointer = MemorySaver()
        self.system_prompt = ""

        # Configuration state
        self.configured = False

        # Logging
        self.log: List[tuple[str, str]] = []

        # Configure the agent
        self.configure()

    def _setup_default_tools(self) -> None:
        """Set up default tools for the agent."""
        # Add native LangChain tools
        self.native_tools = [
            analyze_vision_scores,
            rank_drugs_by_moa,
        ]

    def configure(self) -> None:
        """Configure the agent workflow as a simple static DAG: prompt -> tool_calling -> retrieval."""
        # Generate system prompt for native tools
        tool_descriptions = []
        for tool in self.native_tools:
            tool_descriptions.append(f"**{tool.name}**: {tool.description}")

        tools_section = (
            "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        )

        self.system_prompt = SYSTEM_PROMPT.format(tools_section=tools_section)

        # Create the workflow as a simple static DAG
        workflow = StateGraph(AgentState)

        def prompt_node(state: AgentState) -> AgentState:
            """Process the user prompt and determine tool needs."""
            print("[DEBUG] prompt_node: Processing user input")
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

            # Bind native tools with auto tool choice, auto doesn't work for lambda
            llm_with_tools = self.llm.bind_tools(self.native_tools, tool_choice="any")

            print(f"[DEBUG] prompt_node: Invoking LLM with {len(messages)} messages")
            response = llm_with_tools.invoke(messages)
            print(f"[DEBUG] prompt_node: Response content: {response.content[:500]}...")

            # Check if tool calls were made
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(
                    f"[DEBUG] prompt_node: Found {len(response.tool_calls)} tool calls"
                )
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"[DEBUG] prompt_node: Tool call {i}: {tool_call}")
            else:
                print("[DEBUG] prompt_node: No tool calls found")

            state["messages"].append(response)
            return state

        def tool_calling_node(state: AgentState) -> AgentState:
            """Execute vision scores tool (or other tools in the future)."""
            print("[DEBUG] tool_calling_node: Checking for tool calls")
            last_message = state["messages"][-1]

            # Check if there are tool calls to execute
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(
                    f"[DEBUG] tool_calling_node: Found {len(last_message.tool_calls)} tool calls"
                )

                # Execute each tool call
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    print(
                        f"[DEBUG] tool_calling_node: Executing {tool_name} with args: {tool_args}"
                    )

                    # Find the tool by name
                    tool_func = None
                    for tool in self.native_tools:
                        if tool.name == tool_name:
                            tool_func = tool
                            break

                    if tool_func:
                        try:
                            # Special handling for vision scores (blind evaluation)
                            if tool_name == "analyze_vision_scores":
                                if "drug_name" in self.tool_config:
                                    result = self._inject_drug_name_in_vision_score(
                                        tool_func, tool_args
                                    )
                                    print(
                                        f"[DEBUG] Injecting drug_name: {self.tool_config['drug_name']} in tool_args"
                                    )
                                else:
                                    print(
                                        "[DEBUG] tool_calling_node: No drug_name configured for blind evaluation"
                                    )
                                    result = tool_func.invoke(tool_args)
                            else:
                                # Execute other tools normally
                                print(
                                    f"[DEBUG] tool_calling_node: Executing {tool_name}."
                                )
                                result = tool_func.invoke(tool_args)

                            print(
                                f"[DEBUG] tool_calling_node: Tool {tool_name} result: {result[:200]}..."
                            )

                            # Create a tool message with the result
                            from langchain_core.messages import ToolMessage

                            tool_message = ToolMessage(
                                content=result, name=tool_name, tool_call_id=tool_id
                            )
                            state["messages"].append(tool_message)

                        except Exception as e:
                            print(
                                f"[DEBUG] tool_calling_node: Error executing {tool_name}: {e}"
                            )
                            # Create error message
                            from langchain_core.messages import ToolMessage

                            error_message = ToolMessage(
                                content=f"Error executing {tool_name}: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                            state["messages"].append(error_message)
                    else:
                        print(f"[DEBUG] tool_calling_node: Tool {tool_name} not found")
            else:
                print("[DEBUG] tool_calling_node: No tool calls to execute")

            return state

        def retrieval_node(state: AgentState) -> AgentState:
            """Generate final response based on tool results."""
            print("[DEBUG] retrieval_node: Generating final response")
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

            # Add a prompt to synthesize the final response if tools were used
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                synthesis_prompt = """
Based on the tool execution results above, provide a comprehensive and helpful response to the user's original question.
Synthesize the information from the tools and present it in a clear, organized way.
"""
                messages.append(HumanMessage(content=synthesis_prompt))

            # Use LLM without tools for final response
            print(
                f"[DEBUG] retrieval_node: Generating synthesis with {len(messages)} messages"
            )
            final_response = self.llm.invoke(messages)
            print(
                f"[DEBUG] retrieval_node: Final response length: {len(final_response.content) if hasattr(final_response, 'content') else 'unknown'}"
            )

            state["messages"].append(final_response)

            # Save the summary for drug ranking
            if hasattr(final_response, "content") and final_response.content:
                summary_content = str(final_response.content)
                state["summary"] = summary_content
                print(
                    f"[DEBUG] retrieval_node: Saved summary: {summary_content[:200]}..."
                )

            return state

        def should_do_drug_ranking(state: AgentState) -> str:
            """Decide whether to proceed with drug ranking based on conversation state."""
            print("[DEBUG] should_do_drug_ranking: Checking conversation state")

            # Check user's original request for ranking-related keywords
            original_message = state["messages"][0] if state["messages"] else None
            if original_message and hasattr(original_message, "content"):
                content = str(original_message.content).lower()
                ranking_keywords = [
                    "rank",
                    "ranking",
                    "compare",
                    "comparison",
                    "mechanism",
                    "moa",
                ]

                if any(keyword in content for keyword in ranking_keywords):
                    print(
                        "[DEBUG] should_do_drug_ranking: Ranking keywords found in user request"
                    )
                    return "drug_ranking"

            print("[DEBUG] should_do_drug_ranking: No drug ranking needed")
            return "end"

        def drug_ranking_node(state: AgentState) -> AgentState:
            """Rank drugs based on the generated summary."""
            print("[DEBUG] drug_ranking_node: Starting drug ranking")

            # Check if we have a summary
            summary = state.get("summary")
            if not summary:
                print("[DEBUG] drug_ranking_node: No summary available, skipping")
                return state

            print("[DEBUG] drug_ranking_node: Ranking drugs using existing summary")

            # Create a simple prompt for the LLM to use ONLY the ranking tool
            ranking_prompt = f"""Use the rank_drugs_by_moa tool to rank drugs based on their mechanism of action similarity to the analysis summary.

Use the EXACT summary that was already generated (do NOT re-analyze vision scores):

Summary: {summary}"""

            # TODO: maybe remove the system prompt here?
            # messages = (
            #     [SystemMessage(content=self.system_prompt)]
            #     + state["messages"]
            #     + [HumanMessage(content=ranking_prompt)]
            # )
            messages = [HumanMessage(content=ranking_prompt)]

            # IMPORTANT: Only bind the ranking tool, not all tools
            llm_with_ranking_tool = self.llm.bind_tools(
                [rank_drugs_by_moa], tool_choice="required"
            )

            print("[DEBUG] drug_ranking_node: Calling LLM with ONLY ranking tool")
            response = llm_with_ranking_tool.invoke(messages)
            state["messages"].append(response)

            # Execute any tool calls (should only be ranking tool)
            if hasattr(response, "tool_calls") and response.tool_calls:
                print(
                    f"[DEBUG] drug_ranking_node: Executing {len(response.tool_calls)} tool calls"
                )

                # Only execute the first ranking tool call to avoid duplicates
                ranking_executed = False
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    print(f"[DEBUG] drug_ranking_node: Executing tool: {tool_name}")

                    # Only allow ranking tool execution and only execute it once
                    if tool_name == "rank_drugs_by_moa" and not ranking_executed:
                        try:
                            result = rank_drugs_by_moa.invoke(tool_args)
                            print(
                                "[DEBUG] drug_ranking_node: rank_drugs_by_moa executed successfully"
                            )

                            from langchain_core.messages import ToolMessage

                            ranking_tool_message = ToolMessage(
                                content=result, name=tool_name, tool_call_id=tool_id
                            )
                            state["messages"].append(ranking_tool_message)
                            ranking_executed = True
                        except Exception as e:
                            print(
                                f"[DEBUG] drug_ranking_node: Error executing {tool_name}: {e}"
                            )
                    elif tool_name == "rank_drugs_by_moa" and ranking_executed:
                        print(
                            "[DEBUG] drug_ranking_node: Skipping duplicate ranking tool call"
                        )
                    else:
                        print(
                            f"[DEBUG] drug_ranking_node: Unexpected tool call: {tool_name} - ignoring"
                        )

                # Get final response after tool execution
                final_messages = (
                    [SystemMessage(content=self.system_prompt)]
                    + [ranking_tool_message]
                    + [
                        HumanMessage(
                            content="Based on the drug ranking tool results above, provide the final ranked list of drugs with their relevance scores and reasoning."
                        )
                    ]
                )
                final_response = self.llm.invoke(final_messages)
                print(
                    f"[DEBUG] drug_ranking_node: Final response: {final_response.content}"
                )
                state["messages"].append(final_response)

                if hasattr(final_response, "content") and final_response.content:
                    state["drug_rankings"] = str(final_response.content)
                    print("[DEBUG] drug_ranking_node: Stored final drug rankings")
            else:
                print(
                    "[DEBUG] drug_ranking_node: No tool calls made - this should not happen"
                )

            return state

        # Clean workflow with proper conditional branching
        workflow.add_node("prompt", prompt_node)
        workflow.add_node("tool_calling", tool_calling_node)
        workflow.add_node("retrieval", retrieval_node)
        workflow.add_node("drug_ranking", drug_ranking_node)

        # Simple edges with clean conditional logic
        workflow.add_edge(START, "prompt")
        workflow.add_edge("prompt", "tool_calling")
        workflow.add_edge("tool_calling", "retrieval")

        # Conditional edge based on conversation state
        workflow.add_conditional_edges(
            "retrieval",
            should_do_drug_ranking,
            {"drug_ranking": "drug_ranking", "end": END},
        )

        workflow.add_edge("drug_ranking", END)

        # Compile workflow
        self.app = workflow.compile()
        self.app.checkpointer = self.checkpointer

        self.configured = True
        # mermaid_code = self.app.get_graph().draw_mermaid()
        # print(mermaid_code)

    def _inject_drug_name_in_vision_score(
        self, tool_func: Any, tool_args: Dict[str, Any]
    ) -> Any:  # noqa: ANN401
        # Create a partial version of the tool function with drug_name pre-bound
        underlying_func = tool_func.func  # noqa: E1101
        partial_func = partial(
            underlying_func,
            drug_name=self.tool_config["drug_name"],
        )

        print(
            f"[DEBUG] _execute_vision_scores_with_injection: Using partial with drug_name: {self.tool_config['drug_name']}"
        )
        return partial_func(**tool_args)

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

        # Initialize state
        inputs = {"messages": [HumanMessage(content=prompt)]}
        config = {"recursion_limit": 10, "configurable": {"thread_id": 42}}

        self.log = []
        final_response = ""

        # Stream the workflow
        try:
            if self.app is None:
                raise RuntimeError("Agent not configured. Call configure() first.")

            final_state = None
            for step in self.app.stream(inputs, stream_mode="values", config=config):
                final_state = step  # Keep track of the final state
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
                            final_response = str(message.content)

            # If we have drug rankings, include them in the final response
            if (
                final_state
                and "drug_rankings" in final_state
                and final_state["drug_rankings"]
            ):
                final_response = final_state["drug_rankings"]

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

    def get_drug_rankings(self) -> Optional[str]:
        """Get the latest drug rankings from the most recent run.

        Returns:
            JSON string with drug rankings or None if not available
        """
        # This would need to be stored from the latest run
        # For now, return None - in practice, you'd want to store this in the instance
        return None

    def get_summary(self) -> Optional[str]:
        """Get the latest summary from the most recent run.

        Returns:
            Summary string or None if not available
        """
        # This would need to be stored from the latest run
        # For now, return None - in practice, you'd want to store this in the instance
        return None

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
