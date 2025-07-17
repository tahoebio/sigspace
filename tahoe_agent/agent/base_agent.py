"""Base agent implementation for Tahoe Agent."""

import json
import copy
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from tahoe_agent.agent._prompts import (
    SYSTEM_PROMPT,
    DRUG_RANKING_PROMPT,
    SUMMARY_PROMPT,
)
from tahoe_agent.llm import get_llm, SourceType
from tahoe_agent.model.retriever import Retriever

# from tahoe_agent.tool.base_tool import python_executor, web_search
from tahoe_agent.tool.vision_scores import analyze_vision_scores
from tahoe_agent.tool.drug_ranking import get_drug_list
from tahoe_agent.utils import pretty_print
from functools import partial
from pydantic import BaseModel, Field
from typing import List as TypingList
from tahoe_agent.logging_config import get_logger


class DrugRanking(BaseModel):
    """Individual drug ranking with score."""

    drug: str = Field(description="Name of the drug")
    score: float = Field(description="Relevance score for the drug")


class DrugRankings(BaseModel):
    """Structured output for drug rankings."""

    rankings: TypingList[DrugRanking] = Field(
        description="List of drug rankings ordered by relevance score"
    )


class AgentState(TypedDict):
    """State of the agent."""

    messages: List[BaseMessage]
    signature_summary: Optional[str]  # Store the summary from retrieval step
    drug_rankings: Optional[str]  # Store the final drug ranking results
    structured_rankings: Optional[
        TypingList[DrugRanking]
    ]  # Store structured drug rankings


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

        # Create a separate, clean LLM for summarization by copying the base LLM
        self.summarizer_llm = copy.copy(self.llm)

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
        self.logger = get_logger()

        # Configure the agent
        self.configure()

    def _setup_default_tools(self) -> None:
        """Set up default tools for the agent."""
        # Add native LangChain tools
        self.native_tools = [analyze_vision_scores]

    def configure(self) -> None:
        """Configures the agent's workflow for vision analysis and drug ranking."""
        tool_descriptions = [
            f"**{t.name}**: {t.description}" for t in self.native_tools
        ]
        tools_section = "\n".join(tool_descriptions) or "No tools available"
        self.system_prompt = SYSTEM_PROMPT.format(tools_section=tools_section)

        workflow = StateGraph(AgentState)

        # Node 1: Plan the next step, which may involve calling a tool.
        def plan_step(state: AgentState) -> AgentState:
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            llm_with_tools = self.llm.bind_tools(
                [analyze_vision_scores], tool_choice="any"
            )
            response = llm_with_tools.invoke(messages)
            state["messages"].append(response)
            return state

        # Node 2: Execute the vision scores tool if requested by the planner.
        def execute_vision_tool(state: AgentState) -> AgentState:
            last_message = state["messages"][-1]
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return state

            tool_call = last_message.tool_calls[0]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            result = self._inject_drug_name_in_vision_score(
                analyze_vision_scores, tool_args
            )

            from langchain_core.messages import ToolMessage

            self.logger.info("[execute_vision_tool] Tool message: %s", result)
            tool_message = ToolMessage(
                content=result, name=analyze_vision_scores.name, tool_call_id=tool_id
            )
            state["messages"].append(tool_message)
            return state

        # Node 3: Summarize the results from the vision tool.
        def summarize_results(state: AgentState) -> AgentState:
            self.logger.info("[summarize_results] --- Summarizing Results ---")
            tool_output = state["messages"][-1].content

            messages = [
                HumanMessage(content=SUMMARY_PROMPT.format(tool_output=tool_output))
            ]

            # DO NOT REMOVE THIS COMMENTED CODE
            # print("\n[--- Message Contents ---]")
            # for i, msg in enumerate(messages):
            #     print(f"\nMessage {i}:")
            #     if hasattr(msg, "content"):
            #         print(f"Content: {msg.content}")
            #     else:
            #         print("No content attribute")

            response = self.summarizer_llm.invoke(messages)
            state["messages"].append(response)
            if hasattr(response, "content") and response.content:
                summary_text = str(response.content)
                state["signature_summary"] = summary_text
                self.logger.info(
                    f"[summarize_results] Summary generated: {summary_text}..."
                )
            else:
                self.logger.warning(
                    "[summarize_results] Warning: No summary was generated."
                )
            return state

        # Node 4: Rank drugs based on the summary and return structured output.
        def rank_drugs(state: AgentState) -> AgentState:
            self.logger.info("[rank_drugs] --- Ranking Drugs ---")
            summary = state.get("signature_summary")
            if not summary:
                self.logger.error(
                    "[rank_drugs] Error: No summary available for drug ranking."
                )
                return state

            # Since get_drug_list takes no arguments, we can invoke it directly.
            self.logger.info("[rank_drugs] Invoking get_drug_list tool directly...")
            drug_list_str = get_drug_list.invoke({})
            self.logger.info(
                f"[rank_drugs] Result from get_drug_list tool: {str(drug_list_str)[:300]}..."
            )

            # Generate final ranked list with structured output
            final_messages = [
                HumanMessage(
                    content=DRUG_RANKING_PROMPT.format(
                        drug_list=drug_list_str, summary=summary
                    )
                ),
            ]

            self.logger.info("[rank_drugs] Messages content for final ranking LLM:")
            for msg in final_messages:
                self.logger.info(
                    f"[rank_drugs] [{msg.__class__.__name__}]: {msg.content}..."
                )

            llm_with_structured_output = self.llm.with_structured_output(DrugRankings)
            structured_response = llm_with_structured_output.invoke(final_messages)

            self.logger.info(
                f"[rank_drugs] Structured response from ranking LLM: {structured_response}"
            )

            if isinstance(structured_response, DrugRankings):
                top_50_rankings = structured_response.rankings[:50]
                rankings_text = (
                    f"Top {len(top_50_rankings)} Drug Rankings:\n"
                    + "\n".join(
                        [
                            f"{i+1}. {r.drug}: {r.score:.3f}"
                            for i, r in enumerate(top_50_rankings)
                        ]
                    )
                )
                state["drug_rankings"] = rankings_text
                state["structured_rankings"] = top_50_rankings
            else:
                rankings_text = (
                    f"Error: Unexpected response format: {structured_response}"
                )
                state["drug_rankings"] = rankings_text

            state["messages"].append(AIMessage(content=rankings_text))
            return state

        # Conditional router to decide whether to perform drug ranking.
        def should_rank_drugs(state: AgentState) -> str:
            original_message = state["messages"][0]
            content = str(original_message.content).lower()
            ranking_keywords = ["rank", "ranking", "compare", "mechanism", "moa"]
            if any(keyword in content for keyword in ranking_keywords):
                return "rank_drugs"
            return "end"

        # Define the workflow graph
        workflow.add_node("plan_step", plan_step)
        workflow.add_node("execute_vision_tool", execute_vision_tool)
        workflow.add_node("summarize_results", summarize_results)
        workflow.add_node("rank_drugs", rank_drugs)

        workflow.set_entry_point("plan_step")
        workflow.add_edge("plan_step", "execute_vision_tool")
        workflow.add_edge("execute_vision_tool", "summarize_results")

        workflow.add_conditional_edges(
            "summarize_results",
            should_rank_drugs,
            {"rank_drugs": "rank_drugs", "end": END},
        )
        workflow.add_edge("rank_drugs", END)

        self.app = workflow.compile(checkpointer=self.checkpointer)
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

    def run(
        self, prompt: str, **kwargs: Any
    ) -> tuple[
        List[tuple[str, str]],
        str,
        Optional[TypingList[DrugRanking]],
        Optional[str],
    ]:
        """Run the agent with a given prompt.

        Args:
            prompt: User prompt
            **kwargs: Additional arguments

        Returns:
            Tuple of (log, final_response, structured_rankings, summary)
        """
        if not self.configured:
            self.configure()

        # Initialize state
        inputs = {"messages": [HumanMessage(content=prompt)]}
        config = {"recursion_limit": 10, "configurable": {"thread_id": 42}}

        self.log = []
        final_response = ""
        structured_rankings = None
        summary = None

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
                    self.logger.info(f"[run] {role}: {content}")
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

            if (
                final_state
                and "structured_rankings" in final_state
                and final_state["structured_rankings"]
            ):
                structured_rankings = final_state["structured_rankings"]

            if (
                final_state
                and "signature_summary" in final_state
                and final_state["signature_summary"]
            ):
                summary = final_state["signature_summary"]

        except Exception as e:
            error_msg = f"Error during execution: {str(e)}"
            self.log.append(("error", error_msg))
            final_response = error_msg

        return self.log, final_response, structured_rankings, summary

    def chat(self, prompt: str) -> str:
        """Simple chat interface.

        Args:
            prompt: User prompt

        Returns:
            Agent response
        """
        _, response, _, _ = self.run(prompt)
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
