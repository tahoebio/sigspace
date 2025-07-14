"""Retriever for selecting relevant tools and resources."""

import contextlib
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage


class Retriever:
    """Retrieve tools and resources based on queries."""

    def __init__(self):
        pass

    def prompt_based_retrieval(
        self,
        query: str,
        resources: Dict[str, List[Any]],
        llm: Optional[BaseChatModel] = None,
    ) -> Dict[str, List[Any]]:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with resource types as keys and lists of resources as values
            llm: Optional LLM instance to use for retrieval

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources
        """
        if llm is None:
            from tahoe_agent.llm import get_llm

            llm = get_llm("gpt-4o-mini")

        # Create a prompt for the LLM to select relevant resources
        prompt = f"""
You are an expert assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

{self._format_resources_for_prompt(resources)}

For each category, respond with ONLY the indices of the relevant items in the following format:
{self._generate_response_format(resources)}

If a category has no relevant items, use an empty list, e.g., TOOLS: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. Include resources that could provide useful functionality
3. Don't exclude resources just because they're not explicitly mentioned in the query
4. When in doubt, include it rather than exclude it
"""

        # Invoke the LLM
        if hasattr(llm, "invoke"):
            # For LangChain-style LLMs
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        else:
            # For other LLM interfaces
            response_content = str(llm(prompt))

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(response_content, resources)

        # Get the selected resources
        selected_resources = {}
        for category, indices in selected_indices.items():
            if category in resources:
                selected_resources[category] = [
                    resources[category][i]
                    for i in indices
                    if i < len(resources[category])
                ]
            else:
                selected_resources[category] = []

        return selected_resources

    def _format_resources_for_prompt(self, resources: Dict[str, List[Any]]) -> str:
        """Format resources for inclusion in the prompt."""
        formatted_sections = []

        for category, resource_list in resources.items():
            formatted_items = []
            for i, resource in enumerate(resource_list):
                if isinstance(resource, dict):
                    # Handle dictionary format
                    name = resource.get("name", f"Resource {i}")
                    description = resource.get("description", "")
                    formatted_items.append(f"{i}. {name}: {description}")
                elif isinstance(resource, str):
                    # Handle string format
                    formatted_items.append(f"{i}. {resource}")
                else:
                    # Try to extract name and description from objects
                    name = getattr(resource, "name", str(resource))
                    desc = getattr(resource, "description", "")
                    formatted_items.append(f"{i}. {name}: {desc}")

            section_title = category.upper().replace("_", " ")
            section_content = (
                "\n".join(formatted_items) if formatted_items else "None available"
            )
            formatted_sections.append(f"AVAILABLE {section_title}:\n{section_content}")

        return "\n\n".join(formatted_sections)

    def _generate_response_format(self, resources: Dict[str, List[Any]]) -> str:
        """Generate the expected response format based on available resource categories."""
        format_lines = []
        for category in resources.keys():
            category_upper = category.upper()
            format_lines.append(f"{category_upper}: [list of indices]")

        example_lines = []
        for category in resources.keys():
            category_upper = category.upper()
            example_lines.append(f"{category_upper}: [0, 2, 5]")

        format_str = "\n".join(format_lines)
        example_str = "\n".join(example_lines)

        return f"{format_str}\n\nFor example:\n{example_str}"

    def _parse_llm_response(
        self, response: str, resources: Dict[str, List[Any]]
    ) -> Dict[str, List[int]]:
        """Parse the LLM response to extract the selected indices."""
        selected_indices = {category: [] for category in resources.keys()}

        # Extract indices for each category
        for category in resources.keys():
            category_upper = category.upper()
            pattern = rf"{category_upper}:\s*\[(.*?)\]"
            match = re.search(pattern, response, re.IGNORECASE)

            if match and match.group(1).strip():
                with contextlib.suppress(ValueError):
                    indices = [
                        int(idx.strip())
                        for idx in match.group(1).split(",")
                        if idx.strip()
                    ]
                    selected_indices[category] = indices

        return selected_indices

    def similarity_retrieval(
        self, query: str, resources: List[Any], top_k: int = 5
    ) -> List[Any]:
        """Simple similarity-based retrieval (placeholder implementation).

        Args:
            query: The query to match against
            resources: List of resources to search through
            top_k: Number of top results to return

        Returns:
            List of most relevant resources
        """
        # This is a simple implementation - in practice you might want to use
        # embeddings or more sophisticated similarity measures
        query_lower = query.lower()
        scored_resources = []

        for resource in resources:
            score = 0
            if isinstance(resource, dict):
                # Score based on name and description
                name = resource.get("name", "").lower()
                desc = resource.get("description", "").lower()
                text = f"{name} {desc}"
            elif hasattr(resource, "name") and hasattr(resource, "description"):
                # Score based on object attributes
                name = getattr(resource, "name", "").lower()
                desc = getattr(resource, "description", "").lower()
                text = f"{name} {desc}"
            else:
                # Score based on string representation
                text = str(resource).lower()

            # Simple keyword matching
            query_words = query_lower.split()
            for word in query_words:
                if word in text:
                    score += 1

            scored_resources.append((score, resource))

        # Sort by score and return top_k
        scored_resources.sort(key=lambda x: x[0], reverse=True)
        return [resource for score, resource in scored_resources[:top_k]]
