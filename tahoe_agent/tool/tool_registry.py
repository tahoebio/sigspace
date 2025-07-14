"""Tool registry for managing tools in Tahoe Agent."""

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from tahoe_agent.tool.base_tool import BaseTool


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: List[Union[BaseTool, Dict[str, Any]]] = []
        self.tool_index: Dict[str, int] = {}
        self.document_df: Optional[pd.DataFrame] = None
        self._rebuild_document_df()

    def register_tool(self, tool: Union[BaseTool, Dict[str, Any]]) -> int:
        """Register a tool in the registry.

        Args:
            tool: Tool to register (either BaseTool instance or dictionary)

        Returns:
            Index of the registered tool
        """
        if isinstance(tool, BaseTool):
            tool_name = tool.name
        elif isinstance(tool, dict):
            tool_name = tool.get("name", f"tool_{len(self.tools)}")
        else:
            raise ValueError("Tool must be either BaseTool instance or dictionary")

        # Check for existing tool with same name
        if tool_name in self.tool_index:
            # Update existing tool
            existing_index = self.tool_index[tool_name]
            self.tools[existing_index] = tool
            index = existing_index
        else:
            # Add new tool
            index = len(self.tools)
            self.tools.append(tool)
            self.tool_index[tool_name] = index

        self._rebuild_document_df()
        return index

    def get_tool_by_name(self, name: str) -> Optional[Union[BaseTool, Dict[str, Any]]]:
        """Get a tool by name.

        Args:
            name: Name of the tool

        Returns:
            Tool if found, None otherwise
        """
        if name in self.tool_index:
            return self.tools[self.tool_index[name]]
        return None

    def get_tool_by_id(self, tool_id: int) -> Optional[Union[BaseTool, Dict[str, Any]]]:
        """Get a tool by ID.

        Args:
            tool_id: ID of the tool

        Returns:
            Tool if found, None otherwise
        """
        if 0 <= tool_id < len(self.tools):
            return self.tools[tool_id]
        return None

    def remove_tool_by_name(self, name: str) -> bool:
        """Remove a tool by name.

        Args:
            name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self.tool_index:
            return False

        # Get the index of the tool to remove
        remove_index = self.tool_index[name]

        # Remove the tool
        del self.tools[remove_index]

        # Rebuild the index (all indices after the removed one shift down)
        self.tool_index = {}
        for i, tool in enumerate(self.tools):
            if isinstance(tool, BaseTool):
                tool_name = tool.name
            elif isinstance(tool, dict):
                tool_name = tool.get("name", f"tool_{i}")
            else:
                tool_name = f"tool_{i}"
            self.tool_index[tool_name] = i

        self._rebuild_document_df()
        return True

    def list_tools(self) -> List[str]:
        """List all tool names.

        Returns:
            List of tool names
        """
        tool_names = []
        for tool in self.tools:
            if isinstance(tool, BaseTool):
                tool_names.append(tool.name)
            elif isinstance(tool, dict):
                tool_names.append(tool.get("name", "unknown"))
            else:
                tool_names.append(str(tool))
        return tool_names

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools.

        Returns:
            List of tool schemas
        """
        schemas = []
        for tool in self.tools:
            if isinstance(tool, BaseTool):
                schemas.append(tool.to_schema())
            elif isinstance(tool, dict):
                schemas.append(tool)
            else:
                # Try to extract schema from object
                schema = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                }
                schemas.append(schema)
        return schemas

    def search_tools(self, query: str) -> List[Union[BaseTool, Dict[str, Any]]]:
        """Search for tools by query.

        Args:
            query: Search query

        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []

        for tool in self.tools:
            if isinstance(tool, BaseTool):
                text = f"{tool.name} {tool.description}".lower()
            elif isinstance(tool, dict):
                name = tool.get("name", "")
                desc = tool.get("description", "")
                text = f"{name} {desc}".lower()
            else:
                text = str(tool).lower()

            if query_lower in text:
                matching_tools.append(tool)

        return matching_tools

    def _rebuild_document_df(self):
        """Rebuild the document DataFrame for retrieval."""
        docs = []
        for i, tool in enumerate(self.tools):
            if isinstance(tool, BaseTool):
                content = f"{tool.name}: {tool.description}"
            elif isinstance(tool, dict):
                name = tool.get("name", "unknown")
                desc = tool.get("description", "")
                content = f"{name}: {desc}"
            else:
                content = str(tool)

            docs.append([i, content])

        self.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary format.

        Returns:
            Dictionary representation of the registry
        """
        return {
            "tools": self.get_tool_schemas(),
            "tool_index": self.tool_index,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load registry from dictionary format.

        Args:
            data: Dictionary containing registry data
        """
        self.tools = []
        self.tool_index = {}

        for tool_data in data.get("tools", []):
            self.register_tool(tool_data)

    def __len__(self) -> int:
        """Get number of tools in registry."""
        return len(self.tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool exists in registry."""
        return tool_name in self.tool_index

    def __iter__(self):
        """Iterate over tools in registry."""
        return iter(self.tools)
