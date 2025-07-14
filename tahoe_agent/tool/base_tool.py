"""Base tool class for Tahoe Agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """Base class for all tools in Tahoe Agent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        required_parameters: Optional[List[str]] = None,
    ):
        """Initialize the base tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: Dictionary describing the tool's parameters
            required_parameters: List of required parameter names
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.required_parameters = required_parameters or []

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters for tool execution

        Returns:
            Result of tool execution
        """
        pass

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate that all required parameters are provided.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if all required parameters are present

        Raises:
            ValueError: If required parameters are missing
        """
        missing_params = [
            param for param in self.required_parameters if param not in kwargs
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        return True

    def to_schema(self) -> Dict[str, Any]:
        """Convert the tool to a schema dictionary.

        Returns:
            Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required_parameters": self.required_parameters,
        }

    def __call__(self, **kwargs: Any) -> Any:
        """Make the tool callable.

        Args:
            **kwargs: Parameters for tool execution

        Returns:
            Result of tool execution
        """
        self.validate_parameters(**kwargs)
        return self.execute(**kwargs)

    def __str__(self) -> str:
        """String representation of the tool."""
        return f"Tool(name='{self.name}', description='{self.description}')"

    def __repr__(self) -> str:
        """Detailed string representation of the tool."""
        return (
            f"Tool(name='{self.name}', description='{self.description}', "
            f"parameters={self.parameters}, required_parameters={self.required_parameters})"
        )


class PythonTool(BaseTool):
    """A tool that executes Python code."""

    def __init__(self) -> None:
        super().__init__(
            name="python_executor",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
            required_parameters=["code"],
        )

    def execute(self, **kwargs: Any) -> str:
        """Execute Python code and return the result.

        Args:
            **kwargs: Must contain 'code' parameter with Python code to execute

        Returns:
            String result of code execution
        """
        code = kwargs.get("code", "")
        if not code:
            return "Error: No code provided"

        import io
        from contextlib import redirect_stdout, redirect_stderr

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Create a local namespace for execution
                local_namespace: Dict[str, Any] = {}
                exec(code, {"__builtins__": __builtins__}, local_namespace)

            # Get the output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()

            if stderr_content:
                return f"Error: {stderr_content}"
            elif stdout_content:
                return stdout_content
            else:
                return "Code executed successfully (no output)"

        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchTool(BaseTool):
    """A placeholder tool for web search functionality."""

    def __init__(self) -> None:
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            required_parameters=["query"],
        )

    def execute(self, **kwargs: Any) -> str:
        """Execute web search (placeholder implementation).

        Args:
            **kwargs: Must contain 'query', optionally 'num_results'

        Returns:
            String representation of search results
        """
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)

        if not query:
            return "Error: No query provided"

        # This is a placeholder implementation
        # In a real implementation, you would use a search API
        return f"Search results for '{query}' (placeholder - {num_results} results requested)"
