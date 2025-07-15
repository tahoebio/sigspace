"""LangChain native tools for Tahoe Agent."""

from typing import Any, Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ========================================
# LangChain Native Tools
# ========================================


class PythonExecutorArgs(BaseModel):
    """Schema for Python executor arguments."""

    code: str = Field(description="Python code to execute")


@tool(args_schema=PythonExecutorArgs)
def python_executor(code: str) -> str:
    """Execute Python code and return the result.

    Args:
        code: Python code to execute

    Returns:
        String result of code execution
    """
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


class WebSearchArgs(BaseModel):
    """Schema for web search arguments."""

    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")


@tool(args_schema=WebSearchArgs)
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        String representation of search results
    """
    if not query:
        return "Error: No query provided"

    # This is a placeholder implementation
    # In a real implementation, you would use a search API
    return (
        f"Search results for '{query}' (placeholder - {num_results} results requested)"
    )
