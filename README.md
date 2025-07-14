# Tahoe Agent

A minimal agent implementation inspired by the Biomni framework, built with LangChain and LangGraph.

## Features

- **Multi-LLM Support**: OpenAI, Anthropic, Google Gemini, Ollama, Lambda Labs, and custom model serving
- **Tool System**: Easy-to-use tool registration and execution
- **Retrieval**: Optional tool retrieval for large tool sets
- **Extensible**: Simple base classes for tools and tasks
- **Conversational**: Memory and conversation history
- **LangGraph Workflow**: Modern agent workflow management

## Installation

```bash
cd tahoe-agent
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from tahoe_agent import BaseAgent

# Create an agent
agent = BaseAgent(
    llm="claude-3-5-sonnet-20241022",  # or any supported model
    temperature=0.7
)

# Chat with the agent
response = agent.chat("What's 2+2?")
print(response)

# Run with full logging
log, response = agent.run("Calculate the factorial of 5")
for role, content in log:
    print(f"{role}: {content}")
```

### Using Different LLMs

```python
# OpenAI
agent = BaseAgent(llm="gpt-4")

# Ollama (local)
agent = BaseAgent(llm="llama2")

# Lambda Labs
agent = BaseAgent(
    llm="hermes-3-llama-3.1-405b-fp8",
    source="Lambda",
    api_key="your-lambda-api-key"
)

# Custom model serving
agent = BaseAgent(
    llm="custom-model",
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)
```

### Adding Custom Tools

```python
from tahoe_agent import BaseAgent
from tahoe_agent.tool import BaseTool

# Method 1: Using a simple function
def calculate_square(number: int) -> int:
    """Calculate the square of a number."""
    return number ** 2

agent = BaseAgent()
agent.add_tool(calculate_square)

# Method 2: Creating a custom tool class
class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get weather information for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            },
            required_parameters=["city"]
        )

    def execute(self, city: str) -> str:
        # Your weather API call here
        return f"Weather in {city}: Sunny, 72°F"

weather_tool = WeatherTool()
agent.add_tool(weather_tool)

# Use the tools
response = agent.chat("What's the square of 8 and the weather in San Francisco?")
```

### Using Retrieval

For agents with many tools, enable retrieval to automatically select relevant tools:

```python
agent = BaseAgent(
    llm="claude-3-5-sonnet-20241022",
    use_retriever=True  # Enable tool retrieval
)

# Add many tools...
agent.add_tool(tool1)
agent.add_tool(tool2)
# ... many more tools

# The agent will automatically select relevant tools based on the query
response = agent.chat("I need to analyze some data")
```

### Working with Tasks

```python
from tahoe_agent.task import SimpleTask, DataAnalysisTask

# Simple task
def process_data(data):
    return [x * 2 for x in data]

task = SimpleTask(
    name="double_data",
    description="Double all values in a list",
    func=process_data
)

result = task.run(data=[1, 2, 3, 4, 5])
print(result)  # [2, 4, 6, 8, 10]

# Data analysis task
analysis_task = DataAnalysisTask(
    name="sales_analysis",
    description="Analyze sales data",
    data_source="sales.csv",
    analysis_type="trend_analysis"
)

result = analysis_task.run()
```

## Architecture

### Core Components

1. **BaseAgent**: Main agent class that orchestrates LLM interactions and tool execution
2. **ToolRegistry**: Manages available tools and their schemas
3. **Retriever**: Selects relevant tools based on queries (optional)
4. **BaseTool**: Base class for creating custom tools
5. **BaseTask**: Base class for defining reusable tasks

### Workflow

1. User provides input
2. (Optional) Retriever selects relevant tools
3. System prompt is generated with available tools
4. LLM processes input and decides on tool usage
5. Tools are executed as needed
6. Results are returned to user

## Extending the Agent

### Custom Tool Example

```python
from tahoe_agent.tool import BaseTool
import requests

class APITool(BaseTool):
    def __init__(self, api_url: str, api_key: str):
        super().__init__(
            name="api_call",
            description="Make API calls to external services",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint"},
                    "method": {"type": "string", "description": "HTTP method"},
                    "data": {"type": "object", "description": "Request data"}
                },
                "required": ["endpoint", "method"]
            },
            required_parameters=["endpoint", "method"]
        )
        self.api_url = api_url
        self.api_key = api_key

    def execute(self, endpoint: str, method: str = "GET", data: dict = None) -> str:
        url = f"{self.api_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.request(method, url, headers=headers, json=data)
        return response.json()
```

### Custom Agent Example

```python
from tahoe_agent import BaseAgent

class SpecializedAgent(BaseAgent):
    def __init__(self, domain: str, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self._setup_domain_tools()

    def _setup_domain_tools(self):
        """Add domain-specific tools."""
        if self.domain == "data_science":
            # Add data science tools
            pass
        elif self.domain == "web_dev":
            # Add web development tools
            pass

    def _generate_system_prompt(self, tools=None):
        """Override to add domain-specific instructions."""
        base_prompt = super()._generate_system_prompt(tools)
        domain_prompt = f"\nYou are specialized in {self.domain}. "
        return base_prompt + domain_prompt

# Usage
data_agent = SpecializedAgent(
    domain="data_science",
    llm="claude-3-5-sonnet-20241022"
)
```

## Configuration

### Environment Variables

Set up your API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export LAMBDA_API_KEY="your-lambda-key"
```

### Tool Timeout

```python
agent = BaseAgent(timeout_seconds=300)  # 5 minute timeout for tool execution
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black tahoe_agent/
isort tahoe_agent/
```

### Type Checking

```bash
mypy tahoe_agent/
```

## Comparison with Biomni

This skeleton provides a simplified version of the Biomni agent with:

- **Similarities**: LangGraph workflows, tool registry, retrieval system, multi-LLM support
- **Simplifications**: No specialized domain tools, simplified prompt generation, basic retrieval
- **Extensions**: Better type hints, modular design, easier customization

## License

MIT License - feel free to use and modify as needed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Examples

See the `examples/` directory for more detailed usage examples and tutorials.
