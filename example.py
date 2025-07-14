#!/usr/bin/env python3
"""
Simple example demonstrating the Tahoe Agent with Typer CLI.

This example shows:
1. Basic agent creation and usage
2. Adding custom tools
3. Using different LLMs (OpenAI, Anthropic, Lambda)
4. Tool retrieval

Usage:
    python example.py basic
    python example.py lambda-ai --model hermes-3-llama-3.1-405b-fp8
    python example.py anthropic --model claude-3-5-sonnet-20241022
    python example.py tools --provider openai
"""

from typing import Optional

import typer
from typing_extensions import Annotated

from tahoe_agent import BaseAgent
from tahoe_agent.tool import BaseTool

app = typer.Typer(help="Tahoe Agent Demo - Test different AI providers and features")


# Custom tool for demonstrations
class TemperatureConverter(BaseTool):
    def __init__(self):
        super().__init__(
            name="convert_temperature",
            description="Convert temperature between Celsius and Fahrenheit",
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Temperature value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (C or F)",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (C or F)",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
            required_parameters=["value", "from_unit", "to_unit"],
        )

    def execute(self, value: float, from_unit: str, to_unit: str) -> str:
        from_unit = from_unit.upper()
        to_unit = to_unit.upper()

        if from_unit == to_unit:
            return f"{value}°{to_unit}"

        if from_unit == "C" and to_unit == "F":
            result = (value * 9 / 5) + 32
            return f"{value}°C = {result:.2f}°F"
        elif from_unit == "F" and to_unit == "C":
            result = (value - 32) * 5 / 9
            return f"{value}°F = {result:.2f}°C"
        else:
            return "Invalid units. Use C or F."


@app.command()
def basic(
    provider: Annotated[str, typer.Option(help="AI provider")] = "openai",
    model: Annotated[Optional[str], typer.Option(help="Model name")] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Basic agent usage demo."""

    # Set defaults based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        elif provider == "lambda":
            model = "hermes-3-llama-3.1-405b-fp8"
        else:
            typer.echo(f"Unknown provider: {provider}", err=True)
            raise typer.Exit(1)

    source_map = {"openai": "OpenAI", "anthropic": "Anthropic", "lambda": "Lambda"}
    source = source_map.get(provider)

    typer.echo(f"🤖 Basic Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)
        log, response = agent.run("Calculate 15 * 23 + 7")

        typer.echo("💬 Question: Calculate 15 * 23 + 7")
        typer.echo(f"🎯 Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        _show_setup_help(provider)
        raise typer.Exit(1)


@app.command()
def lambda_ai(
    model: Annotated[
        str, typer.Option(help="Lambda model name")
    ] = "hermes-3-llama-3.1-405b-fp8",
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Lambda AI specific demo."""

    typer.echo(f"🚀 Lambda AI Demo ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source="Lambda", temperature=temperature)
        log, response = agent.run(
            "Hello! Can you help me with a simple math problem: what is 42 * 17?"
        )

        typer.echo(
            "💬 Question: Hello! Can you help me with a simple math problem: what is 42 * 17?"
        )
        typer.echo(f"🎯 Lambda Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Lambda demo failed: {e}", err=True)
        _show_setup_help("lambda")
        raise typer.Exit(1)


@app.command()
def anthropic(
    model: Annotated[
        str, typer.Option(help="Anthropic model name")
    ] = "claude-3-5-sonnet-20241022",
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Anthropic specific demo."""

    typer.echo(f"🧠 Anthropic Demo ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source="Anthropic", temperature=temperature)
        log, response = agent.run("Write a short haiku about programming.")

        typer.echo("💬 Question: Write a short haiku about programming.")
        typer.echo(f"🎯 Anthropic Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Anthropic demo failed: {e}", err=True)
        _show_setup_help("anthropic")
        raise typer.Exit(1)


@app.command()
def tools(
    provider: Annotated[str, typer.Option(help="AI provider")] = "openai",
    model: Annotated[Optional[str], typer.Option(help="Model name")] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Custom tools demonstration."""

    # Set defaults based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        elif provider == "lambda":
            model = "hermes-3-llama-3.1-405b-fp8"
        else:
            typer.echo(f"Unknown provider: {provider}", err=True)
            raise typer.Exit(1)

    source_map = {"openai": "OpenAI", "anthropic": "Anthropic", "lambda": "Lambda"}
    source = source_map.get(provider)

    typer.echo(f"🔧 Custom Tools Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        # Create agent and add custom tools
        agent = BaseAgent(llm=model, source=source, temperature=temperature)

        # Add a simple function tool
        def fibonacci(n: int) -> int:
            """Calculate the nth Fibonacci number."""
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        agent.add_tool(fibonacci)
        agent.add_tool(TemperatureConverter())

        # Test the tools
        typer.echo("🧮 Testing Fibonacci tool...")
        log, response = agent.run("What's the 10th Fibonacci number?")
        typer.echo(f"🎯 Agent: {response}")

        typer.echo("\n🌡️  Testing Temperature Converter tool...")
        log, response = agent.run("Convert 100°F to Celsius")
        typer.echo(f"🎯 Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        _show_setup_help(provider)
        raise typer.Exit(1)


@app.command()
def conversation(
    provider: Annotated[str, typer.Option(help="AI provider")] = "openai",
    model: Annotated[Optional[str], typer.Option(help="Model name")] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Conversation with memory demonstration."""

    # Set defaults based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        elif provider == "lambda":
            model = "hermes-3-llama-3.1-405b-fp8"
        else:
            typer.echo(f"Unknown provider: {provider}", err=True)
            raise typer.Exit(1)

    source_map = {"openai": "OpenAI", "anthropic": "Anthropic", "lambda": "Lambda"}
    source = source_map.get(provider)

    typer.echo(f"💭 Conversation Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)

        # Have a conversation with memory
        prompts = [
            "My name is Alice and I like Python programming.",
            "What did I tell you about myself?",
            "Can you help me calculate 2^8?",
            "What's my name again?",
        ]

        for i, prompt in enumerate(prompts, 1):
            typer.echo(f"\n{i}. 👤 User: {prompt}")
            log, response = agent.run(prompt)
            typer.echo(f"   🎯 Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        _show_setup_help(provider)
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """List example models for each provider."""

    typer.echo("📋 Available Models by Provider")
    typer.echo("=" * 40)

    typer.echo("\n🔵 OpenAI:")
    typer.echo("  • gpt-4o")
    typer.echo("  • gpt-4o-mini")
    typer.echo("  • gpt-4-turbo")

    typer.echo("\n🟣 Anthropic:")
    typer.echo("  • claude-3-5-sonnet-20241022")
    typer.echo("  • claude-3-5-haiku-20241022")
    typer.echo("  • claude-3-opus-20240229")

    typer.echo("\n🚀 Lambda:")
    typer.echo("  • hermes-3-llama-3.1-405b-fp8")
    typer.echo("  • llama-3.1-405b-instruct-fp8")
    typer.echo("  • llama-3.1-70b-instruct")


def _show_setup_help(provider: str) -> None:
    """Show setup instructions for a provider."""
    typer.echo("\n🔧 Setup Instructions:")

    if provider == "openai":
        typer.echo("  export OPENAI_API_KEY='your-openai-api-key'")
    elif provider == "anthropic":
        typer.echo("  export ANTHROPIC_API_KEY='your-anthropic-api-key'")
    elif provider == "lambda":
        typer.echo("  export LAMBDA_API_KEY='your-lambda-api-key'")


if __name__ == "__main__":
    app()
