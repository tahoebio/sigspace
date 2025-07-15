#!/usr/bin/env python3
"""
Vision Scores Analysis example demonstrating the Tahoe Agent with LangChain native tools.

This example shows:
1. Vision scores analysis with different LLM providers
2. Native LangChain tool integration
3. Static DAG workflow (prompt -> tool_calling -> retrieval)
4. Configurable path management (NEW)

Usage:
    python example.py vision-scores --data-path /Users/gpalla/Datasets/tahoe --cell-name HS-578T
    python example.py vision-scores --provider anthropic --model claude-3-5-sonnet-20241022
    python example.py vision-scores --provider lambda --model hermes-3-llama-3.1-405b-fp8
    python example.py models
    python example.py paths  # NEW: Show path configuration
"""

from typing import Optional

import typer
from typing_extensions import Annotated

from tahoe_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths  # NEW

app = typer.Typer(
    help="Tahoe Agent Vision Scores Demo - Analyze vision scores with different AI providers"
)


@app.command()
def vision_scores(
    data_path: Annotated[
        str, typer.Option(help="Path to directory containing h5ad files")
    ] = "/Users/gpalla/Datasets/tahoe",
    cell_name: Annotated[
        Optional[str], typer.Option(help="Cell name to analyze")
    ] = None,
    drug_name: Annotated[str, typer.Option(help="Drug name to analyze")] = "Adagrasib",
    provider: Annotated[str, typer.Option(help="AI provider")] = "lambda",
    model: Annotated[
        Optional[str], typer.Option(help="Model name")
    ] = "hermes-3-llama-3.1-405b-fp8",
    # NEW: Path configuration options
    custom_data_dir: Annotated[
        Optional[str], typer.Option(help="Custom data directory")
    ] = None,
    custom_results_dir: Annotated[
        Optional[str], typer.Option(help="Custom results directory")
    ] = None,
) -> None:
    """Vision scores analysis demonstration."""

    # NEW: Configure custom paths if provided
    if custom_data_dir or custom_results_dir:
        typer.echo("🔧 Configuring custom paths...")
        config_kwargs = {}
        if custom_data_dir:
            config_kwargs["data_dir"] = custom_data_dir
        if custom_results_dir:
            config_kwargs["results_dir"] = custom_results_dir
        configure_paths(**config_kwargs)

        # Show the updated configuration
        paths = get_paths()
        typer.echo(f"  Data directory: {paths.data_dir}")
        typer.echo(f"  Results directory: {paths.results_dir}")

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

    typer.echo(f"🔬 Vision Scores Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    from tahoe_agent.tool.vision_scores import analyze_vision_scores

    # Test direct vision scores function call first
    typer.echo("\n🔧 Testing direct vision scores function call...")
    result = analyze_vision_scores.invoke(
        input={
            "cell_name": cell_name,
            "drug_name": drug_name,  # NEW
        }
    )
    typer.echo(result)
    typer.echo("-" * 50 + "\n")

    try:
        # Create agent (vision scores tool is added by default)
        agent = BaseAgent(llm=model, source=source, temperature=0.7)  # type: ignore

        # Test the vision scores tool
        typer.echo(f"📊 Analyzing vision scores for cell: {cell_name}")
        typer.echo(f"💊 Drug: {drug_name}")  # NEW
        typer.echo(f"📂 Data path: {data_path}")

        # Create a prompt for the agent to use the tool
        typer.echo("\n🤖 Testing vision scores analysis with agent...")
        prompt = f"Use the analyze_vision_scores tool to analyze the vision scores for cell '{cell_name}' and drug '{drug_name}'. Show me the top features and provide insights about the results."

        log, response = agent.run(prompt)

        # Show the full log for debugging
        typer.echo("\n📝 Full execution log:")
        for i, (role, content) in enumerate(log):
            typer.echo(
                f"  {i+1}. [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        typer.echo(f"\n🎯 Final Agent Response: {response}")

    except Exception as e:
        typer.echo(f"❌ Vision scores demo failed: {e}", err=True)
        typer.echo("\n💡 Note: Make sure the h5ad files exist in the specified path:")
        typer.echo("  • 20250417.diff_vision_scores_pseudobulk.public.h5ad")
        typer.echo("  • 20250417.vision_scores_pseudobulk.public.h5ad")
        _show_setup_help(provider)
        raise typer.Exit(1)


@app.command()
def basic_demo(
    provider: Annotated[str, typer.Option(help="AI provider")] = "lambda",
    model: Annotated[Optional[str], typer.Option(help="Model name")] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Basic vision scores demo with simplified prompt."""

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

    typer.echo(f"🔬 Basic Vision Scores Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)  # type: ignore

        # Simple prompt to test the tool
        prompt = "Analyze vision scores for cell HS-578T and drug TestDrug using differential scores from /Users/gpalla/Datasets/tahoe"

        log, response = agent.run(prompt)

        typer.echo(f"💬 Question: {prompt}")
        typer.echo(f"🎯 Agent: {response}")

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


# NEW: Path configuration command
@app.command()
def paths(
    data_dir: Annotated[
        Optional[str], typer.Option(help="Set custom data directory")
    ] = None,
    results_dir: Annotated[
        Optional[str], typer.Option(help="Set custom results directory")
    ] = None,
    vision_diff_filename: Annotated[
        Optional[str], typer.Option(help="Set custom diff filename")
    ] = None,
    vision_pseudo_filename: Annotated[
        Optional[str], typer.Option(help="Set custom pseudo filename")
    ] = None,
) -> None:
    """Show or configure path settings."""

    # Configure paths if any are provided
    if any([data_dir, results_dir, vision_diff_filename, vision_pseudo_filename]):
        typer.echo("🔧 Configuring custom paths...")
        config_kwargs = {}
        if data_dir:
            config_kwargs["data_dir"] = data_dir
        if results_dir:
            config_kwargs["results_dir"] = results_dir
        if vision_diff_filename:
            config_kwargs["vision_diff_filename"] = vision_diff_filename
        if vision_pseudo_filename:
            config_kwargs["vision_pseudo_filename"] = vision_pseudo_filename

        configure_paths(**config_kwargs)
        typer.echo("✅ Paths updated!")

    # Show current configuration
    paths_config = get_paths()
    typer.echo("\n📁 Current Path Configuration:")
    typer.echo("=" * 40)

    for key, value in paths_config.to_dict().items():
        typer.echo(f"  {key}: {value}")

    typer.echo("\n💡 You can also set paths via environment variables:")
    typer.echo("  export TAHOE_DATA_DIR=/custom/data/path")
    typer.echo("  export TAHOE_RESULTS_DIR=/custom/results/path")
    typer.echo("  export TAHOE_VISION_DIFF_FILE=custom_diff.h5ad")
    typer.echo("  export TAHOE_VISION_PSEUDO_FILE=custom_pseudo.h5ad")


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
