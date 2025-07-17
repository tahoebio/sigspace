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

from typing import Optional, Dict

import typer
from typing_extensions import Annotated

from tahoe_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths  # NEW
from tahoe_agent.llm import SourceType


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

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
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
        agent = BaseAgent(llm=model, source=source, temperature=0.7)

        # Test the vision scores tool
        typer.echo(f"📊 Analyzing vision scores for cell: {cell_name}")
        typer.echo(f"💊 Drug: {drug_name}")  # NEW
        typer.echo(f"📂 Data path: {data_path}")

        # Create a prompt for the agent to use the tool
        typer.echo("\n🤖 Testing vision scores analysis with agent...")
        prompt = f"Use the analyze_vision_scores tool to analyze the vision scores for cell '{cell_name}' and drug '{drug_name}'. Show me the top features and provide insights about the results."

        log, response, _, _ = agent.run(prompt)

        # Show the full log for debugging
        typer.echo("\n📝 Full execution log:")
        for i, (role, content) in enumerate(log):
            typer.echo(
                f"  {i+1}. [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        typer.echo(f"\n🎯 Final Agent Response: {response}")
        # typer.echo(f"🔍 Summary: {state['summary']}")
        # typer.echo(f"🔍 Drug Rankings: {state['drug_rankings']}")

    except Exception as e:
        typer.echo(f"❌ Vision scores demo failed: {e}", err=True)
        typer.echo("\n💡 Note: Make sure the h5ad files exist in the specified path:")
        typer.echo("  • 20250417.diff_vision_scores_pseudobulk.public.h5ad")
        typer.echo("  • 20250417.vision_scores_pseudobulk.public.h5ad")
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

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
    source = source_map.get(provider)

    typer.echo(f"🔬 Basic Vision Scores Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)

        # Simple prompt to test the tool
        prompt = "Analyze vision scores for cell HS-578T and drug TestDrug using differential scores from /Users/gpalla/Datasets/tahoe"

        log, response, _, _ = agent.run(prompt)

        typer.echo(f"💬 Question: {prompt}")
        typer.echo(f"🎯 Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def drug_ranking(
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
    ] = "llama-4-maverick-17b-128e-instruct-fp8",  # "hermes-3-llama-3.1-405b-fp8", #"deepseek-r1-671b",  #
    # NEW: Path configuration options
    custom_data_dir: Annotated[
        Optional[str], typer.Option(help="Custom data directory")
    ] = None,
    custom_results_dir: Annotated[
        Optional[str], typer.Option(help="Custom results directory")
    ] = None,
) -> None:
    """Test drug ranking functionality."""

    # NEW: Configure custom paths if provided
    if custom_data_dir or custom_results_dir:
        typer.echo("🔧 Configuring custom paths...")
        config_kwargs = {}
        if custom_data_dir:
            config_kwargs["data_dir"] = custom_data_dir
        if custom_results_dir:
            config_kwargs["results_dir"] = custom_results_dir
        configure_paths(**config_kwargs)
        typer.echo(f"   Data directory: {get_paths().data_dir}")
        typer.echo(f"   Results directory: {get_paths().results_dir}")

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

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
    source = source_map.get(provider)

    typer.echo(f"💊 Drug Ranking Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        # Initialize agent with hidden drug configuration
        agent = BaseAgent(
            llm=model,
            source=source,
            temperature=0.7,
            tool_config={"drug_name": drug_name},  # Hidden from LLM
        )

        # Build prompt based on whether cell_name is provided
        base_prompt = "Use the analyze_vision_scores tool to analyze the vision scores"
        if cell_name is not None:
            base_prompt += f" for cell line '{cell_name}'"

        prompt = f"""{base_prompt}.

        Show me the top features and provide insights about the results.

        After the analysis, when the summary is available, rank drugs based on how well their mechanisms of action match the observed biological signatures.
        """

        typer.echo("💬 Testing drug ranking with BLIND prompt (drug name hidden):")
        typer.echo(f"   Hidden drug: {drug_name}")
        typer.echo(f"   Prompt: {prompt[:100]}...")
        typer.echo("\n🤖 Running agent workflow...")

        log, response, structured_rankings, summary = agent.run(prompt)

        # Show execution log
        typer.echo("\n📝 Execution log:")
        for i, (role, content) in enumerate(log):
            typer.echo(
                f"  {i+1}. [{role}]: {content[:150]}{'...' if len(content) > 150 else ''}"
            )

        typer.echo(f"\n🎯 Final Drug Rankings (Hidden drug was: {drug_name}):")
        typer.echo("=" * 50)
        typer.echo(response)

        # Save structured results to a file
        # if structured_rankings or summary:
        #     results_path = get_paths().results_dir / "drug_rankings.json"
        #     typer.echo(f"\n💾 Saving structured results to {results_path}...")
        #     results_data = {}
        #     if summary:
        #         results_data["summary"] = summary
        #     if structured_rankings:
        #         # Convert Pydantic models to dictionaries for JSON serialization
        #         results_data["rankings"] = [
        #             r.dict() for r in structured_rankings
        #         ]
        #     with open(results_path, "w") as f:
        #         json.dump(results_data, f, indent=2)
        #     typer.echo("   Done.")

    except Exception as e:
        typer.echo(f"❌ Drug ranking demo failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
