#!/usr/bin/env python3
"""
Vision Scores Analysis example demonstrating the Tahoe Agent with LangChain native tools.

This example shows:
1. Vision scores analysis with different LLM providers
2. Native LangChain tool integration
3. Static DAG workflow (prompt -> tool_calling -> retrieval)
4. Configurable path management (NEW)

Usage:
    python example.py vision-scores --data-path /Users/rohit/Desktop/tahoe_data --cell-name HS-578T
    python example.py vision-scores --provider anthropic --model claude-3-5-sonnet-20241022
    python example.py vision-scores --provider lambda --model hermes-3-llama-3.1-405b-fp8
    python example.py models
    python example.py paths  # NEW: Show path configuration
"""

from typing import Optional

import typer
from typing_extensions import Annotated

import pathlib
import pandas as pd
from tahoe_agent.paths import get_paths

from tahoe_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths  # NEW
from tahoe_agent.llm import SourceType
import typing

app = typer.Typer(
    help="Tahoe Agent Vision Scores Demo - Analyze vision scores with different AI providers"
)


@app.command()
def vision_scores(
    data_path: Annotated[
        str, typer.Option(help="Path to directory containing h5ad files")
    ] = "/Users/rohit/Desktop/tahoe_data",
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
        agent = BaseAgent(
            llm=model,
            source=typing.cast(SourceType, source_map[provider]),
            temperature=0.7,
            tool_config={"drug_name": drug_name},  # <-- This is required for blind evaluation!
        )

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
        agent = BaseAgent(llm=model, source=typing.cast(SourceType, source_map[provider]), temperature=temperature)  # type: ignore

        # Simple prompt to test the tool
        prompt = "Analyze vision scores for cell HS-578T and drug TestDrug using differential scores from /Users/rohit/Desktop/tahoe_data"

        log, response = agent.run(prompt)

        typer.echo(f"💬 Question: {prompt}")
        typer.echo(f"🎯 Agent: {response}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def drug_ranking(
    data_path: Annotated[
        str, typer.Option(help="Path to directory containing h5ad files")
    ] = "/Users/rohit/Desktop/tahoe_data",
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

    source_map = {"openai": "OpenAI", "anthropic": "Anthropic", "lambda": "Lambda"}
    source = source_map.get(provider)

    typer.echo(f"💊 Drug Ranking Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        # Initialize agent with hidden drug configuration
        agent = BaseAgent(
            llm=model,
            source=typing.cast(SourceType, source_map[provider]),
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

        log, response = agent.run(prompt)

        # Show execution log
        typer.echo("\n📝 Execution log:")
        for i, (role, content) in enumerate(log):
            typer.echo(
                f"  {i+1}. [{role}]: {content[:150]}{'...' if len(content) > 150 else ''}"
            )

        typer.echo(f"\n🎯 Final Drug Rankings (Hidden drug was: {drug_name}):")
        typer.echo("=" * 50)
        typer.echo(response)

    except Exception as e:
        typer.echo(f"❌ Drug ranking demo failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def batch_vision_scores(
    data_path: Annotated[
        str, typer.Option(help="Path to directory containing h5ad files")
    ] = "/Users/rohit/Desktop/tahoe_data",
    cell_name: Annotated[
        Optional[str], typer.Option(help="Cell name to analyze")
    ] = None,
    provider: Annotated[str, typer.Option(help="AI provider")] = "lambda",
    model: Annotated[
        Optional[str], typer.Option(help="Model name")
    ] = "hermes-3-llama-3.1-405b-fp8",
    custom_data_dir: Annotated[
        Optional[str], typer.Option(help="Custom data directory")
    ] = None,
    custom_results_dir: Annotated[
        Optional[str], typer.Option(help="Custom results directory")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing summaries")] = False,
) -> None:
    """Batch vision scores analysis for all drugs (optionally filtered by cell name), using the agent."""
    import pathlib
    # Configure custom paths if provided
    if custom_data_dir or custom_results_dir:
        typer.echo("🔧 Configuring custom paths...")
        config_kwargs = {}
        if custom_data_dir:
            config_kwargs["data_dir"] = custom_data_dir
        if custom_results_dir:
            config_kwargs["results_dir"] = custom_results_dir
        configure_paths(**config_kwargs)
        paths = get_paths()
        typer.echo(f"  Data directory: {paths.data_dir}")
        typer.echo(f"  Results directory: {paths.results_dir}")
    else:
        paths = get_paths()

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

    # Load drugs from CSV
    drugs_csv = paths.drugs_file
    df = pd.read_csv(drugs_csv)
    drugs = df["drug"].unique().tolist()

    # Prepare output directory
    output_dir = paths.results_dir / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"🔬 Batch Vision Scores Analysis (Agent) - {provider.title()} ({model})")
    typer.echo(f"Processing {len(drugs)} drugs...")
    typer.echo(f"Results will be saved in: {output_dir}")

    # Create agent (vision scores tool is added by default)
    from tahoe_agent import BaseAgent
    import typing

    drugs = drugs[:3]

    for drug in drugs:
        agent = BaseAgent(
            llm=model,
            source=typing.cast(SourceType, source),
            temperature=0.7,
            tool_config={"drug_name": drug},
        )

        summary_path = output_dir / f"{drug}.txt"
        if summary_path.exists() and not overwrite:
            typer.echo(f"[batch_vision_scores] Skipping {drug} (already exists)")
            continue
        typer.echo(f"[batch_vision_scores] Analyzing drug: {drug} (cell: {cell_name})")
        try:
            # Build the same prompt as in vision_scores
            prompt = f"Use the analyze_vision_scores tool to analyze the vision scores for cell '{cell_name}' and drug '{drug}'. Show me the top features and provide insights about the results."
            log, response = agent.run(prompt)
            with open(summary_path, "w") as f:
                f.write(response)
            typer.echo(f"[batch_vision_scores] Saved result for {drug} to {summary_path}")
        except Exception as e:
            typer.echo(f"[batch_vision_scores] Error for {drug}: {e}", err=True)

    # After batch is done, run embedding comparison
    typer.echo("\n[batch_vision_scores] Running embedding comparison on generated summaries...")
    from benchmark.compare_embeddings import compare_embeddings
    compare_embeddings(output_dir, drugs_csv, output_dir)
    typer.echo(f"[batch_vision_scores] Embedding comparison complete. Plots saved in: {output_dir}")

if __name__ == "__main__":
    app()
