#!/usr/bin/env python3
"""
GSEA Scores Analysis example demonstrating the Tahoe Agent with LangChain native tools.

This example shows:
1. GSEA scores analysis with different LLM providers
2. Native LangChain tool integration
3. Static DAG workflow (prompt -> tool_calling -> retrieval)
4. Configurable path management (NEW)

Usage:
    python example_gsea.py gsea-scores --data-path ~/sigspace2/sid/Datasets/tahoe --custom-data-dir ~/sigspace2/sid/Datasets/tahoe --cell-name c_15
    python example_gsea.py drug-ranking --data-path ~/sigspace2/sid/Datasets/tahoe --custom-data-dir ~/sigspace2/sid/Datasets/tahoe --cell-name c_15
"""

from typing import Optional

import typer
from typing_extensions import Annotated
from tahoe_agent.logging_config import get_logger

from tahoe_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths  # NEW

logger = get_logger()

app = typer.Typer(
    help="Tahoe Agent GSEA Scores Demo - Analyze GSEA scores with different AI providers"
)


@app.command()
def gsea_scores(
    data_path: Annotated[
        str, typer.Option(help="Path to directory containing h5ad files")
    ] = "~/sigspace2/sid/Datasets/tahoe",
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
    """GSEA scores analysis demonstration."""
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

    typer.echo(f"🔬 GSEA Scores Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    from tahoe_agent.tool.gsea_scores import analyze_gsea_scores

    # Test direct GSEA scores function call first
    typer.echo("\n🔧 Testing direct GSEA scores function call...")
    result = analyze_gsea_scores.invoke(
        input={
            "cell_name": cell_name,
            "drug_name": drug_name,  # NEW
        }
    )
    typer.echo(result)
    typer.echo("-" * 50 + "\n")

    try:
        # Create agent (GSEA scores tool is added by default)
        agent = BaseAgent(
            llm=model,
            source=source,
            temperature=0.7,
            tool_config={"drug_name": drug_name},
        )  # Hidden from LLM  # type: ignore

        # Test the GSEA scores tool
        typer.echo(f"📊 Analyzing GSEA scores for cell: {cell_name}")
        typer.echo(f"💊 Drug: {drug_name}")  # NEW
        typer.echo(f"📂 Data path: {data_path}")

        # Create a prompt for the agent to use the tool
        typer.echo("\n🤖 Testing GSEA scores analysis with agent...")
        prompt = f"Use the analyze_gsea_scores tool to analyze the GSEA scores for cell '{cell_name}' and drug '{drug_name}'. Show me the top features and provide insights about the results."

        log, response, _, _ = agent.run(prompt)

        # Show the full log for debugging
        typer.echo("\n📝 Full execution log:")
        for i, (role, content) in enumerate(log):
            typer.echo(
                f"  {i+1}. [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        # Show the full log for debugging
        logger.info("[vision_scores] \n📝 Full execution log:")
        for i, (role, content) in enumerate(log):
            logger.info(
                f"[vision_scores]   {i+1}. [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        logger.info(f"[vision_scores] \n🎯 Final Agent Response: {response}")

    except Exception as e:
        typer.echo(f"❌ GSEA scores demo failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def basic_demo(
    provider: Annotated[str, typer.Option(help="AI provider")] = "lambda",
    model: Annotated[Optional[str], typer.Option(help="Model name")] = None,
    temperature: Annotated[float, typer.Option(help="Temperature")] = 0.7,
) -> None:
    """Basic GSEA scores demo with simplified prompt."""

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

    typer.echo(f"🔬 Basic GSEA Scores Demo - {provider.title()} ({model})")
    typer.echo("=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)  # type: ignore

        # Simple prompt to test the tool
        prompt = "Analyze GSEA scores for cell c_15 and drug TestDrug using differential scores from ~/sigspace2/sid/Datasets/tahoe"

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
    ] = "~/sigspace2/sid/Datasets/tahoe",
    cell_name: Annotated[
        Optional[str], typer.Option(help="Cell name to analyze")
    ] = None,
    drug_name: Annotated[str, typer.Option(help="Drug name to analyze")] = "Adagrasib",
    provider: Annotated[str, typer.Option(help="AI provider")] = "lambda",
    model: Annotated[
        Optional[str], typer.Option(help="Model name")
    ] = "deepseek-r1-671b",  # "hermes-3-llama-3.1-405b-fp8"
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
            source=source,
            temperature=0.7,
            tool_config={"drug_name": drug_name},  # Hidden from LLM
        )

        # Build prompt based on whether cell_name is provided
        base_prompt = "Use the analyze_gsea_scores tool to analyze the GSEA scores"
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


if __name__ == "__main__":
    app()
