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
from tahoe_agent.logging_config import get_logger

logger = get_logger()


app = typer.Typer(
    help="Tahoe Agent Vision Scores Demo - Analyze vision scores with different AI providers"
)


@app.command()
def signature_analysis(
    analysis_type: Annotated[
        str, typer.Option(help="Type of analysis to run: 'vision' or 'gsea'")
    ] = "vision",
    cell_name: Annotated[
        Optional[str], typer.Option(help="Cell name to analyze")
    ] = None,
    drug_name: Annotated[str, typer.Option(help="Drug name to analyze")] = "Adagrasib",
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
) -> None:
    """Signature analysis demonstration."""
    if custom_data_dir or custom_results_dir:
        logger.info("[signature_analysis] 🔧 Configuring custom paths...")
        config_kwargs = {}
        if custom_data_dir:
            config_kwargs["data_dir"] = custom_data_dir
        if custom_results_dir:
            config_kwargs["results_dir"] = custom_results_dir
        configure_paths(**config_kwargs)

        paths = get_paths()
        logger.info(f"[signature_analysis]   Data directory: {paths.data_dir}")
        logger.info(f"[signature_analysis]   Results directory: {paths.results_dir}")

    if model is None:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        elif provider == "lambda":
            model = "hermes-3-llama-3.1-405b-fp8"
        else:
            logger.error(f"[signature_analysis] Unknown provider: {provider}")
            raise typer.Exit(1)

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
    source = source_map.get(provider)

    logger.info(
        f"[signature_analysis] 🔬 Signature Analysis Demo - {provider.title()} ({model})"
    )
    logger.info("[signature_analysis] " + "=" * 50)

    try:
        agent = BaseAgent(
            llm=model,
            source=source,
            temperature=0.7,
            tool_config={"drug_name": drug_name},
        )

        logger.info(
            f"[signature_analysis] 📊 Analyzing signatures for drug: {drug_name}"
        )
        if cell_name:
            logger.info(f"[signature_analysis] 🧬 Cell: {cell_name}")

        if analysis_type == "vision":
            prompt = f"Use the analyze_vision_scores tool to analyze the vision scores for drug '{drug_name}'"
            if cell_name:
                prompt += f" in cell line '{cell_name}'"
            prompt += (
                ". Show me the top features and provide insights about the results."
            )
        elif analysis_type == "gsea":
            prompt = f"Use the analyze_gsea_scores tool to analyze the GSEA scores for drug '{drug_name}'"
            if cell_name:
                prompt += f" in cell line '{cell_name}'"
            prompt += (
                ". Show me the top enrichments and provide insights about the results."
            )
        else:
            logger.error(
                f"[signature_analysis] ❌ Invalid analysis type: {analysis_type}"
            )
            raise typer.Exit(1)

        logger.info(f"[signature_analysis] \n🤖 Sending prompt: {prompt}")

        log, response, _, _ = agent.run(prompt)

        logger.info("[signature_analysis] \n📝 Full execution log:")
        for i, (role, content) in enumerate(log):
            logger.info(
                f"[signature_analysis]   {i+1}. [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}"
            )

        logger.info(f"[signature_analysis] \n🎯 Final Agent Response: {response}")

    except Exception as e:
        logger.error(f"[signature_analysis] ❌ Signature analysis demo failed: {e}")
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
            logger.error(f"[basic_demo] Unknown provider: {provider}")
            raise typer.Exit(1)

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
    source = source_map.get(provider)

    logger.info(
        f"[basic_demo] 🔬 Basic Vision Scores Demo - {provider.title()} ({model})"
    )
    logger.info("[basic_demo] " + "=" * 50)

    try:
        agent = BaseAgent(llm=model, source=source, temperature=temperature)

        # Simple prompt to test the tool
        prompt = "Analyze vision scores for cell HS-578T and drug TestDrug using differential scores from /Users/gpalla/Datasets/tahoe"

        log, response, _, _ = agent.run(prompt)

        logger.info(f"[basic_demo] 💬 Question: {prompt}")
        logger.info(f"[basic_demo] 🎯 Agent: {response}")

    except Exception as e:
        logger.error(f"[basic_demo] ❌ Error: {e}")
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
        logger.info("[drug_ranking] 🔧 Configuring custom paths...")
        config_kwargs = {}
        if custom_data_dir:
            config_kwargs["data_dir"] = custom_data_dir
        if custom_results_dir:
            config_kwargs["results_dir"] = custom_results_dir
        configure_paths(**config_kwargs)
        logger.info(f"[drug_ranking]   Data directory: {get_paths().data_dir}")
        logger.info(f"[drug_ranking]   Results directory: {get_paths().results_dir}")

    # Set defaults based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        elif provider == "lambda":
            model = "hermes-3-llama-3.1-405b-fp8"
        else:
            logger.error(f"[drug_ranking] Unknown provider: {provider}")
            raise typer.Exit(1)

    source_map: Dict[str, SourceType] = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "lambda": "Lambda",
    }
    source = source_map.get(provider)

    logger.info(f"[drug_ranking] 💊 Drug Ranking Demo - {provider.title()} ({model})")
    logger.info("[drug_ranking] " + "=" * 50)

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

        logger.info(
            "[drug_ranking] 💬 Testing drug ranking with BLIND prompt (drug name hidden):"
        )
        logger.info(f"[drug_ranking]    Hidden drug: {drug_name}")
        logger.info(f"[drug_ranking]    Prompt: {prompt[:100]}...")
        logger.info("[drug_ranking] \n🤖 Running agent workflow...")

        log, response, structured_rankings, summary = agent.run(prompt)

        # Show execution log
        logger.info("[drug_ranking] \n📝 Execution log:")
        for i, (role, content) in enumerate(log):
            logger.info(
                f"[drug_ranking]   {i+1}. [{role}]: {content[:150]}{'...' if len(content) > 150 else ''}"
            )

        logger.info(
            f"[drug_ranking] \n🎯 Final Drug Rankings (Hidden drug was: {drug_name}):"
        )
        logger.info("[drug_ranking] " + "=" * 50)
        logger.info(f"[drug_ranking] {response}")

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
        logger.error(f"[drug_ranking] ❌ Drug ranking demo failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
