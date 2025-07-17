"""
example usage of the tahoe agent

python3 example.py summarize-vision-scores-one-drug --data-path {} --results-directory {} (one drug, no benchmark)
python3 example.py summarize-vision-scores-all-drugs --data-path {} --results-directory {} (all drugs, embedding benchmark)
python3 example.py rank-drugs-by-summary-one-drug --data-path {} --results-directory {} (one drug, no benchmark)
"""

from benchmark.analyze_embeddings import analyze_embeddings
import pandas as pd
from tahoe_agent import BaseAgent
from tahoe_agent.llm import SourceType
from tahoe_agent.logging_config import get_logger
from tahoe_agent.paths import configure_paths, get_paths
import typer
import typing
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer(help = "tahoe agent demo")
logger = get_logger()

@app.command()
def summarize_vision_scores_one_drug(
    data_path: Annotated[str, typer.Option(help = "path to directory containing data files")],
    results_directory: Annotated[Optional[str], typer.Option(help = "results directory")],
    cell_line: Annotated[Optional[str], typer.Option(help = "cell line to analyze")] = "HS-578T",
    drug_name: Annotated[str, typer.Option(help = "drug name to analyze")] = "Adagrasib",
    provider: Annotated[str, typer.Option(help = "provider")] = "Lambda",
    model: Annotated[Optional[str], typer.Option(help = "model name")] = "deepseek-r1-0528"
) -> None:

    logger.info("[summarize_vision_scores_one_drug] configuring custom paths:")
    logger.info(f"[summarize_vision_scores_one_drug] data directory: {data_path}")
    logger.info(f"[summarize_vision_scores_one_drug] results directory: {results_directory}")

    config_kwargs = {}
    config_kwargs["data_dir"] = data_path
    config_kwargs["results_dir"] = results_directory
    configure_paths(**config_kwargs)

    results_directory = get_paths().results_dir / f"{provider.title()}_{model}_summaries" # type: ignore
    results_directory.mkdir(parents = True, exist_ok = True) # type: ignore

    logger.info(f"[summarize_vision_scores_one_drug] single drug demonstration of vision score summary for {provider.title()} ({model})")

    try:
        agent = BaseAgent(
            llm = model, # type: ignore
            source = typing.cast(SourceType, provider),
            temperature = 0.7,
            tool_config = {"drug_name": drug_name}
        )

        logger.info(f"[summarize_vision_scores_one_drug] analyzing vision scores for:")
        logger.info(f"[summarize_vision_scores_one_drug] cell line: {cell_line}")
        logger.info(f"[summarize_vision_scores_one_drug] drug: {drug_name}")

        prompt = f"use the analyze_vision_scores tool to analyze the vision scores for cell '{cell_line}' and drug '{drug_name}'"

        _, response = agent.run(prompt)

        file = results_directory / f"{drug_name}_summary.txt" # type: ignore
        with open(file, "w") as f:
            f.write(response)
            
        logger.info(f"[summarize_vision_scores_one_drug] saved model summary to: {file}")

    except Exception as e:
        logger.error(f"demonstration failed: {e}")
        raise typer.Exit(1)

@app.command()
def summarize_vision_scores_all_drugs(
    data_path: Annotated[str, typer.Option(help = "path to directory containing data files")],
    results_directory: Annotated[Optional[str], typer.Option(help = "results directory")],
    cell_line: Annotated[Optional[str], typer.Option(help = "cell line to analyze")] = "HS-578T",
    provider: Annotated[str, typer.Option(help = "provider")] = "Lambda",
    model: Annotated[Optional[str], typer.Option(help = "model name")] = "deepseek-r1-0528"
) -> None:
    
    logger.info("[summarize_vision_scores_all_drugs] configuring custom paths:")
    logger.info(f"[summarize_vision_scores_all_drugs] data directory: {data_path}")
    logger.info(f"[summarize_vision_scores_all_drugs] results directory: {results_directory}")

    config_kwargs = {}
    config_kwargs["data_dir"] = data_path
    config_kwargs["results_dir"] = results_directory
    configure_paths(**config_kwargs)

    logger.info(f"[summarize_vision_scores_all_drugs] multiple drug demonstration of vision score summary for {provider.title()} ({model})")

    drugs = pd.read_csv(get_paths().drugs_file)["drug"].tolist()
    for drug_name in drugs[:5]:
        try:
            summarize_vision_scores_one_drug(data_path = data_path, results_directory = results_directory, cell_line = cell_line,
                                                 drug_name = drug_name, provider = provider, model = model)
        
        except Exception as e:
            logger.error(f"[summarize_vision_scores_all_drugs] error for {drug_name}: {e}")
    
    analyze_embeddings(get_paths().results_dir / f"{provider.title()}_{model}_summaries", get_paths().drugs_file)

@app.command()
def rank_drugs_by_summary_one_drug(
    data_path: Annotated[str, typer.Option(help = "path to directory containing data files")],
    results_directory: Annotated[Optional[str], typer.Option(help = "results directory")],
    cell_line: Annotated[Optional[str], typer.Option(help = "cell line to analyze")] = "HS-578T",
    drug_name: Annotated[str, typer.Option(help = "drug name to analyze")] = "Adagrasib",
    provider: Annotated[str, typer.Option(help = "provider")] = "Lambda",
    model: Annotated[Optional[str], typer.Option(help = "model name")] = "deepseek-r1-0528"
) -> None:

    logger.info("[rank_drugs_by_summary] configuring custom paths:")
    logger.info(f"[rank_drugs_by_summary] data directory: {data_path}")
    logger.info(f"[rank_drugs_by_summary] results directory: {results_directory}")

    config_kwargs = {}
    config_kwargs["data_dir"] = data_path
    config_kwargs["results_dir"] = results_directory
    configure_paths(**config_kwargs)

    results_directory = get_paths().results_dir / f"{provider.title()}_{model}_summaries" # type: ignore
    results_directory.mkdir(parents = True, exist_ok = True) # type: ignore

    logger.info(f"[rank_drugs_by_summary_one_drug] single drug demonstration of model rankings for {provider.title()} ({model})")

    try:
        agent = BaseAgent(
            llm = model, # type: ignore
            source = typing.cast(SourceType, provider),
            temperature = 0.7,
            tool_config = {"drug_name": drug_name}
        )

        logger.info(f"[rank_drugs_by_summary_one_drug] analyzing vision scores for:")
        logger.info(f"[rank_drugs_by_summary_one_drug] cell line: {cell_line}")
        logger.info(f"[rank_drugs_by_summary_one_drug] drug: {drug_name}")

        base_prompt = f"use the analyze_vision_scores tool to analyze the vision scores for cell '{cell_line}'"
        prompt = (
            f"{base_prompt}; "
            "show me the top features and provide insights about the results; "
            "after the analysis, when the summary is available, rank drugs based on how well their mechanisms of action match the observed biological signatures"
        )

        _, response = agent.run(prompt)

        file = results_directory / f"ranking_{drug_name}.txt" # type: ignore
        with open(file, "w") as f:
            f.write(response)
            
        logger.info(f"[rank_drugs_by_summary_one_drug] saved model rankings to: {file}")

    except Exception as e:
        logger.error(f"demonstration failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()