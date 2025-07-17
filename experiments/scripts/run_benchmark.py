import hydra
import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tahoe_agent.agent.base_agent import BaseAgent
from tahoe_agent.logging_config import get_logger
from tahoe_agent.paths import configure_paths, get_paths

logger = get_logger()

@hydra.main(version_base = None, config_path = "../config", config_name = "summary_comparison")
def main(cfg: DictConfig) -> None:
    logger.info("🔧 configuring custom paths...")
    config_kwargs = {}
    if cfg.custom_data_dir:
        config_kwargs["data_dir"] = cfg.custom_data_dir
    if cfg.custom_results_dir:
        config_kwargs["results_dir"] = Path(cfg.custom_results_dir) / cfg.model
    configure_paths(**config_kwargs)
    logger.info("[main] starting benchmark run...")

    paths = get_paths()
    logger.info(f"  data directory: {paths.data_dir}")
    logger.info(f"  results directory: {paths.results_dir}")

    agent = BaseAgent(
        llm = cfg.model,
        source = cfg.source,
        temperature = cfg.temperature,
        tool_config = {"drug_name": cfg.drug_name},  # Hidden from LLM
    )

    base_prompt = f"Use the {cfg.tool} tool to analyze the gene signatures"
    if cfg.cell_name is not None:
        base_prompt += f" for cell line '{cfg.cell_name}'"

    prompt = f"""{base_prompt}.

    Show me the top features and provide insights about the results.
    After the analysis, when the summary is available, rank drugs based on how well their mechanisms of action match the observed biological signatures.
    """

    logger.info("🤖 running agent workflow...")
    log, response, structured_rankings, summary = agent.run(prompt)

    logger.info(f"\n🎯 final drug rankings (hidden drug was: {cfg.drug_name}):")
    logger.info("=" * 50)
    logger.info(response)
    logger.info("\nsummary\n")
    logger.info(summary)
    logger.info("\nstructured rankings\n")
    logger.info(structured_rankings)

    # Save summary
    os.makedirs(Path(paths.results_dir), exist_ok = True)
    summary_path = paths.get_results_file(
        f"summary_{cfg.drug_name}_{cfg.cell_name}.txt"
    )
    logger.info(f"\n💾 saving summary to {summary_path}...")
    with open(summary_path, "w") as f:
        f.write(summary)

    # Convert structured rankings to DataFrame and save as CSV
    if structured_rankings:
        rankings_path = paths.get_results_file(
            f"drugrank_{cfg.drug_name}_{cfg.cell_name}.csv"
        )
        logger.info(f"💾 saving drug rankings to {rankings_path}...")

        # Convert list of DrugRanking objects to DataFrame
        rankings_data = {
            "drug": [r.drug for r in structured_rankings],
            "score": [r.score for r in structured_rankings],
        }
        rankings_df = pd.DataFrame(rankings_data)

        # Save to CSV
        rankings_df.to_csv(rankings_path, index = False)
        logger.info("✅ results saved successfully")

    # create embeddings for the summary
    model = SentenceTransformer("all-MiniLM-L6-v2")
    summary_embedding = model.encode(summary)
    embedding_path = paths.get_results_file(
        f"embedding_{cfg.drug_name}_{cfg.cell_name}.npz"
    )
    logger.info(f"💾 saving summary embedding to {embedding_path}...")
    np.savez_compressed(embedding_path, embedding = summary_embedding)


if __name__ == "__main__":
    main()
