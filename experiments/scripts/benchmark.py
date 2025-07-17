import hydra
from omegaconf import DictConfig
from tahoe_agent.agent.base_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths
from tahoe_agent.logging_config import get_logger
import pandas as pd
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

logger = get_logger()


@hydra.main(
    version_base=None, config_path="../config", config_name="summary_comparison"
)
def main(cfg: DictConfig) -> None:
    """Run summary comparison experiment with different models and temperatures."""
    logger.info("🔧 Configuring custom paths...")
    config_kwargs = {}
    if cfg.custom_data_dir:
        config_kwargs["data_dir"] = cfg.custom_data_dir
    if cfg.custom_results_dir:
        config_kwargs["results_dir"] = Path(cfg.custom_results_dir) / cfg.model
    configure_paths(**config_kwargs)

    paths = get_paths()
    logger.info(f"  Data directory: {paths.data_dir}")
    logger.info(f"  Results directory: {paths.results_dir}")

    agent = BaseAgent(
        llm=cfg.model,
        source=cfg.source,
        temperature=cfg.temperature,
        tool_config={"drug_name": cfg.drug_name},  # Hidden from LLM
    )

    base_prompt = "Use the analyze_vision_scores tool to analyze the vision scores"
    if cfg.cell_name is not None:
        base_prompt += f" for cell line '{cfg.cell_name}'"

    prompt = f"""{base_prompt}.

    Show me the top features and provide insights about the results.

    After the analysis, when the summary is available, rank drugs based on how well their mechanisms of action match the observed biological signatures.
    """

    logger.info("🤖 Running agent workflow...")
    log, response, structured_rankings, summary = agent.run(prompt)

    logger.info(f"\n🎯 Final Drug Rankings (Hidden drug was: {cfg.drug_name}):")
    logger.info("=" * 50)
    logger.info(response)
    logger.info("\nSUMMARY\n")
    logger.info(summary)
    logger.info("\nSTRUCTURED RANKINGS\n")
    logger.info(structured_rankings)

    # Save summary
    os.makedirs(Path(paths.results_dir), exist_ok=True)
    summary_path = paths.get_results_file(
        f"summary_{cfg.drug_name}_{cfg.cell_name}.txt"
    )
    logger.info(f"\n💾 Saving summary to {summary_path}...")
    with open(summary_path, "w") as f:
        f.write(summary)

    # Convert structured rankings to DataFrame and save as CSV
    if structured_rankings:
        rankings_path = paths.get_results_file(
            f"drugrank_{cfg.drug_name}_{cfg.cell_name}.csv"
        )
        logger.info(f"💾 Saving drug rankings to {rankings_path}...")

        # Convert list of DrugRanking objects to DataFrame
        rankings_data = {
            "drug": [r.drug for r in structured_rankings],
            "score": [r.score for r in structured_rankings],
        }
        rankings_df = pd.DataFrame(rankings_data)

        # Save to CSV
        rankings_df.to_csv(rankings_path, index=False)
        logger.info("✅ Results saved successfully")

    # create embeddings for the summary
    model = SentenceTransformer("all-MiniLM-L6-v2")
    summary_embedding = model.encode(summary)
    embedding_path = paths.get_results_file(
        f"embedding_{cfg.drug_name}_{cfg.cell_name}.npz"
    )
    logger.info(f"💾 Saving summary embedding to {embedding_path}...")
    np.savez_compressed(embedding_path, embedding=summary_embedding)


if __name__ == "__main__":
    main()
