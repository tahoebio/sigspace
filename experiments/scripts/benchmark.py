import hydra
from omegaconf import DictConfig
from tahoe_agent.agent.base_agent import BaseAgent
from tahoe_agent.paths import configure_paths, get_paths
from tahoe_agent.logging_config import get_logger

logger = get_logger()


@hydra.main(
    version_base=None, config_path="../config", config_name="summary_comparison"
)
def main(cfg: DictConfig) -> None:
    """Run summary comparison experiment with different models and temperatures."""

    # Configure paths if provided
    if cfg.custom_data_dir or cfg.custom_results_dir:
        logger.info("🔧 Configuring custom paths...")
        config_kwargs = {}
        if cfg.custom_data_dir:
            config_kwargs["data_dir"] = cfg.custom_data_dir
        if cfg.custom_results_dir:
            config_kwargs["results_dir"] = cfg.custom_results_dir
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


if __name__ == "__main__":
    main()
