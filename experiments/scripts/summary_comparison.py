import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(
    version_base=None, config_path="../config", config_name="summary_comparison"
)
def main(cfg: DictConfig) -> None:
    """Run summary comparison experiment with different models and temperatures."""

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Test prompt
    prompt = (
        "What are the key differences between supervised and unsupervised learning?"
    )

    print(cfg)
    print(prompt)
    # # Run experiment for each config
    # results = []
    # for model in cfg.model:
    #     for temp in cfg.temperature:
    #         # Initialize agent with current config
    #         agent = BaseAgent(
    #             llm=model, temperature=temp, source=cfg.source, base_url=cfg.base_url
    #         )

    #         # Get response
    #         response = agent.chat(prompt)

    #         # Store result
    #         result = {"model": model, "temperature": temp, "response": response}
    #         results.append(result)

    # # Save results
    # output_file = results_dir / "summary_comparison_results.json"
    # with open(output_file, "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
