# tahoe agent

a minimal agent implementation inspired by the [biomni framework](https://github.com/snap-stanford/Biomni), built with lang-chain and lang-graph

### installation

```bash
pip install -e ".[dev]"
```

### benchmark

`tahoe_agent/experiments/config/summary_comparison.yaml` is the main configuration file for running benchmarking experiments with the agent; it is used by hydra to manage experiment parameters and allows for flexible, reproducible runs

- **model_source_map**: maps logical names to their actual model identifiers and providers; add new configurations here

- **temperature**: controls the randomness / creativity of model outputs (higher values make model outputs more random and creative; lower values make them more focused and deterministic)

- **cell_name**: specifies the cell line to analyze; set to a cell line name or null to analyze across all cell lines

- **custom_data_dir**: specifies path to input data directory

- **custom_results_dir**: specifies path to output directory

- **hydra.sweeper.params**: defines parameter sweeps for running multiple experiments (cartesian product of input parameters):
  - **model_source_pair**: selects which model and provider combination to use based off of the model source mapping
  - **tool**: specifies the analysis tool to use (either `analyze_vision_scores` or `analyze_gsea_scores`)
  - **drug_name**: list of drug(s) to process of form `choice("drug_A", "drug_B", "drug_C", ...)`
