# tahoe agent

SigSpace: an LLM-based agent for drug response signature interpretation


### installation

```bash
git clone https://github.com/tahoebio/sigspace.git
cd sigspace
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

### Notes
Inspired by the [biomni framework](https://github.com/snap-stanford/Biomni), built with lang-chain and lang-graph

### Citations
@article {Khurana2025.12.02.691945,
	author = {Khurana, Rohit and Mangla, Ishita and Palla, Giovanni and Sanghi, Siddhant and Merico, Daniele},
	title = {SigSpace: an LLM-based agent for drug response signature interpretation},
	elocation-id = {2025.12.02.691945},
	year = {2025},
	doi = {10.64898/2025.12.02.691945},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/05/2025.12.02.691945},
	eprint = {https://www.biorxiv.org/content/early/2025/12/05/2025.12.02.691945.full.pdf},
	journal = {bioRxiv}
}
