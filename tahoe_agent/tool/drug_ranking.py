"""Drug ranking tool for Tahoe Agent."""

import json
from langchain_core.tools import tool
from tahoe_agent.paths import get_paths
import pandas as pd
from tahoe_agent._constants import VisionScoreColumns


@tool
def get_drug_list() -> str:
    """
    Provides a list of drugs that were tested on single-cell cancer data following perturbation.
    This list is intended for ranking based on mechanism of action similarity.

    Returns:
        A string containing an introduction and a comma-separated list of drugs.
    """
    try:
        drugs_df = pd.read_csv(get_paths().drugs_file)
        drug_list = drugs_df[VisionScoreColumns.DRUG].unique().tolist()
    except Exception as e:
        return json.dumps({"error": f"Failed to load drug list: {str(e)}"})

    intro = "The following drugs were tested for their effects on single-cell cancer data following perturbation.\nThis list will be used for ranking based on mechanism of action similarity."
    drug_list_str = ", ".join(drug_list)

    return f"{intro}\n\n**DRUG LIST:**\n{drug_list_str}"
