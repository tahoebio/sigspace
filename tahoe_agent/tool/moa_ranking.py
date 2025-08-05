"""MOA ranking tool for Tahoe Agent."""

import json
from langchain_core.tools import tool
from tahoe_agent.paths import get_paths
import pandas as pd


@tool
def get_moa_list() -> str:
    """
    Provides a list of mechanisms of action (MOAs) that were observed in single-cell cancer data following drug perturbation.
    This list is intended for ranking based on mechanism of action similarity to the observed biological signatures.

    Returns:
        A string containing an introduction and a comma-separated list of MOAs.
    """
    try:
        drugs_df = pd.read_csv(get_paths().drugs_file)

        fine_moas = drugs_df["moa-fine"].dropna().unique().tolist()
        fine_moas = [
            moa for moa in fine_moas if moa and moa.lower() != "unclear" and moa.strip()
        ]
        fine_moas.sort()

    except Exception as e:
        return json.dumps({"error": f"Failed to load MOA list: {str(e)}"})

    intro = "The following mechanisms of action (MOAs) were observed in single-cell cancer data following drug perturbation.\nThis list will be used for ranking based on MOA similarity to the observed biological signatures."
    moa_list_str = ", ".join(fine_moas)

    return f"{intro}\n\n**MOA LIST:**\n{moa_list_str}"
