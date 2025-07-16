"""Drug ranking tool for Tahoe Agent."""

import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tahoe_agent.paths import get_paths
import pandas as pd
from tahoe_agent._constants import VisionScoreColumns


class DrugRankingArgs(BaseModel):
    """Schema for drug ranking arguments."""

    summary: str = Field(
        description="Summary from the vision scores analysis to rank drugs against"
    )


@tool(args_schema=DrugRankingArgs)
def rank_drugs_by_moa(summary: str) -> str:
    """Rank drugs based on their mechanism of action (MoA) similarity to a biological summary.

    You are a pharmacology expert specializing in drug mechanisms of action (MoA) analysis.
    Given the summary from the previous LLM and the drug list, based on your understanding of MoA,
    rank each drug as most likely to be the one of which the MoA is described in the summary.

    Args:
        summary: Biological summary to compare against (from vision scores analysis)

    Returns:
        Structured prompt for the LLM to analyze drug mechanisms of action
    """
    if not summary or not isinstance(summary, str):
        return json.dumps({"error": "Summary must be a non-empty string"})

    # Load drug list from file
    try:
        drugs_df = pd.read_csv(get_paths().drugs_file)
        drug_list = drugs_df[VisionScoreColumns.DRUG].unique().tolist()
    except Exception as e:
        return json.dumps({"error": f"Failed to load drug list: {str(e)}"})

    # Prepare the structured prompt for the LLM to use
    ranking_instructions = (
        "Based on the following biological summary and drug list, rank each drug by how closely its mechanism of action (MoA) matches the biological processes described in the summary.\n\n"
        "**SUMMARY FROM VISION SCORES ANALYSIS:**\n"
        f"{summary}\n\n"
        "**DRUG LIST TO RANK:**\n"
        f"{', '.join(drug_list)}\n\n"
        "**RANKING CRITERIA:**\n"
        "- Analyze each drug's mechanism of action (MoA)\n"
        "- Compare how each drug's MoA relates to the biological processes in the summary\n"
        "- Assign relevance scores from 0.0 to 1.0 (1.0 = most relevant MoA match)\n"
        "- Focus on mechanistic similarities, not just therapeutic area\n\n"
    )
    return ranking_instructions
