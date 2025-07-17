"""Vision scores analysis tool for Tahoe Agent."""

from typing import Optional, Dict, Any

import anndata as ad  # type: ignore
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tahoe_agent.paths import get_paths
from tahoe_agent._constants import VisionScoreColumns


class VisionScoresArgs(BaseModel):
    """Schema for vision scores analysis arguments."""

    cell_name: Optional[str] = Field(
        default=None,
        description=f"Name of the cell to analyze (from {VisionScoreColumns.CELL_NAME} column). If None, analyze across all cell lines.",
    )


def analyze_signatures(
    adata: ad.AnnData,
    drug_name: str,
    cell_name: Optional[str] = None,
    concentration: float = 5.0,
) -> Dict[str, Any]:
    """Analyze drug signatures for a specific drug and concentration, with or without cell line.

    Args:
        adata: AnnData object with vision scores
        drug_name: Name of the drug to analyze
        cell_name: Name of the cell to analyze. If None, analyze across all cell lines.
        concentration: Drug concentration to filter on (default: 5.0)

    Returns:
        Dictionary containing top_250, bottom_250 signatures, drug_name, and optionally cell_name
    """
    # Initialize variables
    all_signatures: np.ndarray
    scores: np.ndarray
    total_analyzed: int

    if cell_name is None:
        # Case 1: Analyze across all cell lines
        mask = (adata.obs[VisionScoreColumns.DRUG] == drug_name) & (
            adata.obs[VisionScoreColumns.CONCENTRATION] == concentration
        )

        if not mask.any():
            available_drugs = adata.obs[VisionScoreColumns.DRUG].unique()[:10]
            available_concs = adata.obs[VisionScoreColumns.CONCENTRATION].unique()
            raise ValueError(
                f"No data found for drug '{drug_name}' at concentration {concentration}. "
                f"Available drugs: {list(available_drugs)}, "
                f"Available concentrations: {list(available_concs)}"
            )

        # Get data for this drug at specified concentration across all cell lines
        data = adata.X[mask, :]

        # Convert to dense array if sparse
        if hasattr(data, "toarray"):
            data = data.toarray()

        # Create repeated signature names array matching the data shape
        all_signatures = np.repeat(adata.var_names, data.shape[0])

        # Flatten all vision scores across cell lines into a 1D array
        scores = data.flatten()
        total_analyzed = len(scores)

    else:
        # Case 2: Analyze specific drug-cell combination
        mask = (
            (adata.obs[VisionScoreColumns.DRUG] == drug_name)
            & (adata.obs[VisionScoreColumns.CELL_NAME] == cell_name)
            & (adata.obs[VisionScoreColumns.CONCENTRATION] == concentration)
        )

        if not mask.any():
            available_cells = adata.obs[
                adata.obs[VisionScoreColumns.DRUG] == drug_name
            ][VisionScoreColumns.CELL_NAME].unique()[:10]
            available_concs = adata.obs[
                adata.obs[VisionScoreColumns.DRUG] == drug_name
            ][VisionScoreColumns.CONCENTRATION].unique()
            raise ValueError(
                f"No data found for drug '{drug_name}', cell '{cell_name}' at concentration {concentration}. "
                f"Available cells for this drug: {list(available_cells)}, "
                f"Available concentrations: {list(available_concs)}"
            )

        # Get the specific data point(s) and convert to dense array if needed
        data = adata.X[mask, :]
        if hasattr(data, "toarray"):
            data = data.toarray()

        # Create signature names array matching the flattened data shape
        # If we have multiple replicates, repeat gene names for each replicate
        all_signatures = np.repeat(adata.var_names, data.shape[0])

        # Since we filtered by cell line, drug and concentration,
        # we can just flatten the array to get 1D scores
        scores = data.flatten()
        total_analyzed = sum(mask)

        assert scores.ndim == all_signatures.ndim

    # Get indices for top/bottom 250 scores and corresponding signatures
    score_indices = np.argsort(scores)
    bottom_250_indices = score_indices[:250]  # Bottom 250 in ascending order
    top_250_indices = score_indices[-250:][::-1]  # Top 250 in descending order

    # Get corresponding signatures and scores in sorted order
    top_250_signatures = all_signatures[top_250_indices]
    bottom_250_signatures = all_signatures[bottom_250_indices]

    # Create result dictionary
    result = {
        "drug_name": drug_name,
        "cell_name": cell_name,
        "concentration": concentration,
        "total_analyzed": total_analyzed,
        "top_250": [
            {
                "feature": top_250_signatures[i],
                "score": float(scores[top_250_indices[i]]),
            }
            for i in range(len(top_250_indices))
        ],
        "bottom_250": [
            {
                "feature": bottom_250_signatures[i],
                "score": float(scores[bottom_250_indices[i]]),
            }
            for i in range(len(bottom_250_indices))
        ],
    }

    return result


@tool(args_schema=VisionScoresArgs)
def analyze_vision_scores(
    cell_name: Optional[str] = None,
    drug_name: Optional[str] = None,  # Hidden parameter injected by agent
) -> str:
    """Analyze vision scores for a specific drug and optionally a specific cell line.

    This tool analyzes biological signatures for a drug. The drug name is automatically
    provided by the system and should not be specified by the user.

    Args:
        cell_name: Name of the cell to analyze (from {VisionScoreColumns.CELL_NAME} column). If None, analyze across all cell lines.
        drug_name: [HIDDEN] Drug name injected by the agent system

    Returns:
        Formatted string with analysis results or error message
    """
    if not drug_name or not isinstance(drug_name, str):
        return "Error: Drug name must be configured in the agent system for blind evaluation"

    try:
        paths = get_paths()
        file_path = paths.vision_diff_file
        if not file_path.exists():
            return f"Error: Vision scores file not found: {file_path}"

        adata = ad.read_h5ad(file_path)
        required_cols = [VisionScoreColumns.DRUG, VisionScoreColumns.CONCENTRATION]
        if cell_name:
            required_cols.append(VisionScoreColumns.CELL_NAME)

        missing_cols = [col for col in required_cols if col not in adata.obs.columns]
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}"

        signatures = analyze_signatures(adata, drug_name, cell_name, concentration=5.0)

        # Use a list to build the string parts for better performance and readability
        output_parts = []

        if cell_name is None:
            output_parts.append(
                "# Vision Scores Analysis Results (Across All Cell Lines)\n"
            )
            output_parts.append(f"**Data File:** {file_path.name}\n")
        else:
            output_parts.append(
                "# Vision Scores Analysis Results (Specific Cell Line)\n"
            )
            output_parts.append(f"**Cell Line:** {cell_name}\n")
            output_parts.append(f"**Data File:** {file_path.name}\n")
<<<<<<< HEAD

        output_parts.append("\n## Top 250 Signatures (Highest Scores):\n")
        for i, item in enumerate(signatures["top_250"], 1):
            output_parts.append(f"{i}. {item['feature']}: {item['score']:.6f}\n")

        output_parts.append("\n## Bottom 250 Signatures (Lowest Scores):\n")
        for i, item in enumerate(signatures["bottom_250"], 1):
            output_parts.append(f"{i}. {item['feature']}: {item['score']:.6f}\n")

=======

        output_parts.append("\n## Top 10 Signatures (Highest Absolute Scores):\n")
        for i, item in enumerate(signatures["top_250"][:10], 1):
            output_parts.append(f"{i:2d}. **{item['feature']}**: {item['score']:.6f}\n")

        output_parts.append("\n## Bottom 10 Signatures (Lowest Absolute Scores):\n")
        for i, item in enumerate(signatures["bottom_250"][:10], 1):
            output_parts.append(f"{i:2d}. **{item['feature']}**: {item['score']:.6f}\n")

>>>>>>> Sid01123/structure-output
        if cell_name is None:
            analysis_context = "mean scores across all cell lines for this drug"
        else:
            analysis_context = "specific drug-cell line combination"
        summary_text = f"""
## Analysis Summary:
- **Total signatures analyzed:** {len(adata.var_names)}
- **Top 250 signatures identified**
- **Bottom 250 signatures identified**
- **Analysis based on {analysis_context}**

The vision scores represent the importance or activity level of different biological pathways and gene sets for this drug{' across all tested cell lines' if cell_name is None else ' in this specific cell line'}.
"""
        output_parts.append(summary_text)

        return "".join(output_parts)

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: Vision scores analysis failed: {str(e)}"
