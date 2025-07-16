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
    concentration: float = 5.0,
    cell_name: Optional[str] = None,
) -> Dict[str, Any]:

    print ("Here is the drug name: ", drug_name)
    """Analyze drug signatures for a specific drug and concentration, with or without cell line.

    Args:
        adata: AnnData object with vision scores in layers['scores']
        drug_name: Name of the drug to analyze
        cell_name: Name of the cell to analyze. If None, analyze across all cell lines.
        concentration: Drug concentration to filter on (default: 5.0)

    Returns:
        Dictionary containing top_250, bottom_250 signatures, drug_name, and optionally cell_name
    """
    # Get the scores matrix from layers
    scores_matrix = adata.layers['scores']
    
    # Initialize variables
    all_signatures: np.ndarray
    scores: np.ndarray
    total_analyzed: int

    if cell_name == 'None':
        cell_name = None

    if cell_name is None:
        print("Analyzing across all cell lines")
        mask = (adata.obs['drug'] == drug_name) & (
            adata.obs['concentration'] == concentration
        )

        if not mask.any():
            available_drugs = adata.obs['drug'].unique()[:10]
            available_concs = adata.obs['concentration'].unique()
            raise ValueError(
                f"No data found for drug '{drug_name}' at concentration {concentration}. "
                f"Available drugs: {list(available_drugs)}, "
                f"Available concentrations: {list(available_concs)}"
            )

        # Convert mask to numpy array
        mask_np = mask.to_numpy()

        # Index the scores matrix with numpy mask
        data = scores_matrix[mask_np, :]

        if hasattr(data, "toarray"):
            data = data.toarray()

        # Median across all cell lines per signature
        scores = np.median(data, axis=0)
        all_signatures = adata.var_names
        total_analyzed = data.shape[0]
    else:
        # Case 2: Analyze specific drug-cell combination
        mask = (
            (adata.obs['drug'] == drug_name)
            & (adata.obs['cell_line'] == cell_name)  # Note: cell_line not cell_name
            & (adata.obs['concentration'] == concentration)
        )

        if not mask.any():
            available_cells = adata.obs[
                adata.obs['drug'] == drug_name
            ]['cell_line'].unique()[:10]
            available_concs = adata.obs[
                adata.obs['drug'] == drug_name
            ]['concentration'].unique()
            raise ValueError(
                f"No data found for drug '{drug_name}', cell '{cell_name}' at concentration {concentration}. "
                f"Available cells for this drug: {list(available_cells)}, "
                f"Available concentrations: {list(available_concs)}"
            )

        # Get the specific data point(s) and convert to dense array if needed
        data = scores_matrix[mask, :]
        if hasattr(data, "toarray"):
            data = data.toarray()

        # Create signature names array matching the flattened data shape
        # If we have multiple replicates, repeat gene names for each replicate
        all_signatures = np.repeat(adata.var_names, data.shape[0])

        # Since we filtered by cell line, drug and concentration,
        # we can just flatten the array to get 1D scores
        scores = data.flatten()
        total_analyzed = mask.sum()

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
    # Input validation - drug_name should be injected by the agent
    if not drug_name or not isinstance(drug_name, str):
        return "Error: Drug name must be configured in the agent system for blind evaluation"

    try:
        # Get path configuration and load data
        paths = get_paths()
        

        # Use default vision scores file
        file_path = paths.vision_diff_file

        if not file_path.exists():
            return f"Error: Vision scores file not found: {file_path}"

        # Load data and check required columns
        adata = ad.read_h5ad(file_path)
        required_cols = [VisionScoreColumns.DRUG, VisionScoreColumns.CONCENTRATION]
        if cell_name is not None:
            required_cols.append(VisionScoreColumns.CELL_NAME)

        missing_cols = [col for col in required_cols if col not in adata.obs.columns]
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}"

        # Analyze signatures
        concentration = 5.0
        signatures = analyze_signatures(adata, drug_name, concentration, cell_name)

        # Create header based on whether cell_name is provided
        if cell_name is None:
            header = f"""# Vision Scores Analysis Results (Across All Cell Lines)

**Concentration:** {signatures['concentration']}
**Data File:** {file_path.name}
**Total Cell Lines Analyzed:** {signatures['total_analyzed']}"""
        else:
            header = f"""# Vision Scores Analysis Results (Specific Cell Line)

**Cell Line:** {cell_name}
**Concentration:** {signatures['concentration']}
**Data File:** {file_path.name}
**Data Points Used:** {signatures['total_analyzed']}"""

        # Add signatures (same format for both cases)
        result = header + "\n\n## Top 10 Signatures (Highest Absolute Scores):\n\n"

        for i, item in enumerate(signatures["top_250"][:10], 1):
            result += f"{i:2d}. **{item['feature']}**: {item['score']:.6f}\n"

        result += "\n## Bottom 10 Signatures (Lowest Absolute Scores):\n\n"

        for i, item in enumerate(signatures["bottom_250"][:10], 1):
            result += f"{i:2d}. **{item['feature']}**: {item['score']:.6f}\n"

        # Add summary (same for both cases)
        if cell_name is None:
            analysis_context = "mean scores across all cell lines for this drug"
        else:
            analysis_context = "specific drug-cell line combination"

        result += f"""
## Analysis Summary:
- **Total signatures analyzed:** {len(adata.var_names)}
- **Top 250 signatures identified** (showing top 10 above)
- **Bottom 250 signatures identified** (showing bottom 10 above)
- **Analysis based on {analysis_context}**

The vision scores represent the importance or activity level of different biological pathways and gene sets for this drug{' across all tested cell lines' if cell_name is None else ' in this specific cell line'}.
"""

        return result

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: Vision scores analysis failed: {str(e)}"
