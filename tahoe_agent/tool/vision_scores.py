"""Vision scores analysis tool for Tahoe Agent."""

import pathlib

import anndata as ad  # type: ignore
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class VisionScoresArgs(BaseModel):
    """Schema for vision scores analysis arguments."""

    data_path: str = Field(
        description="Path to the directory containing h5ad files or direct path to h5ad file"
    )
    cell_name: str = Field(
        description="Name of the cell to analyze (from Cell_Name_Vevo column)"
    )
    use_diff_scores: bool = Field(
        default=True,
        description="Whether to use differential vision scores (True) or regular vision scores (False)",
    )


@tool(args_schema=VisionScoresArgs)
def analyze_vision_scores(
    data_path: str, cell_name: str, use_diff_scores: bool = True
) -> str:
    """Analyze vision scores from h5ad files and return top 10 vision scores for a specific cell.

    Args:
        data_path: Path to the directory containing h5ad files or direct path to h5ad file
        cell_name: Name of the cell to analyze (from Cell_Name_Vevo column)
        use_diff_scores: Whether to use differential vision scores (True) or regular vision scores (False)

    Returns:
        Formatted string with the top 10 vision scores and analysis results
    """
    if not data_path:
        return "Error: No data_path provided"
    if not cell_name:
        return "Error: No cell_name provided"

    try:
        # Check if data_path is a file or directory
        data_path_obj = pathlib.Path(data_path)

        if data_path_obj.is_file():
            # data_path is already pointing to a specific file
            file_path = data_path_obj
        else:
            # data_path is a directory, construct the file path
            vision_diff = "20250417.diff_vision_scores_pseudobulk.public.h5ad"
            vision_pseudo = "20250417.vision_scores_pseudobulk.public.h5ad"

            # Choose which file to use based on use_diff_scores parameter
            filename = vision_diff if use_diff_scores else vision_pseudo
            file_path = data_path_obj / filename

        if not file_path.exists():
            return f"Error: Vision scores file not found: {file_path}"

        # Load the h5ad file
        adata = ad.read_h5ad(file_path)

        # Check if cell_name exists in the data
        if "Cell_Name_Vevo" not in adata.obs.columns:
            return "Error: Cell_Name_Vevo column not found in the data"

        # Find the cell index
        cell_mask = adata.obs["Cell_Name_Vevo"] == cell_name
        if not cell_mask.any():
            available_cells = adata.obs["Cell_Name_Vevo"].unique()[
                :10
            ]  # Show first 10 for reference
            return f"Error: Cell '{cell_name}' not found. Available cells include: {list(available_cells)}"

        # Get the cell index (should be only one match)
        cell_idx = np.where(cell_mask)[0][0]

        # Get vision scores for this cell
        cell_scores = adata.X[cell_idx, :]

        # Get variable names (features)
        var_names = adata.var_names.tolist()

        # Handle sparse matrices if necessary
        if hasattr(cell_scores, "toarray"):
            cell_scores = cell_scores.toarray().flatten()
        elif hasattr(cell_scores, "flatten"):
            cell_scores = cell_scores.flatten()

        # Get top 10 indices
        top_indices = np.argsort(cell_scores)[-10:][::-1]  # Top 10 in descending order

        # Format results as a string
        analysis_type = (
            "differential_vision_scores" if use_diff_scores else "vision_scores"
        )

        result = f"""# Vision Scores Analysis Results

**Cell:** {cell_name}
**Analysis Type:** {analysis_type}
**Total Features:** {len(var_names)}
**Data Shape:** {adata.n_obs} cells × {adata.n_vars} features

## Top 10 Vision Scores:

"""

        for i, idx in enumerate(top_indices, 1):
            feature_name = var_names[idx]
            score = float(cell_scores[idx])
            result += f"{i:2d}. **{feature_name}**: {score:.6f}\n"

        # Add statistics
        result += f"""
## Statistics:
- **Mean Score:** {float(np.mean(cell_scores)):.6f}
- **Std Dev:** {float(np.std(cell_scores)):.6f}
- **Min Score:** {float(np.min(cell_scores)):.6f}
- **Max Score:** {float(np.max(cell_scores)):.6f}
- **Median Score:** {float(np.median(cell_scores)):.6f}

The vision scores represent the importance or activity level of different biological pathways and gene sets for this specific cell type.
"""

        return result

    except Exception as e:
        return f"Error: Vision scores analysis failed: {str(e)}"
