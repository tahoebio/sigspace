"""Tests for Tahoe Agent tools."""

import pathlib
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd  # type: ignore

from tahoe_agent.tool.vision_scores import analyze_vision_scores
from tahoe_agent._constants import VisionScoreColumns


def test_analyze_vision_scores() -> None:
    """Test analyze_vision_scores tool with and without cell_name."""
    # Create mock AnnData object
    mock_adata = MagicMock()

    # Mock observations (metadata) - using the exact column names the function expects
    mock_obs = pd.DataFrame(
        {
            VisionScoreColumns.DRUG: ["Adagrasib"] * 10,
            VisionScoreColumns.CELL_NAME: ["HS-578T"] * 5 + ["HT-29"] * 5,
            VisionScoreColumns.CONCENTRATION: [5.0] * 10,
        }
    )
    mock_adata.obs = mock_obs
    mock_adata.obs.columns = mock_obs.columns

    # Mock variable names (features) - needs to support array indexing
    num_features = 5000
    mock_var_names = np.array([f"Gene_{i}" for i in range(num_features)])
    mock_adata.var_names = mock_var_names

    # Mock expression data - must match number of features
    np.random.seed(42)
    mock_adata.X = np.random.randn(10, num_features)

    # Use the actual VisionScoreColumns enum
    with patch(
        "tahoe_agent.tool.vision_scores.ad.read_h5ad", return_value=mock_adata
    ), patch("tahoe_agent.tool.vision_scores.get_paths") as mock_get_paths:
        # Mock paths
        mock_paths = MagicMock()
        mock_paths.vision_diff_file = pathlib.Path("test_vision.h5ad")
        mock_get_paths.return_value = mock_paths

        # Mock the path exists check
        with patch("pathlib.Path.exists", return_value=True):
            # Test 1: Drug only (cell_name=None)
            result1 = analyze_vision_scores.invoke(
                input={
                    "drug_name": "Adagrasib",
                    "cell_name": None,
                }
            )

            # Test 2: Drug + cell
            result2 = analyze_vision_scores.invoke(
                input={
                    "drug_name": "Adagrasib",
                    "cell_name": "HS-578T",
                }
            )

    # Print results
    print("\n" + "=" * 80)
    print("RESULT 1 - Drug only (cell_name=None):")
    print("=" * 80)
    print(result1)

    print("\n" + "=" * 80)
    print("RESULT 2 - Drug + cell (cell_name='HS-578T'):")
    print("=" * 80)
    print(result2)
    print("=" * 80)

    # # Basic assertions
    # assert "Vision Scores Analysis Results (Across All Cell Lines)" in result1
    # assert "Vision Scores Analysis Results (Specific Cell Line)" in result2
    # assert "Adagrasib" in result1
    # assert "Adagrasib" in result2
    # assert "HS-578T" in result2
    # assert "Top 10 Signatures" in result1
    # assert "Top 10 Signatures" in result2
