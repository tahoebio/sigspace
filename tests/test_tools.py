"""Tests for Tahoe Agent tools."""

import pathlib
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd  # type: ignore
import json
from pprint import pprint
from tahoe_agent.tool.vision_scores import analyze_vision_scores
from tahoe_agent.tool.gsea_scores import analyze_gsea_scores
from tahoe_agent.tool.drug_ranking import rank_drugs_by_moa
from tahoe_agent._constants import VisionScoreColumns
from tahoe_agent._constants import GSEAScoreColumns

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

    # Basic assertions
    assert "Vision Scores Analysis Results (Across All Cell Lines)" in result1
    assert "Vision Scores Analysis Results (Specific Cell Line)" in result2
    assert "Adagrasib" in result1
    assert "Adagrasib" in result2
    assert "HS-578T" in result2
    assert "Top 10 Signatures" in result1
    assert "Top 10 Signatures" in result2


def test_analyze_gsea_scores() -> None:
    """Test analyze_gsea_scores tool with and without cell_name."""
    # Create mock AnnData object
    mock_adata = MagicMock()

    # Mock observations (metadata) - using the exact column names the function expects
    mock_obs = pd.DataFrame(
        {
            GSEAScoreColumns.DRUG: ["Adagrasib"] * 10,
            GSEAScoreColumns.CELL_LINE: ["c_15"] * 5 + ["c_40"] * 5,
            GSEAScoreColumns.CONCENTRATION: [5.0] * 10,
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

    # Use the actual GSEAScoreColumns enum
    with patch(
        "tahoe_agent.tool.gsea_scores.ad.read_h5ad", return_value=mock_adata
    ), patch("tahoe_agent.tool.gsea_scores.get_paths") as mock_get_paths:
        # Mock paths
        mock_paths = MagicMock()
        mock_paths.gsea_scores_file = pathlib.Path("test_gsea.h5ad")
        mock_get_paths.return_value = mock_paths

        # Mock the path exists check
        with patch("pathlib.Path.exists", return_value=True):
            # Test 1: Drug only (cell_name=None)
            result1 = analyze_gsea_scores.invoke(
                input={
                    "drug_name": "Adagrasib",
                    "cell_name": None,
                }
            )

            # Test 2: Drug + cell
            result2 = analyze_gsea_scores.invoke(
                input={
                    "drug_name": "Adagrasib",
                    "cell_name": "c_15",
                }
            )

    # Print results
    print("\n" + "=" * 80)
    print("RESULT 1 - Drug only (cell_name=None):")
    print("=" * 80)
    print(result1)

    print("\n" + "=" * 80)
    print("RESULT 2 - Drug + cell (cell_name='c_15'):")
    print("=" * 80)
    print(result2)
    print("=" * 80)

    # Basic assertions
    assert "GSEA Scores Analysis Results (Across All Cell Lines)" in result1
    assert "GSEA Scores Analysis Results (Specific Cell Line)" in result2
    assert "Adagrasib" in result1
    assert "Adagrasib" in result2
    assert "c_15" in result2
    assert "Top 10 Signatures" in result1
    assert "Top 10 Signatures" in result2

def test_rank_drugs_by_moa() -> None:
    """Test rank_drugs_by_moa tool functionality."""

    summary = "DNA damage response pathways are significantly altered, with increased apoptosis markers."

    # Mock the drug list file
    mock_drugs_df = pd.DataFrame(
        {"drug_name": ["Cisplatin", "Paclitaxel", "Methotrexate"]}
    )

    with patch(
        "tahoe_agent.tool.drug_ranking.pd.read_csv", return_value=mock_drugs_df
    ), patch("tahoe_agent.tool.drug_ranking.get_paths") as mock_get_paths:
        # Mock paths
        mock_paths = MagicMock()
        mock_paths.drugs_file = pathlib.Path("test_drugs.csv")
        mock_get_paths.return_value = mock_paths

        # Test valid input
        result = rank_drugs_by_moa.invoke({"summary": summary})
        parsed = json.loads(result)

        pprint(parsed, indent=2)

        assert "drug_list" in parsed
        assert parsed["drug_list"] == ["Cisplatin", "Paclitaxel", "Methotrexate"]
        assert parsed["summary"] == summary
        assert "ranking_instructions" in parsed
        assert parsed["total_drugs"] == 3
        assert parsed["status"] == "ready_for_ranking"

        # Check that instructions contain key elements
        instructions = parsed["ranking_instructions"]
        assert "mechanism of action" in instructions.lower()
        assert "moa" in instructions.lower()
        assert "json" in instructions.lower()

        # Test empty summary
        result_empty_summary = rank_drugs_by_moa.invoke({"summary": ""})
        parsed_empty_summary = json.loads(result_empty_summary)
        assert "error" in parsed_empty_summary

    # Test file read error
    with patch(
        "tahoe_agent.tool.drug_ranking.pd.read_csv",
        side_effect=Exception("File not found"),
    ):
        result_file_error = rank_drugs_by_moa.invoke({"summary": summary})
        parsed_file_error = json.loads(result_file_error)
        assert "error" in parsed_file_error
        assert "Failed to load drug list" in parsed_file_error["error"]
