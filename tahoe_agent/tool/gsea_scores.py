""""GSEA scores analysis tool for Tahoe Agent."""

from typing import Optional, Dict, Any

import anndata as ad  # type: ignore
import numpy as np
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tahoe_agent.paths import get_paths
from tahoe_agent._constants import GSEAScoreColumns


# ───────────────────── Pydantic schema for tool args ────────────────────────
class GSEAScoresArgs(BaseModel):
    """Schema for GSEA score analysis arguments."""

    cell_name: Optional[str] = Field(
        default=None,
        description="Name of the cell line to analyse (omit to aggregate across all lines).",
    )
    concentration: Optional[float] = Field(
        default=None,
        description="Drug concentration in µM; if omitted, the highest available dose is used.",
    )
    min_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "For aggregated mode: fraction of lines that must show padj < 0.05 "
            "in the SAME NES direction."
        ),
    )


# ───────────────────────── Helper functions ─────────────────────────────────
def _format_conc(val: float) -> str:
    return f"{val:.1f}" if val >= 1 else f"{val:.2f}"


def _extract_conc(label: str) -> float:
    m = re.search(r"_(\d*\.?\d+)uM", label)
    return float(m.group(1)) if m else np.nan


# ───────────────────────── Core analysis helper ─────────────────────────────
def analyze_gsea_signatures(
    adata: ad.AnnData,
    drug_name: str,
    *,
    cell_name: Optional[str] = None,
    concentration: Optional[float] = None,
    min_fraction: float = 0.25,
    pval_threshold: float = 0.05,
    score_layer: str = "gsea_nes",
    padj_layer: str = "gsea_padj",
) -> Dict[str, Any]:
    """Analyse NES & padj for a given drug/dose, returning positive and negative tables."""
    obs = adata.obs

    # Choose dose (highest if unspecified)
    if concentration is None:
        mask = obs[GSEAScoreColumns.DRUG_CONC].str.startswith(f"{drug_name}_")
        if not mask.any():
            raise ValueError(f"Drug '{drug_name}' not found.")
        concentration = float(
            np.nanmax(obs.loc[mask, GSEAScoreColumns.DRUG_CONC].apply(_extract_conc))
        )
    conc_str = _format_conc(concentration)

    # Slice matrices
    nes  = adata.layers[score_layer]
    padj = adata.layers[padj_layer]
    if hasattr(nes,  "toarray"): nes  = nes.toarray()
    if hasattr(padj, "toarray"): padj = padj.toarray()

    var_names = np.asarray(adata.var_names)
    rows = obs[GSEAScoreColumns.DRUG_CONC].str.match(
        rf"^{re.escape(drug_name)}_{conc_str}0*uM$"
    )
    if not rows.any():
        raise ValueError(f"No samples for {drug_name} at {concentration} µM.")

    nes, padj, obs = nes[rows], padj[rows], obs.iloc[rows.values]

    # Cell-specific mode
    if cell_name:
        m = obs[GSEAScoreColumns.CELL_LINE] == cell_name
        if not m.any():
            raise ValueError(
                f"Cell line '{cell_name}' absent for this drug/dose."
            )
        nes_vec  = nes[m].mean(axis=0)
        padj_vec = padj[m].mean(axis=0)
        sig_table = [
            {"gene_set": gs, "nes": float(n), "padj": float(p)}
            for gs, n, p in zip(var_names, nes_vec, padj_vec)
        ]
    # Aggregated mode
    else:
        pos = ((padj < pval_threshold) & (nes > 0)).sum(axis=0) / nes.shape[0]
        neg = ((padj < pval_threshold) & (nes < 0)).sum(axis=0) / nes.shape[0]
        keep = (pos >= min_fraction) | (neg >= min_fraction)
        if not keep.any():
            raise ValueError(
                "No gene sets meet padj & direction-consistency criteria."
            )
        nes_mean    = nes[:, keep].mean(axis=0)
        padj_median = np.median(padj[:, keep], axis=0)
        sig_table = [
            {
                "gene_set": var_names[idx],
                "nes": float(nes_mean[j]),
                "padj": float(padj_median[j]),
            }
            for j, idx in enumerate(np.where(keep)[0])
        ]

    # Split into positive / negative lists
    pos_signatures = sorted(
        (r for r in sig_table if r["nes"] > 0), key=lambda d: -d["nes"]
    )
    neg_signatures = sorted(
        (r for r in sig_table if r["nes"] < 0), key=lambda d: d["nes"]
    )

    return {
        "drug_name": drug_name,
        "cell_name": cell_name,
        "concentration": concentration,
        "total_signatures": len(sig_table),
        "pos_signatures": pos_signatures,
        "neg_signatures": neg_signatures,
    }


# ─────────────────────────── Markdown wrapper ───────────────────────────────
@tool(args_schema=GSEAScoresArgs)
def analyze_gsea_scores(
    *,
    cell_name: Optional[str] = None,
    concentration: Optional[float] = None,
    min_fraction: float = 0.25,
    drug_name: Optional[str] = None,  # hidden param injected by agent
) -> str:
    """Analyse GSEA scores for a drug (optionally a specific cell line)."""
    if not drug_name or not isinstance(drug_name, str):
        return "Error: Drug name must be configured in the agent system."

    paths = get_paths()
    file_path = paths.data_dir / "gsea_all_sparse.h5ad"
    if not file_path.exists():
        return f"Error: GSEA file not found: {file_path}"

    try:
        adata = ad.read_h5ad(file_path)
        required = [GSEAScoreColumns.DRUG_CONC] + (
            [GSEAScoreColumns.CELL_LINE] if cell_name else []
        )
        if missing := [c for c in required if c not in adata.obs.columns]:
            return f"Error: Missing required columns: {missing}"

        res = analyze_gsea_signatures(
            adata,
            drug_name,
            cell_name=cell_name,
            concentration=concentration,
            min_fraction=min_fraction,
        )

        # Build output parts
        out: list[str] = []
        if cell_name is None:
            out.append("# GSEA Analysis Results (Across All Cell Lines)\n")
        else:
            out.append("# GSEA Analysis Results (Specific Cell Line)\n")
            out.append(f"**Cell Line:** {cell_name}\n")
        out.append(f"**Data File:** {file_path.name}\n")

        out.append("\n## Top Enrichments - UP-regulated (highest positive NES):\n")
        for i, row in enumerate(res["pos_signatures"], 1):
            out.append(
                f"{i:2d}. **{row['gene_set']}** — NES {row['nes']:.3f}; "
                f"padj {row['padj']:.2e}\n"
            )

        out.append("\n## Top Enrichments - DOWN-regulated (most negative NES):\n")
        for i, row in enumerate(res["neg_signatures"], 1):
            out.append(
                f"{i:2d}. **{row['gene_set']}** — NES {row['nes']:.3f}; "
                f"padj {row['padj']:.2e}\n"
            )

        context = (
            "aggregated statistics across retained lines"
            if cell_name is None
            else "mean of replicates for this line"
        )
        out.append(
            f"""
## Analysis Summary:
- **Total gene sets analysed:** {len(adata.var_names)}
- **Reported gene sets:** {res['total_signatures']}
- Results reflect {context}.
"""
        )
        return "".join(out)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: GSEA score analysis failed: {e}"
