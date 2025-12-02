#!/usr/bin/env python3
"""
Tahoe GSEA k-NN MOA ranking (plain vote counts) + MRR — no top/bottom-N, no name cleaning,
with imputation for unranked MOAs in the final metrics.

What this does
--------------
1) For each drug (at a chosen dose), aggregate replicates: mean NES per gene set.
2) Keep gene sets:
     • If --use-fraction-gate: keep sets where ≥ min_fraction of rows have padj < p and the SAME NES sign
       (computed per drug, per set); otherwise keep ALL sets.
3) Build binary matrices using ALL kept sets (no top/bottom-N):
     • M_up[d, g]   = 1 if kept and aggregated NES > 0
     • M_down[d, g] = 1 if kept and aggregated NES < 0
4) Distances: Jaccard on M_up and M_down, averaged (diag = +inf).
5) Map drugs → MOA directly from CSV (exact strings; drop 'unclear'/NA only).
6) For each drug, rank MOAs by plain vote counts among its k nearest neighbors (score = count / k);
   write one CSV per drug to --out-dir/moarank/.
7) Print MRR, MRR@1/5/10, Hits@1/5/10 and random baselines.
   (Imputes ranks for MOAs not appearing in the k-NN ranked list.)

Outputs:
  - One CSV per drug in --out-dir/moarank:
      moarank_{cellToken}_{drugToken}_rankings.csv
    Columns: drug, moa, score, rank  (score = count/k)

  - One summary CSV per k:
      moarank_summary_{cellToken}_k{k}.csv
    Columns: drug, true_moa, true_rank  (true_rank is imputed when the true MOA isn't in the list)

Usage
-----
python knn_moa_ranking_gsea_v3_correct.py \
    --gsea-h5ad path/to/gsea_scores.h5ad \
    --drugs-csv data/drugs.csv \
    --out-dir ./rankings \
    --k 10 15 \
    [--drug-conc-col DRUG_CONC] \
    [--cell-col CELL_LINE] \
    [--concentration 5.0] \
    [--auto-highest] \
    [--use-fraction-gate] \
    [--min-fraction 0.25] \
    [--pval-threshold 0.05] \
    [--show-unmapped 10]
"""

from __future__ import annotations
import argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple
import os
import re
import csv

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances

# -------------------------- Simple helpers --------------------------

UNCLEAR = {"unclear"}

def token_for_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s

def simple_drug_to_moa(csv_path: str) -> Tuple[Dict[str, str], List[str]]:
    """Exact-string drug → moa-fine mapping; drop NA/empty/'unclear'; keep first occurrence."""
    df = pd.read_csv(csv_path, usecols=["drug", "moa-fine"])
    df["drug"] = df["drug"].astype(str).str.strip()
    df["moa-fine"] = df["moa-fine"].astype(str).str.strip()
    df = df[~df["moa-fine"].str.lower().isin(UNCLEAR)]
    df = df.drop_duplicates(subset="drug", keep="first")
    drug_to_moa = dict(zip(df["drug"], df["moa-fine"]))
    moas_sorted = sorted(df["moa-fine"].unique().tolist())
    return drug_to_moa, moas_sorted

def parse_drug_and_conc(s: str) -> Tuple[str, float]:
    m = re.match(r"^(.*)_(\d*\.?\d+)uM$", s, flags=re.IGNORECASE)
    if not m:
        return s, float("nan")
    return m.group(1), float(m.group(2))

def _to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else arr

# -------------------------- Build M_up / M_down (ALL kept sets) --------------------------

def build_binary_matrices_from_gsea(
    adata: ad.AnnData,
    *,
    drug_conc_col: str = "drug_conc",
    cell_col: Optional[str] = None,
    cell_id_value: Optional[str] = None,
    concentration: Optional[float] = None,
    auto_highest: bool = False,
    use_fraction_gate: bool = False,
    min_fraction: float = 0.25,
    pval_threshold: float = 0.05,
    nes_layer: str = "gsea_nes",
    padj_layer: str = "gsea_padj",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Build M_up/M_down (drugs x gene_sets) from GSEA NES.
    **Skips** a drug entirely if: (a) no rows at chosen dose/cell, or
    (b) fraction gate is ON and no gene set passes the gate.
    Returns:
        M_up, M_down, drugs_kept, gene_sets, skipped_drugs
    """
    # --- fetch & basic setup ---
    if nes_layer not in adata.layers or padj_layer not in adata.layers:
        raise ValueError("Expected GSEA layers 'gsea_nes' and 'gsea_padj' in adata.layers")
    nes_all  = adata.layers[nes_layer].toarray()  if hasattr(adata.layers[nes_layer], "toarray")  else adata.layers[nes_layer]
    padj_all = adata.layers[padj_layer].toarray() if hasattr(adata.layers[padj_layer], "toarray") else adata.layers[padj_layer]

    if drug_conc_col not in adata.obs.columns:
        raise ValueError(f"adata.obs must contain '{drug_conc_col}'")
    drug_conc_vals = adata.obs[drug_conc_col].astype(str).values

    cell_vals = None
    if cell_col:
        if cell_col not in adata.obs.columns:
            raise ValueError(f"adata.obs missing '{cell_col}'")
        cell_vals = adata.obs[cell_col].astype(str).values

    # Parse "Drug_5uM" → (drug, 5.0)
    parsed      = [parse_drug_and_conc(s) for s in drug_conc_vals]
    base_drugs  = np.array([p[0] for p in parsed], dtype=object)
    conc_vals   = np.array([p[1] for p in parsed], dtype=float)

    all_drugs   = sorted(list({str(x) for x in base_drugs}))
    gene_sets   = list(map(str, adata.var_names))
    n_gs        = len(gene_sets)

    # We'll build rows only for drugs we keep
    rows_up:   List[np.ndarray] = []
    rows_down: List[np.ndarray] = []
    drugs_kept: List[str] = []
    skipped: List[str] = []

    for d in all_drugs:
        mask_d  = (base_drugs == d)
        if not np.any(mask_d):
            skipped.append(d); continue

        # choose concentration
        concs_d = conc_vals[mask_d]
        if concentration is not None:
            chosen_conc = float(concentration)
        elif auto_highest:
            chosen_conc = float(np.nanmax(concs_d))
        else:
            vals, cnts  = np.unique(concs_d[~np.isnan(concs_d)], return_counts=True)
            chosen_conc = float(vals[np.argmax(cnts)]) if len(vals) else float("nan")

        # rows for (drug, conc[, cell])
        rows = mask_d & np.isclose(conc_vals, chosen_conc, rtol=0, atol=1e-6)
        if cell_vals is not None and cell_id_value is not None:
            rows &= (cell_vals == str(cell_id_value))
        if not np.any(rows):
            skipped.append(d); continue

        nes, padj = nes_all[rows, :], padj_all[rows, :]

        # ---------- fraction gate ----------
        if use_fraction_gate:
            pos_mask  = (padj < pval_threshold) & (nes > 0)
            neg_mask  = (padj < pval_threshold) & (nes < 0)
            pos_frac  = pos_mask.sum(axis=0) / pos_mask.shape[0]
            neg_frac  = neg_mask.sum(axis=0) / neg_mask.shape[0]
            keep_gs   = (pos_frac >= min_fraction) | (neg_frac >= min_fraction)
            if not np.any(keep_gs):
                skipped.append(d); continue
            nes = nes[:, keep_gs]
            kept_idx = np.where(keep_gs)[0]
        else:
            kept_idx = np.arange(nes.shape[1])

        # ---------- aggregation (mean NES) ----------
        per_gs_nes = np.mean(nes, axis=0)  # mean across rows/replicates

        # Build binary row over ALL gene sets; only mark kept_idx by sign
        up_row   = np.zeros(n_gs, dtype=np.uint8)
        down_row = np.zeros(n_gs, dtype=np.uint8)
        up_row[kept_idx[per_gs_nes > 0]]   = 1
        down_row[kept_idx[per_gs_nes < 0]] = 1

        rows_up.append(up_row)
        rows_down.append(down_row)
        drugs_kept.append(d)

    if not drugs_kept:
        raise ValueError("No drugs remained after gating/row selection.")

    M_up   = np.vstack(rows_up)
    M_down = np.vstack(rows_down)
    return M_up, M_down, drugs_kept, gene_sets, skipped


# -------------------------- Distances & Ranking --------------------------

def averaged_jaccard_distance(M_up: np.ndarray, M_down: np.ndarray) -> np.ndarray:
    U = M_up.astype(bool)
    D = M_down.astype(bool)
    Du = pairwise_distances(U, metric="jaccard")
    Dd = pairwise_distances(D, metric="jaccard")
    Davg = (Du + Dd) / 2.0
    np.fill_diagonal(Davg, np.inf)
    return Davg

def rank_moas_counts(D_row: np.ndarray, labels: List[str], k: int) -> List[Tuple[str, float]]:
    nn_idx = np.argsort(D_row)[:k]
    nn_labels = [labels[j] for j in nn_idx]
    counts = Counter(nn_labels)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kf = float(k) if k > 0 else 1.0
    return [(moa, c / kf) for moa, c in ranked]

# -------------------------- Metrics helpers (with imputation) --------------------------

def metrics_from_ranks(ranks: List[float], ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute MRR/MRR@K/Hits@K from a list of numeric ranks (1-based).
    Convention: rank <= 0 or NaN contributes 0.
    """
    r = np.asarray(ranks, dtype=float)
    valid = np.isfinite(r) & (r > 0)
    inv = np.zeros_like(r, dtype=float)
    inv[valid] = 1.0 / r[valid]

    out = {"MRR": float(inv[valid].mean()) if valid.any() else float("nan")}
    for K in ks:
        at_k = valid & (r <= K)
        out[f"MRR@{K}"] = float(inv[at_k].mean()) if at_k.any() else 0.0
        out[f"Hits@{K}"] = float(at_k.mean() * 100.0) if valid.any() else 0.0
    return out

def rank_with_impute(ranked_moas: List[str], true_moa: str, unique_moas: List[str]) -> float:
    """
    If true_moa is present, return its 1-based rank.
    Otherwise impute as mean of the unranked positions (L+1..M), where
    L = len(ranked_moas), M = len(unique_moas). If L==M, return 0 (no credit).
    """
    if true_moa in ranked_moas:
        return float(ranked_moas.index(true_moa) + 1)

    L = len(ranked_moas)
    M = len(unique_moas)
    if L < M:
        # mean of consecutive integers L+1..M = (L+1 + M) / 2
        return float((L + 1 + M) / 2.0)
    else:
        # fully ranked but true_moa absent (shouldn't happen with counts) → 0 credit
        return 0.0

def random_baselines_moa(num_classes: int, ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Expected random baselines when the true class is uniformly among M classes.
    E[MRR]   = H_M / M
    E[MRR@K] = H_K / M
    E[Hits@K]= K / M
    """
    def H(n: int) -> float:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=float)))
    M = num_classes
    out = {"MRR": H(M)/M}
    for K in ks:
        out[f"MRR@{K}"] = H(min(K, M))/M
        out[f"Hits@{K}"] = (min(K, M)/M)*100.0
    return out

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="GSEA k-NN MOA ranking (ALL kept sets; no name cleaning) with imputation")
    ap.add_argument("--gsea-h5ad", required=True, help="Path to GSEA scores .h5ad (gsea_nes/gsea_padj)")
    ap.add_argument("--drugs-csv", required=True, help="Path to data/drugs.csv (columns: drug, moa-fine)")
    ap.add_argument("--out-dir", required=True, help="Directory to write moarank CSVs")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 15], help="k values for k-NN")
    ap.add_argument("--drug-conc-col", default="drug_conc", help="Obs column with 'Drug_5uM' strings")
    ap.add_argument("--cell-col", default=None, help="Optional obs cell column")
    ap.add_argument("--cell-id", default=None, help="Optional cell id/name filter")
    ap.add_argument("--concentration", type=float, default=None, help="Fixed concentration in µM")
    ap.add_argument("--auto-highest", action="store_true", help="Use highest available dose when --concentration not set")
    ap.add_argument("--use-fraction-gate", action="store_true", help="Require min fraction with padj<thr and same NES sign")
    ap.add_argument("--min-fraction", type=float, default=0.25, help="Min fraction for the gate")
    ap.add_argument("--pval-threshold", type=float, default=0.05, help="padj threshold for the gate")
    ap.add_argument("--show-unmapped", type=int, default=10, help="How many unmapped drug names to print")
    args = ap.parse_args()

    # Prepare output dir
    moarank_dir = os.path.join(args.out_dir, "moarank")
    os.makedirs(moarank_dir, exist_ok=True)

    # Load AnnData and build matrices (ALL kept sets)
    adata = ad.read_h5ad(args.gsea_h5ad)
    M_up, M_down, drugs_kept, gene_sets, skipped = build_binary_matrices_from_gsea(
        adata=adata,
        drug_conc_col=args.drug_conc_col,
        cell_col=args.cell_col,
        cell_id_value=args.cell_id,
        concentration=args.concentration,
        auto_highest=args.auto_highest,
        use_fraction_gate=args.use_fraction_gate,
        min_fraction=args.min_fraction,
        pval_threshold=args.pval_threshold,
        nes_layer="gsea_nes",
        padj_layer="gsea_padj",
    )

    # Direct drug → MOA mapping (exact strings)
    drug_to_moa, fine_moas = simple_drug_to_moa(args.drugs_csv)

    if skipped:
        print(f"[info] Skipped {len(skipped)} drugs with no rows at the chosen dose/cell or no gene sets passing the gate.")
        print("       e.g.", skipped[:10])

    # Keep only drugs with a mapping
    keep_mask = [d in drug_to_moa for d in drugs_kept]
    unmapped = [d for d, keep in zip(drugs_kept, keep_mask) if not keep]
    if unmapped:
        print(f"[warn] {len(unmapped)} drugs had no MOA mapping; first few:", unmapped[:args.show_unmapped])

    # Filter to mapped drugs
    M_up   = M_up[keep_mask]
    M_down = M_down[keep_mask]
    drugs_final = [d for d, k in zip(drugs_kept, keep_mask) if k]
    labels      = [drug_to_moa[d] for d in drugs_final]

    # Distances
    D_avg = averaged_jaccard_distance(M_up, M_down)

    # Unique MOAs for baselines / imputation
    unique_moas = sorted(list(set(labels)))

    cell_token = token_for_filename(args.cell_id) if (args.cell_col and args.cell_id) else "allcells"

    # For each k: rankings + imputed ranks + metrics
    for k in args.k:
        print(f"\n=== k = {k} ===")

        # Numeric ranks for every mapped drug (observed or imputed)
        true_ranks_all: List[float] = []

        # Also write a per-k summary
        summary_rows = []

        for i, (drug_name_raw, true_moa) in enumerate(zip(drugs_final, labels)):
            # Get ranked MOAs via k-NN counts
            scores = rank_moas_counts(D_avg[i], labels, k=k)   # [(moa, score)]
            ranked_moas = [m for m, _ in scores]

            # Impute the true MOA's rank if absent
            true_rank = rank_with_impute(ranked_moas, true_moa, unique_moas)
            true_ranks_all.append(true_rank)
            summary_rows.append({"drug": drug_name_raw, "true_moa": true_moa, "true_rank": f"{true_rank:.6f}"})

            # Per-drug CSV of the full ranked MOA list (by count/k)
            drug_token = token_for_filename(drug_name_raw)
            out_path = os.path.join(moarank_dir, f"moarank_{cell_token}_{drug_token}_rankings.csv")
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["drug", "moa", "score", "rank"])
                for r, (moa, sc) in enumerate(scores, start=1):
                    w.writerow([drug_name_raw, moa, f"{sc:.6f}", r])

        # Write per-k summary of true ranks (observed + imputed)
        summary_path = os.path.join(moarank_dir, f"moarank_summary_{cell_token}_k{k}.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["drug", "true_moa", "true_rank"])
            w.writeheader()
            w.writerows(summary_rows)

        # Metrics from numeric ranks (include imputations)
        metrics = metrics_from_ranks(true_ranks_all, ks=[1, 5, 10])
        baselines = random_baselines_moa(num_classes=len(unique_moas), ks=[1, 5, 10])

        print(f"MRR:   {metrics['MRR']:.3f}   (exp {baselines['MRR']:.3f})")
        print(f"MRR@1: {metrics['MRR@1']:.3f} (exp {baselines['MRR@1']:.3f}) | "
              f"Hits@1: {metrics['Hits@1']:.1f}% (exp {baselines['Hits@1']:.1f}%)")
        print(f"MRR@5: {metrics['MRR@5']:.3f} (exp {baselines['MRR@5']:.3f}) | "
              f"Hits@5: {metrics['Hits@5']:.1f}% (exp {baselines['Hits@5']:.1f}%)")
        print(f"MRR@10:{metrics['MRR@10']:.3f} (exp {baselines['MRR@10']:.3f}) | "
              f"Hits@10:{metrics['Hits@10']:.1f}% (exp {baselines['Hits@10']:.1f}%)")

    print(f"\nWrote MOA rankings and summaries to: {moarank_dir}")

if __name__ == "__main__":
    main()

