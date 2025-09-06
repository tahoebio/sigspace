#!/usr/bin/env python3
"""
Tahoe Vision k-NN MOA ranking (plain vote counts) + MRR, using a SIMPLE drug→MOA map.

Outputs:
  - One CSV per drug in --out-dir/moarank:
      moarank_{cellToken}_{drugToken}_rankings.csv
    Columns: drug, moa, score, rank  (score = count/k)

Also prints MOA ranking metrics:
  - MRR, MRR@1/5/10, Hits@1/5/10, plus random baselines.

Usage:
  python tahoe_vision_knn_moarank.py \
      --vision-h5ad path/to/vision_diff_scores.h5ad \
      --drugs-csv data/drugs.csv \
      --out-dir ./rankings \
      --concentration 5.0 \
      --top-n 250 \
      --k 10 15 \
      [--drug-col drug] [--conc-col concentration] [--cell-col cell_name] [--cell-id CELL123] \
      [--seed 0] [--show-unmapped 10]
"""

from __future__ import annotations
import argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple
import os
import csv

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances

# -------------------------- Simple drug → MOA mapping --------------------------

UNCLEAR = {"unclear", "unknown", "na", "n/a", "none", "unspecified", "other", ""}

def simple_drug_to_moa(csv_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    SIMPLE mapping:
      - read 'drug' and 'moa-fine'
      - strip strings
      - drop rows with unclear/empty MOA
      - keep the first occurrence per drug
    Returns:
      drug_to_moa: {drug -> moa-fine}
      fine_moas: sorted list of unique moa-fine values kept
    """
    df = pd.read_csv(csv_path, usecols=["drug", "moa-fine"])
    df["drug"] = df["drug"].astype(str).str.strip()
    df["moa-fine"] = df["moa-fine"].astype(str).str.strip()
    df = df[~df["moa-fine"].str.lower().isin(UNCLEAR)]
    df = df.drop_duplicates(subset="drug", keep="first")
    drug_to_moa = dict(zip(df["drug"], df["moa-fine"]))
    fine_moas = sorted(df["moa-fine"].unique().tolist())
    return drug_to_moa, fine_moas

def token_for_filename(s: str) -> str:
    """Safe token for filenames (keep alnum, dot, underscore, dash)."""
    s = (s or "").strip().replace("/", "-").replace("\\", "-")
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "-", s)

# -------------------------- Binary matrices from Vision scores --------------------------

def _to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else arr

def build_binary_matrices_from_vision(
    adata: ad.AnnData,
    concentration: float = 5.0,
    top_n: int = 250,
    drug_col: str = "drug",
    conc_col: str = "concentration",
    cell_col: Optional[str] = None,
    cell_id_value: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build M_up and M_down (drugs x gene_sets) from adata.layers['scores'].
    Per drug: aggregate by median across rows matching (drug, conc[, cell]).
    Then set top_n (highest) to 1 in M_up and bottom_n (lowest) to 1 in M_down.
    """
    if "scores" not in adata.layers:
        raise ValueError("Expected Vision scores in adata.layers['scores']")
    scores_mat = _to_dense(adata.layers["scores"])

    if drug_col not in adata.obs.columns or conc_col not in adata.obs.columns:
        raise ValueError(f"adata.obs must contain '{drug_col}' and '{conc_col}'")
    drug_obs = adata.obs[drug_col].astype(str).values
    conc_obs = adata.obs[conc_col].values
    cell_obs = adata.obs[cell_col].astype(str).values if (cell_col and cell_col in adata.obs.columns) else None

    gene_sets = list(map(str, adata.var_names))
    n_gs = len(gene_sets)
    n_top = min(top_n, n_gs)

    drugs_raw = sorted(list({str(x) for x in drug_obs}))
    M_up = np.zeros((len(drugs_raw), n_gs), dtype=np.uint8)
    M_down = np.zeros_like(M_up)
    drug_to_idx = {d: i for i, d in enumerate(drugs_raw)}

    for d in drugs_raw:
        mask = (drug_obs == d) & (conc_obs == concentration)
        if cell_obs is not None and cell_id_value is not None:
            mask &= (cell_obs == str(cell_id_value))
        if not np.any(mask):
            continue

        data = scores_mat[mask, :]  # (n_rows, n_gene_sets)
        per_gs = np.median(data, axis=0)

        order = np.argsort(per_gs)           # low -> high
        bottom_idx = order[:n_top]
        top_idx = order[-n_top:][::-1]

        di = drug_to_idx[d]
        M_up[di, top_idx] = 1
        M_down[di, bottom_idx] = 1

    return M_up, M_down, drugs_raw, gene_sets

# -------------------------- Distances & k-NN --------------------------

def averaged_jaccard_distance(M_up: np.ndarray, M_down: np.ndarray) -> np.ndarray:
    """Jaccard distances for up/down, averaged. Diagonal set to +inf to exclude self."""
    U = M_up.astype(bool)
    D = M_down.astype(bool)
    Du = pairwise_distances(U, metric="jaccard")
    Dd = pairwise_distances(D, metric="jaccard")
    Davg = (Du + Dd) / 2.0
    np.fill_diagonal(Davg, np.inf)
    return Davg

# -------------------------- Ranking MOAs (plain counts) --------------------------

def rank_moas_counts(D_row: np.ndarray, labels: List[str], k: int) -> List[Tuple[str, float]]:
    """
    For a single drug (one row of the distance matrix):
      - take k nearest neighbors (self excluded via +inf on diagonal)
      - count neighbors per MOA
      - return list of (moa, score=count/k), sorted by score desc then moa name
    """
    nn_idx = np.argsort(D_row)[:k]
    nn_labels = [labels[j] for j in nn_idx]
    counts = Counter(nn_labels)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kf = float(k) if k > 0 else 1.0
    return [(moa, c / kf) for moa, c in ranked]

# -------------------------- Metrics (MRR etc.) --------------------------

def mrr_and_hits_from_ranklists(true_labels: List[str],
                                ranked_lists: List[List[str]],
                                ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    true_labels[i] is the ground-truth MOA for query i.
    ranked_lists[i] is the list of MOAs in ranked order for query i.
    Returns MRR, MRR@K, Hits@K.
    """
    def recip(rank: int) -> float:
        return 0.0 if rank <= 0 else 1.0 / float(rank)

    N = len(true_labels)
    ranks = []
    for t, rlist in zip(true_labels, ranked_lists):
        try:
            r = rlist.index(t) + 1  # 1-based
        except ValueError:
            r = 0
        ranks.append(r)

    ranks = np.array(ranks)
    MRR = float(np.mean([recip(int(r)) for r in ranks])) if N else float("nan")
    out = {"MRR": MRR}

    for K in ks:
        rr_at_k = np.mean([1.0/float(r) if (r > 0 and r <= K) else 0.0 for r in ranks]) if N else float("nan")
        hit_at_k = np.mean([1.0 if (r > 0 and r <= K) else 0.0 for r in ranks])*100.0 if N else float("nan")
        out[f"MRR@{K}"] = float(rr_at_k)
        out[f"Hits@{K}"] = float(hit_at_k)
    return out

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
    ap = argparse.ArgumentParser(description="Tahoe Vision k-NN MOA ranking (counts) + MRR (simple drug→MOA)")
    ap.add_argument("--vision-h5ad", required=True, help="Path to vision_diff_scores.h5ad")
    ap.add_argument("--drugs-csv", required=True, help="Path to data/drugs.csv (with 'drug' and 'moa-fine')")
    ap.add_argument("--out-dir", required=True, help="Directory to write moarank CSVs")
    ap.add_argument("--concentration", type=float, default=5.0, help="Drug concentration filter (default 5.0)")
    ap.add_argument("--top-n", type=int, default=250, help="Top/Bottom N signatures per drug (default 250)")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 15], help="k values for k-NN (default 10 15)")
    ap.add_argument("--drug-col", default="drug", help="Obs column for drug (default 'drug')")
    ap.add_argument("--conc-col", default="concentration", help="Obs column for concentration (default 'concentration')")
    ap.add_argument("--cell-col", default=None, help="Optional obs cell column (e.g., 'cell_name' or 'cell_id')")
    ap.add_argument("--cell-id", default=None, help="Optional specific cell id/name to filter on")
    ap.add_argument("--show-unmapped", type=int, default=10, help="How many unmapped drug names to print")
    args = ap.parse_args()

    # Output dir
    moarank_dir = os.path.join(args.out_dir, "moarank")
    os.makedirs(moarank_dir, exist_ok=True)

    # Load AnnData and build binary matrices
    adata = ad.read_h5ad(args.vision_h5ad)
    M_up, M_down, drugs_raw, gene_sets = build_binary_matrices_from_vision(
        adata=adata,
        concentration=args.concentration,
        top_n=args.top_n,
        drug_col=args.drug_col,
        conc_col=args.conc_col,
        cell_col=args.cell_col,
        cell_id_value=args.cell_id,
    )

    # SIMPLE drug → MOA mapping (no canonicalization)
    drug_to_moa, fine_moas = simple_drug_to_moa(args.drugs_csv)

    # Keep only drugs with MOA mapping
    keep_mask = [d in drug_to_moa for d in drugs_raw]
    unmapped = [d for d, keep in zip(drugs_raw, keep_mask) if not keep]
    if unmapped:
        print(f"[warn] {len(unmapped)} drugs had no MOA mapping; first few:", unmapped[:args.show_unmapped])

    M_up = M_up[keep_mask, :]
    M_down = M_down[keep_mask, :]
    drugs_f_raw = [d for d, k in zip(drugs_raw, keep_mask) if k]
    labels = [drug_to_moa[d] for d in drugs_f_raw]  # MOA per drug (aligned)

    # Distances
    D_avg = averaged_jaccard_distance(M_up, M_down)

    # Unique MOAs for baselines
    unique_moas = sorted(list(set(labels)))

    # Cell token for filenames
    cell_token = token_for_filename(args.cell_id) if (args.cell_col and args.cell_id) else "allcells"

    # For each k: rankings + metrics
    for k in args.k:
        print(f"\n=== k = {k} ===")
        ranked_lists: List[List[str]] = []
        true_labels: List[str] = []

        for i, (drug_name_raw, true_moa) in enumerate(zip(drugs_f_raw, labels)):
            scores = rank_moas_counts(D_avg[i], labels, k=k)   # [(moa, score)]
            ranked_moas = [m for m, _ in scores]
            ranked_lists.append(ranked_moas)
            true_labels.append(true_moa)

            # Write per-drug CSV
            drug_token = token_for_filename(drug_name_raw)
            out_path = os.path.join(moarank_dir, f"moarank_{cell_token}_{drug_token}_rankings.csv")
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["drug", "moa", "score", "rank"])
                for r, (moa, sc) in enumerate(scores, start=1):
                    w.writerow([drug_name_raw, moa, f"{sc:.6f}", r])

        # Metrics vs random baseline
        metrics = mrr_and_hits_from_ranklists(true_labels, ranked_lists, ks=[1,5,10])
        baselines = random_baselines_moa(num_classes=len(unique_moas), ks=[1,5,10])

        print(f"MRR:   {metrics['MRR']:.3f}   (exp {baselines['MRR']:.3f})")
        print(f"MRR@1: {metrics['MRR@1']:.3f} (exp {baselines['MRR@1']:.3f}) | "
              f"Hits@1: {metrics['Hits@1']:.1f}% (exp {baselines['Hits@1']:.1f}%)")
        print(f"MRR@5: {metrics['MRR@5']:.3f} (exp {baselines['MRR@5']:.3f}) | "
              f"Hits@5: {metrics['Hits@5']:.1f}% (exp {baselines['Hits@5']:.1f}%)")
        print(f"MRR@10:{metrics['MRR@10']:.3f} (exp {baselines['MRR@10']:.3f}) | "
              f"Hits@10:{metrics['Hits@10']:.1f}% (exp {baselines['Hits@10']:.1f}%)")

    print(f"\nWrote MOA rankings to: {moarank_dir}")

if __name__ == "__main__":
    main()
