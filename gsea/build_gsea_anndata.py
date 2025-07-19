#!/usr/bin/env python3
"""
build_gsea_anndata.py
Processes multiple fgsea parquet files and stores NES/padj in sparse AnnData layers.
Designed to stay memory-efficient and support large-scale processing.

Run:
    python build_gsea_anndata.py
    python build_gsea_anndata.py --subset 5  # Only use first 5 parquet files
"""
import argparse
import os
import re
import time
from pathlib import Path
from glob import glob
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_group_id(group_id):
    match = re.match(r"\[\('(.+?)',\s*([\d\.]+),\s*'uM'\)\]_(\d+)", group_id)
    if match:
        drug, conc, plate = match.groups()
        return f"{drug}_{conc}uM", plate
    return None, None

def process_file_sparse(path):
    start = time.time()
    try:
        print(f"\n Reading {path}...")
        df = pd.read_parquet(path, columns=["pathway", "NES", "padj", "group_id", "cell_line"])
        print(f"Loaded {os.path.basename(path)} in {time.time() - start:.2f}s with {len(df):,} rows")

        df["pathway"] = df["pathway"].str.strip().str.upper()
        df["cell_line"] = df["cell_line"].astype(str)

        extract_df = df["group_id"].str.extract(r"\[\('(.+?)',\s*([\d\.]+),\s*'uM'\)\]_(\d+)")
        extract_df.columns = ["drug", "conc", "plate"]
        extract_df["drug_conc"] = extract_df["drug"] + "_" + extract_df["conc"] + "uM"
        df = pd.concat([df, extract_df[["drug_conc", "plate"]]], axis=1)
        df = df.dropna(subset=["drug_conc", "plate"])
        print(f"Parsed group_id for {len(df):,} rows")

        grouped = df.groupby(["drug_conc", "cell_line", "plate"])
        avg_padj = grouped["padj"].mean().reset_index().rename(columns={"padj": "mean_padj"})
        best_plate = avg_padj.loc[avg_padj.groupby(["drug_conc", "cell_line"])["mean_padj"].idxmin()]
        merged = pd.merge(df, best_plate, on=["drug_conc", "cell_line", "plate"])
        print(f"Selected best plates: {len(merged):,} rows retained")

        records = [
            (f"{row['drug_conc']}__{row['cell_line']}", row["pathway"], row["NES"], row["padj"])
            for _, row in merged.iterrows()
        ]
        print(f"Built {len(records):,} records in {time.time() - start:.2f}s total")
        return records
    except Exception as e:
        print(f" Error processing {path}: {e}")
        return []

def build_sparse_gsea_anndata(parquet_dir, out_path, max_threads=8, subset_files=None):
    overall_start = time.time()
    all_paths = sorted(glob(f"{parquet_dir}/c_*.parquet"))
    if subset_files:
        all_paths = all_paths[:subset_files]
    print(f"Found {len(all_paths)} parquet files to process.")
    print("Starting parallel processing...")

    cache_nes = defaultdict(list)
    cache_padj = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_file_sparse, path) for path in all_paths]
        for i, f in enumerate(as_completed(futures), 1):
            print(f" Finished {i}/{len(futures)} files")
            for sample, pathway, nes, padj in f.result():
                cache_nes[(sample, pathway)].append(nes)
                cache_padj[(sample, pathway)].append(padj)

    print(f"\n Parallel processing done in {time.time() - overall_start:.2f}s")
    print(f"Unique rows: {len(set(k[0] for k in cache_nes))} | Unique gene sets: {len(set(k[1] for k in cache_nes))}")
    print("Constructing sparse NES and padj matrices...")
    build_start = time.time()

    sample_keys, gene_sets, nes_vals, padj_vals = [], [], [], []
    for (sample, gs), vals in cache_nes.items():
        sample_keys.append(sample)
        gene_sets.append(gs)
        nes_vals.append(np.mean(vals))
        padj_vals.append(np.mean(cache_padj[(sample, gs)]))

    rows, row_index = pd.factorize(sample_keys, sort=False)
    cols, col_index = pd.factorize(gene_sets, sort=False)
    shape = (row_index.size, col_index.size)

    X_nes = sp.coo_matrix((np.array(nes_vals, dtype=np.float32), (rows, cols)), shape=shape).tocsr()
    X_padj = sp.coo_matrix((np.array(padj_vals, dtype=np.float32), (rows, cols)), shape=shape).tocsr()
    print(f"Matrices constructed in {time.time() - build_start:.2f}s")

    obs_df = pd.Series(row_index, name="sample_id").str.split("__", expand=True)
    obs_df.columns = ["drug_conc", "cell_line"]
    obs_df.index = row_index
    var_df = pd.DataFrame(index=pd.Index(col_index.astype(str), name="pathway"))

    print("Saving to AnnData...")
    adata = ad.AnnData(
        X=sp.csr_matrix(shape, dtype=np.float32),
        obs=obs_df,
        var=var_df,
        layers={"gsea_nes": X_nes, "gsea_padj": X_padj},
    )
    adata.var_names_make_unique()
    adata.write(out_path, compression="gzip")
    print(f"Done! Saved to {out_path}")
    print(f"Total time: {time.time() - overall_start:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Build sparse GSEA AnnData from multiple fgsea parquet files")
    parser.add_argument("--parquet-dir", default="/home/ubuntu/sigspace2/data/gsea", help="Input directory")
    parser.add_argument("--output", default="/home/ubuntu/sigspace2/results/gsea_all_sparse.h5ad", help="Output AnnData file")
    parser.add_argument("--max-threads", type=int, default=8, help="Max threads to use")
    parser.add_argument("--subset", type=int, default=None, help="Subset to first N files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_sparse_gsea_anndata(
        parquet_dir=args.parquet_dir,
        out_path=args.output,
        max_threads=args.max_threads,
        subset_files=args.subset,
    )

if __name__ == "__main__":
    main()
