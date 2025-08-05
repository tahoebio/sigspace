"""
build_gsea_anndata_fast.py

to build GSEA h5ad in format:
- adata.layers["gsea_nes"]: NES scores from GSEA
- adata.layers["gsea_padj"]: p-values adjusted (FDR) from GSEA
- adata.layers["gsea_mask"]: 1 for valid NES/padj, 0 for missing values
- adata.obs: sample_id, drug_conc, cell_line
- adata.var: gene set names

This version uses pyarrow for faster parquet processing and is designed to handle large datasets efficiently.
"""

import argparse
import os
import time
import gc
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import scipy.sparse as sp
import anndata as ad

PATTERN = r"\[\('(?P<drug>[^']+)',\s*(?P<conc>[0-9.]+),\s*'uM'\)\]_(?P<plate>\d+)"
NEEDED = ["pathway", "NES", "padj", "group_id", "cell_line"]


def arrow_select_best_plate(tab: pa.Table) -> pa.Table:
    ex = pc.extract_regex(tab["group_id"], PATTERN).combine_chunks()

    drug = ex.field("drug")
    conc = ex.field("conc")
    plate = ex.field("plate")

    drug_conc_temp = pc.binary_join_element_wise(drug, conc, "_")
    drug_conc = pc.binary_join_element_wise(drug_conc_temp, "uM", "")

    tab = (
        tab.append_column("drug_conc", drug_conc)
        .append_column("plate", plate)
        .filter(pc.invert(pc.is_null(drug_conc)))
    )

    agg = tab.group_by(["drug_conc", "cell_line", "plate"]).aggregate(
        [("padj", "mean")]
    )
    agg = agg.rename_columns(["drug_conc", "cell_line", "plate", "padj_mean"])

    agg_df = agg.to_pandas()
    best_plate_idx = agg_df.groupby(["drug_conc", "cell_line"])["padj_mean"].idxmin()
    best_rows_df = agg_df.loc[best_plate_idx]
    best_rows = pa.Table.from_pandas(best_rows_df)

    return tab.join(
        best_rows, keys=["drug_conc", "cell_line", "plate"], join_type="inner"
    )


def build(
    parquet_dir: str,
    out: str,
    subset: int | None = None,
    threads: int = 8,
    debug: bool = False,
):
    os.environ["ARROW_NUM_THREADS"] = str(threads)
    paths = sorted(glob(f"{parquet_dir}/c_*.parquet"))
    if subset:
        paths = paths[:subset]
    print(f"{len(paths)} files -> {out}")

    row_id, col_id = {}, {}
    row_idx, col_idx, nes_vals, padj_vals = [], [], [], []

    t0 = time.perf_counter()
    for i, p in enumerate(paths, 1):
        t_start = time.perf_counter()
        tab = pq.read_table(p, columns=NEEDED, memory_map=True)
        if debug:
            print(f"\nLoaded {Path(p).name} with {tab.num_rows:,} rows")

        tab = arrow_select_best_plate(tab)
        if debug:
            print(f"Selected best plates → {tab.num_rows:,} rows after filtering")

        pdf = tab.select(
            ["drug_conc", "cell_line", "pathway", "NES", "padj"]
        ).to_pandas(types_mapper=pd.ArrowDtype)

        num_na_nes = pdf["NES"].isna().sum()
        num_na_padj = pdf["padj"].isna().sum()
        if debug and (num_na_nes > 0 or num_na_padj > 0):
            print(f"Missing NES: {num_na_nes:,} | Missing padj: {num_na_padj:,}")
            na_rows = pdf[pdf["NES"].isna() | pdf["padj"].isna()]
            print("NA Rows:")
            print(na_rows.to_string(index=False, max_rows=10))

        pdf = pdf.dropna(subset=["NES", "padj"])

        if debug:
            print("Sample rows:\n", pdf.head(3).to_string(index=False))

        samples = pdf["drug_conc"].astype(str) + "__" + pdf["cell_line"].astype(str)
        pathways = pdf["pathway"].str.strip().str.upper()

        for s, g, nes, q in zip(samples, pathways, pdf["NES"], pdf["padj"]):
            r = row_id.setdefault(s, len(row_id))
            c = col_id.setdefault(g, len(col_id))
            row_idx.append(r)
            col_idx.append(c)
            nes_vals.append(nes)
            padj_vals.append(q)

        del tab, pdf
        gc.collect()
        dt = time.perf_counter() - t_start
        print(
            f"  {i:>2}/{len(paths)}  {Path(p).name:<20} {dt:>5.1f}s  rows so far {len(row_idx):,}"
        )

    shape = (len(row_id), len(col_id))
    nes_m = sp.coo_matrix(
        (np.array(nes_vals, dtype=np.float32), (row_idx, col_idx)), shape
    ).tocsr()
    padj_m = sp.coo_matrix(
        (np.array(padj_vals, dtype=np.float32), (row_idx, col_idx)), shape
    ).tocsr()
    mask_m = sp.coo_matrix(
        (np.ones(len(nes_vals), dtype=np.uint8), (row_idx, col_idx)), shape
    ).tocsr()

    obs = (
        pd.Series(row_id)
        .sort_values()
        .rename_axis("sample_id")
        .reset_index(name="row")
        .set_index("row")["sample_id"]
        .str.split("__", expand=True)
    )
    obs.columns = ["drug_conc", "cell_line"]

    var = pd.DataFrame(index=pd.Index(sorted(col_id, key=col_id.get), name="pathway"))

    adata = ad.AnnData(
        X=sp.csr_matrix(shape, dtype=np.float32),
        obs=obs,
        var=var,
        layers={"gsea_nes": nes_m, "gsea_padj": padj_m, "gsea_mask": mask_m},
    )
    adata.var_names_make_unique()
    adata.write(out, compression="gzip")

    print(
        f"\n{out} written  |  samples {shape[0]:,}  pathways {shape[1]:,}  "
        f"elapsed {time.perf_counter()-t0:.1f}s"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="/home/ubuntu/sigspace/data/gsea_47")
    ap.add_argument(
        "--output", default="/home/ubuntu/sigspace/results/gsea_all_cell_lines.h5ad"
    )
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--subset", type=int, help="first N files (debug)")
    ap.add_argument("--debug", action="store_true", help="print debug info")
    args = ap.parse_args()

    build(args.parquet_dir, args.output, args.subset, args.threads, args.debug)
