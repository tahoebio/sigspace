#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

UNCLEAR = {"unclear","unknown","na","n/a","none","unspecified","other",""}

def load_drug2moa(drugs_csv):
    """Return {drug→moa} dict and per-MoA class sizes."""
    df = pd.read_csv(drugs_csv, usecols=["drug","moa-fine"])
    df["drug"] = df["drug"].astype(str).str.strip()
    df["moa-fine"] = df["moa-fine"].astype(str).str.strip()
    df = df[~df["moa-fine"].str.lower().isin(UNCLEAR)]
    df = df.drop_duplicates(subset="drug", keep="first")
    moa_size = df.groupby("moa-fine")["drug"].nunique().rename("NumDrugs")
    print (moa_size)
    return dict(zip(df["drug"], df["moa-fine"])), moa_size

def knn_mean_rank_by_moa(moarank_dir: str, drugs_csv: str, cell_token: str = "allcells") -> pd.DataFrame:
    """Summarize mean true-MoA rank for all drugs from moarank CSVs."""
    drug2moa, _ = load_drug2moa(drugs_csv)
    files = glob.glob(os.path.join(moarank_dir, f"moarank_{cell_token}_*_rankings.csv"))
    print (len(files))
    if not files:
        raise FileNotFoundError(f"No moarank files found for cell_token='{cell_token}' in {moarank_dir}")

    rows = []
    for f in files:
        df = pd.read_csv(f)  # columns: drug, moa, score, rank
        drug = df["drug"].iloc[0]
        true_moa = drug2moa.get(drug)
        if true_moa is None:
            continue
        if true_moa == "Sonic inhibitor":
            r = df.loc[df["moa"] == true_moa, "rank"]
            print (df)
        r = df.loc[df["moa"] == true_moa, "rank"]

        if not r.empty:
            rows.append({"drug": drug, "MoA": true_moa, "TrueRank": int(r.min())})

    df_all = pd.DataFrame(rows)
    df_summary = (
        df_all.groupby("MoA")
        .agg(NumDrugs=("drug", "nunique"), MeanTrueRank=("TrueRank", "mean"))
        .reset_index()
    )
    return df_summary

def main():
    ap = argparse.ArgumentParser(description="LLM vs k-NN: mean true-MoA rank vs class size")
    ap.add_argument("--llm-table", required=True, help="CSV with columns: MoA, NumDrugs, MeanTrueRank")
    ap.add_argument("--moarank-dir", required=True, help="Directory containing moarank CSVs")
    ap.add_argument("--drugs-csv", required=True, help="drugs.csv with drug, moa-fine")
    ap.add_argument("--out", default="llm_vs_knn_scatter.png")
    args = ap.parse_args()

    # Load tables
    llm = pd.read_csv(args.llm_table)[["MoA","NumDrugs","MeanTrueRank"]].dropna()
    knn = knn_mean_rank_by_moa(args.moarank_dir, args.drugs_csv, cell_token="allcells")
    knn.to_csv("knn_summary_moa_rank_2.csv", index=False)

    # Correlations
    rho_llm = llm["NumDrugs"].corr(llm["MeanTrueRank"], method="spearman")
    rho_knn = knn["NumDrugs"].corr(knn["MeanTrueRank"], method="spearman")
    print(f"Spearman rho — LLM: {rho_llm:.2f} | k-NN: {rho_knn:.2f}")

    # # Scatter
    # plt.figure(figsize=(7.5,5.5))
    # plt.scatter(llm["NumDrugs"], llm["MeanTrueRank"], alpha=0.9, label=f"LLM (Spearman's ρ={rho_llm:.2f})")
    # plt.scatter(knn["NumDrugs"], knn["MeanTrueRank"], alpha=0.9, marker="s", label=f"k-NN (Spearman's ρ={rho_knn:.2f})")
    # plt.xlabel("Number of drugs per MoA")
    # plt.ylabel("Mean rank of the true MoA (lower is better)")
    # plt.title("Mean true-MoA rank vs #drugs/MoA")
    # plt.grid(True, linestyle=":", linewidth=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(args.out, dpi=300)
    # print("Saved:", args.out)
    # --- Scatter plot (LLM vs k-NN) ---
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # LLM points
    ax.scatter(llm["NumDrugs"], llm["MeanTrueRank"],
               s=80, alpha=0.9, color="#2E86AB", edgecolor="white", linewidth=0.7,
               label=f"LLM (Spearman’s ρ={rho_llm:.2f})")

    # k-NN points
    ax.scatter(knn["NumDrugs"], knn["MeanTrueRank"],
               s=80, alpha=0.9, marker="s", color="#E67E22", edgecolor="white", linewidth=0.7,
               label=f"k-NN (Spearman’s ρ={rho_knn:.2f})")

    # Labels & title
    ax.set_xlabel("Number of drugs per MoA", fontsize=14, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Mean rank of the true MoA", fontsize=14, fontweight="bold", color="#2C3E50")
    ax.set_title("Mean true-MoA rank vs. MoA size", fontsize=16, fontweight="bold", color="#2C3E50")

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Grid and legend
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6, color="#7F8C8D")
    ax.legend(frameon=True, fontsize=12, loc="upper right")

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:", args.out)

if __name__ == "__main__":
    main()
