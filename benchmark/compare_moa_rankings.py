import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns


def plot_rank_matrix_heatmap(rank_matrix, moas, ranking_directory, cell_line_str):
    """Plot heatmap showing rank matrix for MOA rankings."""
    fig, ax = plt.subplots(figsize=(20, 16))
    mask = rank_matrix == 0

    base_cmap = plt.get_cmap("RdBu_r")
    cmap = base_cmap.with_extremes(bad="#e0e0e0")

    rank_matrix_masked = np.ma.masked_where(mask, rank_matrix)

    nonzero = rank_matrix[~mask]
    vmin = np.min(nonzero) if nonzero.size > 0 else 1
    vmax = np.max(nonzero) if nonzero.size > 0 else 1

    sns.heatmap(
        rank_matrix_masked,
        annot=False,
        cmap=cmap,
        xticklabels=moas,
        yticklabels=moas,
        ax=ax,
        cbar_kws={"shrink": 0.5, "label": "ranks"},
        square=True,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("ranked MOA")
    ax.set_ylabel("true MOA")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=2)
    fig.subplots_adjust(bottom=0.3, right=0.85, top=0.95, left=0.2)
    fig.savefig(
        ranking_directory / "figures" / f"moa_rank_matrix_heatmap_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_surprise_histogram(surprise, n, ranking_directory, cell_line_str):
    """Plot histogram of self-ranks (surprise metric)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(range(0, n + 2))
    ax.hist(surprise, bins=bins, color="orange", edgecolor="black")
    ax.set_xlabel("self-rank")
    ax.set_ylabel("count")
    ax.set_title("Distribution of Self-Ranks (Surprise Metric)")
    fig.savefig(
        ranking_directory
        / "figures"
        / f"moa_surprise_metric_histogram_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def combine_moa_rankings(files):
    """Combine MOA ranking files into a dictionary mapping ground truth drugs to ranked MOAs."""
    truth_to_ranks = {}
    for file in files:
        name = os.path.basename(file)
        if name.startswith("moarank"):
            parts = name.split("_")
            # The new format: moarank_{model}_{analyze}_{scores}_{moa_ranking}_{drug}_{cell_line}_{lambda}.csv
            # So: -3 is drug, -2 is cell line, -1 is lambda (with .csv or not)
            if len(parts) < 4:
                continue  # skip malformed names
            drug = parts[-3]
            df = pd.read_csv(file)
            # Get ranked MOAs (only those with non-zero scores)
            ranked_moas = df[df["score"] > 0]["moa"].tolist()
            truth_to_ranks[drug] = ranked_moas
    return truth_to_ranks


def analyze_moa_rankings(ranking_directory, drugs_file, cell_line=None):
    """Analyze MOA ranking outputs and generate visualizations."""
    ranking_directory = pathlib.Path(ranking_directory)
    drugs_file = pathlib.Path(drugs_file)
    drug_data = pd.read_csv(drugs_file)
    drugs = drug_data["drug"].tolist()
    cell_line_str = "None" if cell_line is None else cell_line

    # Find MOA ranking files
    files = [
        f
        for f in ranking_directory.glob("*.csv")
        if "moarank" in f.name and cell_line_str in f.name
    ]

    if not files:
        print(
            f"No MOA ranking files found in {ranking_directory} for cell line {cell_line_str}"
        )
        return

    rank_dictionary = combine_moa_rankings(files)

    # Get unique MOAs from the data
    unique_moas = sorted(
        [moa for moa in drug_data["moa-fine"].dropna().unique() if moa != "unclear"]
    )
    moa_to_idx = {moa: idx for idx, moa in enumerate(unique_moas)}

    # Create rank matrix
    rank_matrix = np.zeros((len(unique_moas), len(unique_moas)))

    # Fill rank matrix
    for drug in drugs:
        if drug not in rank_dictionary:
            continue

        gt_moa = drug_data[drug_data["drug"] == drug]["moa-fine"].iloc[0]
        if pd.isna(gt_moa):
            continue

        gt_moa_idx = moa_to_idx[gt_moa]
        ranked_moas = rank_dictionary[drug]

        for rank, ranked_moa in enumerate(ranked_moas):
            if ranked_moa in moa_to_idx:
                ranked_moa_idx = moa_to_idx[ranked_moa]
                rank_matrix[gt_moa_idx, ranked_moa_idx] = rank + 1

    # Create figures directory if it doesn't exist
    (ranking_directory / "figures").mkdir(exist_ok=True)

    # Generate plots
    plot_rank_matrix_heatmap(rank_matrix, unique_moas, ranking_directory, cell_line_str)

    # Calculate surprise metric (self-ranks)
    surprise = [rank_matrix[i, i] for i in range(len(unique_moas))]
    plot_surprise_histogram(
        surprise, len(unique_moas), ranking_directory, cell_line_str
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="analyze MOA ranking outputs and visualize results"
    )
    parser.add_argument(
        "-r",
        "--ranking_directory",
        type=str,
        required=True,
        help="directory containing MOA ranking CSV files (moarank_*.csv)",
    )
    parser.add_argument(
        "-d",
        "--drugs_file",
        type=str,
        required=True,
        help="path to the file containing drug information",
    )
    parser.add_argument(
        "-c",
        "--cell_line",
        type=str,
        default=None,
        help="cell line to analyze, else omit if analyzing across all cell lines (defaults to None)",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    ranking_directory = arguments.ranking_directory
    drugs_file = arguments.drugs_file
    cell_line = arguments.cell_line
    analyze_moa_rankings(ranking_directory, drugs_file, cell_line)
