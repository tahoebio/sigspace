import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns


def plot_rank_matrix_heatmap(rank_matrix, drugs, ranking_directory, cell_line_str):
    fig, ax = plt.subplots(figsize=(20, 16))
    mask = rank_matrix == 0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    rank_matrix_masked = np.ma.masked_where(mask, rank_matrix)
    sns.heatmap(
        rank_matrix_masked,
        annot=False,
        cmap=cmap,
        xticklabels=drugs,
        yticklabels=drugs,
        ax=ax,
        cbar_kws={"shrink": 0.5, "label": "ranks"},
        square=True,
        mask=mask,
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("ranked drug")
    ax.set_ylabel("true drug")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=2)
    fig.subplots_adjust(bottom=0.3, right=0.85, top=0.95, left=0.2)
    fig.savefig(
        ranking_directory / "figures" / f"rank_matrix_heatmap_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_surprise_histogram(surprise, n, ranking_directory, cell_line_str):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(range(0, n + 2))
    ax.hist(surprise, bins=bins, color="orange", edgecolor="black")
    ax.set_xlabel("self-rank")
    ax.set_ylabel("count")
    fig.savefig(
        ranking_directory
        / "figures"
        / f"surprise_metric_histogram_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_moa_analysis(
    same_moa_ranks, different_moa_ranks, ranking_directory, cell_line_str
):
    same_moa_mean = np.mean(same_moa_ranks)
    different_moa_mean = np.mean(different_moa_ranks)
    mean_difference = different_moa_mean - same_moa_mean
    fig, (ax_one, ax_two) = plt.subplots(1, 2, figsize=(15, 6))
    sns.kdeplot(data=same_moa_ranks, ax=ax_one, label="same moa", color="blue")
    sns.kdeplot(data=different_moa_ranks, ax=ax_one, label="different moa", color="red")
    ax_one.axvline(same_moa_mean, color="blue", linestyle="--", alpha=0.5)
    ax_one.axvline(different_moa_mean, color="red", linestyle="--", alpha=0.5)
    ax_one.text(
        0.02,
        0.98,
        f"mean difference: {mean_difference:.1f}",
        transform=ax_one.transAxes,
        verticalalignment="top",
    )
    ax_one.set_xlabel("rank")
    ax_one.set_ylabel("density")
    ax_one.legend()
    sns.ecdfplot(data=same_moa_ranks, ax=ax_two, label="same moa", color="blue")
    sns.ecdfplot(
        data=different_moa_ranks, ax=ax_two, label="different moa", color="red"
    )
    ax_two.set_xlabel("rank")
    ax_two.set_ylabel("cumulative probability")
    ax_two.set_title("cumulative distribution of ranks")
    stats_text = (
        f"same moa:\n"
        f"  mean: {same_moa_mean:.1f}\n"
        f"  median: {np.median(same_moa_ranks):.1f}\n"
        f"different moa:\n"
        f"  mean: {different_moa_mean:.1f}\n"
        f"  median: {np.median(different_moa_ranks):.1f}\n"
        f"mean difference: {mean_difference:.1f}"
    )
    ax_two.text(
        1.05,
        0.5,
        stats_text,
        transform=ax_two.transAxes,
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(
        ranking_directory / "figures" / f"moa_analysis_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_moa_enrichment(topk_moa_fractions, top_ks, ranking_directory, cell_line_str):
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_frac = np.mean(topk_moa_fractions, axis=0)
    ax.plot(top_ks, mean_frac, marker="o")
    ax.set_xlabel("top k")
    ax.set_ylabel("fraction with same moa")
    fig.savefig(
        ranking_directory / "figures" / f"moa_enrichment_topk_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_target_jaccard(topk_jaccards, top_ks, ranking_directory, cell_line_str):
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_jaccard = np.mean(topk_jaccards, axis=0)
    ax.plot(top_ks, mean_jaccard, marker="o")
    ax.set_xlabel("top k")
    ax.set_ylabel("mean jaccard similarity")
    fig.savefig(
        ranking_directory / "figures" / f"target_jaccard_topk_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_first_match_histogram(ranks, label, ranking_directory, cell_line_str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ranks, bins=range(1, max(ranks) + 2), color="purple", edgecolor="black")
    ax.set_xlabel(f"rank of first {label}")
    ax.set_ylabel("count")
    fig.savefig(
        ranking_directory
        / "figures"
        / f"first_{label.replace(' ', '_')}_rank_{cell_line_str}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def combine_rankings(files):
    truth_to_ranks = {}
    for file in files:
        name = os.path.basename(file)
        if name.startswith("drugrank"):
            parts = name.split("_")
            drug = "_".join(parts[1:-1])
            df = pd.read_csv(file)
            ranks = df["drug"].tolist()
            truth_to_ranks[drug] = ranks
    return truth_to_ranks


def analyze_rankings(ranking_directory, drugs_file, cell_line=None):
    ranking_directory = pathlib.Path(ranking_directory)
    drugs_file = pathlib.Path(drugs_file)
    drug_data = pd.read_csv(drugs_file)
    drugs = drug_data["drug"].tolist()
    n = len(drugs)
    cell_line_str = "None" if cell_line is None else cell_line
    files = list(ranking_directory.glob(f"drugrank_*_{cell_line_str}.csv"))
    rank_dictionary = combine_rankings(files)
    rank_matrix = np.zeros((n, n))
    for i, drug in enumerate(drugs):
        if drug not in rank_dictionary:
            continue
        for rank, ranked_drug in enumerate(rank_dictionary[drug]):
            if ranked_drug not in drugs:
                continue
            j = drugs.index(ranked_drug)
            rank_matrix[i, j] = rank + 1
    moa_map = dict(zip(drug_data["drug"], drug_data["moa-fine"]))
    target_map = {
        row["drug"]: set(
            [t.strip() for t in str(row["targets"]).split(",") if t.strip()]
        )
        if pd.notna(row["targets"])
        else set()
        for _, row in drug_data.iterrows()
    }
    plot_rank_matrix_heatmap(rank_matrix, drugs, ranking_directory, cell_line_str)
    surprise = [rank_matrix[i, i] for i in range(n)]
    first_key = next(iter(rank_dictionary))
    plot_surprise_histogram(
        surprise, len(rank_dictionary[first_key]), ranking_directory, cell_line_str
    )
    same_moa_ranks = []
    different_moa_ranks = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rank_matrix[i, j] != 0:
                (
                    same_moa_ranks
                    if moa_map[drugs[i]] == moa_map[drugs[j]]
                    else different_moa_ranks
                ).append(rank_matrix[i, j])
    same_moa_ranks = np.array(same_moa_ranks)
    different_moa_ranks = np.array(different_moa_ranks)
    plot_moa_analysis(
        same_moa_ranks, different_moa_ranks, ranking_directory, cell_line_str
    )
    top_ks = [1, 5, 10, 20]
    topk_moa_fractions = []
    topk_jaccards = []
    first_same_moa_ranks = []
    first_shared_target_ranks = []
    for drug in drugs:
        if drug not in rank_dictionary:
            continue
        ranked_list = rank_dictionary[drug]
        gt_moa = moa_map[drug]
        gt_targets = target_map[drug]
        moa_fractions = []
        jaccards = []
        for k in top_ks:
            topk = ranked_list[:k]
            if not topk:
                moa_fractions.append(np.nan)
                jaccards.append(np.nan)
                continue
            moa_fractions.append(
                np.mean([moa_map.get(d, None) == gt_moa for d in topk])
            )
            jaccards.append(
                np.mean(
                    [
                        len(gt_targets & target_map.get(d, set()))
                        / len(gt_targets | target_map.get(d, set()))
                        if (len(gt_targets | target_map.get(d, set())) > 0)
                        else 0.0
                        for d in topk
                    ]
                )
            )
        topk_moa_fractions.append(moa_fractions)
        topk_jaccards.append(jaccards)
        first_moa = next(
            (
                i + 1
                for i, d in enumerate(ranked_list)
                if d != drug and moa_map.get(d, None) == gt_moa
            ),
            None,
        )
        if first_moa is not None:
            first_same_moa_ranks.append(first_moa)
        first_target = next(
            (
                i + 1
                for i, d in enumerate(ranked_list)
                if d != drug and len(gt_targets & target_map.get(d, set())) > 0
            ),
            None,
        )
        if first_target is not None:
            first_shared_target_ranks.append(first_target)
    topk_moa_fractions = np.array(topk_moa_fractions)
    topk_jaccards = np.array(topk_jaccards)
    plot_moa_enrichment(topk_moa_fractions, top_ks, ranking_directory, cell_line_str)
    plot_target_jaccard(topk_jaccards, top_ks, ranking_directory, cell_line_str)
    if first_same_moa_ranks:
        plot_first_match_histogram(
            first_same_moa_ranks, "same MOA drug", ranking_directory, cell_line_str
        )
    if first_shared_target_ranks:
        plot_first_match_histogram(
            first_shared_target_ranks,
            "shared target drug",
            ranking_directory,
            cell_line_str,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="analyze drug ranking outputs and visualize results"
    )
    parser.add_argument(
        "-r",
        "--ranking_directory",
        type=str,
        help="directory containing ranking CSV files (drugrank_{ground_truth}_{cell_line}.csv)",
    )
    parser.add_argument(
        "-d",
        "--drugs_file",
        type=str,
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
    analyze_rankings(ranking_directory, drugs_file, cell_line)
