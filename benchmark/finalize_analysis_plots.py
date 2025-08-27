import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def combine_rankings(files, tag):
    truth_to_ranks = {}

    for file in files:
        name = os.path.basename(file)
        if name.startswith(tag):
            rankings = pd.read_csv(file)

            parts = name.split("_")
            truth_to_ranks[parts[-3]] = (
                rankings["drug"].tolist()
                if tag == "drugrank"
                else rankings["moa"].tolist()
            )

    return truth_to_ranks


def plot_rank_matrix_heatmap(rank_matrix, experiment_drugs, reference, save_file):
    fig, ax = plt.subplots(figsize=(20, 16))

    mask = rank_matrix == 0
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e0e0e0")

    rank_matrix_masked = np.ma.masked_where(mask, rank_matrix)

    nonzero = rank_matrix[~mask]
    vmin = np.min(nonzero) if nonzero.size > 0 else 1
    vmax = np.max(nonzero) if nonzero.size > 0 else 1

    sns.heatmap(
        rank_matrix_masked,
        annot=False,
        cmap=cmap,
        xticklabels=reference,
        yticklabels=experiment_drugs,
        ax=ax,
        cbar_kws={"shrink": 0.5, "label": "ranks"},
        square=True,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("ranked")
    ax.set_ylabel("truth")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=2)
    fig.subplots_adjust(bottom=0.3, right=0.85, top=0.95, left=0.2)
    fig.savefig(save_file, bbox_inches="tight", dpi=300)

    plt.close(fig)


def plot_moa_distributions(same_moa_ranks, different_moa_ranks, save_file):
    same_moa_mean = np.mean(np.array(same_moa_ranks))
    different_moa_mean = np.mean(np.array(different_moa_ranks))

    fig, (ax_one, ax_two) = plt.subplots(1, 2, figsize=(15, 6))

    sns.kdeplot(data=same_moa_ranks, ax=ax_one, label="same moa", color="blue")
    sns.kdeplot(data=different_moa_ranks, ax=ax_one, label="different moa", color="red")
    ax_one.axvline(same_moa_mean, color="blue", linestyle="--", alpha=0.5)
    ax_one.axvline(different_moa_mean, color="red", linestyle="--", alpha=0.5)
    ax_one.set_xlabel("rank")
    ax_one.set_ylabel("density")
    ax_one.set_title("out_of_moa vs. in_moa density distributions")
    ax_one.legend()

    sns.ecdfplot(data=same_moa_ranks, ax=ax_two, label="same moa", color="blue")
    sns.ecdfplot(
        data=different_moa_ranks, ax=ax_two, label="different moa", color="red"
    )
    ax_two.set_xlabel("rank")
    ax_two.set_ylabel("cumulative probability")
    ax_two.set_title("cumulative distributions of ranks")

    fig.tight_layout()
    fig.savefig(save_file, bbox_inches="tight", dpi=300)

    plt.close()


def plot_binned_true_drug_ranks(
    rank_matrix, experiment_drugs, all_drugs, save_file, k=50
):
    num_experiments = len(experiment_drugs)
    num_drugs = len(all_drugs)

    bin_specs = [
        (1, 5, "1-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 30, "21-30"),
        (31, 40, "31-40"),
        (41, 50, "41-50"),
    ]

    bin_labels = [b[2] for b in bin_specs] + ["not found"]
    observed_counts = OrderedDict((label, 0) for label in bin_labels)

    true_ranks = [
        rank_matrix[i, all_drugs.index(true_drug)]
        for i, true_drug in enumerate(experiment_drugs)
    ]

    for r in true_ranks:
        if r == 0:
            observed_counts["not found"] += 1
        else:
            for low, high, label in bin_specs:
                if low <= r <= high:
                    observed_counts[label] += 1

    observed_proportions = {k: v / num_experiments for k, v in observed_counts.items()}

    expected_proportions = OrderedDict()
    for low, high, label in bin_specs:
        expected_proportions[label] = (high - low + 1) / num_drugs
    expected_proportions["not found"] = (num_drugs - k) / num_drugs

    df = pd.DataFrame(
        {
            "bin": list(observed_proportions.keys())
            + list(expected_proportions.keys()),
            "proportion": list(observed_proportions.values())
            + list(expected_proportions.values()),
            "type": ["observed"] * len(observed_proportions)
            + ["baseline"] * len(expected_proportions),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="bin", y="proportion", hue="type", ax=ax)
    ax.set_xlabel("true drug rank (binned)")
    ax.set_ylabel("proportion of experiments")
    ax.set_title(
        f"binned true drug rank (top {k} reported) - observed vs. random baseline"
    )
    ax.set_ylim(0, max(df["proportion"].max() * 1.15, 0.15))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
    ax.legend(title="", frameon=False)

    fig.tight_layout()
    fig.savefig(save_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_moa_ranking_performance_by_class(
    rank_matrix, experiment_drugs, unique_moas, moa_to_drugs, save_file
):
    moa_performance = {}
    for moa, drugs in moa_to_drugs.items():
        ranks = []
        for drug in drugs:
            rank = rank_matrix[experiment_drugs.index(drug), unique_moas.index(moa)]
            if rank > 0:
                ranks.append(rank)

        if ranks:
            moa_performance[moa] = {
                "mean rank": np.mean(ranks),
                "std rank": np.std(ranks),
                "ranks": ranks,
            }

    moa_labels = list(moa_performance.keys())
    mean_ranks = [moa_performance[moa]["mean rank"] for moa in moa_labels]
    counts = [len(moa_performance[moa]["ranks"]) for moa in moa_labels]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(
        range(len(moa_labels)), mean_ranks, alpha=0.8, color="skyblue", edgecolor="navy"
    )
    ax.set_xlabel("drug class")
    ax.set_ylabel("mean rank of true moa")
    ax.set_xticks(range(len(moa_labels)))
    ax.set_xticklabels([label for label in moa_labels], rotation=45, ha="right")

    for _, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"n = {count}",
            ha="center",
            va="center",
            fontsize=9,
            color="navy",
            fontweight="normal",
        )

    plt.tight_layout()
    fig.savefig(save_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def analyze_drug_rankings(ranking_directory, drugs_file, cell_line):
    drug_metadata = pd.read_csv(drugs_file)
    all_drugs = drug_metadata["drug"].tolist()
    all_moas = drug_metadata["moa-fine"].tolist()

    rank_dictionary = combine_rankings(
        [
            os.path.join(ranking_directory, f)
            for f in os.listdir(ranking_directory)
            if f.endswith(".csv")
        ],
        "drugrank",
    )

    rank_dictionary = {
        drug: ranks for drug, ranks in rank_dictionary.items() if drug in all_drugs
    }

    experiment_drugs = list(rank_dictionary.keys())

    rank_matrix = np.zeros((len(experiment_drugs), len(all_drugs)))

    for i, drug in enumerate(experiment_drugs):
        for rank, ranked_drug in enumerate(rank_dictionary[drug]):
            if ranked_drug in all_drugs:
                rank_matrix[i, all_drugs.index(ranked_drug)] = rank + 1

    drug_to_moa = {drug: all_moas[all_drugs.index(drug)] for drug in experiment_drugs}

    drug_to_targets = {}
    for drug in experiment_drugs:
        row = drug_metadata[drug_metadata["drug"] == drug].iloc[0]

        drug_to_targets[drug] = (
            set([t.strip() for t in str(row["targets"]).split(",") if t.strip()])
            if pd.notna(row["targets"])
            else set()
        )

    # PLOT 1 - HEATMAP OF RANK MATRIX FOR DRUGRANK TASK
    plot_rank_matrix_heatmap(
        rank_matrix,
        experiment_drugs,
        all_drugs,
        os.path.join(
            ranking_directory, "figures", f"rank_matrix_heatmap_{cell_line}.png"
        ),
    )

    # PLOT 2 - OUT_OF_MOA VS. IN_MOA DISTRIBUTION FOR DRUGRANK TASK
    same_moa_ranks = []
    different_moa_ranks = []

    for i in range(len(experiment_drugs)):
        for j in range(len(all_drugs)):
            if i != j and rank_matrix[i, j] != 0:
                (
                    same_moa_ranks
                    if drug_to_moa[experiment_drugs[i]] == all_moas[j]
                    else different_moa_ranks
                ).append(rank_matrix[i, j])

    plot_moa_distributions(
        same_moa_ranks,
        different_moa_ranks,
        os.path.join(ranking_directory, "figures", f"moa_analysis_{cell_line}.png"),
    )

    # PLOT 3 - BINNED TRUE DRUG RANKS FROM TOP 50 FOR DRUGRANK TASK
    plot_binned_true_drug_ranks(
        rank_matrix,
        experiment_drugs,
        all_drugs,
        os.path.join(
            ranking_directory, "figures", f"binned_true_drug_ranks_{cell_line}.png"
        ),
    )


def analyze_moa_rankings(ranking_directory, drugs_file, cell_line):
    drug_metadata = pd.read_csv(drugs_file)
    all_drugs = drug_metadata["drug"].tolist()
    all_moas = drug_metadata["moa-fine"].tolist()

    unique_moas = sorted(list(set([moa for moa in all_moas if moa != "unclear"])))

    rank_dictionary = combine_rankings(
        [
            os.path.join(ranking_directory, f)
            for f in os.listdir(ranking_directory)
            if f.endswith(".csv")
        ],
        "moarank",
    )

    rank_dictionary = {
        drug: ranks for drug, ranks in rank_dictionary.items() if drug in all_drugs
    }

    experiment_drugs = list(rank_dictionary.keys())

    rank_matrix = np.zeros((len(experiment_drugs), len(unique_moas)))

    for i, drug in enumerate(experiment_drugs):
        for rank, ranked_moa in enumerate(rank_dictionary[drug]):
            if ranked_moa in unique_moas:
                rank_matrix[i, unique_moas.index(ranked_moa)] = rank + 1

    moa_to_drugs = {}
    for drug in experiment_drugs:
        moa = all_moas[all_drugs.index(drug)]
        if moa != "unclear":
            moa_to_drugs.setdefault(moa, []).append(drug)

    # PLOT 1 - HEATMAP OF RANK MATRIX FOR MOARANK TASK
    plot_rank_matrix_heatmap(
        rank_matrix,
        experiment_drugs,
        unique_moas,
        os.path.join(
            ranking_directory, "figures", f"rank_matrix_heatmap_{cell_line}.png"
        ),
    )

    # PLOT 2 - MOA RANKING PERFORMANCE BY DRUG CLASS
    plot_moa_ranking_performance_by_class(
        rank_matrix,
        experiment_drugs,
        unique_moas,
        moa_to_drugs,
        os.path.join(
            ranking_directory, "figures", f"moa_performance_by_class_{cell_line}.png"
        ),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="analyze drug ranking & moa ranking outputs and visualize results"
    )
    parser.add_argument(
        "-r",
        "--ranking_directory",
        type=str,
        help="directory containing ranking csv files (either drugrank or moarank)",
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
        default="None",
        help="cell line to analyze, else omit if analyzing across all cell lines (defaults to 'None')",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()

    ranking_directory = arguments.ranking_directory
    drugs_file = arguments.drugs_file
    cell_line = arguments.cell_line

    os.makedirs(os.path.join(ranking_directory, "figures"), exist_ok=True)

    if os.path.basename(os.path.normpath(ranking_directory)) == "drugrank":
        analyze_drug_rankings(ranking_directory, drugs_file, cell_line)
    elif os.path.basename(os.path.normpath(ranking_directory)) == "moarank":
        analyze_moa_rankings(ranking_directory, drugs_file, cell_line)
    else:
        raise ValueError(f"invalid ranking directory: {ranking_directory}")
