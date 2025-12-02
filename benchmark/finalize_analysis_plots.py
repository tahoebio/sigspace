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
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(22, 18))

    sorted_experiment_drugs = sorted(experiment_drugs)
    sorted_reference = sorted(reference)

    sorted_experiment_indices = [
        experiment_drugs.index(drug) for drug in sorted_experiment_drugs
    ]
    sorted_reference_indices = [reference.index(drug) for drug in sorted_reference]

    sorted_rank_matrix = rank_matrix[
        np.ix_(sorted_experiment_indices, sorted_reference_indices)
    ]

    mask = sorted_rank_matrix == 0
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e0e0e0")

    rank_matrix_masked = np.ma.masked_where(mask, sorted_rank_matrix)

    nonzero = sorted_rank_matrix[~mask]
    vmin = np.min(nonzero) if nonzero.size > 0 else 1
    vmax = np.max(nonzero) if nonzero.size > 0 else 1

    sns.heatmap(
        rank_matrix_masked,
        annot=False,
        cmap=cmap,
        xticklabels=sorted_reference,
        yticklabels=sorted_experiment_drugs,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "drug rank"},
        square=True,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.1,
        linecolor="white",
    )

    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("ranked drugs", fontsize=12, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("true drugs", fontsize=12, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "drug ranking matrix heatmap",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
        pad=20,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=2)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#BDC3C7")
    ax.set_axisbelow(True)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, colors="#2C3E50")
    cbar.set_label("drug rank", fontsize=12, fontweight="bold", color="#2C3E50")

    fig.subplots_adjust(bottom=0.3, right=0.88, top=0.92, left=0.2)
    fig.savefig(save_file, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def plot_moa_distributions(same_moa_ranks, different_moa_ranks, save_file):
    same_moa_mean = np.mean(np.array(same_moa_ranks))
    different_moa_mean = np.mean(np.array(different_moa_ranks))

    plt.style.use("default")
    fig, (ax_one, ax_two) = plt.subplots(1, 2, figsize=(16, 7))

    same_color = "#2E86AB"
    different_color = "#A23B72"

    sns.kdeplot(
        data=same_moa_ranks,
        ax=ax_one,
        label="same moa",
        color=same_color,
        linewidth=2.5,
        alpha=0.8,
    )
    sns.kdeplot(
        data=different_moa_ranks,
        ax=ax_one,
        label="different moa",
        color=different_color,
        linewidth=2.5,
        alpha=0.8,
    )

    ax_one.axvline(
        same_moa_mean, color=same_color, linestyle="--", alpha=0.7, linewidth=2
    )
    ax_one.axvline(
        different_moa_mean,
        color=different_color,
        linestyle="--",
        alpha=0.7,
        linewidth=2,
    )

    ax_one.set_xlabel("drug rank", fontsize=12, fontweight="bold", color="#2C3E50")
    ax_one.set_ylabel("density", fontsize=12, fontweight="bold", color="#2C3E50")
    ax_one.set_title(
        "rank distributions: same vs. different moa",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
        pad=20,
    )
    ax_one.tick_params(axis="both", which="major", labelsize=10, colors="#2C3E50")
    ax_one.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#BDC3C7")
    ax_one.set_axisbelow(True)

    legend1 = ax_one.legend(
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor="#BDC3C7",
        loc="upper right",
    )
    legend1.get_title().set_fontweight("bold")
    legend1.get_title().set_color("#2C3E50")

    sns.ecdfplot(
        data=same_moa_ranks,
        ax=ax_two,
        label="same moa",
        color=same_color,
        linewidth=2.5,
    )
    sns.ecdfplot(
        data=different_moa_ranks,
        ax=ax_two,
        label="different moa",
        color=different_color,
        linewidth=2.5,
    )

    ax_two.set_xlabel("drug rank", fontsize=12, fontweight="bold", color="#2C3E50")
    ax_two.set_ylabel(
        "cumulative probability", fontsize=12, fontweight="bold", color="#2C3E50"
    )
    ax_two.set_title(
        "cumulative rank distributions",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
        pad=20,
    )
    ax_two.tick_params(axis="both", which="major", labelsize=10, colors="#2C3E50")
    ax_two.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#BDC3C7")
    ax_two.set_axisbelow(True)
    ax_two.set_ylim(0, 1)

    legend2 = ax_two.legend(
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor="#BDC3C7",
        loc="lower right",
    )
    legend2.get_title().set_fontweight("bold")
    legend2.get_title().set_color("#2C3E50")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.08, right=0.95, bottom=0.1)
    fig.savefig(save_file, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def plot_binned_true_drug_ranks(
    rank_matrix, experiment_drugs, all_drugs, save_file, k=50
):
    num_experiments = len(experiment_drugs)
    num_drugs = len(all_drugs)

    bin_specs = [
        (1, 10, "1-10"),
        (11, 30, "11-30"),
        (31, 50, "31-50"),
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
        expected_proportions[label] = max(0, min(high, k) - low + 1) / num_drugs
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

    def reciprocal(r):
        return 0.0 if r <= 0 else 1.0 / float(r)

    def rr_at_k(r, k_cut):
        return 1.0 / float(r) if (r > 0 and r <= k_cut) else 0.0

    def hit_at_k(r, k_cut):
        return 1.0 if (r > 0 and r <= k_cut) else 0.0

    def harmonic(n: int):
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=float)))

    rr_all = np.array([reciprocal(r) for r in true_ranks])
    rr20 = np.array([rr_at_k(r, 20) for r in true_ranks])
    rr50 = np.array([rr_at_k(r, 50) for r in true_ranks])
    hit20 = np.array([hit_at_k(r, 20) for r in true_ranks])
    hit50 = np.array([hit_at_k(r, 50) for r in true_ranks])

    obs_MRR = float(rr_all.mean()) if num_experiments else float("nan")
    obs_MRR20 = float(rr20.mean()) if num_experiments else float("nan")
    obs_MRR50 = float(rr50.mean()) if num_experiments else float("nan")
    obs_H20 = float(hit20.mean() * 100.0) if num_experiments else float("nan")
    obs_H50 = float(hit50.mean() * 100.0) if num_experiments else float("nan")

    exp_MRR = harmonic(num_drugs) / num_drugs
    exp_MRR20 = harmonic(min(20, num_drugs)) / num_drugs
    exp_MRR50 = harmonic(min(50, num_drugs)) / num_drugs
    exp_H20 = (min(20, num_drugs) / num_drugs) * 100.0
    exp_H50 = (min(50, num_drugs) / num_drugs) * 100.0

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 8))

    observed_color = "#2E86AB"
    baseline_color = "#A23B72"
    colors = [observed_color, baseline_color]

    sns.barplot(
        data=df,
        x="bin",
        y="proportion",
        hue="type",
        ax=ax,
        palette=colors,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.85,
    )

    ax.set_xlabel(
        "true drug rank (binned)", fontsize=14, fontweight="bold", color="#2C3E50"
    )
    ax.set_ylabel(
        "proportion of experiments", fontsize=14, fontweight="bold", color="#2C3E50"
    )
    ax.set_title(
        f"binned true drug rank analysis (top {k} reported)\nobserved vs. random baseline performance",
        fontsize=16,
        fontweight="bold",
        color="#2C3E50",
        pad=20,
    )

    max_prop = df["proportion"].max()
    ax.set_ylim(0, max(max_prop * 1.25, 0.15))

    ax.tick_params(axis="both", which="major", labelsize=12, colors="#2C3E50")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
        label.set_fontweight("medium")

    legend = ax.legend(
        title_fontsize=12,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor="#BDC3C7",
        loc="upper right",
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#2C3E50")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.4f",
            fontsize=10,
            fontweight="bold",
            color="#2C3E50",
            padding=3,
        )

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#BDC3C7")
    ax.set_axisbelow(True)

    metrics_text = f"""
• MRR: {obs_MRR:.3f} (baseline: {exp_MRR:.3f})
• MRR@20: {obs_MRR20:.3f} (baseline: {exp_MRR20:.3f})
• MRR@50: {obs_MRR50:.3f} (baseline: {exp_MRR50:.3f})
• Hits@20: {obs_H20:.4f}% (baseline: {exp_H20:.4f}%)
• Hits@50: {obs_H50:.4f}% (baseline: {exp_H50:.4f}%)
    """

    ax.text(
        0.02,
        0.98,
        metrics_text.strip(),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.95,
            edgecolor="#2E86AB",
            linewidth=1.5,
        ),
        color="#2C3E50",
        fontweight="medium",
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.85, left=0.08, right=0.95, bottom=0.1)
    fig.savefig(save_file, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

    return {
        "N": num_experiments,
        "M": num_drugs,
        "MRR": obs_MRR,
        "MRR@20": obs_MRR20,
        "MRR@50": obs_MRR50,
        "Hits@20": obs_H20,
        "Hits@50": obs_H50,
        "exp_MRR": exp_MRR,
        "exp_MRR@20": exp_MRR20,
        "exp_MRR@50": exp_MRR50,
        "exp_Hits@20": exp_H20,
        "exp_Hits@50": exp_H50,
    }


def plot_moa_ranking_performance_by_class(
    rank_matrix, experiment_drugs, unique_moas, moa_to_drugs, save_file
):
    moa_performance = {}
    all_true_ranks = []

    for moa, drugs in moa_to_drugs.items():
        ranks = []
        for drug in drugs:
            rank = rank_matrix[experiment_drugs.index(drug), unique_moas.index(moa)]
            if rank > 0:
                ranks.append(rank)

            all_true_ranks.append(rank)
        if ranks:
            moa_performance[moa] = {
                "mean rank": np.mean(ranks),
                "std rank": np.std(ranks),
                "ranks": ranks,
            }

    moa_data = [(moa, moa_performance[moa]) for moa in moa_performance.keys()]
    moa_data.sort(key=lambda x: len(x[1]["ranks"]), reverse=True)

    moa_labels = [item[0] for item in moa_data]
    mean_ranks = [item[1]["mean rank"] for item in moa_data]
    counts = [len(item[1]["ranks"]) for item in moa_data]

    def reciprocal(r):
        return 0.0 if r <= 0 else 1.0 / float(r)

    def rr_at_k(r, k):
        return 1.0 / float(r) if (r > 0 and r <= k) else 0.0

    def hit_at_k(r, k):
        return 1.0 if (r > 0 and r <= k) else 0.0

    rr = np.array([reciprocal(r) for r in all_true_ranks])
    observed_metrics = {
        "MRR": float(rr.mean()),
        "MRR@1": float(np.mean([rr_at_k(r, 1) for r in all_true_ranks])),
        "MRR@5": float(np.mean([rr_at_k(r, 5) for r in all_true_ranks])),
        "MRR@10": float(np.mean([rr_at_k(r, 10) for r in all_true_ranks])),
        "Hits@1": float(np.mean([hit_at_k(r, 1) for r in all_true_ranks]) * 100.0),
        "Hits@5": float(np.mean([hit_at_k(r, 5) for r in all_true_ranks]) * 100.0),
        "Hits@10": float(np.mean([hit_at_k(r, 10) for r in all_true_ranks]) * 100.0),
    }

    M = len(unique_moas)

    def H(n):
        return sum(1.0 / i for i in range(1, n + 1))

    expected_metrics = {
        "MRR": H(M) / M,
        "MRR@1": H(1) / M,
        "MRR@5": H(5) / M,
        "MRR@10": H(10) / M,
        "Hits@1": (1 / M) * 100.0,
        "Hits@5": (5 / M) * 100.0,
        "Hits@10": (10 / M) * 100.0,
    }

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(16, 8))

    baseline = (M + 1) / 2.0

    colors = ["#FF6B35" if rank < baseline else "#2E86AB" for rank in mean_ranks]
    edge_colors = ["#D63031" if rank < baseline else "#1A5F7A" for rank in mean_ranks]

    bars = ax.bar(
        range(len(moa_labels)),
        mean_ranks,
        alpha=0.85,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.5,
    )

    ax.set_xlabel(
        "drug class (sorted by sample size)",
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.set_ylabel(
        "mean rank of true moa", fontsize=12, fontweight="bold", color="#2C3E50"
    )
    ax.set_title(
        "moa ranking performance by drug class",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
        pad=20,
    )
    ax.set_xticks(range(len(moa_labels)))
    ax.set_xticklabels(
        [label for label in moa_labels], rotation=45, ha="right", fontsize=10
    )

    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"n={count}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            rotation=90,
        )

    ax.axhline(baseline, linestyle="--", linewidth=2, color="#2C3E50", alpha=0.8)
    ymin, ymax = ax.get_ylim()
    if baseline > 0.98 * ymax:
        ax.set_ylim(ymin, baseline * 1.05)

    ax.text(
        0.995,
        (baseline - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
        f"random baseline = {baseline:.1f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=10,
        color="#2C3E50",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor="#2C3E50",
            linewidth=1,
        ),
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#FF6B35", edgecolor="#D63031", label="above baseline (better)"
        ),
        Patch(facecolor="#2E86AB", edgecolor="#1A5F7A", label="below baseline (worse)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#BDC3C7")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=10, colors="#2C3E50")

    lines = []
    for k in ["MRR", "MRR@1", "MRR@5", "MRR@10", "Hits@1", "Hits@5", "Hits@10"]:
        obs = observed_metrics[k]
        exp = expected_metrics[k]
        if "Hits" in k:
            lines.append(f"{k}: {obs:.4f}% (exp {exp:.4f}%)")
        else:
            lines.append(f"{k}: {obs:.3f} (exp {exp:.3f})")

    metrics_txt = "\n".join(lines)
    ax.annotate(
        metrics_txt,
        xy=(0.995, 0.995),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, linewidth=0),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.08, right=0.95, bottom=0.15)
    fig.savefig(save_file, bbox_inches="tight", dpi=300, facecolor="white")
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
        ranked_moas = rank_dictionary[drug]
        num_ranked = len(ranked_moas)
        num_total = len(unique_moas)

        if num_ranked < num_total:
            unranked_positions = list(range(num_ranked + 1, num_total + 1))
            imputed_rank = np.mean(unranked_positions)
        else:
            imputed_rank = 0

        for rank, ranked_moa in enumerate(ranked_moas):
            if ranked_moa in unique_moas:
                rank_matrix[i, unique_moas.index(ranked_moa)] = rank + 1

        for moa in unique_moas:
            if moa not in ranked_moas:
                rank_matrix[i, unique_moas.index(moa)] = imputed_rank

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
