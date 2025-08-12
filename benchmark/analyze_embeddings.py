import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_similarity_distribution(same_moa, different_moa, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        same_moa,
        color="blue",
        label="same moa",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
        ax=ax,
    )
    sns.histplot(
        different_moa,
        color="red",
        label="different moa",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tsne_embeddings(tsne_embeddings, available_moas, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_moas = list(sorted(set(available_moas)))
    palette = sns.color_palette("hls", len(unique_moas))
    color_map = {
        moa: f"#{int(palette[i][0] * 255):02x}{int(palette[i][1] * 255):02x}{int(palette[i][2] * 255):02x}"
        for i, moa in enumerate(unique_moas)
    }
    for moa in unique_moas:
        indices = [i for i, m in enumerate(available_moas) if m == moa]
        ax.scatter(
            tsne_embeddings[indices, 0],
            tsne_embeddings[indices, 1],
            label=moa,
            c=color_map[moa],
            alpha=0.7,
            s=30,
        )
    ax.set_xlabel("t-sne 1")
    ax.set_ylabel("t-sne 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_embedding_vs_target_jaccard(embeddings, drugs, target_map, output_path):
    pairs = list(combinations(range(len(drugs)), 2))
    cos_sims = []
    jaccards = []
    for i, j in pairs:
        set_i = target_map[drugs[i]]
        set_j = target_map[drugs[j]]
        if not set_i and not set_j:
            continue
        jac = len(set_i & set_j) / len(set_i | set_j) if (set_i | set_j) else 0.0
        sim = cosine_similarity(embeddings[i : i + 1], embeddings[j : j + 1])[0, 0]
        cos_sims.append(sim)
        jaccards.append(jac)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(jaccards, cos_sims, alpha=0.3, s=10)
    ax.set_xlabel("gene target jaccard similarity")
    ax.set_ylabel("embedding cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tsne_colored_by_target(tsne_embeddings, drugs, target_map, gene, output_path):
    has_target = [gene in target_map[d] for d in drugs]
    (pathlib.Path(output_path).parent).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=has_target,
        cmap="coolwarm",
        alpha=0.7,
        s=30,
    )
    ax.set_xlabel("t-sne 1")
    ax.set_ylabel("t-sne 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_target_mean_similarity_heatmap(
    embeddings, drugs, target_map, output_path, min_drugs=2, max_targets=50
):
    target_to_drugs = {}
    for d, targets in target_map.items():
        for t in targets:
            if t:
                target_to_drugs.setdefault(t, []).append(d)
    filtered = {t: ds for t, ds in target_to_drugs.items() if len(ds) >= min_drugs}
    filtered = dict(list(filtered.items())[:max_targets])
    means = []
    labels = []
    for t, ds in filtered.items():
        idxs = [drugs.index(d) for d in ds if d in drugs]
        if len(idxs) < 2:
            continue
        sims = []
        for i, j in combinations(idxs, 2):
            sim = cosine_similarity(embeddings[i : i + 1], embeddings[j : j + 1])[0, 0]
            sims.append(sim)
        if sims:
            means.append(np.mean(sims))
            labels.append(t)
    (pathlib.Path(output_path).parent).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(20, 0.4 * len(labels)), 2))
    sns.heatmap(
        np.array(means)[None, :],
        annot=False,
        cmap="RdBu_r",
        cbar=True,
        ax=ax,
        xticklabels=[label.lower() for label in labels],
        yticklabels=[],
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("mes")
    ax.set_xlabel("gene target")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_embeddings(summary_directory, drugs_file, cell_line=None):
    summary_directory = pathlib.Path(summary_directory)
    drugs_file = pathlib.Path(drugs_file)
    drugs_data = pd.read_csv(drugs_file)
    drugs = drugs_data["drug"].tolist()
    moas = drugs_data["moa-fine"].tolist()
    target_map = {
        row["drug"]: set(
            [t.strip() for t in str(row["targets"]).split(",") if t.strip()]
        )
        if pd.notna(row["targets"])
        else set()
        for _, row in drugs_data.iterrows()
    }
    available_embeddings = []
    available_drugs = []
    available_moas = []
    for drug, moa in zip(drugs, moas):
        embedding_path = None
        for f in summary_directory.glob("embedding*"):
            fname = f.name
            if drug in fname and (cell_line is None or (str(cell_line) in fname)):
                embedding_path = f
                break
        if embedding_path is not None and embedding_path.exists():
            embedding = np.load(embedding_path)["embedding"]
            available_embeddings.append(embedding)
            available_drugs.append(drug)
            available_moas.append(moa)
    embeddings = np.stack(available_embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    same_moa = []
    different_moa = []
    for i in range(len(available_drugs)):
        for j in range(i + 1, len(available_drugs)):
            (
                same_moa if available_moas[i] == available_moas[j] else different_moa
            ).append(similarity_matrix[i, j])

    (summary_directory / "figures").mkdir(parents=True, exist_ok=True)
    similarity_plot_path = summary_directory / "figures" / "similarity_distribution.png"
    plot_similarity_distribution(same_moa, different_moa, similarity_plot_path)

    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(len(available_drugs) - 1, 30)
    )
    tsne_embeddings = tsne.fit_transform(embeddings)
    tsne_plot_path = summary_directory / "figures" / "tsne_embeddings.png"
    plot_tsne_embeddings(tsne_embeddings, available_moas, tsne_plot_path)

    jaccard_plot_path = (
        summary_directory / "figures" / "embedding_vs_target_jaccard.png"
    )
    plot_embedding_vs_target_jaccard(
        embeddings, available_drugs, target_map, jaccard_plot_path
    )

    example_gene = "EGFR"
    drugs_with_gene = [d for d in available_drugs if example_gene in target_map[d]]
    if len(drugs_with_gene) >= 2:
        tsne_gene_plot_path = (
            summary_directory / "figures" / f"tsne_colored_by_{example_gene}.png"
        )
        plot_tsne_colored_by_target(
            tsne_embeddings,
            available_drugs,
            target_map,
            example_gene,
            tsne_gene_plot_path,
        )

    heatmap_path = summary_directory / "figures" / "target_mean_similarity_heatmap.png"
    plot_target_mean_similarity_heatmap(
        embeddings, available_drugs, target_map, heatmap_path
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="analyze summary embeddings and visualize results"
    )
    parser.add_argument(
        "-c",
        "--cell_line",
        type=str,
        help="cell line to analyze, else omit if analyzing across all cell lines",
    )
    parser.add_argument(
        "-d",
        "--drugs_file",
        type=str,
        help="path to the file containing drug information",
    )
    parser.add_argument(
        "-s",
        "--summary_directory",
        type=str,
        help="directory to save summary outputs and plots",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    cell_line = arguments.cell_line
    drugs_file = arguments.drugs_file
    summary_directory = arguments.summary_directory
    analyze_embeddings(summary_directory, drugs_file, cell_line)
