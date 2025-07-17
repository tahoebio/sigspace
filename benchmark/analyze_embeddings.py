import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def analyze_embeddings(summary_directory, drugs_file, cell_line=None):
    summary_directory = pathlib.Path(summary_directory)
    drugs_file = pathlib.Path(drugs_file)

    drugs_data = pd.read_csv(drugs_file)
    drugs = drugs_data["drug"].tolist()
    moas = drugs_data["moa-fine"].tolist()

    available_embeddings = []
    available_drugs = []
    available_moas = []

    for drug, moa in zip(drugs, moas):
        embedding_path = summary_directory / f"embedding_{drug}_{cell_line}.npz"

        if embedding_path.exists():
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

    # --- PLOT 1: distribution of cosine similarity scores for same vs. different moa's ---
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

    similarity_plot_path = summary_directory / "similarity_distribution.png"
    fig.savefig(similarity_plot_path)

    # --- PLOT 2: visualization of embedding space for the sentence transformer ---
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(len(available_drugs) - 1, 30)
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

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
    ax.legend()

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    tsne_plot_path = summary_directory / "tsne_embeddings.png"
    fig.savefig(tsne_plot_path)


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
