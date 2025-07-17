import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pathlib
from tahoe_agent.logging_config import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger()

def generate_summary_embedding(summary):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(summary)

def analyze_embeddings(summary_directory, drugs_file):
    logger.info("[analyze_embeddings] analyzing embeddings on generated summaries")
    
    summary_directory = pathlib.Path(summary_directory)
    drugs_file = pathlib.Path(drugs_file)
    
    drugs_data = pd.read_csv(drugs_file)
    drugs = drugs_data["drug"].tolist()
    moas = drugs_data["moa-fine"].tolist()

    available_summaries = []
    available_drugs = []
    available_moas = []
    
    for drug, moa in zip(drugs, moas):
        summary_path = summary_directory / f"{drug}_summary.txt"
        print(summary_path)
        if summary_path.exists():
            with open(summary_path, "r") as f:
                available_summaries.append(f.read())
                available_drugs.append(drug)
                available_moas.append(moa)

    logger.info(f"[analyze_embeddings] found {len(available_drugs)} drugs with summaries")    
    logger.info("[analyze_embeddings] computing summary embeddings")

    embeddings = np.stack([generate_summary_embedding(summary) for summary in available_summaries])

    similarity_matrix = cosine_similarity(embeddings)
    
    same_moa = []
    different_moa = []

    for i in range(len(available_drugs)):
        for j in range(i + 1, len(available_drugs)):
            (same_moa if available_moas[i] == available_moas[j] else different_moa).append(similarity_matrix[i, j])

    logger.info(f"[analyze_embeddings] found {len(same_moa)} same moa pairs and {len(different_moa)} different moa pairs")

    # --- PLOT 1: distribution of cosine similarity scores for same vs. different moa's ---
    logger.info("[analyze_embeddings] creating cosine similarity distribution plot")
    fig, ax = plt.subplots(figsize = (8, 5))
    sns.histplot(same_moa, color = "blue", label = "same moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)
    sns.histplot(different_moa, color = "red", label = "different moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)

    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    
    similarity_plot_path = summary_directory / "similarity_distribution.png"
    fig.savefig(similarity_plot_path)

    logger.info(f"[analyze_embeddings] saved similarity distribution plot to: {similarity_plot_path}")

    # --- PLOT 2: visualization of embedding space for the sentence transformer ---
    logger.info("[analyze_embeddings] creating t-SNE visualization")
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = min(len(available_drugs) - 1, 30))
    tsne_embeddings = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize = (10, 7))

    unique_moas = list(sorted(set(available_moas)))
    palette = sns.color_palette("hls", len(unique_moas))
    color_map = {moa: f"#{int(palette[i][0] * 255):02x}{int(palette[i][1] * 255):02x}{int(palette[i][2] * 255):02x}" for i, moa in enumerate(unique_moas)}

    for moa in unique_moas:
        indices = [i for i, m in enumerate(available_moas) if m == moa]
        ax.scatter(tsne_embeddings[indices, 0], 
                   tsne_embeddings[indices, 1], 
                   label = moa, c = color_map[moa], alpha = 0.7, s = 30)
    
    ax.set_xlabel("t-sne 1")
    ax.set_ylabel("t-sne 2")
    ax.legend()

    ax.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")
    fig.tight_layout()

    tsne_plot_path = summary_directory / "tsne_embeddings.png"
    fig.savefig(tsne_plot_path)

    logger.info(f"[analyze_embeddings] saved t-SNE plot to: {tsne_plot_path}")