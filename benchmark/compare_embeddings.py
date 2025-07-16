import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pathlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_summary(summary):
    model = get_model()
    embedding = model.encode(summary, convert_to_numpy = True)
    return embedding

def compare_embeddings(summary_dir, drugs_csv, output_dir=None):
    summary_dir = pathlib.Path(summary_dir)
    drugs_csv = pathlib.Path(drugs_csv)
    if output_dir is None:
        output_dir = pathlib.Path(".")
    else:
        output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(drugs_csv)
    drugs = df["drug"].tolist()
    moas = df["moa-fine"].tolist()

    summaries = []
    missing = []
    for drug in drugs:
        summary_path = summary_dir / f"{drug}.txt"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summaries.append(f.read())
        else:
            summaries.append("")
            missing.append(drug)
    if missing:
        print(f"[compare_embeddings] Warning: Missing summaries for {len(missing)} drugs: {missing}")

    embeddings = np.stack([embed_summary(summary) for summary in summaries])

    similarity_matrix = cosine_similarity(embeddings)
    
    same_moa = []
    different_moa = []

    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            (same_moa if moas[i] == moas[j] else different_moa).append(similarity_matrix[i, j])

    # --- distribution of cosine similarity scores for same vs. different moa's ---

    fig, ax = plt.subplots(figsize = (8, 5))
    sns.histplot(same_moa, color = "blue", label = "same moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)
    sns.histplot(different_moa, color = "red", label = "different moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)

    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    
    fig.savefig(output_dir / "similarity_distribution.png")

    # --- visualization of embedding space for the sentence transformer ---

    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 30)
    tsne_embeddings = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize = (10, 7))

    unique_moas = list(sorted(set(moas)))
    palette = sns.color_palette("hls", len(unique_moas))
    color_map = {moa: f"#{int(palette[i][0]*255):02x}{int(palette[i][1]*255):02x}{int(palette[i][2]*255):02x}" for i, moa in enumerate(unique_moas)}

    for moa in unique_moas:
        indices = [i for i, m in enumerate(moas) if m == moa]
        ax.scatter(tsne_embeddings[indices, 0], 
                   tsne_embeddings[indices, 1], 
                   label = moa, c = color_map[moa], alpha = 0.7, s = 30)
    
    ax.set_xlabel("tsne 1")
    ax.set_ylabel("tsne 2")
    ax.legend()

    ax.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")
    fig.tight_layout()

    fig.savefig(output_dir / "tsne_embeddings.png")