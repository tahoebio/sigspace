import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "all-MiniLM-L6-v2"
DRUG_METADATA = "/Users/rohit/Desktop/tahoe_agent/benchmark/drug_metadata.csv"

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

def main():
    df = pd.read_csv(DRUG_METADATA)

    drugs = df["drug"].tolist()
    moas = df["moa-fine"].tolist()

    summaries = [f"This is a summary for {drug}. It describes the effect of the drug on gene expression and biological pathways. Mechanism: {moa}." for drug, moa in zip(drugs, moas)]

    embeddings = np.stack([embed_summary(summary) for summary in summaries])

    similarity_matrix = cosine_similarity(embeddings)
    
    same_moa = []
    different_moa = []

    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            (same_moa if moas[i] == moas[j] else different_moa).append(similarity_matrix[i, j])

    fig, ax = plt.subplots(figsize = (8, 5))
    sns.histplot(same_moa, color = "blue", label = "same moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)
    sns.histplot(different_moa, color = "red", label = "different moa", kde = True, stat = "density", bins = 30, alpha = 0.6, ax = ax)

    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    
    fig.savefig("similarity_distribution.png")

    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 30)
    tsne_embeddings = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize = (10, 7))

    unique_moas = list(sorted(set(moas)))
    palette = sns.color_palette("hls", len(unique_moas))
    color_map = {moa: palette[i] for i, moa in enumerate(unique_moas)}

    for moa in unique_moas:
        indices = [i for i, m in enumerate(moas) if m == moa]
        ax.scatter(tsne_embeddings[indices, 0], 
                   tsne_embeddings[indices, 1], 
                   label = moa, color = color_map[moa], alpha = 0.7, s = 30)
    
    ax.set_xlabel("tsne 1")
    ax.set_ylabel("tsne 2")
    ax.legend()

    ax.legend(bbox_to_anchor = (1.05, 1), loc = "upper left", borderaxespad = 0.)
    fig.tight_layout()

    fig.savefig("tsne_embeddings.png")

if __name__ == "__main__":
    main()