import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def combine_rankings(files):
    truth_to_ranks = {}

    for file in files:
        name = os.path.basename(file)
        drug = name[len("ranking_"):-len(".json")]

        with open(file, "r") as f:
            data = json.load(f)
            truth_to_ranks[drug] = data["ranks"]
    
    return truth_to_ranks

def simulate_outputs(n, drugs, output_directory = "ranks"):
    os.makedirs(output_directory, exist_ok = True)

    for i in range(n):
        ranks = np.random.permutation(np.arange(1, n + 1)).tolist()
        data = {"ranks": [drugs[i - 1] for i in ranks]}

        with open(os.path.join(output_directory, f"ranking_{drugs[i]}.json"), "w") as f:
            json.dump(data, f)

    return [os.path.join(output_directory, f"ranking_{drugs[i]}.json") for i in range(n)]

drug_data = pd.read_csv("benchmark/drug_metadata.csv")
drugs = drug_data["drug"].tolist()
drugs = [drug.replace("/", "or") for drug in drugs]

n = len(drugs)

files = simulate_outputs(n, drugs)
rank_dictionary = combine_rankings(files)
rank_matrix = np.zeros((n, n))

for i, drug in enumerate(drugs):
    for rank, ranked_drug in enumerate(rank_dictionary[drug]):
        j = drugs.index(ranked_drug)
        rank_matrix[i, j] = rank + 1

moa_map = {row["drug"].replace("/", "or"): row["moa-fine"] for _, row in drug_data.iterrows()}

# --- heatmap of the ranking matrix ---

fig, ax = plt.subplots(figsize = (20, 16))

sns.heatmap(rank_matrix, annot = False, cmap = "RdBu_r", xticklabels = drugs, yticklabels = drugs, ax = ax,
                cbar_kws = {"shrink": 0.5, "label": "ranks"}, square = True)

ax.tick_params(left = False, bottom = False) # type: ignore
ax.set_xlabel("ranked drug")
ax.set_ylabel("true drug")

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "center", fontsize = 2) # type: ignore
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, ha = "right", fontsize = 2) # type: ignore

fig.subplots_adjust(bottom = 0.3, right = 0.85, top = 0.95, left = 0.2) # type: ignore

fig.savefig("rank_matrix_heatmap.png", bbox_inches = "tight", dpi = 300)
plt.close(fig)

# --- histogram of the "surprise" metric ---

surprise = []
for i in range(n):
    surprise.append(rank_matrix[i, i])

fig, ax = plt.subplots(figsize = (7, 4))

ax.hist(surprise, bins = np.arange(1, n + 2) - 0.5, color = "orange", edgecolor = "black") # type: ignore
ax.set_xlabel("self-rank")
ax.set_ylabel("count")

fig.savefig("surprise_metric_histogram.png", bbox_inches = "tight", dpi = 300)
plt.close(fig)

# --- analyze and visualize moa relationships ---

same_moa_ranks = []
different_moa_ranks = []

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        
        (same_moa_ranks if moa_map[drugs[i]] == moa_map[drugs[j]] else different_moa_ranks).append(rank_matrix[i, j])

same_moa_ranks = np.array(same_moa_ranks)
different_moa_ranks = np.array(different_moa_ranks)

same_moa_mean = np.mean(same_moa_ranks)
different_moa_mean = np.mean(different_moa_ranks)
mean_difference = different_moa_mean - same_moa_mean

fig, (ax_one, ax_two) = plt.subplots(1, 2, figsize = (15, 6))

sns.kdeplot(data = same_moa_ranks, ax = ax_one, label = "same moa", color = "blue") # type: ignore
sns.kdeplot(data = different_moa_ranks, ax = ax_one, label = "different moa", color = "red") # type: ignore

ax_one.axvline(same_moa_mean, color = "blue", linestyle = "--", alpha = 0.5) # type: ignore
ax_one.axvline(different_moa_mean, color = "red", linestyle = "--", alpha = 0.5) # type: ignore
ax_one.text(0.02, 0.98, f"mean difference: {mean_difference:.1f}", transform = ax_one.transAxes, verticalalignment = "top") # type: ignore

ax_one.set_xlabel("rank") # type: ignore
ax_one.set_ylabel("density") # type: ignore
ax_one.legend() # type: ignore

sns.ecdfplot(data = same_moa_ranks, ax = ax_two, label = "same moa", color = "blue")  # type: ignore
sns.ecdfplot(data = different_moa_ranks, ax = ax_two, label = "different moa", color = "red")  # type: ignore

ax_two.set_xlabel("rank") # type: ignore
ax_two.set_ylabel("cumulative probability") # type: ignore
ax_two.set_title("cumulative distribution of ranks") # type: ignore

stats_text = (
    f"same moa:\n"
    f"  mean: {same_moa_mean:.1f}\n"
    f"  median: {np.median(same_moa_ranks):.1f}\n"
    f"different moa:\n"
    f"  mean: {different_moa_mean:.1f}\n"
    f"  median: {np.median(different_moa_ranks):.1f}\n"
    f"mean difference: {mean_difference:.1f}"
)

ax_two.text(1.05, 0.5, stats_text, transform = ax_two.transAxes, verticalalignment = "center", bbox = dict(facecolor = "white", alpha = 0.8)) # type: ignore

fig.tight_layout() # type: ignore
fig.savefig("moa_analysis.png", bbox_inches = "tight", dpi = 300)
plt.close()