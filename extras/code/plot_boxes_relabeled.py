from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# === Load Excel file ===
file_path = Path("/scratch/radv/share/glioseg/new_run/VALIDATION/metrics_boxplot.xlsx")

multiclass_df = pd.read_excel(
    file_path, sheet_name="Multiclass dice score", engine="openpyxl", index_col=0
)

# === Rename models ===
rename_map = {
    "nnUNet task 001-082": "nnUNet",
    "nnUNet task 001-082_relabeled": "nnUNet rlb",
    "nnUNet task 500": "BraTS '21",
    "nnUNet task 500_relabeled": "BraTS '21 rlb",
    "HDglio": "HD-GLIO",
    "HDglio_relabeled": "HD-GLIO rlb",
    "DeepSCAN": "DeepScan",
    "DeepSCAN_relabeled": "DeepScan rlb",
    "FETS": "FeTS",
    "FETS_relabeled": "FeTS rlb",
    "STAPLE": "STAPLE",
    "STAPLE_relabeled": "STAPLE rlb",
    "SIMPLE": "SIMPLE",
    "SIMPLE_relabeled": "SIMPLE rlb",
    "MV": "MajVote",
    "MV_relabeled": "MajVote rlb",
}

# === Reshape wide → long ===
mc_clean = (
    multiclass_df[list(rename_map.keys())]
    .rename(columns=rename_map)
    .melt(var_name="Model", value_name="Dice")
)

# Add columns for relabeled vs original
mc_clean["Relabeled"] = mc_clean["Model"].apply(lambda x: "Relabeled" if "rlb" in x else "Original")
mc_clean["BaseModel"] = mc_clean["Model"].apply(lambda x: x.replace(" rlb", ""))

# === Order models ===
ensembles = ["STAPLE", "SIMPLE", "MajVote"]
non_ensembles = [m for m in mc_clean["BaseModel"].unique() if m not in ensembles]

# Order non-ensemble models by relabeled mean
non_ens_order = (
    mc_clean[mc_clean["BaseModel"].isin(non_ensembles) & (mc_clean["Relabeled"] == "Relabeled")]
    .groupby("BaseModel")["Dice"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# Order ensembles by relabeled mean
ens_order = (
    mc_clean[mc_clean["BaseModel"].isin(ensembles) & (mc_clean["Relabeled"] == "Relabeled")]
    .groupby("BaseModel")["Dice"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# Combine final order
ordered_models = non_ens_order + ens_order

# === Define color palette ===
palette = {
    "Original": (86 / 255, 180 / 255, 233 / 255),  # blue
    "Relabeled": (230 / 255, 159 / 255, 0 / 255),  # orange
}

# === Plot ===
plt.figure(figsize=(16, 7))
ax = sns.boxplot(
    data=mc_clean,
    x="BaseModel",
    y="Dice",
    hue="Relabeled",
    palette=palette,
    order=ordered_models,
    showcaps=True,
    fliersize=5,
    linewidth=1.2,
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    flierprops={
        "marker": "D",
        "markerfacecolor": "gray",
        "markeredgecolor": "black",
        "markersize": 5,
    },
)

# === Add vertical black line to separate individual models vs ensembles ===
num_individual_models = len(non_ens_order)
ax.axvline(num_individual_models - 0.5, color="black", linestyle="-", linewidth=1.2)

# === Axis labels and legend ===
plt.xticks(rotation=0, fontsize=9)
plt.xlabel("")
plt.ylabel("Multi-class Dice Score", fontsize=12, weight="bold")
plt.legend(title="Output", title_fontsize=9, fontsize=9, loc="lower left")

# === Add per-model mean ± std below boxes ===
stats = mc_clean.groupby(["BaseModel", "Relabeled"])["Dice"].agg(["mean", "std"]).reset_index()

for i, model in enumerate(ordered_models):
    for relab_state in ["Original", "Relabeled"]:
        subset = stats[(stats["BaseModel"] == model) & (stats["Relabeled"] == relab_state)]
        if not subset.empty:
            offset = -0.09 if relab_state == "Original" else -0.16
            color = palette[relab_state]
            ax.text(
                i,
                offset,
                f"{subset['mean'].values[0]:.2f} ± {subset['std'].values[0]:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                weight="bold",
            )

plt.tight_layout()

# === Save figure ===
output_path = Path("/scratch/radv/share/glioseg/new_run/VALIDATION/dice_boxplots_multiclass.svg")
plt.savefig(output_path, bbox_inches="tight")
print(f"✅ Saved plot to: {output_path}")
