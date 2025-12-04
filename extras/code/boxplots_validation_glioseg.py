import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data_dir = "/home/gmosquerarojas/glioseg/additional_validation_metrics.xlsx"

# Read excel file in the sheet "Dice score whole tumor"

df_wt_dsc = pd.read_excel(data_dir, sheet_name="Dice score whole tumor", skiprows=1)

# Take the columns from the first to the last, excluding the last two rows. 
wt_dsc = df_wt_dsc.iloc[:-2, 1:8].values
wt_dsc_names = df_wt_dsc.columns.values[1:8]

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=wt_dsc, palette="Set2", width=0.6)
plt.xticks(range(len(wt_dsc_names)), wt_dsc_names, rotation=45)
plt.title("Whole tumor Dice score: validation set")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.savefig("WT_DSC_BP.png")

# Read excel file in the sheet "Dice score multi class"
df_mc_dsc = pd.read_excel(data_dir, sheet_name="Multiclass dice score", skiprows=1)
# Take the columns from the first to the last, excluding the last two rows.
mc_dsc = df_mc_dsc.iloc[:-2, 1:8].values
mc_dsc_names = df_mc_dsc.columns.values[1:8]
# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=mc_dsc, palette="Set2", width=0.6)
plt.xticks(range(len(mc_dsc_names)), mc_dsc_names, rotation=45)
plt.title("Multi class Dice score: validation set")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.savefig("MC_DSC_BP.png")


# Create a boxplot for the metric value accross all patients 
# # Read the Excel file and skip the method row
# df = pd.read_excel(data_dir, sheet_name="Boxplot", skiprows=1)

# # Force column names to strings
# df.columns = ["Whole_MV", "Whole_STAPLE", "Multi_MV", "Multi_STAPLE"]

# # Melt to long format
# df_long = df.melt(var_name="Group_Method", value_name="Score")

# # Force Group_Method to be string before applying string operations
# df_long["Group_Method"] = df_long["Group_Method"].astype(str)

# # Derive 'Type' and 'Method'
# df_long["Type"] = df_long["Group_Method"].apply(lambda x: "Whole Tumor Dice Score" if "Whole" in x else "Multi Class Dice Score")
# df_long["Method"] = df_long["Group_Method"].apply(lambda x: "Majority Voting" if "MV" in x else "STAPLE")

# # Drop NaNs if they exist
# df_long = df_long.dropna(subset=["Score"])

# # Plot
# plt.figure(figsize=(8, 6))

# # Adjust position of the boxes manually by creating a 'Type' ordering
# sns.boxplot(data=df_long, x="Type", y="Score", hue="Method", palette="Set2", width=0.6, dodge=True)

# # Adding space between boxplots by modifying the x-axis limits
# plt.title("Overlap metrics for validation set")
# plt.ylabel("Value")
# plt.xlabel("Metric")
# plt.ylim(0,1)
# plt.legend(title="Method")
# plt.tight_layout()

# # Save figure
# plt.savefig("dice_scores_boxplot.png", dpi=300)
# plt.show()