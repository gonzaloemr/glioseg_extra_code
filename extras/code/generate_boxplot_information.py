from pathlib import Path

import pandas as pd


# === INPUT / OUTPUT paths ===
input_file = Path(
    "/scratch/radv/share/glioseg/new_run_corrected/VALIDATION/METRICS/additional_validation_metrics.xlsx"
)
output_file = input_file.parent.parent / "metrics_boxplot.xlsx"

# === SHEETS to process ===
sheets_to_keep = ["Dice score whole tumor", "Multiclass dice score"]

# === Function to clean columns ===
def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 'original' and 'relabeled' versions of each model.
    Drop *_relabeled_cc, *_TN, *_TN_relabeled, etc.
    """
    keep_cols = [c for c in df.columns if not any(substr in c for substr in ["_cc", "_TN"])]
    return df[keep_cols]


# === Process and save ===
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for sheet in sheets_to_keep:
        print(f"Processing sheet: {sheet}")

        # Read sheet, using first column as index (Patient IDs)
        df = pd.read_excel(
            input_file,
            sheet_name=sheet,
            header=1,  # skip merged header row
            index_col=0,  # patient column
            engine="openpyxl",
        )

        # Drop last two rows (mean, std)
        df = df.iloc[:-2]

        # Rename the index to "Patient" and make it a regular column
        df.index.name = "Patient"
        df.reset_index(inplace=True)

        # Keep only desired columns
        cleaned_df = filter_columns(df)

        # Reorder columns: Patient first
        cols = ["Patient"] + [c for c in cleaned_df.columns if c != "Patient"]
        cleaned_df = cleaned_df[cols]

        # Save cleaned sheet
        cleaned_df.to_excel(writer, sheet_name=sheet, index=False)

print(f"\n✅ Cleaned Excel file saved to:\n{output_file}")
