from pathlib import Path

import matplotlib.pyplot as plt

import glioseg.constants as constants


def generate_boxplots_from_excel(filename: str, output_dir: str | Path) -> None:
    """
    Reads an Excel file containing metric values and generates boxplots per label for each metric.
    
    Each sheet represents a metric, and for each label in the sheet, a boxplot is created showing 
    the distribution of values across patients. The generated plots are saved in the specified output directory.

    Args:
        filename (str): Path to the Excel file containing the metric data.
        output_dir (str | Path): Directory where boxplots will be saved.
    """
    output_dir = Path(output_dir)  # Convert to Path object if it's a string
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Load Excel file
    xls = pd.ExcelFile(filename)

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        # Skip the first three rows (headers) and reset index
        df_data = df.iloc[3:].reset_index(drop=True)

        # Extract labels from the second header row (row index 1)
        labels = []
        current_label = None
        for col_value in df.iloc[1, 1:]:
            if pd.notna(col_value):  # A new label starts here
                current_label = col_value
            labels.append(current_label)

        # Extract numeric data
        df_data = df_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")  # Ignore patient column

        # Get unique labels
        unique_labels = list(filter(lambda x: x is not None, set(labels)))

        # Generate boxplots for each label
        col_start = 0
        for label in unique_labels:
            num_models = labels.count(label)
            col_end = col_start + num_models

            plt.figure(figsize=(8, 6))
            plt.boxplot(df_data.iloc[:, col_start:col_end].T.dropna().values, labels=df.iloc[2, col_start:col_end])
            plt.title(f"Boxplot for {sheet_name} - {label} accross different models")
            plt.xlabel("Models")
            plt.ylabel(sheet_name)
            plt.xticks(rotation=90)

            # Save plot
            plot_filename = output_dir / f"boxplot_{sheet_name}_{label}.png"
            plt.savefig(plot_filename, bbox_inches="tight")
            plt.close()

            col_start = col_end  # Move to the next label

    print(f"Boxplots saved in {output_dir}")

if __name__ == "__main__":

    import pandas as pd 
    file_name= "/scratch/radv/share/glioseg/patients_clean_3/VALIDATION/METRICS/validation_metrics.xlsx"
    output_dir= "/scratch/radv/share/glioseg/patients_clean_3/VALIDATION/PLOTS/"
    xls = pd.ExcelFile(file_name)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        df_data = df.iloc[3:-2].reset_index(drop=True)
        # Extract labels from the second header row (row index 1)
        labels = []
        current_label = None
        for col_value in df.iloc[1, 1:]:
            if pd.notna(col_value): # A new label starts here
                current_label = col_value
            labels.append(current_label)
        df_data = df_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        # Get unique labels
        unique_labels = list(constants.TUMOR_LABELS.values())[1:-1]
        # print(unique_labels)
        col_start = 0
        models_names = list(df.iloc[2, 1:].values)

        for label in unique_labels:
            
            num_models = labels.count(label)
            
            col_end = col_start + num_models

            plt.figure(figsize=(8, 6))
            
            df_data_label = df_data.iloc[:,col_start:col_end].values
            models_label = models_names[col_start:col_end]
            print(models_label)

            plt.boxplot(df_data_label, tick_labels=models_label)
            plt.title(f"Boxplot for {sheet_name} - {label} accross different models")
            plt.xlabel("Models")
            plt.ylabel(sheet_name)
            plt.xticks(rotation=90)

            #Save plot
            plot_filename = Path(output_dir).joinpath(f"boxplot_{sheet_name.replace(' ' ,'_')}_{label.replace(' ','_')}.png")
            plt.savefig(plot_filename, bbox_inches="tight")
            plt.close()

            col_start = col_end
